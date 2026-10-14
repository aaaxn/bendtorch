import argparse
import json
import math
import shutil
from pathlib import Path

from tools.autograd_broadcast import (
    CAPACITY_EXCEEDED,
    MAX_ELEMENTS,
    MAX_RANK,
    RANK_EXCEEDED,
    program,
    run_lanes,
    snapshot,
    tensor_source,
    word,
)
from tools.autograd_gradient import SHAPE_MISMATCH, contribution, same_gradient
from tools.autograd_reduce import bits, f32, value
from tools.common import ROOT, Evidence, GateError, bend

NEGATIVE_ZERO = 0x80000000
RUNTIME_CONTROLS = {
    "grad-held-reads-none": (
        "    case Held{gradient}:\n      Some{gradient}",
        "    case Held{gradient}:\n      None{}",
    ),
    "reset-keeps-held": (
        "def reset(slot: Slot) -> Slot:\n  Unset{}",
        "def reset(slot: Slot) -> Slot:\n  slot",
    ),
}


def handwritten() -> list[dict]:
    one_two = contribution([2], [word(1), word(2)])
    return [
        {"id": "never_set_is_none", "shape": [2], "before": [], "after": []},
        {"id": "scalar_never_set", "shape": [], "before": [], "after": []},
        {"id": "empty_never_set", "shape": [0, 3], "before": [], "after": []},
        {
            "id": "reset_drops_held",
            "shape": [2],
            "before": [one_two, one_two],
            "after": [],
        },
        {
            "id": "reset_keeps_negative_zero",
            "shape": [],
            "before": [contribution([], [word(5)])],
            "after": [contribution([], [NEGATIVE_ZERO])],
        },
        {
            "id": "reset_then_two",
            "shape": [2],
            "before": [one_two],
            "after": [contribution([2], [NEGATIVE_ZERO, word(3)]), one_two],
        },
        {
            "id": "refused_stays_none",
            "shape": [2],
            "before": [contribution([3])],
            "after": [contribution([1])],
        },
        {"id": "slot_rank_five", "shape": [1, 1, 1, 1, 1], "before": [], "after": []},
        {"id": "slot_capacity", "shape": [1025, 1024], "before": [], "after": []},
    ]


def cases() -> list[dict]:
    result = handwritten()
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate autograd lifecycle case identifier")
    return result


def accumulate(shape: list[int], total: list[int] | None, items: list[dict]):
    outcomes = []
    for item in items:
        if item["shape"] != shape:
            outcomes.append({"refused": SHAPE_MISMATCH, "returned": item})
            continue
        outcomes.append({"accumulated": True})
        if total is None:
            total = list(item["words"])
        else:
            total = [
                bits(f32(value(a) + value(b)))
                for a, b in zip(total, item["words"], strict=True)
            ]
    return outcomes, total


def grad(shape: list[int], total: list[int] | None) -> dict | None:
    return None if total is None else {"shape": shape, "words": total}


def reference(case: dict) -> dict:
    shape = case["shape"]
    if len(shape) > MAX_RANK:
        return {"error": RANK_EXCEEDED}
    if math.prod(shape) > MAX_ELEMENTS:
        return {"error": CAPACITY_EXCEEDED}
    before, held = accumulate(shape, None, case["before"])
    after, total = accumulate(shape, None, case["after"])
    return {
        "before": before,
        "held": grad(shape, held),
        "cleared": None,
        "after": after,
        "final": grad(shape, total),
        "gradient": {
            "shape": shape,
            "words": [0] * math.prod(shape) if total is None else total,
        },
    }


def arguments(case: dict) -> str:
    def items(rows: list[dict]) -> str:
        return ", ".join(tensor_source(row["shape"], row["words"]) for row in rows)

    return (
        f"{json.dumps(case['shape'])}, [{items(case['before'])}], "
        f"[{items(case['after'])}]"
    )


def outcomes(payload: list[str], identifier: str) -> tuple[list[dict], list[str]]:
    result = []
    while payload and (
        payload[0] == "S3:accumulated" or payload[0].startswith("S3:refused:")
    ):
        head, payload = payload[0], payload[1:]
        if head == "S3:accumulated":
            result.append({"accumulated": True})
            continue
        returned, payload = snapshot(payload, identifier)
        result.append(
            {
                "refused": int(head.removeprefix("S3:refused:")),
                "returned": returned,
            }
        )
    return result, payload


def read_grad(payload: list[str], identifier: str) -> tuple[dict | None, list[str]]:
    if payload[:1] == ["S3:grad:none"]:
        return None, payload[1:]
    if payload[:1] == ["S3:grad:some"]:
        return snapshot(payload[1:], identifier)
    raise GateError(f"{identifier}: missing grad read")


def parse(text: str, identifiers: list[str]) -> list[dict]:
    rows = text.splitlines()
    results = []
    index = 0
    while index < len(rows):
        begin = rows[index]
        if not begin.startswith("S3:begin:"):
            raise GateError(f"Unframed autograd output: {begin!r}")
        identifier = begin.removeprefix("S3:begin:")
        end = rows.index("S3:end", index)
        payload = rows[index + 1 : end]
        index = end + 1
        if len(payload) == 1 and payload[0].startswith("S3:error:"):
            results.append(
                {
                    "id": identifier,
                    "error": int(payload[0].removeprefix("S3:error:")),
                }
            )
            continue
        before, payload = outcomes(payload, identifier)
        held, payload = read_grad(payload, identifier)
        if payload[:1] != ["S3:reset"]:
            raise GateError(f"{identifier}: missing reset")
        cleared, payload = read_grad(payload[1:], identifier)
        after, payload = outcomes(payload, identifier)
        final, payload = read_grad(payload, identifier)
        if payload[:1] != ["S3:gradient"]:
            raise GateError(f"{identifier}: missing zeros-materializing gradient")
        gradient, extra = snapshot(payload[1:], identifier)
        if extra:
            raise GateError(f"{identifier}: trailing output rows")
        results.append(
            {
                "id": identifier,
                "before": before,
                "held": held,
                "cleared": cleared,
                "after": after,
                "final": final,
                "gradient": gradient,
            }
        )
    if [row["id"] for row in results] != identifiers:
        raise GateError(
            "Autograd lifecycle inventory or order differs from the requests"
        )
    return results


def same_tensor(seen: dict | None, wanted: dict | None) -> bool:
    if seen is None or wanted is None:
        return seen is wanted
    return seen["shape"] == wanted["shape"] and same_gradient(
        seen["words"], wanted["words"]
    )


def same_outcomes(seen: list[dict], wanted: list[dict]) -> bool:
    return len(seen) == len(wanted) and all(
        a == b
        if "accumulated" in b
        else a.get("refused") == b["refused"]
        and same_tensor(a["returned"], b["returned"])
        for a, b in zip(seen, wanted, strict=True)
    )


def verify(selected: list[dict], observations: list[dict], lane: str) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete autograd lifecycle case inventory")
    nones = errors = 0
    for case, row in zip(selected, observations, strict=True):
        identifier = case["id"]
        expected = reference(case)
        if "error" in expected:
            if row != {"id": identifier, "error": expected["error"]}:
                raise GateError(f"{lane} {identifier}: expected error {expected}")
            errors += 1
            continue
        for key in ("before", "after"):
            if not same_outcomes(row[key], expected[key]):
                raise GateError(f"{lane} {identifier}: wrong {key} outcomes")
        for key in ("held", "cleared", "final", "gradient"):
            if not same_tensor(row[key], expected[key]):
                raise GateError(f"{lane} {identifier}: wrong {key} read")
            nones += row[key] is None
    return {"cases": len(selected), "none_reads": nones, "errors": errors}


def run_gate(tool_root: Path, output: Path) -> dict:
    selected = cases()

    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root, "lifecycle", arguments)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        return observations, verify(selected, observations, lane)

    summary = run_lanes(tool_root, output, "lifecycle", selected, render, check)
    summary["runtime_controls"] = runtime_controls(
        tool_root.resolve(), output.resolve() / "runtime-controls", selected
    )
    return summary


def runtime_controls(tool_root: Path, output: Path, selected: list[dict]) -> list[str]:
    evidence = Evidence(output, tool_root)
    rejected = []
    for name, (original, mutated) in RUNTIME_CONTROLS.items():
        copy = output / name
        shutil.copytree(ROOT / "src", copy / "src")
        target = copy / "src/autograd/gradient.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: runtime control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        source = copy / "lifecycle.bend"
        source.write_text(program(selected, copy, copy, "lifecycle", arguments))
        script = copy / "lifecycle.js"
        evidence.run(f"{name}-javascript-build", bend(tool_root, source, "-o", script))
        run = evidence.run(
            f"{name}-javascript-run", [bend(tool_root)[0], str(script)], timeout=1800
        )
        try:
            verify(selected, parse(run.stdout, [case["id"] for case in selected]), name)
        except GateError:
            rejected.append(name)
            continue
        raise GateError(f"{name}: the lifecycle lane accepted a broken accessor")
    evidence.save()
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
