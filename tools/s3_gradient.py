import argparse
import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

from tools.common import ROOT, GateError
from tools.s3_broadcast import (
    CAPACITY,
    CAPACITY_EXCEEDED,
    RANK_EXCEEDED,
    SPECIAL_WORDS,
    counting_words,
    is_nan,
    program,
    run_lanes,
    same_words,
    snapshot,
    tensor_source,
    word,
)
from tools.s3_reduce import bits, f32, value

SHAPE_MISMATCH = 102


def scaled_words(count: int, scale: float) -> list[int]:
    return [bits(f32((index + 1) * scale)) for index in range(count)]


def contribution(shape: list[int], words: list[int] | None = None) -> dict:
    count = math.prod(shape)
    return {"shape": shape, "words": counting_words(count) if words is None else words}


def handwritten() -> list[dict]:
    big = [[word(1e8)], [word(1)], [word(-1e8)], [word(1)]]
    tenths = scaled_words(6, 0.1)
    return [
        {"id": "first_moves_in", "shape": [2, 3], "items": [contribution([2, 3])]},
        {
            "id": "second_adds",
            "shape": [2, 3],
            "items": [contribution([2, 3]), contribution([2, 3], tenths)],
        },
        {
            "id": "refusal_between_contributions",
            "shape": [2, 3],
            "items": [
                contribution([2, 3]),
                contribution([3]),
                contribution([2, 3], tenths),
            ],
        },
        {"id": "unset_reads_zeros", "shape": [2, 3], "items": []},
        {"id": "unset_scalar_reads_zero", "shape": [], "items": []},
        {"id": "unset_empty", "shape": [0, 3], "items": []},
        {
            "id": "empty_contributions",
            "shape": [0, 3],
            "items": [contribution([0, 3]), contribution([0, 3])],
        },
        {
            "id": "ordered_cancellation",
            "shape": [],
            "items": [contribution([], item) for item in big],
        },
        {
            "id": "ordered_cancellation_reversed",
            "shape": [],
            "items": [contribution([], item) for item in big[::-1]],
        },
        {
            "id": "negative_zero_moves_in",
            "shape": [],
            "items": [contribution([], [0x80000000])],
        },
        {
            "id": "negative_zeros_add",
            "shape": [],
            "items": [contribution([], [0x80000000]), contribution([], [0x80000000])],
        },
        {
            "id": "negative_zero_then_zero",
            "shape": [],
            "items": [contribution([], [0x80000000]), contribution([], [0])],
        },
        {
            "id": "special_words",
            "shape": [8],
            "items": [
                contribution([8], list(SPECIAL_WORDS)),
                contribution([8], list(SPECIAL_WORDS[::-1])),
            ],
        },
        {"id": "only_refusals", "shape": [2], "items": [contribution([3])]},
        {
            "id": "same_count_other_dimensions",
            "shape": [2, 3],
            "items": [contribution([3, 2]), contribution([6]), contribution([2, 3])],
        },
        {
            "id": "broadcastable_is_refused",
            "shape": [1, 3],
            "items": [contribution([3]), contribution([2, 3]), contribution([1, 3])],
        },
        {
            "id": "rank_four",
            "shape": [2, 1, 2, 1],
            "items": [
                contribution([2, 1, 2, 1]),
                contribution([2, 1, 2, 1], scaled_words(4, 0.25)),
            ],
        },
        {
            "id": "many_contributions",
            "shape": [2],
            "items": [
                contribution([2], scaled_words(2, 0.1 * (step + 1)))
                for step in range(8)
            ],
        },
        {"id": "slot_rank_five", "shape": [1, 1, 1, 1, 1], "items": []},
        {"id": "slot_capacity", "shape": [1025, 1024], "items": []},
        {
            "id": "wide_rows",
            "shape": [32, 64],
            "items": [
                contribution([32, 64]),
                contribution([32, 64], scaled_words(2048, 0.5)),
            ],
        },
    ]


def exhaustive() -> list[dict]:
    extents = (0, 1, 2)
    shapes = [
        list(shape)
        for rank in range(3)
        for shape in itertools.product(extents, repeat=rank)
    ]
    result = []
    for slot in shapes:
        name = "x".join(map(str, slot))
        count = math.prod(slot)
        result.append({"id": f"small_{name}_unset", "shape": slot, "items": []})
        for other in shapes:
            result.append(
                {
                    "id": f"small_{name}_with_" + "x".join(map(str, other)),
                    "shape": slot,
                    "items": [
                        contribution(slot),
                        contribution(other, scaled_words(math.prod(other), 0.1)),
                        contribution(slot, scaled_words(count, 1 / 3)),
                    ],
                }
            )
    return result


def cases() -> list[dict]:
    result = handwritten() + exhaustive()
    for case in result:
        for item in case["items"]:
            if len(item["words"]) != math.prod(item["shape"]):
                raise GateError(f"{case['id']}: contribution words disagree")
            if len(item["words"]) > 2048:
                raise GateError(f"{case['id']}: literal exceeds the JS stack limit")
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate S3 gradient case identifier")
    return result


def reference(case: dict) -> dict:
    shape = case["shape"]
    if len(shape) > 4:
        return {"error": RANK_EXCEEDED}
    if math.prod(shape) > CAPACITY:
        return {"error": CAPACITY_EXCEEDED}
    total = None
    outcomes = []
    for item in case["items"]:
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
    if total is None:
        total = [0] * math.prod(shape)
    return {"outcomes": outcomes, "gradient": {"shape": shape, "words": total}}


def arguments(case: dict) -> str:
    items = ", ".join(
        tensor_source(item["shape"], item["words"]) for item in case["items"]
    )
    return f"{json.dumps(case['shape'])}, [{items}]"


def parse(text: str, identifiers: list[str]) -> list[dict]:
    rows = text.splitlines()
    results = []
    index = 0
    while index < len(rows):
        begin = rows[index]
        if not begin.startswith("S3:begin:"):
            raise GateError(f"Unframed S3 output: {begin!r}")
        identifier = begin.removeprefix("S3:begin:")
        end = rows.index("S3:end", index)
        payload = rows[index + 1 : end]
        index = end + 1
        outcomes = []
        while (
            payload
            and payload[0] not in ("S3:gradient",)
            and not payload[0].startswith("S3:error:")
        ):
            head, payload = payload[0], payload[1:]
            if head == "S3:accumulated":
                outcomes.append({"accumulated": True})
            elif head.startswith("S3:refused:"):
                returned, payload = snapshot(payload, identifier)
                outcomes.append(
                    {
                        "refused": int(head.removeprefix("S3:refused:")),
                        "returned": returned,
                    }
                )
            else:
                raise GateError(f"{identifier}: unexpected row {head!r}")
        if len(payload) == 1 and payload[0].startswith("S3:error:"):
            results.append(
                {"id": identifier, "error": int(payload[0].removeprefix("S3:error:"))}
            )
            if outcomes:
                raise GateError(f"{identifier}: outcomes before a slot error")
            continue
        if payload[:1] != ["S3:gradient"]:
            raise GateError(f"{identifier}: missing gradient")
        gradient, extra = snapshot(payload[1:], identifier)
        if extra:
            raise GateError(f"{identifier}: trailing output rows")
        results.append({"id": identifier, "outcomes": outcomes, "gradient": gradient})
    if [row["id"] for row in results] != identifiers:
        raise GateError("S3 gradient inventory or order differs from the requests")
    return results


def same_gradient(observed: list[int], expected: list[int]) -> bool:
    return len(observed) == len(expected) and all(
        a == b or (is_nan(a) and is_nan(b))
        for a, b in zip(observed, expected, strict=True)
    )


def verify(selected: list[dict], observations: list[dict], lane: str) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete S3 gradient case inventory")
    gradients = refusals = errors = 0
    for case, row in zip(selected, observations, strict=True):
        identifier = case["id"]
        expected = reference(case)
        if "error" in expected:
            if row != {"id": identifier, "error": expected["error"]}:
                raise GateError(f"{lane} {identifier}: expected error {expected}")
            errors += 1
            continue
        if "outcomes" not in row or len(row["outcomes"]) != len(expected["outcomes"]):
            raise GateError(f"{lane} {identifier}: wrong outcome count")
        for seen, wanted in zip(row["outcomes"], expected["outcomes"], strict=True):
            if "accumulated" in wanted:
                if seen != wanted:
                    raise GateError(f"{lane} {identifier}: contribution was refused")
                continue
            returned = wanted["returned"]
            if (
                seen.get("refused") != wanted["refused"]
                or seen["returned"]["shape"] != returned["shape"]
                or not same_words(seen["returned"]["words"], returned["words"], lane)
            ):
                raise GateError(
                    f"{lane} {identifier}: refusal changed the contribution"
                )
            refusals += 1
        gradient = row["gradient"]
        if gradient["shape"] != expected["gradient"]["shape"] or not same_gradient(
            gradient["words"], expected["gradient"]["words"]
        ):
            raise GateError(f"{lane} {identifier}: wrong gradient words")
        gradients += 1
    return {
        "cases": len(selected),
        "gradients": gradients,
        "refusals": refusals,
        "errors": errors,
    }


def lanes(
    selected: list[dict],
) -> tuple[Callable[[Path, Path], str], Callable[[str, str], tuple[list, dict]]]:
    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root, "accumulate", arguments)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        return observations, verify(selected, observations, lane)

    return render, check


def run_stage(tool_root: Path, output: Path) -> dict:
    selected = cases()
    return run_lanes(tool_root, output, "gradient", selected, *lanes(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
