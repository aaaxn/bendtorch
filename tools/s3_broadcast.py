import argparse
import itertools
import json
import math
import os
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

from tools.common import ROOT, Evidence, GateError, bend, check_sources

CAPACITY = 1 << 20
NOT_BROADCASTABLE = 100
RANK_EXCEEDED = 1
CAPACITY_EXCEEDED = 2
INCOMPATIBLE_BROADCAST = 6
SPECIAL_WORDS = (
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x7F7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC01234,
    0xFFA00001,
)


def word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def counting_words(count: int) -> list[int]:
    return [word(float(index + 1)) for index in range(count)]


def handwritten() -> list[dict]:
    one_two_three = counting_words(3)
    return [
        {"id": "row_to_matrix", "shape": [3], "target": [2, 3]},
        {"id": "scalar_to_square", "shape": [], "target": [2, 2], "words": [word(7)]},
        {"id": "column_to_matrix", "shape": [2, 1], "target": [2, 3]},
        {"id": "row_to_empty_rows", "shape": [1, 3], "target": [0, 3]},
        {"id": "same_shape", "shape": [2, 3], "target": [2, 3]},
        {"id": "rank_four", "shape": [2, 1, 3], "target": [2, 2, 1, 3]},
        {"id": "special_words", "shape": [8, 1], "target": [8, 2]},
        {"id": "larger_than_target", "shape": [3], "target": [1]},
        {"id": "higher_rank_input", "shape": [2, 3], "target": [3]},
        {"id": "incompatible", "shape": [2], "target": [3]},
        {"id": "empty_to_one", "shape": [0], "target": [1]},
        {"id": "target_rank_five", "shape": [1], "target": [1, 1, 1, 1, 1]},
        {"id": "target_capacity", "shape": [1], "target": [1025, 1024]},
        {"id": "output_capacity", "shape": [1024, 1], "target": [1, 1025]},
        {"id": "wide_square", "shape": [1, 64], "target": [64, 64]},
        {
            "id": "words_follow_projection",
            "shape": [1, 3],
            "target": [2, 3],
            "words": one_two_three,
        },
    ]


def exhaustive() -> list[dict]:
    extents = (0, 1, 2)
    inputs = [
        list(shape)
        for rank in range(3)
        for shape in itertools.product(extents, repeat=rank)
    ]
    targets = [
        list(shape)
        for rank in range(4)
        for shape in itertools.product(extents, repeat=rank)
    ]
    return [
        {
            "id": "small_"
            + "x".join(map(str, shape))
            + "_to_"
            + "x".join(map(str, target)),
            "shape": shape,
            "target": target,
        }
        for shape in inputs
        for target in targets
    ]


def cases() -> list[dict]:
    result = []
    for case in handwritten() + exhaustive():
        count = math.prod(case["shape"])
        if "words" not in case:
            case["words"] = (
                list(SPECIAL_WORDS)
                if case["id"] == "special_words"
                else counting_words(count)
            )
        if len(case["words"]) != count:
            raise GateError(f"{case['id']}: input words disagree with its shape")
        result.append(case)
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate S3 broadcast case identifier")
    return result


def reference(case: dict) -> dict:
    shape, target, words = case["shape"], case["target"], case["words"]
    if len(target) > 4:
        return {"error": RANK_EXCEEDED}
    if math.prod(target) > CAPACITY:
        return {"error": CAPACITY_EXCEEDED}
    rank = max(len(shape), len(target))
    left = [1] * (rank - len(shape)) + shape
    right = [1] * (rank - len(target)) + target
    output = []
    for a, b in zip(left, right, strict=True):
        if a != b and a != 1 and b != 1:
            return {"error": INCOMPATIBLE_BROADCAST}
        output.append(b if a == 1 else a)
    if math.prod(output) > CAPACITY:
        return {"error": CAPACITY_EXCEEDED}
    if output != target:
        return {"error": NOT_BROADCASTABLE}
    values = []
    for flat in range(math.prod(target)):
        coordinate = []
        for extent in reversed(target):
            coordinate.append(flat % extent)
            flat //= extent
        coordinate.reverse()
        source = 0
        for extent, index in zip(left, coordinate, strict=True):
            source = source * extent + (0 if extent == 1 else index)
        values.append(words[source])
    return {"shape": target, "words": values}


def tensor_source(shape: list[int], words: list[int]) -> str:
    values = ", ".join(f"Tensor.from_bits({value})" for value in words)
    return f"Core.construct({json.dumps(shape)}, [{values}])"


def target_arguments(case: dict) -> str:
    return (
        f"{tensor_source(case['shape'], case['words'])}, {json.dumps(case['target'])}"
    )


def program(
    selected: list[dict],
    directory: Path,
    source_root: Path = ROOT,
    call: str = "broadcast",
    arguments: Callable[[dict], str] = target_arguments,
) -> str:
    if not selected:
        raise GateError(f"Refusing to generate an empty S3 {call} corpus")
    lines = ["import Base"]
    for path, alias in (
        ("src/autograd/harness.bend", "Harness"),
        ("src/tensor/harness.bend", "Tensor"),
        ("src/tensor/core.bend", "Core"),
    ):
        relative = os.path.relpath(source_root / path, directory)
        lines.append(f"import ./{relative} as {alias}")
    for index, case in enumerate(selected):
        lines += [
            "",
            f"def run_case_{index}() -> IO(Unit):",
            "  do IO<Unit>:",
            f'    IO.print("S3:begin:{case["id"]}")',
            f"    Harness.{call}({arguments(case)})",
            '    IO.print("S3:end")',
        ]
    batches = range(0, len(selected), 16)
    for batch, start in enumerate(batches):
        lines += ["", f"def run_batch_{batch}() -> IO(Unit):", "  do IO<Unit>:"]
        lines += [
            f"    run_case_{index}()"
            for index in range(start, min(start + 16, len(selected)))
        ]
    lines += ["", "def main() -> IO(Unit):", "  do IO<Unit>:"]
    lines += [f"    run_batch_{batch}()" for batch in range(len(batches))]
    return "\n".join(lines) + "\n"


def snapshot(rows: list[str], identifier: str) -> tuple[dict, list[str]]:
    if len(rows) < 2 or not rows[0].startswith("S2:shape:"):
        raise GateError(f"{identifier}: missing tensor shape")
    shape = json.loads(rows[0].removeprefix("S2:shape:"))
    if not rows[1].startswith("S2:count:"):
        raise GateError(f"{identifier}: missing element count")
    count = int(rows[1].removeprefix("S2:count:"))
    if count != math.prod(shape):
        raise GateError(f"{identifier}: element count disagrees with shape")
    words = rows[2 : 2 + count]
    if len(words) != count or not all(row.startswith("S2:word:") for row in words):
        raise GateError(f"{identifier}: missing or extra tensor words")
    values = [int(row.removeprefix("S2:word:")) for row in words]
    return {"shape": shape, "words": values}, rows[2 + count :]


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
        if payload[:1] != ["S3:retained"]:
            raise GateError(f"{identifier}: missing retained input")
        retained, rest = snapshot(payload[1:], identifier)
        if len(rest) == 1 and rest[0].startswith("S3:error:"):
            outcome = {"error": int(rest[0].removeprefix("S3:error:"))}
        elif rest[:1] == ["S3:output"]:
            outcome, extra = snapshot(rest[1:], identifier)
            if extra:
                raise GateError(f"{identifier}: trailing output rows")
        else:
            raise GateError(f"{identifier}: missing broadcast outcome")
        results.append({"id": identifier, "retained": retained, "outcome": outcome})
    if [row["id"] for row in results] != identifiers:
        raise GateError("S3 broadcast inventory or order differs from the requests")
    return results


def is_nan(value: int) -> bool:
    return value & 0x7F800000 == 0x7F800000 and value & 0x7FFFFF != 0


def same_words(observed: list[int], expected: list[int], lane: str) -> bool:
    if lane == "javascript":
        return len(observed) == len(expected) and all(
            a == b or (is_nan(a) and is_nan(b))
            for a, b in zip(observed, expected, strict=True)
        )
    return observed == expected


def verify(
    selected: list[dict],
    observations: list[dict],
    lane: str,
    expected_result: Callable[[dict], dict] = reference,
    same_output: Callable[[list[int], list[int], str], bool] = same_words,
) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete S3 case inventory")
    tensors = errors = 0
    for case, row in zip(selected, observations, strict=True):
        identifier = case["id"]
        retained = row["retained"]
        if retained["shape"] != case["shape"] or not same_words(
            retained["words"], case["words"], lane
        ):
            raise GateError(f"{lane} {identifier}: input owner changed")
        expected = expected_result(case)
        outcome = row["outcome"]
        if "error" in expected:
            if outcome != expected:
                raise GateError(f"{lane} {identifier}: expected error {expected}")
            errors += 1
            continue
        if outcome.get("shape") != expected["shape"] or not same_output(
            outcome["words"], expected["words"], lane
        ):
            raise GateError(f"{lane} {identifier}: wrong output words")
        tensors += 1
    return {"cases": len(selected), "tensors": tensors, "errors": errors}


def framed(
    selected: list[dict],
    call: str,
    expected_result: Callable[[dict], dict],
    same_output: Callable[[list[int], list[int], str], bool] = same_words,
) -> tuple[Callable[[Path, Path], str], Callable[[str, str], tuple[list, dict]]]:
    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root, call)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        summary = verify(selected, observations, lane, expected_result, same_output)
        return observations, summary

    return render, check


def run_lanes(
    tool_root: Path,
    output: Path,
    call: str,
    selected: list[dict],
    render: Callable[[Path, Path], str],
    check: Callable[[str, str], tuple[list, dict]],
) -> dict:
    tool_root, output = tool_root.resolve(), output.resolve()
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    shutil.copytree(ROOT / "src", output / "source/src")
    (output / "requests.json").write_text(json.dumps(selected, indent=2) + "\n")
    source = output / f"{call}.bend"
    source.write_text(render(output, output / "source"))
    evidence.run(
        f"{call}-check",
        bend(tool_root, source, "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    summary = {}
    for lane in ("native", "javascript"):
        if lane == "native":
            emitted = output / f"{call}.c"
            evidence.run(f"{call}-c-export", bend(tool_root, source, "-o", emitted))
            target = output / call
            evidence.run(
                f"{call}-native-build",
                [
                    str(tool_root / "llvm/usr/bin/clang-19"),
                    "-std=c11",
                    "-O3",
                    str(emitted),
                    "-lpthread",
                    "-lm",
                    "-o",
                    str(target),
                ],
            )
            command = [str(target), "--threads", "1", "--gpu", "off"]
        else:
            target = output / f"{call}.js"
            evidence.run(
                f"{call}-javascript-build", bend(tool_root, source, "-o", target)
            )
            command = [bend(tool_root)[0], str(target)]
        run = evidence.run(f"{call}-{lane}-run", command, timeout=1800)
        observations, summary[lane] = check(run.stdout, lane)
        (output / f"{lane}-results.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    evidence.save()
    return summary


def run_stage(tool_root: Path, output: Path) -> dict:
    selected = cases()
    return run_lanes(
        tool_root,
        output,
        "broadcast",
        selected,
        *framed(selected, "broadcast", reference),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
