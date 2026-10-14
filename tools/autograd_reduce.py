import argparse
import itertools
import json
import math
import struct
from pathlib import Path

from tools.autograd_broadcast import (
    CAPACITY_EXCEEDED,
    INCOMPATIBLE_BROADCAST,
    MAX_ELEMENTS,
    MAX_RANK,
    RANK_EXCEEDED,
    SPECIAL_WORDS,
    counting_words,
    framed,
    is_nan,
    run_lanes,
    word,
)
from tools.common import ROOT, GateError

NOT_REDUCIBLE = 101


def f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(number: float) -> int:
    return struct.unpack("<I", struct.pack("<f", number))[0]


def handwritten() -> list[dict]:
    big = [word(1e8), word(1), word(-1e8), word(1)]
    return [
        {"id": "rows_to_row", "shape": [2, 3], "target": [3]},
        {"id": "columns_to_column", "shape": [2, 3], "target": [2, 1]},
        {"id": "all_to_scalar", "shape": [2, 3], "target": []},
        {"id": "empty_rows_to_zeros", "shape": [0, 3], "target": [1, 3]},
        {"id": "empty_to_scalar", "shape": [0], "target": []},
        {"id": "same_shape", "shape": [2, 3], "target": [2, 3]},
        {"id": "rank_four", "shape": [2, 2, 1, 3], "target": [2, 1, 3]},
        {"id": "ordered_cancellation", "shape": [4], "target": [], "words": big},
        {
            "id": "ordered_cancellation_reversed",
            "shape": [4],
            "target": [],
            "words": big[::-1],
        },
        {"id": "special_words_rows", "shape": [8, 1], "target": [1]},
        {
            "id": "negative_zero_fiber",
            "shape": [2],
            "target": [],
            "words": [0x80000000, 0x80000000],
        },
        {"id": "special_words_pairs", "shape": [2, 4], "target": [4]},
        {
            "id": "finite_overflow",
            "shape": [2],
            "target": [],
            "words": [0x7F7FFFFF, 0x7F7FFFFF],
        },
        {
            "id": "negative_overflow_rows",
            "shape": [2, 2],
            "target": [2],
            "words": [0xFF7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, word(1)],
        },
        {"id": "target_larger", "shape": [3], "target": [2, 3]},
        {"id": "target_wider", "shape": [1], "target": [3]},
        {"id": "incompatible", "shape": [3], "target": [2]},
        {"id": "target_rank_five", "shape": [1], "target": [1, 1, 1, 1, 1]},
        {"id": "target_capacity", "shape": [1], "target": [1025, 1024]},
        {"id": "wide_rows", "shape": [32, 64], "target": [1, 64]},
    ]


def exhaustive() -> list[dict]:
    extents = (0, 1, 2)
    shapes = [
        list(shape)
        for rank in range(4)
        for shape in itertools.product(extents, repeat=rank)
    ]
    targets = [
        list(shape)
        for rank in range(3)
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
        for shape in shapes
        for target in targets
    ]


def cases() -> list[dict]:
    result = []
    for case in handwritten() + exhaustive():
        count = math.prod(case["shape"])
        if "words" not in case:
            case["words"] = (
                list(SPECIAL_WORDS)
                if case["id"].startswith("special_words")
                else counting_words(count)
            )
        if len(case["words"]) != count:
            raise GateError(f"{case['id']}: input words disagree with its shape")
        result.append(case)
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate autograd reduce case identifier")
    return result


def reference(case: dict) -> dict:
    shape, target, words = case["shape"], case["target"], case["words"]
    if len(target) > MAX_RANK:
        return {"error": RANK_EXCEEDED}
    if math.prod(target) > MAX_ELEMENTS:
        return {"error": CAPACITY_EXCEEDED}
    rank = max(len(shape), len(target))
    left = [1] * (rank - len(target)) + target
    right = [1] * (rank - len(shape)) + shape
    output = []
    for a, b in zip(left, right, strict=True):
        if a != b and a != 1 and b != 1:
            return {"error": INCOMPATIBLE_BROADCAST}
        output.append(b if a == 1 else a)
    if math.prod(output) > MAX_ELEMENTS:
        return {"error": CAPACITY_EXCEEDED}
    if output != shape:
        return {"error": NOT_REDUCIBLE}
    totals = [0.0] * math.prod(target)
    for flat in range(math.prod(shape)):
        coordinate = []
        remainder = flat
        for extent in reversed(shape):
            coordinate.append(remainder % extent)
            remainder //= extent
        coordinate.reverse()
        position = 0
        for extent, index in zip(left, coordinate, strict=True):
            position = position * extent + (0 if extent == 1 else index)
        totals[position] = f32(totals[position] + value(words[flat]))
    return {"shape": target, "words": [bits(total) for total in totals]}


def same_sums(observed: list[int], expected: list[int], lane: str) -> bool:
    return len(observed) == len(expected) and all(
        a == b or (is_nan(a) and is_nan(b))
        for a, b in zip(observed, expected, strict=True)
    )


def run_gate(tool_root: Path, output: Path) -> dict:
    selected = cases()
    return run_lanes(
        tool_root,
        output,
        "reduce",
        selected,
        *framed(selected, "reduce", reference, same_sums),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
