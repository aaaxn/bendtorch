"""Signed axes, indices and -1 reshape sizes (src/tensor/signed.bend).

Each request runs PyTorch-style arguments, negative ones included, through
the signed operations on the native and JavaScript lanes, and both must print
this module's reference line for line. oracle/tensor/signed.py replays the
same requests with PyTorch; tests/test_tensor_signed.py compares the
reference with that record.
"""

import argparse
import json
import math
import os
from pathlib import Path

from tools.autograd_broadcast import is_nan, run_lanes, word
from tools.common import ROOT, GateError

RANK_EXCEEDED = 1
CAPACITY_EXCEEDED = 2
COORDINATE_RANK = 3
COORDINATE_OUT_OF_BOUNDS = 4
FLAT_OUT_OF_BOUNDS = 8
AXES_LENGTH = 12
AXIS_OUT_OF_BOUNDS = 13
DUPLICATE_AXIS = 14
ELEMENT_COUNT = 15
REPEATED_INFERRED = 17
AMBIGUOUS_INFERRED = 18
MAX_RANK = 4
MAX_ELEMENTS = 1 << 20
QUIET_NAN = 0x7FC00001
NEGATIVE_ZERO = 0x80000000
NEGATIVE_INFINITY = 0xFF800000
LARGE = 1 << 24
MAX_U32 = (1 << 32) - 1

# A grid whose words a copy must keep bit for bit: -0.0, a NaN payload, a
# subnormal and an infinity among ordinary words.
GRID = ([2, 3], [word(0.5), NEGATIVE_ZERO, QUIET_NAN, 1, NEGATIVE_INFINITY, word(3.0)])
CUBE = ([2, 3, 4], [word(float(i)) for i in range(24)])
EMPTY = ([0, 3], [])
SCALAR = ([], [word(5.0)])
VECTOR = ([3], [word(1.0), word(2.0), word(3.0)])
EMPTY_VECTOR = ([0], [])


def request(identifier: str, tensor: tuple, op: str, **arguments) -> dict:
    shape, words = tensor
    return {"id": identifier, "shape": shape, "words": words, "op": op, **arguments}


def cases() -> list[dict]:
    rows = []
    for name, tensor, sizes in (
        ("all", GRID, [-1]),
        ("first", GRID, [-1, 2]),
        ("last", GRID, [3, -1]),
        ("one", GRID, [-1, 6]),
        ("middle", GRID, [6, -1, 1]),
        ("rank4", GRID, [-1, 1, 1, 1]),
        ("exact", GRID, [3, 2]),
        ("indivisible", GRID, [-1, 4]),
        ("twice", GRID, [-1, -1]),
        ("twice_before_count", GRID, [-1, -1, 7]),
        ("zero_other", GRID, [-1, 0]),
        ("other_zero", GRID, [0, -1]),
        ("exact_count", GRID, [5]),
        ("empty_all", EMPTY, [-1]),
        ("empty_first", EMPTY, [-1, 3]),
        ("empty_last", EMPTY, [3, -1]),
        ("empty_pair", EMPTY, [2, -1]),
        ("empty_ambiguous", EMPTY, [-1, 0]),
        ("empty_ambiguous_last", EMPTY, [0, -1]),
        ("empty_ambiguous_three", EMPTY, [-1, 0, 2]),
        ("empty_twice", EMPTY, [0, -1, -1]),
        # Exact sizes whose product passes the native Nat limit of 2^48: the
        # inference classifies it against the count and never forms it.
        ("empty_large", EMPTY, [-1, LARGE, LARGE]),
        ("empty_large_last", EMPTY, [LARGE, LARGE, -1]),
        ("large_mismatch", GRID, [-1, LARGE, LARGE]),
        ("largest_mismatch", GRID, [-1, MAX_U32, MAX_U32]),
        ("empty_large_ambiguous", EMPTY, [-1, 0, LARGE, LARGE]),
        ("empty_large_ambiguous_middle", EMPTY, [-1, LARGE, 0, LARGE]),
        ("past_count", GRID, [-1, 7]),
        # Empty shapes whose other sizes multiply past 2^63: bendtorch takes
        # them, PyTorch refuses them (docs/torch-differences.md).
        ("empty_huge", EMPTY, [-1, MAX_U32, MAX_U32]),
        ("empty_huge_exact", EMPTY, [0, MAX_U32, MAX_U32]),
        ("scalar_all", SCALAR, [-1]),
        ("scalar_column", SCALAR, [-1, 1]),
        ("scalar_rank0", SCALAR, []),
        ("scalar_indivisible", SCALAR, [-1, 2]),
    ):
        rows.append(request(f"reshape_{name}", tensor, "reshape", sizes=sizes))
    for name, tensor, dims in (
        ("last_first", GRID, [-1, 0]),
        ("both_negative", GRID, [-2, -1]),
        ("swapped", GRID, [-1, -2]),
        ("cube", CUBE, [-1, 0, -2]),
        ("cube_mixed", CUBE, [2, -3, 1]),
        ("repeated", GRID, [1, -1]),
        ("below", GRID, [0, -3]),
        ("below_first", GRID, [-3, 0]),
        ("past", GRID, [0, 2]),
        ("short", GRID, [-1]),
        ("long", GRID, [0, 1, -1]),
        ("scalar", SCALAR, []),
        ("scalar_axis", SCALAR, [-1]),
        ("range_after_repeat", CUBE, [0, -3, 5]),
    ):
        rows.append(request(f"permute_{name}", tensor, "permute", dims=dims))
    for name, tensor, dims in (
        ("last_first", GRID, [-1, 0]),
        ("first_last", GRID, [0, -1]),
        ("both_negative", GRID, [-2, -1]),
        ("same", GRID, [-1, -1]),
        ("mixed", GRID, [1, -2]),
        ("cube", CUBE, [-1, -3]),
        ("cube_middle", CUBE, [0, -2]),
        ("below", GRID, [-3, 0]),
        ("past_first", GRID, [2, -3]),
        ("scalar", SCALAR, [0, 0]),
        ("scalar_last", SCALAR, [-1, 0]),
        ("scalar_both", SCALAR, [-1, -1]),
        ("scalar_below", SCALAR, [0, -2]),
        ("scalar_past", SCALAR, [1, 0]),
        ("vector", VECTOR, [-1, 0]),
        ("vector_below", VECTOR, [-2, 0]),
    ):
        rows.append(request(f"transpose_{name}", tensor, "transpose", dims=dims))
    for name, tensor, index in (
        ("last_row", GRID, [-1, 0]),
        ("both_negative", GRID, [-2, -3]),
        ("last_column", GRID, [1, -1]),
        ("cube", CUBE, [-1, -2, -3]),
        ("below_row", GRID, [-3, 0]),
        ("below_column", GRID, [0, -4]),
        ("past_row", GRID, [2, 0]),
        ("past_column", GRID, [-1, 3]),
        ("empty", EMPTY, [-1, 0]),
        ("scalar", SCALAR, []),
    ):
        rows.append(request(f"get_{name}", tensor, "get", index=index))
    for name, tensor, index in (
        ("last", GRID, [-1, -1]),
        ("cube", CUBE, [0, -1, 2]),
        ("below", GRID, [-3, 0]),
    ):
        rows.append(request(f"set_{name}", tensor, "set", index=index, value=word(9.0)))
    for name, tensor, index in (
        ("last", GRID, -1),
        ("first", GRID, -6),
        ("end", GRID, 5),
        ("below", GRID, -7),
        ("past", GRID, 6),
        ("scalar", SCALAR, -1),
        ("scalar_first", SCALAR, 0),
        ("empty", EMPTY_VECTOR, -1),
    ):
        rows.append(request(f"get_flat_{name}", tensor, "get_flat", index=index))
    for name, tensor, index in (("second_last", GRID, -2), ("below", GRID, -7)):
        rows.append(
            request(
                f"set_flat_{name}", tensor, "set_flat", index=index, value=word(9.0)
            )
        )
    return rows


# PyTorch's wrap of a position among `limit` axes or elements; a position
# outside -limit..limit-1 becomes `limit` if negative and stays itself
# otherwise, as signed.bend's wrap.
def wrap(position: int, limit: int) -> int:
    if position >= 0:
        return position
    return position + limit if position >= -limit else limit


def inferred(count: int, sizes: list[int]) -> list[int] | int:
    if sizes.count(-1) > 1:
        return REPEATED_INFERRED
    if -1 not in sizes:
        return sizes
    product = math.prod(size for size in sizes if size != -1)
    if product == 0:
        return AMBIGUOUS_INFERRED if count == 0 else ELEMENT_COUNT
    if count % product:
        return ELEMENT_COUNT
    return [count // product if size == -1 else size for size in sizes]


def snapshot(shape: list[int], words: list[int]) -> list[str]:
    return [
        f"tensor:shape:{json.dumps(shape)}",
        f"tensor:count:{len(words)}",
        *[f"tensor:word:{item}" for item in words],
    ]


def refused(code: int) -> list[str]:
    return [f"tensor:error:{code}"]


def strides(shape: list[int]) -> list[int]:
    return [math.prod(shape[axis + 1 :]) for axis in range(len(shape))]


def permuted(shape: list[int], words: list[int], axes: list[int]) -> list[str]:
    if len(axes) != len(shape):
        return refused(AXES_LENGTH)
    if any(axis >= len(shape) for axis in axes):
        return refused(AXIS_OUT_OF_BOUNDS)
    if len(set(axes)) != len(axes):
        return refused(DUPLICATE_AXIS)
    output = [shape[axis] for axis in axes]
    step = strides(shape)
    moved = []
    for flat in range(math.prod(output)):
        coordinates, rest = [], flat
        for extent in reversed(output):
            coordinates.append(rest % extent)
            rest //= extent
        coordinates.reverse()
        source = sum(
            coordinate * step[axis]
            for coordinate, axis in zip(coordinates, axes, strict=True)
        )
        moved.append(words[source])
    return snapshot(output, moved)


def flat_index(shape: list[int], coordinates: list[int]) -> int:
    if len(coordinates) != len(shape):
        return -COORDINATE_RANK
    if any(c >= extent for c, extent in zip(coordinates, shape, strict=True)):
        return -COORDINATE_OUT_OF_BOUNDS
    return sum(c * s for c, s in zip(coordinates, strides(shape), strict=True))


def coordinates(shape: list[int], index: list[int]) -> list[int]:
    wrapped = [wrap(i, extent) for i, extent in zip(index, shape, strict=False)]
    return wrapped + [wrap(i, 0) for i in index[len(shape) :]]


def written(shape: list[int], words: list[int], at: int, value: int) -> list[str]:
    if at < 0:
        return refused(-at)
    if at >= len(words):
        return refused(FLAT_OUT_OF_BOUNDS)
    return snapshot(shape, words[:at] + [value] + words[at + 1 :])


def read(words: list[int], at: int) -> list[str]:
    if at < 0:
        return refused(-at)
    if at >= len(words):
        return refused(FLAT_OUT_OF_BOUNDS)
    return snapshot([], [words[at]])


def reference(case: dict) -> list[str]:
    shape, words, op = case["shape"], case["words"], case["op"]
    if op == "reshape":
        sizes = inferred(len(words), case["sizes"])
        if isinstance(sizes, int):
            return refused(sizes)
        if len(sizes) > MAX_RANK:
            return refused(RANK_EXCEEDED)
        if math.prod(sizes) > MAX_ELEMENTS:
            return refused(CAPACITY_EXCEEDED)
        if math.prod(sizes) != len(words):
            return refused(ELEMENT_COUNT)
        return snapshot(sizes, words)
    if op == "permute":
        return permuted(shape, words, [wrap(d, len(shape)) for d in case["dims"]])
    if op == "transpose":
        limit = max(len(shape), 1)
        a, b = (wrap(d, limit) for d in case["dims"])
        if a >= limit or b >= limit:
            return refused(AXIS_OUT_OF_BOUNDS)
        axes = list(range(len(shape)))
        if axes:
            axes[a], axes[b] = axes[b], axes[a]
        return permuted(shape, words, axes)
    if op == "get":
        return read(words, flat_index(shape, coordinates(shape, case["index"])))
    if op == "set":
        at = flat_index(shape, coordinates(shape, case["index"]))
        return written(shape, words, at, case["value"])
    if op == "get_flat":
        return read(words, wrap(case["index"], len(words)))
    if op == "set_flat":
        return written(shape, words, wrap(case["index"], len(words)), case["value"])
    raise GateError(f"Unknown signed request: {op}")


def position_source(position: int) -> str:
    if position >= 0:
        return f"Signed.First{{{position}}}"
    return f"Signed.Last{{{-1 - position}}}"


def positions_source(positions: list[int]) -> str:
    return "[" + ", ".join(position_source(p) for p in positions) + "]"


# Only -1 is an Inferred size; PyTorch refuses every other negative size, and
# Signed.Extent cannot hold one.
def extent_source(size: int) -> str:
    if size == -1:
        return "Signed.Inferred{}"
    if size < 0:
        raise GateError(f"A reshape size below -1 has no Extent: {size}")
    return f"Signed.Exact{{{size}}}"


def request_source(case: dict) -> str:
    op = case["op"]
    if op == "reshape":
        extents = ", ".join(extent_source(size) for size in case["sizes"])
        return f"Harness.Reshape{{[{extents}]}}"
    if op == "permute":
        return f"Harness.Permute{{{positions_source(case['dims'])}}}"
    if op == "transpose":
        a, b = (position_source(d) for d in case["dims"])
        return f"Harness.Transpose{{{a}, {b}}}"
    if op == "get":
        return f"Harness.Get{{{positions_source(case['index'])}}}"
    if op == "set":
        return f"Harness.Set{{{positions_source(case['index'])}, {case['value']}}}"
    if op == "get_flat":
        return f"Harness.GetFlat{{{position_source(case['index'])}}}"
    if op == "set_flat":
        index = position_source(case["index"])
        return f"Harness.SetFlat{{{index}, {case['value']}}}"
    raise GateError(f"Unknown signed request: {op}")


def program(selected: list[dict], directory: Path, source_root: Path) -> str:
    if not selected:
        raise GateError("Refusing to generate an empty signed corpus")
    lines = ["import Base"]
    for path, alias in (
        ("src/tensor/signed-harness.bend", "Harness"),
        ("src/tensor/harness.bend", "Tensors"),
        ("src/tensor/core.bend", "Core"),
        ("src/tensor/signed.bend", "Signed"),
    ):
        relative = os.path.relpath(source_root / path, directory)
        lines.append(f"import ./{relative} as {alias}")
    for index, case in enumerate(selected):
        words = ", ".join(f"Tensors.from_bits({item})" for item in case["words"])
        tensor = f"Core.construct({json.dumps(case['shape'])}, [{words}])"
        lines += [
            "",
            f"def run_case_{index}() -> IO(Unit):",
            "  do IO<Unit>:",
            f'    IO.print("tensor:begin:{case["id"]}")',
            f"    Harness.run({request_source(case)}, {tensor})",
            '    IO.print("tensor:end")',
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


def parse(text: str, identifiers: list[str]) -> list[dict]:
    rows = text.splitlines()
    results = []
    index = 0
    while index < len(rows):
        begin = rows[index]
        if not begin.startswith("tensor:begin:"):
            raise GateError(f"Unframed signed output: {begin!r}")
        try:
            end = rows.index("tensor:end", index)
        except ValueError as error:
            raise GateError(f"{begin}: missing tensor:end") from error
        identifier = begin.removeprefix("tensor:begin:")
        results.append({"id": identifier, "lines": rows[index + 1 : end]})
        index = end + 1
    if [row["id"] for row in results] != identifiers:
        raise GateError("Signed inventory or order differs from the requests")
    return results


# Every request only moves or selects words, so the native lane must print
# the reference's words bit for bit, NaN payloads included. JavaScript keeps
# no NaN payload, so there a NaN only has to be a NaN.
def same_row(seen: str, wanted: str, native: bool) -> bool:
    prefix = "tensor:word:"
    if native or not (seen.startswith(prefix) and wanted.startswith(prefix)):
        return seen == wanted
    left, right = int(seen.removeprefix(prefix)), int(wanted.removeprefix(prefix))
    return left == right or (is_nan(left) and is_nan(right))


def verify(selected: list[dict], observations: list[dict], lane: str) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete signed case inventory")
    summary = {"cases": len(selected), "errors": 0, "words": 0}
    for case, row in zip(selected, observations, strict=True):
        expected = reference(case)
        seen = row["lines"]
        if len(seen) != len(expected) or not all(
            same_row(a, b, lane == "native")
            for a, b in zip(seen, expected, strict=True)
        ):
            raise GateError(f"{lane} {case['id']}: output differs from the reference")
        summary["errors"] += sum(line.startswith("tensor:error:") for line in expected)
        summary["words"] += sum(line.startswith("tensor:word:") for line in expected)
    return summary


def run_gate(tool_root: Path, output: Path) -> dict:
    selected = cases()

    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        return observations, verify(selected, observations, lane)

    return run_lanes(tool_root, output, "signed", selected, render, check)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
