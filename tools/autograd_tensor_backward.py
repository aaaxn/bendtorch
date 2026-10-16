import argparse
import json
import math
import os
import shutil
import struct
from pathlib import Path

from tools.autograd_backward import (
    DANGLING_OPERAND,
    NEGATIVE_ZERO,
    ONE,
    OTHER_NAN,
    QUIET_NAN,
    UNKNOWN_NODE,
    add_word,
    mul_word,
    neg_word,
    sub_word,
)
from tools.autograd_broadcast import (
    CAPACITY_EXCEEDED,
    INCOMPATIBLE_BROADCAST,
    MAX_ELEMENTS,
    MAX_RANK,
    RANK_EXCEEDED,
    is_nan,
    run_lanes,
    word,
)
from tools.common import ROOT, Evidence, GateError, bend

VALUE_COUNT_MISMATCH = 10
MATMUL_RANK = 105
MATMUL_INNER = 106
INFINITY = 0x7F800000
BINARY = ("add", "sub", "mul", "matmul", "matmul_nt")
UNARY = ("sum", "relu")

RUNTIME_CONTROLS = {
    "seed-newest-node": (
        "ran(pass(entries, count, output, at_head(count, output)))",
        "ran(pass(entries, count, output, True{}))",
    ),
    "mul-vjp-swaps-operands": (
        "multiplied_first(a, b, left, right, x, Elementwise.mul(g, y))",
        "multiplied_first(a, b, left, right, y, Elementwise.mul(g, x))",
    ),
    "sub-vjp-drops-sign": (
        "Elementwise.binary(Elementwise.NegateLeft{}, g,",
        "Elementwise.binary(Elementwise.KeepLeft{}, g,",
    ),
    "matmul-vjp-skips-transpose": (
        "matmul_second(b, right, x, g, left_product(a, left, copy, "
        "Transforms.transpose_dynamic(y, 0, 1)))",
        "matmul_second(b, right, x, g, left_product(a, left, copy, "
        "Transforms.transpose_dynamic(y, 0, 0)))",
    ),
    "nt-reads-weight-untransposed": (
        "src/tensor/matmul.bend",
        "  product(Kept{}, Kept{}, left, right)",
        "  product(Kept{}, Swapped{}, left, right)",
    ),
    "nt-vjp-tn-swaps-operands": (
        "(left, multiplied(target, shape, Matrix.matmul_tn(copy, flipped)))",
        "(left, multiplied(target, shape, Matrix.matmul_tn(flipped, copy)))",
    ),
    "nt-vjp-weight-gradient-untransposed": (
        "src/tensor/matmul.bend",
        "  product(Swapped{}, Swapped{}, left, right)",
        "  product(Kept{}, Swapped{}, left, right)",
    ),
    "linear-drops-bias": (
        "unrecorded(add(tape, product, bias))",
        "unrecorded((tape, Done{product}))",
    ),
    "linear-resolves-bias-after-recording": (
        "linear_checked(x, w, bias, entries, Nat.is_lt(bias, count))",
        "linear_checked(x, w, bias, entries, True{})",
    ),
    "linear-keeps-refused-product": (
        "unrecorded(add(tape, product, bias))",
        "add(tape, product, bias)",
    ),
    "linear-bias-gradient-sums-rows": (
        "src/autograd/reduce.bend",
        "        fiber(rest, others, first_span(rest, others), inner, "
        "Nat.add(base, Nat.mul(right, digit)), first_digit(rest, inner),",
        "        fiber(rest, others, first_span(rest, others), inner, "
        "Nat.add(base, Nat.mul(left, digit)), first_digit(rest, inner),",
    ),
    "merge-overwrites-slot": (
        "merged_entry(node, shape, value, saved, Gradient.accumulate(shape, slot, g), "
        "merge(rest, more))",
        "merged_entry(node, shape, value, saved, "
        "Gradient.Accumulated{Gradient.Held{g}}, merge(rest, more))",
    ),
    "relu-passes-negatives": (
        "src/tensor/elementwise.bend",
        "  zero_if(below_zero(x), x)",
        "  zero_if(False{}, x)",
    ),
    "relu-zeroes-negative-zero": (
        "src/tensor/elementwise.bend",
        "Bool.and(U32.is_lt(2147483648, word), U32.is_le(word, 4286578688))",
        "Bool.and(U32.is_le(2147483648, word), U32.is_le(word, 4286578688))",
    ),
    "relu-backward-passes-gradient-at-zero": (
        "src/tensor/elementwise.bend",
        "  zero_if(at_most_zero(x), g)",
        "  zero_if(below_zero(x), g)",
    ),
    "relu-vjp-reads-gradient-as-input": (
        "relu_passed(a, input, zero, "
        "Elementwise.binary(Elementwise.ReluBackward{}, x, g))",
        "relu_passed(a, input, zero, "
        "Elementwise.binary(Elementwise.ReluBackward{}, g, x))",
    ),
    "reduction-folds-equal-shapes": (
        "delivered_as(target, shape, SumToSize.sum_to_size(g, shape))",
        "delivered_as(target, shape, Reduce.reduce_to(g, shape))",
    ),
}


def f(*numbers: float | int) -> list[int]:
    return [number if isinstance(number, int) else word(number) for number in numbers]


def leaf(shape: list[int], words: list[int]) -> dict:
    return {"op": "leaf", "shape": shape, "words": words}


def op(name: str, a: int, b: int | None = None) -> dict:
    return {"op": name, "a": a} if b is None else {"op": name, "a": a, "b": b}


def run(output: int) -> dict:
    return {"op": "backward", "output": output}


def reads(count: int) -> list[dict]:
    return [{"op": "read", "id": index} for index in range(count)]


def forge(node: dict, shape: list[int], words: list[int]) -> dict:
    return {"op": "forge", "node": node, "shape": shape, "words": words}


ZERO_GRAD = {"op": "zero_grad"}


def linear(x: int, w: int, bias: int) -> dict:
    return {"op": "linear", "a": x, "b": w, "bias": bias}


def program_case(identifier: str, steps: list[dict], output: int) -> dict:
    count = sum(step["op"] in ("leaf", *UNARY, *BINARY) for step in steps)
    count += 2 * sum(step["op"] == "linear" for step in steps)
    return {"id": identifier, "steps": [*steps, run(output), *reads(count)]}


def handwritten() -> list[dict]:
    grid = leaf([2, 3], f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    row = leaf([3], f(0.5, -1.0, 2.0))
    column = leaf([2, 1], f(-2.0, 0.25))
    scalar = leaf([], f(3.0))
    unused = [scalar, leaf([], f(2.0)), leaf([3], f(1.0, 2.0, 3.0))]
    unused += [op("mul", 0, 1), op("add", 2, 2)]
    left = leaf([2, 3], f(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
    right = leaf([3, 2], f(2.0, 1.0, -1.0, 4.0, 0.5, -3.0))
    weight = leaf([3, 4], f(*[0.5 * k - 2.0 for k in range(12)]))
    bias = leaf([4], f(0.25, -0.5, 1.0, 2.0))
    layer = leaf([4, 3], f(*[0.5 * k - 2.0 for k in range(12)]))
    layer_bias = leaf([4], f(0.25, -0.5, 1.0, 2.0))
    return [
        program_case(
            "mul_broadcast_row", [grid, row, op("mul", 0, 1), op("sum", 2)], 3
        ),
        program_case(
            "add_broadcast_row", [grid, row, op("add", 0, 1), op("sum", 2)], 3
        ),
        program_case(
            "sub_broadcast_column", [grid, column, op("sub", 0, 1), op("sum", 2)], 3
        ),
        program_case(
            "mul_broadcast_column", [column, grid, op("mul", 0, 1), op("sum", 2)], 3
        ),
        program_case("sub_scalar", [row, scalar, op("sub", 0, 1), op("sum", 2)], 3),
        program_case(
            "mul_keepdim_row",
            [grid, leaf([1, 3], f(2.0, -0.5, 4.0)), op("mul", 0, 1), op("sum", 2)],
            3,
        ),
        program_case("seed_non_scalar", [grid, row, op("mul", 0, 1)], 2),
        program_case("square", [row, op("mul", 0, 0), op("sum", 1)], 2),
        program_case("add_self", [row, op("add", 0, 0), op("sum", 1)], 2),
        program_case("sub_self", [row, op("sub", 0, 0), op("sum", 1)], 2),
        program_case("matmul", [left, right, op("matmul", 0, 1), op("sum", 2)], 3),
        program_case("matmul_seed", [left, right, op("matmul", 0, 1)], 2),
        program_case(
            "matmul_square",
            [leaf([2, 2], f(1.0, -2.0, 0.5, 3.0)), op("matmul", 0, 0), op("sum", 1)],
            2,
        ),
        program_case(
            "linear",
            [left, weight, bias, op("matmul", 0, 1), op("add", 3, 2), op("sum", 4)],
            5,
        ),
        # Linear in PyTorch's layout: W is [out, in], so the product reads W
        # transposed in place, and the bias is an Add of [out] to every row.
        program_case(
            "linear_layer",
            [left, layer, layer_bias, linear(0, 1, 2), op("sum", 4)],
            5,
        ),
        program_case("linear_seed", [left, layer, layer_bias, linear(0, 1, 2)], 4),
        program_case(
            "linear_rounds",
            [
                leaf([3, 2], f(0.1, 0.2, 0.3, 0.7, 1.0 / 3.0, -0.6)),
                leaf([2, 2], f(1.0 / 3.0, 3.0, -0.1, 7.0)),
                leaf([2], f(0.1, -0.3)),
                linear(0, 1, 2),
                op("mul", 4, 4),
                op("sum", 5),
            ],
            6,
        ),
        program_case(
            "linear_two_layers",
            [
                left,
                layer,
                layer_bias,
                leaf([2, 4], f(*[0.25 * k - 1.0 for k in range(8)])),
                leaf([2], f(0.5, -0.25)),
                linear(0, 1, 2),
                linear(6, 3, 4),
                op("sum", 8),
            ],
            9,
        ),
        program_case(
            "matmul_nt_self",
            [left, op("matmul_nt", 0, 0), op("sum", 1)],
            2,
        ),
        program_case(
            "linear_inner_mismatch",
            [left, leaf([4, 2], f(*range(1, 9))), layer_bias, linear(0, 1, 2)],
            0,
        ),
        # A refused linear leaves the tape as it was: the next node takes
        # the id the product would have taken.
        program_case(
            "linear_bias_mismatch",
            [
                left,
                layer,
                leaf([3], f(1.0, 2.0, 3.0)),
                linear(0, 1, 2),
                leaf([], f(2.0)),
                op("mul", 3, 0),
            ],
            4,
        ),
        # The bias id is the incoming entry count, the id the product would
        # take: it is unknown, not the product.
        program_case(
            "linear_bias_next",
            [
                left,
                layer,
                linear(0, 1, 2),
                layer_bias,
                linear(0, 1, 2),
                op("sum", 4),
            ],
            5,
        ),
        program_case(
            "linear_bias_unknown",
            [
                left,
                layer,
                layer_bias,
                linear(0, 1, 9),
                leaf([], f(2.0)),
                op("mul", 3, 0),
                op("sum", 4),
            ],
            5,
        ),
        program_case(
            "matmul_nt_rank",
            [leaf([], f(2.0)), layer, op("matmul_nt", 0, 1)],
            1,
        ),
        program_case(
            "matmul_rounds",
            [
                leaf([2, 2], f(0.1, 0.2, 0.3, 0.7)),
                leaf([2, 2], f(1.0 / 3.0, 3.0, -0.1, 7.0)),
                op("matmul", 0, 1),
                op("mul", 2, 2),
                op("sum", 3),
            ],
            4,
        ),
        program_case(
            "rank_four",
            [
                leaf([2, 1, 2, 1], f(1.0, 2.0, 3.0, 4.0)),
                leaf([1, 3, 1, 2], f(0.5, -1.0, 2.0, 0.25, -4.0, 1.5)),
                op("mul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "empty_rows",
            [leaf([0, 3], []), row, op("add", 0, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "rank_zero",
            [scalar, leaf([], f(-0.5)), op("mul", 0, 1), op("sub", 2, 0)],
            3,
        ),
        program_case("sum_of_scalar", [scalar, op("sum", 0), op("mul", 1, 1)], 2),
        {"id": "seed_selected_output", "steps": [*unused, run(3), *reads(5)]},
        {"id": "seed_newest_when_selected", "steps": [*unused, run(4), *reads(5)]},
        {"id": "leaf_output", "steps": [*unused, run(2), *reads(5)]},
        {"id": "backward_twice", "steps": [*unused, run(3), run(3), *reads(5)]},
        {
            "id": "backward_two_outputs",
            "steps": [*unused, run(3), run(4), run(3), *reads(5)],
        },
        {
            "id": "zero_grad_then_backward",
            "steps": [*unused, run(3), ZERO_GRAD, *reads(5), run(4), *reads(5)],
        },
        program_case(
            "interior_output",
            [
                grid,
                row,
                op("mul", 0, 1),
                op("add", 2, 0),
                op("mul", 3, 3),
                op("sum", 4),
            ],
            3,
        ),
        program_case(
            "fan_out_arrival_order",
            [
                leaf([2], f(1.0, 1.0)),
                leaf([2], f(1.0, -1e8)),
                leaf([2], f(1e8, 1e8)),
                leaf([2], f(-1e8, 1.0)),
                op("mul", 0, 1),
                op("mul", 0, 2),
                op("mul", 0, 3),
                op("add", 4, 5),
                op("add", 7, 6),
            ],
            8,
        ),
        program_case(
            "broadcast_fold_order",
            [
                leaf([4, 1], f(1e8, 1.0, -1e8, 1.0)),
                leaf([4, 2], f(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
                op("mul", 1, 0),
                op("sum", 2),
                leaf([2], f(1.0, 2.0)),
                op("mul", 1, 4),
                op("sum", 5),
            ],
            3,
        ),
        {
            "id": "negative_zero_moves_in",
            "steps": [
                leaf([2], f(4.0, -1.0)),
                leaf([2], f(NEGATIVE_ZERO, 0.0)),
                op("mul", 0, 1),
                op("mul", 1, 1),
                ZERO_GRAD,
                run(2),
                *reads(4),
                run(3),
                *reads(4),
            ],
        },
        program_case(
            "negative_zero_sub",
            [leaf([2], f(NEGATIVE_ZERO, 0.0)), op("sub", 0, 0), op("sub", 1, 0)],
            2,
        ),
        program_case(
            "specials",
            [
                leaf([3], f(INFINITY, 0.0, QUIET_NAN)),
                leaf([3], f(0.0, 2.0, OTHER_NAN)),
                op("mul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "overflow_and_subnormal",
            [
                leaf([2], f(3e38, 1)),
                leaf([2], f(10.0, 0.5)),
                op("mul", 0, 1),
                op("sub", 2, 1),
                op("sum", 3),
            ],
            4,
        ),
        {
            "id": "unknown_operands",
            "steps": [
                row,
                op("add", 0, 5),
                op("mul", 5, 0),
                op("sub", 0, 1),
                op("matmul", 3, 0),
                op("sum", 2),
                op("sum", 0),
                run(1),
                *reads(3),
            ],
        },
        {"id": "unknown_output_empty", "steps": [run(0), *reads(1)]},
        {
            "id": "unknown_output",
            "steps": [row, op("sum", 0), run(2), run(7), *reads(3)],
        },
        {
            "id": "incompatible_shapes",
            "steps": [
                grid,
                leaf([2], f(1.0, 2.0)),
                op("add", 0, 1),
                op("sub", 1, 0),
                op("mul", 0, 1),
                op("matmul", 0, 1),
                op("matmul", 0, 0),
                leaf([2, 2], f(1.0, 2.0, 3.0, 4.0)),
                op("matmul", 2, 0),
                op("matmul", 0, 2),
                op("sum", 0),
                run(3),
                *reads(5),
            ],
        },
        {
            "id": "invalid_leaf",
            "steps": [
                leaf([2, 2], f(1.0, 2.0, 3.0)),
                row,
                op("sum", 0),
                run(1),
                *reads(2),
            ],
        },
        {
            "id": "dangling_self",
            "steps": [
                row,
                forge({"op": "add", "a": 1, "b": 0}, [3], f(1.0, 2.0, 3.0)),
                run(0),
                run(1),
                *reads(2),
            ],
        },
        {
            "id": "dangling_newer",
            "steps": [
                row,
                forge({"op": "sum", "a": 9}, [], f(-2.0)),
                row,
                run(0),
                run(2),
                *reads(3),
            ],
        },
        {
            "id": "dangling_missing",
            "steps": [
                scalar,
                leaf([], f(1.0)),
                forge({"op": "add", "a": 0, "b": 2}, [], f(4.0)),
                run(2),
                *reads(3),
            ],
        },
        # relu at every edge word, times weights that reach the gradient
        # unchanged: the gradient of relu's input is the weight's word where
        # the input is above zero or NaN, and +0.0 where it is at most zero.
        program_case(
            "relu_edges",
            [
                leaf([len(EDGES)], list(EDGES)),
                op("relu", 0),
                leaf([len(EDGES)], [EDGE_WEIGHTS[k % 6] for k in range(len(EDGES))]),
                op("mul", 1, 2),
                op("sum", 3),
            ],
            4,
        ),
        program_case(
            "relu_seed",
            [
                leaf([2, 3], f(-1.0, 0.0, NEGATIVE_ZERO, 2.0, 1, 0x80000001)),
                op("relu", 0),
            ],
            1,
        ),
        program_case(
            "relu_twice_and_fan_out",
            [
                leaf([4], f(-1.5, 0.5, NEGATIVE_ZERO, 3.0)),
                op("relu", 0),
                op("relu", 1),
                op("mul", 2, 0),
                op("sum", 3),
            ],
            4,
        ),
        program_case(
            "relu_rank_zero",
            [leaf([], f(NEGATIVE_ZERO)), op("relu", 0), op("mul", 1, 1)],
            2,
        ),
        program_case(
            "relu_empty",
            [leaf([0, 3], []), op("relu", 0), op("sum", 1)],
            2,
        ),
        # Linear -> ReLU -> Linear, the layers of an MLP.
        program_case(
            "relu_two_layers",
            [
                left,
                layer,
                layer_bias,
                leaf([2, 4], f(*[0.25 * k - 1.0 for k in range(8)])),
                leaf([2], f(0.5, -0.25)),
                linear(0, 1, 2),
                op("relu", 6),
                linear(7, 3, 4),
                op("sum", 9),
            ],
            10,
        ),
        {
            "id": "relu_unknown_operand",
            "steps": [row, op("relu", 3), op("relu", 0), run(1), *reads(2)],
        },
    ]


# The F32 edge words: +0.0, -0.0, the smallest and largest denormals, the
# smallest normal, 1.0, the largest finite, the infinities, a signaling, a
# quiet and the all-ones NaN, each followed by its negative.
EDGES = (
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
    0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
    0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000,
    0x7FFFFFFF, 0xFFFFFFFF,
)  # fmt: skip
EDGE_WEIGHTS = (word(2.5), NEGATIVE_ZERO, word(-3.0), 1, word(0.75), QUIET_NAN)

# Words whose sums and products in these scripts stay exact, so every order
# of addition gives the same word.
EXACT = tuple(
    word(number) for number in (1.0, -2.0, 0.5, 3.0, -0.25, 1.5, -4.0, 0.0, 2.5, -1.0)
)
# Words that round, cancel and overflow.
ROUNDING = (
    word(1.0),
    word(-2.0),
    word(0.5),
    word(3.0),
    word(0.1),
    word(-0.3),
    word(1e8),
    word(-1e8),
    word(7.25),
    word(1e-3),
    0,
    NEGATIVE_ZERO,
)

# Words for programs with relu: both zeros, values on either side of zero
# and a denormal on each side.
RELU_POOL = (
    word(1.0),
    word(-2.0),
    word(0.5),
    word(-0.25),
    word(3.0),
    word(-1.5),
    0,
    NEGATIVE_ZERO,
    1,
    0x80000001,
)

SHAPES = ([], [2], [3], [1, 3], [2, 1], [2, 3], [3, 2], [2, 2])


def generated(
    prefix: str, pool: tuple[int, ...], count: int, seed: int, relu: bool = False
) -> list[dict]:
    state = seed
    result = []

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    for index in range(count):
        shapes: list[list[int]] = []
        steps = []
        for _ in range(2 + draw(3)):
            shape = SHAPES[draw(len(SHAPES))]
            steps.append(
                leaf(shape, [pool[draw(len(pool))] for _ in range(math.prod(shape))])
            )
            shapes.append(shape)
        for _ in range(3 + draw(8)):
            position = len(shapes)
            # With relu, three draws in eight record a relu.
            choice = draw(8 if relu else 5)
            a = draw(position)
            if choice >= 5:
                steps.append(op("relu", a))
                shapes.append(shapes[a])
                continue
            if choice == 4:
                steps.append(op("sum", a))
                shapes.append([])
                continue
            if choice == 3:
                partners = [
                    b
                    for b in range(position)
                    if len(shapes[a]) == 2
                    and len(shapes[b]) == 2
                    and shapes[a][1] == shapes[b][0]
                ]
                if partners:
                    b = partners[draw(len(partners))]
                    steps.append(op("matmul", a, b))
                    shapes.append([shapes[a][0], shapes[b][1]])
                    continue
                choice = draw(3)
            partners = [
                b
                for b in range(position)
                if broadcast(shapes[a], shapes[b]) is not None
            ]
            b = partners[draw(len(partners))]
            steps.append(op(BINARY[choice], a, b))
            shapes.append(broadcast(shapes[a], shapes[b]) or [])
        count = len(shapes)
        outputs = [count - 1 - draw(min(count, 4)) for _ in range(1 + draw(2))]
        steps += [run(output) for output in outputs]
        steps += reads(count)
        if draw(3) == 0:
            steps += [ZERO_GRAD, run(count - 1), *reads(count)]
        result.append({"id": f"{prefix}_{index:02d}", "steps": steps})
    return result


def cases() -> list[dict]:
    result = (
        handwritten()
        + generated("generated", EXACT, 32, 0x5DEECE66)
        + generated("rounding", ROUNDING, 24, 0x2545F491)
        + generated("relu", RELU_POOL, 16, 0x1F2E3D4C, relu=True)
    )
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate autograd tensor backward case identifier")
    return result


# Tensors in the reference are (shape, words) pairs of F32 bit words.


# IEEE comparisons with zero, through the binary32 value: a NaN compares false.
def below_zero(number: int) -> bool:
    return struct.unpack("<f", struct.pack("<I", number))[0] < 0


def at_most_zero(number: int) -> bool:
    return struct.unpack("<f", struct.pack("<I", number))[0] <= 0


# PyTorch's relu, clamp_min(x, 0): +0.0 where x < 0, else x's own word.
def relu_word(number: int) -> int:
    return 0 if below_zero(number) else number


# PyTorch's threshold_backward(g, relu(x), 0): +0.0 where x <= 0, else g.
def relu_backward_word(x: int, g: int) -> int:
    return 0 if at_most_zero(x) else g


def broadcast(left: list[int], right: list[int]) -> list[int] | None:
    rank = max(len(left), len(right))
    padded_left = [1] * (rank - len(left)) + left
    padded_right = [1] * (rank - len(right)) + right
    output = []
    for a, b in zip(padded_left, padded_right, strict=True):
        if a != b and a != 1 and b != 1:
            return None
        output.append(b if a == 1 else a)
    return output


def source_index(shape: list[int], output: list[int], flat: int) -> int:
    coordinate = []
    for extent in reversed(output):
        coordinate.append(flat % extent)
        flat //= extent
    coordinate.reverse()
    padded = [1] * (len(output) - len(shape)) + shape
    position = 0
    for extent, index in zip(padded, coordinate, strict=True):
        position = position * extent + (0 if extent == 1 else index)
    return position


def elementwise(combine, left: tuple, right: tuple) -> tuple | int:
    output = broadcast(left[0], right[0])
    if output is None:
        return INCOMPATIBLE_BROADCAST
    if len(output) > MAX_RANK:
        return RANK_EXCEEDED
    if math.prod(output) > MAX_ELEMENTS:
        return CAPACITY_EXCEEDED
    words = [
        combine(
            left[1][source_index(left[0], output, flat)],
            right[1][source_index(right[0], output, flat)],
        )
        for flat in range(math.prod(output))
    ]
    return (output, words)


# sum_to_size: the words themselves when the shapes agree, otherwise the
# ordered fold of reduce_to: each target cell adds its fiber in flat order,
# from +0.0.
def sum_to_size(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    if shape == target:
        return (target, list(words))
    totals = [0] * math.prod(target)
    for flat in range(math.prod(shape)):
        position = source_index(target, shape, flat)
        totals[position] = add_word(totals[position], words[flat])
    return (target, totals)


def broadcast_to(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    return (
        target,
        [words[source_index(shape, target, flat)] for flat in range(math.prod(target))],
    )


def total(words: list[int]) -> int:
    accumulator = 0
    for number in words:
        accumulator = add_word(accumulator, number)
    return accumulator


def matmul(left: tuple, right: tuple) -> tuple | int:
    if len(left[0]) != 2 or len(right[0]) != 2:
        return MATMUL_RANK
    (rows, inner), (other, columns) = left[0], right[0]
    if inner != other:
        return MATMUL_INNER
    words = []
    for row in range(rows):
        for column in range(columns):
            accumulator = 0
            for k in range(inner):
                accumulator = add_word(
                    accumulator,
                    mul_word(left[1][row * inner + k], right[1][k * columns + column]),
                )
            words.append(accumulator)
    return ([rows, columns], words)


def transpose(tensor: tuple) -> tuple:
    (rows, columns), words = tensor
    return (
        [columns, rows],
        [
            words[row * columns + column]
            for column in range(columns)
            for row in range(rows)
        ],
    )


# The VJP of a recorded node only combines tensors whose shapes the recording
# already accepted, so a refusal here is a broken reference.
def accepted(result: tuple | int) -> tuple:
    if isinstance(result, int):
        raise GateError(f"A VJP tensor operation was refused with code {result}")
    return result


def vjp(node: dict, gradient: tuple) -> list[tuple[int, tuple]]:
    name = node["op"]
    if name == "leaf":
        return []
    if name == "sum":
        return [(node["a"], broadcast_to(gradient, node["left"]))]
    if name == "relu":
        passed = accepted(elementwise(relu_backward_word, node["saved"][0], gradient))
        return [(node["a"], sum_to_size(passed, node["left"]))]
    if name == "add":
        return [
            (node["a"], sum_to_size(gradient, node["left"])),
            (node["b"], sum_to_size(gradient, node["right"])),
        ]
    if name == "sub":
        negated = (gradient[0], [neg_word(number) for number in gradient[1]])
        return [
            (node["a"], sum_to_size(gradient, node["left"])),
            (node["b"], sum_to_size(negated, node["right"])),
        ]
    if name == "mul":
        first = accepted(elementwise(mul_word, gradient, node["saved"][1]))
        second = accepted(elementwise(mul_word, gradient, node["saved"][0]))
        return [
            (node["a"], sum_to_size(first, node["left"])),
            (node["b"], sum_to_size(second, node["right"])),
        ]
    left, right = node["saved"]
    if name == "matmul_nt":
        return [
            (node["a"], accepted(matmul(gradient, right))),
            (node["b"], accepted(matmul(transpose(gradient), left))),
        ]
    return [
        (node["a"], accepted(matmul(gradient, transpose(right)))),
        (node["b"], accepted(matmul(transpose(left), gradient))),
    ]


def accumulate(held: tuple | None, contribution: tuple | None) -> tuple | None:
    if held is None:
        return contribution
    if contribution is None:
        return held
    return (
        held[0],
        [add_word(a, b) for a, b in zip(held[1], contribution[1], strict=True)],
    )


def gradients(nodes: list[dict], output: int) -> list[tuple | None]:
    found: list[tuple | None] = [None] * len(nodes)
    shape = nodes[output]["shape"]
    found[output] = (shape, [ONE] * math.prod(shape))
    for position in range(output, -1, -1):
        gradient = found[position]
        if gradient is None:
            continue
        for target, contribution in vjp(nodes[position], gradient):
            if target < position:
                found[target] = accumulate(found[target], contribution)
    return found


def snapshot(tensor: tuple) -> list[str]:
    shape, words = tensor
    return [
        f"tensor:shape:{json.dumps(shape)}",
        f"tensor:count:{len(words)}",
        *[f"tensor:word:{number}" for number in words],
    ]


def operands(node: dict) -> list[int]:
    return [node[key] for key in ("a", "b") if key in node]


def forward(name: str, values: list[tuple]) -> tuple | int:
    if name == "sum":
        return ([], [total(values[0][1])])
    if name == "relu":
        return (values[0][0], [relu_word(number) for number in values[0][1]])
    if name == "matmul":
        return matmul(*values)
    if name == "matmul_nt":
        left, right = values
        if len(left[0]) != 2 or len(right[0]) != 2:
            return MATMUL_RANK
        return matmul(left, transpose(right))
    combine = {"add": add_word, "sub": sub_word, "mul": mul_word}[name]
    return elementwise(combine, *values)


def reference(case: dict) -> list[str]:
    nodes: list[dict] = []
    slots: list[tuple | None] = []
    lines = []

    def push(node: dict, shown: bool = True) -> None:
        if shown:
            lines.append(f"autograd:id:{len(nodes)}")
        nodes.append(node)
        slots.append(None)

    def record(name: str, step: dict, shown: bool = True) -> bool:
        ids = operands(step)
        if any(identifier >= len(nodes) for identifier in ids):
            lines.append(f"autograd:error:{UNKNOWN_NODE}")
            return False
        values = [nodes[identifier]["value"] for identifier in ids]
        result = forward(name, values)
        if isinstance(result, int):
            lines.append(f"autograd:error:{result}")
            return False
        node = {"op": name, "a": step["a"], "shape": result[0], "value": result}
        node["left"] = values[0][0]
        if name not in UNARY:
            node |= {"b": step["b"], "right": values[1][0]}
        if name in ("mul", "matmul", "matmul_nt", "relu"):
            node["saved"] = values
        push(node, shown)
        return True

    for step in case["steps"]:
        name = step["op"]
        if name == "leaf":
            shape, words = step["shape"], step["words"]
            if len(shape) > MAX_RANK:
                lines.append(f"autograd:error:{RANK_EXCEEDED}")
            elif math.prod(shape) > MAX_ELEMENTS:
                lines.append(f"autograd:error:{CAPACITY_EXCEEDED}")
            elif len(words) != math.prod(shape):
                lines.append(f"autograd:error:{VALUE_COUNT_MISMATCH}")
            else:
                push({"op": "leaf", "shape": shape, "value": (shape, list(words))})
        elif name in BINARY or name in UNARY:
            record(name, step)
        elif name == "linear":
            # The product, then the bias added to it; only the sum's id is
            # printed. Every id is checked against the incoming tape, and a
            # refused sum drops the product, so a refusal leaves the tape as
            # it was.
            if any(step[key] >= len(nodes) for key in ("a", "b", "bias")):
                lines.append(f"autograd:error:{UNKNOWN_NODE}")
            elif record("matmul_nt", step, shown=False) and not record(
                "add", {"a": len(nodes) - 1, "b": step["bias"]}
            ):
                nodes.pop()
                slots.pop()
        elif name == "forge":
            node = dict(step["node"])
            node |= {"shape": step["shape"], "value": (step["shape"], step["words"])}
            node |= {"left": step["shape"], "right": step["shape"]}
            lines.append("autograd:forged")
            nodes.append(node)
            slots.append(None)
        elif name == "backward":
            output = step["output"]
            if output >= len(nodes):
                lines.append(f"autograd:error:{UNKNOWN_NODE}")
            elif any(
                target >= position
                for position, node in enumerate(nodes)
                for target in operands(node)
            ):
                lines.append(f"autograd:error:{DANGLING_OPERAND}")
            else:
                found = gradients(nodes, output)
                slots = [accumulate(a, b) for a, b in zip(slots, found, strict=True)]
                lines.append("autograd:backward")
        elif name == "zero_grad":
            slots = [None] * len(nodes)
            lines.append("autograd:zero_grad")
        else:
            identifier = step["id"]
            if identifier >= len(nodes):
                lines += [
                    f"autograd:value:error:{UNKNOWN_NODE}",
                    f"autograd:grad:error:{UNKNOWN_NODE}",
                ]
                continue
            slot = slots[identifier]
            lines += ["autograd:value", *snapshot(nodes[identifier]["value"])]
            lines += (
                ["autograd:grad:none"]
                if slot is None
                else ["autograd:grad", *snapshot(slot)]
            )
    return lines


def list_source(numbers: list[int]) -> str:
    return "[" + ", ".join(str(number) for number in numbers) + "]"


def step_source(step: dict) -> str:
    name = step["op"]
    if name == "leaf":
        tensor = f"{list_source(step['shape'])}, {list_source(step['words'])}"
        return f"Harness.Leaf{{{tensor}}}"
    if name == "backward":
        return f"Harness.Run{{{step['output']}n}}"
    if name == "zero_grad":
        return "Harness.ZeroGrad{}"
    if name == "read":
        return f"Harness.Read{{{step['id']}n}}"
    if name == "forge":
        node = step["node"]
        tensor = f"{list_source(step['shape'])}, {list_source(step['words'])}"
        if node["op"] == "add":
            return f"Harness.ForgeAdd{{{node['a']}n, {node['b']}n, {tensor}}}"
        return f"Harness.ForgeSum{{{node['a']}n, {tensor}}}"
    if name == "linear":
        return f"Harness.Linear{{{step['a']}n, {step['b']}n, {step['bias']}n}}"
    fields = ", ".join(f"{identifier}n" for identifier in operands(step))
    title = "".join(part.capitalize() for part in name.split("_"))
    return f"Harness.{title}{{{fields}}}"


def program(selected: list[dict], directory: Path, source_root: Path) -> str:
    if not selected:
        raise GateError("Refusing to generate an empty autograd tensor backward corpus")
    lines = ["import Base"]
    for path, alias in (
        ("src/autograd/tensor-backward-harness.bend", "Harness"),
        ("src/autograd/tensor-backward.bend", "Backward"),
    ):
        relative = os.path.relpath(source_root / path, directory)
        lines.append(f"import ./{relative} as {alias}")
    for index, case in enumerate(selected):
        steps = ", ".join(step_source(step) for step in case["steps"])
        lines += [
            "",
            f"def run_case_{index}() -> IO(Unit):",
            "  do IO<Unit>:",
            f'    IO.print("autograd:begin:{case["id"]}")',
            f"    Harness.script([{steps}], Backward.empty())",
            '    IO.print("autograd:end")',
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
        if not begin.startswith("autograd:begin:"):
            raise GateError(f"Unframed autograd output: {begin!r}")
        try:
            end = rows.index("autograd:end", index)
        except ValueError as error:
            raise GateError(f"{begin}: missing autograd:end") from error
        results.append(
            {
                "id": begin.removeprefix("autograd:begin:"),
                "lines": rows[index + 1 : end],
            }
        )
        index = end + 1
    if [row["id"] for row in results] != identifiers:
        raise GateError(
            "Autograd tensor backward inventory or order differs from the requests"
        )
    return results


# Both lanes must match every word bit for bit, except that a NaN only needs
# to be NaN on both sides: the tensor operations and the gradient slots this
# engine builds on do not specify NaN payloads.
def same_row(seen: str, wanted: str) -> bool:
    prefix = "tensor:word:"
    if seen.startswith(prefix) and wanted.startswith(prefix):
        left, right = seen.removeprefix(prefix), wanted.removeprefix(prefix)
        if left.isdigit() and right.isdigit():
            return left == right or (is_nan(int(left)) and is_nan(int(right)))
    return seen == wanted


def verify(selected: list[dict], observations: list[dict], lane: str) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete autograd tensor backward case inventory")
    summary = {
        "cases": len(selected),
        "backward": 0,
        "errors": 0,
        "none_reads": 0,
        "words": 0,
    }
    for case, row in zip(selected, observations, strict=True):
        expected = reference(case)
        seen = row["lines"]
        if len(seen) != len(expected) or not all(
            same_row(a, b) for a, b in zip(seen, expected, strict=True)
        ):
            raise GateError(f"{lane} {case['id']}: output differs from the reference")
        summary["backward"] += expected.count("autograd:backward")
        summary["errors"] += sum(
            line.startswith(("autograd:error:", "autograd:value:error:"))
            for line in expected
        )
        summary["none_reads"] += expected.count("autograd:grad:none")
        summary["words"] += sum(line.startswith("tensor:word:") for line in expected)
    return summary


def run_gate(tool_root: Path, output: Path) -> dict:
    selected = cases()

    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        return observations, verify(selected, observations, lane)

    summary = run_lanes(tool_root, output, "tensor-backward", selected, render, check)
    summary["runtime_controls"] = runtime_controls(
        tool_root.resolve(), output.resolve() / "runtime-controls", selected
    )
    return summary


def runtime_controls(tool_root: Path, output: Path, selected: list[dict]) -> list[str]:
    evidence = Evidence(output, tool_root)
    rejected = []
    for name, control in RUNTIME_CONTROLS.items():
        path, original, mutated = (
            control
            if len(control) == 3
            else ("src/autograd/tensor-backward.bend", *control)
        )
        copy = output / name
        shutil.copytree(ROOT / "src", copy / "src")
        target = copy / path
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: runtime control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        source = copy / "tensor-backward.bend"
        source.write_text(program(selected, copy, copy))
        script = copy / "tensor-backward.js"
        evidence.run(f"{name}-javascript-build", bend(tool_root, source, "-o", script))
        result = evidence.run(
            f"{name}-javascript-run", [bend(tool_root)[0], str(script)], timeout=1800
        )
        try:
            verify(selected, parse(result.stdout, [c["id"] for c in selected]), name)
        except GateError:
            rejected.append(name)
            continue
        raise GateError(f"{name}: the tensor backward lane accepted a broken engine")
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
