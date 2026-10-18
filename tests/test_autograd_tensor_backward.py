import hashlib
import json
import math
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_backward import DANGLING_OPERAND, NEGATIVE_ZERO, UNKNOWN_NODE
from tools.autograd_broadcast import CAPACITY_EXCEEDED as CAPACITY
from tools.autograd_broadcast import RANK_EXCEEDED, is_nan, word
from tools.autograd_tensor_backward import (
    AXIS_OUT_OF_BOUNDS,
    CHAIN_CONTROLS,
    DUPLICATE_AXIS,
    INDEX_RANK_MISMATCH,
    INDEX_SHAPE_MISMATCH,
    INVALID_LEARNING_RATE,
    MATMUL_INNER,
    MATMUL_RANK,
    NATIVE_CONTROLS,
    NOT_LEAF,
    NOT_LOGITS,
    OUT_OF_RANGE,
    RUNTIME_CONTROLS,
    SGD_CONTROLS,
    STEP_NOT_POSITIVE,
    TARGET_OUT_OF_RANGE,
    VALUE_COUNT_MISMATCH,
    ZERO_GRAD,
    capacity_scripts,
    cases,
    descend_word,
    forge,
    leaf,
    op,
    parse,
    reads,
    reference,
    run,
    run_gate,
    same_row,
    source_index,
    verify,
)
from tools.autograd_tensor_bound import (
    REPLACED,
    Bounded,
    add_word,
    bounded_dot,
    bounded_lines,
    bounded_matmul,
    bounded_reference,
    bounded_total,
    bounded_word,
    compare,
    mul_word,
    within,
)
from tools.common import ROOT, GateError

FIXTURE = ROOT / "oracle/autograd/fixtures/tensor-backward-torch.json"
FIXTURE_SHA256 = "6c88564b86f37b5ab271df7360527c6e0ee38200b388f38ac3b858484d85587e"
# The tape answers UnknownNode where the replay indexes past its Python list
# (IndexError), and the tensor layer's error codes where PyTorch raises
# RuntimeError: shapes that do not broadcast (6), a matmul rank (105) or inner
# dimension (106) mismatch, and a leaf with the wrong number of values (10);
# InvalidLearningRate (110) where torch.optim.SGD raises ValueError.
REJECTIONS = {
    "unknown_operands",
    "unknown_output_empty",
    "unknown_output",
    "incompatible_shapes",
    "invalid_leaf",
    "linear_inner_mismatch",
    "linear_bias_mismatch",
    "linear_bias_next",
    "linear_bias_unknown",
    "matmul_nt_rank",
    "relu_unknown_operand",
    "mse_refused",
    "sgd_unknown_param",
    "sgd_rate_negative",
    "batched_refusals",
}
RUNTIME_ERROR_CODES = {6, MATMUL_RANK, MATMUL_INNER, 10}
# Refusals step by step, as PyTorch's kind and the tape's code. Those of
# cross_entropy_refused: a target equal to the columns (IndexError, 108), too
# few and too many targets (ValueError, 108), logits of rank 3 (RuntimeError,
# 107) and an unknown id (IndexError, 103). Those of sum_dims_refused: an
# unknown id, an axis past the rank (IndexError, AxisOutOfBounds 13), an axis
# listed twice (RuntimeError, DuplicateAxis 14), a list with both (13, as
# PyTorch checks every axis's range first), the U32 word of -1 (13) and axis
# 1 of a rank-0 input (13). Those of sum_dims_negative_refused: -3 on rank 2
# (13), -1 and 1 (14), 0 and -2 (14), 5 and -3 (13), -3 before a repeated 0
# (13), -2 on rank 0 (13), and 0 and -1 on rank 0, both axis 0 (14). Those of
# slice_refused: an unknown id, an axis past the rank (IndexError, 13), a step
# of 0 (ValueError, StepNotPositive 19), a slice of a rank-0 input
# (IndexError, 13); a narrow past the extent (RuntimeError, OutOfRange 20),
# one whose start is the largest U32 (IndexError, 20), one past the rank
# (IndexError, 13) and one of a rank-0 input (RuntimeError, 13); a select at
# the extent (IndexError, 20), one past the rank and one of a rank-0 input
# (IndexError, 13). Those of slice_negative_refused: a slice at an axis
# before -rank and of a rank-0 input (IndexError, 13); a narrow start before
# -extent with and without a length (IndexError, 20), one at -extent whose
# length passes the extent (RuntimeError, 20), and one at an axis before
# -rank (IndexError, 13); a select index before -extent (IndexError, 20), at
# an axis before -rank and of a rank-0 input (IndexError, 13). Those of
# gather_refused: an index_select at an axis past the rank and before -rank
# (IndexError, 13) and with an entry at the extent (RuntimeError,
# OutOfRange 20); a gather whose entry count is not the product of its
# dimensions (RuntimeError from the replay's reshape, ValueCountMismatch 10),
# at an axis past the rank (IndexError, 13), of another rank (RuntimeError,
# IndexRankMismatch 21), wider than the input at another axis (RuntimeError,
# IndexShapeMismatch 22), with an entry at the extent (RuntimeError, 20);
# an unknown id (IndexError, 103); an index past 2^20 elements and one past
# rank 4 (RuntimeError from the replay's reshape and from the rank check,
# CapacityExceeded 2 and RankExceeded 1, the index built as a tensor is),
# and both again at an unknown id, as the index is built first (PyTorch
# builds a rank-5 index, so it raises IndexError for the id there).
STEP_REJECTIONS = {
    "cross_entropy_refused": [
        ("IndexError", TARGET_OUT_OF_RANGE),
        ("ValueError", TARGET_OUT_OF_RANGE),
        ("ValueError", TARGET_OUT_OF_RANGE),
        ("RuntimeError", NOT_LOGITS),
        ("IndexError", UNKNOWN_NODE),
    ],
    "sum_dims_refused": [
        ("IndexError", UNKNOWN_NODE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", DUPLICATE_AXIS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", DUPLICATE_AXIS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
    ],
    "sum_dims_negative_refused": [
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", DUPLICATE_AXIS),
        ("RuntimeError", DUPLICATE_AXIS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", DUPLICATE_AXIS),
    ],
    "slice_refused": [
        ("IndexError", UNKNOWN_NODE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("ValueError", STEP_NOT_POSITIVE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", OUT_OF_RANGE),
        ("IndexError", OUT_OF_RANGE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", OUT_OF_RANGE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
    ],
    "gather_refused": [
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", OUT_OF_RANGE),
        ("RuntimeError", VALUE_COUNT_MISMATCH),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("RuntimeError", INDEX_RANK_MISMATCH),
        ("RuntimeError", INDEX_SHAPE_MISMATCH),
        ("RuntimeError", OUT_OF_RANGE),
        ("IndexError", UNKNOWN_NODE),
        ("RuntimeError", CAPACITY),
        ("RuntimeError", RANK_EXCEEDED),
        ("RuntimeError", CAPACITY),
        ("IndexError", RANK_EXCEEDED),
    ],
    "slice_negative_refused": [
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", OUT_OF_RANGE),
        ("IndexError", OUT_OF_RANGE),
        ("RuntimeError", OUT_OF_RANGE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", OUT_OF_RANGE),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
        ("IndexError", AXIS_OUT_OF_BOUNDS),
    ],
}
FORGED = {"dangling_self", "dangling_newer", "dangling_missing"}
# NaN * NaN: the payload is not specified by the tensor layer.
NAN_PAYLOAD = {
    "specials",
    "relu_edges",
    "relu_nan_gradient",
    "mse_empty",
    "cross_entropy_infinities",
    "cross_entropy_nan",
    "cross_entropy_empty",
    "cross_entropy_empty_wide",
    "cross_entropy_one_class_scaled",
    "cross_entropy_two_class_scaled",
    "sgd_edges",
    "sgd_rate_nan",
    "batched_special_words",
}
# The forward value differs, not the backward: rounding_09 sums six words in
# ascending order where PyTorch's vectorized sum takes another order, and
# rounding_14 has a matmul with inner dimension 1 whose only product is -0.0:
# the ordered dot adds it to +0.0 and gives +0.0, PyTorch's gemm keeps -0.0.
# relu_13 has the same inner-dimension-1 matmul, whose products 0.5 * -1e-45
# and 0.5 * -0.0 are -0.0. sum_dims_rounding sums rows of 20 rounding words
# over axis 1, which PyTorch adds in a vectorized order: 4 of its row sums
# differ, its column sums agree. batched_rounding_01 sums the words of a
# torch.matmul in order, where PyTorch's sum takes another. Every gradient
# must agree word for word, and every value word that differs must lie
# within its rounding-error bound (AGENTS.md, numeric policy, level 2;
# tools/autograd_tensor_bound.py).
FORWARD_ORDER = {
    "rounding_09",
    "rounding_14",
    "relu_13",
    "sum_dims_rounding",
    "batched_rounding_01",
}
# torch.matmul words in the last bits, each pinned as (PyTorch's word, the
# reference's word) at its line, every other line exact, and each pinned
# word within its rounding-error bound. A product with one term (inner
# extent 1, as g @ right^T has for a 1-D right operand, or a [3, 4] @ [4, 1]
# matmul) whose only product is -0.0: the ordered dot adds it to +0.0 and
# gives +0.0, PyTorch keeps -0.0 (batched_rounding_00, batched_rounding_02).
# In batched_rounding_04 the right operand [5, 3] is broadcast over the
# batch [3, 1]: its gradient sums the batch's x^T @ g in batch order, where
# PyTorch folds the batch into one product (1 ulp), and the forward sum
# adds in order, where PyTorch's sum takes another (3 ulp).
PRODUCT_ORDER = {
    "batched_rounding_00": {
        44: (0x80000000, 0x00000000),
        47: (0x80000000, 0x00000000),
        59: (0x80000000, 0x00000000),
        62: (0x80000000, 0x00000000),
        64: (0x80000000, 0x00000000),
        66: (0x80000000, 0x00000000),
    },
    "batched_rounding_02": {
        26: (0x80000000, 0x00000000),
        28: (0x80000000, 0x00000000),
        30: (0x80000000, 0x00000000),
        34: (0x80000000, 0x00000000),
    },
    "batched_rounding_04": {
        133: (0x4C3EBC1D, 0x4C3EBC1E),
        322: (0xDA5529AF, 0xDA5529B2),
    },
}
# mse_loss words in the last bits: PyTorch sums the squares, and the
# broadcast target's gradient, in a cascade where the tape adds left to right.
# For each script, the line index of every word that differs and the pair
# (PyTorch's word, the reference's word); every other line must be exact, and
# each pinned word, PyTorch's and the reference's, must also lie within its
# rounding-error bound.
MSE_ROUNDING = {
    "loss_03": {
        143: (0xC23FCE94, 0xC23FCE93),
        145: (0xC2205F9E, 0xC2205F9F),
        147: (0xC1D6DFCF, 0xC1D6DFCE),
        151: (0x42E662FA, 0x42E662FB),
        163: (0x42E662FA, 0x42E662FB),
        167: (0x44A1786C, 0x44A1786D),
    },
    "loss_05": {
        37: (0x42F4DF00, 0x42F4DF01),
    },
    "loss_07": {
        71: (0x43103CF9, 0x43103CFB),
        83: (0x43103CF9, 0x43103CFB),
        87: (0x43FDE039, 0x43FDE03C),
    },
    "loss_09": {
        127: (0x429B5C42, 0x429B5C41),
    },
    "loss_11": {
        81: (0x42274C91, 0x42274C92),
    },
    "loss_15": {
        77: (0x40E3FDA0, 0x40E3FD9E),
        81: (0x42A58400, 0x42A58401),
        93: (0x42A58400, 0x42A58401),
        97: (0xC39B3B0F, 0xC39B3B10),
    },
}
# cross_entropy words in the last bits: PyTorch's log_softmax forward and
# backward use a vectorized exp whose error is its own, and its nll terms are
# summed in a cascade. These words are built on exp or log, which no bound
# covers. For each script, the line index of every word that differs and the
# pair (PyTorch's word, the reference's word); every other line must be exact,
# and the comparison refuses a pin on a word that has a bound.
LOSS_ROUNDING = {
    "cross_entropy": {
        21: (0x3D6CC5F4, 0x3D6CC5F5),
    },
    "cross_entropy_scaled": {
        28: (0x3BBC8FEE, 0x3BBC8FED),
    },
    "loss_02": {
        57: (0x2FBC0A32, 0x2FBC0A31),
        75: (0x3A70CB7C, 0x3A70CB7B),
        100: (0x40E2749D, 0x40E2749E),
    },
    "loss_06": {
        53: (0x3C52C0AC, 0x3C52C0AD),
        56: (0x362641F6, 0x362641F5),
        65: (0x35B633EC, 0x35B633ED),
        72: (0x38DFB880, 0x38DFB881),
    },
    "loss_08": {
        22: (0x3B4B88EE, 0x3B4B88ED),
    },
    "loss_10": {
        56: (0x3818EE64, 0x3818EE65),
        64: (0x2E5C1E56, 0x2E5C1E54),
        92: (0x32FD4F5E, 0x32FD4F5C),
    },
    "loss_12": {
        65: (0x395829BA, 0x395829B9),
        66: (0x34EC0E51, 0x34EC0E50),
    },
    "loss_14": {
        67: (0x2E12DDE3, 0x2E12DDE4),
        77: (0x3E2DA733, 0x3E2DA732),
    },
}

# PyTorch's CPU kernel for param.add_(grad, alpha=-lr) rounds p + (-lr) * g
# once (a fused multiply-add) on AVX2 and AVX512; the tape rounds the product,
# then the difference. These scripts match a reference that rounds once.
FUSED_UPDATE = {"sgd_two_layers", "sgd_rounding"}
# torch.optim.SGD accepts a non-leaf that retains its gradient, as every
# non-leaf of the replay does; the tape refuses it with NotLeaf.
NON_LEAF = {"sgd_not_leaf"}
# torch.matmul past the capacity of 2^20 elements, which PyTorch accepts: a
# forward output (the product is refused, CapacityExceeded, 2, and the next
# leaf takes its id), and an operand's gradient over the output's batch,
# where the forward output is empty (the backward is refused, and every slot
# read after it holds what it held, as PyTorch's do, since the backward adds
# only zeros). For each script, the line index of every line that differs
# and the pair (PyTorch's line, the reference's line).
# What PyTorch takes and the tape refuses, line by line: a gather index
# narrower than the input at another axis (IndexShapeMismatch 22), an
# index_select and a gather of a rank-0 input (AxisOutOfBounds 13), and two
# empty gather indexes, of another rank (IndexRankMismatch 21) and wider
# than the input at another axis (22), which PyTorch does not check.
INDEX_REFUSALS = {
    "gather_pytorch_accepts": {
        2: ("autograd:id:2", f"autograd:error:{INDEX_SHAPE_MISMATCH}"),
        3: ("autograd:id:3", f"autograd:error:{AXIS_OUT_OF_BOUNDS}"),
        4: ("autograd:id:4", f"autograd:error:{AXIS_OUT_OF_BOUNDS}"),
        5: ("autograd:id:5", f"autograd:error:{INDEX_RANK_MISMATCH}"),
        6: ("autograd:id:6", f"autograd:error:{INDEX_SHAPE_MISMATCH}"),
    },
}
CAPACITY_REFUSALS = {
    "batched_output_capacity": {
        2: ("autograd:id:2", f"autograd:error:{CAPACITY}"),
        3: ("autograd:id:3", "autograd:id:2"),
    },
    "batched_left_gradient_capacity": {
        2083: ("autograd:backward", f"autograd:error:{CAPACITY}"),
    },
    "batched_right_gradient_capacity": {
        2083: ("autograd:backward", f"autograd:error:{CAPACITY}"),
    },
}
# The families that record_rule does not judge; each keeps its own check.
OTHER_FAMILIES = (
    REJECTIONS
    | set(STEP_REJECTIONS)
    | FORGED
    | NAN_PAYLOAD
    | FUSED_UPDATE
    | NON_LEAF
    | set(CAPACITY_REFUSALS)
    | set(INDEX_REFUSALS)
)


def f32(number: int) -> float:
    return struct.unpack("<f", struct.pack("<I", number))[0]


def rounded(exact: Fraction) -> int:
    negative = exact < 0
    magnitude = -exact if negative else exact
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    unit = Fraction(2) ** (max(exponent, -126) - 23)
    steps = magnitude / unit
    whole = steps.numerator // steps.denominator
    rest = steps - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    if whole * unit >= Fraction(2) ** 128 - Fraction(2) ** 103:
        bits = 0x7F800000
    else:
        bits = struct.unpack("<I", struct.pack("<f", float(whole * unit)))[0]
    return bits | 0x80000000 if negative else bits


# p - lr * g rounded once, from the exact value, for finite words; a NaN or an
# infinity propagates as with two roundings.
def fused_word(lr: int, p: int, g: int) -> int:
    rate, held, gradient = f32(lr), f32(p), f32(g)
    if not all(math.isfinite(number) for number in (rate, held, gradient)):
        return descend_word(lr, p, g)
    product = Fraction(rate) * Fraction(gradient)
    exact = Fraction(held) - product
    if exact != 0:
        return rounded(exact)
    negative = math.copysign(1.0, held) < 0 and math.copysign(1.0, rate * gradient) > 0
    return NEGATIVE_ZERO if held == 0 and product == 0 and negative else 0


def case(steps: list[dict]) -> dict:
    return {"id": "case", "steps": steps}


def tensors(lines: list[str], prefix: str) -> list[list[int] | None]:
    found: list[list[int] | None] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == f"autograd:{prefix}:none":
            found.append(None)
        elif line == f"autograd:{prefix}":
            count = int(lines[index + 2].removeprefix("tensor:count:"))
            words = lines[index + 3 : index + 3 + count]
            found.append([int(row.removeprefix("tensor:word:")) for row in words])
            index += 2 + count
        index += 1
    return found


def grads(lines: list[str]) -> list[list[int] | None]:
    return tensors(lines, "grad")


def values(*numbers: float) -> list[int]:
    return [word(number) for number in numbers]


GRID = leaf([2, 3], values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
ROW = leaf([3], values(0.5, -1.0, 2.0))
UNUSED = [
    leaf([], values(3.0)),
    leaf([], values(2.0)),
    leaf([3], values(1.0, 2.0, 3.0)),
    op("mul", 0, 1),
    op("add", 2, 2),
]


class TensorBackwardReferenceTests(unittest.TestCase):
    def test_seed_is_the_selected_output_not_the_newest_node(self):
        found = grads(reference(case([*UNUSED, run(3), *reads(5)])))
        self.assertEqual(found, [values(2.0), values(3.0), None, values(1.0), None])

    def test_a_non_scalar_output_is_seeded_with_ones(self):
        found = grads(reference(case([GRID, ROW, op("mul", 0, 1), run(2), *reads(3)])))
        self.assertEqual(found[2], values(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(found[0], values(0.5, -1.0, 2.0, 0.5, -1.0, 2.0))
        self.assertEqual(found[1], values(5.0, 7.0, 9.0))

    def test_broadcast_contributions_are_reduced_to_the_operand_shape(self):
        column = leaf([2, 1], values(-2.0, 0.25))
        steps = [GRID, column, op("sub", 0, 1), op("sum", 2), run(3), *reads(4)]
        found = grads(reference(case(steps)))
        self.assertEqual(found[0], values(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(found[1], values(-3.0, -3.0))

    def test_fan_out_adds_in_arrival_order(self):
        selected = {row["id"]: row for row in cases()}
        # x's consumers x*a, x*b, x*c send a, b, c; the newest (x*c) arrives
        # first, so each position of x holds (c + b) + a: (-1e8 + 1e8) + 1 = 1 and
        # (1 + 1e8) - 1e8 = 0.
        found = grads(reference(selected["fan_out_arrival_order"]))
        self.assertEqual(found[0], values(1.0, 0.0))

    def test_sub_negates_before_it_reduces(self):
        steps = [
            leaf([2], [NEGATIVE_ZERO, 0]),
            op("sub", 0, 0),
            run(1),
            *reads(2),
        ]
        # g - g at one leaf: 1.0 then -1.0 arrive, and the total is +0.0.
        self.assertEqual(grads(reference(case(steps)))[0], [0, 0])

    def test_matmul_vjps(self):
        left = leaf([2, 3], values(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
        right = leaf([3, 2], values(2.0, 1.0, -1.0, 4.0, 0.5, -3.0))
        found = grads(
            reference(case([left, right, op("matmul", 0, 1), run(2), *reads(3)]))
        )
        # g @ right^T with g all ones: each row sums a row of right.
        self.assertEqual(found[0], values(3.0, 3.0, -2.5, 3.0, 3.0, -2.5))
        # left^T @ g: each row sums a column of left.
        self.assertEqual(found[1], values(4.0, 4.0, -1.75, -1.75, -0.5, -0.5))

    def test_sum_vjp_broadcasts_the_gradient(self):
        steps = [GRID, op("sum", 0), op("mul", 1, 1), run(2), *reads(3)]
        found = grads(reference(case(steps)))
        self.assertEqual(found[0], values(*[42.0] * 6))

    def test_repeated_backward_accumulates(self):
        found = grads(reference(case([*UNUSED, run(3), run(3), *reads(5)])))
        self.assertEqual(found[:2], [values(4.0), values(6.0)])

    def test_zero_grad_then_negative_zero_moves_in(self):
        steps = [
            leaf([2], values(4.0, -1.0)),
            leaf([2], [NEGATIVE_ZERO, 0]),
            op("mul", 0, 1),
            ZERO_GRAD,
            run(2),
            *reads(3),
        ]
        self.assertEqual(grads(reference(case(steps)))[0], [NEGATIVE_ZERO, 0])

    def test_unknown_ids_and_dangling_operands_are_typed(self):
        lines = reference(case([ROW, op("add", 0, 4), run(3), *reads(2)]))
        self.assertEqual(lines.count(f"autograd:error:{UNKNOWN_NODE}"), 2)
        self.assertIn(f"autograd:value:error:{UNKNOWN_NODE}", lines)
        forged = forge({"op": "sum", "a": 0}, [], values(1.0))
        lines = reference(case([forged, run(0)]))
        self.assertEqual(lines[-1], f"autograd:error:{DANGLING_OPERAND}")

    def test_linear_vjps_read_the_weight_in_place(self):
        x = leaf([2, 3], values(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
        weight = leaf([2, 3], values(2.0, 1.0, -1.0, 4.0, 0.5, -3.0))
        bias = leaf([2], values(0.5, -1.0))
        steps = [x, weight, bias, {"op": "linear", "a": 0, "b": 1, "bias": 2}]
        lines = reference(case([*steps, run(4), *reads(5)]))
        found = grads(lines)
        # g @ W with g all ones: each row of x's gradient sums W's rows.
        self.assertEqual(found[0], values(6.0, 1.5, -4.0, 6.0, 1.5, -4.0))
        # g^T @ x: each row of W's gradient sums x's rows.
        self.assertEqual(found[1], values(4.0, -1.75, -0.5, 4.0, -1.75, -0.5))
        # The bias gradient sums g over the batch, column by column.
        self.assertEqual(found[2], values(2.0, 2.0))
        # x @ W^T + b: row 0 is [2 - 2 - 0.5 + 0.5, 4 - 1 - 1.5 - 1], and
        # row 1 is [6 + 0.25 + 1 + 0.5, 12 + 0.125 + 3 - 1].
        self.assertEqual(tensors(lines, "value")[4], values(0.0, 0.5, 7.75, 14.125))

    def test_a_refused_linear_leaves_the_tape_as_it_was(self):
        x = leaf([1, 1], values(2.0))
        weight = leaf([1, 1], values(3.0))
        bias = {"op": "linear", "a": 0, "b": 1, "bias": 2}
        # bias 2 is the id the product would take: it is unknown.
        lines = reference(case([x, weight, bias, leaf([], values(5.0))]))
        self.assertEqual(
            lines,
            [
                "autograd:id:0",
                "autograd:id:1",
                f"autograd:error:{UNKNOWN_NODE}",
                "autograd:id:2",
            ],
        )
        # A bias that does not broadcast to the [2, 3] product drops it.
        column = leaf([2, 1], values(2.0, 4.0))
        rows = leaf([3, 1], values(3.0, 5.0, 7.0))
        short = leaf([2], values(1.0, 2.0))
        steps = [column, rows, short, bias, leaf([], values(5.0))]
        self.assertEqual(
            reference(case(steps))[3:], ["autograd:error:6", "autograd:id:3"]
        )

    def test_a_batched_gradient_past_the_capacity_refuses_the_backward(self):
        rows = {row["id"]: row for row in capacity_scripts()}
        for identifier in (
            "batched_left_gradient_capacity",
            "batched_right_gradient_capacity",
        ):
            with self.subTest(case=identifier):
                lines = reference(rows[identifier])
                first = lines.index("autograd:backward")
                refused = lines.index(f"autograd:error:{CAPACITY}")
                self.assertNotIn("autograd:backward", lines[first + 1 :])
                # The reads after the refusal repeat the first three reads
                # before it: the leaf's slot, the constant's None and the
                # sum's seed, unchanged.
                before = lines[first + 1 : refused]
                after = lines[refused + 1 :]
                self.assertEqual(after, before[: len(after)])
        lines = reference(rows["batched_output_capacity"])
        self.assertEqual(lines[2:4], [f"autograd:error:{CAPACITY}", "autograd:id:2"])

    def test_matmul_errors_are_typed(self):
        lines = reference(case([GRID, ROW, op("matmul", 0, 1), op("matmul", 0, 0)]))
        self.assertEqual(
            lines[2:],
            [f"autograd:error:{MATMUL_RANK}", f"autograd:error:{MATMUL_INNER}"],
        )


class TensorBackwardVerificationTests(unittest.TestCase):
    def frame(self, lines: list[str]) -> str:
        return "\n".join(["autograd:begin:case", *lines, "autograd:end"]) + "\n"

    def test_reference_output_is_accepted(self):
        row = case([*UNUSED, run(3), *reads(5)])
        verify([row], parse(self.frame(reference(row)), ["case"]), "native")

    def test_a_newest_node_seed_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        wrong = reference(case([*UNUSED, run(4), *reads(5)]))
        with self.assertRaises(GateError):
            verify([row], parse(self.frame(wrong), ["case"]), "native")

    def test_a_nan_only_needs_to_be_nan(self):
        row = case([leaf([1], [0x7FC01234]), op("sum", 0), run(1), *reads(2)])
        expected = reference(row)
        changed = [line.replace(str(0x7FC01234), str(0x7FC00000)) for line in expected]
        self.assertNotEqual(changed, expected)
        verify([row], parse(self.frame(changed), ["case"]), "native")

    def test_a_changed_word_or_missing_line_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        expected = reference(row)
        changed = [line.replace(str(word(3.0)), str(word(-3.0))) for line in expected]
        for wrong in (changed, expected[:-1]):
            with self.assertRaises(GateError):
                verify([row], parse(self.frame(wrong), ["case"]), "native")


def word_pairs(seen: list[str], expected: list[str]) -> dict[int, tuple[int, int]]:
    return {
        index: (
            int(a.removeprefix("tensor:word:")),
            int(b.removeprefix("tensor:word:")),
        )
        for index, (a, b) in enumerate(zip(seen, expected, strict=True))
        if a != b
    }


# Level 2 of the numeric policy for the scripts that the refusal, forged,
# NaN-payload, fused-update and non-leaf families do not cover. Each family
# keeps the rule it had before the rounding-error bound, and the bound is
# checked on top of it. `expected` is the reference's lines with their bounds,
# replayed from the script unless a test plants other ones. Returns the
# family; raises GateError on a mismatch.
def record_rule(row: dict, seen: list[str], expected: list[str] | None = None) -> str:
    identifier = row["id"]
    expected = bounded_reference(row) if expected is None else expected
    if len(seen) != len(expected):
        raise GateError(f"{identifier}: {len(seen)} lines, expected {len(expected)}")
    if identifier in FORWARD_ORDER:
        for a, b in zip(seen, expected, strict=True):
            if a != b and not (
                a.startswith("tensor:word:") and b.startswith("tensor:word:")
            ):
                raise GateError(f"{identifier}: {a} differs from {b}")
        if grads(seen) != grads(expected):
            raise GateError(f"{identifier}: a gradient differs from PyTorch's")
        bounded, _ = compare(identifier, seen, expected)
        if not bounded:
            raise GateError(f"{identifier}: no forward word differs")
        return "forward_order"
    for family, pinned in (
        ("mse_rounding", MSE_ROUNDING),
        ("product_order", PRODUCT_ORDER),
    ):
        if identifier in pinned:
            if word_pairs(seen, expected) != pinned[identifier]:
                raise GateError(f"{identifier}: not the pinned words")
            bounded, _ = compare(identifier, seen, expected)
            if bounded != len(pinned[identifier]):
                raise GateError(f"{identifier}: a pinned word is not bounded")
            return family
    if identifier in LOSS_ROUNDING:
        if word_pairs(seen, expected) != LOSS_ROUNDING[identifier]:
            raise GateError(f"{identifier}: not the pinned words")
        compare(identifier, seen, expected, LOSS_ROUNDING[identifier])
        return "loss_rounding"
    if seen != expected:
        raise GateError(f"{identifier}: a line differs from PyTorch's")
    return "exact"


# A sum that leaves out its last term, as a Bend fold that stops one cell
# early would; the reference follows it, so its bound is the bound of the
# wrong arithmetic.
def dropping_total(words: list[int]) -> Bounded:
    return bounded_total(words[:-1] if len(words) > 1 else words)


def dropping_reduce_to(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    fibers: list[list[int]] = [[] for _ in range(math.prod(target))]
    for flat in range(math.prod(shape)):
        fibers[source_index(target, shape, flat)].append(words[flat])
    return (target, [dropping_total(fiber) for fiber in fibers])


def dropping_sum_to_size(tensor: tuple, target: list[int]) -> tuple:
    if tensor[0] == target:
        return (target, list(tensor[1]))
    return dropping_reduce_to(tensor, target)


def dropping_matmul(left: tuple, right: tuple) -> tuple | int:
    result = bounded_matmul(left, right)
    (rows, inner), (_, columns) = left[0], right[0]
    if isinstance(result, int) or inner < 2:
        return result
    words = []
    for row in range(rows):
        for column in range(columns):
            pairs = [
                (left[1][row * inner + k], right[1][k * columns + column])
                for k in range(inner - 1)
            ]
            number = 0
            for a, b in pairs:
                number = add_word(number, mul_word(int(a), int(b)))
            words.append(bounded_dot(pairs, number))
    return (result[0], words)


class TensorBackwardTorchOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = FIXTURE.read_bytes()
        cls.digest = hashlib.sha256(data).hexdigest()
        cls.fixture = json.loads(data)

    def test_fixture_is_pinned_and_covers_current_requests(self):
        self.assertEqual(self.digest, FIXTURE_SHA256)
        self.assertEqual(self.fixture["oracle"]["torch_version"], "2.14.1+cpu")
        requests = json.dumps(cases(), sort_keys=True).encode()
        self.assertEqual(self.fixture["requests"], hashlib.sha256(requests).hexdigest())
        self.assertEqual(
            [row["id"] for row in self.fixture["backward"]],
            [row["id"] for row in cases()],
        )

    def test_reference_matches_pytorch_within_the_policy(self):
        exact = 0
        for row, observed in zip(cases(), self.fixture["backward"], strict=True):
            identifier = row["id"]
            with self.subTest(case=identifier):
                if identifier in FORGED:
                    self.assertEqual(observed["unrepresentable"], "forge")
                    continue
                expected = reference(row)
                seen = observed["lines"]
                self.assertEqual(len(seen), len(expected))
                differ = [(a, b) for a, b in zip(seen, expected, strict=True) if a != b]
                if identifier in STEP_REJECTIONS:
                    self.assertEqual(
                        [
                            (a.rsplit(":", 1)[1], int(b.rsplit(":", 1)[1]))
                            for a, b in differ
                        ],
                        STEP_REJECTIONS[identifier],
                    )
                    for a, b in differ:
                        self.assertEqual(
                            a.split(":")[1],
                            b.split(":")[1].replace("error", "rejected"),
                        )
                    continue
                if identifier in REJECTIONS:
                    for a, b in differ:
                        code = int(b.rsplit(":", 1)[1])
                        if a.endswith("IndexError"):
                            self.assertEqual(code, UNKNOWN_NODE)
                        elif a.endswith("ValueError"):
                            self.assertEqual(code, INVALID_LEARNING_RATE)
                        else:
                            self.assertEqual(a, "autograd:rejected:RuntimeError")
                            self.assertIn(code, RUNTIME_ERROR_CODES)
                        self.assertEqual(
                            a.split(":")[1],
                            b.split(":")[1].replace("error", "rejected"),
                        )
                    continue
                if identifier in NAN_PAYLOAD:
                    self.assertTrue(differ)
                    for a, b in differ:
                        self.assertTrue(is_nan(int(a.removeprefix("tensor:word:"))))
                        self.assertTrue(is_nan(int(b.removeprefix("tensor:word:"))))
                    continue
                if identifier in FUSED_UPDATE:
                    self.assertTrue(differ)
                    fused = reference(row, update=fused_word)
                    self.assertEqual(len(seen), len(fused))
                    for a, b in zip(seen, fused, strict=True):
                        self.assertTrue(same_row(a, b), (a, b))
                    continue
                if identifier in CAPACITY_REFUSALS | INDEX_REFUSALS:
                    self.assertEqual(
                        {
                            index: (a, b)
                            for index, (a, b) in enumerate(
                                zip(seen, expected, strict=True)
                            )
                            if a != b
                        },
                        (CAPACITY_REFUSALS | INDEX_REFUSALS)[identifier],
                    )
                    continue
                if identifier in NON_LEAF:
                    self.assertEqual(
                        differ[0], ("autograd:step", f"autograd:error:{NOT_LEAF}")
                    )
                    for a, b in differ[1:]:
                        self.assertTrue(a.startswith("tensor:word:"))
                        self.assertTrue(b.startswith("tensor:word:"))
                    continue
                if record_rule(row, seen) == "exact":
                    exact += 1
        excluded = (
            REJECTIONS
            | set(STEP_REJECTIONS)
            | FORGED
            | NAN_PAYLOAD
            | FORWARD_ORDER
            | set(MSE_ROUNDING)
            | set(PRODUCT_ORDER)
            | set(LOSS_ROUNDING)
            | FUSED_UPDATE
            | NON_LEAF
            | set(CAPACITY_REFUSALS)
            | set(INDEX_REFUSALS)
        )
        self.assertEqual(exact, len(cases()) - len(excluded))
        self.assertEqual(exact, 282)

    # Planted wrong results on the reference's side, as a wrong Bend result
    # would be: a sum, a broadcast reduction, a sum over axes (reduce_to) and
    # a matmul that leave out their last term. The clean lines of every
    # script pass first. The bound comparison
    # rejects each changed script but loss_07, whose omitted square adds about
    # 9e-5 to a loss of 144 (6 ulp), inside that word's 15-ulp bound; its old
    # pins reject it under the whole rule. slice_parallel's omitted term is one
    # of the 59700 terms of its sum, inside that sum's bound, and
    # index_select_parallel's one of its 45000; their exact records reject
    # them under the whole rule. A word with no bound (the
    # cross_entropy words, built on exp and log) is rejected by its pin, or
    # because it differs with no pin, never by a bound.
    def test_a_dropped_term_is_rejected(self):
        rows = {row["id"]: row for row in cases()}
        seen = {row["id"]: row.get("lines") for row in self.fixture["backward"]}
        compared = [
            identifier for identifier in rows if identifier not in OTHER_FAMILIES
        ]
        for identifier in compared:
            record_rule(rows[identifier], seen[identifier])
            pins = LOSS_ROUNDING.get(identifier)
            compare(
                identifier, seen[identifier], bounded_reference(rows[identifier]), pins
            )
        planted = {
            "total": (
                dropping_total,
                152,
                {"loss_07", "slice_parallel", "index_select_parallel"},
                (134, 13, 2),
            ),
            "sum_to_size": (dropping_sum_to_size, 88, set(), (86, 2, 0)),
            "matmul": (dropping_matmul, 58, set(), (56, 2, 0)),
            "reduce_to": (dropping_reduce_to, 21, set(), (21, 0, 0)),
        }
        for name, (function, changed, missed, reasons) in planted.items():
            with self.subTest(planted=name):
                counts = {"outside the bound": 0, "no bound": 0, "pinned pair": 0}
                accepted = set()
                touched = 0
                for identifier in compared:
                    row = rows[identifier]
                    wrong = bounded_lines(row, REPLACED | {name: function})
                    if wrong == reference(row):
                        continue
                    touched += 1
                    pins = LOSS_ROUNDING.get(identifier)
                    try:
                        compare(identifier, seen[identifier], wrong, pins)
                    except GateError as error:
                        reason = next(key for key in counts if key in str(error))
                        counts[reason] += 1
                    else:
                        accepted.add(identifier)
                    with self.assertRaises(GateError):
                        record_rule(row, seen[identifier], wrong)
                self.assertEqual(touched, changed)
                self.assertEqual(accepted, missed)
                self.assertEqual(tuple(counts.values()), reasons)

    # Words the rule before the rounding-error bound rejected, and that are
    # within the bound: the bound comparison alone accepts each one, and the
    # whole rule, which checks the bound on top of the old rule, rejects it.
    # The first is a pinned mse_loss word moved 2 ulp, inside its 57-ulp
    # bound; the others are gradient words moved 1 ulp in the scripts whose
    # forward values differ.
    def test_a_word_the_old_rule_rejected_stays_rejected(self):
        rows = {row["id"]: row for row in cases()}
        seen = {row["id"]: row.get("lines") for row in self.fixture["backward"]}
        planted = [
            ("loss_03", 151, 0x42E662FA, 0x42E662FC),
            ("rounding_09", 36, 0x4000418A, 0x4000418B),
            ("rounding_14", 23, 0x40800000, 0x40800001),
            ("relu_13", 18, 0xBFC00000, 0xBFC00001),
        ]
        for identifier, index, old, new in planted:
            with self.subTest(case=identifier):
                lines = list(seen[identifier])
                self.assertEqual(lines[index], f"tensor:word:{old}")
                record_rule(rows[identifier], lines)
                lines[index] = f"tensor:word:{new}"
                expected = bounded_reference(rows[identifier])
                self.assertTrue(within(new, bounded_word(expected[index])))
                compare(identifier, lines, expected)
                with self.assertRaises(GateError):
                    record_rule(rows[identifier], lines)

    def test_arrival_order_and_broadcast_folds_match_pytorch(self):
        rows = {row["id"]: row for row in self.fixture["backward"]}
        selected = {row["id"]: row for row in cases()}
        for identifier in ("fan_out_arrival_order", "broadcast_fold_order", "linear"):
            self.assertEqual(rows[identifier]["lines"], reference(selected[identifier]))
        self.assertEqual(
            grads(rows["fan_out_arrival_order"]["lines"])[0], values(1.0, 0.0)
        )


def canonicalized(lines: list[str], index: int) -> list[str]:
    changed = list(lines)
    changed[index] = f"tensor:word:{CANONICAL}"
    return changed


CANONICAL = 0x7FC00000


class ReluExactWordTests(unittest.TestCase):
    """The native lane holds relu's words exact, NaNs included; any NaN for any
    NaN is still accepted elsewhere and on the JavaScript lane."""

    def selected(self, identifier: str) -> dict:
        return next(row for row in cases() if row["id"] == identifier)

    def word_index(self, lines: list[str], header: str, nth: int, word: int) -> int:
        start = [i for i, line in enumerate(lines) if line == header][nth] + 3
        return lines.index(f"tensor:word:{word}", start)

    def test_reference_lines_pass_on_both_lanes(self):
        for identifier in ("relu_edges", "relu_nan_gradient", "relu_two_layers"):
            row = self.selected(identifier)
            observed = [{"id": identifier, "lines": reference(row)}]
            verify([row], observed, "native", native=True)
            verify([row], observed, "javascript")

    def test_a_canonical_nan_from_relu_fails_only_the_native_check(self):
        row = self.selected("relu_edges")
        lines = reference(row)
        # Node 1 is relu(x): its word for the negative signaling NaN.
        index = self.word_index(lines, "autograd:value", 1, 0xFF800001)
        observed = [{"id": row["id"], "lines": canonicalized(lines, index)}]
        verify([row], observed, "javascript")
        with self.assertRaisesRegex(GateError, "relu of node 0 is not exact"):
            verify([row], observed, "native", native=True)

    def test_a_canonical_nan_in_the_passed_gradient_fails_the_native_check(self):
        row = self.selected("relu_nan_gradient")
        lines = reference(row)
        # Node 0's gradient where x = 1.0: g's own word, a negative NaN.
        index = self.word_index(lines, "autograd:grad", 0, 0xFFC00001)
        observed = [{"id": row["id"], "lines": canonicalized(lines, index)}]
        verify([row], observed, "javascript")
        with self.assertRaisesRegex(GateError, "relu gradient of node 0 is not exact"):
            verify([row], observed, "native", native=True)


class GateRegistrationTests(unittest.TestCase):
    def test_sgd_controls_apply_once_and_name_a_required_law(self):
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(len(SGD_CONTROLS), 9)
        for name, (path, original, mutated, law) in SGD_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual((ROOT / path).read_text().count(original), 1)
                self.assertNotEqual(original, mutated)
                self.assertIn(law.replace("Laws.", "sgd_LAWS."), required)
                self.assertEqual(RUNTIME_CONTROLS[name], (path, original, mutated))
        self.assertIn("zero-grad-keeps-slot", RUNTIME_CONTROLS)

    def test_autograd_gate_requires_the_chain_rule_laws_and_their_controls(self):
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(
            [name for name in required if name.startswith("tensor_chain_LAWS.")],
            [
                "tensor_chain_LAWS.tensor_sweep_chain_rule",
                "tensor_chain_LAWS.tensor_backward_chain_rule",
            ],
        )
        proof = (ROOT / "src/autograd/tensor_chain_PROOF.bend").read_text()
        self.assertEqual(len(CHAIN_CONTROLS), 4)
        for name, (path, original, mutated, location) in CHAIN_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual((ROOT / path).read_text().count(original), 1)
                self.assertNotEqual(original, mutated)
                self.assertIn(f"def {location.removeprefix('Location: ')}(", proof)
                self.assertEqual(RUNTIME_CONTROLS[name], (path, original, mutated))

    def test_autograd_gate_registers_tensor_backward_and_its_laws(self):
        self.assertIn("tensor_backward", SUBGATES)
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(
            sum(name.startswith("tensor_backward_LAWS.") for name in required),
            21,
        )
        self.assertEqual(
            [name for name in required if name.startswith("tensor_vjp_LAWS.")],
            [
                "tensor_vjp_LAWS.tensor_product_replayed",
                "tensor_vjp_LAWS.tensor_add_vjp_values",
                "tensor_vjp_LAWS.tensor_sub_vjp_values",
                "tensor_vjp_LAWS.tensor_mul_vjp_values",
                "tensor_vjp_LAWS.tensor_matmul_vjp_values",
                "tensor_vjp_LAWS.tensor_matmul_nt_vjp_values",
                "tensor_vjp_LAWS.tensor_matmul_nt_frozen_vjp_values",
                "tensor_vjp_LAWS.nt_reads_transpose",
                "tensor_vjp_LAWS.tn_reads_transpose",
                "tensor_vjp_LAWS.tensor_relu_vjp_values",
                "tensor_vjp_LAWS.tensor_mse_vjp_values",
                "tensor_vjp_LAWS.tensor_cross_entropy_vjp_values",
            ],
        )
        controls = [
            name for name in PROOF_CONTROLS if name.startswith("tensor-backward-")
        ]
        self.assertEqual(len(controls), 11)
        self.assertIn("negate-left-keeps-sign", PROOF_CONTROLS)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_native_controls_apply_once(self):
        self.assertEqual(
            set(NATIVE_CONTROLS),
            {"relu-canonicalizes-nan", "relu-backward-canonicalizes-nan"},
        )
        for path, original, mutated in NATIVE_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)
            self.assertIn("quiet(2143289344)", mutated)

    def test_runtime_controls_apply_once(self):
        self.assertEqual(len(RUNTIME_CONTROLS), 56)
        for control in RUNTIME_CONTROLS.values():
            path, original, mutated = (
                control
                if len(control) == 3
                else ("src/autograd/tensor_backward.bend", *control)
            )
            self.assertEqual((ROOT / path).read_text().count(original), 1)
            self.assertNotEqual(original, mutated)


class TensorBackwardRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(
            prefix="autograd-tensor-backward-"
        ) as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))
        self.assertEqual(
            summary["runtime_controls"], [*RUNTIME_CONTROLS, *NATIVE_CONTROLS]
        )


if __name__ == "__main__":
    unittest.main()
