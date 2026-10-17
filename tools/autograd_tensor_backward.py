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
    QUIET,
    QUIET_NAN,
    UNKNOWN_NODE,
    add_word,
    mul_word,
    nan_operand,
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
from tools.autograd_reduce import bits, f32, value
from tools.common import ROOT, Evidence, GateError, bend, native
from tools.tensor_signed import position_source, positions_source, wrap

VALUE_COUNT_MISMATCH = 10
MATMUL_RANK = 105
MATMUL_INNER = 106
NOT_LEAF = 109
INVALID_LEARNING_RATE = 110
INFINITY = 0x7F800000
BINARY = ("add", "sub", "mul", "matmul", "matmul_nt", "mse_loss")
UNARY = ("sum", "sum_dims", "relu", "cross_entropy")
# torch.matmul with batching and 1-D promotion. It is not in BINARY, which
# the generated scripts draw from, so their scripts stay as they were.
PRODUCTS = ("batched_matmul",)
# Basic slicing, each recorded as one Slice node. Not in UNARY either, for the
# same reason.
SLICES = ("slice", "narrow", "select")
# index_select and gather, each recorded as one Gathered node, with an index
# of U32 entries. Not in UNARY either.
GATHERS = ("index_select", "gather")
NOT_LOGITS = 107
TARGET_OUT_OF_RANGE = 108
NEGATIVE_INFINITY = 0xFF800000
# The codes permute gives a bad axis, which sum over axes reuses.
AXIS_OUT_OF_BOUNDS = 13
DUPLICATE_AXIS = 14
# A slice's refusals: a step of 0, and a narrow or select past the extent.
STEP_NOT_POSITIVE = 19
OUT_OF_RANGE = 20
# A gather index of another rank than the input, or with another extent at
# another axis.
INDEX_RANK_MISMATCH = 21
INDEX_SHAPE_MISMATCH = 22

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
        "  Matrix.matmul_tn(copy, flipped)",
        "  Matrix.matmul_tn(flipped, copy)",
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
    # The bias gradient's column sum runs the strided loop of
    # `Reduce.fiber_fast`; stepping it by the target's stride sums each row's
    # first word instead.
    "linear-bias-gradient-sums-rows": (
        "src/autograd/reduce.bend",
        "      strided(fuel, right, base, digit, single_offset(",
        "      strided(fuel, left, base, digit, single_offset(",
    ),
    # torch.matmul: a stretched batch axis read at the cell's own index
    # instead of 0, a gradient delivered over the output's batch without
    # summing it to its operand's shape, and left^T @ g reading g by rows.
    "batched-stretched-axis-reads-own-index": (
        "src/tensor/batched-matmul.bend",
        "    case True{}:\n      0n\n    case False{}:\n      index\n",
        "    case True{}:\n      index\n    case False{}:\n      index\n",
    ),
    "batched-vjp-skips-batch-sum": (
        "      (first, second, product) = triple\n"
        "      contribution_of(reduced(target, shape, product))\n",
        "      (first, second, product) = triple\n"
        "      Done{Contribution{target, shape, product}}\n",
    ),
    "batched-right-gradient-reads-g-by-rows": (
        "src/tensor/batched-matmul.bend",
        "  Layout{Matrix.Swapped{}, Matrix.Swapped{}, rows, inner, columns, "
        "l1, l2, o1, o2, o2}",
        "  Layout{Matrix.Swapped{}, Matrix.Kept{}, rows, inner, columns, "
        "l1, l2, o1, o2, o2}",
    ),
    # A sum over axes: g of the dropped shape broadcast without the reduced
    # axes put back, the axis after each listed one summed, and keepdim
    # ignored.
    "sum-dims-vjp-broadcasts-dropped-shape": (
        "restored(shape, vjp_sum(a, input, "
        "Transforms.reshape_equal(shape, kept, tensor, "
        "Transforms.elements_equal(shape, kept, equal))))",
        "restored(shape, vjp_sum(a, input, Core.Packed{shape, tensor}))",
    ),
    "sum-dims-sums-next-axis": (
        "src/autograd/sum-dims.bend",
        "Bool.pick(U32, reduces(axes, axis), 1, extent)",
        "Bool.pick(U32, reduces(axes, U32.add(axis, 1)), 1, extent)",
    ),
    "sum-dims-ignores-keepdim": (
        "src/autograd/sum-dims.bend",
        "    case True{}:\n      Done{output}",
        "    case True{}:\n"
        "      transformed(Transforms.reshape_dynamic(output, dropped))",
    ),
    # Slicing: a narrow whose tape plan takes the length for the stop, a
    # select whose tape plan scatters into the next index, a scatter that
    # fills the cells no output cell came from with 1.0 instead of +0.0, and
    # a plan that wraps a start before -extent among the extent, as an index,
    # instead of bounding it at 0, so the plan and the value disagree.
    "narrow-records-the-length-as-stop": (
        "      Slice.plan(shape, axis, first, U32.add(first, length), 1)",
        "      Slice.plan(shape, axis, first, length, 1)",
    ),
    "select-records-the-next-index": (
        "      Slice.plan(shape, axis, first, U32.add(first, 1), 1)",
        "      Slice.plan(shape, axis, U32.add(first, 1), U32.add(first, 2), 1)",
    ),
    # index_select and gather: a backward that overwrites a cell an entry
    # listed twice adds to, and a tape that records no entries.
    "gather-scatter-overwrites": (
        "src/tensor/gather.bend",
        "  (F32.add(acc, x), (g, index))",
        "  (x, (g, index))",
    ),
    "gather-records-no-entries": (
        "    case IndexSelected{dim, entries}:\n      entries",
        "    case IndexSelected{dim, entries}:\n      Nil{}",
    ),
    "slice-plan-wraps-the-start": (
        "      Slice.plan(shape, axis, Signed.bound(slice_extent(shape, axis), start),",
        "      Slice.plan(shape, axis, Signed.wrap(slice_extent(shape, axis), start),",
    ),
    "slice-scatter-fills-ones": (
        "src/tensor/slice.bend",
        "    case None{}:\n      (0.0, state)\n    case Some{+index}:\n      fetched",
        "    case None{}:\n      (1.0, state)\n    case Some{+index}:\n      fetched",
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
    "mean-divides-by-one": (
        "src/autograd/loss.bend",
        "  total_over(Shape.logical_numel(Shape.dimensions(shape)), "
        "Reduction.sum_dynamic(input))",
        "  total_over(1n, Reduction.sum_dynamic(input))",
    ),
    "mse-scale-inverted": (
        "src/autograd/loss.bend",
        "  F32.div(2.0, count_f32(Shape.logical_numel(Shape.dimensions(shape))))",
        "  F32.div(count_f32(Shape.logical_numel(Shape.dimensions(shape))), 2.0)",
    ),
    "cross-entropy-skips-max-shift": (
        "src/autograd/loss.bend",
        "ce_shifted(rows, onehot, target, "
        "Elementwise.binary(Elementwise.Sub{}, logits, maxes),",
        "ce_shifted(rows, onehot, target, "
        "Elementwise.binary(Elementwise.KeepLeft{}, logits, maxes),",
    ),
    "one-hot-shifted-column": (
        "src/autograd/loss.bend",
        "classes(columns, 0n)",
        "classes(columns, 1n)",
    ),
    "cross-entropy-mean-divides-by-one": (
        "src/autograd/loss.bend",
        "  recorded(lp, onehot, total_over(rows, result))",
        "  recorded(lp, onehot, total_over(1n, result))",
    ),
    "one-hot-gradient-zero": (
        "src/tensor/elementwise.bend",
        "      zero_if(Bool.not(is_one(right)), target_gradient(left, count))",
        "      zero_if(True{}, target_gradient(left, count))",
    ),
    "target-gradient-skips-count": (
        "src/tensor/elementwise.bend",
        "  F32.neg(F32.div(g, count))",
        "  F32.neg(g)",
    ),
    "softmax-gradient-swaps-operands": (
        "entropy_second(a, input, product_reduced(a, input, first), onehot, "
        "Elementwise.binary(Elementwise.SoftmaxGrad{count, single}, g, lp))",
        "entropy_second(a, input, product_reduced(a, input, first), onehot, "
        "Elementwise.binary(Elementwise.SoftmaxGrad{count, single}, lp, g))",
    ),
    "softmax-gradient-ignores-one-class": (
        "entropy_first(a, input, Loss.count_f32(Loss.rows(input)), "
        "Loss.one_class(input), lp,",
        "entropy_first(a, input, Loss.count_f32(Loss.rows(input)), False{}, lp,",
    ),
    "reduction-folds-equal-shapes": (
        "delivered_as(target, shape, SumToSize.sum_to_size(g, shape))",
        "delivered_as(target, shape, Reduce.reduce_to(g, shape))",
    ),
    "constant-records-leaf": (
        "pushed(Constant{}, value, Unsaved{}, counted(entries))",
        "pushed(Leaf{}, value, Unsaved{}, counted(entries))",
    ),
    "constant-result-records-frozen": (
        "    case True{}:\n      Constant{}\n",
        "    case True{}:\n      freeze(ids, nodes, count, node)\n",
    ),
    "frozen-keeps-contribution": (
        "kept_one(target, shape, gradient, Nat.is_eq(id, target), "
        "kept_others(id, rest))",
        "kept_one(target, shape, gradient, False{}, kept_others(id, rest))",
    ),
    "frozen-skip-sends-input-gradient": (
        "      vjp_matmul_nt_right(b, right, saved, g)\n",
        "      vjp_matmul_nt(a, b, left, right, saved, g)\n",
    ),
}

# SGD mutants: the tensor backward lanes reject each one, and the autograd
# gate's check of src/autograd/PROOF.bend rejects each at the law named last.
DESCEND_HEAD = (
    "def descend(+lr: F32, entry: Backward.Entry) -> "
    "Backward.Entry & Result<&2, &2, Error, Unit>:\n"
    "  Backward.Entry{node, +shape, value, saved, slot} = entry\n"
    "  match slot:\n"
    "    case Gradient.Unset{}:\n"
    "      (Backward.Entry{node, shape, value, saved, Gradient.Unset{}}, Done{Unit{}})"
)
# Reads an empty slot as zeros, as Gradient.gradient does, and updates with them.
ZEROED = (
    "def zeroed(+lr: F32, node: Backward.Node, +shape: Shape.Shape, "
    "value: Core.Dynamic, saved: Backward.Saved,\n"
    "  zeros: Result<&2, &1, Core.Error, Core.Dynamic>) -> "
    "Backward.Entry & Result<&2, &2, Error, Unit>:\n"
    "  match zeros:\n"
    "    case Fail{error}:\n"
    "      (Backward.Entry{node, shape, value, saved, Gradient.Unset{}}, "
    "Done{Unit{}})\n"
    "    case Done{g}:\n"
    "      descended(node, shape, saved, "
    "Elementwise.binary(Elementwise.Descend{lr}, value, g))\n\n"
)
SGD_CONTROLS = {
    "sgd-step-adds": (
        "src/tensor/elementwise.bend",
        "      F32.sub(left, F32.mul(lr, right))",
        "      F32.add(left, F32.mul(lr, right))",
        "Laws.sgd_descend_scalar",
    ),
    "sgd-step-doubles-lr": (
        "src/autograd/sgd.bend",
        "Elementwise.binary(Elementwise.Descend{lr}, value, g)",
        "Elementwise.binary(Elementwise.Descend{F32.add(lr, lr)}, value, g)",
        "Laws.sgd_step_values",
    ),
    "sgd-step-skips-later-params": (
        "src/autograd/sgd.bend",
        "continued(entries => descend_all(lr, count, rest, entries), ",
        "continued(entries => (entries, Done{Unit{}}), ",
        "Laws.sgd_params_in_order",
    ),
    "sgd-step-updates-empty-slot": (
        "src/autograd/sgd.bend",
        DESCEND_HEAD,
        ZEROED
        + DESCEND_HEAD.replace(
            "      (Backward.Entry{node, shape, value, saved, Gradient.Unset{}}, "
            "Done{Unit{}})",
            "      zeroed(lr, node, shape, value, saved, "
            "Gradient.gradient(shape, Gradient.Unset{}))",
        ),
        "Laws.sgd_empty_slot_kept",
    ),
    "sgd-update-reaches-neighbor": (
        "src/autograd/sgd.bend",
        "kept(entry, descend_at(lr, rest, below, id, Backward.at_head(below, id)))",
        "kept(entry, descend_at(lr, rest, below, id, Backward.at_head(below, 1n+id)))",
        "Laws.sgd_update_reaches_target",
    ),
    "sgd-step-updates-before-checking": (
        "src/autograd/sgd.bend",
        "checked(lr, entries, count, params, refusal(nodes, count, params))",
        "checked(lr, entries, count, params, None{})",
        "Laws.sgd_step_refused",
    ),
    "sgd-step-accepts-non-leaf": (
        "src/autograd/sgd.bend",
        "    case Backward.Mul{a, b, left, right}:\n      Some{NotLeaf{}}",
        "    case Backward.Mul{a, b, left, right}:\n      None{}",
        "Laws.sgd_only_leaves_accepted",
    ),
    "sgd-step-skips-rate-check": (
        "src/autograd/sgd.bend",
        "rated(lr, entries, count, params, rate_refusal(lr))",
        "rated(lr, entries, count, params, None{})",
        "Laws.sgd_step_rate_refused",
    ),
    "sgd-rate-refuses-zero": (
        "src/autograd/sgd.bend",
        "  refused_if(Elementwise.below_zero(lr))",
        "  refused_if(Elementwise.at_most_zero(lr))",
        "Laws.sgd_negative_rate_refused",
    ),
}
RUNTIME_CONTROLS |= {
    name: (path, original, mutated)
    for name, (path, original, mutated, _) in SGD_CONTROLS.items()
}
# Sweeps that break the chain rule: a dropped contribution, one added twice,
# a VJP's contributions delivered in reverse operand order, and a contribution
# sent to the next slot. The proof of tensor-chain-PROOF.bend, checked on its
# own, fails at the named def for each, and the JavaScript lane rejects each.
BUMP_HIT = (
    "# Accumulates a contribution into the slot of node target, in a newest-first\n"
    "# list of count slots; same says whether the head is that node. A target the\n"
    "# list does not hold changes nothing.\n"
    "def bump(slots: List<&1, Gradient.Slot>, +count: Nat, +target: Nat, "
    "+shape: Shape.Shape, gradient: Core.Dynamic, same: Bool) ->\n"
    "  Result<&2, &1, Error, List<&1, Gradient.Slot>>:\n"
    "  match slots count same:\n"
    "    case Nil{} count same:\n"
    "      Done{Nil{}}\n"
    "    case Con{slot, rest} 0n same:\n"
    "      Done{slot <> rest}\n"
    "    case Con{slot, rest} 1n+below True{}:\n"
    "      placed(rest, Gradient.accumulate(shape, slot, gradient))"
)
PLACED_TWICE = """def again(+shape: Shape.Shape, rest: List<&1, Gradient.Slot>,
  outcome: Gradient.Outcome, extra: Core.Dynamic) ->
  Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  match outcome:
    case Gradient.Accumulated{slot}:
      placed(rest, Gradient.accumulate(shape, slot, extra))
    case Gradient.Refused{slot, contribution, error}:
      Fail{GradientError{error}}

def placed_twice(rest: List<&1, Gradient.Slot>, +shape: Shape.Shape,
  slot: Gradient.Slot, pair: Core.Dynamic & Core.Dynamic) ->
  Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  (one, two) = pair
  again(shape, rest, Gradient.accumulate(shape, slot, one), two)

"""
DELIVERED = (
    "      deliver(rest, count, bump(slots, count, target, shape, gradient, "
    "at_head(count, target)))"
)
CHAIN_CONTROLS = {
    "sweep-drops-contribution": (
        "src/autograd/tensor-backward.bend",
        DELIVERED,
        "      deliver(rest, count, Done{slots})",
        "Location: relay",
    ),
    "sweep-adds-contribution-twice": (
        "src/autograd/tensor-backward.bend",
        BUMP_HIT,
        PLACED_TWICE
        + BUMP_HIT.replace(
            "      placed(rest, Gradient.accumulate(shape, slot, gradient))",
            "      placed_twice(rest, shape, slot, Core.clone_dynamic(gradient))",
        ),
        "Location: hit_bumped",
    ),
    "sweep-swaps-delivery-order": (
        "src/autograd/tensor-backward.bend",
        "      deliver(items, count, Done{slots})",
        "      deliver(List.reverse(&1, Contribution, items), count, Done{slots})",
        "Location: fired_stop",
    ),
    "sweep-sends-to-wrong-slot": (
        "src/autograd/tensor-backward.bend",
        DELIVERED,
        "      deliver(rest, count, bump(slots, count, 1n+target, shape, gradient, "
        "at_head(count, 1n+target)))",
        "Location: relay",
    ),
}

RUNTIME_CONTROLS |= {
    name: (path, original, mutated)
    for name, (path, original, mutated, _) in CHAIN_CONTROLS.items()
}
RUNTIME_CONTROLS["zero-grad-keeps-slot"] = (
    "Entry{node, shape, value, saved, Gradient.reset(slot)} <> cleared(rest)",
    "Entry{node, shape, value, saved, slot} <> cleared(rest)",
)

# Controls that only the native lane can reject: the JavaScript lane prints
# every NaN as 0x7FC00000, so a relu that replaces a NaN word with that
# canonical NaN looks right there. Each replaces x's or g's NaN word, sign,
# payload and signaling bit included, with 0x7FC00000.
CANONICAL_NAN = """def quiet(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def canonical_if(nan: Bool, value: F32) -> F32:
  match nan:
    case True{}:
      quiet(2143289344)
    case False{}:
      value

# 0x7FC00000 when the word is a NaN, else the word.
def canonical(+value: F32) -> F32:
  canonical_if(U32.is_lt(2139095040, U32.and(bits(value), 2147483647)), value)

"""
NATIVE_CONTROLS = {
    "relu-canonicalizes-nan": (
        "src/tensor/elementwise.bend",
        "def relu(+x: F32) -> F32:\n  zero_if(below_zero(x), x)",
        CANONICAL_NAN
        + "def relu(+x: F32) -> F32:\n  zero_if(below_zero(x), canonical(x))",
    ),
    "relu-backward-canonicalizes-nan": (
        "src/tensor/elementwise.bend",
        "def relu_backward(+x: F32, g: F32) -> F32:\n  zero_if(at_most_zero(x), g)",
        CANONICAL_NAN + "def relu_backward(+x: F32, g: F32) -> F32:\n"
        "  zero_if(at_most_zero(x), canonical(g))",
    ),
}


def f(*numbers: float | int) -> list[int]:
    return [number if isinstance(number, int) else word(number) for number in numbers]


def leaf(shape: list[int], words: list[int]) -> dict:
    return {"op": "leaf", "shape": shape, "words": words}


# A leaf that takes no gradient, PyTorch's requires_grad=False.
def constant(shape: list[int], words: list[int]) -> dict:
    return {"op": "constant", "shape": shape, "words": words}


def op(name: str, a: int, b: int | None = None) -> dict:
    return {"op": name, "a": a} if b is None else {"op": name, "a": a, "b": b}


# PyTorch's a.sum(dim=axes, keepdim=keepdim).
def dims(a: int, axes: list[int], keepdim: bool = False) -> dict:
    return {"op": "sum_dims", "a": a, "dims": axes, "keepdim": keepdim}


def entropy(a: int, targets: list[int]) -> dict:
    return {"op": "cross_entropy", "a": a, "targets": targets}


def sliced(a: int, dim: int, start: int, stop: int, step: int = 1) -> dict:
    return {
        "op": "slice",
        "a": a,
        "dim": dim,
        "start": start,
        "stop": stop,
        "step": step,
    }


def narrowed(a: int, dim: int, start: int, length: int) -> dict:
    return {"op": "narrow", "a": a, "dim": dim, "start": start, "length": length}


def selected(a: int, dim: int, index: int) -> dict:
    return {"op": "select", "a": a, "dim": dim, "index": index}


def index_selected(a: int, dim: int, entries: list[int]) -> dict:
    return {"op": "index_select", "a": a, "dim": dim, "index": entries}


def gathered(a: int, dim: int, shape: list[int], entries: list[int]) -> dict:
    return {"op": "gather", "a": a, "dim": dim, "index_shape": shape, "index": entries}


def run(output: int) -> dict:
    return {"op": "backward", "output": output}


def reads(count: int) -> list[dict]:
    return [{"op": "read", "id": index} for index in range(count)]


def forge(node: dict, shape: list[int], words: list[int]) -> dict:
    return {"op": "forge", "node": node, "shape": shape, "words": words}


ZERO_GRAD = {"op": "zero_grad"}


def linear(x: int, w: int, bias: int) -> dict:
    return {"op": "linear", "a": x, "b": w, "bias": bias}


def sgd(lr: int, params: list[int]) -> dict:
    return {"op": "step", "lr": lr, "params": params}


def program_case(identifier: str, steps: list[dict], output: int) -> dict:
    count = sum(
        step["op"]
        in ("leaf", "constant", *UNARY, *BINARY, *PRODUCTS, *SLICES, *GATHERS)
        for step in steps
    )
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
        # NaN gradients of either sign, with payloads and a signaling NaN,
        # reach relu through x * w: where x is above zero or NaN, x's
        # gradient is g's own word.
        program_case(
            "relu_nan_gradient",
            [
                leaf([7], f(1.0, 2.0, 0x7FC00000, 0xFFC00001, 3.0, -1.0, 0.5)),
                op("relu", 0),
                leaf(
                    [7],
                    [
                        0xFFC00001,
                        QUIET_NAN,
                        0xFF800001,
                        0x7FA00001,
                        NEGATIVE_ZERO,
                        0xFFC00001,
                        0xFFFFFFFF,
                    ],
                ),
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
        *sgd_scripts(left, layer, layer_bias),
        *constant_scripts(left, layer, layer_bias),
    ]


# Constants, leaves that take no gradient: no VJP sends one a contribution,
# so its gradient reads None, and an operation whose operands are all
# constants records a constant. Linear of a constant input skips g @ W.
def constant_scripts(x: dict, layer: dict, layer_bias: dict) -> list[dict]:
    batch = constant([2, 3], x["words"])
    target = constant([4, 3], f(*[0.5 - 0.125 * k for k in range(12)]))
    return [
        # The first layer of the MNIST step: the input is a constant.
        program_case(
            "constant_linear_input",
            [batch, layer, layer_bias, linear(0, 1, 2), op("sum", 4)],
            5,
        ),
        program_case(
            "constant_linear_relu_linear",
            [
                batch,
                layer,
                layer_bias,
                leaf([2, 4], f(*[0.25 * k - 1.0 for k in range(8)])),
                leaf([2], f(0.5, -0.25)),
                linear(0, 1, 2),
                op("relu", 6),
                linear(7, 3, 4),
                entropy(9, [1, 0]),
            ],
            10,
        ),
        # The weight is the constant: x @ W^T sends only g @ W, to x.
        program_case(
            "constant_weight",
            [
                leaf([2, 3], x["words"]),
                constant([4, 3], layer["words"]),
                op("matmul_nt", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        # A target that takes no gradient, as in mse_loss(prediction, y).
        program_case(
            "constant_mse_target",
            [layer, target, op("mse_loss", 0, 1)],
            2,
        ),
        # Every operand a constant: the product is a constant too, and the
        # sum that reads it with a leaf is frozen at it.
        program_case(
            "constant_of_constants",
            [
                batch,
                constant([2, 3], f(0.5, -1.0, 2.0, 0.25, 4.0, -0.5)),
                op("mul", 0, 1),
                leaf([2, 3], f(1.0, -2.0, 0.5, 3.0, 0.25, -1.0)),
                op("mul", 2, 3),
                op("relu", 0),
                op("add", 4, 5),
                op("sum", 6),
            ],
            7,
        ),
        program_case(
            "constant_matmul_left",
            [
                batch,
                leaf([3, 2], f(1.0, 0.5, -1.0, 2.0, 0.25, -0.5)),
                op("matmul", 0, 1),
                op("relu", 1),
                op("matmul", 0, 3),
                op("sub", 2, 4),
                op("sum", 5),
            ],
            6,
        ),
        # SGD keeps a constant, whose slot is empty.
        {
            "id": "constant_sgd_keeps",
            "steps": [
                batch,
                layer,
                layer_bias,
                linear(0, 1, 2),
                op("sum", 4),
                run(5),
                sgd(word(0.5), [0, 1, 2]),
                *reads(6),
            ],
        },
    ]


# SGD steps: p - lr * g for every listed leaf whose slot holds g.
def sgd_scripts(x: dict, layer: dict, layer_bias: dict) -> list[dict]:
    tenth, third = word(0.1), word(1.0 / 3.0)
    head = leaf([2, 4], f(*[0.25 * k - 1.0 for k in range(8)]))
    head_bias = leaf([2], f(0.5, -0.25))
    mlp = [linear(0, 1, 2), op("relu", 6), linear(7, 3, 4), op("sum", 9)]
    again = [linear(0, 1, 2), op("relu", 12), linear(13, 3, 4), op("sum", 15)]
    pair = [leaf([3], f(0.5, -1.0, 2.0)), leaf([3], f(3.0, 0.25, -2.0))]
    trained = [op("mul", 0, 1), op("sum", 2), run(3)]
    # Each p is paired with a g from another edge class: x * w seeds 1.0, so
    # p's gradient is w's word and w's is p's.
    weights = [EDGES[(7 * k + 3) % len(EDGES)] for k in range(len(EDGES))]
    return [
        # Linear -> ReLU -> Linear: forward, backward, step, zero_grad, then
        # a second forward, backward and step on the updated parameters.
        {
            "id": "sgd_two_layers",
            "steps": [
                x,
                layer,
                layer_bias,
                head,
                head_bias,
                *mlp,
                run(10),
                sgd(tenth, [1, 2, 3, 4]),
                *[{"op": "read", "id": index} for index in range(1, 5)],
                ZERO_GRAD,
                *again,
                run(16),
                sgd(tenth, [1, 2, 3, 4]),
                *reads(17),
            ],
        },
        {
            "id": "sgd_edges",
            "steps": [
                leaf([len(EDGES)], list(EDGES)),
                leaf([len(EDGES)], weights),
                op("mul", 0, 1),
                op("sum", 2),
                run(3),
                sgd(third, [0, 1]),
                *reads(4),
            ],
        },
        # Products that round, cancel, underflow to denormals and overflow.
        {
            "id": "sgd_rounding",
            "steps": [
                leaf([12], list(ROUNDING)),
                leaf([12], [ROUNDING[(5 * k + 2) % 12] for k in range(12)]),
                leaf([12], f(*[1e-38, -3e-39, 1e-45, 2.5e38, -3.4e38, 0.75] * 2)),
                op("mul", 0, 1),
                op("mul", 3, 2),
                op("sum", 4),
                run(5),
                sgd(word(0.01), [0, 1, 2]),
                sgd(third, [2, 0]),
                *reads(6),
            ],
        },
        # A parameter that backward never reaches keeps None and its value;
        # after zero_grad, a step changes nothing.
        {
            "id": "sgd_empty_slot",
            "steps": [
                *pair,
                leaf([], f(4.0)),
                *trained,
                sgd(tenth, [0, 1, 2]),
                *reads(5),
                ZERO_GRAD,
                sgd(tenth, [0, 1, 2]),
                *reads(5),
            ],
        },
        # A step before any backward and a step over no parameters change
        # nothing; a parameter listed twice is updated twice.
        {
            "id": "sgd_no_gradients",
            "steps": [*pair, sgd(tenth, [0, 1]), *reads(2)],
        },
        {
            "id": "sgd_duplicate",
            "steps": [
                *pair,
                *trained,
                sgd(tenth, []),
                sgd(tenth, [1, 0, 1]),
                *reads(4),
            ],
        },
        # Every id is checked before any update: an unknown id is refused and
        # the earlier parameter keeps its value.
        {
            "id": "sgd_unknown_param",
            "steps": [*pair, *trained, sgd(tenth, [0, 4]), *reads(4)],
        },
        {
            "id": "sgd_not_leaf",
            "steps": [*pair, *trained, sgd(tenth, [0, 2]), *reads(4)],
        },
        # The learning rate is checked after the ids and before any update,
        # as PyTorch's `lr < 0.0`: -0.25, -inf and the smallest negative
        # denormal are refused and the tape keeps its values; -0.0, +0.0,
        # NaNs of either sign and +inf are accepted.
        {
            "id": "sgd_rate_negative",
            "steps": [
                *pair,
                *trained,
                sgd(word(-0.25), [0, 1]),
                sgd(0xFF800000, [0, 1]),
                sgd(0x80000001, [0, 1]),
                *reads(4),
            ],
        },
        {
            "id": "sgd_rate_signed_zero",
            "steps": [
                leaf([4], f(0.0, NEGATIVE_ZERO, 1.5, NEGATIVE_ZERO)),
                leaf([4], f(2.0, -3.0, 0.5, 2.0)),
                *trained,
                sgd(NEGATIVE_ZERO, [0, 1]),
                *reads(4),
                sgd(0, [0, 1]),
                *reads(4),
            ],
        },
        {
            "id": "sgd_rate_nan",
            "steps": [
                *pair,
                *trained,
                sgd(0x7FC00000, [0]),
                sgd(0xFFC00000, [1]),
                *reads(4),
            ],
        },
        {
            "id": "sgd_rate_infinity",
            "steps": [
                leaf([3], f(0.5, -1.0, 2.0)),
                leaf([3], f(3.0, 0.0, -2.0)),
                *trained,
                sgd(INFINITY, [0, 1]),
                *reads(4),
            ],
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

# Words for the rounding sums over axes.
SUM_POOL = tuple(
    word(number) for number in (0.1, -0.3, 7.25, 1e-3, 1.0, -2.0, 0.5, 3.0, 1e8, -1e8)
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


def losses() -> list[dict]:
    logits = leaf(
        [4, 3], f(1.0, 2.0, 0.5, -1.0, 0.0, 3.0, 2.5, 2.5, -0.5, 0.1, -0.3, 7.25)
    )
    targets = [1, 2, 0, 2]
    x = leaf([2, 3], f(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
    y = leaf([2, 3], f(0.5, 1.0, 0.5, -1.0, 2.0, 0.75))
    big = f(100.0, 200.0, 89.0, -1e30, 0.0, 1e30)
    infinities = f(INFINITY, 1.0, 2.0, 1.0, NEGATIVE_INFINITY, 0.5)
    infinities += f(NEGATIVE_INFINITY, NEGATIVE_INFINITY, NEGATIVE_INFINITY)
    layer = leaf([4, 3], f(*[0.25 * k - 1.0 for k in range(12)]))
    return [
        program_case("mse", [x, y, op("mse_loss", 0, 1)], 2),
        program_case(
            "mse_broadcast_target",
            [x, leaf([3], f(0.1, -0.3, 1e-3)), op("mse_loss", 0, 1)],
            2,
        ),
        # Six cells, so 2 / N rounds, and a scale of 3 after the loss, so
        # (2 / N * d) * g differs from (d * g) * (2 / N).
        program_case(
            "mse_scaled",
            [
                leaf([2, 3], f(0.1, -0.3, 7.25, 1e-3, 3.0, -2.0)),
                leaf([2, 3], f(1.0, 0.5, 0.1, 1e8, -1e-3, -2.0)),
                op("mse_loss", 0, 1),
                leaf([], f(3.0)),
                op("mul", 2, 3),
            ],
            4,
        ),
        program_case(
            "mse_equal_and_zeros",
            [
                leaf([4], f(1.0, 0.0, NEGATIVE_ZERO, 0.0)),
                leaf([4], f(1.0, NEGATIVE_ZERO, 0.0, 0.0)),
                op("mse_loss", 0, 1),
            ],
            2,
        ),
        program_case(
            "mse_specials",
            [
                leaf([4], f(INFINITY, 1e30, QUIET_NAN, 1.0)),
                leaf([4], f(1.0, -1e30, 2.0, INFINITY)),
                op("mse_loss", 0, 1),
            ],
            2,
        ),
        program_case(
            "mse_empty",
            [leaf([0, 3], []), leaf([3], f(1.0, 2.0, 3.0)), op("mse_loss", 0, 1)],
            2,
        ),
        {
            "id": "mse_refused",
            "steps": [
                x,
                leaf([2], f(1.0, 2.0)),
                op("mse_loss", 0, 1),
                op("mse_loss", 0, 7),
                op("mse_loss", 0, 0),
                run(2),
                *reads(3),
            ],
        },
        program_case("cross_entropy", [logits, entropy(0, targets)], 1),
        program_case(
            "cross_entropy_scaled",
            [
                leaf(
                    [3, 4],
                    f(0.1, -0.3, 7.25, 1e-3, 3.0, -2.0, 0.5, 0.5, 1.0, 2.0, 3.0, 4.0),
                ),
                entropy(0, [3, 0, 2]),
                leaf([], f(3.0)),
                op("mul", 1, 2),
            ],
            3,
        ),
        # Logits whose unshifted exp overflows, and a row of -1e30 and 1e30.
        program_case(
            "cross_entropy_large",
            [leaf([2, 3], big), entropy(0, [0, 1])],
            1,
        ),
        program_case(
            "cross_entropy_large_target",
            [leaf([2, 3], big), entropy(0, [2, 2])],
            1,
        ),
        # +inf in a row, -inf at a target, and a row of -inf only.
        program_case(
            "cross_entropy_infinities",
            [leaf([3, 3], infinities), entropy(0, [1, 1, 0])],
            1,
        ),
        program_case(
            "cross_entropy_nan",
            [leaf([2, 3], f(1.0, QUIET_NAN, 2.0, 0.5, 0.25, 0.0)), entropy(0, [0, 2])],
            1,
        ),
        # Equal logits, and a tie of +0.0 and -0.0 in a one-class row.
        program_case(
            "cross_entropy_ties",
            [
                leaf([2, 3], f(5.0, 5.0, 5.0, 0.0, NEGATIVE_ZERO, 0.0)),
                entropy(0, [1, 1]),
            ],
            1,
        ),
        program_case(
            "cross_entropy_one_class",
            [leaf([3, 1], f(2.0, NEGATIVE_ZERO, -7.5)), entropy(0, [0, 0, 0])],
            1,
        ),
        program_case(
            "cross_entropy_empty",
            [leaf([0, 3], []), entropy(0, [])],
            1,
        ),
        # No rows and more columns than a tensor holds cells: valid, since
        # [0, columns] has no cell, so the loss is NaN and the gradient empty.
        program_case(
            "cross_entropy_empty_wide",
            [leaf([0, 1048577], []), entropy(0, [])],
            1,
        ),
        scaled_entropy("one_class", [2, 1], f(2.0, -7.5), [0, 0]),
        scaled_entropy("two_class", [2, 2], f(0.0, 1.0, 3.0, -1.0), [1, 0]),
        # The logits also feed a sum, so their slot adds the sum's gradient
        # before the two cross_entropy contributions.
        program_case(
            "cross_entropy_fan_out",
            [logits, entropy(0, targets), op("sum", 0), op("add", 1, 2)],
            3,
        ),
        {
            "id": "cross_entropy_refused",
            "steps": [
                logits,
                entropy(0, [1, 2, 3, 0]),
                entropy(0, [1, 2, 0]),
                entropy(0, [1, 2, 0, 0, 1]),
                leaf([1, 2, 3], f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
                entropy(1, [0]),
                entropy(5, [0]),
                op("sum", 0),
                run(2),
                *reads(3),
            ],
        },
        # Linear, relu, Linear and cross_entropy: the MLP of a classifier.
        program_case(
            "cross_entropy_mlp",
            [
                leaf([2, 3], f(1.0, -2.0, 0.5, 3.0, 0.25, -1.0)),
                layer,
                leaf([4], f(0.25, -0.5, 1.0, 2.0)),
                leaf([3, 4], f(*[0.5 - 0.125 * k for k in range(12)])),
                leaf([3], f(0.1, 0.0, -0.1)),
                linear(0, 1, 2),
                op("relu", 6),
                linear(7, 3, 4),
                entropy(9, [2, 0]),
            ],
            10,
        ),
        program_case(
            "mse_linear",
            [
                x,
                layer,
                leaf([4], f(0.25, -0.5, 1.0, 2.0)),
                linear(0, 1, 2),
                leaf([2, 4], f(*[0.5 * k for k in range(8)])),
                op("mse_loss", 4, 5),
            ],
            6,
        ),
    ]


# The loss of logits times g = +0.0, -0.0, 1.0 and NaN, one backward each,
# the slots emptied between them. With one class PyTorch's row sum of the
# one-hot gradient is its one word, so at g = +0.0 the logits' gradient is
# +0.0, where adding +0.0 to that row sum would give -0.0.
def scaled_entropy(
    name: str, shape: list[int], words: list[int], targets: list[int]
) -> dict:
    steps = [leaf(shape, words), entropy(0, targets)]
    for scale in (0, NEGATIVE_ZERO, ONE, QUIET_NAN):
        steps += [leaf([], [scale]), op("mul", 1, len(steps))]
    count = len(steps)
    for output in range(3, count, 2):
        if output > 3:
            steps.append(ZERO_GRAD)
        steps += [run(output), *reads(count)]
    return {"id": f"cross_entropy_{name}_scaled", "steps": steps}


def loss_generated(count: int, seed: int) -> list[dict]:
    state = seed

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    def number() -> int:
        return word((draw(4001) - 2000) / 173.0)

    result = []
    for index in range(count):
        rows, columns = 1 + draw(6), 1 + draw(12)
        shape = [rows, columns]
        steps = [leaf(shape, [number() for _ in range(rows * columns)])]
        if index % 2 == 0:
            steps.append(entropy(0, [draw(columns) for _ in range(rows)]))
        else:
            other = [shape, [columns], [rows, 1], []][draw(4)]
            steps.append(leaf(other, [number() for _ in range(math.prod(other))]))
            steps.append(op("mse_loss", 0, 1))
        node = len(steps) - 1
        if draw(2):
            steps += [leaf([], [number()]), op("mul", node, node + 1)]
        output = len(steps) - 1
        steps += [run(output), *reads(len(steps))]
        result.append({"id": f"loss_{index:02d}", "steps": steps})
    return result


# torch.matmul scripts: 1-D promotion, batches, broadcast batches, the
# rank-2 pair that records `matmul`, refusals, empty batches and products,
# constants, special words, and generated products over rounding words.
BATCHED_SHAPES = (
    ([3], [3]),
    ([2, 3], [3]),
    ([3], [3, 2]),
    ([2, 2, 3], [2, 3, 2]),
    ([2, 3], [4, 3, 2]),
    ([4, 2, 3], [3, 2]),
    ([2, 1, 2, 3], [3, 3, 2]),
    ([3, 1, 2], [2, 1, 2, 3]),
    ([3], [2, 3, 2]),
    ([2, 2, 3], [3]),
    ([1, 3], [2, 3, 1]),
    ([2, 3], [3, 2]),
)


def counted_words(count: int, pool: tuple[int, ...], start: int) -> list[int]:
    return [pool[(start + 3 * index) % len(pool)] for index in range(count)]


# x @ w, times a weight of the product's shape, summed: the product's
# gradient is the weight, a different word in each cell, so a VJP that reads
# g in the wrong order gives other words.
def weighted(x: dict, w: dict, left: list[int], right: list[int], seed: int) -> list:
    plan = batched_plan(left, right)
    if isinstance(plan, int):
        raise GateError(f"A batched script refuses its own product: {plan}")
    shape = plan[3]
    weights = leaf(shape, counted_words(math.prod(shape), EXACT, 7 * seed + 3))
    return [x, w, op("batched_matmul", 0, 1), weights, op("mul", 2, 3), op("sum", 4)]


# torch.matmul refusals of capacity, with no input word past 2^20. The
# forward output [1, 1025, 1025] of [1, 1025, 0] @ [1, 0, 1025] has more than
# 2^20 elements: the product is refused and the next leaf takes its id. In
# the other two the forward output [1025, 0] is empty, but one operand's
# gradient over the output's batch, before it is summed to the operand's
# shape, is [1025, 1025]: [1025] @ [1025, 1025, 0] for the left operand's,
# [1025, 0, 1025] @ [1025] for the right one's. A first backward from a sum
# of the leaf fills its slot; the backward through the product is refused
# and leaves every slot as it was. The other operand is a constant, so
# PyTorch's record of each slot read after the refusal is the same.
def capacity_scripts() -> list[dict]:
    ones = [ONE] * 1025
    return [
        {
            "id": "batched_output_capacity",
            "steps": [
                leaf([1, 1025, 0], []),
                leaf([1, 0, 1025], []),
                op("batched_matmul", 0, 1),
                leaf([], f(1.0)),
                *reads(2),
            ],
        },
        {
            "id": "batched_left_gradient_capacity",
            "steps": [
                leaf([1025], ones),
                constant([1025, 1025, 0], []),
                op("sum", 0),
                op("batched_matmul", 0, 1),
                op("sum", 3),
                run(2),
                *reads(5),
                run(4),
                *reads(3),
            ],
        },
        {
            "id": "batched_right_gradient_capacity",
            "steps": [
                constant([1025, 0, 1025], []),
                leaf([1025], ones),
                op("sum", 1),
                op("batched_matmul", 0, 1),
                op("sum", 3),
                run(2),
                *reads(5),
                run(4),
                *reads(3),
            ],
        },
    ]


def batched() -> list[dict]:
    result = []
    for index, (left, right) in enumerate(BATCHED_SHAPES):
        x = leaf(left, counted_words(math.prod(left), EXACT, index))
        w = leaf(right, counted_words(math.prod(right), EXACT, 5 * index + 1))
        name = "x".join(map(str, left)) + "_by_" + "x".join(map(str, right))
        result.append(
            program_case(f"batched_{name}", weighted(x, w, left, right, index), 5)
        )
    stack = leaf([2, 2, 2], f(1.0, -2.0, 0.5, 3.0, 2.5, -1.0, 0.0, 1.5))
    vector = leaf([3], f(0.5, -1.0, 2.0))
    special = leaf(
        [2, 1, 3], [INFINITY, NEGATIVE_ZERO, QUIET_NAN, word(2.0), 0, NEGATIVE_INFINITY]
    )
    weights = leaf([3, 2], f(1.0, -0.5, NEGATIVE_ZERO, 2.0, 0.0, -3.0))
    result += [
        program_case(
            "batched_dot_seed",
            [vector, leaf([3], f(4.0, 0.25, -1.0)), op("batched_matmul", 0, 1)],
            2,
        ),
        program_case("batched_self_dot", [vector, op("batched_matmul", 0, 0)], 1),
        program_case(
            "batched_self_stack", [stack, op("batched_matmul", 0, 0), op("sum", 1)], 2
        ),
        program_case(
            "batched_seed",
            [stack, leaf([2, 1], f(3.0, -0.5)), op("batched_matmul", 0, 1)],
            2,
        ),
        program_case(
            "batched_special_words",
            [special, weights, op("batched_matmul", 0, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "batched_constant_left",
            [
                constant([2, 2, 3], counted_words(12, EXACT, 2)),
                leaf([3, 2], f(1.0, 2.0, -1.0, 0.5, 3.0, -2.0)),
                op("batched_matmul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "batched_constant_right",
            [
                leaf([3], f(1.0, -2.0, 0.5)),
                constant([2, 3, 2], counted_words(12, EXACT, 7)),
                op("batched_matmul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "batched_layer_relu",
            [
                leaf([2, 2, 3], counted_words(12, EXACT, 4)),
                leaf([3, 4], counted_words(12, EXACT, 9)),
                leaf([4], f(0.25, -0.5, 1.0, -2.0)),
                op("batched_matmul", 0, 1),
                op("add", 3, 2),
                op("relu", 4),
                leaf([4], f(1.0, -1.0, 0.5, 2.0)),
                op("batched_matmul", 5, 6),
                op("sum", 7),
            ],
            8,
        ),
        program_case(
            "batched_empty_batch",
            [
                leaf([0, 2, 3], []),
                leaf([3, 2], f(1.0, 2.0, -1.0, 0.5, 3.0, -2.0)),
                op("batched_matmul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "batched_zero_inner",
            [
                leaf([2, 3, 0], []),
                leaf([0, 4], []),
                op("batched_matmul", 0, 1),
                op("sum", 2),
            ],
            3,
        ),
        program_case(
            "batched_empty_vector",
            [leaf([0, 3], []), vector, op("batched_matmul", 0, 1), op("sum", 2)],
            3,
        ),
        {
            "id": "batched_refusals",
            "steps": [
                leaf([], f(2.0)),
                vector,
                leaf([2], f(1.0, 2.0)),
                leaf([2, 1, 3], counted_words(6, EXACT, 1)),
                leaf([3, 3, 2], counted_words(18, EXACT, 2)),
                op("batched_matmul", 0, 1),
                op("batched_matmul", 1, 0),
                op("batched_matmul", 1, 2),
                op("batched_matmul", 3, 4),
                op("batched_matmul", 3, 9),
                op("batched_matmul", 3, 1),
                run(5),
                *reads(6),
            ],
        },
        *capacity_scripts(),
    ]
    state = 0x6C078965
    for index in range(8):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        batch = [[], [2], [3, 1], [1, 2]][state % 4]
        other = [[], [1], [2], [3, 2]][(state >> 4) % 4]
        if broadcast(batch, other) is None:
            other = []
        m, k, n = 1 + (state >> 8) % 3, 2 + (state >> 10) % 4, 1 + (state >> 12) % 3
        left = [k] if (state >> 14) % 5 == 0 else [*batch, m, k]
        right = [k] if (state >> 16) % 5 == 0 and len(left) > 1 else [*other, k, n]
        x = leaf(left, counted_words(math.prod(left), ROUNDING, state >> 18))
        w = leaf(right, counted_words(math.prod(right), ROUNDING, state >> 20))
        result.append(
            program_case(
                f"batched_rounding_{index:02d}", weighted(x, w, left, right, index), 5
            )
        )
    return result


# Sums over axes, PyTorch's sum(dim, keepdim): one axis, several in any
# order, none (every axis), keepdim on and off, rank 0, an axis of extent 1
# that holds -0.0, empty inputs, and the refusals. A product with a weight
# after the sum makes each gradient word its own.
def sum_dims_scripts() -> list[dict]:
    grid = leaf([2, 3], f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    cube = leaf([2, 3, 4], f(*[0.5 * k - 3.0 for k in range(24)]))
    scalar = leaf([], f(-2.5))
    return [
        program_case(
            "sum_dims_rows",
            [grid, dims(0, [1]), leaf([2], f(2.0, -0.5)), op("mul", 1, 2)],
            3,
        ),
        program_case(
            "sum_dims_columns_keepdim",
            [grid, dims(0, [0], True), leaf([1, 3], f(0.5, -1.0, 4.0))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_every_axis",
            [grid, dims(0, []), dims(0, [], True), dims(0, [1, 0])]
            + [op("add", 1, 2), op("mul", 4, 3)],
            5,
        ),
        program_case(
            "sum_dims_two_of_three",
            [cube, dims(0, [2, 0]), leaf([3], f(1.0, -2.0, 0.25)), op("mul", 1, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_middle_keepdim",
            [cube, dims(0, [1], True), leaf([4], f(0.5, -1.0, 2.0, 3.0))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_chain",
            [cube, dims(0, [2]), dims(1, [0], True), op("mul", 2, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_rank_zero",
            [scalar, dims(0, [0]), dims(0, [0], True), dims(0, [])]
            + [op("mul", 1, 2), op("mul", 4, 3)],
            5,
        ),
        program_case(
            "sum_dims_negative_zero",
            [leaf([2, 1], f(NEGATIVE_ZERO, -3.0)), dims(0, [1])]
            + [leaf([2], f(-1.0, 2.0)), op("mul", 1, 2)],
            3,
        ),
        program_case(
            "sum_dims_empty",
            [leaf([0, 3], []), dims(0, [0]), dims(0, [1], True)]
            + [op("sum", 1), op("sum", 2), op("add", 3, 4)],
            5,
        ),
        program_case("sum_dims_seed_ones", [cube, dims(0, [0, 1])], 1),
        program_case(
            "sum_dims_constant",
            [constant([2, 3], grid["words"]), dims(0, [0]), leaf([3], f(1.0, 2.0, 3.0))]
            + [op("mul", 1, 2), dims(3, [0]), op("mul", 4, 4)],
            5,
        ),
        # Rounding words: PyTorch adds a row of 20 in a vectorized order, so
        # the row sums may differ from the ordered fold within their bound;
        # it adds the columns row by row, as the fold does.
        {
            "id": "sum_dims_rounding",
            "steps": [
                leaf(
                    [4, 20], [SUM_POOL[(i * 7 + 3 * (i // 5)) % 10] for i in range(80)]
                ),
                dims(0, [1]),
                dims(0, [0], True),
                run(1),
                run(2),
                *reads(3),
            ],
        },
        {
            "id": "sum_dims_refused",
            "steps": [
                grid,
                dims(1, [0]),
                dims(0, [2]),
                dims(0, [0, 0]),
                dims(0, [0, 0, 5]),
                dims(0, [1, 1, 0]),
                dims(0, [4294967295]),
                scalar,
                dims(1, [1]),
                dims(0, [1]),
                run(2),
                *reads(3),
            ],
        },
        # Negative axes count from the last: -1 is the last axis.
        program_case(
            "sum_dims_negative_rows",
            [grid, dims(0, [-1]), leaf([2], f(2.0, -0.5)), op("mul", 1, 2)],
            3,
        ),
        program_case(
            "sum_dims_negative_keepdim",
            [cube, dims(0, [-2], True), leaf([4], f(0.5, -1.0, 2.0, 3.0))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_negative_mixed",
            [cube, dims(0, [-1, 0]), leaf([3], f(1.0, -2.0, 0.25)), op("mul", 1, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "sum_dims_negative_rank_zero",
            [scalar, dims(0, [-1]), dims(0, [-1], True), op("mul", 1, 2)],
            3,
        ),
        {
            "id": "sum_dims_negative_refused",
            "steps": [
                grid,
                dims(0, [-3]),
                dims(0, [-1, 1]),
                dims(0, [0, -2]),
                dims(0, [5, -3]),
                dims(0, [-3, 0, 0]),
                scalar,
                dims(1, [-2]),
                dims(1, [0, -1]),
                dims(0, [-2]),
                run(2),
                *reads(3),
            ],
        },
    ]


# Sums over random axes of exact words in random shapes of rank 0 to 3,
# mixed with add, mul and other sums, so every order of addition gives the
# same words.
def dims_generated(count: int, seed: int) -> list[dict]:
    state = seed

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    result = []
    for index in range(count):
        rank = draw(4)
        first = [1 + draw(3) for _ in range(rank)]
        shapes = [first]
        steps = [
            leaf(first, [EXACT[draw(len(EXACT))] for _ in range(math.prod(first))])
        ]
        for _ in range(2 + draw(4)):
            position = len(shapes)
            a = draw(position)
            shape = shapes[a]
            if draw(3):
                axes = [axis for axis in range(len(shape)) if draw(2)]
                if len(axes) > 1 and draw(2):
                    axes.reverse()
                keepdim = bool(draw(2))
                plan = sum_dims_plan(shape, axes)
                if isinstance(plan, int):
                    raise GateError("A generated sum over axes was refused")
                steps.append(dims(a, axes, keepdim))
                shapes.append(plan[0] if keepdim else plan[1])
                continue
            other = [extent for extent in shape if draw(2)]
            words = [EXACT[draw(len(EXACT))] for _ in range(math.prod(other))]
            steps += [leaf(other, words), op(("add", "mul")[draw(2)], a, position)]
            shapes += [other, broadcast(shape, other) or []]
        steps.append(op("sum", len(shapes) - 1))
        count = len(shapes) + 1
        steps += [run(count - 1), *reads(count)]
        if draw(2):
            steps += [ZERO_GRAD, run(count - 2), *reads(count)]
        result.append({"id": f"dims_{index:02d}", "steps": steps})
    return result


# Basic slicing: a[..., start:stop:step, ...], narrow and select, each a
# copy whose backward scatters the gradient into +0.0 words. The words only
# move, so every word must be PyTorch's, NaN payloads included where nothing
# else touches them.
def slice_scripts() -> list[dict]:
    grid = leaf([2, 5], f(*[0.5 * k - 1.0 for k in range(10)]))
    cube = leaf([2, 3, 4], f(*[0.25 * k - 3.0 for k in range(24)]))
    line = leaf([4], f(1.0, -2.0, 0.5, 3.0))
    scalar = leaf([], f(-2.5))
    weights = leaf([2, 2], f(1.0, -0.5, 2.0, 0.25))
    special = leaf(
        [2, 3],
        [INFINITY, NEGATIVE_ZERO, OTHER_NAN, word(2.0), NEGATIVE_INFINITY, QUIET_NAN],
    )
    largest = 4294967295
    return [
        program_case(
            "slice_columns_step",
            [grid, sliced(0, 1, 1, 5, 2), weights, op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "slice_rows",
            [grid, sliced(0, 0, 1, 2), leaf([1, 5], f(1.0, 2.0, 3.0, 4.0, 5.0))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case("slice_seed_ones", [grid, sliced(0, 1, 1, 5, 2)], 1),
        program_case(
            "slice_stop_clamped",
            [cube, sliced(0, 2, 1, largest, 2), op("mul", 1, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "slice_start_past_the_extent",
            [cube, sliced(0, 1, largest, largest), op("sum", 1)],
            2,
        ),
        program_case(
            "slice_stop_before_start",
            [grid, sliced(0, 1, 4, 2), op("sum", 1), sliced(0, 1, 3, 3), op("sum", 3)]
            + [op("add", 2, 4)],
            5,
        ),
        program_case(
            "slice_two_axes",
            [cube, sliced(0, 1, 0, 3, 2), sliced(1, 2, 1, 4, 2), op("mul", 2, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "slice_overlapping",
            [line, sliced(0, 0, 0, 3), sliced(0, 0, 1, 4), op("mul", 1, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "slice_step_past_the_extent",
            [line, sliced(0, 0, 1, 4, 7), op("mul", 1, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "narrow_inner",
            [cube, narrowed(0, 2, 1, 2), op("mul", 1, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "narrow_empty_at_the_end",
            [grid, narrowed(0, 1, 5, 0), op("sum", 1)],
            2,
        ),
        program_case(
            "select_middle",
            [cube, selected(0, 1, 2), leaf([4], f(1.0, -1.0, 2.0, 0.5))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "select_to_rank_zero",
            [line, selected(0, 0, 3), selected(0, 0, 1), op("mul", 1, 2)],
            3,
        ),
        program_case(
            "select_twice",
            [cube, selected(0, 0, 1), selected(1, 1, 3), op("mul", 2, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "slice_special_words",
            [special, sliced(0, 1, 0, 3, 2), selected(0, 0, 1)],
            2,
        ),
        program_case(
            "slice_constant",
            [constant([2, 5], grid["words"]), sliced(0, 1, 0, 4, 2), weights]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "slice_rank_four",
            [
                leaf([2, 1, 3, 2], f(*[k - 5.0 for k in range(12)])),
                sliced(0, 2, 2, 3),
                narrowed(1, 3, 1, 1),
                op("mul", 2, 2),
                op("sum", 3),
            ],
            4,
        ),
        # Inputs with no element whose other extents multiply past 2^48 - 1,
        # the largest Nat Bend's runtime holds: slices, a narrow and a select
        # at the axis of the zero and away from it, each seeded at once.
        {
            "id": "slice_empty_wide_outer",
            "steps": [
                leaf([2**24, 2**24, 0], []),
                sliced(0, 2, 0, 0),
                selected(0, 0, 1),
                narrowed(0, 1, 2**24, 0),
                run(1),
                run(2),
                run(3),
                *reads(4),
            ],
        },
        {
            "id": "slice_empty_wide_inner",
            "steps": [
                leaf([0, 2**24, 2**24], []),
                sliced(0, 0, 0, 0),
                narrowed(0, 0, 0, 0),
                selected(0, 2, 2**24 - 1),
                run(1),
                run(2),
                run(3),
                *reads(4),
            ],
        },
        # A slice whose axis is 10^8 long: its length takes one division.
        {
            "id": "slice_long_axis",
            "steps": [
                leaf([0, 10**8], []),
                sliced(0, 1, 0, largest),
                sliced(0, 1, 1, largest, 3),
                run(1),
                run(2),
                *reads(3),
            ],
        },
        # Refusals: an unknown id, an axis past the rank, a zero step, a
        # rank-0 input; a narrow past the extent, one whose start is the
        # largest U32, one past the rank and one of a rank-0 input; a select
        # at the extent, one past the rank and one of a rank-0 input.
        {
            "id": "slice_refused",
            "steps": [
                grid,
                sliced(1, 0, 0, 1),
                sliced(0, 2, 0, 1),
                sliced(0, 1, 0, 5, 0),
                scalar,
                sliced(1, 0, 0, 1),
                narrowed(0, 1, 3, 3),
                narrowed(0, 1, largest, 2),
                narrowed(0, 2, 0, 1),
                narrowed(1, 0, 0, 1),
                selected(0, 1, 5),
                selected(0, 2, 0),
                selected(1, 0, 0),
                sliced(0, 1, 1, 4, 2),
                run(2),
                *reads(3),
            ],
        },
        # More than 2^15 cells, so both builds fork: x = a * b * c of
        # [3, 200, 100], built on the tape from 303 words, and its slice
        # 1:200 along the middle axis. The gradients of a, b and c sum the
        # products of the scattered ones; exact words keep every sum exact in
        # any order.
        {
            "id": "slice_parallel",
            "steps": [
                leaf([3, 1, 1], f(1.0, -2.0, 0.5)),
                leaf([1, 200, 1], counted_words(200, EXACT, 1)),
                leaf([1, 1, 100], counted_words(100, EXACT, 4)),
                op("mul", 0, 1),
                op("mul", 3, 2),
                sliced(4, 1, 1, 200),
                op("sum", 5),
                run(6),
                *reads(3),
                {"op": "read", "id": 6},
            ],
        },
    ]


# PyTorch's signed positions: an axis counted from the last, a slice's start
# and stop counted from the end and clamped at 0 below -extent, a narrow's
# start in -extent..extent and a select's index in -extent..extent-1.
def signed_slice_scripts() -> list[dict]:
    grid = leaf([2, 5], f(*[0.5 * k - 1.0 for k in range(10)]))
    cube = leaf([2, 3, 4], f(*[0.25 * k - 3.0 for k in range(24)]))
    line = leaf([4], f(1.0, -2.0, 0.5, 3.0))
    scalar = leaf([], f(-2.5))
    return [
        program_case(
            "slice_negative_bounds",
            [grid, sliced(0, -1, -4, -1), op("mul", 1, 1), op("sum", 2)],
            3,
        ),
        program_case(
            "slice_negative_axis_step",
            [cube, sliced(0, -2, -3, 3, 2), sliced(1, -1, 1, -1), op("mul", 2, 2)]
            + [op("sum", 3)],
            4,
        ),
        # A start before -extent is clamped to 0, a stop before -extent ends
        # the slice at its start, and a negative start past a positive stop
        # gives an empty axis.
        program_case(
            "slice_negative_clamped",
            [grid, sliced(0, 1, -100, 3), sliced(0, 1, 1, -100), op("sum", 2)]
            + [sliced(0, 1, -2, 2), op("sum", 4), op("add", 3, 5), op("sum", 1)]
            + [op("add", 6, 7)],
            8,
        ),
        program_case(
            "narrow_negative",
            [cube, narrowed(0, -1, -3, 2), narrowed(1, -3, -2, 2), op("mul", 2, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "narrow_negative_whole_axis",
            [grid, narrowed(0, 1, -5, 5), narrowed(0, -1, -1, 0), op("sum", 1)]
            + [op("sum", 2), op("add", 3, 4)],
            5,
        ),
        program_case(
            "select_negative",
            [cube, selected(0, -1, -1), selected(1, -2, -2), op("mul", 2, 2)]
            + [op("sum", 3)],
            4,
        ),
        program_case(
            "select_negative_to_rank_zero",
            [line, selected(0, -1, -4), selected(0, 0, -1), op("mul", 1, 2)],
            3,
        ),
        # Refusals: an axis before -rank and a rank-0 input (AxisOutOfBounds);
        # a narrow start before -extent, with and without a length, and one
        # at -extent whose length passes the extent (OutOfRange); a narrow
        # whose axis is before -rank and whose start is before -extent too
        # (AxisOutOfBounds first); a select before -extent (OutOfRange), at
        # an axis before -rank and of a rank-0 input (AxisOutOfBounds).
        {
            "id": "slice_negative_refused",
            "steps": [
                grid,
                sliced(0, -3, 0, 1),
                scalar,
                sliced(1, -1, 0, 1),
                narrowed(0, -1, -6, 0),
                narrowed(0, 1, -6, 1),
                narrowed(0, 1, -5, 6),
                narrowed(0, -3, -9, 0),
                selected(0, -1, -6),
                selected(0, -3, 0),
                selected(1, -1, -1),
                sliced(0, -1, -4, -1, 2),
                run(2),
                *reads(3),
            ],
        },
    ]


# Slices, narrows and selects of exact words on random axes of random shapes
# of rank 1 to 3, with starts, stops and steps past the extent, mixed with
# mul and other slices.
def slices_generated(count: int, seed: int) -> list[dict]:
    state = seed

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    result = []
    for index in range(count):
        first = [1 + draw(4) for _ in range(1 + draw(3))]
        shapes = [first]
        steps = [leaf(first, counted_words(math.prod(first), EXACT, draw(10)))]
        for _ in range(2 + draw(4)):
            a = draw(len(shapes))
            shape = shapes[a]
            if not shape or draw(4) == 0:
                steps.append(op("mul", a, a))
                shapes.append(shape)
                continue
            dim = draw(len(shape))
            kind = draw(3) if shape[dim] else 0
            if kind == 0:
                step = sliced(a, dim, draw(shape[dim] + 2), draw(shape[dim] + 3))
                step["step"] = 1 + draw(3)
            elif kind == 1:
                start = draw(shape[dim] + 1)
                step = narrowed(a, dim, start, draw(shape[dim] - start + 1))
            else:
                step = selected(a, dim, draw(shape[dim]))
            plan = slice_plan(shape, step)
            if isinstance(plan, int):
                raise GateError("A generated slice was refused")
            steps.append(step)
            shapes.append(plan[3])
        steps.append(op("sum", len(shapes) - 1))
        total = len(shapes) + 1
        steps += [run(total - 1), *reads(total)]
        result.append({"id": f"slices_{index:02d}", "steps": steps})
    return result


# The same with signed axes and positions: axes in -rank..rank-1, slice
# starts and stops in -extent-2..extent+2, narrow starts in -extent..extent
# and select indices in -extent..extent-1, none refused.
def signed_slices_generated(count: int, seed: int) -> list[dict]:
    state = seed

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    def between(low: int, high: int) -> int:
        return low + draw(high - low + 1)

    result = []
    for index in range(count):
        first = [1 + draw(4) for _ in range(1 + draw(3))]
        shapes = [first]
        steps = [leaf(first, counted_words(math.prod(first), EXACT, draw(10)))]
        for _ in range(2 + draw(4)):
            a = draw(len(shapes))
            shape = shapes[a]
            if not shape or draw(4) == 0:
                steps.append(op("mul", a, a))
                shapes.append(shape)
                continue
            dim = between(-len(shape), len(shape) - 1)
            extent = shape[dim]
            kind = draw(3) if extent else 0
            if kind == 0:
                start = between(-extent - 2, extent + 2)
                step = sliced(a, dim, start, between(-extent - 2, extent + 2))
                step["step"] = 1 + draw(3)
            elif kind == 1:
                start = between(-extent, extent)
                room = extent - (start + extent if start < 0 else start)
                step = narrowed(a, dim, start, draw(room + 1))
            else:
                step = selected(a, dim, between(-extent, extent - 1))
            plan = slice_plan(shape, step)
            if isinstance(plan, int):
                raise GateError("A generated signed slice was refused")
            steps.append(step)
            shapes.append(plan[3])
        steps.append(op("sum", len(shapes) - 1))
        total = len(shapes) + 1
        steps += [run(total - 1), *reads(total)]
        result.append({"id": f"signed_slices_{index:02d}", "steps": steps})
    return result


# index_select and gather: copies whose backward adds each gradient word into
# +0.0 words at the input cell it came from, so an entry listed twice gets
# the sum of both words, in output order.
def gather_scripts() -> list[dict]:
    grid = leaf([3, 4], f(*[0.5 * k - 2.0 for k in range(12)]))
    cube = leaf([2, 3, 4], f(*[0.25 * k - 3.0 for k in range(24)]))
    scalar = leaf([], f(-2.5))
    special = leaf(
        [2, 3],
        [INFINITY, NEGATIVE_ZERO, OTHER_NAN, word(2.0), NEGATIVE_INFINITY, QUIET_NAN],
    )
    return [
        program_case(
            "index_select_rows_repeated",
            [grid, index_selected(0, 0, [2, 0, 2]), leaf([3, 4], f(*range(1, 13)))]
            + [op("mul", 1, 2), op("sum", 3)],
            4,
        ),
        program_case(
            "index_select_last_axis",
            [cube, index_selected(0, -1, [3, 3, 0, 1, 3]), op("mul", 1, 1)]
            + [op("sum", 2)],
            3,
        ),
        program_case(
            "index_select_empty_index",
            [grid, index_selected(0, 1, []), op("sum", 1)],
            2,
        ),
        # The gradient words of one input row, added in output order: words
        # that round, so another order gives another word.
        program_case(
            "index_select_rounding_order",
            [
                leaf([3, 2], f(*range(6))),
                index_selected(0, 0, [1, 0, 1, 1, 2, 1]),
                leaf([6, 2], counted_words(12, ROUNDING, 4)),
                op("mul", 1, 2),
                op("sum", 3),
            ],
            4,
        ),
        program_case(
            "gather_columns_repeated",
            [grid, gathered(0, 1, [3, 2], [3, 3, 0, 1, 2, 2]), op("mul", 1, 1)]
            + [op("sum", 2)],
            3,
        ),
        program_case(
            "gather_rows_negative_axis",
            [grid, gathered(0, -2, [2, 4], [2, 0, 1, 2, 2, 2, 0, 0]), op("mul", 1, 1)]
            + [op("sum", 2)],
            3,
        ),
        # A gather longer than the input along its axis.
        program_case(
            "gather_longer_axis",
            [
                cube,
                gathered(0, 1, [2, 5, 4], counted_words(40, (0, 1, 2), 1)),
                leaf([2, 5, 4], counted_words(40, ROUNDING, 2)),
                op("mul", 1, 2),
                op("sum", 3),
            ],
            4,
        ),
        program_case(
            "gather_special_words",
            [special, gathered(0, 1, [2, 2], [2, 0, 1, 1])],
            1,
        ),
        program_case(
            "index_select_constant",
            [constant([3, 4], grid["words"]), index_selected(0, 1, [1, 1]), grid]
            + [index_selected(2, 1, [0, 3]), op("mul", 1, 3), op("sum", 4)],
            5,
        ),
        # Refusals: an axis past the rank and before -rank, an entry at the
        # extent; a gather whose entry count is not the product of its
        # dimensions, at an axis past the rank, of another rank, wider than
        # the input at another axis, with an entry at the extent; an unknown
        # id; a gather index past 2^20 elements (its product, 2^48, is never
        # formed) and one past rank 4.
        {
            "id": "gather_refused",
            "steps": [
                grid,
                index_selected(0, 2, [0]),
                index_selected(0, -3, [0]),
                index_selected(0, 1, [1, 4]),
                gathered(0, 1, [3, 2], [0, 1, 2]),
                gathered(0, 2, [3, 1], [0, 0, 0]),
                gathered(0, 1, [3], [0, 0, 0]),
                gathered(0, 1, [4, 1], [0, 0, 0, 0]),
                gathered(0, 0, [1, 4], [0, 1, 3, 2]),
                index_selected(5, 0, [0]),
                gathered(0, 0, [1, 4], [0, 1, 2, 2]),
                gathered(0, 0, [2**24, 2**24], []),
                gathered(0, 0, [1, 1, 1, 1, 1], [0]),
                run(1),
                *reads(2),
            ],
        },
        # What PyTorch takes and bendtorch refuses: a gather index narrower
        # than the input at another axis (IndexShapeMismatch), an index_select
        # and a gather of a rank-0 input, which PyTorch reads as one element
        # along axis 0 (AxisOutOfBounds), and two empty gather indexes, of
        # another rank (IndexRankMismatch) and wider than the input at another
        # axis (IndexShapeMismatch), which PyTorch takes unchecked.
        {
            "id": "gather_pytorch_accepts",
            "steps": [
                grid,
                scalar,
                gathered(0, 1, [2, 4], [0, 1, 2, 3, 3, 2, 1, 0]),
                index_selected(1, 0, [0]),
                gathered(1, 0, [], [0]),
                gathered(0, 1, [0], []),
                gathered(0, 1, [4, 0], []),
            ],
        },
        # Inputs and indexes with no element whose other extents multiply to
        # 2^48, past the largest Nat Bend's runtime holds: no product of them
        # is formed.
        {
            "id": "gather_empty_wide",
            "steps": [
                leaf([0, 2**24, 2**24], []),
                gathered(0, 1, [0, 2**24, 2**24], []),
                index_selected(0, 2, [5, 1]),
                run(1),
                run(2),
                *reads(3),
            ],
        },
        # More than 2^15 cells, so both builds fork: x = a * b * c of
        # [3, 200, 100], then 150 of its 200 rows along the middle axis,
        # some twice. The gradients of a, b and c sum the products of the
        # added ones; exact words keep every sum exact in any order.
        {
            "id": "index_select_parallel",
            "steps": [
                leaf([3, 1, 1], f(1.0, -2.0, 0.5)),
                leaf([1, 200, 1], counted_words(200, EXACT, 1)),
                leaf([1, 1, 100], counted_words(100, EXACT, 4)),
                op("mul", 0, 1),
                op("mul", 3, 2),
                index_selected(4, 1, [(7 * k) % 200 for k in range(150)]),
                op("sum", 5),
                run(6),
                *reads(3),
                {"op": "read", "id": 6},
            ],
        },
    ]


# index_selects and gathers of exact words at random signed axes of random
# shapes of rank 1 to 3, with repeated entries, mixed with mul.
def gathers_generated(count: int, seed: int) -> list[dict]:
    state = seed

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    result = []
    for index in range(count):
        first = [1 + draw(4) for _ in range(1 + draw(3))]
        shapes = [first]
        steps = [leaf(first, counted_words(math.prod(first), EXACT, draw(10)))]
        for _ in range(2 + draw(3)):
            a = draw(len(shapes))
            shape = shapes[a]
            if not shape or draw(4) == 0:
                steps.append(op("mul", a, a))
                shapes.append(shape)
                continue
            dim = draw(2 * len(shape)) - len(shape)
            axis = dim % len(shape)
            extent = shape[axis]
            length = draw(5)
            if not extent:
                length = 0
            if draw(2):
                entries = [draw(extent) for _ in range(length)] if extent else []
                step = index_selected(a, dim, entries)
            else:
                index_shape = [*shape[:axis], length, *shape[axis + 1 :]]
                total = math.prod(index_shape)
                entries = [draw(extent) for _ in range(total)] if extent else []
                step = gathered(a, dim, index_shape, entries)
            plan = gather_plan(shape, step)
            if isinstance(plan, int):
                raise GateError(f"A generated gather was refused: {plan}")
            steps.append(step)
            shapes.append(plan[0])
        steps.append(op("sum", len(shapes) - 1))
        total = len(shapes) + 1
        steps += [run(total - 1), *reads(total)]
        result.append({"id": f"gathers_{index:02d}", "steps": steps})
    return result


def cases() -> list[dict]:
    result = (
        handwritten()
        + generated("generated", EXACT, 32, 0x5DEECE66)
        + generated("rounding", ROUNDING, 24, 0x2545F491)
        + generated("relu", RELU_POOL, 16, 0x1F2E3D4C, relu=True)
        + losses()
        + loss_generated(16, 0x3C6EF372)
        + sum_dims_scripts()
        + dims_generated(16, 0x6C078965)
        + batched()
        + slice_scripts()
        + slices_generated(16, 0x41C64E6D)
        + signed_slice_scripts()
        + signed_slices_generated(16, 0x2F6B3A91)
        + gather_scripts()
        + gathers_generated(16, 0x1B873593)
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


# F32 division, exp and log as the native lane computes them: the binary64
# operation on the two F32 values, rounded to F32 (exp and log are the C
# library's binary64 exp and log). A NaN operand gives a NaN; its payload is
# not specified.
def div_word(left: int, right: int) -> int:
    nan = nan_operand(left, right, False)
    if nan is not None:
        return nan
    a, b = value(left), value(right)
    if b == 0:
        if a == 0:
            return QUIET_NAN
        negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
        return NEGATIVE_INFINITY if negative else INFINITY
    if math.isinf(a) and math.isinf(b):
        return QUIET_NAN
    return bits(f32(a / b))


def exp_word(number: int) -> int:
    if is_nan(number):
        return number | QUIET
    try:
        return bits(f32(math.exp(value(number))))
    except OverflowError:
        return INFINITY


def log_word(number: int) -> int:
    if is_nan(number):
        return number | QUIET
    x = value(number)
    if x == 0:
        return NEGATIVE_INFINITY
    if x < 0:
        return QUIET_NAN
    if math.isinf(x):
        return INFINITY
    return bits(f32(math.log(x)))


def count_word(count: int) -> int:
    return word(float(count))


def greater(left: int, right: int) -> bool:
    return value(left) > value(right)


# PyTorch's maximum: NaN when either word is NaN, else the larger, the later
# word on a tie.
def max_word(held: int, number: int) -> int:
    if is_nan(held) or (not is_nan(number) and greater(held, number)):
        return held
    return number


def mse_forward(left: tuple, right: tuple) -> tuple | int:
    d = elementwise(sub_word, left, right)
    if isinstance(d, int):
        return d
    e = accepted(elementwise(sub_word, right, left))
    squares = [mul_word(number, number) for number in d[1]]
    loss = div_word(total(squares), count_word(len(squares)))
    return ([], [loss]), e, d


def entropy_forward(logits: tuple, targets: list[int]) -> tuple | int:
    shape, words = logits
    if len(shape) != 2:
        return NOT_LOGITS
    rows, columns = shape
    if len(targets) != rows or any(target >= columns for target in targets):
        return TARGET_OUT_OF_RANGE
    lp, onehot, picked = [], [], []
    for row in range(rows):
        line = words[row * columns : (row + 1) * columns]
        held = line[0]
        for number in line[1:]:
            held = max_word(held, number)
        shifted = [sub_word(number, held) for number in line]
        lse = log_word(total([exp_word(number) for number in shifted]))
        lp += [sub_word(number, lse) for number in shifted]
        onehot += [ONE if column == targets[row] else 0 for column in range(columns)]
    picked = [neg_word(a) if b == ONE else 0 for a, b in zip(lp, onehot, strict=True)]
    loss = div_word(total(picked), count_word(rows))
    return ([], [loss]), (shape, lp), (shape, onehot)


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


# reduce_to: each target cell adds its fiber in flat order, from +0.0, even
# when the shapes agree.
def reduce_to(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    totals = [0] * math.prod(target)
    for flat in range(math.prod(shape)):
        position = source_index(target, shape, flat)
        totals[position] = add_word(totals[position], words[flat])
    return (target, totals)


# sum_to_size: the words themselves when the shapes agree, otherwise the
# ordered fold of reduce_to.
def sum_to_size(tensor: tuple, target: list[int]) -> tuple:
    if tensor[0] == target:
        return (target, list(tensor[1]))
    return reduce_to(tensor, target)


# The kept and dropped dimensions of a sum over `axes`, or the code of its
# refusal: the signed axes wrapped among the rank (1 for a rank-0 input),
# then an axis out of range first and an axis listed twice next, as PyTorch
# checks them. No axes reduce every axis.
def sum_dims_plan(shape: list[int], axes: list[int]) -> tuple[list, list] | int:
    limit = max(len(shape), 1)
    axes = [wrap(axis, limit) for axis in axes]
    if any(axis >= limit for axis in axes):
        return AXIS_OUT_OF_BOUNDS
    if len(set(axes)) != len(axes):
        return DUPLICATE_AXIS
    reduced = [not axes or axis in axes for axis in range(len(shape))]
    kept = [1 if gone else extent for gone, extent in zip(reduced, shape, strict=True)]
    dropped = [extent for gone, extent in zip(reduced, shape, strict=True) if not gone]
    return kept, dropped


# reduce_to onto the kept shape, then, without keepdim, the same words with
# the dropped dimensions.
def sum_dims(tensor: tuple, step: dict) -> tuple | int:
    plan = sum_dims_plan(tensor[0], step["dims"])
    if isinstance(plan, int):
        return plan
    kept, dropped = plan
    summed = reduce_to(tensor, kept)
    return summed if step["keepdim"] else (dropped, summed[1])


def broadcast_to(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    return (
        target,
        [words[source_index(shape, target, flat)] for flat in range(math.prod(target))],
    )


# A slice's axis, its indices along the axis, its shape with the axis kept
# and the shape it returns, or the code of its refusal. The axis is wrapped
# among the rank, and one outside -rank..rank-1 (every axis of a rank-0 input)
# is refused first, then a step of 0, or a narrow or select out of range. A
# slice's start and stop count from the end when negative and are clamped to
# the axis as PyTorch clamps them; a narrow takes a start in -extent..extent
# and a select an index in -extent..extent-1.
def slice_plan(shape: list[int], step: dict) -> tuple[int, range, list, list] | int:
    dim = wrap(step["dim"], len(shape))
    if dim >= len(shape):
        return AXIS_OUT_OF_BOUNDS
    extent = shape[dim]
    if step["op"] == "slice":
        start, stop = (
            max(0, p + extent) if p < 0 else p for p in (step["start"], step["stop"])
        )
        stride = step["step"]
        if stride == 0:
            return STEP_NOT_POSITIVE
    elif step["op"] == "narrow":
        if step["start"] < -extent:
            return OUT_OF_RANGE
        start = wrap(step["start"], extent)
        stop, stride = start + step["length"], 1
        if stop > extent:
            return OUT_OF_RANGE
    else:
        start = wrap(step["index"], extent)
        stop, stride = start + 1, 1
        if start >= extent:
            return OUT_OF_RANGE
    low = min(start, extent)
    indices = range(low, max(low, min(stop, extent)), stride)
    kept = [*shape[:dim], len(indices), *shape[dim + 1 :]]
    output = [*kept[:dim], *kept[dim + 1 :]] if step["op"] == "select" else kept
    return dim, indices, kept, output


# Output cell (o, k, r) of [outer, len(indices), inner] and the input cell
# (o, indices[k], r) of [outer, extent, inner] it copies, as flat indices. An
# input with no element has none, whatever the other extents.
def slice_cells(shape: list[int], dim: int, indices: range) -> list[tuple]:
    if math.prod(shape) == 0:
        return []
    outer, extent = math.prod(shape[:dim]), shape[dim]
    inner = math.prod(shape[dim + 1 :])
    return [
        ((o * len(indices) + k) * inner + r, (o * extent + i) * inner + r)
        for o in range(outer)
        for k, i in enumerate(indices)
        for r in range(inner)
    ]


# An index_select's or a gather's output shape and, for each output cell, the
# input cell it copies, as flat indices; or the code of its refusal, in
# order: a gather index whose dimensions no tensor has (past rank 4 or 2^20
# elements), a gather's entry count, the axis, a gather index's rank and its
# extents at the other axes, then an entry at or past the extent. Output cell
# (o, k, r) of [outer, length, inner] reads input cell (o, j, r) of [outer,
# extent, inner], with j entry k of an index_select's index, and the entry
# at the output cell itself of a gather's.
def gather_plan(shape: list[int], step: dict) -> tuple[list, list] | int:
    entries = step["index"]
    gather = step["op"] == "gather"
    if gather and len(step["index_shape"]) > MAX_RANK:
        return RANK_EXCEEDED
    if gather and math.prod(step["index_shape"]) > MAX_ELEMENTS:
        return CAPACITY_EXCEEDED
    if gather and len(entries) != math.prod(step["index_shape"]):
        return VALUE_COUNT_MISMATCH
    dim = wrap(step["dim"], len(shape))
    if dim >= len(shape):
        return AXIS_OUT_OF_BOUNDS
    if gather:
        index_shape = step["index_shape"]
        if len(index_shape) != len(shape):
            return INDEX_RANK_MISMATCH
        if any(index_shape[d] != shape[d] for d in range(len(shape)) if d != dim):
            return INDEX_SHAPE_MISMATCH
    extent = shape[dim]
    if any(entry >= extent for entry in entries):
        return OUT_OF_RANGE
    length = step["index_shape"][dim] if gather else len(entries)
    output = [*shape[:dim], length, *shape[dim + 1 :]]
    if math.prod(output) > MAX_ELEMENTS:
        return CAPACITY_EXCEEDED
    outer, inner = math.prod(shape[:dim]), math.prod(shape[dim + 1 :])
    pairs = []
    for o in range(outer):
        for k in range(length):
            for r in range(inner):
                at = (o * length + k) * inner + r
                j = entries[at] if gather else entries[k]
                pairs.append((at, (o * extent + j) * inner + r))
    return output, pairs


def gather_forward(tensor: tuple, step: dict) -> tuple | int:
    plan = gather_plan(tensor[0], step)
    if isinstance(plan, int):
        return plan
    output, pairs = plan
    words = [0] * math.prod(output)
    for target, source in pairs:
        words[target] = tensor[1][source]
    return (output, words)


def slice_forward(tensor: tuple, step: dict) -> tuple | int:
    plan = slice_plan(tensor[0], step)
    if isinstance(plan, int):
        return plan
    dim, indices, kept, output = plan
    words = [0] * math.prod(kept)
    for target, source in slice_cells(tensor[0], dim, indices):
        words[target] = tensor[1][source]
    return (output, words)


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


# torch.matmul's plan: a 1-D left operand is the row [1, k] and a 1-D right
# one the column [k, 1], the output dropping that row or column; the
# dimensions before the last two are batch dimensions, which broadcast. The
# checks come in the order Batched.plan makes them: each operand's rank (1 to
# 4), the inner extents, the batch dimensions, the output's capacity. Two
# rank-2 operands are `matmul` itself.
def batched_plan(left: list[int], right: list[int]) -> tuple | int:
    if not 1 <= len(left) <= MAX_RANK or not 1 <= len(right) <= MAX_RANK:
        return MATMUL_RANK
    rows = [1, *left] if len(left) == 1 else left
    columns = [*right, 1] if len(right) == 1 else right
    if rows[-1] != columns[-2]:
        return MATMUL_INNER
    batch = broadcast(rows[:-2], columns[:-2])
    if batch is None:
        return INCOMPATIBLE_BROADCAST
    output = batch + rows[-2:-1] * (len(left) > 1) + columns[-1:] * (len(right) > 1)
    if math.prod(output) > MAX_ELEMENTS:
        return CAPACITY_EXCEEDED
    return rows, columns, batch, output


# The words of the matrix that batch `flat` of the output reads from an
# operand whose batch dimensions are `own`: a stretched axis reads index 0.
def batch_matrix(
    words: list, own: list[int], batch: list[int], flat: int, size: int
) -> list:
    start = source_index(own, batch, flat) * size
    return list(words[start : start + size])


# torch.matmul: each batch is `matmul` of its two matrices, in batch order.
def batched_matmul(left: tuple, right: tuple) -> tuple | int:
    if len(left[0]) == 2 and len(right[0]) == 2:
        return matmul(left, right)
    plan = batched_plan(left[0], right[0])
    if isinstance(plan, int):
        return plan
    rows, columns, batch, output = plan
    (m, k), n = rows[-2:], columns[-1]
    words = []
    for flat in range(math.prod(batch)):
        a = batch_matrix(left[1], rows[:-2], batch, flat, m * k)
        b = batch_matrix(right[1], columns[:-2], batch, flat, k * n)
        words += accepted(matmul(([m, k], a), ([k, n], b)))[1]
    return (output, words)


# The VJP of torch.matmul: g @ right^T and left^T @ g for each batch of the
# output, the left operand's product with its row dropped when it is 1-D and
# the right one's with its column dropped, each summed to its operand's shape.
# Each product is built over the output's batch before the sum, as
# Batched.left_gradient and right_gradient build it, so one with more than
# 2^20 elements is refused with CapacityExceeded, the left one first, even
# when both operands and the forward output fit.
def batched_vjp(node: dict, gradient: tuple) -> list[tuple[int, tuple]] | int:
    left, right = node["saved"]
    plan = batched_plan(left[0], right[0])
    if isinstance(plan, int):
        raise GateError(f"A recorded torch.matmul was refused with code {plan}")
    rows, columns, batch, _ = plan
    (m, k), n = rows[-2:], columns[-1]
    left_dims = batch + ([k] if len(left[0]) == 1 else [m, k])
    right_dims = batch + ([k] if len(right[0]) == 1 else [k, n])
    if max(math.prod(left_dims), math.prod(right_dims)) > MAX_ELEMENTS:
        return CAPACITY_EXCEEDED
    first: list = []
    second: list = []
    for flat in range(math.prod(batch)):
        a = batch_matrix(left[1], rows[:-2], batch, flat, m * k)
        b = batch_matrix(right[1], columns[:-2], batch, flat, k * n)
        g = list(gradient[1][flat * m * n : (flat + 1) * m * n])
        first += accepted(matmul(([m, n], g), transpose(([k, n], b))))[1]
        second += accepted(matmul(transpose(([m, k], a)), ([m, n], g)))[1]
    return [
        (node["a"], sum_to_size((left_dims, first), node["left"])),
        (node["b"], sum_to_size((right_dims, second), node["right"])),
    ]


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


def vjp(node: dict, gradient: tuple) -> list[tuple[int, tuple]] | int:
    name = node["op"]
    if name in ("leaf", "constant"):
        return []
    if name == "mse_loss":
        e, d = node["saved"]
        scale = div_word(word(2.0), count_word(len(d[1])))

        def scaled(g: int, number: int) -> int:
            return mul_word(mul_word(scale, number), g)

        first = accepted(elementwise(scaled, gradient, d))
        second = accepted(elementwise(scaled, gradient, e))
        return [
            (node["a"], sum_to_size(first, node["left"])),
            (node["b"], sum_to_size(second, node["right"])),
        ]
    if name == "cross_entropy":
        lp, onehot = node["saved"]
        count = count_word(node["left"][0])

        def target(g: int, mark: int) -> int:
            return neg_word(div_word(g, count)) if mark == ONE else 0

        # PyTorch's row sum of the one-hot gradient: the target word alone
        # in a row of one column (-0.0 kept), the target word + 0.0 otherwise.
        single = node["left"][1] == 1

        def rest(g: int, number: int) -> int:
            target = neg_word(div_word(g, count))
            row_sum = target if single else add_word(target, 0)
            return neg_word(mul_word(exp_word(number), row_sum))

        first = accepted(elementwise(target, gradient, onehot))
        second = accepted(elementwise(rest, gradient, lp))
        return [
            (node["a"], sum_to_size(first, node["left"])),
            (node["a"], sum_to_size(second, node["left"])),
        ]
    if name == "sum":
        return [(node["a"], broadcast_to(gradient, node["left"]))]
    if name in SLICES:
        # slice_backward: the gradient words at the cells the slice read,
        # +0.0 everywhere else.
        words = [0] * math.prod(node["left"])
        for target, source in slice_cells(node["left"], node["dim"], node["indices"]):
            words[source] = gradient[1][target]
        return [(node["a"], (node["left"], words))]
    if name in GATHERS:
        # index_add and scatter_add: each gradient word added, in output
        # order, to +0.0 at the input cell it came from.
        words = [0] * math.prod(node["left"])
        for target, source in node["pairs"]:
            words[source] = add_word(words[source], gradient[1][target])
        return [(node["a"], (node["left"], words))]
    if name == "sum_dims":
        unsqueezed = (node["kept"], gradient[1])
        return [(node["a"], broadcast_to(unsqueezed, node["left"]))]
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
    if name == "batched_matmul":
        return batched_vjp(node, gradient)
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


# The working gradients of a sweep from output, or the error code of the
# first VJP that the tensor layer refuses: then the sweep delivers nothing,
# and every slot stays as it was.
def gradients(nodes: list[dict], output: int) -> list[tuple | None] | int:
    found: list[tuple | None] = [None] * len(nodes)
    shape = nodes[output]["shape"]
    found[output] = (shape, [ONE] * math.prod(shape))
    for position in range(output, -1, -1):
        gradient = found[position]
        if gradient is None:
            continue
        frozen = nodes[position].get("frozen", ())
        contributions = vjp(nodes[position], gradient)
        if isinstance(contributions, int):
            return contributions
        for target, contribution in contributions:
            if target < position and target not in frozen:
                found[target] = accumulate(found[target], contribution)
    return found


def snapshot(tensor: tuple) -> list[str]:
    shape, words = tensor
    return [
        f"tensor:shape:{json.dumps(shape)}",
        f"tensor:count:{len(words)}",
        *[f"tensor:word:{number}" for number in words],
    ]


# One cell of an SGD step: p - lr * g, the product rounded, then the
# difference.
def descend_word(lr: int, p: int, g: int) -> int:
    return sub_word(p, mul_word(lr, g))


def operands(node: dict) -> list[int]:
    return [node[key] for key in ("a", "b") if key in node]


def forward(name: str, values: list[tuple], step: dict) -> tuple | int:
    if name == "mse_loss":
        return mse_forward(*values)
    if name == "cross_entropy":
        return entropy_forward(values[0], step["targets"])
    if name == "sum":
        return ([], [total(values[0][1])])
    if name in SLICES:
        return slice_forward(values[0], step)
    if name in GATHERS:
        return gather_forward(values[0], step)
    if name == "sum_dims":
        return sum_dims(values[0], step)
    if name == "relu":
        return (values[0][0], [relu_word(number) for number in values[0][1]])
    if name == "matmul":
        return matmul(*values)
    if name == "batched_matmul":
        return batched_matmul(*values)
    if name == "matmul_nt":
        left, right = values
        if len(left[0]) != 2 or len(right[0]) != 2:
            return MATMUL_RANK
        return matmul(left, transpose(right))
    combine = {"add": add_word, "sub": sub_word, "mul": mul_word}[name]
    return elementwise(combine, *values)


# The reference's lines for one script. With a trace, it also records each
# node's op and operands ("nodes"), the backward outputs ("outputs"), and, for
# each read, where its value and gradient words start in the lines ("reads",
# keyed by the read's epoch and the node id; an epoch is a run of reads with
# no other step between them).
def reference(case: dict, trace: dict | None = None, update=descend_word) -> list[str]:
    nodes: list[dict] = []
    slots: list[tuple | None] = []
    lines = []
    epoch = 0
    reads: dict[tuple[int, int], tuple[int, int, int | None]] = {}
    outputs: set[int] = set()

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
        result = forward(name, values, step)
        if isinstance(result, int):
            lines.append(f"autograd:error:{result}")
            return False
        saved = None
        if name in ("mse_loss", "cross_entropy"):
            result, *saved = result
        node = {"op": name, "a": step["a"], "shape": result[0], "value": result}
        # Operands that are constants get nothing; with every operand a
        # constant, the result is a constant too.
        node["frozen"] = [i for i in ids if nodes[i]["op"] == "constant"]
        if len(node["frozen"]) == len(ids):
            node["op"] = "constant"
        node["left"] = values[0][0]
        if name not in UNARY and name not in SLICES and name not in GATHERS:
            node |= {"b": step["b"], "right": values[1][0]}
        if name in GATHERS:
            plan = gather_plan(values[0][0], step)
            if isinstance(plan, int):
                raise GateError(f"A recorded gather refuses its own input: {plan}")
            node["pairs"] = plan[1]
        if name in SLICES:
            plan = slice_plan(values[0][0], step)
            if isinstance(plan, int):
                raise GateError(f"A recorded slice refuses its own input: {plan}")
            node |= {"dim": plan[0], "indices": plan[1]}
        if name in ("mul", "matmul", "matmul_nt", "relu", *PRODUCTS):
            node["saved"] = values
        if name == "sum_dims":
            plan = sum_dims_plan(values[0][0], step["dims"])
            node["kept"] = plan[0] if isinstance(plan, tuple) else None
        if saved is not None:
            node["saved"] = saved
        push(node, shown)
        return True

    for step in case["steps"]:
        name = step["op"]
        if name != "read":
            epoch += 1
        if name in ("leaf", "constant"):
            shape, words = step["shape"], step["words"]
            if len(shape) > MAX_RANK:
                lines.append(f"autograd:error:{RANK_EXCEEDED}")
            elif math.prod(shape) > MAX_ELEMENTS:
                lines.append(f"autograd:error:{CAPACITY_EXCEEDED}")
            elif len(words) != math.prod(shape):
                lines.append(f"autograd:error:{VALUE_COUNT_MISMATCH}")
            else:
                push({"op": name, "shape": shape, "value": (shape, list(words))})
        elif name in (*BINARY, *UNARY, *PRODUCTS, *SLICES, *GATHERS):
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
            outputs.add(output)
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
                if isinstance(found, int):
                    lines.append(f"autograd:error:{found}")
                    continue
                slots = [accumulate(a, b) for a, b in zip(slots, found, strict=True)]
                lines.append("autograd:backward")
        elif name == "zero_grad":
            slots = [None] * len(nodes)
            lines.append("autograd:zero_grad")
        elif name == "step":
            # Every id is checked, in list order, before any update.
            refused = next(
                (
                    UNKNOWN_NODE if target >= len(nodes) else NOT_LEAF
                    for target in step["params"]
                    if target >= len(nodes)
                    or nodes[target]["op"] not in ("leaf", "constant")
                ),
                None,
            )
            # Then the learning rate, as PyTorch's `lr < 0.0`, a NaN compares
            # false.
            (rate,) = struct.unpack("<f", struct.pack("<I", step["lr"]))
            if refused is None and rate < 0.0:
                refused = INVALID_LEARNING_RATE
            if refused is not None:
                lines.append(f"autograd:error:{refused}")
                continue
            for target in step["params"]:
                slot = slots[target]
                if slot is None:
                    continue
                shape, words = nodes[target]["value"]
                changed = [
                    update(step["lr"], p, g)
                    for p, g in zip(words, slot[1], strict=True)
                ]
                nodes[target]["value"] = (shape, changed)
            lines.append("autograd:step")
        else:
            identifier = step["id"]
            if identifier >= len(nodes):
                lines += [
                    f"autograd:value:error:{UNKNOWN_NODE}",
                    f"autograd:grad:error:{UNKNOWN_NODE}",
                ]
                continue
            slot = slots[identifier]
            value_start = len(lines) + 3
            lines += ["autograd:value", *snapshot(nodes[identifier]["value"])]
            grad_start = None if slot is None else len(lines) + 3
            lines += (
                ["autograd:grad:none"]
                if slot is None
                else ["autograd:grad", *snapshot(slot)]
            )
            count = len(nodes[identifier]["value"][1])
            reads[(epoch, identifier)] = (value_start, count, grad_start)
    if trace is not None:
        trace["nodes"] = [(node["op"], operands(node)) for node in nodes]
        trace["outputs"] = outputs
        trace["reads"] = reads
    return lines


def words_at(lines: list[str], start: int, count: int) -> list[int]:
    return [
        int(line.removeprefix("tensor:word:")) for line in lines[start : start + count]
    ]


# relu moves words and never computes one, so on the native lane its words
# must be exact, NaN sign, payload and signaling bit included, though the
# reference allows any NaN for any NaN elsewhere. For each read of a relu
# node r of input a in one epoch: r's value is relu of a's value, word for
# word; and when a feeds only r and no backward starts at a, a's gradient is
# relu_backward of a's value and r's gradient, both as this lane printed them,
# so every passed word is g's own word. Both slots accumulate in the same
# order, so the sums agree too.
def relu_exact(case: dict, seen: list[str], lane: str) -> int:
    trace: dict = {}
    reference(case, trace)
    nodes, reads, outputs = trace["nodes"], trace["reads"], trace["outputs"]
    consumers = [0] * len(nodes)
    for _, ids in nodes:
        for identifier in ids:
            if identifier < len(nodes):
                consumers[identifier] += 1
    checked = 0
    for (epoch, r), (value_r, count, grad_r) in reads.items():
        op_name, ids = nodes[r]
        if op_name != "relu" or (epoch, ids[0]) not in reads:
            continue
        a = ids[0]
        value_a, _, grad_a = reads[(epoch, a)]
        xs = words_at(seen, value_a, count)
        if words_at(seen, value_r, count) != [relu_word(x) for x in xs]:
            raise GateError(f"{lane} {case['id']}: relu of node {a} is not exact")
        checked += count
        if consumers[a] != 1 or a in outputs:
            continue
        if (grad_r is None) != (grad_a is None):
            raise GateError(f"{lane} {case['id']}: node {a} and relu disagree on None")
        if grad_r is None or grad_a is None:
            continue
        gs = words_at(seen, grad_r, count)
        if words_at(seen, grad_a, count) != [
            relu_backward_word(x, g) for x, g in zip(xs, gs, strict=True)
        ]:
            raise GateError(
                f"{lane} {case['id']}: relu gradient of node {a} is not exact"
            )
        checked += count
    return checked


def list_source(numbers: list[int]) -> str:
    return "[" + ", ".join(str(number) for number in numbers) + "]"


def step_source(step: dict) -> str:
    name = step["op"]
    if name in ("leaf", "constant"):
        # A large leaf of tools/autograd_parallel.py computes its words with
        # the `spread` its program defines instead of listing them.
        words = (
            f"spread({len(step['words'])}n, {step['spread']}n)"
            if "spread" in step
            else list_source(step["words"])
        )
        tensor = f"{list_source(step['shape'])}, {words}"
        return f"Harness.{name.capitalize()}{{{tensor}}}"
    if name == "backward":
        return f"Harness.Run{{{step['output']}n}}"
    if name == "zero_grad":
        return "Harness.ZeroGrad{}"
    if name == "step":
        params = "[" + ", ".join(f"{target}n" for target in step["params"]) + "]"
        return f"Harness.Step{{{step['lr']}, {params}}}"
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
    if name == "cross_entropy":
        return f"Harness.CrossEntropy{{{step['a']}n, {list_source(step['targets'])}}}"
    if name == "slice":
        fields = ", ".join(
            position_source(step[key]) for key in ("dim", "start", "stop")
        )
        return f"Harness.Slice{{{step['a']}n, {fields}, {step['step']}}}"
    if name == "narrow":
        fields = ", ".join(position_source(step[key]) for key in ("dim", "start"))
        return f"Harness.Narrow{{{step['a']}n, {fields}, {step['length']}}}"
    if name == "select":
        fields = ", ".join(position_source(step[key]) for key in ("dim", "index"))
        return f"Harness.Select{{{step['a']}n, {fields}}}"
    if name == "index_select":
        dim, entries = position_source(step["dim"]), list_source(step["index"])
        return f"Harness.IndexSelect{{{step['a']}n, {dim}, {entries}}}"
    if name == "gather":
        dim, shape = position_source(step["dim"]), list_source(step["index_shape"])
        # A large index of tools/autograd_parallel.py computes its entries
        # with the `cycle` its program defines.
        entries = (
            f"cycle({len(step['index'])}n, {step['cycle'][0]}n, {step['cycle'][1]}n)"
            if "cycle" in step
            else list_source(step["index"])
        )
        return f"Harness.Gather{{{step['a']}n, {dim}, {shape}, {entries}}}"
    if name == "sum_dims":
        keepdim = "True{}" if step["keepdim"] else "False{}"
        axes = positions_source(step["dims"])
        return f"Harness.SumDims{{{step['a']}n, {axes}, {keepdim}}}"
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
        ("src/tensor/signed.bend", "Signed"),
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


def verify(
    selected: list[dict], observations: list[dict], lane: str, native: bool = False
) -> dict:
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
        if native:
            relu_exact(case, seen, lane)
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
        return observations, verify(selected, observations, lane, lane == "native")

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
    for name, (path, original, mutated) in NATIVE_CONTROLS.items():
        copy = output / name
        shutil.copytree(ROOT / "src", copy / "src")
        target = copy / path
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: native control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        source = copy / "tensor-backward.bend"
        source.write_text(program(selected, copy, copy))
        emitted = copy / "tensor-backward.c"
        executable = copy / "tensor-backward"
        evidence.run(f"{name}-c-export", bend(tool_root, source, "-o", emitted))
        evidence.run(
            f"{name}-native-build",
            [
                str(tool_root / "llvm/usr/bin/clang-19"),
                "-std=c11",
                "-O3",
                str(emitted),
                "-lpthread",
                "-lm",
                "-o",
                str(executable),
            ],
        )
        result = evidence.run(f"{name}-native-run", native(executable), timeout=1800)
        observations = parse(result.stdout, [c["id"] for c in selected])
        try:
            verify(selected, observations, name, native=True)
        except GateError as error:
            # Only the exact relu words may reject it: the reference allows
            # the canonical NaN.
            if "is not exact" not in str(error):
                raise GateError(
                    f"{name}: rejected for another reason: {error}"
                ) from error
            rejected.append(name)
            continue
        raise GateError(
            f"{name}: the native tensor backward lane accepted a broken engine"
        )
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
