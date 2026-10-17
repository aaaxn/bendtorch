import json
import shutil
from functools import partial
from pathlib import Path

from tools.autograd_backward import run_gate as backward
from tools.autograd_broadcast import run_gate as broadcast
from tools.autograd_gradient import run_gate as gradient
from tools.autograd_lifecycle import run_gate as lifecycle
from tools.autograd_reduce import run_gate as reduce
from tools.autograd_sum_to_size import run_gate as sum_to_size
from tools.autograd_tensor_backward import CHAIN_CONTROLS, SGD_CONTROLS
from tools.autograd_tensor_backward import run_gate as tensor_backward
from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_laws,
    check_sources,
    digest,
    parallel,
    required_exports,
)

SUBGATES = {
    "broadcast": broadcast,
    "reduce": reduce,
    "sum_to_size": sum_to_size,
    "gradient": gradient,
    "lifecycle": lifecycle,
    "backward": backward,
    "tensor_backward": tensor_backward,
}

PROOF_CONTROLS = {
    "keep-left-returns-right": (
        "src/tensor/elementwise.bend",
        "    case KeepLeft{}:\n      left",
        "    case KeepLeft{}:\n      right",
    ),
    "broadcast-returns-zero-owner": (
        "src/autograd/broadcast.bend",
        "    case Done{output}:\n      (input, Done{output})",
        "    case Done{output}:\n      (zeros, Done{output})",
    ),
    "broadcast-skips-target-check": (
        "src/autograd/broadcast.bend",
        "      fitted(input, target, plan, Layout.words_same("
        "Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target)))",
        "      fitted(input, target, plan, True{})",
    ),
    "reduce-overwrites-instead-of-adding": (
        "src/autograd/reduce.bend",
        "(F32.add(accumulator, word), kept)",
        "(word, kept)",
    ),
    "reduce-reverses-addition-order": (
        "src/autograd/reduce.bend",
        "(F32.add(accumulator, word), kept)",
        "(F32.add(word, accumulator), kept)",
    ),
    "reduce-starts-from-one": (
        "src/autograd/reduce.bend",
        "first_digit(rows, target), 0.0, array)",
        "first_digit(rows, target), 1.0, array)",
    ),
    "reduce-walks-target-stride": (
        "src/autograd/reduce.bend",
        "Nat.add(base, Nat.mul(right, digit))",
        "Nat.add(base, Nat.mul(left, digit))",
    ),
    "reduce-skips-target-check": (
        "src/autograd/reduce.bend",
        "      fitted(Core.Packed{shape, tensor}, target, plan,\n"
        "        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), "
        "Shape.dimensions(shape)))",
        "      fitted(Core.Packed{shape, tensor}, target, plan, True{})",
    ),
    "sum-to-size-identity-folds": (
        "src/autograd/sum-to-size.bend",
        "    case True{}:\n      cloned(shape, Core.clone(shape, tensor))",
        "    case True{}:\n      Reduce.reduce_to(Core.Packed{shape, tensor}, target)",
    ),
    "sum-to-size-skips-dimension-check": (
        "src/autograd/sum-to-size.bend",
        "  decided(shape, tensor, target, Layout.words_same("
        "Shape.dimensions(shape), Shape.dimensions(target)))",
        "  decided(shape, tensor, target, True{})",
    ),
    "sum-to-size-always-folds": (
        "src/autograd/sum-to-size.bend",
        "  decided(shape, tensor, target, Layout.words_same("
        "Shape.dimensions(shape), Shape.dimensions(target)))",
        "  decided(shape, tensor, target, False{})",
    ),
    "sum-dims-keepdim-drops-axes": (
        "src/autograd/sum-dims.bend",
        "    case True{}:\n      Done{output}",
        "    case True{}:\n"
        "      transformed(Transforms.reshape_dynamic(output, dropped))",
    ),
    "sum-dims-keeps-reduced-axes": (
        "src/autograd/sum-dims.bend",
        "    case False{}:\n"
        "      transformed(Transforms.reshape_dynamic(output, dropped))",
        "    case False{}:\n      Done{output}",
    ),
    "sum-dims-counts-axes-from-one": (
        "src/autograd/sum-dims.bend",
        "  kept_from(axes, dimensions, 0)",
        "  kept_from(axes, dimensions, 1)",
    ),
    "sum-dims-marks-next-axis": (
        "src/autograd/sum-dims.bend",
        "Bool.pick(U32, reduces(axes, axis), 1, extent)",
        "Bool.pick(U32, reduces(axes, U32.add(axis, 1)), 1, extent)",
    ),
    "sum-dims-drops-kept-axes": (
        "src/autograd/sum-dims.bend",
        "    case True{}:\n      rest\n    case False{}:\n      extent <> rest",
        "    case True{}:\n      extent <> rest\n    case False{}:\n      rest",
    ),
    "sum-dims-empty-reduces-nothing": (
        "src/autograd/sum-dims.bend",
        "    case Nil{}:\n      True{}\n    case +head <> +tail:",
        "    case Nil{}:\n      False{}\n    case +head <> +tail:",
    ),
    "sum-dims-rank-zero-has-no-axis": (
        "src/autograd/sum-dims.bend",
        "    case Nil{}:\n      1\n",
        "    case Nil{}:\n      0\n",
    ),
    "tensor-backward-sum-dims-vjp-skips-unsqueeze": (
        "src/autograd/tensor-backward.bend",
        "restored(shape, vjp_sum(a, input, "
        "Transforms.reshape_equal(shape, kept, tensor, "
        "Transforms.elements_equal(shape, kept, equal))))",
        "restored(shape, vjp_sum(a, input, Core.Packed{shape, tensor}))",
    ),
    "tensor-backward-sum-dims-wraps-among-rank": (
        "src/autograd/tensor-backward.bend",
        "Signed.wrap_all(SumDims.limit(Shape.dimensions(shape)), axes)",
        "Signed.wrap_all(Shape.rank(shape), axes)",
    ),
    "gradient-overwrites-instead-of-adding": (
        "src/autograd/gradient.bend",
        "F32.add(accumulated, value)",
        "value",
    ),
    "gradient-reverses-addition-order": (
        "src/autograd/gradient.bend",
        "F32.add(accumulated, value)",
        "F32.add(value, accumulated)",
    ),
    "gradient-drops-first-contribution": (
        "src/autograd/gradient.bend",
        "    case Unset{}:\n      Accumulated{Held{contribution}}",
        "    case Unset{}:\n      Accumulated{Unset{}}",
    ),
    "gradient-skips-contribution-check": (
        "src/autograd/gradient.bend",
        "  checked(shape, slot, contribution_shape, tensor,\n"
        "    Layout.words_same(Shape.dimensions(contribution_shape), "
        "Shape.dimensions(shape)))",
        "  checked(shape, slot, contribution_shape, tensor, True{})",
    ),
    "gradient-skips-slot-check": (
        "src/autograd/gradient.bend",
        "  held_checked(contribution, total_shape, total, Layout.words_same("
        "Shape.dimensions(total_shape), Shape.dimensions(shape)))",
        "  held_checked(contribution, total_shape, total, True{})",
    ),
    "gradient-unset-reads-ones": (
        "src/autograd/gradient.bend",
        "Core.construct_values(shape, Core.new(shape, 0.0))",
        "Core.construct_values(shape, Core.new(shape, 1.0))",
    ),
    "grad-held-reads-none": (
        "src/autograd/gradient.bend",
        "    case Held{gradient}:\n      Some{gradient}",
        "    case Held{gradient}:\n      None{}",
    ),
    "reset-keeps-held": (
        "src/autograd/gradient.bend",
        "def reset(slot: Slot) -> Slot:\n  Unset{}",
        "def reset(slot: Slot) -> Slot:\n  slot",
    ),
    "backward-seeds-newest-node": (
        "src/autograd/backward.bend",
        "unit => pass(rest, below, output), Nat.is_eq(below, output))",
        "unit => pass(rest, below, output), True{})",
    ),
    "backward-lookup-aliases": (
        "src/autograd/backward.bend",
        "picked(A, head, unit => find(A, rest, below, id), Nat.is_eq(below, id))",
        "picked(A, head, unit => find(A, rest, below, id), Nat.is_le(below, id))",
    ),
    "backward-bump-aliases": (
        "src/autograd/backward.bend",
        "unit => bump(rest, below, target, contribution), Nat.is_eq(below, target))",
        "unit => bump(rest, below, target, contribution), Nat.is_le(below, target))",
    ),
    "backward-drops-later-contribution": (
        "src/autograd/backward.bend",
        "      Some{F32.add(held, value)}",
        "      Some{held}",
    ),
    "backward-mul-vjp-swaps-operands": (
        "src/autograd/backward.bend",
        "[Contribution{a, F32.mul(g, right)}, Contribution{b, F32.mul(g, left)}]",
        "[Contribution{a, F32.mul(g, left)}, Contribution{b, F32.mul(g, right)}]",
    ),
    "backward-sub-vjp-drops-sign": (
        "src/autograd/backward.bend",
        "[Contribution{a, g}, Contribution{b, F32.neg(g)}]",
        "[Contribution{a, g}, Contribution{b, g}]",
    ),
    "backward-overwrites-slot": (
        "src/autograd/backward.bend",
        "Entry{node, value, accumulate(slot, gradient)} <> merge(rest, more)",
        "Entry{node, value, gradient} <> merge(rest, more)",
    ),
    "backward-wiring-check-bypassed": (
        "src/autograd/backward.bend",
        "      Bool.and(operands_below(node, below), wired(rest, below))",
        "      wired(rest, below)",
    ),
    "backward-zero-grad-keeps-slot": (
        "src/autograd/backward.bend",
        "Entry{node, value, None{}} <> cleared(rest)",
        "Entry{node, value, slot} <> cleared(rest)",
    ),
    "negate-left-keeps-sign": (
        "src/tensor/elementwise.bend",
        "    case NegateLeft{}:\n      F32.neg(left)",
        "    case NegateLeft{}:\n      left",
    ),
    "tensor-backward-seeds-newest-node": (
        "src/autograd/tensor-backward.bend",
        "ran(pass(entries, count, output, at_head(count, output)))",
        "ran(pass(entries, count, output, True{}))",
    ),
    "tensor-backward-seeds-zeros": (
        "src/autograd/tensor-backward.bend",
        "seeded(node, shape, value, saved, slot, rest, below, Core.new(shape, 1.0))",
        "seeded(node, shape, value, saved, slot, rest, below, Core.new(shape, 0.0))",
    ),
    "tensor-backward-head-aliases": (
        "src/autograd/tensor-backward.bend",
        "    case 1n+below:\n      Nat.is_eq(below, id)",
        "    case 1n+below:\n      Nat.is_le(below, id)",
    ),
    "tensor-backward-bump-overwrites": (
        "src/autograd/tensor-backward.bend",
        "      placed(rest, Gradient.accumulate(shape, slot, gradient))",
        "      placed(rest, Gradient.Accumulated{Gradient.Held{gradient}})",
    ),
    "tensor-backward-merge-overwrites-slot": (
        "src/autograd/tensor-backward.bend",
        "merged_entry(node, shape, value, saved, Gradient.accumulate(shape, slot, g), "
        "merge(rest, more))",
        "merged_entry(node, shape, value, saved, "
        "Gradient.Accumulated{Gradient.Held{g}}, merge(rest, more))",
    ),
    "tensor-backward-unreached-skips-older": (
        "src/autograd/tensor-backward.bend",
        "finished(Entry{node, shape, value, saved, slot}, Gradient.Unset{}, "
        "rec(Done{pending}))",
        "finished(Entry{node, shape, value, saved, slot}, Gradient.Unset{}, "
        "rec(Done{Nil{}}))",
    ),
    "tensor-backward-sum-vjp-wrong-target": (
        "src/autograd/tensor-backward.bend",
        "Done{[Contribution{a, input, expanded}]}",
        "Done{[Contribution{0n, input, expanded}]}",
    ),
    "tensor-backward-wiring-check-bypassed": (
        "src/autograd/tensor-backward.bend",
        "      Bool.and(operands_below(node, below), wired(rest, below))",
        "      wired(rest, below)",
    ),
    "tensor-backward-zero-grad-keeps-slot": (
        "src/autograd/tensor-backward.bend",
        "Entry{node, shape, value, saved, Gradient.reset(slot)} <> cleared(rest)",
        "Entry{node, shape, value, saved, slot} <> cleared(rest)",
    ),
}


ERROR_CONTROLS = {
    "broadcast-shape-error-renamed": (
        "src/autograd/broadcast.bend",
        "    case Fail{error}:\n"
        "      (input, Fail{TensorError{Core.ShapeError{error}}})\n"
        "    case Done{+plan}:\n      fitted(",
        "    case Fail{error}:\n"
        "      (input, Fail{NotBroadcastable{}})\n"
        "    case Done{+plan}:\n      fitted(",
        "src/autograd/PROOF.bend",
        "Laws.broadcast_shape_error",
    ),
    "broadcast-not-target-as-shape-error": (
        "src/autograd/broadcast.bend",
        "      (input, Fail{NotBroadcastable{}})",
        "      (input, Fail{TensorError{Core.ShapeError{"
        "Shape.IncompatibleBroadcast{}}}})",
        "src/autograd/PROOF.bend",
        "fitted_not_target",
    ),
    "reduce-shape-error-renamed": (
        "src/autograd/reduce.bend",
        "    case Fail{error}:\n"
        "      (Core.Packed{shape, tensor}, Fail{TensorError{Core.ShapeError{error}}})",
        "    case Fail{error}:\n"
        "      (Core.Packed{shape, tensor}, Fail{NotReducible{}})",
        "src/autograd/PROOF.bend",
        "Laws.reduce_shape_error",
    ),
    "reduce-not-reducible-as-shape-error": (
        "src/autograd/reduce.bend",
        "      (input, Fail{NotReducible{}})",
        "      (input, Fail{TensorError{Core.ShapeError{"
        "Shape.IncompatibleBroadcast{}}}})",
        "src/autograd/PROOF.bend",
        "fitted_not_reducible",
    ),
    "sum-dims-out-of-bounds-as-duplicate": (
        "src/autograd/sum-dims.bend",
        "    case False{} unique:\n      Some{Layout.AxisOutOfBounds{}}",
        "    case False{} unique:\n      Some{Layout.DuplicateAxis{}}",
        "src/autograd/PROOF.bend",
        "Laws.sum_dims_axis_out_of_bounds",
    ),
    "sum-dims-duplicate-as-out-of-bounds": (
        "src/autograd/sum-dims.bend",
        "    case True{} False{}:\n      Some{Layout.DuplicateAxis{}}",
        "    case True{} False{}:\n      Some{Layout.AxisOutOfBounds{}}",
        "src/autograd/PROOF.bend",
        "Laws.sum_dims_duplicate_axis",
    ),
    "sum-dims-refusal-as-reduce-error": (
        "src/autograd/sum-dims.bend",
        "      Fail{TransformError{Transforms.LayoutError{error}}}",
        "      Fail{ReduceError{Reduce.NotReducible{}}}",
        "src/autograd/PROOF.bend",
        "Laws.sum_dims_plan_refused",
    ),
    "matmul-inner-as-rank": (
        "src/tensor/matmul.bend",
        "      Fail{InnerDimensionMismatch{}}\n"
        "    case Shape.BTrue{equal}:\n      plan_output(",
        "      Fail{RankMismatch{}}\n    case Shape.BTrue{equal}:\n      plan_output(",
        "src/tensor/PROOF.bend",
        "compatible_inner",
    ),
    "matmul-right-rank-as-inner": (
        "src/tensor/matmul.bend",
        "    case rest:\n      Fail{RankMismatch{}}\n\ndef plan_left(",
        "    case rest:\n      Fail{InnerDimensionMismatch{}}\n\ndef plan_left(",
        "src/tensor/PROOF.bend",
        "plan_right_rank",
    ),
    "matmul-left-rank-as-inner": (
        "src/tensor/matmul.bend",
        "    case rest:\n      Fail{RankMismatch{}}\n\ndef plan(",
        "    case rest:\n      Fail{InnerDimensionMismatch{}}\n\ndef plan(",
        "src/tensor/PROOF.bend",
        "plan_left_rank",
    ),
    "flatten-rank-as-bounds": (
        "src/shape/core.bend",
        "      Fail{CoordinateRankMismatch{}}\n    case BTrue{equal}:\n"
        "      flatten_checked.fits(",
        "      Fail{CoordinateOutOfBounds{}}\n    case BTrue{equal}:\n"
        "      flatten_checked.fits(",
        "src/shape/PROOF.bend",
        "flatten_ranked_rank",
    ),
    "flatten-bounds-as-rank": (
        "src/shape/core.bend",
        "      Fail{CoordinateOutOfBounds{}}\n"
        "    case BTrue{equal}:\n      flat_checked(",
        "      Fail{CoordinateRankMismatch{}}\n"
        "    case BTrue{equal}:\n      flat_checked(",
        "src/shape/PROOF.bend",
        "flatten_fits_bounds",
    ),
    "gradient-wrong-shape-other-error": (
        "src/autograd/gradient.bend",
        "      Refused{slot, Core.Packed{contribution_shape, tensor}, ShapeMismatch{}}",
        "      Refused{slot, Core.Packed{contribution_shape, tensor}, "
        "TensorError{Core.StorageError{Storage.InvalidIndex{}}}}",
        "src/autograd/PROOF.bend",
        "Location: checked_refusal",
    ),
    "backward-unknown-as-dangling": (
        "src/autograd/backward.bend",
        "    case False{} sound:\n      Fail{UnknownNode{}}",
        "    case False{} sound:\n      Fail{DanglingOperand{}}",
        "src/autograd/PROOF.bend",
        "Laws.backward_unknown",
    ),
    "backward-dangling-as-unknown": (
        "src/autograd/backward.bend",
        "    case True{} False{}:\n      Fail{DanglingOperand{}}",
        "    case True{} False{}:\n      Fail{UnknownNode{}}",
        "src/autograd/PROOF.bend",
        "Laws.backward_dangling",
    ),
    "tensor-backward-unknown-as-dangling": (
        "src/autograd/tensor-backward.bend",
        "      (Tape{entries}, Fail{UnknownNode{}})\n    case True{} False{}:",
        "      (Tape{entries}, Fail{DanglingOperand{}})\n    case True{} False{}:",
        "src/autograd/PROOF.bend",
        "Laws.tensor_backward_unknown",
    ),
    "tensor-backward-dangling-as-unknown": (
        "src/autograd/tensor-backward.bend",
        "      (Tape{entries}, Fail{DanglingOperand{}})",
        "      (Tape{entries}, Fail{UnknownNode{}})",
        "src/autograd/PROOF.bend",
        "Laws.tensor_backward_dangling",
    ),
}


def proof_controls(evidence: Evidence, source: Path) -> None:
    parallel(
        *(
            partial(
                mutant_control,
                evidence,
                source,
                name,
                path,
                original,
                mutated,
                "src/autograd/PROOF.bend",
                ("SOME PROOFS FAIL",),
            )
            for name, (path, original, mutated) in PROOF_CONTROLS.items()
        ),
        *(
            partial(
                mutant_control,
                evidence,
                source,
                name,
                path,
                original,
                mutated,
                root,
                ("SOME PROOFS FAIL", location),
            )
            for name, (path, original, mutated, root, location) in (
                ERROR_CONTROLS.items()
            )
        ),
        *(
            partial(
                mutant_control,
                evidence,
                source,
                name,
                path,
                original,
                mutated,
                "src/autograd/PROOF.bend",
                ("SOME PROOFS FAIL", f"Location: {law}"),
            )
            for name, (path, original, mutated, law) in SGD_CONTROLS.items()
        ),
        *(
            partial(
                mutant_control,
                evidence,
                source,
                name,
                path,
                original,
                mutated,
                "src/autograd/tensor-chain-PROOF.bend",
                ("SOME PROOFS FAIL", location),
            )
            for name, (path, original, mutated, location) in CHAIN_CONTROLS.items()
        ),
    )


def mutant_control(
    evidence: Evidence,
    source: Path,
    name: str,
    path: str,
    original: str,
    mutated: str,
    root: str,
    contains: tuple[str, ...],
) -> None:
    copy = evidence.directory / "controls" / name
    shutil.copytree(source / "src", copy / "src")
    target = copy / path
    text = target.read_text()
    if text.count(original) != 1:
        raise GateError(f"{name}: control anchor is missing or ambiguous")
    target.write_text(text.replace(original, mutated))
    evidence.run(
        f"control-{name}",
        bend(evidence.tool_root, copy / root, "--check-only"),
        expected_code=1,
        contains=contains[0] if len(contains) == 1 else contains,
    )


def run_gate(tool_root: Path, output: Path) -> None:
    tool_root, output = tool_root.resolve(), output.resolve()
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    sources = [
        *sorted((ROOT / "src").rglob("*.bend")),
        *sorted((ROOT / "verification").rglob("*.bend")),
        ROOT / "LAWS.bend",
        ROOT / "PROOF.bend",
        ROOT / "verification/autograd.json",
        ROOT / "verification/references.json",
        *sorted((ROOT / "tools").glob("autograd*.py")),
        ROOT / "tools/common.py",
    ]
    frozen = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        frozen[str(relative)] = digest(target)
    (output / "source-inventory.json").write_text(json.dumps(frozen, indent=2) + "\n")
    names = list(SUBGATES)
    _, _, *results = parallel(
        partial(
            check_laws,
            evidence,
            "autograd",
            output / "source/src/autograd/PROOF.bend",
            required_exports(output / "source/verification/autograd.json"),
            timeout=1800,
        ),
        partial(proof_controls, evidence, output / "source"),
        *(partial(SUBGATES[name], tool_root, output / name) for name in names),
    )
    summary = dict(zip(names, results, strict=True))
    (output / "subgates.json").write_text(json.dumps(summary, indent=2) + "\n")
    changed = [
        path for path, expected in frozen.items() if digest(ROOT / path) != expected
    ]
    if changed:
        raise GateError(f"Autograd sources changed during the gate: {changed}")
    evidence.save()
    print(f"Autograd gate passed ({', '.join(SUBGATES)}): {output}")
