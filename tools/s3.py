import argparse
import json
import shutil
from pathlib import Path

from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_inventory,
    check_sources,
    digest,
)
from tools.s3_broadcast import run_stage as broadcast
from tools.s3_gradient import run_stage as gradient
from tools.s3_lifecycle import run_stage as lifecycle
from tools.s3_reduce import run_stage as reduce
from tools.s3_sum_to_size import run_stage as sum_to_size

SUBGATES = {
    "broadcast": broadcast,
    "reduce": reduce,
    "sum_to_size": sum_to_size,
    "gradient": gradient,
    "lifecycle": lifecycle,
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
        "F32.add(accumulated, value)",
        "value",
    ),
    "reduce-reverses-addition-order": (
        "src/autograd/reduce.bend",
        "F32.add(accumulated, value)",
        "F32.add(value, accumulated)",
    ),
    "reduce-starts-from-one": (
        "src/autograd/reduce.bend",
        "allocated(input, target, plan, Core.new(target, 0.0))",
        "allocated(input, target, plan, Core.new(target, 1.0))",
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
}


# Each mutant returns a different error for a request whose error identity a
# law states; the proofs must reject it at that law's proof.
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
    "matmul-inner-as-rank": (
        "src/tensor/matmul.bend",
        "      Fail{InnerDimensionMismatch{}}",
        "      Fail{RankMismatch{}}",
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
}


def proof_controls(evidence: Evidence, source: Path) -> None:
    for name, (path, original, mutated) in PROOF_CONTROLS.items():
        copy = evidence.directory / "controls" / name
        shutil.copytree(source / "src", copy / "src")
        target = copy / path
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"control-{name}",
            bend(evidence.tool_root, copy / "src/autograd/PROOF.bend", "--check-only"),
            expected_code=1,
            contains="SOME PROOFS FAIL",
        )
    for name, (path, original, mutated, root, location) in ERROR_CONTROLS.items():
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
            contains=("SOME PROOFS FAIL", location),
        )


def run_stage(tool_root: Path, output: Path) -> None:
    tool_root, output = tool_root.resolve(), output.resolve()
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    sources = [
        *sorted((ROOT / "src").rglob("*.bend")),
        *sorted((ROOT / "verification").rglob("*.bend")),
        ROOT / "LAWS.bend",
        ROOT / "PROOF.bend",
        ROOT / "verification/s3.json",
        ROOT / "verification/references.json",
        *sorted((ROOT / "tools").glob("s3*.py")),
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
    required = json.loads((output / "source/verification/s3.json").read_text())[
        "required_exports"
    ]
    root = output / "source/verification/s3/PROOF.bend"
    for name, mode in (("source", "--check-only"), ("kernel", "--verdict")):
        evidence.run(
            f"s3-{name}-check",
            bend(tool_root, root, mode),
            contains="ALL PROOFS CHECK",
            timeout=1800,
        )
    artifact = output / "PROOF.bendtt"
    evidence.run("s3-export", bend(tool_root, root, "-o", artifact))
    check_inventory(artifact.read_text(), required)
    evidence.run(
        "s3-direct-kernel",
        [str(tool_root / "kernel/bendtt"), str(artifact)],
        contains="ALL PROOFS CHECK",
        timeout=1800,
    )
    for name in required:
        missing = artifact.read_text().replace(name + " :", "omitted :", 1)
        try:
            check_inventory(missing, required)
        except GateError:
            continue
        raise GateError(f"The inventory accepted an omitted declaration: {name}")
    proof_controls(evidence, output / "source")
    summary = {name: gate(tool_root, output / name) for name, gate in SUBGATES.items()}
    (output / "subgates.json").write_text(json.dumps(summary, indent=2) + "\n")
    changed = [
        path for path, expected in frozen.items() if digest(ROOT / path) != expected
    ]
    if changed:
        raise GateError(f"S3 sources changed during the gate: {changed}")
    evidence.save()
    print(f"S3 gate passed ({', '.join(SUBGATES)}): {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_stage(args.tool_root, args.output)


if __name__ == "__main__":
    main()
