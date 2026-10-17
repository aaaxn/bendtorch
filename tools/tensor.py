import json
import math
import os
import shutil
import sys
from functools import partial
from pathlib import Path

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
from tools.tensor_families import run_gate as pointwise_family
from tools.tensor_numerics import special, verify_case
from tools.tensor_ordered import verify_ordered_case, verify_ordered_lane
from tools.tensor_parallel import run_gate as parallel_builds
from tools.tensor_runtime import (
    CORPUS,
    ORDERED_CONTROLS,
    ORDERED_REFERENCE,
    POLICY,
    POLICY_V3,
    SUCCESS_GROUPS,
    SUPPLEMENT,
    U32Range,
    corpus_cases,
    decode_case,
    error_name,
    load_corpus,
    run_native_corpus,
    run_supplement,
)
from tools.tensor_signed import run_gate as signed_requests


def verify_lane(corpus: dict, observations: list[dict], *, native: bool) -> dict:
    expected = []
    for case in corpus_cases(corpus):
        if not native:
            try:
                request = decode_case(case)
            except U32Range:
                continue
            if "compile_negative" in request:
                continue
        expected.append(case["id"])
    if [row["id"] for row in observations] != expected:
        raise GateError(
            "Tensor acceptance requires the complete ordered lane inventory"
        )
    observed = {row["id"]: row for row in observations}
    accepted = [
        verify_case(case, observed[case["id"]]["result"])
        for group in SUCCESS_GROUPS
        for case in corpus[group]
    ]
    errors = []
    for case in corpus["error_cases"]:
        identifier = case["id"]
        if identifier not in observed:
            continue
        row = observed[identifier]
        category = case["expected_error"]
        if category == "UnsupportedDtype":
            if (
                row.get("boundary") != "static_f32_type"
                or row.get("compile_negative") != case["request"]["dtype"]
                or "error" in row
            ):
                raise GateError(f"{identifier}: missing static F32 rejection")
        elif category == "U32Range":
            if row.get("boundary") != "decoder" or row.get("error") != category:
                raise GateError(f"{identifier}: missing pretransport rejection")
        elif (
            row.get("error") != category
            or "raw_error" not in row
            or error_name(row["raw_error"]) != category
        ):
            raise GateError(f"{identifier}: wrong native API error")
        errors.append({"id": identifier, "category": category})
    shapes = []
    for case in corpus["shape_cases"]:
        if observed[case["id"]].get("shape_result") != case["expected"]:
            raise GateError(f"{case['id']}: wrong shape-only result")
        shapes.append(case["id"])
    constructed_nan_words = {}
    for group in SUCCESS_GROUPS:
        for case in corpus[group]:
            if case["op"] != "construct":
                continue
            supplied = decode_case(case)["inputs"]["a"]["f32_bits"]
            actual = observed[case["id"]]["result"]["f32_bits"]
            for original, word in zip(supplied, actual, strict=True):
                if special(original) != "nan":
                    continue
                if (
                    original in constructed_nan_words
                    and constructed_nan_words[original] != word
                ):
                    raise GateError("Inconsistent same-lane constructed NaN words")
                constructed_nan_words[original] = word
    retained_binary = []
    for case in corpus_cases(corpus):
        if case["op"] not in ("add", "sub", "mul"):
            continue
        try:
            request = decode_case(case)
        except U32Range:
            continue
        if "compile_negative" in request:
            continue
        identifier = case["id"]
        row = observed[identifier]
        inputs = request["inputs"]
        construction_failed = any(
            len(encoded["shape"]) > 4
            or math.prod(encoded["shape"]) > 1048576
            or (
                not encoded.get("fill_zero")
                and len(encoded["f32_bits"]) != math.prod(encoded["shape"])
            )
            for encoded in (inputs["a"], inputs["b"])
        )
        if construction_failed:
            if row.get("boundary") != "construction" or "retained" in row:
                raise GateError(f"{identifier}: wrong pre-call construction boundary")
            continue
        if "boundary" in row or set(row.get("retained", {})) != {"left", "right"}:
            raise GateError(f"{identifier}: missing actual binary retained operands")
        for owner, name in (("left", "a"), ("right", "b")):
            encoded = inputs[name]
            words = (
                ["00000000"] * math.prod(encoded["shape"])
                if encoded.get("fill_zero")
                else encoded["f32_bits"]
            )
            expected_words = []
            for word in words:
                if special(word) == "nan":
                    if word not in constructed_nan_words:
                        raise GateError(f"{identifier}: missing same-lane NaN control")
                    word = constructed_nan_words[word]
                expected_words.append(word)
            if row["retained"][owner] != {
                "shape": encoded["shape"],
                "f32_bits": expected_words,
            }:
                raise GateError(f"{identifier}: wrong retained {owner} operand")
        retained_binary.append(
            {"id": identifier, "outcome": "error" if "error" in row else "success"}
        )
    return {
        "numerical_cases": accepted,
        "errors": errors,
        "shape_cases": shapes,
        "retained_binary": retained_binary,
    }


def verify_supplement(supplement: dict, observations: list[dict]) -> dict:
    observed = {row["id"]: row for row in observations}
    expected = [case["id"] for case in supplement["cases"] + supplement["error_cases"]]
    if [row["id"] for row in observations] != expected:
        raise GateError("Supplement acceptance requires the complete inventory")
    accepted, ordered, errors = [], [], []
    for case in supplement["cases"]:
        if "result" not in observed[case["id"]]:
            raise GateError(f"{case['id']}: supplement case did not succeed")
        result = observed[case["id"]]["result"]
        accepted.append(verify_case(case, result))
        if "ordered_reference" in case:
            ordered.append(
                verify_ordered_case(
                    {"id": case["id"], "result": case["ordered_reference"]}, result
                )
            )
    for case in supplement["error_cases"]:
        row = observed[case["id"]]
        if (
            row.get("error") != case["expected_error"]
            or error_name(row.get("raw_error", "")) != case["expected_error"]
        ):
            raise GateError(f"{case['id']}: wrong supplement error")
        errors.append({"id": case["id"], "category": case["expected_error"]})
    return {"numerical_cases": accepted, "ordered_reference": ordered, "errors": errors}


# Mutants of ReLU, its gradient and the comparisons with zero they use, each
# applied to a copy of src/tensor/elementwise.bend, and the law of
# src/tensor/relu-LAWS.bend whose proof rejects it. Each law named here is
# false for the mutant.
RELU_CONTROLS = {
    "relu-passes-negatives": (
        "  zero_if(below_zero(x), x)",
        "  zero_if(False{}, x)",
        "Laws.scalar_relu_left_below",
    ),
    "relu-zeroes-negative-zero": (
        "  Bool.and(U32.is_lt(2147483648, word), U32.is_le(word, 4286578688))\n",
        "  Bool.and(U32.is_le(2147483648, word), U32.is_le(word, 4286578688))\n",
        "Laws.below_zero_words",
    ),
    "relu-zeroes-negative-nan": (
        "  Bool.and(U32.is_lt(2147483648, word), U32.is_le(word, 4286578688))\n",
        "  Bool.and(U32.is_lt(2147483648, word), U32.is_le(word, 4294967295))\n",
        "Laws.below_zero_words",
    ),
    "relu-backward-passes-gradient-at-zero": (
        "  zero_if(at_most_zero(x), g)",
        "  zero_if(below_zero(x), g)",
        "Laws.scalar_relu_backward_zeroed",
    ),
    "relu-backward-ignores-gradient": (
        "  zero_if(at_most_zero(x), g)",
        "  zero_if(at_most_zero(x), x)",
        "scalar_relu_backward_passed_at_one",
    ),
    "relu-backward-zeroes-negative-nan": (
        "U32.is_le(word, 4286578688)))",
        "U32.is_le(word, 4294967295)))",
        "Laws.at_most_zero_words",
    ),
    "relu-backward-swaps-operands": (
        "      relu_backward(left, right)",
        "      relu_backward(right, left)",
        "Laws.scalar_relu_backward_zeroed",
    ),
}


# Instances of a relu law, checked on their own. A mutant that keeps the law
# a proof checks first true is rejected by an instance of the law it breaks:
# a gradient that passes x instead of g still gives +0.0 where x is at most
# zero, so it is checked at x = 1.0, where scalar_relu_backward_passed says
# the gradient is g's own word, 2.5. The check names the two words.
RELU_WITNESSES = {
    "scalar_relu_backward_passed_at_one": (
        "def scalar_relu_backward_passed_at_one() -> "
        "{Elementwise.scalar(Elementwise.ReluBackward{}, 1.0, 2.5) == 2.5 : F32}:\n"
        "  {==}\n",
        ("- expected : 1.0", "- observed : 2.5"),
    ),
}


def relu_witness(directory: Path, location: str) -> Path:
    source = directory / "relu-witness.bend"
    source.write_text(
        "import Base\nimport ./src/tensor/elementwise.bend as Elementwise\n\n"
        + RELU_WITNESSES[location][0]
    )
    return source


def relu_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "relu-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/elementwise.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        if location in RELU_WITNESSES:
            checked, words = relu_witness(copy, location), RELU_WITNESSES[location][1]
        else:
            checked, words = copy / "src/tensor/relu-PROOF.bend", ()
        evidence.run(
            f"relu-control-{name}",
            bend(evidence.tool_root, checked, "--check-only"),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}", *words),
        )

    def accepted(location: str) -> None:
        copy = evidence.directory / "relu-controls" / f"accepted-{location}"
        shutil.copytree(source_root / "src", copy / "src")
        evidence.run(
            f"relu-witness-{location}",
            bend(evidence.tool_root, relu_witness(copy, location), "--check-only"),
            contains="ALL PROOFS CHECK",
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in RELU_CONTROLS.items()
        ),
        *(partial(accepted, location) for location in RELU_WITNESSES),
    )


# Mutants of the scalar operations of the mean and the losses, each applied
# to a copy of src/tensor/elementwise.bend, and the law of
# src/tensor/loss-ops-LAWS.bend whose proof rejects it. Each law named here
# is false for the mutant.
LOSS_CONTROLS = {
    "divide-multiplies": (
        "      F32.div(left, right)",
        "      F32.mul(left, right)",
        "Laws.scalar_divide",
    ),
    "matches-inverted": (
        "      zero_if(Bool.not(U32.is_eq(bits(left), bits(right))), 1.0)",
        "      zero_if(U32.is_eq(bits(left), bits(right)), 1.0)",
        "Laws.scalar_matches_same",
    ),
    "pick-keeps-sign": (
        "      zero_if(Bool.not(is_one(right)), F32.neg(left))",
        "      zero_if(Bool.not(is_one(right)), left)",
        "Laws.scalar_pick_negated_target",
    ),
    "mse-grad-drops-scale": (
        "      F32.mul(F32.mul(scale, right), left)",
        "      F32.mul(right, left)",
        "Laws.scalar_mse_grad",
    ),
    "mse-grad-scales-last": (
        "      F32.mul(F32.mul(scale, right), left)",
        "      F32.mul(F32.mul(right, left), scale)",
        "Laws.scalar_mse_grad",
    ),
    "one-hot-grad-inverted": (
        "      zero_if(Bool.not(is_one(right)), target_gradient(left, count))",
        "      zero_if(is_one(right), target_gradient(left, count))",
        "Laws.scalar_one_hot_grad_target",
    ),
    "softmax-grad-drops-zero-add": (
        "    case False{}:\n      F32.add(t, 0.0)\n",
        "    case False{}:\n      t\n",
        "Laws.scalar_softmax_grad_classes",
    ),
    "softmax-grad-adds-zero-to-one-class": (
        "    case True{}:\n      t\n",
        "    case True{}:\n      F32.add(t, 0.0)\n",
        "Laws.scalar_softmax_grad_one_class",
    ),
}


def loss_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "loss-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/elementwise.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"loss-control-{name}",
            bend(
                evidence.tool_root,
                copy / "src/tensor/loss-ops-PROOF.bend",
                "--check-only",
            ),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}"),
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in LOSS_CONTROLS.items()
        )
    )


# Mutants of the signed wrap, the reshape size inference and signed slicing,
# each applied to a copy of src/tensor/signed.bend, and the definition of
# src/tensor/signed-PROOF.bend that rejects it: a law, or the lemma that
# proves one.
SIGNED_CONTROLS = {
    "last-wraps-to-limit-minus-k": (
        "U32.sub(U32.sub(limit, 1), k), limit)",
        "U32.sub(limit, k), limit)",
        "last_wrapped",
    ),
    "last-accepts-limit": (
        "Bool.pick(U32, U32.is_lt(k, limit),",
        "Bool.pick(U32, U32.is_le(k, limit),",
        "last_wrapped",
    ),
    "outside-wraps-to-zero": (
        "U32.sub(U32.sub(limit, 1), k), limit)",
        "U32.sub(U32.sub(limit, 1), k), 0)",
        "last_wrapped",
    ),
    "first-counts-from-end": (
        "    case First{k}:\n      k\n",
        "    case First{k}:\n      U32.sub(limit, k)\n",
        "Laws.signed_first_kept",
    ),
    "coordinates-wrap-among-zero": (
        "wrap(extent, head) <> wrap_each(rest, tail)",
        "wrap(0, head) <> wrap_each(rest, tail)",
        "Laws.signed_wrap_each_step",
    ),
    "permute-skips-wrap": (
        "Transforms.permute_copy(input, source, wrap_all(Shape.rank(input), axes))",
        "Transforms.permute_copy(input, source, wrap_all(0, axes))",
        "Laws.signed_permute_wraps",
    ),
    "transpose-wraps-among-rank": (
        "wrap(axis_limit(input), a), wrap(axis_limit(input), b))",
        "wrap(Shape.rank(input), a), wrap(Shape.rank(input), b))",
        "Laws.signed_transpose_wraps",
    ),
    "flat-wraps-among-rank": (
        "Core.read_flat(shape, tensor, wrap(Shape.count(shape), index))",
        "Core.read_flat(shape, tensor, wrap(Shape.rank(shape), index))",
        "Laws.signed_read_flat_wraps",
    ),
    "inferred-ignores-remainder": (
        "    case 1n+rest:\n      Fail{count_mismatch()}\n\n# A product past",
        "    case 1n+rest:\n      Done{filled(extents, U32.from_nat(quotient))}\n\n"
        "# A product past",
        "indivisible_decided",
    ),
    "ambiguous-as-count-mismatch": (
        "    case 0n:\n      Fail{AmbiguousInferred{}}",
        "    case 0n:\n      Fail{count_mismatch()}",
        "Laws.signed_inferred_ambiguous",
    ),
    "second-inferred-accepted": (
        "        case 1n+rest:\n          Fail{RepeatedInferred{}}",
        "        case 1n+rest:\n"
        "          inferred_one(count, extents, capped(count, extents))",
        "Laws.signed_inferred_repeated",
    ),
    "inferred-counts-as-zero": (
        "    case Inferred{} <> tail:\n      exact_product(tail)",
        "    case Inferred{} <> tail:\n      Nat.mul(0n, exact_product(tail))",
        "capped_classifies",
    ),
    "zero-size-keeps-product": (
        "    case 0n:\n      ZeroProduct{}\n    case 1n+(+less):",
        "    case 0n:\n      rest\n    case 1n+(+less):",
        "times_classify",
    ),
    "empty-list-multiplies-to-zero": (
        "    case Nil{}:\n      cap(bound, 1n)",
        "    case Nil{}:\n      cap(bound, 0n)",
        "capped_classifies",
    ),
    "past-size-kept": (
        "    case True{}:\n      Beyond{}\n    case False{}:\n"
        "      cap(bound, Nat.mul(size, product))",
        "    case True{}:\n      Within{size}\n    case False{}:\n"
        "      cap(bound, Nat.mul(size, product))",
        "times_scaled",
    ),
    "past-count-infers-one": (
        "      Done{filled(extents, 0)}\n    case 1n+rest:\n"
        "      Fail{count_mismatch()}",
        "      Done{filled(extents, 1)}\n    case 1n+rest:\n"
        "      Fail{count_mismatch()}",
        "size_beyond",
    ),
    "past-count-accepted": (
        "      Done{filled(extents, 0)}\n    case 1n+rest:\n"
        "      Fail{count_mismatch()}",
        "      Done{filled(extents, 0)}\n    case 1n+rest:\n"
        "      Done{filled(extents, 0)}",
        "indivisible_beyond",
    ),
    "cap-always-beyond": (
        "  bounded(product, Nat.is_le(product, bound))\n",
        "  Beyond{}\n",
        "times_scaled",
    ),
    # Signed slicing: a slice bound before -limit left at limit instead of
    # 0, or no Last before -limit (both rejected where bound_inside rewrites
    # the bound); every First before -limit; a slice whose start
    # is wrapped as an index, or whose axis is wrapped among the element
    # count; a select whose index is bounded like a slice position; a narrow
    # that takes every start; a narrow start before -extent that skips the
    # axis check.
    "bound-outside-to-limit": (
        "Bool.pick(U32, before(limit, position), 0, wrap(limit, position))",
        "Bool.pick(U32, before(limit, position), limit, wrap(limit, position))",
        "bound_inside",
    ),
    "before-takes-every-last": (
        "    case Last{k}:\n      Bool.not(U32.is_lt(k, limit))",
        "    case Last{k}:\n      False{}",
        "bound_inside",
    ),
    "before-holds-every-first": (
        "    case First{k}:\n      False{}\n    case Last{k}:\n      Bool.not(",
        "    case First{k}:\n      True{}\n    case Last{k}:\n      Bool.not(",
        "Laws.signed_bound_first",
    ),
    "slice-wraps-the-start": (
        "bound(n, start), bound(n, stop), step)",
        "wrap(n, start), bound(n, stop), step)",
        "Laws.signed_slice_bounds",
    ),
    "slice-axis-among-the-count": (
        "  +axis = wrap(Shape.rank(shape), dim)\n"
        "  +n = extent(Shape.dimensions(shape), axis)\n  Slice.slice(",
        "  +axis = wrap(Shape.count(shape), dim)\n"
        "  +n = extent(Shape.dimensions(shape), axis)\n  Slice.slice(",
        "Laws.signed_slice_bounds",
    ),
    "select-bounds-the-index": (
        "axis, wrap(extent(Shape.dimensions(shape), axis), index))",
        "axis, bound(extent(Shape.dimensions(shape), axis), index))",
        "Laws.signed_select_wraps",
    ),
    "narrow-takes-every-start": (
        "wrap(n, start), length, before(n, start))",
        "wrap(n, start), length, False{})",
        "Laws.signed_narrow_last_inside",
    ),
    "narrow-before-skips-the-axis": (
        "    case Some{error}:\n      (input, Fail{error})\n    case None{}:\n"
        "      (input, Fail{Slice.OutOfRange{}})",
        "    case Some{error}:\n      (input, Fail{Slice.OutOfRange{}})\n"
        "    case None{}:\n      (input, Fail{Slice.OutOfRange{}})",
        "Laws.signed_narrow_before_axis",
    ),
}


def signed_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "signed-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/signed.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"signed-control-{name}",
            bend(
                evidence.tool_root,
                copy / "src/tensor/signed-PROOF.bend",
                "--check-only",
            ),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}"),
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in SIGNED_CONTROLS.items()
        )
    )


# Mutants of torch.matmul (src/tensor/batched-matmul.bend), each applied to
# a copy of the source, and where the proof of
# src/tensor/batched-matmul-PROOF.bend fails: the law, or the lemma of the
# law, that is false for the mutant. A rank-2 pair sent to the plan breaks
# batched_matmul_matrices, a rank-0 left operand read as a vector breaks
# batched_matmul_left_scalar, a plan that skips the inner check breaks the
# refusal of a dot of unequal lengths (dot_compatible), and a cell that
# folds from -0.0 breaks the fold of batched_build_cells (from_run). Three
# break the geometry stated from PyTorch's definition: a stretched batch
# axis read at the output's own index instead of 0 breaks
# batched_build_logical_cells (stretch_spec, the shape module's broadcast
# index), an axis of extent 1 that keeps its own extent instead of the
# other one's breaks batched_plan_geometry (axis_left), and a 1-D left
# operand whose row the output keeps breaks it too (matrix_part).
LEFT_OPERAND = (
    "def left_operand(dims: List<&2, U32>) -> "
    "Result<&2, &2, Matrix.Error, Operand>:\n  match dims:\n"
)
BATCHED_CONTROLS = {
    "batched-routes-matrices-to-plan": (
        "    case True{}:\n"
        "      Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b})\n",
        "    case True{}:\n      planned(left, right, a, b, plan_dims(dims, other))\n",
        "Laws.batched_matmul_matrices",
    ),
    "batched-accepts-rank-zero-left": (
        LEFT_OPERAND,
        LEFT_OPERAND
        + "    case Nil{}:\n      Done{Operand{Batch{0n, 1, 1}, 1, 1, True{}}}\n",
        "Laws.batched_matmul_left_scalar",
    ),
    "batched-skips-inner-check": (
        "    case False{}:\n      Fail{Matrix.InnerDimensionMismatch{}}\n",
        "    case False{}:\n      shaped(left, right)\n",
        "dot_compatible",
    ),
    "batched-dot-starts-from-negative-zero": (
        "      Matrix.product_run(rest, ls, rs, lp, rp, 0.0, ",
        "      Matrix.product_run(rest, ls, rs, lp, rp, F32.neg(0.0), ",
        "from_run",
    ),
    "batched-stretched-axis-reads-own-index": (
        "    case True{}:\n      0n\n    case False{}:\n      index\n",
        "    case True{}:\n      index\n    case False{}:\n      index\n",
        "stretch_spec",
    ),
    "batched-axis-keeps-the-stretched-extent": (
        "    case False{} True{} _:\n      Done{r}\n",
        "    case False{} True{} _:\n      Done{l}\n",
        "axis_left",
    ),
    "batched-keeps-the-promoted-row": (
        "    case True{}:\n      column_dims(columns, right_vector)\n",
        "    case True{}:\n      rows <> column_dims(columns, right_vector)\n",
        "matrix_part",
    ),
}


def batched_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "batched-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/batched-matmul.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"batched-control-{name}",
            bend(
                evidence.tool_root,
                copy / "src/tensor/batched-matmul-PROOF.bend",
                "--check-only",
            ),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}"),
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in BATCHED_CONTROLS.items()
        )
    )


# Mutants of basic slicing (src/tensor/slice.bend), each applied to a copy of
# the source, and where the proof of src/tensor/slice-PROOF.bend fails: the
# first law, or lemma of a law, whose proof no longer checks. Each mutant
# makes at least one slice law false. The clamp mutants all fail at the
# proof of the clamped case, which follows Nat.min and Nat.max: a start or
# stop left unclamped falsifies slice_start_clamped or slice_stop_clamped, a
# start one past slice_start_kept, a stop one short slice_stop_kept, a stop
# before the start kept as it is slice_stop_empty. A length one short
# falsifies slice_length_reaches, one that counts the whole span
# slice_length_below; both fail at below_span, whose proof follows the
# length's form. A plan that starts at 0 instead of the request's start
# falsifies slice_plan_axis, a view of an empty input that loses the extent
# too (sized_extent); a plan that takes every input as empty falsifies
# slice_plan_view (its proof fails first at slice_plan_axis), and a zero
# check that skips the first dimension slice_plan_empty_view (zero_head). A
# doubled inner count falsifies slice_view_around (view_around), a resized
# axis that keeps its extent slice_resized_around, a select that keeps the
# axis select_dropped_around, an output one cell longer slice_planned
# (planned_checked). A cell that reads without the start, or every index
# instead of every step-th, falsifies slice_build_cells (source_at); a
# backward that reads the input's row of the gradient slice_scatter_hit
# (target_hit), one that fills the cells between steps slice_scatter_miss
# (missed_k). A zero step or an axis at the rank accepted, a refusal renamed,
# a narrow that checks only its start or slices one more, and a select that
# accepts the extent or slices two falsify the law of that check.
SLICE_CONTROLS = {
    "slice-start-not-clamped": (
        "  +low = Nat.min(start, extent)\n",
        "  +low = start\n",
        "Laws.slice_start_clamped",
    ),
    "slice-start-one-past": (
        "  +low = Nat.min(start, extent)\n",
        "  +low = Nat.min(Nat.add(start, 1n), extent)\n",
        "Laws.slice_start_clamped",
    ),
    "slice-stop-not-clamped": (
        "  Bounds{low, Nat.max(low, Nat.min(stop, extent))}",
        "  Bounds{low, Nat.max(low, stop)}",
        "Laws.slice_stop_clamped",
    ),
    "slice-stop-one-short": (
        "  Bounds{low, Nat.max(low, Nat.min(stop, extent))}",
        "  Bounds{low, Nat.max(low, Nat.min(Nat.sub(stop, 1n), extent))}",
        "Laws.slice_stop_clamped",
    ),
    "slice-stop-before-the-start-kept": (
        "  Bounds{low, Nat.max(low, Nat.min(stop, extent))}",
        "  Bounds{low, Nat.min(stop, extent)}",
        "Laws.slice_stop_clamped",
    ),
    "slice-length-drops-the-first": (
        "      1n+Nat.div(rest, step)\n",
        "      Nat.div(rest, step)\n",
        "below_span",
    ),
    "slice-length-counts-the-span": (
        "      1n+Nat.div(rest, step)\n",
        "      1n+Nat.div(1n+rest, step)\n",
        "below_span",
    ),
    "slice-plan-starts-at-zero": (
        "holds_zero(dimensions)), U32.to_nat(start),",
        "holds_zero(dimensions)), 0n,",
        "Laws.slice_plan_axis",
    ),
    "slice-empty-view-loses-the-extent": (
        "      View{0n, axis_extent(dimensions, dim), 0n}",
        "      View{0n, 0n, 0n}",
        "sized_extent",
    ),
    "slice-plan-skips-the-zero-check": (
        "sized(dimensions, U32.to_nat(dim), holds_zero(dimensions))",
        "sized(dimensions, U32.to_nat(dim), True{})",
        "Laws.slice_plan_axis",
    ),
    "holds-zero-ignores-the-head": (
        "      Bool.or(Nat.is_eq(U32.to_nat(extent), 0n), holds_zero(rest))",
        "      holds_zero(rest)",
        "zero_head",
    ),
    "slice-view-inner-doubled": (
        "      View{outer, U32.to_nat(extent), Shape.logical_numel(rest)}",
        "      View{outer, U32.to_nat(extent), Nat.mul(2n, Shape.logical_numel(rest))}",
        "view_around",
    ),
    "slice-resized-keeps-the-extent": (
        "    case head <> rest 0n:\n      extent <> rest\n",
        "    case head <> rest 0n:\n      head <> rest\n",
        "Laws.slice_resized_around",
    ),
    "select-dropped-keeps-the-axis": (
        "    case head <> rest 0n:\n      rest\n",
        "    case head <> rest 0n:\n      head <> rest\n",
        "Laws.select_dropped_around",
    ),
    "slice-output-one-longer": (
        "      +output = resized(dimensions, U32.to_nat(dim),"
        " U32.from_nat(plan_length(plan)))",
        "      +output = resized(dimensions, U32.to_nat(dim),"
        " U32.from_nat(1n+plan_length(plan)))",
        "planned_checked",
    ),
    "slice-reads-without-the-start": (
        "Nat.add(start, Nat.mul(Nat.mod(row, length), step))",
        "Nat.mul(Nat.mod(row, length), step)",
        "source_at",
    ),
    "slice-reads-every-index": (
        "Nat.add(start, Nat.mul(Nat.mod(row, length), step))",
        "Nat.add(start, Nat.mod(row, length))",
        "source_at",
    ),
    "slice-backward-reads-the-input-row": (
        "Nat.add(Nat.mul(Nat.div(row, extent), length), k)",
        "Nat.add(Nat.mul(Nat.div(row, extent), extent), k)",
        "target_hit",
    ),
    "slice-backward-hits-between-steps": (
        "Bool.and(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k,"
        " step)), i))",
        "Nat.is_lt(k, length)",
        "missed_k",
    ),
    "slice-accepts-a-zero-step": (
        "    case True{} False{}:\n      Some{StepNotPositive{}}\n",
        "    case True{} False{}:\n      None{}\n",
        "Laws.slice_step_refused",
    ),
    "slice-accepts-the-rank": (
        "  refused(Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(dimensions))",
        "  refused(Nat.is_le(U32.to_nat(dim), Shape.rank_dimensions(dimensions))",
        "Laws.slice_axis_refused",
    ),
    "slice-renames-a-refusal": (
        "    case Fail{error}:\n      (Core.Packed{shape, tensor}, Fail{error})\n"
        "    case Done{Planned{+plan, +output}}:",
        "    case Fail{error}:\n      (Core.Packed{shape, tensor},"
        " Fail{OutOfRange{}})\n"
        "    case Done{Planned{+plan, +output}}:",
        "Laws.slice_refused",
    ),
    "narrow-checks-only-the-start": (
        "        Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), "
        "extent_at(Shape.dimensions(shape), dim)))",
        "        Nat.is_le(U32.to_nat(start),"
        " extent_at(Shape.dimensions(shape), dim)))",
        "Laws.narrow_out_of_range",
    ),
    "narrow-slices-one-more": (
        "      slice(Core.Packed{shape, tensor}, dim, start, U32.add(start,"
        " length), 1)",
        "      slice(Core.Packed{shape, tensor}, dim, start, "
        "U32.add(U32.add(start, length), 1), 1)",
        "Laws.narrow_slices",
    ),
    "narrow-renames-the-axis-error": (
        "    case Some{error}:\n      (Core.Packed{shape, tensor}, Fail{error})\n"
        "    case None{}:\n      narrow_ranged",
        "    case Some{error}:\n      (Core.Packed{shape, tensor},"
        " Fail{OutOfRange{}})\n"
        "    case None{}:\n      narrow_ranged",
        "Laws.narrow_axis_refused",
    ),
    "select-accepts-the-extent": (
        "      select_ranged(shape, tensor, dim, index, Nat.is_lt(U32.to_nat(index), "
        "extent_at(Shape.dimensions(shape), dim)))",
        "      select_ranged(shape, tensor, dim, index, Nat.is_le(U32.to_nat(index), "
        "extent_at(Shape.dimensions(shape), dim)))",
        "Laws.select_out_of_range",
    ),
    "select-slices-two": (
        "slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 1), 1))",
        "slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 2), 1))",
        "Laws.select_slices",
    ),
    "select-renames-the-axis-error": (
        "    case Some{error}:\n      (Core.Packed{shape, tensor}, Fail{error})\n"
        "    case None{}:\n      select_ranged",
        "    case Some{error}:\n      (Core.Packed{shape, tensor},"
        " Fail{OutOfRange{}})\n"
        "    case None{}:\n      select_ranged",
        "Laws.select_axis_refused",
    ),
}


def slice_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "slice-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/slice.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"slice-control-{name}",
            bend(
                evidence.tool_root, copy / "src/tensor/slice-PROOF.bend", "--check-only"
            ),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}"),
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in SLICE_CONTROLS.items()
        )
    )


# Mutants of index_select and gather (src/tensor/gather.bend), each applied to
# a copy of the source, and where the proof of src/tensor/gather-PROOF.bend
# fails: the law, or the lemma of a law, whose proof no longer checks. An
# index_select that reads the next entry, or a gather that reads the entry
# of its row's first cell, falsifies gather_build_cells (entry_same), and a
# source that strides by the output's length instead of the input's extent
# too (source_at). An entry check that accepts the extent falsifies the
# refusal and planning laws (below_inside), as do a shape check that skips
# an extent (matches_others) and an entry count one short (count_length);
# an index_select without its axis check, a gather without its count check
# or one that counts against the index's rank, a rank mismatch reported as a
# shape mismatch, and a gather whose length comes from the input fail the
# law of that check. A backward that
# overwrites instead of adding (accumulated), adds every entry's word
# (compared_is), skips the first gradient cell or starts from -0.0
# (scatter_keeps) falsifies gather_scatter_cells.
GATHER_CONTROLS = {
    "select-reads-the-next-entry": (
        "    case Selected{}:\n"
        "      k\n"
        "    case Gathered{}:\n"
        "      position(length, inner, o, k, r)",
        "    case Selected{}:\n"
        "      1n+k\n"
        "    case Gathered{}:\n"
        "      position(length, inner, o, k, r)",
        "entry_same",
    ),
    "gather-reads-the-row-entry": (
        "    case Selected{}:\n"
        "      k\n"
        "    case Gathered{}:\n"
        "      position(length, inner, o, k, r)",
        "    case Selected{}:\n"
        "      k\n"
        "    case Gathered{}:\n"
        "      position(length, inner, o, k, 0n)",
        "entry_same",
    ),
    "source-strides-by-the-length": (
        "  position(extent, inner, Nat.div(Nat.div(at, inner), "
        "length), j, Nat.mod(at, inner))",
        "  position(length, inner, Nat.div(Nat.div(at, inner), "
        "length), j, Nat.mod(at, inner))",
        "source_at",
    ),
    "below-accepts-the-extent": (
        "      Bool.and(Nat.is_lt(U32.to_nat(head), extent), below(rest, extent))",
        "      Bool.and(Nat.is_le(U32.to_nat(head), extent), below(rest, extent))",
        "below_inside",
    ),
    "matches-skips-an-extent": (
        "      Bool.and(Nat.is_eq(U32.to_nat(extent), "
        "U32.to_nat(other)), matches(rest, more, next))",
        "      matches(rest, more, next)",
        "matches_others",
    ),
    "count-one-short": (
        "    case head <> rest:\n      1n+count(rest)",
        "    case head <> rest:\n      count(rest)",
        "count_length",
    ),
    "select-skips-the-axis-check": (
        "      Fail{LayoutError{Layout.AxisOutOfBounds{}}}\n"
        "    case True{}:\n"
        "      ranged(Selected{}",
        "      ranged(Selected{}, dimensions, axis, entries, "
        "count(entries), below(entries, Slice.axis_extent(dimensions, axis)))\n"
        "    case True{}:\n"
        "      ranged(Selected{}",
        "Laws.index_select_axis_refused",
    ),
    "gather-skips-the-count-check": (
        "    case False{}:\n      Fail{TensorError{Core.ValueCountMismatch{}}}",
        "    case False{}:\n"
        "      gather_axis(dimensions, axis, index, entries, "
        "Nat.is_lt(axis, Shape.rank_dimensions(dimensions)))",
        "Laws.gather_count_refused",
    ),
    "count-against-the-rank": (
        "    Nat.is_eq(count(entries), Shape.elements(shape)))",
        "    Nat.is_eq(count(entries), "
        "Shape.rank_dimensions(Shape.dimensions(shape))))",
        "Laws.gather_count_refused",
    ),
    "rank-mismatch-renamed": (
        "      Fail{IndexRankMismatch{}}",
        "      Fail{IndexShapeMismatch{}}",
        "Laws.gather_rank_refused",
    ),
    "gather-length-from-the-input": (
        "      ranged(Gathered{}, dimensions, axis, entries, "
        "Slice.axis_extent(index, axis),",
        "      ranged(Gathered{}, dimensions, axis, entries, "
        "Slice.axis_extent(dimensions, axis),",
        "Laws.gather_out_of_range",
    ),
    "scatter-overwrites": (
        "  (F32.add(acc, x), (g, index))",
        "  (x, (g, index))",
        "accumulated",
    ),
    "scatter-adds-every-entry": (
        "  summed(Nat.is_eq(U32.to_nat(j), i), at, acc, g, index)",
        "  summed(True{}, at, acc, g, index)",
        "compared_is",
    ),
    "scatter-starts-at-the-second": (
        "Nat.mod(at, inner), length, 0n, (0.0, state))",
        "Nat.mod(at, inner), length, 1n, (0.0, state))",
        "scatter_keeps",
    ),
    "scatter-starts-at-negative-zero": (
        "Nat.mod(at, inner), length, 0n, (0.0, state))",
        "Nat.mod(at, inner), length, 0n, (F32.neg(0.0), state))",
        "scatter_keeps",
    ),
}


def gather_controls(evidence: Evidence, source_root: Path) -> None:
    def control(name: str, original: str, mutated: str, location: str) -> None:
        copy = evidence.directory / "gather-controls" / name
        shutil.copytree(source_root / "src", copy / "src")
        target = copy / "src/tensor/gather.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        evidence.run(
            f"gather-control-{name}",
            bend(
                evidence.tool_root,
                copy / "src/tensor/gather-PROOF.bend",
                "--check-only",
            ),
            expected_code=1,
            contains=("SOME PROOFS FAIL", f"Location: {location}"),
        )

    parallel(
        *(
            partial(control, name, *control_text)
            for name, control_text in GATHER_CONTROLS.items()
        )
    )


def certificate_controls(evidence: Evidence, source_root: Path) -> None:
    imports = "import Base\n"
    for path, alias in (
        ("src/shape/core.bend", "Shape"),
        ("src/storage/main.bend", "Storage"),
        ("src/tensor/core.bend", "Core"),
        ("src/tensor/layout.bend", "Layout"),
        ("src/tensor/transforms.bend", "Transforms"),
    ):
        relative = os.path.relpath(source_root / path, evidence.directory)
        imports += f"import ./{relative} as {alias}\n"
    shapes = """
def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def input_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def target_shape() -> Shape.Shape:
  Shape.SShape{[3, 2], 6n, {==}, Unit{}, Unit{}}

def wrong_shape() -> Shape.Shape:
  Shape.SShape{[4], 4n, {==}, Unit{}, Unit{}}

def square_shape() -> Shape.Shape:
  Shape.SShape{[2, 2], 4n, {==}, Unit{}, Unit{}}

def cube_shape() -> Shape.Shape:
  Shape.SShape{[2, 2, 2], 8n, {==}, Unit{}, Unit{}}

def volume_shape() -> Shape.Shape:
  Shape.SShape{[2, 3, 4], 24n, {==}, Unit{}, Unit{}}

def empty_tensor() -> Core.Tensor<empty_shape()>:
  Core.TTensor{Storage.Empty{}, {==}}

def mapped_word.result(-shape: Shape.Shape, -axes: List<&2, U32>,
  -plan: Layout.Plan<shape, axes>, -index: U32,
  result: Result<&2, &2, Layout.Error,
    Layout.Mapping<shape, axes, plan, index>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{mapping}:
      Some{Layout.source_index(shape, axes, plan, index, mapping)}

def mapped_word.planned(+shape: Shape.Shape, +axes: List<&2, U32>, +index: U32,
  result: Result<&2, &2, Layout.Error, Layout.Plan<shape, axes>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{+plan}:
      mapped_word.result(shape, axes, plan, index,
        Layout.map_index(shape, axes, plan, index))

def mapped_word(+shape: Shape.Shape, +axes: List<&2, U32>, +index: U32) ->
  Maybe<&2, U32>:
  mapped_word.planned(shape, axes, index, Layout.permutation(shape, axes))

def composed_word(+shape: Shape.Shape, +axes: List<&2, U32>,
  middle: Maybe<&2, U32>) -> Maybe<&2, U32>:
  match middle:
    case None{}:
      None{}
    case Some{index}:
      mapped_word(shape, axes, index)
"""
    controls = {
        "tensor-length": (
            "def witness() -> Core.Tensor<empty_shape()>:\n"
            "  Core.TTensor{Storage.Empty{}, {==}}\n",
            "def witness() -> Core.Tensor<scalar_shape()>:\n"
            "  Core.TTensor{Storage.Empty{}, {==}}\n",
        ),
        "reader-outcome": (
            "def witness() -> Core.Read<empty_shape(), empty_tensor(), 0>:\n"
            "  Core.Read{empty_tensor(), "
            "Fail{Core.StorageError{Storage.OutOfBounds{}}}, {==}}\n",
            "def witness() -> Core.Read<empty_shape(), empty_tensor(), 0>:\n"
            "  Core.Read{empty_tensor(), Done{0.0}, {==}}\n",
        ),
        "value-count": (
            "def witness() -> Result<&2, &1, Core.Error, Core.Tensor<empty_shape()>>:\n"
            "  Core.from_values_counted(empty_shape(), [], Shape.BTrue{{==}})\n",
            "def witness() -> "
            "Result<&2, &1, Core.Error, Core.Tensor<scalar_shape()>>:\n"
            "  Core.from_values_counted(scalar_shape(), [], Shape.BTrue{{==}})\n",
        ),
        "reshape-count": (
            "def witness(tensor: Core.Tensor<input_shape()>) -> Core.Dynamic:\n"
            "  Transforms.reshape_equal(input_shape(), target_shape(), tensor, {==})\n",
            "def witness(tensor: Core.Tensor<input_shape()>) -> Core.Dynamic:\n"
            "  Transforms.reshape_equal(input_shape(), wrong_shape(), tensor, {==})\n",
        ),
        "permutation-output": (
            "def witness() -> Layout.Plan<input_shape(), [1, 0]>:\n"
            "  Layout.Permutation{target_shape(), {==}, {==}, {==}}\n",
            "def witness() -> Layout.Plan<input_shape(), [1, 0]>:\n"
            "  Layout.Permutation{input_shape(), {==}, {==}, {==}}\n",
        ),
        "permutation-identity-map": (
            "def witness() -> {mapped_word(square_shape(), [1, 0], 1) == "
            "Some{2} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {mapped_word(square_shape(), [1, 0], 1) == "
            "Some{1} : Maybe<&2, U32>}:\n  {==}\n",
        ),
        "permutation-wrong-axis": (
            "def witness() -> {mapped_word(volume_shape(), [2, 0, 1], 1) == "
            "Some{4} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {mapped_word(volume_shape(), [1, 0, 2], 1) == "
            "Some{4} : Maybe<&2, U32>}:\n  {==}\n",
        ),
        "permutation-wrong-inverse": (
            "def witness() -> {composed_word(cube_shape(), [1, 2, 0], "
            "mapped_word(cube_shape(), [2, 0, 1], 1)) == "
            "Some{1} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {composed_word(cube_shape(), [1, 2, 0], "
            "mapped_word(cube_shape(), [1, 2, 0], 1)) == "
            "Some{1} : Maybe<&2, U32>}:\n  {==}\n",
        ),
        "transpose-requested-map": (
            "def witness() -> {mapped_word(square_shape(), "
            "Layout.transpose_axes(Shape.dimensions(square_shape()), 0, 1, 0), "
            "1) == Some{2} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {mapped_word(square_shape(), "
            "Layout.transpose_axes(Shape.dimensions(square_shape()), 0, 1, 0), "
            "1) == Some{1} : Maybe<&2, U32>}:\n  {==}\n",
        ),
        "transpose-wrong-pair": (
            "def witness() -> {mapped_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 2, 0), "
            "1) == Some{4} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {mapped_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 1, 0), "
            "1) == Some{4} : Maybe<&2, U32>}:\n  {==}\n",
        ),
        "transpose-wrong-inverse": (
            "def witness() -> {composed_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 2, 0), "
            "mapped_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 2, 0), "
            "1)) == Some{1} : Maybe<&2, U32>}:\n  {==}\n",
            "def witness() -> {composed_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 1, 0), "
            "mapped_word(cube_shape(), "
            "Layout.transpose_axes(Shape.dimensions(cube_shape()), 0, 2, 0), "
            "1)) == Some{1} : Maybe<&2, U32>}:\n  {==}\n",
        ),
    }

    def control(name: str, label: str, body: str) -> None:
        source = evidence.directory / f"{name}-{label}.bend"
        source.write_text(imports + shapes + "\n" + body)
        for mode in ("--check-only", "--verdict"):
            evidence.run(
                f"{name}-{label}-{mode[2:]}",
                bend(evidence.tool_root, source, mode),
                expected_code=0 if label == "accepted" else 1,
                contains=(
                    "ALL PROOFS CHECK"
                    if label == "accepted"
                    else (
                        "SOME PROOFS FAIL",
                        "Location: witness",
                        "expected :",
                        "observed :",
                    )
                ),
            )
        if label == "accepted":
            artifact = source.with_suffix(".bendtt")
            evidence.run(
                f"{name}-export", bend(evidence.tool_root, source, "-o", artifact)
            )
            evidence.run(
                f"{name}-direct-kernel",
                [str(evidence.tool_root / "kernel/bendtt"), str(artifact)],
                contains="ALL PROOFS CHECK",
            )

    parallel(
        *(
            partial(control, name, label, body)
            for name, (accepted, rejected) in controls.items()
            for label, body in (("accepted", accepted), ("rejected", rejected))
        )
    )
    positive = evidence.directory / "accepted-f32.bend"
    positive.write_text(
        imports + "\ndef witness(values: List<&2, F32>) -> "
        "Result<&2, &1, Core.Error, Core.Dynamic>:\n"
        + "  Core.construct([1], values)\n"
    )
    for mode in ("--check-only", "--verdict"):
        evidence.run(
            f"accepted-f32-{mode[2:]}",
            bend(evidence.tool_root, positive, mode),
            contains="ALL PROOFS CHECK",
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
        ROOT / "verification/tensor.json",
        ROOT / "verification/tensor-operators.json",
        ROOT / "verification/references.json",
        ROOT / "tools/tensor.py",
        ROOT / "tools/tensor_runtime.py",
        ROOT / "tools/tensor_families.py",
        ROOT / "tools/tensor_numerics.py",
        ROOT / "tools/tensor_ordered.py",
        ROOT / "tools/tensor_parallel.py",
        ROOT / "tools/tensor_signed.py",
        ROOT / "probes/parallel_builds.bend",
        ROOT / "tools/common.py",
        ROOT / "tools/cache.py",
        ROOT / "tools/storage.py",
        ROOT / "oracle/tensor/ordered_reference.py",
        ROOT / "oracle/tensor/tests/test_ordered_reference.py",
        ROOT / "tests/test_tensor_ordered.py",
        ROOT / "tests/test_tensor_ordered_integration.py",
        ROOT / "oracle/tensor/ORDERED_REFERENCE.md",
        ROOT / "oracle/tensor/generate.py",
        ROOT / "oracle/tensor/transpose_reference.py",
        ROOT / "oracle/tensor/fixtures/scalar-transpose-correction.json",
        ROOT / "tests/test_scalar_transpose.py",
        ROOT / "oracle/tensor/SCALAR_TRANSPOSE.md",
        CORPUS,
        POLICY,
        POLICY_V3,
        SUPPLEMENT,
        ROOT / "oracle/tensor/supplement.py",
        ORDERED_REFERENCE,
        ORDERED_CONTROLS,
    ]
    frozen = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        frozen[str(relative)] = digest(target)
    (output / "source-inventory.json").write_text(json.dumps(frozen, indent=2) + "\n")

    def runtime() -> None:
        evidence.run(
            "pytorch-absent",
            [
                sys.executable,
                "-c",
                "import importlib.util; "
                "assert importlib.util.find_spec('torch') is None; "
                "print('PyTorch is absent from tensor execution')",
            ],
            expected_output="PyTorch is absent from tensor execution\n",
        )

        # Each task checks its own results before it returns, so the first
        # failure in sequential order is the one the gate reports.
        def corpus() -> dict:
            execution = run_native_corpus(
                tool_root, output / "runtime", javascript=True
            )
            summary = execution["summary"]
            if summary["javascript"]["status"] != "completed":
                raise GateError(
                    "The generated-Bend JavaScript lane is mandatory "
                    "for the tensor gate"
                )
            if (
                summary["execution_mode"] != "source_frozen_run"
                or summary["live_source_drift"]
            ):
                raise GateError("A screening run is not final tensor evidence")
            runtime_sources = json.loads(
                (output / "runtime/identity.json").read_text()
            )["source_sha256"]
            mismatched = [
                path
                for path, expected in runtime_sources.items()
                if frozen.get(path) != expected
            ]
            if mismatched:
                raise GateError(
                    f"Tensor proof and executable snapshots disagree: {mismatched}"
                )
            corpus = load_corpus(output / "source" / CORPUS.relative_to(ROOT))
            records = json.loads(
                (output / "source" / ORDERED_REFERENCE.relative_to(ROOT)).read_text()
            )
            acceptance = {}
            for lane, observations, native in (
                ("native", execution["results"], True),
                ("javascript", execution["javascript_results"], False),
            ):
                acceptance[lane] = verify_lane(corpus, observations, native=native)
                acceptance[lane]["ordered_reference"] = verify_ordered_lane(
                    corpus, records, observations
                )
                acceptance[lane]["ordered_controls"] = summary["ordered_controls"][lane]
            return acceptance

        def supplement() -> dict:
            result = run_supplement(tool_root, output / "supplement")
            return {
                lane: verify_supplement(result["supplement"], observations)
                for lane, observations in result["results"].items()
            }

        acceptance, supplement_acceptance, family, builds, signed = parallel(
            corpus,
            supplement,
            partial(pointwise_family, tool_root, output / "pointwise-family"),
            partial(parallel_builds, tool_root, output / "parallel-builds"),
            partial(signed_requests, tool_root, output / "signed"),
        )
        for lane, accepted in supplement_acceptance.items():
            acceptance[lane]["supplement"] = accepted
        acceptance["pointwise_family"] = family
        acceptance["parallel_builds"] = builds
        acceptance["signed"] = signed
        (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        links = evidence.run(
            "native-dependencies", ["ldd", str(output / "runtime/corpus")]
        )
        libraries = [
            line.split()[0].lower()
            for line in links.stdout.splitlines()
            if line.split()
        ]
        if any(
            name in library
            for library in libraries
            for name in ("cuda", "nvrtc", "libtorch", "libc10")
        ):
            raise GateError(
                "Tensor native executable links a forbidden compute backend"
            )
        certificate_controls(evidence, output / "runtime/source")
        relu_controls(evidence, output / "source")
        loss_controls(evidence, output / "source")
        signed_controls(evidence, output / "source")
        batched_controls(evidence, output / "source")
        slice_controls(evidence, output / "source")
        gather_controls(evidence, output / "source")

    parallel(
        partial(
            check_laws,
            evidence,
            "tensor",
            output / "source/src/tensor/PROOF.bend",
            required_exports(output / "source/verification/tensor.json"),
        ),
        runtime,
    )
    changed = [
        path for path, expected in frozen.items() if digest(ROOT / path) != expected
    ]
    if changed:
        raise GateError(f"Tensor sources changed during the gate: {changed}")
    evidence.save()
    print(f"Tensor source proofs and native/JS numerical conformance passed: {output}")
