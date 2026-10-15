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
        ROOT / "tools/common.py",
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
        ROOT / "docs/scalar-transpose.md",
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

        acceptance, supplement_acceptance, family = parallel(
            corpus,
            supplement,
            partial(pointwise_family, tool_root, output / "pointwise-family"),
        )
        for lane, accepted in supplement_acceptance.items():
            acceptance[lane]["supplement"] = accepted
        acceptance["pointwise_family"] = family
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
