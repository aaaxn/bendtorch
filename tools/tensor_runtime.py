import argparse
import json
import os
import platform
import re
import shutil
from pathlib import Path

from tools import common
from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_sources,
    digest,
    native,
)
from tools.shape import broadcast_oracle, shape_oracle
from tools.storage import instrument_copy_audit
from tools.tensor_ordered import verify_ordered_case

CORPUS = ROOT / "oracle/tensor/fixtures/s2-v2.json"
POLICY = ROOT / "oracle/tensor/numerical-policy.json"
POLICY_V3 = ROOT / "oracle/tensor/numerical-policy-v3.json"
SUPPLEMENT = ROOT / "oracle/tensor/fixtures/s2-supplement-v1.json"
ORDERED_REFERENCE = ROOT / "oracle/tensor/fixtures/ordered-reference.json"
ORDERED_CONTROLS = ROOT / "oracle/tensor/fixtures/ordered-controls.json"
CORPUS_SHA256 = "35d996e7147b79444a78862fd1ffbfec44ed8728e5ea242ab96b7719bb251ed2"
POLICY_SHA256 = "0a0e28b2e4666aa9a859af9a091cf0a2cad93ac10f7bb4c68413f1250853d9a4"
ORDERED_REFERENCE_SHA256 = (
    "428308966f75824217b98cfee0795f76d450feb3dc91f370073abf2c757bfcb8"
)
POLICY_V3_SHA256 = "e3b1bd0807b1e6f686a57574c64f961fc9de6fedec97f0a74a3bc2142add001f"
SUPPLEMENT_SHA256 = "64069566511462006eb67044099170ab8db24f73880e13cd362b9e2994fd9460"
ORDERED_CONTROLS_SHA256 = (
    "4fa88466a9e82d60c812ba8423897e11118d9303d7139d5db907ec3a978f7a7b"
)
MAX_U32 = (1 << 32) - 1
SUCCESS_GROUPS = ("finite_cases", "edge_cases", "nonfinite_cases")
ERROR_CODES = {
    1: "RankExceeded",
    2: "CapacityExceeded",
    3: "CoordinateRankMismatch",
    4: "CoordinateOutOfBounds",
    5: "FlatIndexOutOfBounds",
    6: "IncompatibleBroadcast",
    7: "CapacityExceeded",
    8: "FlatIndexOutOfBounds",
    9: "InvalidIndex",
    10: "DataLengthMismatch",
    11: "StorageLengthMismatch",
    12: "InvalidPermutation",
    13: "AxisOutOfBounds",
    14: "InvalidPermutation",
    15: "ReshapeElementCountMismatch",
    16: "MappingError",
}


class U32Range(GateError):
    pass


def u32(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_U32:
        raise U32Range(f"U32Range: {value!r}")
    return value


def u32_list(values: list) -> list[int]:
    return [u32(value) for value in values]


def tensor_input(encoded: dict) -> dict:
    shape = u32_list(encoded["shape"])
    if "f32_bits" not in encoded:
        return {"shape": shape, "fill_zero": True}
    words = encoded["f32_bits"]
    if not isinstance(words, list) or any(
        not isinstance(word, str) or re.fullmatch(r"[0-9a-f]{8}", word) is None
        for word in words
    ):
        raise GateError("Malformed F32 input word")
    return {"shape": shape, "f32_bits": words.copy()}


def decode_case(case: dict) -> dict:
    identifier = case["id"]
    if (
        not isinstance(identifier, str)
        or re.fullmatch(r"[A-Za-z0-9_]+", identifier) is None
    ):
        raise GateError("Invalid case identifier")
    op = case["op"]
    if op == "shape_construct":
        return {"id": identifier, "op": op, "shape": u32_list(case["request"]["shape"])}
    if "inputs" in case:
        inputs = {name: tensor_input(value) for name, value in case["inputs"].items()}
        parameters = case.get("parameters", {})
    else:
        request = case["request"]
        inputs = {"a": tensor_input(request)}
        if "dtype" in request:
            return {"id": identifier, "op": op, "compile_negative": request["dtype"]}
        if "b_shape" in request:
            inputs["b"] = tensor_input({"shape": request["b_shape"]})
        if "bias_shape" in request:
            inputs["bias"] = tensor_input({"shape": request["bias_shape"]})
        parameters = {
            key: request[key]
            for key in ("coordinate", "index", "dims", "axes")
            if key in request
        }
        if "target_shape" in request:
            parameters["shape"] = request["target_shape"]
    decoded_parameters = {
        key: u32(value) if key == "index" else u32_list(value)
        for key, value in parameters.items()
    }
    required = {
        "construct": (),
        "clone": (),
        "sum": (),
        "get": ("coordinate",),
        "get_flat": ("index",),
        "reshape": ("shape",),
        "permute_copy": ("axes",),
        "transpose_copy": ("dims",),
        "add": (),
        "sub": (),
        "mul": (),
        "matmul": (),
        "matmul_add": (),
    }
    if op not in required:
        raise GateError(f"Unknown operation: {op}")
    if any(key not in decoded_parameters for key in required[op]):
        raise GateError(f"Missing parameters for {identifier}")
    if op == "transpose_copy":
        dimensions = decoded_parameters["dims"]
        if not isinstance(dimensions, list) or len(dimensions) != 2:
            raise GateError("Transpose requires two dimensions")
    if op in ("add", "sub", "mul", "matmul", "matmul_add") and "b" not in inputs:
        raise GateError(f"Missing right input for {identifier}")
    if op == "matmul_add" and "bias" not in inputs:
        raise GateError(f"Missing bias input for {identifier}")
    return {
        "id": identifier,
        "op": op,
        "inputs": inputs,
        "parameters": decoded_parameters,
    }


def corpus_cases(corpus: dict) -> list[dict]:
    return [
        case
        for group in (*SUCCESS_GROUPS, "error_cases", "shape_cases")
        for case in corpus[group]
    ]


def load_corpus(path: Path = CORPUS) -> dict:
    if digest(path) != CORPUS_SHA256 or digest(POLICY) != POLICY_SHA256:
        raise GateError("Frozen tensor corpus or numerical policy hash mismatch")
    if (
        digest(ORDERED_REFERENCE) != ORDERED_REFERENCE_SHA256
        or digest(ORDERED_CONTROLS) != ORDERED_CONTROLS_SHA256
    ):
        raise GateError("Frozen tensor ordered reference or controls hash mismatch")
    corpus = json.loads(path.read_text())
    if corpus["policy_sha256"] != POLICY_SHA256:
        raise GateError("Corpus policy identity mismatch")
    if [len(corpus[group]) for group in SUCCESS_GROUPS] != [141, 3, 17]:
        raise GateError("Frozen success case inventory changed")
    if len(corpus["error_cases"]) != 45 or len(corpus["shape_cases"]) != 7:
        raise GateError("Frozen boundary case inventory changed")
    cases = corpus_cases(corpus)
    if len({case["id"] for case in cases}) != len(cases):
        raise GateError("Duplicate corpus case identifier")
    return corpus


def load_supplement(path: Path = SUPPLEMENT) -> dict:
    if digest(path) != SUPPLEMENT_SHA256 or digest(POLICY_V3) != POLICY_V3_SHA256:
        raise GateError("Tensor supplement or numerical policy v3 hash mismatch")
    supplement = json.loads(path.read_text())
    if (
        supplement["policy_sha256"] != POLICY_V3_SHA256
        or supplement["base_corpus"]["sha256"] != CORPUS_SHA256
    ):
        raise GateError("Supplement policy or base corpus identity mismatch")
    if len(supplement["cases"]) != 29 or len(supplement["error_cases"]) != 2:
        raise GateError("Frozen supplement inventory changed")
    identifiers = [case["id"] for case in supplement["cases"]] + [
        case["id"] for case in supplement["error_cases"]
    ]
    if len(set(identifiers)) != len(identifiers):
        raise GateError("Duplicate supplement case identifier")
    return supplement


def error_name(raw: str) -> str:
    domain, separator, code = raw.partition(":")
    if not separator:
        raise GateError(f"Malformed native error: {raw}")
    if domain == "matrix" and code in ("rank", "inner"):
        return "MatmulRankMismatch" if code == "rank" else "MatmulInnerMismatch"
    if domain not in ("core", "shape", "transform") or not code.isdecimal():
        raise GateError(f"Unknown native error: {raw}")
    number = int(code)
    if number not in ERROR_CODES:
        raise GateError(f"Unknown native error: {raw}")
    return ERROR_CODES[number]


def parse_native_payload(payload: list[str], identifier: str) -> dict:
    if len(payload) == 1 and payload[0].startswith("tensor:error:"):
        raw = payload[0].removeprefix("tensor:error:")
        return {"error": error_name(raw), "raw_error": raw}
    if len(payload) < 2 or not payload[0].startswith("tensor:shape:"):
        raise GateError(f"Missing native shape: {identifier}")
    try:
        dimensions = json.loads(payload[0].removeprefix("tensor:shape:"))
        if not isinstance(dimensions, list):
            raise GateError("Native shape is not a list")
        dimensions = u32_list(dimensions)
        if len(dimensions) > 4:
            raise GateError("Native output rank exceeds the API bound")
        count_row = payload[1]
        shape_only = count_row.startswith("tensor:numel:")
        prefix = "tensor:numel:" if shape_only else "tensor:count:"
        if not count_row.startswith(prefix):
            raise GateError("Missing native element count")
        count = u32(int(count_row.removeprefix(prefix)))
        logical_count = 1
        for dimension in dimensions:
            logical_count *= dimension
        if count != logical_count or count > 1 << 20:
            raise GateError("Native logical count disagrees with shape")
        if shape_only:
            if len(payload) != 2:
                raise GateError("Shape-only result contains tensor words")
            return {
                "shape_result": {
                    "shape": dimensions,
                    "rank": len(dimensions),
                    "numel": count,
                }
            }
        words = []
        for row in payload[2:]:
            if not row.startswith("tensor:word:"):
                raise GateError("Invalid native word record")
            words.append(f"{u32(int(row.removeprefix('tensor:word:'))):08x}")
        if len(words) != count:
            raise GateError("Missing or extra logical words, including padding")
    except (ValueError, TypeError) as error:
        raise GateError(f"Malformed native payload: {identifier}") from error
    return {"result": {"shape": dimensions, "f32_bits": words}}


def parse_native_output(text: str, case_ids: list[str] | None = None) -> list[dict]:
    rows = text.splitlines()
    results = []
    seen = set()
    index = 0
    while index < len(rows):
        begin = rows[index]
        index += 1
        if not begin.startswith("tensor:begin:"):
            raise GateError(f"Unframed native output: {begin!r}")
        identifier = begin.removeprefix("tensor:begin:")
        if not identifier or identifier in seen:
            raise GateError("Empty or duplicate native case identifier")
        seen.add(identifier)
        payload = []
        while index < len(rows) and rows[index] != "tensor:end":
            payload.append(rows[index])
            index += 1
        if index == len(rows):
            raise GateError(f"Truncated native case: {identifier}")
        index += 1
        retained = None
        construction = payload[:1] == ["tensor:boundary:construction"]
        if construction:
            payload = payload[1:]
        elif payload[:1] == ["tensor:binary:called"]:
            markers = ("tensor:retained:left", "tensor:retained:right", "tensor:output")
            if any(payload.count(marker) != 1 for marker in markers):
                raise GateError(f"Missing or duplicate retained section: {identifier}")
            left, right, output = (payload.index(marker) for marker in markers)
            if left != 1 or not left < right < output:
                raise GateError(f"Reordered retained sections: {identifier}")
            retained = {}
            for owner, section in (
                ("left", payload[left + 1 : right]),
                ("right", payload[right + 1 : output]),
            ):
                snapshot = parse_native_payload(section, identifier)
                if "result" not in snapshot:
                    raise GateError(f"Retained owner is not a tensor: {identifier}")
                retained[owner] = snapshot["result"]
            payload = payload[output + 1 :]
        observation = parse_native_payload(payload, identifier)
        if construction:
            if not observation.get("raw_error", "").startswith("core:"):
                raise GateError(
                    f"Construction boundary is not a core error: {identifier}"
                )
            observation["boundary"] = "construction"
        if retained is not None:
            if "shape_result" in observation:
                raise GateError(f"Binary output is shape-only: {identifier}")
            observation["retained"] = retained
        results.append({"id": identifier, **observation})
    if case_ids is not None and [result["id"] for result in results] != case_ids:
        raise GateError(
            "Native case inventory or order differs from generated requests"
        )
    return results


def input_source(encoded: dict) -> str:
    dimensions = json.dumps(encoded["shape"])
    if encoded.get("fill_zero"):
        return f"Harness.zeroes({dimensions})"
    values = ", ".join(
        f"Harness.from_bits({int(word, 16)})" for word in encoded["f32_bits"]
    )
    return f"Core.construct({dimensions}, [{values}])"


def case_source(case: dict) -> str:
    op = case["op"]
    if op == "shape_construct":
        return f"Harness.shape({json.dumps(case['shape'])})"
    inputs = case["inputs"]
    a = input_source(inputs["a"])
    parameters = case["parameters"]
    if op in ("add", "sub", "mul"):
        b = input_source(inputs["b"])
        return f"Harness.binary(Harness.{op.title()}{{}}, {a}, {b})"
    if op in ("matmul", "matmul_add"):
        b = input_source(inputs["b"])
        if op == "matmul_add":
            return f"Harness.biased({input_source(inputs['bias'])}, {a}, {b})"
        return f"Harness.matrix(None{{}}, {a}, {b})"
    operations = {
        "construct": "Construct{}",
        "clone": "Clone{}",
        "sum": "Sum{}",
        "get": f"Get{{{json.dumps(parameters.get('coordinate', []))}}}",
        "get_flat": f"GetFlat{{{parameters.get('index', 0)}}}",
        "reshape": f"Reshape{{{json.dumps(parameters.get('shape', []))}}}",
        "permute_copy": f"Permute{{{json.dumps(parameters.get('axes', []))}}}",
        "transpose_copy": "Transpose{"
        + ", ".join(map(str, parameters.get("dims", [])))
        + "}",
    }
    return f"Harness.unary(Harness.{operations[op]}, {a})"


def bend_program(cases: list[dict], directory: Path, source_root: Path = ROOT) -> str:
    imports = ["import Base"]
    for module, alias in (
        ("harness", "Harness"),
        ("core", "Core"),
        ("elementwise", "Elementwise"),
    ):
        relative = os.path.relpath(source_root / f"src/tensor/{module}.bend", directory)
        if not relative.startswith("."):
            relative = "./" + relative
        imports.append(f"import {relative} as {alias}")
    if not cases:
        raise GateError("Refusing to generate an empty native corpus")
    definitions = []
    for index, case in enumerate(cases):
        definitions.extend(
            [
                "",
                f"def run_case_{index}() -> IO(Unit):",
                "  do IO<Unit>:",
                f'    IO.print("tensor:begin:{case["id"]}")',
                "    " + case_source(case),
                '    IO.print("tensor:end")',
            ]
        )
    batches = list(range(0, len(cases), 16))
    for batch, start in enumerate(batches):
        definitions.extend(
            ["", f"def run_batch_{batch}() -> IO(Unit):", "  do IO<Unit>:"]
            + [
                f"    run_case_{index}()"
                for index in range(start, min(start + 16, len(cases)))
            ]
        )
    definitions.extend(
        ["", "def main() -> IO(Unit):", "  do IO<Unit>:"]
        + [f"    run_batch_{batch}()" for batch in range(len(batches))]
    )
    return "\n".join(imports + definitions) + "\n"


def negative_program(dtype: str, directory: Path, source_root: Path = ROOT) -> str:
    names = {
        "float64": "Float64",
        "float16": "Float16",
        "bfloat16": "BFloat16",
        "int32": "Int32",
        "int64": "Int64",
        "bool": "Boolean",
        "complex64": "Complex64",
    }
    if dtype not in names:
        raise GateError(f"Unknown unsupported dtype certificate: {dtype}")
    name = names[dtype]
    relative = os.path.relpath(source_root / "src/tensor/core.bend", directory)
    return (
        f"import Base\nimport {relative} as Core\n\n"
        f"type {name} is Data:\n  Other{{payload: U32}}\n\n"
        f"def rejected(values: List<&2, {name}>) -> "
        "Result<&2, &1, Core.Error, Core.Dynamic>:\n"
        "  Core.construct([1], values)\n"
    )


def copy_sources(output: Path) -> dict:
    pending = [ROOT / "src/tensor/harness.bend"]
    source_hashes = {}
    while pending:
        source = pending.pop().resolve()
        relative = str(source.relative_to(ROOT))
        if relative in source_hashes:
            continue
        content = source.read_bytes()
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_hashes[relative] = digest(target)
        pending.extend(
            source.parent / imported
            for imported in re.findall(
                r"^import (\S+\.bend) as ", content.decode(), re.MULTILINE
            )
        )
    for relative in (
        "tools/tensor_runtime.py",
        "tools/common.py",
        "tools/storage.py",
        "tools/tensor_ordered.py",
        "oracle/tensor/ordered_reference.py",
        "oracle/tensor/tests/test_ordered_reference.py",
        "tests/test_tensor_ordered.py",
        "tests/test_tensor_ordered_integration.py",
        "oracle/tensor/fixtures/ordered-reference.json",
        "oracle/tensor/fixtures/ordered-controls.json",
        "oracle/tensor/ORDERED_REFERENCE.md",
        "oracle/tensor/generate.py",
        "oracle/tensor/transpose_reference.py",
        "oracle/tensor/fixtures/scalar-transpose-correction.json",
        "tests/test_scalar_transpose.py",
        "docs/scalar-transpose.md",
        "verification/references.json",
    ):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return source_hashes


def constructs(encoded: dict) -> bool:
    shape = shape_oracle(encoded["shape"])
    return "count" in shape and (
        "fill_zero" in encoded or len(encoded["f32_bits"]) == shape["count"]
    )


def product_shape(request: dict) -> list[int] | None:
    """Output shape of the matmul a request runs, None if none runs."""
    if request.get("op") not in ("matmul", "matmul_add"):
        return None
    inputs = request["inputs"]
    if not all(constructs(encoded) for encoded in inputs.values()):
        return None
    left, right = inputs["a"]["shape"], inputs["b"]["shape"]
    if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
        return None
    product = [left[0], right[1]]
    return product if "count" in shape_oracle(product) else None


def built_cells(request: dict) -> int:
    """Output cells of the elementwise build a request runs, 0 if none runs."""
    op = request.get("op")
    if op not in ("add", "sub", "mul", "matmul_add"):
        return 0
    inputs = request["inputs"]
    if not all(constructs(encoded) for encoded in inputs.values()):
        return 0
    left, right = inputs["a"]["shape"], inputs["b"]["shape"]
    if op == "matmul_add":
        product = product_shape(request)
        if product is None:
            return 0
        left, right = product, inputs["bias"]["shape"]
    return broadcast_oracle(left, right).get("count", 0)


def product_cells(request: dict) -> int:
    """Output cells of the parallel matmul build a request runs, 0 if none."""
    product = product_shape(request)
    return 0 if product is None else shape_oracle(product)["count"]


def build_joins(cells: int) -> int:
    """`blk_node` calls of one elementwise or matmul build.

    The build joins every inner node of its balanced output tree once, by
    `ANode` in `fork` or `tree_join`. Storage of `cells` cells has depth
    `ceil(log2(cells))`, so the tree has `2^depth - 1` inner nodes.
    """
    return 0 if cells == 0 else (1 << (cells - 1).bit_length()) - 1


def expected_copy_audit(requests: list[dict]) -> tuple[int, int]:
    return 0, sum(
        build_joins(built_cells(request)) + build_joins(product_cells(request))
        for request in requests
    )


def check_copy_audit(stderr: str, requests: list[dict]) -> None:
    expected = expected_copy_audit(requests)
    found = re.findall(r"^STORAGE_COPY_AUDIT (\d+) (\d+)$", stderr, re.MULTILINE)
    if [tuple(map(int, row)) for row in found] != [expected]:
        raise GateError(
            "Native copy audit reported "
            + (", ".join(" ".join(row) for row in found) or "nothing")
            + f"; expected {expected[0]} splits and {expected[1]} joins, "
            "exactly the joins of the elementwise and matmul output builds"
        )


def build_native(evidence: Evidence, cases: list[dict], name: str) -> Path:
    program = evidence.directory / f"{name}.bend"
    program.write_text(
        bend_program(cases, evidence.directory, evidence.directory / "source")
    )
    for mode in ("--check-only", "--verdict"):
        evidence.run(
            f"{name}-{mode[2:]}",
            bend(evidence.tool_root, program, mode),
            contains="ALL PROOFS CHECK",
        )
    emitted = evidence.directory / f"{name}.c"
    evidence.run(f"{name}-c-export", bend(evidence.tool_root, program, "-o", emitted))
    executable = evidence.directory / name
    evidence.run(
        f"{name}-build",
        [
            str(evidence.tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(emitted),
            "-lpthread",
            "-lm",
            "-o",
            str(executable),
        ],
    )
    return executable


def native_command(executable: Path, threads: int | None = None) -> list[str]:
    return native(executable, threads=threads)


def run_native_corpus(
    tool_root: Path,
    output: Path,
    corpus_path: Path = CORPUS,
    *,
    benchmark: bool = False,
    javascript: bool = False,
    screening: bool = False,
) -> dict:
    tool_root, output = tool_root.resolve(), output.resolve()
    corpus = load_corpus(corpus_path)
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    freeze = {
        "corpus_sha256": digest(corpus_path),
        "policy_sha256": digest(POLICY),
        "ordered_reference_sha256": digest(ORDERED_REFERENCE),
        "ordered_controls_sha256": digest(ORDERED_CONTROLS),
    }
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    source_hashes = copy_sources(output)
    references = output / "references"
    references.mkdir()
    shutil.copyfile(corpus_path, references / corpus_path.name)
    shutil.copyfile(POLICY, references / "numerical-policy.json")
    identity = {
        "hostname": platform.node(),
        "workspace": str(ROOT),
        "threads": common.THREADS,
        "gpu": "off",
        "execution_mode": "screening" if screening else "source_frozen_run",
        "bendtt_sha256": digest(tool_root / "kernel/bendtt"),
        "clang_sha256": digest(tool_root / "llvm/usr/bin/clang-19"),
        "source_sha256": source_hashes,
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    transported, boundary_results, negatives = [], [], []
    for case in corpus_cases(corpus):
        try:
            request = decode_case(case)
        except U32Range as error:
            boundary_results.append(
                {
                    "id": case["id"],
                    "error": "U32Range",
                    "boundary": "decoder",
                    "diagnostic": str(error),
                }
            )
            continue
        if "compile_negative" in request:
            negatives.append(request)
        else:
            transported.append(request)
    (output / "requests.json").write_text(json.dumps(transported, indent=2) + "\n")
    (output / "decoder-results.json").write_text(
        json.dumps(boundary_results, indent=2) + "\n"
    )
    executable = build_native(evidence, transported, "corpus")
    native = evidence.run("corpus-native-run", native_command(executable))
    identifiers = [case["id"] for case in transported]
    results = parse_native_output(native.stdout, identifiers)
    (output / "native-results.json").write_text(json.dumps(results, indent=2) + "\n")
    emitted = output / "corpus.c"
    audited = output / "corpus-audit.c"
    audited.write_text(instrument_copy_audit(emitted.read_text()))
    audit_binary = output / "corpus-audit"
    evidence.run(
        "corpus-audit-build",
        [
            str(tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(audited),
            "-lpthread",
            "-lm",
            "-o",
            str(audit_binary),
        ],
    )
    audit = evidence.run("corpus-audit-run", native_command(audit_binary))
    check_copy_audit(audit.stderr, transported)
    if parse_native_output(audit.stdout, identifiers) != results:
        raise GateError("Copy-audit executable changed native observations")
    for request in negatives:
        name = request["id"]
        fixture = output / f"{name}.bend"
        fixture.write_text(
            negative_program(request["compile_negative"], output, output / "source")
        )
        for mode in ("--check-only", "--verdict"):
            evidence.run(
                f"{name}-{mode[2:]}",
                bend(tool_root, fixture, mode),
                expected_code=1,
                contains="expected : List<&2, F32>",
            )
        boundary_results.append(
            {
                "id": name,
                "compile_negative": request["compile_negative"],
                "boundary": "static_f32_type",
            }
        )
    by_id = {result["id"]: result for result in results + boundary_results}
    results = [by_id[case["id"]] for case in corpus_cases(corpus)]
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    controls = json.loads(
        (output / "source" / ORDERED_CONTROLS.relative_to(ROOT)).read_text()
    )["cases"]
    control_requests = [decode_case(case) for case in controls]
    control_ids = [case["id"] for case in controls]
    control_binary = build_native(evidence, control_requests, "ordered-controls")
    control_run = evidence.run(
        "ordered-controls-native-run", native_command(control_binary)
    )
    native_controls = parse_native_output(control_run.stdout, control_ids)
    control_acceptance = {
        "native": [
            verify_ordered_case(
                {"id": case["id"], "result": case["ordered_result"]}, row["result"]
            )
            for case, row in zip(controls, native_controls, strict=True)
        ]
    }
    (output / "ordered-controls-native-results.json").write_text(
        json.dumps(native_controls, indent=2) + "\n"
    )
    javascript_results = []
    javascript_failure = None
    if javascript:
        try:
            js_program = output / "corpus.js"
            evidence.run(
                "javascript-executable-build",
                bend(tool_root, output / "corpus.bend", "-o", js_program),
            )
            js = evidence.run(
                "javascript-executable-run", [bend(tool_root)[0], str(js_program)]
            )
            javascript_results = parse_native_output(js.stdout, identifiers)
            (output / "javascript-executable-results.json").write_text(
                json.dumps(javascript_results, indent=2) + "\n"
            )
            control_js = output / "ordered-controls.js"
            evidence.run(
                "ordered-controls-javascript-build",
                bend(tool_root, output / "ordered-controls.bend", "-o", control_js),
            )
            control_js_run = evidence.run(
                "ordered-controls-javascript-run", [bend(tool_root)[0], str(control_js)]
            )
            javascript_controls = parse_native_output(
                control_js_run.stdout, control_ids
            )
            (output / "ordered-controls-javascript-results.json").write_text(
                json.dumps(javascript_controls, indent=2) + "\n"
            )
            control_acceptance["javascript"] = [
                verify_ordered_case(
                    {"id": case["id"], "result": case["ordered_result"]}, row["result"]
                )
                for case, row in zip(controls, javascript_controls, strict=True)
            ]
        except GateError as error:
            javascript_failure = str(error)
    if benchmark:
        selected_ids = [
            "add_broadcast_3",
            "matmul_rectangular_0",
            "golden_ab",
            "golden_ab_plus_bias",
        ]
        selected = [request for request in transported if request["id"] in selected_ids]
        small = build_native(evidence, selected, "screen")
        screens = []
        for repetition in range(3):
            timing = output / f"screen-{repetition}.time.json"
            observed = evidence.run(
                f"screen-{repetition}",
                [
                    "/usr/bin/time",
                    "-f",
                    '{"elapsed_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M}',
                    "-o",
                    str(timing),
                    *native_command(small, threads=1),
                ],
            )
            if parse_native_output(
                observed.stdout, [case["id"] for case in selected]
            ) != [by_id[case["id"]] for case in selected]:
                raise GateError("Native screen changed corpus observations")
            measured = json.loads(timing.read_text())
            measured["driver_wall_seconds"] = evidence.records[-1]["seconds"]
            screens.append(measured)
        (output / "screen.json").write_text(
            json.dumps(
                {
                    "case_ids": [case["id"] for case in selected],
                    "inputs": {
                        case["id"]: {
                            name: value["shape"]
                            for name, value in case["inputs"].items()
                        }
                        for case in selected
                    },
                    "includes": (
                        "process startup, construction, computation and serialization"
                    ),
                    "runs": screens,
                },
                indent=2,
            )
            + "\n"
        )
    if (
        digest(corpus_path) != freeze["corpus_sha256"]
        or digest(POLICY) != freeze["policy_sha256"]
        or digest(ORDERED_REFERENCE) != freeze["ordered_reference_sha256"]
        or digest(ORDERED_CONTROLS) != freeze["ordered_controls_sha256"]
    ):
        raise GateError("Frozen references changed during native execution")
    changed_sources = [
        relative
        for relative, expected in source_hashes.items()
        if digest(ROOT / relative) != expected
    ]
    if changed_sources and not screening:
        raise GateError("Numerical source changed during native execution")
    summary = {
        "execution_mode": "screening" if screening else "source_frozen_run",
        "live_source_drift": changed_sources,
        "success_cases": 161,
        "observed_tensor_results": sum("result" in row for row in results),
        "native_error_cases": 34,
        "decoder_error_cases": 4,
        "compile_negative_cases": 7,
        "shape_cases": 7,
        "copy_audit": {
            "splits": 0,
            "elementwise_build_joins": sum(
                build_joins(built_cells(request)) for request in transported
            ),
            "matmul_build_joins": sum(
                build_joins(product_cells(request)) for request in transported
            ),
        },
        "numerical_acceptance": (
            "not performed; root must call tools.tensor_numerics.verify_case "
            "and tools.tensor_ordered.verify_ordered_lane"
        ),
        "freeze": freeze,
        "ordered_controls": control_acceptance,
        "javascript": (
            {"status": "failed", "diagnostic": javascript_failure}
            if javascript_failure
            else {
                "status": "completed" if javascript else "not_requested",
                "lane": "generated_bend_javascript_executable",
                "observed_tensor_results": sum(
                    "result" in row for row in javascript_results
                ),
                "observed_api_errors": sum(
                    "error" in row for row in javascript_results
                ),
                "observed_shape_results": sum(
                    "shape_result" in row for row in javascript_results
                ),
                "abi_export": "not established by this executable lane",
            }
        ),
    }
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    evidence.save()
    return {
        "results": results,
        "javascript_results": javascript_results,
        "summary": summary,
        "evidence_directory": str(output),
    }


def run_supplement(tool_root: Path, output: Path, path: Path = SUPPLEMENT) -> dict:
    tool_root, output = tool_root.resolve(), output.resolve()
    supplement = load_supplement(path)
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    source_hashes = copy_sources(output)
    requests = [
        decode_case(case) for case in supplement["cases"] + supplement["error_cases"]
    ]
    identifiers = [request["id"] for request in requests]
    executable = build_native(evidence, requests, "supplement")
    native = evidence.run("supplement-native-run", native_command(executable))
    results = {"native": parse_native_output(native.stdout, identifiers)}
    program = output / "supplement.js"
    evidence.run(
        "supplement-javascript-build",
        bend(tool_root, output / "supplement.bend", "-o", program),
    )
    js = evidence.run("supplement-javascript-run", [bend(tool_root)[0], str(program)])
    results["javascript"] = parse_native_output(js.stdout, identifiers)
    for lane, rows in results.items():
        (output / f"supplement-{lane}-results.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
    if digest(path) != SUPPLEMENT_SHA256 or any(
        digest(ROOT / relative) != expected
        for relative, expected in source_hashes.items()
    ):
        raise GateError("Supplement or numerical source changed during execution")
    evidence.save()
    return {"supplement": supplement, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen tensor corpus on native Bend CPU"
    )
    parser.add_argument("--tools", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--javascript", action="store_true")
    parser.add_argument("--screening", action="store_true")
    args = parser.parse_args()
    result = run_native_corpus(
        args.tools,
        args.output,
        benchmark=args.benchmark,
        javascript=args.javascript,
        screening=args.screening,
    )
    print(
        json.dumps(
            {"evidence_directory": result["evidence_directory"], **result["summary"]},
            indent=2,
        )
    )
    if result["summary"]["javascript"]["status"] == "failed":
        raise GateError(result["summary"]["javascript"]["diagnostic"])


if __name__ == "__main__":
    main()
