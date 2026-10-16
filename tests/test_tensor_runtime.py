import copy
import json
import platform
import tempfile
import unittest
from pathlib import Path

from tools import common
from tools.common import ROOT, Evidence, GateError, bend, digest
from tools.storage import instrument_copy_audit
from tools.tensor_numerics import special, verify_envelope
from tools.tensor_runtime import (
    CORPUS,
    CORPUS_SHA256,
    POLICY,
    POLICY_SHA256,
    SUCCESS_GROUPS,
    U32Range,
    bend_program,
    build_joins,
    built_cells,
    case_source,
    check_copy_audit,
    corpus_cases,
    decode_case,
    expected_copy_audit,
    load_corpus,
    native_command,
    negative_program,
    parse_native_output,
    permuted_cells,
    product_cells,
    run_native_corpus,
    u32,
)


class RuntimeAdapterTests(unittest.TestCase):
    def test_frozen_corpus_has_all_cases_and_words(self):
        corpus = load_corpus()
        self.assertEqual(digest(CORPUS), CORPUS_SHA256)
        self.assertEqual(digest(POLICY), POLICY_SHA256)
        self.assertEqual(len(corpus_cases(corpus)), 213)
        successes = [case for group in SUCCESS_GROUPS for case in corpus[group]]
        self.assertEqual(len(successes), 161)
        self.assertEqual(
            sum(
                len(tensor["f32_bits"])
                for case in successes
                for tensor in [*case["inputs"].values(), case["outputs"]["result"]]
            ),
            2921,
        )

    def test_mutated_corpus_is_rejected_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "corpus.json"
            mutated.write_bytes(CORPUS.read_bytes() + b"\n")
            original = mutated.read_bytes()
            with self.assertRaisesRegex(GateError, "hash mismatch"):
                load_corpus(mutated)
            self.assertEqual(mutated.read_bytes(), original)

    def test_u32_decoder_rejects_before_source_transport(self):
        self.assertEqual(u32(0), 0)
        self.assertEqual(u32(0xFFFFFFFF), 0xFFFFFFFF)
        for value in (-1, 1 << 32, True, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(U32Range):
                u32(value)
        corpus = load_corpus()
        rejected = []
        for case in corpus["error_cases"]:
            try:
                decode_case(case)
            except U32Range:
                rejected.append(case["id"])
        self.assertEqual(
            rejected,
            [
                "negative_dimension",
                "over_u32_dimension",
                "negative_coordinate",
                "transpose_negative_axis",
            ],
        )

    def test_decoder_keeps_coordinate_dims_axes_and_shape(self):
        corpus = load_corpus()
        for case in corpus["finite_cases"]:
            if case["op"] not in ("get", "transpose_copy", "permute_copy", "reshape"):
                continue
            decoded = decode_case(case)
            self.assertEqual(decoded["parameters"], case["parameters"])
            self.assertEqual(
                decoded["inputs"]["a"]["shape"], case["inputs"]["a"]["shape"]
            )
        case = next(
            case for case in corpus["error_cases"] if case["id"] == "reshape_count"
        )
        self.assertEqual(decode_case(case)["parameters"], {"shape": [5]})

    def test_frozen_binary_requests_select_public_arithmetic(self):
        corpus = load_corpus()
        for case in corpus_cases(corpus):
            if case["op"] not in ("add", "sub", "mul"):
                continue
            with self.subTest(case=case["id"]):
                source = bend_program([decode_case(case)], ROOT / ".evidence")
                self.assertIn(
                    f"Harness.binary(Harness.{case['op'].title()}{{}},", source
                )
                self.assertNotIn("Harness.Generic{", source)
                self.assertNotIn("Elementwise.binary(", source)

    def test_raw_f32_words_never_pass_through_host_float(self):
        case: dict = {
            "id": "all_bits",
            "op": "construct",
            "inputs": {
                "a": {
                    "shape": [4],
                    "f32_bits": ["80000000", "00000001", "7f800001", "ffffffff"],
                }
            },
        }
        original = copy.deepcopy(case)
        program = bend_program([decode_case(case)], ROOT / ".evidence")
        for word in case["inputs"]["a"]["f32_bits"]:
            self.assertIn(f"Harness.from_bits({int(word, 16)})", program)
        self.assertNotIn("outputs", program)
        self.assertEqual(case, original)
        for word in ("7fc0000", "7FC00000", "-0000001", "zzzzzzzz", 7):
            bad = copy.deepcopy(case)
            bad["inputs"]["a"]["f32_bits"] = [word]
            with self.assertRaisesRegex(GateError, "F32 input word"):
                decode_case(bad)

    def test_shape_only_and_dtype_cases_are_not_fabricated_tensor_errors(self):
        corpus = load_corpus()
        shape = decode_case(corpus["shape_cases"][3])
        source = bend_program([shape], ROOT / ".evidence")
        self.assertIn("Harness.shape([1048576])", source)
        self.assertNotIn("Core.construct(", source)
        for case in corpus["error_cases"]:
            if case["expected_error"] != "UnsupportedDtype":
                continue
            request = decode_case(case)
            self.assertEqual(request["compile_negative"], case["request"]["dtype"])
            self.assertNotIn("error", request)
            source = negative_program(request["compile_negative"], ROOT / ".evidence")
            self.assertIn("Core.construct([1], values)", source)
            self.assertNotIn("Fail{", source)

    def test_native_word_parser_preserves_all_payload_bits_and_scalar_shape(self):
        text = "\n".join(
            [
                "tensor:begin:scalar",
                "tensor:shape:[]",
                "tensor:count:1",
                "tensor:word:2147483648",
                "tensor:end",
                "tensor:begin:words",
                "tensor:shape:[3]",
                "tensor:count:3",
                "tensor:word:1",
                "tensor:word:2139095041",
                "tensor:word:4294967295",
                "tensor:end",
                "",
            ]
        )
        observed = parse_native_output(text, ["scalar", "words"])
        self.assertEqual(observed[0]["result"], {"shape": [], "f32_bits": ["80000000"]})
        self.assertEqual(
            observed[1]["result"]["f32_bits"], ["00000001", "7f800001", "ffffffff"]
        )

    def test_parser_accepts_empty_and_shape_only_without_padding(self):
        text = (
            "tensor:begin:empty\ntensor:shape:[2, 0, 3]\ntensor:count:0\ntensor:end\n"
            "tensor:begin:huge_zero\ntensor:shape:[4294967295,4294967295,4294967295,0]\n"
            "tensor:numel:0\ntensor:end\n"
            "tensor:begin:capacity\ntensor:shape:[1024,1024]\ntensor:numel:1048576\ntensor:end\n"
        )
        results = parse_native_output(text)
        self.assertEqual(results[0]["result"], {"shape": [2, 0, 3], "f32_bits": []})
        self.assertEqual(results[1]["shape_result"]["numel"], 0)
        self.assertEqual(results[2]["shape_result"]["numel"], 1048576)

    def test_error_parser_preserves_raw_code_and_maps_declared_semantics(self):
        for raw, expected in (
            ("core:8", "FlatIndexOutOfBounds"),
            ("transform:12", "InvalidPermutation"),
            ("transform:14", "InvalidPermutation"),
            ("transform:15", "ReshapeElementCountMismatch"),
            ("shape:2", "CapacityExceeded"),
            ("matrix:rank", "MatmulRankMismatch"),
            ("matrix:inner", "MatmulInnerMismatch"),
        ):
            rows = parse_native_output(
                f"tensor:begin:error\ntensor:error:{raw}\ntensor:end\n"
            )
            self.assertEqual(
                rows, [{"id": "error", "error": expected, "raw_error": raw}]
            )

    def test_binary_parser_preserves_actual_retained_owners_on_success_and_error(self):
        owners = (
            "tensor:binary:called\ntensor:retained:left\n"
            "tensor:shape:[4]\ntensor:count:4\ntensor:word:2147483648\n"
            "tensor:word:1\ntensor:word:2143289345\ntensor:word:4294967295\n"
            "tensor:retained:right\ntensor:shape:[]\ntensor:count:1\n"
            "tensor:word:1077936128\ntensor:output\n"
        )
        retained = {
            "left": {
                "shape": [4],
                "f32_bits": ["80000000", "00000001", "7fc00001", "ffffffff"],
            },
            "right": {"shape": [], "f32_bits": ["40400000"]},
        }
        for output, expected in (
            (
                "tensor:shape:[0]\ntensor:count:0\n",
                {"result": {"shape": [0], "f32_bits": []}},
            ),
            (
                "tensor:error:core:6\n",
                {"error": "IncompatibleBroadcast", "raw_error": "core:6"},
            ),
        ):
            with self.subTest(output=output):
                text = f"tensor:begin:binary\n{owners}{output}tensor:end\n"
                self.assertEqual(
                    parse_native_output(text),
                    [{"id": "binary", **expected, "retained": retained}],
                )

    def test_parser_distinguishes_pre_call_construction_errors_without_retention(self):
        for raw in ("core:1", "core:2", "core:10"):
            text = (
                "tensor:begin:construction\ntensor:boundary:construction\n"
                f"tensor:error:{raw}\ntensor:end\n"
            )
            row = parse_native_output(text)[0]
            self.assertEqual(row["boundary"], "construction")
            self.assertEqual(row["raw_error"], raw)
            self.assertNotIn("retained", row)
        for payload in (
            "tensor:shape:[0]\ntensor:count:0\n",
            "tensor:error:matrix:rank\n",
            "tensor:error:shape:1\n",
            "tensor:error:core:1\ntensor:error:core:1\n",
            "tensor:binary:called\ntensor:error:core:1\n",
        ):
            with self.subTest(payload=payload), self.assertRaises(GateError):
                parse_native_output(
                    f"tensor:begin:a\ntensor:boundary:construction\n{payload}tensor:end\n"
                )

    def test_binary_parser_rejects_missing_duplicate_swapped_and_corrupt_sections(self):
        left = "tensor:shape:[1]\ntensor:count:1\ntensor:word:1073741824\n"
        right = "tensor:shape:[1]\ntensor:count:1\ntensor:word:1077936128\n"
        output = "tensor:shape:[1]\ntensor:count:1\ntensor:word:1084227584\n"
        valid = (
            f"tensor:begin:a\ntensor:binary:called\ntensor:retained:left\n{left}"
            f"tensor:retained:right\n{right}tensor:output\n{output}tensor:end\n"
        )
        corruptions = (
            valid.replace("tensor:binary:called\n", ""),
            valid.replace("tensor:binary:called\n", "tensor:binary:called\n" * 2),
            valid.replace(f"tensor:retained:left\n{left}", ""),
            valid.replace(f"tensor:retained:right\n{right}", ""),
            valid.replace("tensor:retained:right\n", "tensor:retained:left\n"),
            valid.replace(
                "tensor:retained:right\n",
                f"tensor:retained:left\n{left}tensor:retained:right\n",
            ),
            valid.replace(
                f"tensor:retained:left\n{left}tensor:retained:right\n{right}",
                f"tensor:retained:right\n{right}tensor:retained:left\n{left}",
            ),
            valid.replace("tensor:output\n", ""),
            valid.replace("tensor:output\n", "tensor:output\n" * 2),
            valid.replace(left, left.replace("[1]", "[2]"), 1),
            valid.replace(left, left.replace("tensor:word:1073741824\n", ""), 1),
            valid.replace(left, left + "tensor:word:0\n", 1),
            valid.replace(left, left.replace("1073741824", "4294967296"), 1),
            valid.replace(left, "tensor:shape:[1]\ntensor:numel:1\n", 1),
            valid.replace(left, "tensor:error:core:8\n", 1),
            valid.replace(output, "tensor:shape:[1]\ntensor:numel:1\n"),
        )
        for text in corruptions:
            with self.subTest(text=text), self.assertRaises(GateError):
                parse_native_output(text)

    def test_parser_rejects_missing_duplicate_extra_and_malformed_observations(self):
        valid = (
            "tensor:begin:a\ntensor:shape:[1]\ntensor:count:1\n"
            "tensor:word:0\ntensor:end\n"
        )
        corruptions = (
            valid.replace("tensor:end\n", ""),
            valid + valid,
            "unframed\n" + valid,
            valid.replace("tensor:count:1", "tensor:count:2"),
            valid.replace("tensor:word:0\n", ""),
            valid.replace("tensor:word:0\n", "tensor:word:0\ntensor:word:0\n"),
            valid.replace("tensor:word:0", "tensor:word:-1"),
            valid.replace("tensor:word:0", "tensor:word:4294967296"),
            valid.replace("tensor:shape:[1]", "tensor:shape:[-1]"),
            valid.replace("tensor:shape:[1]", "tensor:shape:null"),
            valid.replace("tensor:word:0", "tensor:error:core:8"),
            "tensor:begin:a\ntensor:error:foreign:1\ntensor:end\n",
            "tensor:begin:a\ntensor:error:core:99\ntensor:end\n",
        )
        for text in corruptions:
            with self.subTest(text=text), self.assertRaises(GateError):
                parse_native_output(text)
        with self.assertRaisesRegex(GateError, "inventory"):
            parse_native_output(valid, ["a", "missing"])
        with self.assertRaisesRegex(GateError, "inventory"):
            parse_native_output("", ["missing"])

    def test_native_command_disables_gpu_and_sets_the_thread_count(self):
        self.assertEqual(
            native_command(Path("/native")),
            ["/native", "--threads", str(common.THREADS), "--gpu", "off"],
        )
        self.assertEqual(
            native_command(Path("/native"), threads=1),
            ["/native", "--threads", "1", "--gpu", "off"],
        )

    def test_generated_program_imports_the_preserved_source_snapshot(self):
        case = decode_case(load_corpus()["finite_cases"][0])
        directory = ROOT / ".evidence"
        program = bend_program([case], directory, directory / "source")
        self.assertIn("import ./source/src/tensor/harness.bend as Harness", program)
        self.assertIn("import ./source/src/tensor/core.bend as Core", program)

    def test_generated_io_batches_preserve_every_case_without_a_deep_main(self):
        cases = [decode_case(case) for case in load_corpus()["finite_cases"][:40]]
        program = bend_program(cases, ROOT / ".evidence")
        for case in cases:
            self.assertEqual(program.count(f'tensor:begin:{case["id"]}"'), 1)
        for index in range(40):
            self.assertEqual(program.count(f"    run_case_{index}()\n"), 1)
        main = program.split("def main() -> IO(Unit):", 1)[1]
        self.assertEqual(main.count("run_batch_"), 3)
        self.assertNotIn("run_case_", main)


def binary_request(op: str, *shapes: list[int]) -> dict:
    names = ("a", "b", "bias")
    return {
        "op": op,
        "inputs": {
            name: {"shape": shape, "fill_zero": True}
            for name, shape in zip(names, shapes, strict=False)
        },
    }


def transform_request(op: str, shape: list[int], key: str, axes: list[int]) -> dict:
    return {
        "op": op,
        "inputs": {"a": {"shape": shape, "fill_zero": True}},
        "parameters": {key: axes},
    }


class CopyAuditTests(unittest.TestCase):
    def test_build_joins_every_inner_node_of_the_balanced_output(self):
        for cells, joins in (
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 3),
            (15, 15),
            (21, 31),
            (32768, 32767),
            (32769, 65535),
            (1 << 20, (1 << 20) - 1),
        ):
            with self.subTest(cells=cells):
                self.assertEqual(build_joins(cells), joins)

    def test_built_cells_counts_only_outputs_the_build_writes(self):
        for request, cells in (
            (binary_request("add", [3], []), 3),
            (binary_request("mul", [2, 1, 3, 1], [1, 3]), 18),
            (binary_request("sub", [0, 3], [1, 3]), 0),
            (binary_request("add", [1 << 20], []), 1 << 20),
            (binary_request("add", [2, 3], [4, 3]), 0),
            (binary_request("add", [1024, 1], [1, 1025]), 0),
            (binary_request("add", [1, 1, 1, 1, 1], [1]), 0),
            (binary_request("add", [1], [1048577]), 0),
            (
                {
                    "op": "add",
                    "inputs": {
                        "a": {"shape": [2], "f32_bits": ["00000000"]},
                        "b": {"shape": [1], "fill_zero": True},
                    },
                },
                0,
            ),
            (binary_request("matmul_add", [2, 3], [3, 2], [2]), 4),
            (binary_request("matmul_add", [2, 3], [4, 2], [2]), 0),
            (binary_request("matmul", [2, 3], [3, 2]), 0),
            (binary_request("clone", [4]), 0),
        ):
            with self.subTest(request=request):
                self.assertEqual(built_cells(request), cells)

    def test_product_cells_counts_only_products_the_build_writes(self):
        for request, cells in (
            (binary_request("matmul", [2, 3], [3, 2]), 4),
            (binary_request("matmul", [2, 0], [0, 3]), 6),
            (binary_request("matmul", [0, 3], [3, 2]), 0),
            (binary_request("matmul", [2, 3], [4, 2]), 0),
            (binary_request("matmul", [2, 3, 1], [3, 2]), 0),
            (binary_request("matmul", [3], [3, 2]), 0),
            (binary_request("matmul_add", [2, 3], [3, 2], [2]), 4),
            (binary_request("matmul_add", [2, 3], [4, 2], [2]), 0),
            (binary_request("matmul", [1024, 2048], [2048, 1025]), 0),
            (binary_request("add", [3], []), 0),
        ):
            with self.subTest(request=request):
                self.assertEqual(product_cells(request), cells)

    def test_permuted_cells_counts_only_outputs_the_build_writes(self):
        for request, cells in (
            (transform_request("permute_copy", [2, 3], "axes", [1, 0]), 6),
            (transform_request("permute_copy", [2, 3, 4], "axes", [2, 0, 1]), 24),
            (transform_request("permute_copy", [], "axes", []), 1),
            (transform_request("permute_copy", [2, 3], "axes", [0]), 0),
            (transform_request("permute_copy", [2, 3], "axes", [0, 0]), 0),
            (transform_request("permute_copy", [2, 3], "axes", [0, 2]), 0),
            (transform_request("permute_copy", [0, 3], "axes", [1, 0]), 0),
            (transform_request("transpose_copy", [2, 3], "dims", [0, 1]), 6),
            (transform_request("transpose_copy", [2, 3], "dims", [1, 1]), 6),
            (transform_request("transpose_copy", [2, 3], "dims", [0, 2]), 0),
            (transform_request("transpose_copy", [], "dims", [0, 0]), 1),
            (transform_request("transpose_copy", [], "dims", [1, 0]), 0),
            (binary_request("add", [3], [3]), 0),
        ):
            with self.subTest(request=request):
                self.assertEqual(permuted_cells(request), cells)

    def test_matmul_add_joins_both_builds(self):
        request = binary_request("matmul_add", [2, 3], [3, 3], [3])
        self.assertEqual(expected_copy_audit([request]), (0, 7 + 7))

    def test_frozen_corpus_expects_842_joins_and_no_split(self):
        transported = []
        for case in corpus_cases(load_corpus()):
            try:
                request = decode_case(case)
            except U32Range:
                continue
            if "compile_negative" not in request:
                transported.append(request)
        self.assertEqual(len(transported), 202)
        self.assertEqual(expected_copy_audit(transported), (0, 842))

    def test_audit_rejects_any_split_or_join_beyond_the_builds(self):
        requests = [binary_request("add", [3], []), binary_request("sum", [4])]
        check_copy_audit("STORAGE_COPY_AUDIT 0 3\n", requests)
        for stderr in (
            "STORAGE_COPY_AUDIT 0 4\n",
            "STORAGE_COPY_AUDIT 0 2\n",
            "STORAGE_COPY_AUDIT 1 3\n",
            "STORAGE_COPY_AUDIT 0 3\nSTORAGE_COPY_AUDIT 0 3\n",
            "",
        ):
            with self.subTest(stderr=stderr):
                with self.assertRaises(GateError):
                    check_copy_audit(stderr, requests)


class NativeCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT / ".evidence"
        parent.mkdir(parents=True, exist_ok=True)
        cls.directory = Path(
            tempfile.mkdtemp(prefix="tensor-runtime-tests-", dir=parent)
        )
        cls.execution = run_native_corpus(
            ROOT / ".tools",
            cls.directory / "run",
            javascript=True,
            screening=True,
        )
        cls.corpus = load_corpus()
        cls.results = {row["id"]: row for row in cls.execution["results"]}
        cls.javascript_results = {
            row["id"]: row for row in cls.execution["javascript_results"]
        }
        entry_cases = [
            decode_case(
                {
                    "id": f"generic_{op}",
                    "op": op,
                    "inputs": {
                        "a": {"shape": [2], "f32_bits": ["40000000", "40a00000"]},
                        "b": {"shape": [2], "f32_bits": ["40400000", "40800000"]},
                    },
                }
            )
            for op in ("add", "sub", "mul")
        ]
        mismatch = next(
            case
            for case in cls.corpus["error_cases"]
            if case["id"] == "broadcast_mismatch"
        )
        entry_cases.append(
            decode_case({**mismatch, "id": "generic_broadcast_mismatch"})
        )
        for name, invalid in (
            ("rank", {"shape": [1, 1, 1, 1, 1]}),
            ("capacity", {"shape": [1048577]}),
            ("length", {"shape": [2], "f32_bits": ["00000000"]}),
        ):
            for side in ("a", "b"):
                inputs = {"a": {"shape": [1]}, "b": {"shape": [1]}}
                inputs[side] = invalid
                entry_cases.append(
                    decode_case(
                        {
                            "id": f"construction_{side}_{name}",
                            "op": "add",
                            "inputs": inputs,
                        }
                    )
                )
        evidence = Evidence(cls.directory / "entry-points", ROOT / ".tools")
        program = evidence.directory / "entry-points.bend"
        source = bend_program(
            entry_cases, evidence.directory, cls.directory / "run/source"
        )
        for request in entry_cases[:4]:
            public = case_source(request)
            generic = public.replace(
                f"Harness.{request['op'].title()}{{}}",
                f"Harness.Generic{{Elementwise.{request['op'].title()}{{}}}}",
                1,
            )
            source = source.replace(public, generic)
        program.write_text(source)
        for mode in ("--check-only", "--verdict"):
            evidence.run(
                f"entry-points-{mode[2:]}",
                bend(evidence.tool_root, program, mode),
                contains="ALL PROOFS CHECK",
            )
        emitted = evidence.directory / "entry-points.c"
        evidence.run(
            "entry-points-c-export", bend(evidence.tool_root, program, "-o", emitted)
        )
        audited = evidence.directory / "entry-points-audit.c"
        audited.write_text(instrument_copy_audit(emitted.read_text()))
        executable = evidence.directory / "entry-points"
        evidence.run(
            "entry-points-build",
            [
                str(evidence.tool_root / "llvm/usr/bin/clang-19"),
                "-std=c11",
                "-O3",
                str(audited),
                "-lpthread",
                "-lm",
                "-o",
                str(executable),
            ],
        )
        native = evidence.run("entry-points-native-run", native_command(executable))
        check_copy_audit(native.stderr, entry_cases)
        javascript = evidence.directory / "entry-points.js"
        evidence.run(
            "entry-points-js-export",
            bend(evidence.tool_root, program, "-o", javascript),
        )
        js = evidence.run(
            "entry-points-js-run", [bend(evidence.tool_root)[0], str(javascript)]
        )
        identifiers = [request["id"] for request in entry_cases]
        cls.entry_results = {
            "native": {
                row["id"]: row
                for row in parse_native_output(native.stdout, identifiers)
            },
            "javascript": {
                row["id"]: row for row in parse_native_output(js.stdout, identifiers)
            },
        }

    def test_complete_native_inventory_has_tensor_results_for_every_success(self):
        self.assertEqual(len(self.results), 213)
        for group in SUCCESS_GROUPS:
            for case in self.corpus[group]:
                with self.subTest(case=case["id"]):
                    actual = self.results[case["id"]]["result"]
                    self.assertEqual(
                        actual["shape"], case["outputs"]["result"]["shape"]
                    )
                    self.assertEqual(
                        len(actual["f32_bits"]),
                        len(case["outputs"]["result"]["f32_bits"]),
                    )

    def test_public_binary_outputs_pass_frozen_numerical_acceptance_in_both_lanes(self):
        for lane, observed in (
            ("native", self.results),
            ("javascript", self.javascript_results),
        ):
            for group in SUCCESS_GROUPS:
                for case in self.corpus[group]:
                    if case["op"] not in ("add", "sub", "mul"):
                        continue
                    with self.subTest(lane=lane, case=case["id"]):
                        verify_envelope(case, observed[case["id"]]["result"])

    def test_public_binary_calls_retain_corresponding_inputs_even_on_error(self):
        for lane, observed in (
            ("native", self.results),
            ("javascript", self.javascript_results),
        ):
            classified = next(
                case
                for case in self.corpus["nonfinite_cases"]
                if case["id"] == "construct_all_classifications"
            )
            constructed_words = dict(
                zip(
                    classified["inputs"]["a"]["f32_bits"],
                    observed[classified["id"]]["result"]["f32_bits"],
                    strict=True,
                )
            )
            for case in corpus_cases(self.corpus):
                if case["op"] not in ("add", "sub", "mul"):
                    continue
                request = decode_case(case)
                row = observed[case["id"]]
                with self.subTest(lane=lane, case=case["id"]):
                    self.assertNotIn("boundary", row)
                    self.assertEqual(set(row["retained"]), {"left", "right"})
                    for owner, name in (("left", "a"), ("right", "b")):
                        original = request["inputs"][name]
                        actual = row["retained"][owner]
                        self.assertEqual(actual["shape"], original["shape"])
                        if original.get("fill_zero"):
                            self.assertTrue(
                                all(word == "00000000" for word in actual["f32_bits"])
                            )
                        else:
                            self.assertEqual(
                                len(actual["f32_bits"]), len(original["f32_bits"])
                            )
                            for word, expected in zip(
                                actual["f32_bits"], original["f32_bits"], strict=True
                            ):
                                if special(expected) == "nan":
                                    self.assertEqual(special(word), "nan")
                                    self.assertEqual(word, constructed_words[expected])
                                else:
                                    self.assertEqual(word, expected)

    def test_generic_binary_entry_is_executable_and_discriminates_operations(self):
        for lane, observed in self.entry_results.items():
            for op, words in (
                ("add", ["40a00000", "41100000"]),
                ("sub", ["bf800000", "3f800000"]),
                ("mul", ["40c00000", "41a00000"]),
            ):
                with self.subTest(lane=lane, op=op):
                    row = observed[f"generic_{op}"]
                    self.assertEqual(row["result"], {"shape": [2], "f32_bits": words})
                    self.assertEqual(
                        row["retained"],
                        {
                            "left": {
                                "shape": [2],
                                "f32_bits": ["40000000", "40a00000"],
                            },
                            "right": {
                                "shape": [2],
                                "f32_bits": ["40400000", "40800000"],
                            },
                        },
                    )
            mismatch = observed["generic_broadcast_mismatch"]
            self.assertEqual(mismatch["error"], "IncompatibleBroadcast")
            self.assertEqual(mismatch["raw_error"], "core:6")
            self.assertEqual(
                mismatch["retained"], self.results["broadcast_mismatch"]["retained"]
            )
            self.assertNotIn("boundary", mismatch)

    def test_constructor_failures_do_not_claim_a_binary_call_or_retained_owners(self):
        for lane, observed in self.entry_results.items():
            for side in ("a", "b"):
                for name, error in (
                    ("rank", "RankExceeded"),
                    ("capacity", "CapacityExceeded"),
                    ("length", "DataLengthMismatch"),
                ):
                    with self.subTest(lane=lane, side=side, failure=name):
                        row = observed[f"construction_{side}_{name}"]
                        self.assertEqual(row["error"], error)
                        self.assertEqual(row["boundary"], "construction")
                        self.assertNotIn("retained", row)

    def test_native_errors_decoder_errors_and_static_certificates_are_distinct(self):
        for case in self.corpus["error_cases"]:
            with self.subTest(case=case["id"]):
                observed = self.results[case["id"]]
                if case["expected_error"] == "UnsupportedDtype":
                    self.assertEqual(observed["boundary"], "static_f32_type")
                    self.assertNotIn("error", observed)
                    for mode in ("check-only", "verdict"):
                        diagnostic = (
                            self.directory / "run" / f"{case['id']}-{mode}.stderr"
                        ).read_text()
                        self.assertIn("expected : List<&2, F32>", diagnostic)
                else:
                    self.assertEqual(observed["error"], case["expected_error"])
                    if case["expected_error"] == "U32Range":
                        self.assertEqual(observed["boundary"], "decoder")
                    else:
                        self.assertIn("raw_error", observed)

    def test_shape_only_boundaries_are_native_and_storage_free(self):
        for case in self.corpus["shape_cases"]:
            self.assertEqual(self.results[case["id"]]["shape_result"], case["expected"])

    def test_generated_js_executable_has_all_tensor_and_api_boundary_results(self):
        summary = self.execution["summary"]
        self.assertEqual(summary["javascript"]["status"], "completed")
        self.assertEqual(summary["javascript"]["observed_tensor_results"], 161)
        self.assertEqual(summary["javascript"]["observed_api_errors"], 34)
        self.assertEqual(summary["javascript"]["observed_shape_results"], 7)
        self.assertEqual(
            summary["javascript"]["abi_export"],
            "not established by this executable lane",
        )
        self.assertEqual(len(self.javascript_results), 202)
        for group in SUCCESS_GROUPS:
            for case in self.corpus[group]:
                with self.subTest(case=case["id"]):
                    actual = self.javascript_results[case["id"]]["result"]
                    self.assertEqual(
                        actual["shape"], case["outputs"]["result"]["shape"]
                    )
                    self.assertEqual(
                        len(actual["f32_bits"]),
                        len(case["outputs"]["result"]["f32_bits"]),
                    )
        for case in self.corpus["error_cases"]:
            if case["expected_error"] in ("UnsupportedDtype", "U32Range"):
                self.assertNotIn(case["id"], self.javascript_results)
                continue
            self.assertEqual(
                self.javascript_results[case["id"]]["error"], case["expected_error"]
            )
        for case in self.corpus["shape_cases"]:
            self.assertEqual(
                self.javascript_results[case["id"]]["shape_result"], case["expected"]
            )

    def test_native_copy_audit_counts_only_build_joins_and_evidence_is_hashed(self):
        evidence = self.directory / "run"
        self.assertEqual(
            (evidence / "corpus-audit-run.stderr").read_text(),
            "STORAGE_COPY_AUDIT 0 842\n",
        )
        report = json.loads((evidence / "report.json").read_text())
        self.assertEqual(report["hostname"], platform.node())
        self.assertTrue(all(command["passed"] for command in report["commands"]))
        for relative, expected in report["artifacts"].items():
            self.assertEqual(digest(evidence / relative), expected)
        self.assertIn("source/src/tensor/harness.bend", report["artifacts"])
        self.assertIn("corpus.c", report["artifacts"])
        self.assertIn("corpus-native-run.stdout", report["artifacts"])
        self.assertIn("corpus.js", report["artifacts"])
        self.assertIn("javascript-executable-run.stdout", report["artifacts"])
        self.assertIn("javascript-executable-results.json", report["artifacts"])


if __name__ == "__main__":
    unittest.main()
