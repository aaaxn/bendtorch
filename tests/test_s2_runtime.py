import copy
import json
import platform
import tempfile
import unittest
from pathlib import Path

from tools.common import ROOT, Evidence, GateError, bend, digest
from tools.s1_storage import instrument_copy_audit
from tools.s2_numerics import special, verify_case
from tools.s2_runtime import (
    CORPUS,
    CORPUS_SHA256,
    POLICY,
    POLICY_SHA256,
    SUCCESS_GROUPS,
    U32Range,
    bend_program,
    case_source,
    corpus_cases,
    decode_case,
    load_corpus,
    native_command,
    negative_program,
    parse_native_output,
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
                "S2:begin:scalar",
                "S2:shape:[]",
                "S2:count:1",
                "S2:word:2147483648",
                "S2:end",
                "S2:begin:words",
                "S2:shape:[3]",
                "S2:count:3",
                "S2:word:1",
                "S2:word:2139095041",
                "S2:word:4294967295",
                "S2:end",
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
            "S2:begin:empty\nS2:shape:[2, 0, 3]\nS2:count:0\nS2:end\n"
            "S2:begin:huge_zero\nS2:shape:[4294967295,4294967295,4294967295,0]\n"
            "S2:numel:0\nS2:end\n"
            "S2:begin:capacity\nS2:shape:[1024,1024]\nS2:numel:1048576\nS2:end\n"
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
            rows = parse_native_output(f"S2:begin:error\nS2:error:{raw}\nS2:end\n")
            self.assertEqual(
                rows, [{"id": "error", "error": expected, "raw_error": raw}]
            )

    def test_binary_parser_preserves_actual_retained_owners_on_success_and_error(self):
        owners = (
            "S2:binary:called\nS2:retained:left\n"
            "S2:shape:[4]\nS2:count:4\nS2:word:2147483648\n"
            "S2:word:1\nS2:word:2143289345\nS2:word:4294967295\n"
            "S2:retained:right\nS2:shape:[]\nS2:count:1\n"
            "S2:word:1077936128\nS2:output\n"
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
                "S2:shape:[0]\nS2:count:0\n",
                {"result": {"shape": [0], "f32_bits": []}},
            ),
            (
                "S2:error:core:6\n",
                {"error": "IncompatibleBroadcast", "raw_error": "core:6"},
            ),
        ):
            with self.subTest(output=output):
                text = f"S2:begin:binary\n{owners}{output}S2:end\n"
                self.assertEqual(
                    parse_native_output(text),
                    [{"id": "binary", **expected, "retained": retained}],
                )

    def test_parser_distinguishes_pre_call_construction_errors_without_retention(self):
        for raw in ("core:1", "core:2", "core:10"):
            text = (
                "S2:begin:construction\nS2:boundary:construction\n"
                f"S2:error:{raw}\nS2:end\n"
            )
            row = parse_native_output(text)[0]
            self.assertEqual(row["boundary"], "construction")
            self.assertEqual(row["raw_error"], raw)
            self.assertNotIn("retained", row)
        for payload in (
            "S2:shape:[0]\nS2:count:0\n",
            "S2:error:matrix:rank\n",
            "S2:error:shape:1\n",
            "S2:error:core:1\nS2:error:core:1\n",
            "S2:binary:called\nS2:error:core:1\n",
        ):
            with self.subTest(payload=payload), self.assertRaises(GateError):
                parse_native_output(
                    f"S2:begin:a\nS2:boundary:construction\n{payload}S2:end\n"
                )

    def test_binary_parser_rejects_missing_duplicate_swapped_and_corrupt_sections(self):
        left = "S2:shape:[1]\nS2:count:1\nS2:word:1073741824\n"
        right = "S2:shape:[1]\nS2:count:1\nS2:word:1077936128\n"
        output = "S2:shape:[1]\nS2:count:1\nS2:word:1084227584\n"
        valid = (
            f"S2:begin:a\nS2:binary:called\nS2:retained:left\n{left}"
            f"S2:retained:right\n{right}S2:output\n{output}S2:end\n"
        )
        corruptions = (
            valid.replace("S2:binary:called\n", ""),
            valid.replace("S2:binary:called\n", "S2:binary:called\n" * 2),
            valid.replace(f"S2:retained:left\n{left}", ""),
            valid.replace(f"S2:retained:right\n{right}", ""),
            valid.replace("S2:retained:right\n", "S2:retained:left\n"),
            valid.replace(
                "S2:retained:right\n", f"S2:retained:left\n{left}S2:retained:right\n"
            ),
            valid.replace(
                f"S2:retained:left\n{left}S2:retained:right\n{right}",
                f"S2:retained:right\n{right}S2:retained:left\n{left}",
            ),
            valid.replace("S2:output\n", ""),
            valid.replace("S2:output\n", "S2:output\n" * 2),
            valid.replace(left, left.replace("[1]", "[2]"), 1),
            valid.replace(left, left.replace("S2:word:1073741824\n", ""), 1),
            valid.replace(left, left + "S2:word:0\n", 1),
            valid.replace(left, left.replace("1073741824", "4294967296"), 1),
            valid.replace(left, "S2:shape:[1]\nS2:numel:1\n", 1),
            valid.replace(left, "S2:error:core:8\n", 1),
            valid.replace(output, "S2:shape:[1]\nS2:numel:1\n"),
        )
        for text in corruptions:
            with self.subTest(text=text), self.assertRaises(GateError):
                parse_native_output(text)

    def test_parser_rejects_missing_duplicate_extra_and_malformed_observations(self):
        valid = "S2:begin:a\nS2:shape:[1]\nS2:count:1\nS2:word:0\nS2:end\n"
        corruptions = (
            valid.replace("S2:end\n", ""),
            valid + valid,
            "unframed\n" + valid,
            valid.replace("S2:count:1", "S2:count:2"),
            valid.replace("S2:word:0\n", ""),
            valid.replace("S2:word:0\n", "S2:word:0\nS2:word:0\n"),
            valid.replace("S2:word:0", "S2:word:-1"),
            valid.replace("S2:word:0", "S2:word:4294967296"),
            valid.replace("S2:shape:[1]", "S2:shape:[-1]"),
            valid.replace("S2:shape:[1]", "S2:shape:null"),
            valid.replace("S2:word:0", "S2:error:core:8"),
            "S2:begin:a\nS2:error:foreign:1\nS2:end\n",
            "S2:begin:a\nS2:error:core:99\nS2:end\n",
        )
        for text in corruptions:
            with self.subTest(text=text), self.assertRaises(GateError):
                parse_native_output(text)
        with self.assertRaisesRegex(GateError, "inventory"):
            parse_native_output(valid, ["a", "missing"])
        with self.assertRaisesRegex(GateError, "inventory"):
            parse_native_output("", ["missing"])

    def test_native_command_always_disables_gpu_and_sets_one_thread(self):
        self.assertEqual(
            native_command(Path("/native")),
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
            self.assertEqual(program.count(f'S2:begin:{case["id"]}"'), 1)
        for index in range(40):
            self.assertEqual(program.count(f"    run_case_{index}()\n"), 1)
        main = program.split("def main() -> IO(Unit):", 1)[1]
        self.assertEqual(main.count("run_batch_"), 3)
        self.assertNotIn("run_case_", main)


class NativeCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT / ".evidence"
        parent.mkdir(parents=True, exist_ok=True)
        cls.directory = Path(tempfile.mkdtemp(prefix="s2-runtime-tests-", dir=parent))
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
        if native.stderr != "STORAGE_COPY_AUDIT 0 0\n":
            raise GateError("Entry-point checks copied live packed arrays")
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
                        verify_case(case, observed[case["id"]]["result"])

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

    def test_native_copy_audit_is_zero_and_evidence_is_hash_addressed(self):
        evidence = self.directory / "run"
        self.assertEqual(
            (evidence / "corpus-audit-run.stderr").read_text(),
            "STORAGE_COPY_AUDIT 0 0\n",
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
