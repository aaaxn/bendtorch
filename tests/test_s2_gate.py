import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.common import ROOT, Evidence, GateError
from tools.s2 import certificate_controls, verify_lane
from tools.s2_runtime import (
    ERROR_CODES,
    SUCCESS_GROUPS,
    corpus_cases,
    decode_case,
    load_corpus,
)


class StageAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus()
        success = {
            case["id"]: {"id": case["id"], "result": case["outputs"]["result"]}
            for group in SUCCESS_GROUPS
            for case in cls.corpus[group]
        }
        cls.native: list[dict] = []
        cls.javascript: list[dict] = []
        for case in corpus_cases(cls.corpus):
            identifier = case["id"]
            shared = False
            if identifier in success:
                row = success[identifier]
            elif "expected" in case:
                row = {"id": identifier, "shape_result": case["expected"]}
            else:
                category = case["expected_error"]
                if category == "UnsupportedDtype":
                    shared = True
                    row = {
                        "id": identifier,
                        "boundary": "static_f32_type",
                        "compile_negative": case["request"]["dtype"],
                    }
                elif category == "U32Range":
                    shared = True
                    row = {
                        "id": identifier,
                        "boundary": "decoder",
                        "error": category,
                    }
                else:
                    if category == "MatmulRankMismatch":
                        raw = "matrix:rank"
                    elif category == "MatmulInnerMismatch":
                        raw = "matrix:inner"
                    else:
                        code = next(
                            key
                            for key, value in ERROR_CODES.items()
                            if value == category
                        )
                        raw = f"core:{code}"
                    row = {"id": identifier, "error": category, "raw_error": raw}
            if case["op"] in ("add", "sub", "mul") and not shared:
                inputs = decode_case(case)["inputs"]
                row["retained"] = {
                    owner: {
                        "shape": inputs[name]["shape"],
                        "f32_bits": inputs[name].get(
                            "f32_bits", ["00000000"] * math.prod(inputs[name]["shape"])
                        ),
                    }
                    for owner, name in (("left", "a"), ("right", "b"))
                }
            cls.native.append(copy.deepcopy(row))
            if not shared:
                cls.javascript.append(copy.deepcopy(row))

    def test_required_inventory_covers_every_aggregated_tensor_law(self):
        required = json.loads((ROOT / "verification/s2.json").read_text())[
            "required_exports"
        ]
        laws = set()
        for line in (ROOT / "src/tensor/LAWS.bend").read_text().splitlines():
            if not line.startswith("import ./"):
                continue
            path = ROOT / "src/tensor" / line.split()[1]
            module = path.relative_to(ROOT / "src/tensor").with_suffix("")
            prefix = (
                "_2e_._2f_.._2f_src_2f_tensor_2f_"
                + module.as_posix().replace("-", "_2d_").replace("/", "_2f_")
                + "."
            )
            laws.update(
                prefix + declaration.removeprefix("law ").split(":", 1)[0]
                for declaration in path.read_text().splitlines()
                if declaration.startswith("law ")
            )
        self.assertTrue(laws)
        self.assertLessEqual(laws, set(required))
        self.assertEqual(len(required), len(set(required)))

    def test_complete_reference_observations_satisfy_each_lane(self):
        for native, rows in ((True, self.native), (False, self.javascript)):
            with self.subTest(native=native):
                acceptance = verify_lane(self.corpus, rows, native=native)
                self.assertEqual(len(acceptance["numerical_cases"]), 161)
                self.assertEqual(len(acceptance["errors"]), 45 if native else 34)
                self.assertEqual(len(acceptance["shape_cases"]), 7)
                self.assertEqual(len(acceptance["retained_binary"]), 67)

    def test_retained_operands_are_required_and_exact_on_success_and_error(self):
        for native, observations in ((True, self.native), (False, self.javascript)):
            for outcome in ("result", "error"):
                original = next(
                    row
                    for row in observations
                    if outcome in row
                    and "retained" in row
                    and row["retained"]["left"] != row["retained"]["right"]
                    and row["retained"]["left"]["f32_bits"]
                )
                for mutation in ("missing", "swapped", "shape", "word", "length"):
                    rows = copy.deepcopy(observations)
                    row = next(row for row in rows if row["id"] == original["id"])
                    if mutation == "missing":
                        del row["retained"]
                    elif mutation == "swapped":
                        row["retained"]["left"], row["retained"]["right"] = (
                            row["retained"]["right"],
                            row["retained"]["left"],
                        )
                    elif mutation == "shape":
                        row["retained"]["left"]["shape"] = [0, 2]
                    elif mutation == "word":
                        row["retained"]["left"]["f32_bits"][0] = "40000000"
                    else:
                        row["retained"]["left"]["f32_bits"].append("00000000")
                    with (
                        self.subTest(native=native, outcome=outcome, mutation=mutation),
                        self.assertRaisesRegex(GateError, "retained"),
                    ):
                        verify_lane(self.corpus, rows, native=native)

    def test_nan_retention_uses_matching_same_lane_construction_words(self):
        rows = copy.deepcopy(self.javascript)
        construction = next(
            row for row in rows if row["id"] == "construct_all_classifications"
        )
        original_nan_words = {"7fc01234", "ffc05678", "7f800001"}
        construction["result"]["f32_bits"] = [
            "7fc00000" if word in original_nan_words else word
            for word in construction["result"]["f32_bits"]
        ]
        for row in rows:
            for owner in row.get("retained", {}).values():
                owner["f32_bits"] = [
                    "7fc00000" if word in original_nan_words else word
                    for word in owner["f32_bits"]
                ]
        verify_lane(self.corpus, rows, native=False)
        row = next(
            row
            for row in rows
            if any(
                "7fc00000" in owner["f32_bits"]
                for owner in row.get("retained", {}).values()
            )
        )
        owner = next(
            owner
            for owner in row["retained"].values()
            if "7fc00000" in owner["f32_bits"]
        )
        owner["f32_bits"][owner["f32_bits"].index("7fc00000")] = "7fc01234"
        with self.assertRaisesRegex(GateError, "wrong retained"):
            verify_lane(self.corpus, rows, native=False)

    def test_construction_failure_is_not_an_actual_binary_return(self):
        corpus = copy.deepcopy(self.corpus)
        case = next(case for case in corpus["error_cases"] if case["op"] == "add")
        case["request"]["shape"] = [1, 1, 1, 1, 1]
        case["expected_error"] = "RankExceeded"
        for native, observations in ((True, self.native), (False, self.javascript)):
            rows = copy.deepcopy(observations)
            row = next(row for row in rows if row["id"] == case["id"])
            row.update(
                error="RankExceeded", raw_error="core:1", boundary="construction"
            )
            retained = row.pop("retained")
            verify_lane(corpus, rows, native=native)
            row["retained"] = retained
            with self.assertRaisesRegex(GateError, "pre-call construction boundary"):
                verify_lane(corpus, rows, native=native)
            del row["retained"]
            del row["boundary"]
            with self.assertRaisesRegex(GateError, "pre-call construction boundary"):
                verify_lane(corpus, rows, native=native)
        rows = copy.deepcopy(self.native)
        row = next(row for row in rows if "retained" in row)
        row["boundary"] = "construction"
        with self.assertRaisesRegex(GateError, "actual binary retained"):
            verify_lane(self.corpus, rows, native=True)

    def test_actual_mapping_controls_require_rejected_and_kernel_checked_twins(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Mock(spec=Evidence)
            evidence.directory = Path(directory)
            evidence.tool_root = ROOT / ".tools"
            certificate_controls(evidence, ROOT)
            commands = {call.args[0]: call for call in evidence.run.call_args_list}
            for name in (
                "permutation-identity-map",
                "permutation-wrong-axis",
                "permutation-wrong-inverse",
                "transpose-requested-map",
                "transpose-wrong-pair",
                "transpose-wrong-inverse",
            ):
                with self.subTest(name=name):
                    for mode in ("check-only", "verdict"):
                        accepted = commands[f"{name}-accepted-{mode}"]
                        rejected = commands[f"{name}-rejected-{mode}"]
                        self.assertEqual(accepted.kwargs["expected_code"], 0)
                        self.assertEqual(rejected.kwargs["expected_code"], 1)
                        self.assertIn("Location: witness", rejected.kwargs["contains"])
                    self.assertIn(f"{name}-export", commands)
                    self.assertIn(f"{name}-direct-kernel", commands)
                    accepted_source = (
                        evidence.directory / f"{name}-accepted.bend"
                    ).read_text()
                    rejected_source = (
                        evidence.directory / f"{name}-rejected.bend"
                    ).read_text()
                    self.assertIn("Layout.map_index(", accepted_source)
                    self.assertIn("Layout.map_index(", rejected_source)
                    self.assertNotEqual(accepted_source, rejected_source)

    def test_missing_duplicate_reordered_or_wrong_lane_inventory_is_rejected(self):
        changes = (
            self.native[:-1],
            [*self.native, self.native[-1]],
            [self.native[1], self.native[0], *self.native[2:]],
            self.javascript,
        )
        for rows in changes:
            with (
                self.subTest(rows=len(rows)),
                self.assertRaisesRegex(GateError, "complete ordered lane inventory"),
            ):
                verify_lane(self.corpus, rows, native=True)

    def test_exact_numeric_corruption_fails_on_each_lane(self):
        case = next(
            case
            for case in self.corpus["finite_cases"]
            if case["comparison_policy"] == "exact_golden"
        )
        for native, observations in ((True, self.native), (False, self.javascript)):
            rows = copy.deepcopy(observations)
            row = next(row for row in rows if row["id"] == case["id"])
            row["result"]["f32_bits"][0] = "00000000"
            with (
                self.subTest(native=native),
                self.assertRaisesRegex(GateError, "exact-word mismatch"),
            ):
                verify_lane(self.corpus, rows, native=native)

    def test_wrong_error_label_or_raw_diagnostic_is_rejected(self):
        for field, value in (
            ("error", "IncompatibleBroadcast"),
            ("raw_error", "core:6"),
        ):
            rows = copy.deepcopy(self.native)
            row = next(row for row in rows if row.get("raw_error") == "core:1")
            row[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(GateError, "wrong native API error"),
            ):
                verify_lane(self.corpus, rows, native=True)

    def test_dtype_is_not_counted_as_an_invented_runtime_error(self):
        rows = copy.deepcopy(self.native)
        row = next(row for row in rows if "compile_negative" in row)
        row["error"] = "UnsupportedDtype"
        with self.assertRaisesRegex(GateError, "missing static F32 rejection"):
            verify_lane(self.corpus, rows, native=True)

    def test_decoder_boundary_and_shape_requests_are_checked(self):
        rows = copy.deepcopy(self.native)
        row = next(row for row in rows if row.get("boundary") == "decoder")
        row["boundary"] = "native"
        with self.assertRaisesRegex(GateError, "missing pretransport rejection"):
            verify_lane(self.corpus, rows, native=True)
        rows = copy.deepcopy(self.native)
        row = next(row for row in rows if "shape_result" in row)
        row["shape_result"]["numel"] += 1
        with self.assertRaisesRegex(GateError, "wrong shape-only result"):
            verify_lane(self.corpus, rows, native=True)


if __name__ == "__main__":
    unittest.main()
