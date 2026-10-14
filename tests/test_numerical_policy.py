import copy
import json
import unittest

from tools.common import ROOT, GateError, digest
from tools.tensor import verify_supplement
from tools.tensor_numerics import verify_case, verify_case_v3
from tools.tensor_runtime import (
    CORPUS,
    POLICY,
    POLICY_SHA256,
    POLICY_V3,
    load_corpus,
    load_supplement,
)


def shifted(case: dict, index: int, steps: int) -> dict:
    actual = copy.deepcopy(case["outputs"]["result"])
    word = int(actual["f32_bits"][index], 16)
    actual["f32_bits"][index] = f"{word + steps:08x}"
    return actual


class PolicyV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = load_corpus()
        cls.cases = {
            case["id"]: case
            for group in ("finite_cases", "edge_cases", "nonfinite_cases")
            for case in corpus[group]
        }
        cls.supplement = load_supplement()

    def test_version_two_stays_frozen(self):
        self.assertEqual(digest(POLICY), POLICY_SHA256)
        frozen = json.loads(POLICY.read_text())
        self.assertEqual(frozen["policy_version"], 2)
        v3 = json.loads(POLICY_V3.read_text())
        self.assertEqual(v3["policy_version"], 3)
        self.assertEqual(v3["supersedes"]["sha256"], POLICY_SHA256)
        for name, rules in frozen["policies"].items():
            for key, value in rules.items():
                self.assertEqual(v3["policies"][name][key], value)

    def test_every_frozen_pytorch_result_satisfies_version_three(self):
        for case in self.cases.values():
            with self.subTest(case=case["id"]):
                verify_case_v3(case, case["outputs"]["result"])
        for case in self.supplement["cases"]:
            with self.subTest(case=case["id"]):
                verify_case_v3(case, case["outputs"]["result"])

    def test_exact_word_lane_rejects_one_ulp_that_version_two_accepts(self):
        case = self.cases["random_add_1"]
        actual = shifted(case, 3, 1)
        verify_case(case, actual)
        with self.assertRaisesRegex(GateError, "not the round-to-nearest word"):
            verify_case_v3(case, actual)

    def test_reduction_zero_sign_must_match_pytorch(self):
        for identifier, index in (
            ("sum_cancellation_0", 0),
            ("sum_cancellation_3", 0),
            ("matmul_cancellation", 0),
        ):
            case = self.cases[identifier]
            actual = copy.deepcopy(case["outputs"]["result"])
            self.assertEqual(actual["f32_bits"][index], "00000000")
            actual["f32_bits"][index] = "80000000"
            verify_case(case, actual)
            with (
                self.subTest(case=identifier),
                self.assertRaisesRegex(GateError, "zero sign differs"),
            ):
                verify_case_v3(case, actual)

    def test_reduction_envelope_is_otherwise_unchanged(self):
        case = self.cases["random_sum_0"]
        verify_case_v3(case, shifted(case, 0, 60))
        with self.assertRaisesRegex(GateError, "exceeds registered reduction"):
            verify_case_v3(case, shifted(case, 0, 61))


class SupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supplement = load_supplement()

    def observations(self) -> list[dict]:
        rows = [
            {"id": case["id"], "result": case["outputs"]["result"]}
            for case in self.supplement["cases"]
        ]
        rows += [
            {
                "id": case["id"],
                "error": case["expected_error"],
                "raw_error": "shape:6"
                if case["expected_error"] == "IncompatibleBroadcast"
                else "matrix:inner",
            }
            for case in self.supplement["error_cases"]
        ]
        return rows

    def test_base_corpus_is_unchanged_and_bound(self):
        self.assertEqual(self.supplement["base_corpus"]["sha256"], digest(CORPUS))
        base = load_corpus()
        identifiers = {
            case["id"]
            for group in ("finite_cases", "edge_cases", "nonfinite_cases")
            for case in base[group]
        }
        for case in self.supplement["cases"]:
            self.assertNotIn(case["id"], identifiers)

    def test_covers_the_audited_gaps(self):
        identifiers = {case["id"] for case in self.supplement["cases"]}
        for required in (
            "matmul_add_scalar_bias",
            "matmul_add_row_bias",
            "matmul_add_column_bias",
            "matmul_add_cancellation",
            "matmul_add_zero_inner",
            "matmul_add_negative_zero_bias",
            "get_special_0_0",
            "get_special_0_1",
            "get_special_1_0",
            "get_special_1_1",
            "add_scalar_left",
            "ieee_add_negative_overflow",
        ):
            self.assertIn(required, identifiers)
        self.assertEqual(
            [case["expected_error"] for case in self.supplement["error_cases"]],
            ["IncompatibleBroadcast", "MatmulInnerMismatch"],
        )

    def test_pytorch_observations_pass_and_mutations_fail(self):
        summary = verify_supplement(self.supplement, self.observations())
        self.assertEqual(len(summary["numerical_cases"]), 29)
        self.assertEqual(len(summary["ordered_reference"]), 12)
        for identifier, word in (
            ("get_special_0_0", "00000000"),
            ("matmul_add_negative_zero_bias", "80000000"),
            ("matmul_add_cancellation", "3f800000"),
            ("add_scalar_left", "bf400001"),
        ):
            rows = self.observations()
            row = next(row for row in rows if row["id"] == identifier)
            row["result"] = copy.deepcopy(row["result"])
            row["result"]["f32_bits"][0] = word
            with self.subTest(case=identifier), self.assertRaises(GateError):
                verify_supplement(self.supplement, rows)
        rows = self.observations()
        rows[-1]["error"] = "MatmulRankMismatch"
        with self.assertRaises(GateError):
            verify_supplement(self.supplement, rows)

    def test_supplement_files_are_registered(self):
        self.assertTrue((ROOT / "oracle/tensor/supplement.py").is_file())


if __name__ == "__main__":
    unittest.main()
