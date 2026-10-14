import copy
import hashlib
import json
import unittest

from oracle.tensor.transpose_reference import requested_axes, valid_axes
from tools.common import ROOT, GateError
from tools.tensor_numerics import verify_envelope
from tools.tensor_runtime import decode_case, load_corpus


class ScalarTransposeProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / "oracle/tensor/fixtures"
        cls.corpus_bytes = (fixtures / "s2-v1.json").read_bytes()
        cls.corpus = json.loads(cls.corpus_bytes)
        cls.proposal = json.loads(
            (fixtures / "scalar-transpose-correction.json").read_text()
        )

    def test_scalar_axes_accept_identity_and_reject_positive_boundaries(self):
        self.assertTrue(valid_axes(0, 0, 0))
        self.assertEqual(requested_axes(0, 0, 0), [])
        for a, b in self.proposal["scalar_invalid_axes"]:
            with self.subTest(a=a, b=b):
                self.assertFalse(valid_axes(0, a, b))
                with self.assertRaisesRegex(ValueError, "axis out of bounds"):
                    requested_axes(0, a, b)

    def test_positive_rank_requested_mapping_and_bounds_remain(self):
        self.assertEqual(requested_axes(1, 0, 0), [0])
        self.assertEqual(requested_axes(2, 0, 1), [1, 0])
        self.assertEqual(requested_axes(3, 0, 2), [2, 1, 0])
        self.assertEqual(requested_axes(4, 0, 2), [2, 1, 0, 3])
        for rank in range(1, 5):
            for a, b in ((rank, 0), (0, rank), (4294967295, 0)):
                with self.subTest(rank=rank, a=a, b=b):
                    self.assertFalse(valid_axes(rank, a, b))
                    with self.assertRaises(ValueError):
                        requested_axes(rank, a, b)

    def test_negative_axes_remain_decoder_errors(self):
        case = copy.deepcopy(self.proposal["replacement"])
        case["parameters"]["dims"] = [-1, 0]
        with self.assertRaisesRegex(GateError, "U32Range"):
            decode_case(case)

    def test_literal_scalar_expectation_rejects_wrong_value_and_shape(self):
        case = self.proposal["replacement"]
        decoded = decode_case(case)
        self.assertEqual(decoded["parameters"], {"dims": [0, 0]})
        self.assertEqual(decoded["inputs"]["a"]["shape"], [])
        self.assertEqual(case["outputs"]["result"]["f32_bits"], ["40e00000"])
        verify_envelope(case, case["outputs"]["result"])
        for shape, word, classification in (
            ([], "00000000", "positive_zero"),
            ([1], "40e00000", "positive_normal"),
        ):
            actual = {
                "shape": shape,
                "f32_bits": [word],
                "classifications": [classification],
            }
            with self.subTest(shape=shape, word=word):
                with self.assertRaises(GateError):
                    verify_envelope(case, actual)

    def test_one_old_id_moves_without_changing_frozen_inputs_or_policy(self):
        self.assertEqual(
            hashlib.sha256(self.corpus_bytes).hexdigest(),
            self.proposal["base_corpus_sha256"],
        )
        policy = ROOT / "oracle/tensor/numerical-policy.json"
        self.assertEqual(
            hashlib.sha256(policy.read_bytes()).hexdigest(),
            self.proposal["policy_sha256"],
        )
        groups = (
            "finite_cases",
            "edge_cases",
            "nonfinite_cases",
            "error_cases",
            "shape_cases",
        )
        original = {case["id"]: case for group in groups for case in self.corpus[group]}
        historical = original[self.proposal["replaced_id"]]
        self.assertEqual(
            self.proposal["replacement"]["id"], self.proposal["replaced_id"]
        )
        self.assertEqual(historical["request"], self.proposal["historical_request"])
        self.assertEqual(
            historical["expected_error"], self.proposal["historical_expected_error"]
        )
        corrected = {
            **original,
            self.proposal["replaced_id"]: self.proposal["replacement"],
        }
        self.assertEqual(set(corrected), set(original))
        self.assertEqual(
            [
                identifier
                for identifier in original
                if original[identifier] != corrected[identifier]
            ],
            ["transpose_scalar"],
        )
        self.assertEqual(len(original), 208)

    def test_active_correction_preserves_every_other_old_case(self):
        active = load_corpus()
        groups = (
            "finite_cases",
            "edge_cases",
            "nonfinite_cases",
            "error_cases",
            "shape_cases",
        )
        old = {case["id"]: case for group in groups for case in self.corpus[group]}
        current = {case["id"]: case for group in groups for case in active[group]}
        self.assertLessEqual(set(old), set(current))
        self.assertEqual(
            [
                identifier
                for identifier in old
                if old[identifier] != current[identifier]
            ],
            ["transpose_scalar"],
        )
        self.assertEqual(
            set(current) - set(old),
            {
                f"transpose_scalar_{suffix}"
                for suffix in ("axis_a", "axis_b", "axis_both", "u32_a", "u32_b")
            },
        )

    def test_active_scalar_success_and_invalid_axes_are_literal(self):
        active = load_corpus()
        scalar = next(
            case for case in active["finite_cases"] if case["id"] == "transpose_scalar"
        )
        self.assertEqual(decode_case(scalar)["parameters"], {"dims": [0, 0]})
        self.assertEqual(scalar["outputs"]["result"]["shape"], [])
        self.assertEqual(scalar["outputs"]["result"]["f32_bits"], ["40e00000"])
        for case in active["error_cases"]:
            if case["id"].startswith("transpose_scalar_"):
                with self.subTest(case=case["id"]):
                    request = decode_case(case)
                    self.assertEqual(request["inputs"]["a"]["shape"], [])
                    self.assertFalse(valid_axes(0, *request["parameters"]["dims"]))
                    self.assertEqual(case["expected_error"], "AxisOutOfBounds")


if __name__ == "__main__":
    unittest.main()
