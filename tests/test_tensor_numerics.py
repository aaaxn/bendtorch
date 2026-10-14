import copy
import json
import unittest

from tools.common import ROOT, GateError
from tools.tensor_numerics import rational, special, verify_envelope


class NumericalAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = json.loads((ROOT / "oracle/tensor/fixtures/s2-v1.json").read_text())
        cls.cases = {
            case["id"]: case
            for group in ("finite_cases", "edge_cases", "nonfinite_cases")
            for case in corpus[group]
        }

    def test_registered_oracle_is_itself_inside_the_policy(self):
        for case in self.cases.values():
            with self.subTest(case=case["id"]):
                verify_envelope(case, case["outputs"]["result"])

    def test_exact_mapping_rejects_changed_word(self):
        case = next(
            case
            for case in self.cases.values()
            if case["comparison_policy"] == "exact_words"
            and case["outputs"]["result"]["f32_bits"]
        )
        actual = copy.deepcopy(case["outputs"]["result"])
        actual["f32_bits"][0] = "00000000"
        with self.assertRaises(GateError):
            verify_envelope(case, actual)

    def test_missing_values_and_wrong_shape_fail(self):
        case = next(
            case
            for case in self.cases.values()
            if case["outputs"]["result"]["f32_bits"]
        )
        for mutation in ("shape", "length"):
            actual = copy.deepcopy(case["outputs"]["result"])
            if mutation == "shape":
                actual["shape"] = [99]
            else:
                actual["f32_bits"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(GateError):
                verify_envelope(case, actual)

    def test_exceptional_values_never_enter_a_tolerance(self):
        self.assertEqual(special("7f800000"), "positive_infinity")
        self.assertEqual(special("ff800000"), "negative_infinity")
        self.assertEqual(special("7f800001"), "nan")
        case = next(
            case
            for case in self.cases.values()
            if "nan" in case["outputs"]["result"]["classifications"]
        )
        actual = copy.deepcopy(case["outputs"]["result"])
        index = case["outputs"]["result"]["classifications"].index("nan")
        actual["f32_bits"][index] = "00000000"
        with self.assertRaises(GateError):
            verify_envelope(case, actual)

    def _shifted(self, identifier: str, index: int, steps: int) -> dict:
        actual = copy.deepcopy(self.cases[identifier]["outputs"]["result"])
        word = int(actual["f32_bits"][index], 16)
        actual["f32_bits"][index] = f"{word + steps:08x}"
        return actual

    def test_bounded_elementwise_accepts_inside_and_rejects_outside(self):
        for identifier, inside in (("random_add_1", 388), ("random_sub_5", 364)):
            case = self.cases[identifier]
            with self.subTest(case=identifier):
                verify_envelope(case, self._shifted(identifier, 3, inside))
                with self.assertRaisesRegex(GateError, "bounded arithmetic mismatch"):
                    verify_envelope(case, self._shifted(identifier, 3, inside + 1))

    def test_bounded_elementwise_checks_the_pytorch_distance_separately(self):
        case = copy.deepcopy(self.cases["random_add_1"])
        actual = copy.deepcopy(case["independent_reference"]["result"])
        verify_envelope(case, actual)
        torch_word = int(case["outputs"]["result"]["f32_bits"][3], 16)
        case["outputs"]["result"]["f32_bits"][3] = f"{torch_word + 776:08x}"
        with self.assertRaisesRegex(GateError, "bounded arithmetic mismatch"):
            verify_envelope(case, actual)

    def test_reduction_envelope_accepts_inside_and_rejects_outside(self):
        for identifier, inside in (("random_sum_0", 60), ("matmul_rectangular_2", 26)):
            case = self.cases[identifier]
            with self.subTest(case=identifier):
                verify_envelope(case, self._shifted(identifier, 0, inside))
                with self.assertRaisesRegex(GateError, "exceeds registered reduction"):
                    verify_envelope(case, self._shifted(identifier, 0, inside + 1))

    def test_reduction_envelope_rejects_oracle_outside_its_bound(self):
        case = copy.deepcopy(self.cases["random_sum_0"])
        actual = copy.deepcopy(case["independent_reference"]["result"])
        verify_envelope(case, actual)
        torch_word = int(case["outputs"]["result"]["f32_bits"][0], 16)
        case["outputs"]["result"]["f32_bits"][0] = f"{torch_word + 61:08x}"
        with self.assertRaisesRegex(GateError, "pinned oracle exceeds"):
            verify_envelope(case, actual)

    def test_empty_reduction_requires_positive_zero_word(self):
        case = self.cases["sum_shape_1"]
        actual = copy.deepcopy(case["outputs"]["result"])
        actual["f32_bits"][0] = "80000000"
        with self.assertRaisesRegex(GateError, "empty identity is not positive zero"):
            verify_envelope(case, actual)

    def test_fraction_decoder_does_not_round_json_integers(self):
        encoded = {"numerator": str(2**100 + 1), "denominator": str(2**100)}
        self.assertGreater(rational(encoded), 1)


if __name__ == "__main__":
    unittest.main()
