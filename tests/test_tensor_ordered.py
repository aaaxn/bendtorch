import copy
import json
import unittest

from tools.common import ROOT, GateError
from tools.tensor_numerics import verify_case
from tools.tensor_ordered import verify_ordered_case, verify_ordered_lane
from tools.tensor_runtime import parse_native_output


class OrderedAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / "oracle/tensor/fixtures"
        cls.corpus = json.loads((fixtures / "s2-v1.json").read_text())
        cls.records = json.loads((fixtures / "ordered-reference.json").read_text())
        cls.controls = json.loads((fixtures / "ordered-controls.json").read_text())

    def test_compatibility_accepts_both_while_ordered_rejects_reassociation(self):
        case = next(
            case
            for case in self.corpus["finite_cases"]
            if case["id"] == "sum_cancellation_0"
        )
        record = next(row for row in self.records["cases"] if row["id"] == case["id"])
        actual = copy.deepcopy(case["outputs"]["result"])
        verify_case(case, actual)
        self.assertEqual(verify_ordered_case(record, actual)["elements"], 1)
        actual["f32_bits"] = ["3f800000"]
        actual["classifications"] = ["positive_normal"]
        self.assertEqual(verify_case(case, actual)["max_exact_error"]["numerator"], "0")
        with self.assertRaisesRegex(GateError, "primitive-rounding mismatch"):
            verify_ordered_case(record, actual)

    def test_fusion_control_and_existing_native_parser(self):
        case = self.controls["cases"][1]
        record = {"id": case["id"], "result": case["ordered_result"]}
        text = f"S2:begin:{case['id']}\nS2:shape:[1,1]\nS2:count:1\nS2:word:0\nS2:end\n"
        observed = parse_native_output(text)[0]["result"]
        self.assertEqual(verify_ordered_case(record, observed)["elements"], 1)
        with self.assertRaisesRegex(GateError, "primitive-rounding mismatch"):
            verify_ordered_case(record, case["alternative_result"])

    def test_nan_payloads_are_not_a_portable_equality_requirement(self):
        record = {
            "id": "nan",
            "result": {
                "shape": [],
                "f32_bits": ["7fc00000"],
                "classifications": ["nan"],
            },
        }
        for word in ("7f800001", "ffc12345"):
            self.assertEqual(
                verify_ordered_case(record, {"shape": [], "f32_bits": [word]})[
                    "elements"
                ],
                1,
            )
        with self.assertRaises(GateError):
            verify_ordered_case(record, {"shape": [], "f32_bits": ["7f800000"]})

    def test_zero_sign_subnormal_and_infinity_are_exact(self):
        for expected, wrong in (
            ("00000000", "80000000"),
            ("00000001", "00000000"),
            ("ff800000", "7f800000"),
        ):
            record = {"id": "edge", "result": {"shape": [], "f32_bits": [expected]}}
            self.assertEqual(
                verify_ordered_case(record, record["result"])["elements"], 1
            )
            with self.subTest(expected=expected), self.assertRaises(GateError):
                verify_ordered_case(record, {"shape": [], "f32_bits": [wrong]})

    def test_complete_inventory_and_input_binding(self):
        observations = [
            {"id": row["id"], "result": row["result"]} for row in self.records["cases"]
        ]
        self.assertEqual(
            len(verify_ordered_lane(self.corpus, self.records, observations)),
            len(self.records["cases"]),
        )
        for mutation in ("record", "observation", "duplicate", "input"):
            records = copy.deepcopy(self.records)
            actual = copy.deepcopy(observations)
            if mutation == "record":
                records["cases"].pop()
            elif mutation == "observation":
                actual.pop()
            elif mutation == "duplicate":
                actual.append(actual[0])
            else:
                records["cases"][0]["input_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(GateError):
                verify_ordered_lane(self.corpus, records, actual)

    def test_shape_count_and_classification_mutations_fail(self):
        record = {"id": "one", "result": {"shape": [], "f32_bits": ["3f800000"]}}
        for actual in (
            {"shape": [1], "f32_bits": ["3f800000"]},
            {"shape": [], "f32_bits": []},
            {"shape": [], "f32_bits": ["3F800000"]},
            {"shape": [], "f32_bits": ["3f800000"], "classifications": ["nan"]},
        ):
            with self.subTest(actual=actual), self.assertRaises(GateError):
                verify_ordered_case(record, actual)


if __name__ == "__main__":
    unittest.main()
