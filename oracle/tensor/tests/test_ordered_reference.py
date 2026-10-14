import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from ordered_reference import (
    add,
    dyadic,
    multiply,
    ordered_result,
    reference_records,
    rounded,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class OrderedReferenceTests(unittest.TestCase):
    def test_nearest_even_and_carry(self):
        for coefficient, exponent, expected in (
            ((1 << 24) + 1, -24, 0x3F800000),
            ((1 << 24) + 3, -24, 0x3F800002),
            ((1 << 25) - 1, -24, 0x40000000),
            ((1 << 25) - 1, 103, 0x7F800000),
            ((1 << 25) - 2, 103, 0x7F7FFFFF),
            (1, -150, 0x00000000),
            (-1, -150, 0x80000000),
            (3, -150, 0x00000002),
            ((1 << 24) - 1, -150, 0x00800000),
        ):
            with self.subTest(coefficient=coefficient, exponent=exponent):
                self.assertEqual(rounded(coefficient, exponent), expected)

    def test_gradual_underflow_and_signed_zero(self):
        self.assertEqual(add(0x00000001, 0x00000001), 0x00000002)
        self.assertEqual(multiply(0x00800000, 0x3F000000), 0x00400000)
        self.assertEqual(multiply(0x80000001, 0x3F000000), 0x80000000)
        self.assertEqual(add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(add(0x00000000, 0x80000000), 0x00000000)
        self.assertEqual(add(0x3F800000, 0xBF800000), 0x00000000)
        self.assertEqual(multiply(0x80000000, 0x3F800000), 0x80000000)
        self.assertEqual(multiply(0x80000000, 0xBF800000), 0x00000000)

    def test_nonfinites_and_overflow(self):
        self.assertEqual(add(0x7F800000, 0xFF800000), 0x7FC00000)
        self.assertEqual(add(0x7F800000, 0x3F800000), 0x7F800000)
        self.assertEqual(multiply(0xFF800000, 0xBF800000), 0x7F800000)
        self.assertEqual(multiply(0x00000000, 0xFF800000), 0x7FC00000)
        self.assertEqual(add(0x7F800001, 0x3F800000), 0x7FC00000)
        self.assertEqual(add(0x7F7FFFFF, 0x7F7FFFFF), 0x7F800000)
        self.assertEqual(multiply(0xFF7FFFFF, 0x40000000), 0xFF800000)

    def test_cancellation_and_fusion_controls(self):
        controls = json.loads((FIXTURES / "ordered-controls.json").read_text())
        for case in controls["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(ordered_result(case), case["ordered_result"])
                self.assertNotEqual(ordered_result(case), case["alternative_result"])
        self.assertEqual(add(0x4B800000, add(0x3F800000, 0xCB800000)), 0x3F800000)
        a, ae = dyadic(0x3F800001)
        b, be = dyadic(0x3F7FFFFE)
        exact_residual = Fraction(a) * Fraction(2) ** ae * b * Fraction(2) ** be - 1
        self.assertEqual(exact_residual, -Fraction(1, 1 << 46))
        self.assertEqual(rounded(-1, -46), 0xA8800000)

    def test_empty_identity_zero_inner_and_bias(self):
        self.assertEqual(
            ordered_result(
                {"op": "sum", "inputs": {"a": {"shape": [0], "f32_bits": []}}}
            ),
            {
                "shape": [],
                "f32_bits": ["00000000"],
                "classifications": ["positive_zero"],
            },
        )
        case = {
            "op": "matmul_add",
            "inputs": {
                "a": {"shape": [2, 0], "f32_bits": []},
                "b": {"shape": [0, 2], "f32_bits": []},
                "bias": {"shape": [2], "f32_bits": ["80000000", "00000001"]},
            },
        }
        self.assertEqual(
            ordered_result(case)["f32_bits"],
            ["00000000", "00000001", "00000000", "00000001"],
        )
        case["inputs"]["a"]["shape"] = [0, 0]
        self.assertEqual(ordered_result(case)["f32_bits"], [])

    def test_portable_records_match_all_frozen_reduction_inputs(self):
        corpus = json.loads((FIXTURES / "s2-v1.json").read_text())
        records = json.loads((FIXTURES / "ordered-reference.json").read_text())
        self.assertEqual(reference_records(corpus), records)
        by_id = {case["id"]: case["result"] for case in records["cases"]}
        self.assertEqual(by_id["sum_cancellation_0"]["f32_bits"], ["00000000"])
        self.assertEqual(
            by_id["golden_ab"]["f32_bits"],
            ["42680000", "42800000", "430b0000", "431a0000"],
        )
        self.assertEqual(by_id["golden_sum_415"]["f32_bits"], ["43cf8000"])

    def test_frozen_finite_inputs_roundtrip_without_host_floats(self):
        corpus = json.loads((FIXTURES / "s2-v1.json").read_text())
        checked = 0
        for group in ("finite_cases", "edge_cases", "nonfinite_cases"):
            for case in corpus[group]:
                for tensor in case["inputs"].values():
                    for encoded in tensor["f32_bits"]:
                        word = int(encoded, 16)
                        if word & 0x7F800000 == 0x7F800000:
                            continue
                        coefficient, exponent = dyadic(word)
                        with self.subTest(case=case["id"], word=encoded):
                            self.assertEqual(
                                rounded(
                                    coefficient,
                                    exponent,
                                    negative_zero=word == 0x80000000,
                                ),
                                word,
                            )
                        checked += 1
        self.assertGreater(checked, 0)

    def test_cli_check_and_write_preserve_existing_failures(self):
        script = FIXTURES.parent / "ordered_reference.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ordered.json"
            prefix = [sys.executable, "-B", str(script), "--output", str(output)]
            written = subprocess.run(
                [*prefix, "--write"], capture_output=True, text=True, check=False
            )
            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertEqual(
                output.read_bytes(), (FIXTURES / "ordered-reference.json").read_bytes()
            )
            checked = subprocess.run(
                [*prefix, "--check"], capture_output=True, text=True, check=False
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            output.write_text("retained mismatch\n")
            for mode in ("--check", "--write"):
                failed = subprocess.run(
                    [*prefix, mode], capture_output=True, text=True, check=False
                )
                with self.subTest(mode=mode):
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertEqual(output.read_text(), "retained mismatch\n")


if __name__ == "__main__":
    unittest.main()
