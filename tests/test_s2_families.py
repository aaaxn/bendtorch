import copy
import math
import tempfile
import unittest
from pathlib import Path

from tools.common import GateError
from tools.s2_families import program, requests, verify
from tools.s2_runtime import ERROR_CODES, load_corpus


def observation(case: dict, request: dict) -> dict:
    row: dict = {
        "id": case["id"],
        "retained": {
            owner: {
                "shape": request["inputs"][name]["shape"],
                "f32_bits": request["inputs"][name].get(
                    "f32_bits",
                    ["00000000"] * math.prod(request["inputs"][name]["shape"]),
                ),
            }
            for owner, name in (("left", "a"), ("right", "b"))
        },
    }
    if "expected_error" in case:
        category = case["expected_error"]
        code = next(key for key, value in ERROR_CODES.items() if value == category)
        row |= {"error": category, "raw_error": f"core:{code}"}
    else:
        row["result"] = case["outputs"]["result"]
    return row


class PointwiseFamilyCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = requests(load_corpus())
        cls.rows = [observation(case, request) for case, request in cls.selected]

    def test_selects_every_binary_case_that_reaches_bend(self):
        self.assertEqual(len(self.selected), 67)
        self.assertEqual(
            {case["op"] for case, _ in self.selected}, {"add", "sub", "mul"}
        )
        errors = [case for case, _ in self.selected if "expected_error" in case]
        self.assertEqual(len(errors), 4)

    def test_program_calls_the_family_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            text = program([request for _, request in self.selected], Path(directory))
        self.assertEqual(text.count("Family.binary(Elementwise."), 67)
        self.assertNotIn("Harness.binary(", text)
        self.assertIn("contracts/harness.bend as Family", text)

    def test_reference_observations_pass(self):
        summary = verify(self.selected, self.rows)
        self.assertEqual(len(summary["tensors"]), 63)
        self.assertEqual(len(summary["errors"]), 4)

    def test_rejects_a_changed_retained_input(self):
        rows = copy.deepcopy(self.rows)
        index = next(
            i for i, row in enumerate(rows) if row["retained"]["right"]["f32_bits"]
        )
        rows[index]["retained"]["right"]["f32_bits"][0] = "3f800001"
        with self.assertRaisesRegex(GateError, "right input changed"):
            verify(self.selected, rows)

    def test_rejects_a_wrong_output_word(self):
        rows = copy.deepcopy(self.rows)
        index = next(
            i
            for i, (case, _) in enumerate(self.selected)
            if case["id"] == "sub_broadcast_2"
        )
        words = rows[index]["result"]["f32_bits"]
        words[0], words[1] = words[1], words[0]
        with self.assertRaises(GateError):
            verify(self.selected, rows)

    def test_rejects_a_missing_error(self):
        rows = copy.deepcopy(self.rows)
        index = next(i for i, row in enumerate(rows) if "error" in row)
        del rows[index]["error"]
        with self.assertRaisesRegex(GateError, "expected error"):
            verify(self.selected, rows)

    def test_rejects_a_reordered_inventory(self):
        with self.assertRaisesRegex(GateError, "inventory"):
            verify(self.selected, self.rows[1:] + self.rows[:1])


if __name__ == "__main__":
    unittest.main()
