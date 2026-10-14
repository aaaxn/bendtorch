import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.common import ROOT, GateError
from tools.s3 import PROOF_CONTROLS, SUBGATES
from tools.s3_broadcast import (
    CAPACITY_EXCEEDED,
    INCOMPATIBLE_BROADCAST,
    NOT_BROADCASTABLE,
    RANK_EXCEEDED,
    cases,
    parse,
    program,
    reference,
    run_stage,
    verify,
    word,
)


def case(shape: list[int], target: list[int], words: list[int]) -> dict:
    return {"id": "case", "shape": shape, "target": target, "words": words}


def floats(words: list[int]) -> list[float]:
    return [struct.unpack("<f", struct.pack("<I", value))[0] for value in words]


class BroadcastReferenceTests(unittest.TestCase):
    def test_handwritten_examples(self):
        one_two_three = [word(1), word(2), word(3)]
        self.assertEqual(
            floats(reference(case([3], [2, 3], one_two_three))["words"]),
            [1, 2, 3, 1, 2, 3],
        )
        self.assertEqual(
            floats(reference(case([], [2, 2], [word(7)]))["words"]), [7, 7, 7, 7]
        )
        self.assertEqual(
            floats(reference(case([2, 1], [2, 3], [word(1), word(2)]))["words"]),
            [1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(
            reference(case([1, 3], [0, 3], one_two_three)),
            {"shape": [0, 3], "words": []},
        )

    def test_refusals_follow_the_error_codes(self):
        self.assertEqual(
            reference(case([3], [1], [0, 0, 0])), {"error": NOT_BROADCASTABLE}
        )
        self.assertEqual(
            reference(case([2, 3], [3], [0] * 6)), {"error": NOT_BROADCASTABLE}
        )
        self.assertEqual(
            reference(case([2], [3], [0, 0])), {"error": INCOMPATIBLE_BROADCAST}
        )
        self.assertEqual(reference(case([1], [1] * 5, [0])), {"error": RANK_EXCEEDED})
        self.assertEqual(
            reference(case([1], [1025, 1024], [0])), {"error": CAPACITY_EXCEEDED}
        )

    def test_inventory_covers_every_small_pair(self):
        selected = cases()
        small = [row for row in selected if row["id"].startswith("small_")]
        self.assertEqual(len(small), 13 * 40)
        self.assertEqual(len({row["id"] for row in selected}), len(selected))
        outcomes = [reference(row) for row in small]
        self.assertGreater(sum("words" in row for row in outcomes), 100)
        self.assertGreater(sum("error" in row for row in outcomes), 100)

    def test_ordered_words_are_bit_exact_and_keep_specials(self):
        special = next(row for row in cases() if row["id"] == "special_words")
        expected = reference(special)["words"]
        self.assertEqual(expected[::2], special["words"])
        self.assertEqual(expected[1::2], special["words"])


class BroadcastParserTests(unittest.TestCase):
    def output(self, rows: list[str]) -> str:
        return "\n".join(["S3:begin:case", *rows, "S3:end"]) + "\n"

    def test_parse_and_verify_success(self):
        row = case([1], [2], [word(5)])
        text = self.output(
            [
                "S3:retained",
                "S2:shape:[1]",
                "S2:count:1",
                f"S2:word:{word(5)}",
                "S3:output",
                "S2:shape:[2]",
                "S2:count:2",
                f"S2:word:{word(5)}",
                f"S2:word:{word(5)}",
            ]
        )
        observed = parse(text, ["case"])
        self.assertEqual(verify([row], observed, "native")["tensors"], 1)

    def test_wrong_words_owner_or_error_are_rejected(self):
        row = case([1], [2], [word(5)])
        good = [
            "S3:retained",
            "S2:shape:[1]",
            "S2:count:1",
            f"S2:word:{word(5)}",
        ]
        for rows in (
            [
                *good,
                "S3:output",
                "S2:shape:[2]",
                "S2:count:2",
                "S2:word:0",
                "S2:word:0",
            ],
            [*good, "S3:error:100"],
            ["S3:retained", "S2:shape:[1]", "S2:count:1", "S2:word:0", "S3:error:100"],
        ):
            with self.subTest(rows=rows), self.assertRaises(GateError):
                verify([row], parse(self.output(rows), ["case"]), "native")

    def test_missing_cases_are_rejected(self):
        with self.assertRaises(GateError):
            parse("", ["case"])

    def test_nan_payloads_differ_only_on_javascript(self):
        row = case([1], [1], [0x7FC01234])
        rows = [
            "S3:retained",
            "S2:shape:[1]",
            "S2:count:1",
            "S2:word:2143289344",
            "S3:output",
            "S2:shape:[1]",
            "S2:count:1",
            "S2:word:2143289344",
        ]
        observed = parse(self.output(rows), ["case"])
        verify([row], observed, "javascript")
        with self.assertRaises(GateError):
            verify([row], observed, "native")

    def test_program_imports_the_autograd_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            text = program(cases()[:2], Path(directory))
        self.assertIn("src/autograd/harness.bend as Harness", text)
        self.assertIn('IO.print("S3:begin:row_to_matrix")', text)


class StageRegistrationTests(unittest.TestCase):
    def test_s3_gate_registers_broadcast_and_its_laws(self):
        self.assertIn("broadcast", SUBGATES)
        required = json.loads((ROOT / "verification/s3.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(sum("broadcast_2d_LAWS" in name for name in required), 6)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)


class BroadcastRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="s3-broadcast-") as directory:
            summary = run_stage(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))


if __name__ == "__main__":
    unittest.main()
