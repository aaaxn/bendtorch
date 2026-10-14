import json
import tempfile
import unittest
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_broadcast import (
    CAPACITY_EXCEEDED,
    INCOMPATIBLE_BROADCAST,
    RANK_EXCEEDED,
    parse,
    verify,
    word,
)
from tools.autograd_reduce import (
    NOT_REDUCIBLE,
    bits,
    cases,
    reference,
    run_gate,
    same_sums,
    value,
)
from tools.common import ROOT, GateError


def case(shape: list[int], target: list[int], words: list[int]) -> dict:
    return {"id": "case", "shape": shape, "target": target, "words": words}


def counting(count: int) -> list[int]:
    return [word(float(index + 1)) for index in range(count)]


def floats(result: dict) -> list[float]:
    return [value(item) for item in result["words"]]


class ReduceReferenceTests(unittest.TestCase):
    def test_handwritten_examples(self):
        self.assertEqual(floats(reference(case([2, 3], [3], counting(6)))), [5, 7, 9])
        self.assertEqual(floats(reference(case([2, 3], [2, 1], counting(6)))), [6, 15])
        self.assertEqual(floats(reference(case([2, 3], [], counting(6)))), [21])
        self.assertEqual(
            floats(reference(case([2, 3], [2, 3], counting(6)))), [1, 2, 3, 4, 5, 6]
        )

    def test_empty_contribution_gives_zeros_not_an_empty_tensor(self):
        self.assertEqual(
            reference(case([0, 3], [1, 3], [])),
            {"shape": [1, 3], "words": [0, 0, 0]},
        )
        self.assertEqual(reference(case([0], [], [])), {"shape": [], "words": [0]})

    def test_accumulation_is_ordered_f32(self):
        big = [word(1e8), word(1), word(-1e8), word(1)]
        self.assertEqual(floats(reference(case([4], [], big))), [1.0])
        self.assertEqual(floats(reference(case([4], [], big[::-1]))), [0.0])
        negative_zero = reference(case([2], [], [0x80000000, 0x80000000]))
        self.assertEqual(negative_zero["words"], [0])

    def test_finite_overflow_rounds_to_signed_infinity(self):
        largest = 0x7F7FFFFF
        self.assertEqual(
            reference(case([2], [], [largest, largest]))["words"], [0x7F800000]
        )
        negative = largest | 0x80000000
        self.assertEqual(
            reference(case([2, 2], [2], [negative, largest, negative, word(1)]))[
                "words"
            ],
            [0xFF800000, largest],
        )

    def test_refusals_follow_the_error_codes(self):
        self.assertEqual(
            reference(case([3], [2, 3], counting(3))), {"error": NOT_REDUCIBLE}
        )
        self.assertEqual(reference(case([1], [3], [0])), {"error": NOT_REDUCIBLE})
        self.assertEqual(
            reference(case([3], [2], counting(3))), {"error": INCOMPATIBLE_BROADCAST}
        )
        self.assertEqual(reference(case([1], [1] * 5, [0])), {"error": RANK_EXCEEDED})
        self.assertEqual(
            reference(case([1], [1025, 1024], [0])), {"error": CAPACITY_EXCEEDED}
        )

    def test_inventory_covers_every_small_pair(self):
        selected = cases()
        small = [row for row in selected if row["id"].startswith("small_")]
        self.assertEqual(len(small), 40 * 13)
        self.assertEqual(len({row["id"] for row in selected}), len(selected))
        outcomes = [reference(row) for row in small]
        self.assertGreater(sum("words" in row for row in outcomes), 100)
        self.assertGreater(sum("error" in row for row in outcomes), 100)

    def test_reduce_inverts_broadcast_counts(self):
        ones = [word(1)] * 6
        self.assertEqual(floats(reference(case([2, 3], [1, 3], ones))), [2, 2, 2])
        self.assertEqual(bits(2.0), word(2))


class ReduceVerificationTests(unittest.TestCase):
    def output(self, rows: list[str]) -> str:
        return "\n".join(["S3:begin:case", *rows, "S3:end"]) + "\n"

    def retained(self) -> list[str]:
        return [
            "S3:retained",
            "S2:shape:[2]",
            "S2:count:2",
            f"S2:word:{word(1)}",
            f"S2:word:{word(2)}",
        ]

    def test_ordered_sum_is_accepted(self):
        row = case([2], [], [word(1), word(2)])
        text = self.output(
            [
                *self.retained(),
                "S3:output",
                "S2:shape:[]",
                "S2:count:1",
                f"S2:word:{word(3)}",
            ]
        )
        observed = parse(text, ["case"])
        summary = verify([row], observed, "native", reference, same_sums)
        self.assertEqual(summary["tensors"], 1)

    def test_overwrite_or_wrong_sum_is_rejected(self):
        row = case([2], [], [word(1), word(2)])
        for result in (word(2), word(1), 0):
            text = self.output(
                [
                    *self.retained(),
                    "S3:output",
                    "S2:shape:[]",
                    "S2:count:1",
                    f"S2:word:{result}",
                ]
            )
            with self.subTest(result=result), self.assertRaises(GateError):
                verify([row], parse(text, ["case"]), "native", reference, same_sums)


class StageRegistrationTests(unittest.TestCase):
    def test_s3_gate_registers_reduce_and_its_laws(self):
        self.assertIn("reduce", SUBGATES)
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(sum("reduce_2d_LAWS" in name for name in required), 7)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)


class ReduceRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="autograd-reduce-") as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))


if __name__ == "__main__":
    unittest.main()
