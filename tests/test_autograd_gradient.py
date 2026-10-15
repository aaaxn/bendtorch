import json
import tempfile
import unittest
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_broadcast import CAPACITY_EXCEEDED, RANK_EXCEEDED, word
from tools.autograd_gradient import (
    SHAPE_MISMATCH,
    cases,
    contribution,
    parse,
    reference,
    run_gate,
    verify,
)
from tools.autograd_reduce import value
from tools.common import ROOT, GateError


def case(shape: list[int], items: list[dict]) -> dict:
    return {"id": "case", "shape": shape, "items": items}


def scalar(number: float) -> dict:
    return contribution([], [word(number)])


def floats(result: dict) -> list[float]:
    return [value(item) for item in result["gradient"]["words"]]


class GradientReferenceTests(unittest.TestCase):
    def test_unset_slot_reads_zeros_of_its_shape(self):
        self.assertEqual(reference(case([2, 3], []))["gradient"]["words"], [0] * 6)
        self.assertEqual(reference(case([], []))["gradient"]["words"], [0])
        self.assertEqual(reference(case([0, 3], []))["gradient"]["words"], [])

    def test_first_contribution_moves_in_and_later_ones_add(self):
        first = contribution([3], [word(1), word(2), word(3)])
        second = contribution([3], [word(10), word(20), word(30)])
        self.assertEqual(floats(reference(case([3], [first]))), [1, 2, 3])
        self.assertEqual(floats(reference(case([3], [first, second]))), [11, 22, 33])

    def test_accumulation_is_ordered_f32_and_never_an_overwrite(self):
        big = [scalar(1e8), scalar(1), scalar(-1e8), scalar(1)]
        self.assertEqual(floats(reference(case([], big))), [1.0])
        self.assertEqual(floats(reference(case([], big[::-1]))), [0.0])
        self.assertEqual(floats(reference(case([], [scalar(2), scalar(5)]))), [7.0])

    def test_first_contribution_keeps_negative_zero(self):
        negative = contribution([], [0x80000000])
        positive = contribution([], [0])
        self.assertEqual(
            reference(case([], [negative]))["gradient"]["words"], [0x80000000]
        )
        self.assertEqual(
            reference(case([], [negative, negative]))["gradient"]["words"],
            [0x80000000],
        )
        self.assertEqual(
            reference(case([], [negative, positive]))["gradient"]["words"], [0]
        )

    def test_mismatched_shapes_are_refused_and_skipped(self):
        result = reference(
            case([2], [scalar(1), contribution([2]), contribution([1, 2])])
        )
        self.assertEqual(
            [outcome.get("refused") for outcome in result["outcomes"]],
            [SHAPE_MISMATCH, None, SHAPE_MISMATCH],
        )
        self.assertEqual(floats(result), [1, 2])

    def test_invalid_slot_shapes_follow_the_error_codes(self):
        self.assertEqual(reference(case([1] * 5, [])), {"error": RANK_EXCEEDED})
        self.assertEqual(
            reference(case([1025, 1024], [])), {"error": CAPACITY_EXCEEDED}
        )

    def test_inventory_covers_every_small_pair(self):
        selected = cases()
        small = [row for row in selected if row["id"].startswith("small_")]
        self.assertEqual(len(small), 13 + 13 * 13)
        self.assertEqual(len({row["id"] for row in selected}), len(selected))


class GradientVerificationTests(unittest.TestCase):
    def output(self, rows: list[str]) -> str:
        return "\n".join(["S3:begin:case", *rows, "S3:end"]) + "\n"

    def gradient(self, words: list[int]) -> list[str]:
        return [
            "S3:gradient",
            "tensor:shape:[]",
            "tensor:count:1",
            *[f"tensor:word:{item}" for item in words],
        ]

    def test_ordered_sum_is_accepted(self):
        row = case([], [scalar(1), scalar(2)])
        text = self.output(
            ["S3:accumulated", "S3:accumulated", *self.gradient([word(3)])]
        )
        summary = verify([row], parse(text, ["case"]), "native")
        self.assertEqual(summary["gradients"], 1)

    def test_overwrite_or_zero_start_is_rejected(self):
        row = case([], [scalar(1), scalar(2)])
        for result in (word(2), word(1), 0):
            text = self.output(
                [
                    "S3:accumulated",
                    "S3:accumulated",
                    *self.gradient([result]),
                ]
            )
            with self.subTest(result=result), self.assertRaises(GateError):
                verify([row], parse(text, ["case"]), "native")

    def test_refusal_must_return_the_contribution(self):
        row = case([], [contribution([1], [word(4)])])
        returned = ["tensor:shape:[1]", "tensor:count:1"]
        accepted = self.output(
            [
                "S3:refused:102",
                *returned,
                f"tensor:word:{word(4)}",
                *self.gradient([0]),
            ]
        )
        self.assertEqual(
            verify([row], parse(accepted, ["case"]), "native")["refusals"], 1
        )
        changed = self.output(
            [
                "S3:refused:102",
                *returned,
                f"tensor:word:{word(5)}",
                *self.gradient([0]),
            ]
        )
        with self.assertRaises(GateError):
            verify([row], parse(changed, ["case"]), "native")

    def test_a_missing_outcome_is_rejected(self):
        row = case([], [scalar(1), scalar(2)])
        text = self.output(["S3:accumulated", *self.gradient([word(3)])])
        with self.assertRaises(GateError):
            verify([row], parse(text, ["case"]), "native")


class GateRegistrationTests(unittest.TestCase):
    def test_autograd_gate_registers_gradient_and_its_laws(self):
        self.assertEqual(
            list(SUBGATES),
            [
                "broadcast",
                "reduce",
                "sum_to_size",
                "gradient",
                "lifecycle",
                "backward",
                "tensor_backward",
            ],
        )
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(sum("gradient_2d_LAWS" in name for name in required), 10)
        controls = [name for name in PROOF_CONTROLS if name.startswith("gradient-")]
        self.assertEqual(len(controls), 6)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)


class GradientRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="autograd-gradient-") as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))


if __name__ == "__main__":
    unittest.main()
