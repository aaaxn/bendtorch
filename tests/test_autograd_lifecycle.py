import tempfile
import unittest
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_broadcast import word
from tools.autograd_gradient import contribution
from tools.autograd_lifecycle import (
    NEGATIVE_ZERO,
    RUNTIME_CONTROLS,
    cases,
    parse,
    reference,
    run_gate,
    verify,
)
from tools.common import ROOT, GateError


def case(shape: list[int], before: list[dict], after: list[dict]) -> dict:
    return {"id": "case", "shape": shape, "before": before, "after": after}


def tensor(shape: list[int], words: list[int]) -> list[str]:
    return [
        f"S2:shape:{shape}".replace(" ", ""),
        f"S2:count:{len(words)}",
        *[f"S2:word:{item}" for item in words],
    ]


class LifecycleReferenceTests(unittest.TestCase):
    # PyTorch 2.14.1+cpu: .grad is None before backward and after
    # zero_grad(set_to_none=True); the next backward stores its gradient as is.
    def test_never_set_reads_none_but_gradient_materializes_zeros(self):
        expected = reference(case([2], [], []))
        self.assertIsNone(expected["held"])
        self.assertIsNone(expected["final"])
        self.assertEqual(expected["gradient"], {"shape": [2], "words": [0, 0]})

    def test_reset_reads_none_and_next_contribution_moves_in(self):
        expected = reference(
            case(
                [],
                [contribution([], [word(5)])],
                [contribution([], [NEGATIVE_ZERO])],
            )
        )
        self.assertEqual(expected["held"], {"shape": [], "words": [word(5)]})
        self.assertIsNone(expected["cleared"])
        self.assertEqual(expected["final"], {"shape": [], "words": [NEGATIVE_ZERO]})

    def test_zero_fill_reset_would_lose_negative_zero(self):
        expected = reference(
            case([], [contribution([], [0])], [contribution([], [NEGATIVE_ZERO])])
        )
        self.assertNotEqual(expected["final"]["words"], [0])


class LifecycleVerificationTests(unittest.TestCase):
    def frame(self, rows: list[str]) -> str:
        return "\n".join(["S3:begin:case", *rows, "S3:end"]) + "\n"

    def test_zeros_instead_of_none_is_rejected(self):
        row = case([2], [], [])
        good = self.frame(
            [
                "S3:grad:none",
                "S3:reset",
                "S3:grad:none",
                "S3:grad:none",
                "S3:gradient",
                *tensor([2], [0, 0]),
            ]
        )
        verify([row], parse(good, ["case"]), "native")
        bad = self.frame(
            [
                "S3:grad:some",
                *tensor([2], [0, 0]),
                "S3:reset",
                "S3:grad:none",
                "S3:grad:none",
                "S3:gradient",
                *tensor([2], [0, 0]),
            ]
        )
        with self.assertRaises(GateError):
            verify([row], parse(bad, ["case"]), "native")

    def test_reset_that_keeps_the_gradient_is_rejected(self):
        item = contribution([], [word(5)])
        row = case([], [item], [])
        bad = self.frame(
            [
                "S3:accumulated",
                "S3:grad:some",
                *tensor([], [word(5)]),
                "S3:reset",
                "S3:grad:some",
                *tensor([], [word(5)]),
                "S3:grad:some",
                *tensor([], [word(5)]),
                "S3:gradient",
                *tensor([], [word(5)]),
            ]
        )
        with self.assertRaises(GateError):
            verify([row], parse(bad, ["case"]), "native")


class StageRegistrationTests(unittest.TestCase):
    def test_s3_gate_registers_lifecycle_and_its_controls(self):
        self.assertIn("lifecycle", SUBGATES)
        controls = [
            name for name in PROOF_CONTROLS if name.startswith(("grad-", "reset-"))
        ]
        self.assertEqual(len(controls), 2)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_runtime_controls_mutate_the_public_accessors(self):
        source = (ROOT / "src/autograd/gradient.bend").read_text()
        self.assertEqual(
            set(RUNTIME_CONTROLS), {"grad-held-reads-none", "reset-keeps-held"}
        )
        for original, mutated in RUNTIME_CONTROLS.values():
            self.assertEqual(source.count(original), 1)
            self.assertNotEqual(original, mutated)
        harness = (ROOT / "src/autograd/harness.bend").read_text()
        self.assertIn("Gradient.grad(slot)", harness)
        self.assertIn("Gradient.reset(held)", harness)


class LifecycleRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="s3-lifecycle-") as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))
        self.assertEqual(summary["runtime_controls"], list(RUNTIME_CONTROLS))


if __name__ == "__main__":
    unittest.main()
