import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_broadcast import parse, word
from tools.autograd_reduce import NOT_REDUCIBLE
from tools.autograd_reduce import reference as reduce_reference
from tools.autograd_sum_to_size import (
    NEGATIVE_ZERO,
    cases,
    checker,
    reference,
    run_gate,
)
from tools.common import ROOT, GateError


def case(shape: list[int], target: list[int], words: list[int]) -> dict:
    return {"id": "case", "shape": shape, "target": target, "words": words}


def output(rows: list[int], shape: list[int], retained: list[int]) -> str:
    lines = [
        "S3:begin:case",
        "S3:retained",
        f"tensor:shape:{json.dumps(shape)}",
        f"tensor:count:{len(retained)}",
        *[f"tensor:word:{item}" for item in retained],
        "S3:output",
        f"tensor:shape:{json.dumps(shape)}",
        f"tensor:count:{len(rows)}",
        *[f"tensor:word:{item}" for item in rows],
        "S3:end",
    ]
    return "\n".join(lines) + "\n"


class SumToSizeReferenceTests(unittest.TestCase):
    # PyTorch 2.14.1+cpu: for a nonscalar x, x.sum_to_size(x.shape) is x itself.
    # For a scalar, the direct method reduces and returns +0.0, but autograd
    # does not reduce, so the gradient of a same-shape operand keeps -0.0
    # (80000000) for every shape here, scalar included. sum_to_size follows
    # autograd; reduce_to folds from +0.0.
    def test_identity_keeps_negative_zero_like_pytorch(self):
        for shape in ([], [1], [2, 3], [1, 3]):
            words = [NEGATIVE_ZERO] * math.prod(shape)
            with self.subTest(shape=shape):
                self.assertEqual(reference(case(shape, shape, words))["words"], words)
                self.assertEqual(
                    reduce_reference(case(shape, shape, words))["words"],
                    [0] * len(words),
                )

    def test_reduction_starts_from_positive_zero_like_pytorch(self):
        self.assertEqual(
            reference(case([1, 3], [3], [NEGATIVE_ZERO] * 3))["words"], [0, 0, 0]
        )
        self.assertEqual(reference(case([1], [], [NEGATIVE_ZERO]))["words"], [0])

    def test_other_targets_match_reduce_to(self):
        for row in cases():
            if row["shape"] != row["target"]:
                self.assertEqual(reference(row), reduce_reference(row))
        self.assertEqual(
            reference(case([3], [2, 3], [word(1)] * 3)), {"error": NOT_REDUCIBLE}
        )

    def test_identity_rejects_a_folded_word(self):
        row = case([1], [1], [NEGATIVE_ZERO])
        check = checker([row])
        check(output([NEGATIVE_ZERO], [1], [NEGATIVE_ZERO]), "native")
        with self.assertRaises(GateError):
            check(output([0], [1], [NEGATIVE_ZERO]), "native")

    def test_identity_compares_nan_payloads_exactly_on_native(self):
        row = case([1], [1], [0x7FC01234])
        with self.assertRaises(GateError):
            checker([row])(output([0x7FC00000], [1], [0x7FC01234]), "native")

    def test_parse_keeps_case_order(self):
        rows = parse(output([1], [1], [1]), ["case"])
        self.assertEqual(rows[0]["outcome"]["words"], [1])


class GateRegistrationTests(unittest.TestCase):
    def test_autograd_gate_registers_sum_to_size_and_its_laws(self):
        self.assertIn("sum_to_size", SUBGATES)
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(sum("sum_to_size_LAWS" in name for name in required), 3)
        self.assertEqual(sum("sum_to_size_fold_LAWS" in name for name in required), 3)
        controls = [name for name in PROOF_CONTROLS if name.startswith("sum-to-size-")]
        self.assertEqual(len(controls), 3)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_inventory_has_identity_and_reduction_cases(self):
        selected = cases()
        self.assertEqual(len({row["id"] for row in selected}), len(selected))
        identities = [row for row in selected if row["shape"] == row["target"]]
        self.assertGreater(len(identities), 8)
        self.assertGreater(len(selected) - len(identities), 400)


class SumToSizeRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="autograd-sum-to-size-") as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))


if __name__ == "__main__":
    unittest.main()
