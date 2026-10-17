import json
import unittest

from tools.autograd import ERROR_CONTROLS, PROOF_CONTROLS
from tools.autograd_backward import NEGATIVE_ZERO
from tools.autograd_broadcast import word
from tools.autograd_tensor_backward import (
    AXIS_OUT_OF_BOUNDS,
    DUPLICATE_AXIS,
    RUNTIME_CONTROLS,
    dims,
    sum_dims,
    sum_dims_plan,
)
from tools.common import ROOT


def words(*numbers: float) -> list[int]:
    return [word(number) for number in numbers]


GRID = ([2, 3], words(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


class SumDimsReferenceTests(unittest.TestCase):
    def test_one_axis_with_and_without_keepdim(self):
        self.assertEqual(sum_dims(GRID, dims(0, [1])), ([2], words(6.0, 15.0)))
        self.assertEqual(
            sum_dims(GRID, dims(0, [0], True)), ([1, 3], words(5.0, 7.0, 9.0))
        )

    def test_no_axes_sum_every_axis_as_pytorch_dim_empty(self):
        self.assertEqual(sum_dims(GRID, dims(0, [])), ([], words(21.0)))
        self.assertEqual(sum_dims(GRID, dims(0, [], True)), ([1, 1], words(21.0)))
        self.assertEqual(sum_dims(GRID, dims(0, [1, 0])), ([], words(21.0)))

    def test_rank_zero_has_axis_zero(self):
        self.assertEqual(sum_dims_plan([], [0]), ([], []))
        self.assertEqual(sum_dims_plan([], [1]), AXIS_OUT_OF_BOUNDS)

    def test_a_folded_axis_turns_negative_zero_positive(self):
        column = ([2, 1], [NEGATIVE_ZERO, word(-3.0)])
        self.assertEqual(sum_dims(column, dims(0, [1])), ([2], [0, word(-3.0)]))

    def test_empty_axes_give_zeros_or_nothing(self):
        empty = ([0, 3], [])
        self.assertEqual(sum_dims(empty, dims(0, [0])), ([3], [0, 0, 0]))
        self.assertEqual(sum_dims(empty, dims(0, [1], True)), ([0, 1], []))

    def test_refusals_check_every_range_before_repeats(self):
        self.assertEqual(sum_dims_plan([2, 3], [2]), AXIS_OUT_OF_BOUNDS)
        self.assertEqual(sum_dims_plan([2, 3], [0, 0]), DUPLICATE_AXIS)
        self.assertEqual(sum_dims_plan([2, 3], [0, 0, 5]), AXIS_OUT_OF_BOUNDS)
        self.assertEqual(sum_dims_plan([2, 3], [4294967295]), AXIS_OUT_OF_BOUNDS)


class SumDimsRegistrationTests(unittest.TestCase):
    def test_autograd_gate_requires_the_laws_and_their_controls(self):
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        laws = (ROOT / "src/autograd/sum-dims-LAWS.bend").read_text()
        names = [
            line.removeprefix("law ").removesuffix(":")
            for line in laws.splitlines()
            if line.startswith("law ")
        ]
        self.assertEqual(len(names), 17)
        self.assertEqual(
            [name for name in required if name.startswith("sum_2d_dims_2d_LAWS.")],
            [f"sum_2d_dims_2d_LAWS.{name}" for name in names],
        )
        self.assertIn(
            "tensor_2d_backward_2d_LAWS.tensor_sum_dims_vjp_pointwise", required
        )
        proofs = [name for name in PROOF_CONTROLS if "sum-dims-" in name]
        self.assertEqual(len(proofs), 8)
        errors = {name: v for name, v in ERROR_CONTROLS.items() if "sum-dims-" in name}
        self.assertEqual(len(errors), 3)
        proof = (ROOT / "src/autograd/sum-dims-PROOF.bend").read_text()
        for _, original, mutated, _, location in errors.values():
            self.assertIn(f"def {location}(", proof)
            self.assertNotEqual(original, mutated)
        runtime = [name for name in RUNTIME_CONTROLS if name.startswith("sum-dims-")]
        self.assertEqual(len(runtime), 3)
        for name in proofs + list(errors):
            path, original = {**PROOF_CONTROLS, **ERROR_CONTROLS}[name][:2]
            self.assertEqual((ROOT / path).read_text().count(original), 1)


if __name__ == "__main__":
    unittest.main()
