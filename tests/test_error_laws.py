import json
import unittest

from tools.autograd import ERROR_CONTROLS
from tools.common import ROOT

ERROR_LAWS = {
    "verification/shape.json": (
        "LAWS.",
        ["flatten_rank_error", "flatten_bounds_error"],
    ),
    "verification/tensor.json": (
        "matmul_LAWS.",
        ["matmul_left_rank_error", "matmul_right_rank_error", "matmul_inner_error"],
    ),
    "verification/autograd.json": (
        "",
        [
            "broadcast_LAWS.broadcast_shape_error",
            "broadcast_LAWS.broadcast_not_target",
            "reduce_LAWS.reduce_shape_error",
            "reduce_LAWS.reduce_not_reducible",
            "gradient_LAWS.accumulate_wrong_shape_refused",
            "backward_LAWS.value_unknown",
            "backward_LAWS.grad_unknown",
            "backward_LAWS.record_unknown_left",
            "backward_LAWS.record_unknown_right",
            "backward_LAWS.neg_unknown",
            "backward_LAWS.backward_unknown",
            "backward_LAWS.backward_dangling",
            "backward_LAWS.backward_rejects_unbound_operand",
            "tensor_backward_LAWS.tensor_backward_unknown",
            "tensor_backward_LAWS.tensor_backward_dangling",
            "sum_dims_LAWS.sum_dims_axis_out_of_bounds",
            "sum_dims_LAWS.sum_dims_duplicate_axis",
            "sum_dims_LAWS.sum_dims_plan_refused",
        ],
    ),
}


class ErrorLawTests(unittest.TestCase):
    def test_error_laws_are_required_exports(self):
        for path, (prefix, names) in ERROR_LAWS.items():
            required = json.loads((ROOT / path).read_text())["required_exports"]
            for name in names:
                with self.subTest(law=name):
                    self.assertIn(prefix + name, required)

    def test_error_controls_apply_once_and_name_a_proof(self):
        self.assertEqual(len(ERROR_CONTROLS), 17)
        for name, (path, original, mutated, root, location) in ERROR_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual((ROOT / path).read_text().count(original), 1)
                self.assertNotEqual(original, mutated)
                self.assertTrue((ROOT / root).is_file())
                self.assertTrue(location)


if __name__ == "__main__":
    unittest.main()
