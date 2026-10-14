import json
import unittest

from tools.common import ROOT
from tools.s3 import ERROR_CONTROLS

ERROR_LAWS = {
    "verification/s1.json": (
        "_2e_._2f_.._2f_src_2f_shape_2f_LAWS.",
        ["flatten_rank_error", "flatten_bounds_error"],
    ),
    "verification/s2.json": (
        "_2e_._2f_.._2f_src_2f_tensor_2f_matmul_2d_LAWS.",
        ["matmul_left_rank_error", "matmul_right_rank_error", "matmul_inner_error"],
    ),
    "verification/s3.json": (
        "_2e_._2f_.._2f_src_2f_autograd_2f_",
        [
            "broadcast_2d_LAWS.broadcast_shape_error",
            "broadcast_2d_LAWS.broadcast_not_target",
            "reduce_2d_LAWS.reduce_shape_error",
            "reduce_2d_LAWS.reduce_not_reducible",
            "gradient_2d_LAWS.accumulate_wrong_shape_refused",
            "backward_2d_LAWS.value_unknown",
            "backward_2d_LAWS.grad_unknown",
            "backward_2d_LAWS.record_unknown_left",
            "backward_2d_LAWS.record_unknown_right",
            "backward_2d_LAWS.neg_unknown",
            "backward_2d_LAWS.backward_unknown",
            "backward_2d_LAWS.backward_dangling",
            "backward_2d_LAWS.backward_rejects_unbound_operand",
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
        self.assertEqual(len(ERROR_CONTROLS), 12)
        for name, (path, original, mutated, root, location) in ERROR_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual((ROOT / path).read_text().count(original), 1)
                self.assertNotEqual(original, mutated)
                self.assertTrue((ROOT / root).is_file())
                self.assertTrue(location)


if __name__ == "__main__":
    unittest.main()
