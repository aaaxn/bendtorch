import math
import unittest

from tools.autograd_parallel import CONTROLS, cases, same_bytes, spread
from tools.common import ROOT, GateError


class AutogradParallelTests(unittest.TestCase):
    def test_spread_matches_the_bend_words(self):
        # What `main() = show(spread(4n, 7n))` printed with the pinned Bend2.
        self.assertEqual(spread(4, 7), [1082187769, 3221340146, 1065525227, 3204677604])

    def test_some_script_holds_a_tensor_above_2_to_the_15_cells(self):
        largest = max(
            math.prod(step["shape"])
            for case in cases()
            for step in case["steps"]
            if step["op"] == "leaf"
        )
        self.assertGreater(largest, 2**15)

    def test_outputs_that_differ_between_thread_counts_fail(self):
        self.assertEqual(same_bytes({1: "a\n", 8: "a\n"}), "a\n")
        with self.assertRaisesRegex(GateError, "at 8 threads differs"):
            same_bytes({1: "a\n", 8: "b\n"})

    def test_every_control_has_one_anchor_per_edit_and_a_script(self):
        named = {name for case in cases() for name in case["forks"]}
        self.assertEqual(named, set(CONTROLS))
        for name, (path, edits) in CONTROLS.items():
            text = (ROOT / path).read_text()
            for original, _ in edits:
                with self.subTest(control=name, anchor=original):
                    self.assertEqual(text.count(original), 1)


if __name__ == "__main__":
    unittest.main()
