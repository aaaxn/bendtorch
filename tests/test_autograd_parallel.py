import math
import unittest

from tools.autograd_parallel import CONTROLS, cases, fired, pool, same_bytes, spread
from tools.common import ROOT, GateError
from tools.tensor_parallel import THREAD_COUNTS


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

    def test_a_run_that_ignores_the_thread_count_fails(self):
        for threads in THREAD_COUNTS:
            pool(threads, f"autograd:threads:{threads}\n")
        # Every run forced onto one thread, whatever count it was given.
        with self.assertRaisesRegex(
            GateError, r"at 8 threads reported a pool of \['1'\]"
        ):
            for threads in THREAD_COUNTS:
                pool(threads, "autograd:threads:1\n")
        with self.assertRaisesRegex(GateError, r"reported a pool of \[\]"):
            pool(1, "")

    def test_every_control_has_one_anchor_per_edit_and_a_script(self):
        named = {name for case in cases() for name in case["forks"]}
        self.assertEqual(named, set(CONTROLS))
        for name, (path, poison, unforked) in CONTROLS.items():
            text = (ROOT / path).read_text()
            # The unforked twin edits the poisoned source.
            for original, mutated in poison + unforked:
                with self.subTest(control=name, anchor=original):
                    self.assertEqual(text.count(original), 1)
                    text = text.replace(original, mutated)
            with self.subTest(control=name):
                self.assertNotIn("Elementwise.split", "".join(m for _, m in poison))
                self.assertIn("wrong_split", text)

    def test_each_control_poisons_only_its_own_build(self):
        # No two controls edit the same call.
        anchors = [
            original for _, poison, _ in CONTROLS.values() for original, _ in poison[1:]
        ]
        self.assertEqual(len(anchors), len(set(anchors)))

    def test_a_build_that_stops_forking_fails(self):
        # The unforked twin of a control changes no script; were that the
        # control's result, the gate would fail on every script that forks
        # the build.
        selected = cases()
        for name in CONTROLS:
            forking = [case["id"] for case in selected if name in case["forks"]]
            with self.subTest(control=name):
                fired(name, selected, forking)
                with self.assertRaisesRegex(GateError, "the forked build never ran"):
                    fired(name, selected, [])


if __name__ == "__main__":
    unittest.main()
