"""The comparator's own checks, on outputs written here.

Run from oracle/:
  uv run --frozen python -I -m unittest discover -s mnist -p 'test_*.py'
"""

import math
import struct
import unittest

from compare import TEST_ROWS, compared

STEPS = 937
NAN = 0x7FC00000
INFINITY = 0x7F800000


def word(number: float) -> int:
    return struct.unpack("<I", struct.pack("<f", number))[0]


def losses() -> list[int]:
    return [word(2.3 * math.exp(-k / 300)) for k in range(STEPS)]


def output(words: list[int], correct: int = 9242, rows: int = TEST_ROWS) -> str:
    lines = [f"loss {k} {number}" for k, number in enumerate(words)]
    return "\n".join([*lines, f"correct {correct} {rows}"]) + "\n"


class CompareTests(unittest.TestCase):
    def test_equal_outputs_pass(self):
        result = compared(output(losses()), output(losses()), STEPS)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["equal"], STEPS)

    def test_a_non_finite_loss_fails_at_any_step_on_either_side(self):
        for step in (0, STEPS // 2, STEPS - 1):
            for bad in (NAN, INFINITY, INFINITY | 0x80000000):
                for side in ("bend", "torch"):
                    with self.subTest(step=step, word=hex(bad), side=side):
                        changed = losses()
                        changed[step] = bad
                        good, bad_text = output(losses()), output(changed)
                        if side == "bend":
                            result = compared(bad_text, good, STEPS)
                        else:
                            result = compared(good, bad_text, STEPS)
                        self.assertEqual(len(result["failures"]), 1)
                        self.assertIn(
                            f"step {step}: a loss is not finite", result["failures"][0]
                        )

    def test_every_step_is_checked_on_its_own(self):
        for step, scale, fails in (
            (50, 1 + 2e-4, True),
            (50, 1 + 5e-5, False),
            (STEPS - 1, 1 + 2e-2, True),
            (STEPS - 1, 1 + 5e-3, False),
        ):
            with self.subTest(step=step, scale=scale):
                changed = losses()
                value = struct.unpack("<f", struct.pack("<I", changed[step]))[0]
                changed[step] = word(value * scale)
                result = compared(output(changed), output(losses()), STEPS)
                self.assertEqual(bool(result["failures"]), fails, result["failures"])

    def test_the_count_must_be_of_all_test_rows(self):
        for line, wanted in (
            ("correct 9242 12", "not N of 10000"),
            ("correct 10001 10000", "not N of 10000"),
            ("correct -1 10000", "not a result line"),
            ("correct 9242", "not a result line"),
        ):
            with self.subTest(line=line):
                text = output(losses()).replace("correct 9242 10000", line)
                result = compared(text, output(losses()), STEPS)
                self.assertEqual(len(result["failures"]), 1)
                self.assertIn(wanted, result["failures"][0])

    def test_one_count_line_and_every_step_in_order(self):
        good = output(losses())
        for text, wanted in (
            (good + "correct 9242 10000\n", "unexpected line"),
            (good.replace("correct 9242 10000\n", ""), "no correct line"),
            (good.replace("loss 3 ", "loss 4 ", 1), "out of order"),
            (output(losses()[:-1]), "936 loss lines"),
            (good + "1065353216\n", "not a result line"),
        ):
            with self.subTest(wanted=wanted):
                result = compared(text, good, STEPS)
                self.assertEqual(len(result["failures"]), 1)
                self.assertIn(wanted, result["failures"][0])

    def test_counts_outside_the_band_fail(self):
        self.assertEqual(
            compared(output(losses(), 9215), output(losses()), STEPS)["failures"], []
        )
        result = compared(output(losses(), 9214), output(losses()), STEPS)
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("more than 27 rows apart", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
