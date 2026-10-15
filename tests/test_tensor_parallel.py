import unittest

from tools.common import GateError
from tools.tensor_parallel import (
    PRODUCTS,
    ROUNDING,
    check_lines,
    depth,
    expected_splits,
    matmul_splits,
)


def probe_output(**overrides: tuple[int, int]) -> str:
    lines = []
    for name in PRODUCTS:
        splits, differing = overrides.get(name, (expected_splits(name), 0))
        lines.append(f"{name} splits {splits} differing {differing} hash 7")
    return "\n".join(lines) + "\n"


class ParallelProbeTests(unittest.TestCase):
    def test_depth_is_the_storage_depth(self):
        for cells, expected in ((0, 0), (1, 0), (2, 1), (3, 2), (8192, 13), (8193, 14)):
            with self.subTest(cells=cells):
                self.assertEqual(depth(cells), expected)

    def test_matmul_splits_by_work(self):
        for cells, inner, expected in (
            (8192, 64, 6),
            (7680, 40, 5),
            (1024, 8, 0),
            (1024, 16, 1),
            (4, 3, 0),
            (8192, 784, 6),
            (32768, 784, 6),
            (1 << 20, 1024, 6),
            (2, 1 << 20, 1),
            (8192, 0, 0),
        ):
            with self.subTest(cells=cells, inner=inner):
                self.assertEqual(matmul_splits(depth(cells), inner), expected)

    def test_probe_builds_fork(self):
        self.assertTrue(any(expected_splits(name) for name in PRODUCTS))

    def test_rounding_build_forks(self):
        self.assertIn(ROUNDING, PRODUCTS)
        self.assertEqual(expected_splits(ROUNDING), 4)

    def test_check_lines_accepts_the_expected_output(self):
        self.assertEqual(
            check_lines(probe_output()),
            {name: expected_splits(name) for name in PRODUCTS},
        )

    def test_check_lines_rejects_a_differing_word(self):
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"matmul-128x64x64": (6, 1)}))

    def test_check_lines_rejects_a_wrong_split(self):
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"matmul-128x64x64": (0, 0)}))

    def test_check_lines_rejects_missing_or_extra_lines(self):
        complete = probe_output()
        for text in (
            "\n".join(complete.splitlines()[1:]) + "\n",
            complete + "matmul-128x64x64 splits 2 differing 0 hash 7\n",
            complete + "garbage\n",
            "",
        ):
            with self.subTest(text=text):
                with self.assertRaises(GateError):
                    check_lines(text)


if __name__ == "__main__":
    unittest.main()
