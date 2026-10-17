import unittest

from tools.common import GateError
from tools.tensor_parallel import (
    BUILDS,
    ELEMENTWISE,
    LARGE,
    PRODUCTS,
    REDUCTIONS,
    ROUNDING,
    ROUNDING_BUILDS,
    check_lines,
    contiguous,
    depth,
    expected_splits,
    matmul_splits,
    product_splits,
    words,
)


def probe_output(**overrides: tuple[int, int]) -> str:
    lines = []
    for name in BUILDS:
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

    def test_product_splits_at_most_four(self):
        for cells, inner, expected in (
            (8192, 64, 4),
            (7680, 40, 4),
            (1024, 16, 1),
            (8192, 784, 4),
            (4, 3, 0),
        ):
            with self.subTest(cells=cells, inner=inner):
                self.assertEqual(product_splits(depth(cells), inner), expected)

    def test_probe_builds_fork(self):
        self.assertTrue(any(expected_splits(name) for name in PRODUCTS))

    def test_rounding_build_forks(self):
        self.assertIn(ROUNDING, PRODUCTS)
        self.assertEqual(expected_splits(ROUNDING), 4)

    def test_rounding_reductions_fork(self):
        self.assertEqual(len(ROUNDING_BUILDS), 4)
        for name in ROUNDING_BUILDS:
            with self.subTest(name=name):
                self.assertTrue(expected_splits(name))

    def test_only_contiguous_fibers_fork(self):
        self.assertTrue(contiguous((256, 64), (256, 1)))
        self.assertTrue(contiguous((64, 16, 32), (64, 1, 1)))
        self.assertTrue(contiguous((4, 1, 8), (4, 1, 1)))
        self.assertTrue(contiguous((8, 1), (1, 1)))
        self.assertTrue(contiguous((5, 7), ()))
        self.assertFalse(contiguous((256, 128), (1, 128)))
        self.assertFalse(contiguous((16, 256, 256), (256, 256)))
        self.assertFalse(contiguous((64, 16, 32), (16, 1)))
        self.assertFalse(contiguous((8, 1, 4), (1, 1, 4)))

    def test_reduction_splits_by_terms_when_contiguous(self):
        self.assertEqual(
            {name: expected_splits(name) for name in REDUCTIONS},
            {
                "reduce-256x128-to-1x128-rounding": 0,
                "reduce-256x64-to-256x1-rounding": 1,
                "reduce-64x16x32-to-16x1-rounding": 0,
                "reduce-64x16x32-to-64x1x1-rounding": 2,
                "reduce-6x5x4-to-5x1": 0,
                "reduce-130x256-to-1x256-rounding": 0,
                "reduce-256x130-to-256x1-rounding": 2,
                "reduce-65536-to-scalar-rounding": 0,
            },
        )

    def test_elementwise_builds_fork_with_padding(self):
        self.assertEqual(
            {name: expected_splits(name) for name in ELEMENTWISE},
            {name: 2 for name in ELEMENTWISE},
        )

    def test_large_builds_exceed_one_subtree(self):
        self.assertTrue(set(ELEMENTWISE) <= set(LARGE))
        for name in LARGE:
            with self.subTest(name=name):
                self.assertGreater(words(name), 1 << 15)

    def test_check_lines_rejects_an_unforked_elementwise_build(self):
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"mul-100000-own-rounding": (0, 0)}))

    def test_check_lines_accepts_the_expected_output(self):
        self.assertEqual(
            check_lines(probe_output()),
            {name: expected_splits(name) for name in BUILDS},
        )

    def test_check_lines_rejects_a_differing_word(self):
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"matmul-128x64x64": (4, 1)}))
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"reduce-256x64-to-256x1-rounding": (1, 3)}))

    def test_check_lines_rejects_an_unforked_rounding_reduction(self):
        with self.assertRaises(GateError):
            check_lines(probe_output(**{"reduce-64x16x32-to-64x1x1-rounding": (0, 0)}))

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
