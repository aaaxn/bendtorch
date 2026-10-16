"""The bench command: its output parsing, its output check, and one real run.

The real run builds experiments/bench/bench.bend natively, runs the small
`add` workload at one thread and at two, and checks that the command passes
with the stored digests and fails when a stored digest is changed.
"""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from tools.bench import (
    EXPECTED,
    WORKLOADS,
    Sample,
    build,
    expected_digests,
    main,
    parse,
    verdict,
)

CPU = str(min(os.sched_getaffinity(0)))
ADD = next(w for w in WORKLOADS if w.name == "add" and w.n == 1 << 14)


class ParseTests(unittest.TestCase):
    def test_reads_the_time_and_the_digest_among_other_lines(self):
        self.assertEqual(
            parse("warning: x\nms 83\ndigest 3957881504\n"), (83, "3957881504")
        )

    def test_a_run_without_a_digest_is_refused(self):
        with self.assertRaises(KeyError):
            parse("ms 83\n")


class VerdictTests(unittest.TestCase):
    def test_every_run_must_print_the_expected_digest(self):
        expected = {"add 16384": "7"}
        same = [Sample(1, "7", 1.0), Sample(2, "7", 1.0)]
        self.assertEqual(verdict(ADD, same, expected), "ok")
        changed = verdict(ADD, [Sample(1, "7", 1.0), Sample(2, "8", 1.0)], expected)
        self.assertTrue(changed.startswith("CHANGED"))
        self.assertEqual(verdict(ADD, same, {}), "no expected digest")

    def test_every_workload_has_a_stored_digest(self):
        stored = expected_digests(EXPECTED)
        for w in WORKLOADS:
            with self.subTest(workload=w.name, n=w.n):
                self.assertIn(f"{w.name} {w.n}", stored)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.binary = build(Path(__file__).resolve().parents[1], cls.root / "bench")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def bench(self, expected: Path) -> tuple[int, str]:
        out = io.StringIO()
        arguments = ["--only", "add", "--runs", "1", "--cpu", CPU, "--cpus", CPU]
        arguments += [
            "--threads",
            "2",
            "--binary",
            str(self.binary),
            "--expected",
            str(expected),
        ]
        with contextlib.redirect_stdout(out):
            code = main(arguments)
        return code, out.getvalue()

    def test_the_stored_digests_pass(self):
        code, out = self.bench(EXPECTED)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count(" ok\n"), 2, out)

    def test_a_changed_digest_fails(self):
        stored = EXPECTED.read_text().replace("add 16384 ", "add 16384 1", 1)
        changed = self.root / "changed.txt"
        changed.write_text(stored)
        code, out = self.bench(changed)
        self.assertEqual(code, 1, out)
        self.assertIn("CHANGED", out)


if __name__ == "__main__":
    unittest.main()
