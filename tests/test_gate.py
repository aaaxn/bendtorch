import json
import struct
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path

from tools.check import SUITES
from tools.common import (
    ROOT,
    Evidence,
    GateError,
    check_inventory,
    check_omissions,
    omitted_inventory_by_rewrite,
    parallel,
    set_jobs,
)
from tools.toolchain import F32_PROBE


class GateTests(unittest.TestCase):
    def test_inventory_rejects_omissions_and_empty_packages(self):
        for required in (["law"], [], ["law", "law"]):
            with self.subTest(required=required), self.assertRaises(GateError):
                check_inventory("other : *1 =\n  ()\n", required)

    def test_inventory_uses_declarations_not_mentions(self):
        with self.assertRaises(GateError):
            check_inventory("other : *1 =\n  law : *1\n", ["law"])
        check_inventory("law : *1 =\n  ()\n", ["law"])

    def test_omissions_remove_the_declaration_not_a_longer_name(self):
        text = "matmul_2d_core.construct : *1 =\n  ()\ncore.construct : *1 =\n  ()\n"
        check_omissions(text, ["core.construct"])
        twice = "core.construct : *1 =\n  ()\ncore.construct : *1 =\n  ()\n"
        with self.assertRaises(GateError):
            check_omissions(twice, ["core.construct"])

    def test_omission_scan_decides_what_the_rewrite_decides(self):
        texts = [
            "a : *1 =\n  ()\nb : *1 =\n  ()\n",
            "a : *1 =\na : *1 =\nb : *1 =\n",
            "a :b : x\na : *1 =\nb : *1 =\n",
            "a b : *1 =\na : *1 =\na b : *1 =\nb : *1",
            "x\ra : *1\nb : *1 =\r\na : 2 : 3\x0bb : *1\n",
            "a : a : *1\nb : *1\n a : *1\n",
            "b : *1\na : *1\r\n",
            "",
        ]
        for text in texts:
            for required in (["a"], ["b"], ["a", "b"], ["a b", "a"], ["a", "a :b"]):
                outcomes = []
                for check in (check_omissions, omitted_inventory_by_rewrite):
                    try:
                        check(text, required)
                        outcomes.append(None)
                    except GateError as error:
                        outcomes.append(str(error))
                with self.subTest(text=text, required=required):
                    self.assertEqual(outcomes[0], outcomes[1])

    def test_root_and_suites_cover_every_module(self):
        modules = sorted(path.parent.name for path in ROOT.glob("src/*/PROOF.bend"))
        self.assertTrue(modules)
        for root in ("PROOF", "LAWS"):
            text = (ROOT / f"{root}.bend").read_text()
            for module in modules:
                with self.subTest(root=root, module=module):
                    self.assertIn(f"import ./src/{module}/{root}.bend as ", text)
        self.assertLessEqual(set(modules), set(SUITES))

    def test_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            Evidence(path, Path(directory))
            with self.assertRaises(GateError):
                Evidence(path, Path(directory))

    def test_pass_marker_cannot_hide_a_failing_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Evidence(Path(directory) / "run", Path(directory))
            with self.assertRaises(GateError):
                evidence.run(
                    "spoof",
                    [
                        sys.executable,
                        "-c",
                        "print('ALL PROOFS CHECK'); raise SystemExit(1)",
                    ],
                    contains="ALL PROOFS CHECK",
                )
            self.assertFalse(evidence.records[0]["passed"])

    def test_wrong_reason_cannot_pass_a_negative_control(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Evidence(Path(directory) / "run", Path(directory))
            with self.assertRaises(GateError):
                evidence.run(
                    "wrong-reason",
                    [
                        sys.executable,
                        "-c",
                        "print('unrelated failure'); raise SystemExit(1)",
                    ],
                    expected_code=1,
                    contains="unsafe",
                )

    def test_parallel_steps_keep_the_sequential_record_order(self):
        def sleeper(evidence: Evidence, name: str, seconds: float) -> str:
            evidence.run(
                name, [sys.executable, "-c", f"import time; time.sleep({seconds})"]
            )
            return name

        def chain(evidence: Evidence) -> list[str]:
            # The slow first step makes "b" finish before this task's steps.
            return [sleeper(evidence, "a1", 0.3), sleeper(evidence, "a2", 0)]

        orders = {}
        for jobs in (1, 4):
            set_jobs(jobs)
            try:
                with tempfile.TemporaryDirectory() as directory:
                    evidence = Evidence(Path(directory) / "run", Path(directory))
                    sleeper(evidence, "first", 0)
                    results = parallel(
                        partial(chain, evidence),
                        partial(sleeper, evidence, "b", 0),
                    )
                    sleeper(evidence, "last", 0)
                    saved = evidence.directory / "report.json"
                    report = json.loads(saved.read_text())
            finally:
                set_jobs(1)
            self.assertEqual(results, [["a1", "a2"], "b"])
            orders[jobs] = [row["name"] for row in report["commands"]]
        self.assertEqual(orders[1], ["first", "a1", "a2", "b", "last"])
        self.assertEqual(orders[4], orders[1])

    def test_parallel_raises_the_first_failure_in_task_order(self):
        def fail(message: str) -> None:
            raise GateError(message)

        set_jobs(3)
        try:
            with self.assertRaisesRegex(GateError, "^one$"):
                parallel(lambda: None, lambda: fail("one"), lambda: fail("two"))
        finally:
            set_jobs(1)

    def test_every_required_reason_must_appear(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Evidence(Path(directory) / "run", Path(directory))
            command = [
                sys.executable,
                "-c",
                "print('SOME PROOFS FAIL'); raise SystemExit(1)",
            ]
            evidence.run(
                "verdict", command, expected_code=1, contains=("SOME PROOFS FAIL",)
            )
            with self.assertRaises(GateError):
                evidence.run(
                    "missing-location",
                    command,
                    expected_code=1,
                    contains=("SOME PROOFS FAIL", "In LAWS.tree_is_seq:"),
                )

    def test_probe_arithmetic_matches_ieee_single_rounding(self):
        for (label, native, javascript), value in zip(
            F32_PROBE[:4], (1.5 + 2.25, 1.5 - 2.25, 1.5 * 2.25, 1.5 / 2.25), strict=True
        ):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            self.assertEqual((native, javascript), (bits, bits), label)


if __name__ == "__main__":
    unittest.main()
