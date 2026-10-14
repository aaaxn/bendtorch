import struct
import sys
import tempfile
import unittest
from pathlib import Path

from tools.common import Evidence, GateError, check_inventory, check_omissions
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
