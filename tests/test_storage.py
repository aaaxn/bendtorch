import tempfile
import unittest
from pathlib import Path

from tools.common import ROOT, GateError
from tools.s1_storage import (
    cases,
    expected,
    instrument_copy_audit,
    run_stage,
    verify_output,
)


class StorageOracleTests(unittest.TestCase):
    def test_fixture_inventory_is_not_empty_or_truncated(self):
        requests = cases()
        self.assertEqual(len(requests), 55)
        self.assertEqual(len(expected(requests)), 2322)
        self.assertIn(1 << 20, [case["length"] for case in requests])

    def test_errors_and_copy_are_independent_of_implementation(self):
        request = {
            "length": 1,
            "bits": 7,
            "steps": [
                {"op": "copy"},
                {"op": "write", "index": 0, "bits": 11},
                {"op": "read_copy", "index": 0},
                {"op": "read", "index": 1},
            ],
        }
        self.assertEqual(
            expected([request]), ["new:1", "copy", "ok", "v:7", "e:OutOfBounds"]
        )

    def test_missing_or_wrong_observations_fail(self):
        for actual in ("", "v:1\n"):
            with self.assertRaises(GateError):
                verify_output(actual, ["v:2"], "native")

    def test_js_nan_payload_difference_is_explicit_not_a_skip(self):
        verify_output("v:2143289344\n", ["v:2139095041"], "javascript")
        with self.assertRaises(GateError):
            verify_output("v:2139095040\n", ["v:2139095041"], "javascript")
        with self.assertRaises(GateError):
            verify_output("v:2143289344\n", ["v:2139095041"], "native")

    def test_ambiguous_or_missing_instrumentation_is_rejected(self):
        with self.assertRaises(GateError):
            instrument_copy_audit("int main() { return 0; }")


class StorageRuntimeTests(unittest.TestCase):
    def test_actual_bend_storage_with_output_outside_the_repository(self):
        with tempfile.TemporaryDirectory(prefix="storage-unit-") as directory:
            output = Path(directory) / "run"
            self.assertFalse(output.is_relative_to(ROOT))
            run_stage(ROOT / ".tools", output)


if __name__ == "__main__":
    unittest.main()
