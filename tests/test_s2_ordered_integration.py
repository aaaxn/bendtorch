import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.common import GateError, digest
from tools.s2_runtime import (
    ORDERED_CONTROLS,
    ORDERED_CONTROLS_SHA256,
    ORDERED_REFERENCE,
    ORDERED_REFERENCE_SHA256,
    load_corpus,
)


class OrderedFreezeTests(unittest.TestCase):
    def test_new_records_match_the_registered_protected_hashes(self):
        self.assertEqual(digest(ORDERED_REFERENCE), ORDERED_REFERENCE_SHA256)
        self.assertEqual(digest(ORDERED_CONTROLS), ORDERED_CONTROLS_SHA256)
        self.assertEqual(len(load_corpus()["finite_cases"]), 141)

    def test_reference_or_control_drift_stops_corpus_loading(self):
        for name, source in (
            ("ORDERED_REFERENCE", ORDERED_REFERENCE),
            ("ORDERED_CONTROLS", ORDERED_CONTROLS),
        ):
            with tempfile.TemporaryDirectory() as directory:
                changed = Path(directory) / "changed.json"
                changed.write_bytes(source.read_bytes() + b"\n")
                with (
                    self.subTest(source=name),
                    patch(f"tools.s2_runtime.{name}", changed),
                    self.assertRaisesRegex(GateError, "ordered reference or controls"),
                ):
                    load_corpus()


if __name__ == "__main__":
    unittest.main()
