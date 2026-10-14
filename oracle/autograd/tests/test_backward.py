import json
import unittest

import backward
import generate


class S3BackwardOracleTests(unittest.TestCase):
    def test_regeneration_is_byte_identical(self) -> None:
        generate.configure_cpu()
        identity = generate.observed_identity()
        self.assertEqual(identity, generate.EXPECTED_IDENTITY)
        data = generate.json_bytes(backward.build(identity))
        self.assertEqual(data, backward.OUTPUT.read_bytes())

    def test_scalar_words_round_trip(self) -> None:
        for word in (0x80000000, 0x00000001, 0x7FC01234, 0x7F800000, 0x3F800000):
            self.assertEqual(backward.bits(backward.scalar(word)), word)

    def test_unused_leaf_and_newer_node_read_none(self) -> None:
        document = json.loads(backward.OUTPUT.read_text())
        rows = {row["id"]: row for row in document["backward"]}
        lines = rows["seed_selected_output"]["lines"]
        self.assertEqual(lines[-5], "S3:grad:none")
        self.assertEqual(lines[-1], "S3:grad:none")

    def test_unknown_ids_raise_index_error(self) -> None:
        document = json.loads(backward.OUTPUT.read_text())
        rows = {row["id"]: row for row in document["backward"]}
        self.assertIn("S3:rejected:IndexError", rows["unknown_output"]["lines"])


if __name__ == "__main__":
    unittest.main()
