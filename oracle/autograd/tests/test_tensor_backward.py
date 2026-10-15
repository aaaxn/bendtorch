import json
import unittest

import generate
import tensor_backward


class TensorBackwardOracleTests(unittest.TestCase):
    def test_regeneration_is_byte_identical(self) -> None:
        generate.configure_cpu()
        identity = generate.observed_identity()
        self.assertEqual(identity, generate.EXPECTED_IDENTITY)
        data = generate.json_bytes(tensor_backward.build(identity))
        self.assertEqual(data, tensor_backward.OUTPUT.read_bytes())

    def test_snapshot_round_trips_words(self) -> None:
        words = [0x80000000, 0x00000001, 0x7FC01234, 0x7F800000, 0x3F800000, 0]
        value = generate.tensor(words, [2, 3])
        lines = tensor_backward.snapshot(value)
        self.assertEqual(lines[:2], ["tensor:shape:[2, 3]", "tensor:count:6"])
        self.assertEqual([int(line.split(":")[-1]) for line in lines[2:]], words)

    def test_unused_leaf_reads_none_and_unknown_ids_raise_index_error(self) -> None:
        document = json.loads(tensor_backward.OUTPUT.read_text())
        rows = {row["id"]: row for row in document["backward"]}
        lines = rows["seed_selected_output"]["lines"]
        self.assertEqual(lines.count("autograd:grad:none"), 2)
        self.assertIn("autograd:rejected:IndexError", rows["unknown_output"]["lines"])

    def test_a_non_scalar_output_is_seeded_with_ones(self) -> None:
        document = json.loads(tensor_backward.OUTPUT.read_text())
        rows = {row["id"]: row for row in document["backward"]}
        lines = rows["seed_non_scalar"]["lines"]
        start = len(lines) - 1 - lines[::-1].index("autograd:grad")
        self.assertEqual(lines[start + 1], "tensor:shape:[2, 3]")
        self.assertEqual(lines[start + 3 : start + 9], ["tensor:word:1065353216"] * 6)


if __name__ == "__main__":
    unittest.main()
