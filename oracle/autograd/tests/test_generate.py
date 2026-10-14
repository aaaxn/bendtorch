import hashlib
import json
import unittest

import generate


class S3OracleTests(unittest.TestCase):
    def test_regeneration_is_byte_identical(self) -> None:
        generate.configure_cpu()
        identity = generate.observed_identity()
        self.assertEqual(identity, generate.EXPECTED_IDENTITY)
        data = generate.json_bytes(generate.build(identity))
        self.assertEqual(data, generate.OUTPUT.read_bytes())

    def test_words_round_trip_signed_zero_and_nan_payloads(self) -> None:
        words = [0x80000000, 0x00000001, 0x7FC01234, 0xFFA00001, 0x3F800000]
        self.assertEqual(
            generate.words(generate.tensor(words, [5])),
            [f"{word:08x}" for word in words],
        )

    def test_large_results_store_a_digest(self) -> None:
        tensor = generate.tensor([0x3F800000] * 5000, [5000])
        observed = generate.observed(tensor)
        joined = " ".join(["3f800000"] * 5000)
        self.assertEqual(observed["count"], 5000)
        self.assertEqual(
            observed["sha256"], hashlib.sha256(joined.encode()).hexdigest()
        )

    def test_rejections_are_recorded(self) -> None:
        document = json.loads(generate.OUTPUT.read_text())
        rows = {row["id"]: row for row in document["broadcast"]}
        self.assertEqual(rows["incompatible"]["expand"], {"rejected": "RuntimeError"})


if __name__ == "__main__":
    unittest.main()
