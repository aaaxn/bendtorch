import hashlib
import json
import unittest

import generate
import supplement
import torch


class SupplementTests(unittest.TestCase):
    def test_regeneration_is_byte_identical(self) -> None:
        if torch.get_num_interop_threads() != 1:
            generate.configure_cpu()
        torch.set_num_threads(1)
        identity = generate.observed_identity()
        data = generate.json_bytes(supplement.build(identity))
        self.assertEqual(data, supplement.OUTPUT.read_bytes())

    def test_policy_v3_extends_frozen_v2(self) -> None:
        self.assertEqual(
            hashlib.sha256(generate.POLICY_PATH.read_bytes()).hexdigest(),
            generate.POLICY_SHA256,
        )
        v2 = generate.policy()
        v3 = supplement.policy_v3()
        self.assertEqual(v3["supersedes"]["sha256"], generate.POLICY_SHA256)
        for name, rules in v2["policies"].items():
            for key, value in rules.items():
                self.assertEqual(v3["policies"][name][key], value)

    def test_pytorch_is_correctly_rounded_and_zero_signs_agree(self) -> None:
        document = json.loads(supplement.OUTPUT.read_text())
        for case in document["cases"]:
            with self.subTest(case=case["id"]):
                self.assertIsNone(supplement.v3_violation(case))
        base = json.loads(supplement.BASE_CORPUS.read_text())
        for group in ("finite_cases", "edge_cases", "nonfinite_cases"):
            for case in base[group]:
                if case["comparison_policy"] != "bounded_elementwise":
                    continue
                self.assertEqual(
                    case["outputs"]["result"]["f32_bits"],
                    case["independent_reference"]["result"]["f32_bits"],
                )

    def test_base_corpus_hash_is_bound(self) -> None:
        document = json.loads(supplement.OUTPUT.read_text())
        self.assertEqual(
            document["base_corpus"]["sha256"],
            hashlib.sha256(supplement.BASE_CORPUS.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
