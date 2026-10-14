import hashlib
import json
import unittest

from tools.autograd_broadcast import cases as broadcast_cases
from tools.autograd_broadcast import is_nan
from tools.autograd_broadcast import reference as broadcast_reference
from tools.autograd_gradient import cases as gradient_cases
from tools.autograd_gradient import reference as gradient_reference
from tools.autograd_reduce import cases as reduce_cases
from tools.autograd_reduce import reference as reduce_reference
from tools.common import ROOT

FIXTURE = ROOT / "oracle/autograd/fixtures/s3-torch-v1.json"
FIXTURE_SHA256 = "98b66eddc970576327945fb9ab3b90ee930952f27eb6939f35b835548e1f837d"
LITERAL_LIMIT = 4096
# Requests the bounded API rejects although PyTorch accepts them: rank 5 and
# more than 2^20 elements. docs/torch-differences.md lists both limits.
BOUNDED = {
    "broadcast": {"target_rank_five", "target_capacity"},
    "gradient": {"slot_rank_five", "slot_capacity"},
}
# PyTorch reduces this case's signaling NaN to a NaN with another payload.
# A slot with no accumulated contribution is None in PyTorch; the gate's
# gradient read materializes zeros (Gradient.grad gives None).
NAN_PAYLOAD = {"special_words_rows"}


def hexwords(words: list[int]) -> list[str]:
    return [f"{word:08x}" for word in words]


def matches(expected: dict, observed: dict) -> str:
    if "rejected" in observed:
        return "rejected"
    if observed["shape"] != expected["shape"]:
        return "shape"
    wanted = hexwords(expected["words"])
    if "sha256" in observed:
        found = hashlib.sha256(" ".join(wanted).encode()).hexdigest()
        return "exact" if found == observed["sha256"] else "words"
    seen = observed["words"].split() if observed["words"] else []
    if seen == wanted:
        return "exact"
    if len(seen) == len(wanted) and all(
        a == b or (is_nan(int(a, 16)) and is_nan(int(b, 16)))
        for a, b in zip(seen, wanted, strict=True)
    ):
        return "nan"
    return "words"


class AutogradTorchOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = FIXTURE.read_bytes()
        cls.digest = hashlib.sha256(data).hexdigest()
        cls.fixture = json.loads(data)

    def test_fixture_is_pinned_and_covers_current_requests(self):
        self.assertEqual(self.digest, FIXTURE_SHA256)
        self.assertEqual(self.fixture["oracle"]["torch_version"], "2.14.1+cpu")
        for name, rows in (
            ("broadcast", broadcast_cases()),
            ("reduce", reduce_cases()),
            ("gradient", gradient_cases()),
        ):
            requests = json.dumps(rows, sort_keys=True).encode()
            with self.subTest(name=name):
                self.assertEqual(
                    self.fixture["requests"][name], hashlib.sha256(requests).hexdigest()
                )
                self.assertEqual(
                    [row["id"] for row in self.fixture[name]],
                    [row["id"] for row in rows],
                )

    def test_broadcast_matches_pytorch_expand(self):
        exact = 0
        for case, row in zip(broadcast_cases(), self.fixture["broadcast"], strict=True):
            expected = broadcast_reference(case)
            observed = row["expand"]
            with self.subTest(case=case["id"]):
                if "error" in expected:
                    if case["id"] in BOUNDED["broadcast"]:
                        self.assertNotIn("rejected", observed)
                    else:
                        self.assertIn("rejected", observed)
                    continue
                self.assertEqual(matches(expected, observed), "exact")
                exact += 1
        self.assertEqual(exact, 214)

    def test_reduce_to_matches_sum_to_size_and_expand_gradient(self):
        exact = 0
        for case, row in zip(reduce_cases(), self.fixture["reduce"], strict=True):
            expected = reduce_reference(case)
            for name in ("sum_to_size", "expand_gradient"):
                observed = row[name]
                with self.subTest(case=case["id"], call=name):
                    if "error" in expected:
                        self.assertIn("rejected", observed)
                        continue
                    wanted = "nan" if case["id"] in NAN_PAYLOAD else "exact"
                    self.assertEqual(matches(expected, observed), wanted)
                    exact += wanted == "exact"
        self.assertEqual(exact, 2 * 219)

    def test_gradient_accumulation_matches_pytorch_backward(self):
        exact = unset = 0
        for case, row in zip(gradient_cases(), self.fixture["gradient"], strict=True):
            expected = gradient_reference(case)
            with self.subTest(case=case["id"]):
                if "error" in expected:
                    self.assertIn(case["id"], BOUNDED["gradient"])
                    self.assertNotIn("rejected", row)
                    continue
                outcomes = ["accumulated" in item for item in expected["outcomes"]]
                self.assertEqual(
                    ["accumulated" in item for item in row["outcomes"]], outcomes
                )
                if not any(outcomes):
                    self.assertIsNone(row["grad"])
                    self.assertEqual(set(expected["gradient"]["words"]) - {0}, set())
                    unset += 1
                    continue
                self.assertEqual(matches(expected["gradient"], row["grad"]), "exact")
                exact += 1
        self.assertEqual((exact, unset), (184, 17))

    def test_comparison_rejects_a_changed_word(self):
        row = self.fixture["broadcast"][0]
        case = broadcast_cases()[0]
        expected = broadcast_reference(case)
        changed = dict(
            expected, words=[expected["words"][0] ^ 1, *expected["words"][1:]]
        )
        self.assertEqual(matches(changed, row["expand"]), "words")
        negative = dict(expected, words=[0x80000000, *expected["words"][1:]])
        self.assertEqual(matches(negative, row["expand"]), "words")


if __name__ == "__main__":
    unittest.main()
