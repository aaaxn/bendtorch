import hashlib
import json
import re
import unittest

from tools.common import ROOT, GateError
from tools.tensor import SIGNED_CONTROLS
from tools.tensor_signed import (
    AMBIGUOUS_INFERRED,
    AXIS_OUT_OF_BOUNDS,
    COORDINATE_OUT_OF_BOUNDS,
    DUPLICATE_AXIS,
    ELEMENT_COUNT,
    REPEATED_INFERRED,
    cases,
    extent_source,
    inferred,
    position_source,
    reference,
    wrap,
)

FIXTURE = ROOT / "oracle/tensor/fixtures/signed-torch.json"
FIXTURE_SHA256 = "a36983335e31e6f4ed09bc84780633edbc5e1424642c8b75da18885dec3b873e"
# The exception PyTorch raises for each refusal code: an axis, coordinate or
# flat index out of range is IndexError; an axis count other than the rank
# (12), a repeated axis (14), a size count mismatch (15), a second -1 (17)
# and an ambiguous -1 (18) are RuntimeError.
KINDS = {
    4: "IndexError",
    8: "IndexError",
    12: "RuntimeError",
    13: "IndexError",
    14: "RuntimeError",
    15: "RuntimeError",
    17: "RuntimeError",
    18: "RuntimeError",
}
# Both refuse, with another first error: PyTorch's permute wraps and checks
# each axis in turn, so [0, -3, 5] on rank 3 meets -3, which wraps to the
# repeated 0, before 5; the unsigned permute checks every range first.
FIRST_ERROR = {"permute_range_after_repeat": (AXIS_OUT_OF_BOUNDS, "RuntimeError")}
# bendtorch accepts, PyTorch refuses: an empty shape whose other sizes
# multiply past PyTorch's int64 element count and strides. bendtorch's sizes
# are exact naturals and it keeps no strides, so the shape [0, 2^32-1, 2^32-1]
# holds 0 elements.
PYTORCH_REFUSES = {
    "reshape_empty_huge": "RuntimeError",
    "reshape_empty_huge_exact": "RuntimeError",
}


class SignedReferenceTests(unittest.TestCase):
    def test_wrap_counts_negative_positions_from_the_end(self):
        self.assertEqual([wrap(d, 3) for d in (-1, -2, -3)], [2, 1, 0])
        self.assertEqual([wrap(d, 3) for d in (0, 2, 3, 7)], [0, 2, 3, 7])
        self.assertEqual([wrap(d, 3) for d in (-4, -9)], [3, 3])
        self.assertEqual(wrap(-1, 0), 0)

    def test_inferred_size_follows_pytorch(self):
        self.assertEqual(inferred(6, [-1, 2]), [3, 2])
        self.assertEqual(inferred(6, [2, 3]), [2, 3])
        self.assertEqual(inferred(6, [-1, 4]), ELEMENT_COUNT)
        self.assertEqual(inferred(6, [-1, 0]), ELEMENT_COUNT)
        self.assertEqual(inferred(0, [-1, 0]), AMBIGUOUS_INFERRED)
        self.assertEqual(inferred(0, [-1, 3]), [0, 3])
        self.assertEqual(inferred(6, [-1, -1, 7]), REPEATED_INFERRED)
        self.assertEqual(inferred(0, [-1, 1 << 24, 1 << 24]), [0, 1 << 24, 1 << 24])
        self.assertEqual(inferred(6, [-1, 1 << 24, 1 << 24]), ELEMENT_COUNT)
        self.assertEqual(inferred(0, [-1, 1 << 24, 0, 1 << 24]), AMBIGUOUS_INFERRED)

    def test_refusals_keep_the_unsigned_codes(self):
        row = {"shape": [2, 3], "words": list(range(6))}
        self.assertEqual(
            reference({**row, "op": "permute", "dims": [1, -1]}),
            [f"tensor:error:{DUPLICATE_AXIS}"],
        )
        self.assertEqual(
            reference({**row, "op": "get", "index": [-3, 0]}),
            [f"tensor:error:{COORDINATE_OUT_OF_BOUNDS}"],
        )

    def test_sources_name_each_integer_once(self):
        self.assertEqual(position_source(0), "Signed.First{0}")
        self.assertEqual(position_source(-1), "Signed.Last{0}")
        self.assertEqual(position_source(-3), "Signed.Last{2}")
        self.assertEqual(extent_source(-1), "Signed.Inferred{}")
        self.assertEqual(extent_source(0), "Signed.Exact{0}")
        with self.assertRaises(GateError):
            extent_source(-2)


class SignedTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = FIXTURE.read_bytes()
        cls.digest = hashlib.sha256(data).hexdigest()
        cls.fixture = json.loads(data)

    def test_fixture_is_the_pinned_record_of_these_requests(self):
        self.assertEqual(self.digest, FIXTURE_SHA256)
        requests = hashlib.sha256(
            json.dumps(cases(), sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(self.fixture["requests"], requests)
        self.assertEqual(self.fixture["oracle"]["torch_version"], "2.14.1+cpu")

    def test_reference_agrees_with_pytorch(self):
        torch = {row["id"]: row["lines"] for row in self.fixture["signed"]}
        self.assertEqual(list(torch), [case["id"] for case in cases()])
        succeeded = refused = 0
        for case in cases():
            with self.subTest(case=case["id"]):
                expected, seen = reference(case), torch[case["id"]]
                if case["id"] in PYTORCH_REFUSES:
                    self.assertEqual(
                        expected[0], "tensor:shape:[0, 4294967295, 4294967295]"
                    )
                    self.assertEqual(
                        seen, [f"tensor:rejected:{PYTORCH_REFUSES[case['id']]}"]
                    )
                    continue
                if not expected[0].startswith("tensor:error:"):
                    succeeded += 1
                    self.assertEqual(expected, seen)
                    continue
                refused += 1
                code = int(expected[0].removeprefix("tensor:error:"))
                kind = KINDS[code]
                if case["id"] in FIRST_ERROR:
                    self.assertEqual(FIRST_ERROR[case["id"]][0], code)
                    kind = FIRST_ERROR[case["id"]][1]
                self.assertEqual(seen, [f"tensor:rejected:{kind}"])
        self.assertEqual((succeeded, refused), (46, 39))
        self.assertLessEqual(set(PYTORCH_REFUSES), set(torch))


class SignedRegistrationTests(unittest.TestCase):
    def test_tensor_gate_requires_every_signed_law(self):
        required = json.loads((ROOT / "verification/tensor.json").read_text())[
            "required_exports"
        ]
        laws = re.findall(
            r"^law (\w+):", (ROOT / "src/tensor/signed-LAWS.bend").read_text(), re.M
        )
        self.assertEqual(len(laws), 36)
        self.assertEqual(
            [name for name in required if name.startswith("signed_2d_LAWS.")],
            [f"signed_2d_LAWS.{name}" for name in laws],
        )

    def test_controls_apply_once_and_name_a_proof(self):
        source = (ROOT / "src/tensor/signed.bend").read_text()
        proof = (ROOT / "src/tensor/signed-PROOF.bend").read_text()
        self.assertEqual(len(SIGNED_CONTROLS), 18)
        for original, mutated, location in SIGNED_CONTROLS.values():
            with self.subTest(location=location):
                self.assertEqual(source.count(original), 1)
                self.assertNotEqual(original, mutated)
                self.assertIn(f"def {location}(", proof)


if __name__ == "__main__":
    unittest.main()
