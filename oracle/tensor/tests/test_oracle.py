import copy
import hashlib
import json
import math
import struct
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import generate
import torch


class S2OracleTests(unittest.TestCase):
    def test_transport_lists_precede_shape_admission(self) -> None:
        requests = (
            ("get", "coordinate", [1]),
            ("permute_copy", "axes", [0, 1, 2, 3, 4]),
            ("transpose_copy", "dims", [0, 0]),
            ("reshape", "target_shape", [1]),
            ("add", "b_shape", [1]),
            ("sub", "b_shape", [1]),
            ("mul", "b_shape", [1]),
            ("matmul", "b_shape", [1, 1]),
        )
        for shape, category in (
            ([1, 1, 1, 1, 1], "RankExceeded"),
            ([1048577], "CapacityExceeded"),
        ):
            for operation, field, valid in requests:
                for invalid in (-1, 1 << 32, True, False, 0.5, "0", None):
                    with self.subTest(
                        shape=shape, operation=operation, invalid=invalid
                    ):
                        request = {"shape": shape, field: [*valid[:-1], invalid]}
                        self.assertEqual(
                            generate.specified_error(
                                {"op": operation, "request": request}
                            ),
                            "U32Range",
                        )
                self.assertEqual(
                    generate.specified_error(
                        {"op": operation, "request": {"shape": shape, field: valid}}
                    ),
                    category,
                )

    def test_left_shape_transport_precedes_static_dtype_rejection(self) -> None:
        self.assertEqual(
            generate.specified_error(
                {"op": "construct", "request": {"shape": [-1], "dtype": "float64"}}
            ),
            "U32Range",
        )
        self.assertEqual(
            generate.specified_error(
                {
                    "op": "get",
                    "request": {
                        "shape": [1048577],
                        "dtype": "float64",
                        "coordinate": [-1],
                    },
                }
            ),
            "UnsupportedDtype",
        )

    def test_flat_index_transport_boundary_precedes_shape_admission(self) -> None:
        for shape, category in (
            ([1, 1, 1, 1, 1], "RankExceeded"),
            ([1048577], "CapacityExceeded"),
            ([1025, 1024], "CapacityExceeded"),
        ):
            for index in (-1, 1 << 32, False, None):
                with self.subTest(shape=shape, index=index):
                    self.assertEqual(
                        generate.specified_error(
                            {
                                "op": "get_flat",
                                "request": {"shape": shape, "index": index},
                            }
                        ),
                        "U32Range",
                    )
            self.assertEqual(
                generate.specified_error(
                    {"op": "get_flat", "request": {"shape": shape, "index": 0}}
                ),
                category,
            )

    def test_flat_index_transport_boundary_precedes_logical_bounds(self) -> None:
        for index in (-1, 1 << 32, True, False, 0.5, "0", None):
            with self.subTest(index=index):
                self.assertEqual(
                    generate.specified_error(
                        {"op": "get_flat", "request": {"shape": [2], "index": index}}
                    ),
                    "U32Range",
                )
        self.assertIsNone(
            generate.specified_error(
                {"op": "get_flat", "request": {"shape": [2], "index": 1}}
            )
        )
        self.assertEqual(
            generate.specified_error(
                {"op": "get_flat", "request": {"shape": [2], "index": 2}}
            ),
            "FlatIndexOutOfBounds",
        )

    def test_permutation_length_precedes_axis_bounds_and_uniqueness(self) -> None:
        for axes in ([2], [0, 0, 2]):
            with self.subTest(axes=axes):
                self.assertEqual(
                    generate.specified_error(
                        {
                            "op": "permute_copy",
                            "request": {"shape": [2, 3], "axes": axes},
                        }
                    ),
                    "InvalidPermutation",
                )
        self.assertEqual(
            generate.specified_error(
                {
                    "op": "permute_copy",
                    "request": {"shape": [2, 3], "axes": [0, 2]},
                }
            ),
            "AxisOutOfBounds",
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = generate.observed_identity()
        generate.verify_identity(cls.identity)
        generate.configure_cpu()
        cls.fixture_dir = generate.TENSOR_DIR / "fixtures"
        cls.document = json.loads((cls.fixture_dir / "s2-v2.json").read_text())
        cls.successes = [
            case
            for group in ("finite_cases", "edge_cases", "nonfinite_cases")
            for case in cls.document[group]
        ]
        cls.cases = {case["id"]: case for case in cls.successes}

    def test_exact_runtime_identity_and_frozen_policy(self) -> None:
        self.assertEqual(self.identity, generate.EXPECTED_IDENTITY)
        self.assertEqual(self.document["oracle"], self.identity)
        self.assertEqual(self.document["policy_version"], 2)
        self.assertEqual(
            generate.policy()["status"], "frozen_before_any_Bend_comparison"
        )
        self.assertEqual(
            hashlib.sha256(generate.POLICY_PATH.read_bytes()).hexdigest(),
            self.document["policy_sha256"],
        )
        for field in self.identity:
            changed = dict(self.identity)
            changed[field] = "incorrect"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(RuntimeError, "identity mismatch"),
            ):
                generate.build_fixtures(changed)
        with patch.object(generate, "POLICY_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                RuntimeError, "Frozen numerical policy changed"
            ):
                generate.policy()
        numeric = generate.policy()["policies"]["reduction_envelope"]
        self.assertEqual(Fraction(numeric["u"]), generate.U)
        self.assertEqual(Fraction(numeric["eta"]), generate.ETA)

    def test_schema_and_full_case_coverage(self) -> None:
        self.assertEqual(self.document["schema_version"], 1)
        self.assertEqual(self.document["corpus_version"], "s2-v2")
        self.assertEqual(self.document["dtype"], "float32")
        self.assertEqual(self.document["device"], "cpu")
        self.assertIsNone(self.document["implementation_results"])
        self.assertEqual(self.document["random"]["seed"], 0x53320001)
        all_ids = [
            case["id"]
            for case in self.successes
            + self.document["error_cases"]
            + self.document["shape_cases"]
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(self.successes), 161)
        self.assertEqual(len(self.document["finite_cases"]), 141)
        self.assertEqual(len(self.document["edge_cases"]), 3)
        self.assertEqual(len(self.document["nonfinite_cases"]), 17)
        self.assertEqual(len(self.document["error_cases"]), 45)
        self.assertEqual(len(self.document["shape_cases"]), 7)
        self.assertEqual(
            {case["op"] for case in self.successes},
            {
                "construct",
                "clone",
                "get",
                "add",
                "sub",
                "mul",
                "sum",
                "reshape",
                "transpose_copy",
                "permute_copy",
                "matmul",
                "matmul_add",
            },
        )
        self.assertNotIn("mean", {case["op"] for case in self.successes})
        self.assertEqual(
            {
                len(tensor["shape"])
                for case in self.successes
                for tensor in case["inputs"].values()
            },
            {0, 1, 2, 3, 4},
        )

    def test_all_raw_words_and_classifications_are_present(self) -> None:
        covered = set()
        for case in self.successes:
            for tensor in (
                list(case["inputs"].values())
                + list(case["outputs"].values())
                + [case["independent_reference"]["result"]]
            ):
                with self.subTest(case=case["id"]):
                    self.assertEqual(
                        len(tensor["f32_bits"]), math.prod(tensor["shape"])
                    )
                    self.assertEqual(
                        len(tensor["classifications"]), len(tensor["f32_bits"])
                    )
                    classes = []
                    for index, word in enumerate(tensor["f32_bits"]):
                        self.assertRegex(word, "^[0-9a-f]{8}$")
                        integer = int(word, 16)
                        kind = generate.classification(integer)
                        covered.add(kind)
                        self.assertEqual(tensor["classifications"][index], kind)
                        value = struct.unpack(">f", bytes.fromhex(word))[0]
                        if math.isnan(value):
                            classes.append({"index": index, "classification": "nan"})
                        elif math.isinf(value):
                            classes.append(
                                {
                                    "index": index,
                                    "classification": "positive_infinity"
                                    if value > 0
                                    else "negative_infinity",
                                }
                            )
                    self.assertEqual(tensor.get("nonfinite", []), classes)
        self.assertEqual(
            covered,
            {
                "nan",
                "positive_infinity",
                "negative_infinity",
                "positive_zero",
                "negative_zero",
                "positive_subnormal",
                "negative_subnormal",
                "positive_normal",
                "negative_normal",
            },
        )

    def test_independent_exact_binary32_rounding(self) -> None:
        r = generate.rational
        round_word = generate.rounded_word
        self.assertEqual(r(1), Fraction(1, 1 << 149))
        self.assertEqual(r(0x80000001), -Fraction(1, 1 << 149))
        self.assertEqual(r(0x3F800000), 1)
        self.assertEqual(round_word(Fraction(1, 1 << 150)), 0)
        self.assertEqual(round_word(Fraction(3, 1 << 150)), 2)
        self.assertEqual(round_word(-Fraction(1, 1 << 150)), 0x80000000)
        self.assertEqual(round_word(Fraction(1) + Fraction(1, 1 << 24)), 0x3F800000)
        self.assertEqual(round_word(Fraction(1) + Fraction(3, 1 << 24)), 0x3F800002)
        self.assertEqual(round_word(r(0x7F7FFFFF) + Fraction(2) ** 103), 0x7F800000)
        for word in [
            0,
            1,
            2,
            0x007FFFFF,
            0x00800000,
            0x3EAAAAAB,
            0x7F7FFFFF,
            0x80000001,
            0xFF7FFFFF,
        ]:
            with self.subTest(word=word):
                self.assertEqual(round_word(r(word)), word)
        with self.assertRaises(ValueError):
            r(0x7F800000)

    def test_ieee_edges_independently_define_signed_zero_and_overflow(self) -> None:
        step = generate.ieee_binary
        for op, a, b, result in [
            ("add", 0x80000000, 0x80000000, 0x80000000),
            ("add", 0, 0x80000000, 0),
            ("sub", 0x80000000, 0, 0x80000000),
            ("mul", 1, 0x3F000000, 0),
            ("mul", 3, 0x3F000000, 2),
            ("mul", 0, 0xBF800000, 0x80000000),
            ("add", 0x7F7FFFFF, 0x7F7FFFFF, 0x7F800000),
            ("mul", 0x7F800000, 0, 0x7FC00000),
            ("add", 0x7F800000, 0xFF800000, 0x7FC00000),
        ]:
            with self.subTest(op=op, a=a, b=b):
                self.assertEqual(step(op, a, b), result)

    def test_every_torch_output_satisfies_frozen_independent_policy(self) -> None:
        for case in self.successes:
            with self.subTest(case=case["id"]):
                generate.validate_observation(case)
                reference = generate.independent_reference(
                    case["op"],
                    case["inputs"],
                    case.get("parameters", {}),
                    case["comparison_policy"],
                )
                self.assertEqual(reference, case["independent_reference"])

    def test_no_tolerance_filtering_or_missing_output_is_accepted(self) -> None:
        for case_id, word in [
            ("construct_shape_2", "80000000"),
            ("ieee_sum_4", "ff800000"),
            ("ieee_sum_0", "00000000"),
            ("random_add_0", "7fc00000"),
            ("matmul_cancellation", "7f800000"),
            ("sum_shape_1", "80000000"),
        ]:
            changed = copy.deepcopy(self.cases[case_id])
            changed["outputs"]["result"]["f32_bits"][0] = word
            with (
                self.subTest(case=case_id),
                self.assertRaisesRegex(ValueError, "violates"),
            ):
                generate.validate_observation(changed)
        changed = copy.deepcopy(self.cases["random_add_0"])
        changed["outputs"]["result"]["f32_bits"].pop()
        with self.assertRaisesRegex(ValueError, "word count"):
            generate.validate_observation(changed)
        changed = copy.deepcopy(self.cases["ieee_sum_0"])
        changed["outputs"]["result"]["f32_bits"][0] = "ffcabcde"
        generate.validate_observation(changed)

    def test_broadcast_including_zero_one_in_both_directions(self) -> None:
        for op in ("add", "sub", "mul"):
            for index, expected in [
                (4, [0]),
                (5, [0, 3]),
                (6, [3, 0]),
                (7, [1, 0, 3, 1]),
                (8, [0, 3]),
                (9, [0]),
                (10, [0, 3]),
                (11, [3, 0]),
            ]:
                case = self.cases[f"{op}_broadcast_{index}"]
                self.assertEqual(case["outputs"]["result"]["shape"], expected)
                self.assertEqual(case["outputs"]["result"]["f32_bits"], [])
        self.assertEqual(generate.broadcast_shape([2, 1], [1, 3]), [2, 3])
        with self.assertRaisesRegex(ValueError, "IncompatibleBroadcast"):
            generate.broadcast_shape([0], [2])

    def test_row_major_reshape_transpose_and_permutation_are_materialized(self) -> None:
        for case in self.successes:
            if case["op"] == "reshape":
                self.assertEqual(
                    case["inputs"]["a"]["f32_bits"],
                    case["outputs"]["result"]["f32_bits"],
                )
            self.assertNotIn("alias", case)
        transposed = self.cases["transpose_0"]["outputs"]["result"]
        self.assertEqual(transposed["shape"], [3, 2])
        self.assertEqual(
            [generate.rational(word) for word in generate.words(transposed)],
            [1, 4, 2, 5, 3, 6],
        )
        permuted = self.cases["permute_3"]["outputs"]["result"]
        self.assertEqual(permuted["shape"], [3, 2, 1])
        self.assertEqual(
            [generate.rational(word) for word in generate.words(permuted)],
            [1, 4, 2, 5, 3, 6],
        )

    def test_empty_reduction_and_matmul_k0_have_positive_zero_identity(self) -> None:
        for case_id in ("sum_shape_1", "sum_shape_5", "sum_shape_6", "sum_shape_9"):
            result = self.cases[case_id]["outputs"]["result"]
            self.assertEqual(result["shape"], [])
            self.assertEqual(result["f32_bits"], ["00000000"])
        result = self.cases["matmul_rectangular_6"]["outputs"]["result"]
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["f32_bits"], ["00000000"] * 6)
        for case_id in (
            "matmul_rectangular_4",
            "matmul_rectangular_5",
            "matmul_rectangular_7",
        ):
            self.assertEqual(self.cases[case_id]["outputs"]["result"]["f32_bits"], [])

    def test_reduction_envelope_depends_on_length_and_cancellation(self) -> None:
        n3 = generate.envelope([Fraction(1)] * 3, 3)
        n31 = generate.envelope([Fraction(1)] * 31, 31)
        self.assertGreater(
            generate.read_fraction(n31["absolute_error_bound"]),
            generate.read_fraction(n3["absolute_error_bound"]),
        )
        cancellation = self.cases["sum_cancellation_0"]["independent_reference"][
            "envelopes"
        ][0]
        self.assertEqual(generate.read_fraction(cancellation["exact"]), 1)
        self.assertEqual(generate.read_fraction(cancellation["abs_sum"]), 33554433)
        self.assertGreater(
            generate.read_fraction(cancellation["absolute_error_bound"]), 1
        )
        zero = self.cases["sum_cancellation_3"]["independent_reference"]["envelopes"][0]
        self.assertEqual(zero["condition_number"], "infinite")
        self.assertEqual(generate.read_fraction(zero["exact"]), 0)
        for case in self.successes:
            for item in case["independent_reference"].get("envelopes", []):
                n = item["rounding_count"]
                absolute_sum = generate.read_fraction(item["abs_sum"])
                self.assertEqual(
                    generate.read_fraction(item["absolute_error_bound"]),
                    (n * generate.U * absolute_sum + n * generate.ETA)
                    / (1 - n * generate.U),
                )
        with self.assertRaises(ValueError):
            generate.envelope([Fraction(2) ** 120], 1)
        with self.assertRaises(ValueError):
            generate.envelope([], 1 << 24)

    def test_mathematical_goldens_are_separate_and_independent(self) -> None:
        golden = json.loads(
            (self.fixture_dir / "mathematical-goldens-v1.json").read_text()
        )
        ab = [
            [
                sum(a * b for a, b in zip(row, column, strict=True))
                for column in zip(*golden["b"], strict=True)
            ]
            for row in golden["a"]
        ]
        biased = [
            [value + golden["bias"][column] for column, value in enumerate(row)]
            for row in ab
        ]
        self.assertEqual(ab, [[58, 64], [139, 154]])
        self.assertEqual(biased, [[59, 63], [140, 153]])
        self.assertEqual(sum(map(sum, biased)), 415)
        self.assertEqual(golden["ab"], ab)
        self.assertEqual(golden["ab_plus_bias"], biased)
        self.assertEqual(golden["sum_ab_plus_bias"], 415)
        for case_id, expected in [
            ("golden_ab", [58, 64, 139, 154]),
            ("golden_ab_plus_bias", [59, 63, 140, 153]),
            ("golden_sum_415", [415]),
        ]:
            case = self.cases[case_id]
            self.assertEqual(
                [
                    generate.rational(word)
                    for word in generate.words(case["outputs"]["result"])
                ],
                expected,
            )
            self.assertEqual(
                [
                    generate.read_fraction(value)
                    for value in case["independent_reference"]["exact_values"]
                ],
                expected,
            )

    def test_error_contract_and_shape_capacity_boundaries(self) -> None:
        categories = set()
        for case in self.document["error_cases"]:
            categories.add(case["expected_error"])
            with self.subTest(case=case["id"]):
                self.assertEqual(generate.specified_error(case), case["expected_error"])
        self.assertEqual(
            categories,
            set(generate.policy()["policies"]["expected_errors"]["categories"]),
        )
        for case in self.document["shape_cases"]:
            shape = case["request"]["shape"]
            self.assertIsNone(generate.shape_error(shape))
            self.assertEqual(
                case["expected"],
                {"shape": shape, "rank": len(shape), "numel": math.prod(shape)},
            )
        self.assertIsNone(generate.shape_error([1048576]))
        self.assertEqual(generate.shape_error([1048577]), "CapacityExceeded")
        self.assertIsNone(generate.shape_error([4294967295, 4294967295, 4294967295, 0]))

    def test_snapshot_rejects_non_f32_and_cpu_config_is_actual(self) -> None:
        for dtype in (
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.complex64,
        ):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(ValueError, "CPU float32"),
            ):
                generate.snapshot(torch.tensor([1], dtype=dtype))
        config = json.loads((self.fixture_dir / "cpu-config-v2.json").read_text())
        self.assertEqual(config, generate.cpu_provenance(self.identity))
        self.assertEqual(config["cpu"]["num_threads"], 1)
        self.assertEqual(config["cpu"]["num_interop_threads"], 1)
        self.assertTrue(config["cpu"]["deterministic_algorithms"])
        self.assertTrue(config["cpu"]["gradual_underflow_observed"])
        self.assertEqual(config["python_version"], "3.12.12")

    def test_frozen_documents_match_live_generation(self) -> None:
        self.assertEqual(
            (self.fixture_dir / "s2-v2.json").read_bytes(),
            generate.json_bytes(generate.build_fixtures(self.identity)),
        )
        self.assertEqual(
            (self.fixture_dir / "mathematical-goldens-v1.json").read_bytes(),
            generate.json_bytes(generate.exact_goldens()),
        )

    def test_two_actual_regenerations_and_read_only_stale_detection(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".test-regeneration-", dir=generate.TENSOR_DIR
        ) as temporary:
            root = Path(temporary)
            for output in (root / "first", root / "second"):
                completed = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--project",
                        str(generate.ORACLE_DIR),
                        "--frozen",
                        "--no-sync",
                        "python",
                        str(generate.TENSOR_DIR / "generate.py"),
                        "--output-dir",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                for name in (
                    "s2-v2.json",
                    "cpu-config-v2.json",
                    "mathematical-goldens-v1.json",
                ):
                    self.assertEqual(
                        (output / name).read_bytes(),
                        (self.fixture_dir / name).read_bytes(),
                    )
            fixture = root / "first" / "s2-v2.json"
            fixture.write_bytes(b"stale\n")
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(generate.ORACLE_DIR),
                    "--frozen",
                    "--no-sync",
                    "python",
                    str(generate.TENSOR_DIR / "generate.py"),
                    "--check",
                    "--output-dir",
                    str(fixture.parent),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Frozen oracle differs", completed.stderr)
            self.assertEqual(fixture.read_bytes(), b"stale\n")

    def test_scalar_transpose_preserves_words_and_rejects_invalid_axes(self) -> None:
        for word in (0x40E00000, 0x80000000, 0x00000001):
            inputs = {"a": generate.encoded([], [word])}
            with self.subTest(word=word):
                observed = generate.torch_result(
                    "transpose_copy", inputs, {"dims": [0, 0]}
                )
                self.assertEqual(observed, generate.encoded([], [word]))
                tensor = generate.torch_tensor(inputs["a"])
                for dims in ((1, 0), (0, 1), (1, 1), (4294967295, 0), (0, 4294967295)):
                    with self.subTest(dims=dims), self.assertRaises(IndexError):
                        tensor.transpose(*dims)

    def test_historical_scalar_profile_replays_without_changing_active_contract(
        self,
    ) -> None:
        historical = generate.build_fixtures(self.identity, scalar_transpose=False)
        self.assertEqual(
            generate.json_bytes(historical),
            (self.fixture_dir / "s2-v1.json").read_bytes(),
        )
        self.assertEqual(self.document["corpus_version"], "s2-v2")
        self.assertEqual(
            self.cases["transpose_scalar"]["outputs"]["result"]["shape"], []
        )


if __name__ == "__main__":
    unittest.main()
