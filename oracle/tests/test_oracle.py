import hashlib
import json
import math
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import generate
import torch


def values(tensor):
    return [
        struct.unpack("<f", bytes.fromhex(word)[::-1])[0] for word in tensor["f32_bits"]
    ]


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = generate.observed_identity()
        generate.verify_identity(cls.identity)
        generate.configure_cpu()
        cls.fixture_path = generate.ORACLE_DIR / "fixtures" / "toolchain.json"
        cls.document = json.loads(cls.fixture_path.read_text())
        cls.cases = {
            case["id"]: case
            for group in ("finite_cases", "nonfinite_cases")
            for case in cls.document[group]
        }

    def test_actual_identity(self):
        self.assertEqual(self.identity["torch_version"], "2.14.1+cpu")
        self.assertEqual(
            self.identity["torch_git_version"],
            "5c4886908584029761b579af026dcfb627c84070",
        )
        self.assertIsNone(self.identity["cuda_version"])
        self.assertIsNone(self.identity["hip_version"])
        self.assertEqual(self.document["oracle"], self.identity)

    def test_identity_mismatches_are_rejected(self):
        for field in self.identity:
            with self.subTest(field=field):
                identity = dict(self.identity)
                identity[field] = "mismatch"
                with self.assertRaisesRegex(RuntimeError, "Oracle identity mismatch"):
                    generate.verify_identity(identity)
                with self.assertRaisesRegex(RuntimeError, "Oracle identity mismatch"):
                    generate.build_fixtures(identity)

    def test_frozen_fixture_matches_live_oracle(self):
        self.assertEqual(
            self.fixture_path.read_bytes(),
            generate.json_bytes(generate.build_fixtures(self.identity)),
        )

    def test_case_coverage(self):
        self.assertEqual(self.document["dtype"], "float32")
        self.assertEqual(self.document["device"], "cpu")
        self.assertEqual(len(self.document["finite_cases"]), 12)
        self.assertEqual(len(self.document["nonfinite_cases"]), 3)
        self.assertEqual(len(self.cases), 15)
        self.assertEqual(
            set(self.cases),
            {
                "scalar_add",
                "scalar_sum",
                "empty_sum",
                "empty_add_singleton",
                "empty_rows_add_singleton",
                "empty_cols_add_singleton",
                "broadcast_add",
                "matmul_2x3_3x2",
                "matmul_add_bias",
                "sum_2x2",
                "f32_bit_patterns",
                "transpose_view_alias",
                "empty_mean",
                "divide_by_zero",
                "nonfinite_add",
            },
        )
        for case in self.document["finite_cases"]:
            for tensor in case["outputs"].values():
                self.assertNotIn("nonfinite", tensor)
        for case in self.document["nonfinite_cases"]:
            self.assertTrue(case["outputs"]["result"]["nonfinite"])

    def test_every_value_and_nonfinite_classification_is_preserved(self):
        for case in self.cases.values():
            for group in ("inputs", "outputs"):
                for name, tensor in case[group].items():
                    with self.subTest(case=case["id"], group=group, name=name):
                        self.assertEqual(
                            len(tensor["f32_bits"]), math.prod(tensor["shape"])
                        )
                        classes = []
                        for index, (word, value) in enumerate(
                            zip(tensor["f32_bits"], values(tensor))
                        ):
                            self.assertRegex(word, "^[0-9a-f]{8}$")
                            if math.isnan(value):
                                classes.append(
                                    {"index": index, "classification": "nan"}
                                )
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

    def test_scalar_and_empty_results(self):
        for case_id, expected in (
            ("scalar_add", 5.5),
            ("scalar_sum", 3.5),
            ("empty_sum", 0.0),
        ):
            result = self.cases[case_id]["outputs"]["result"]
            self.assertEqual(result["shape"], [])
            self.assertEqual(values(result), [expected])
        for case_id, shape, singleton_shape in (
            ("empty_add_singleton", [0], [1]),
            ("empty_rows_add_singleton", [0, 3], [1, 3]),
            ("empty_cols_add_singleton", [3, 0], [3, 1]),
        ):
            with self.subTest(case=case_id):
                case = self.cases[case_id]
                self.assertEqual(case["inputs"]["a"]["shape"], shape)
                self.assertEqual(case["inputs"]["b"]["shape"], singleton_shape)
                self.assertEqual(
                    case["outputs"]["result"], {"shape": shape, "f32_bits": []}
                )

    def test_broadcast_matmul_bias_and_sum(self):
        self.assertEqual(
            values(self.cases["broadcast_add"]["outputs"]["result"]),
            [11, 21, 31, 12, 22, 32],
        )
        matmul = self.cases["matmul_2x3_3x2"]
        self.assertEqual(matmul["inputs"]["a"]["shape"], [2, 3])
        self.assertEqual(matmul["inputs"]["b"]["shape"], [3, 2])
        self.assertEqual(values(matmul["inputs"]["a"]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(values(matmul["inputs"]["b"]), [7, 8, 9, 10, 11, 12])
        self.assertEqual(matmul["outputs"]["result"]["shape"], [2, 2])
        self.assertEqual(values(matmul["outputs"]["result"]), [58, 64, 139, 154])
        biased = self.cases["matmul_add_bias"]
        self.assertEqual(values(biased["inputs"]["bias"]), [1, -1])
        self.assertEqual(values(biased["outputs"]["result"]), [59, 63, 140, 153])
        total = self.cases["sum_2x2"]
        self.assertEqual(total["inputs"]["a"], biased["outputs"]["result"])
        self.assertEqual(total["outputs"]["result"]["shape"], [])
        self.assertEqual(values(total["outputs"]["result"]), [415])

    def test_transpose_alias_is_a_mutating_view(self):
        case = self.cases["transpose_view_alias"]
        self.assertTrue(case["alias"]["shares_storage"])
        self.assertFalse(case["alias"]["view_is_contiguous"])
        self.assertEqual(case["inputs"]["base"]["stride"], [3, 1])
        self.assertEqual(values(case["inputs"]["base"]), [1, 2, 3, 4, 5, 6])
        before = case["outputs"]["view_before"]
        after = case["outputs"]["view_after"]
        self.assertEqual(before["shape"], [3, 2])
        self.assertEqual(before["stride"], [1, 3])
        self.assertEqual(before["storage_offset"], 0)
        self.assertEqual(values(before), [1, 4, 2, 5, 3, 6])
        self.assertEqual(values(case["outputs"]["base_after"]), [1, 20, 3, 4, 5, 6])
        self.assertEqual(values(after), [1, 4, 20, 5, 3, 6])
        self.assertEqual(after["stride"], [1, 3])

    def test_exact_f32_bits_include_signed_zero_and_subnormals(self):
        case = self.cases["f32_bit_patterns"]
        words = [
            "00000000",
            "80000000",
            "00000001",
            "007fffff",
            "00800000",
            "7f7fffff",
            "3eaaaaab",
        ]
        self.assertEqual(case["inputs"]["a"]["f32_bits"], words)
        self.assertEqual(case["outputs"]["result"]["f32_bits"], words)
        self.assertEqual(math.copysign(1, values(case["outputs"]["result"])[1]), -1)
        nonfinite_words = [0x7FC01234, 0x7F800000, 0xFF800000]
        encoded = generate.snapshot(generate.from_bits(nonfinite_words))
        self.assertEqual(
            encoded["f32_bits"], [f"{word:08x}" for word in nonfinite_words]
        )

    def test_nonfinite_semantics_are_not_numeric_tolerances(self):
        self.assertEqual(
            self.cases["empty_mean"]["outputs"]["result"]["nonfinite"],
            [{"index": 0, "classification": "nan"}],
        )
        for case_id in ("divide_by_zero", "nonfinite_add"):
            self.assertEqual(
                self.cases[case_id]["outputs"]["result"]["nonfinite"],
                [
                    {"index": 0, "classification": "nan"},
                    {"index": 1, "classification": "positive_infinity"},
                    {"index": 2, "classification": "negative_infinity"},
                ],
            )

    def test_non_f32_snapshots_are_rejected(self):
        for dtype in (torch.float64, torch.int32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "CPU float32"):
                    generate.snapshot(torch.tensor([1], dtype=dtype, device="cpu"))

    def test_cpu_configuration_and_lock_are_captured(self):
        document = generate.cpu_provenance(self.identity)
        self.assertEqual(document["oracle"], self.identity)
        self.assertEqual(document["python_version"], "3.12.12")
        self.assertEqual(document["platform"]["machine"], "x86_64")
        self.assertEqual(document["platform"]["system"], "Linux")
        self.assertTrue(document["platform"]["cpu_model"])
        self.assertEqual(document["cpu"]["num_threads"], 1)
        self.assertEqual(document["cpu"]["num_interop_threads"], 1)
        self.assertTrue(document["cpu"]["deterministic_algorithms"])
        self.assertEqual(document["torch_config"], torch.__config__.show())
        self.assertIn("OpenMP", document["torch_parallel_info"])
        self.assertEqual(
            document["lock_sha256"],
            hashlib.sha256((generate.ORACLE_DIR / "uv.lock").read_bytes()).hexdigest(),
        )

    def test_two_frozen_generations_are_byte_identical_and_check_is_read_only(self):
        with tempfile.TemporaryDirectory(
            prefix=".test-regeneration-", dir=generate.ORACLE_DIR
        ) as temporary:
            root = Path(temporary)
            for directory in (root / "first", root / "second"):
                completed = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--frozen",
                        "python",
                        "generate.py",
                        "--output-dir",
                        str(directory),
                    ],
                    cwd=generate.ORACLE_DIR,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                self.assertEqual(
                    (directory / "toolchain.json").read_bytes(),
                    self.fixture_path.read_bytes(),
                )
                self.assertTrue((directory / "cpu_config.json").is_file())
            fixture = root / "first" / "toolchain.json"
            fixture.write_bytes(b"stale\n")
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "python",
                    "generate.py",
                    "--check",
                    "--output-dir",
                    str(fixture.parent),
                ],
                cwd=generate.ORACLE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Frozen oracle differs", completed.stderr)
            self.assertEqual(fixture.read_bytes(), b"stale\n")


if __name__ == "__main__":
    unittest.main()
