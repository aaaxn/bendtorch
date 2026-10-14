import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.common import ROOT, GateError, bend, environment
from tools.shape import (
    CAPACITY,
    MAX_U32,
    TRANSPORT_REJECTIONS,
    broadcast_oracle,
    check_transport,
    encode,
    exhaustive_requests,
    expected_lines,
    flatten_oracle,
    run_gate,
    shape_oracle,
    stream,
    unflatten_oracle,
    verify_output,
)


class ShapeOracleTests(unittest.TestCase):
    def test_scalar_empty_and_capacity_contract(self):
        self.assertEqual(shape_oracle([]), {"dimensions": [], "rank": 0, "count": 1})
        for dimensions in ([0, 3], [MAX_U32, MAX_U32, MAX_U32, 0], [CAPACITY, 2, 0]):
            self.assertEqual(shape_oracle(list(dimensions))["count"], 0)
        self.assertEqual(shape_oracle([CAPACITY])["count"], CAPACITY)
        for dimensions in ([CAPACITY + 1], [65536, 65536], [MAX_U32, MAX_U32]):
            self.assertEqual(shape_oracle(list(dimensions)), {"error": 2})
        self.assertEqual(shape_oracle([1] * 5), {"error": 1})

    def test_right_aligned_zero_broadcast_is_not_max(self):
        self.assertEqual(
            broadcast_oracle([0, 3], [1, 3]),
            {"dimensions": [0, 3], "rank": 2, "count": 0},
        )
        self.assertEqual(
            broadcast_oracle([1, 3], [0, 3]),
            {"dimensions": [0, 3], "rank": 2, "count": 0},
        )
        self.assertEqual(broadcast_oracle([0, 3], [2, 3]), {"error": 6})
        self.assertEqual(broadcast_oracle([1024, 1], [1, 2048]), {"error": 2})

    def test_independent_row_major_examples(self):
        self.assertEqual(flatten_oracle([2, 3], [1, 2]), {"value": 5})
        self.assertEqual(unflatten_oracle([2, 3], 5), {"value": [1, 2]})
        self.assertEqual(flatten_oracle([], []), {"value": 0})
        self.assertEqual(unflatten_oracle([], 0), {"value": []})
        self.assertEqual(flatten_oracle([0, 3], [0, 0]), {"error": 4})
        self.assertEqual(unflatten_oracle([0, 3], 0), {"error": 5})

    def test_omitted_and_wrong_results_cannot_pass(self):
        requests = [
            {
                "action": "shape",
                "dimensions": [2, 3],
                "coordinates": [[1, 2]],
                "indices": [5],
            }
        ]
        lines = expected_lines(requests[0])
        self.assertEqual(lines, ["shape:2,3:2:6", "index:5", "coordinate:1,2"])
        verify_output("\n".join(lines) + "\n", requests, "native")
        for actual in (lines[:2], [*lines, "index:0"], [lines[0], "index:4", lines[2]]):
            with self.assertRaises(GateError):
                verify_output("\n".join(actual) + "\n", requests, "native")

    def test_u32_transport_range_is_rejected_before_encoding(self):
        self.assertEqual(check_transport(), 4)
        for request in TRANSPORT_REJECTIONS:
            with (
                self.subTest(request=request),
                self.assertRaisesRegex(GateError, "U32"),
            ):
                encode(request)


class ShapeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = ROOT / ".evidence"
        scratch.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="unit-", dir=scratch)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "evidence"
        run_gate(ROOT / ".tools", cls.output)
        cls.coverage = json.loads((cls.output / "coverage.json").read_text())
        cls.requests, _ = exhaustive_requests()

    def evaluate(self, requests: list[dict]) -> list[str]:
        path = self.output.parent / "requests.txt"
        path.write_text(stream(requests))
        outputs = [
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment(ROOT / ".tools"),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            for command in (
                [
                    str(self.output / "shapes"),
                    "--threads",
                    "1",
                    "--gpu",
                    "off",
                    str(path),
                ],
                [bend(ROOT / ".tools")[0], str(self.output / "shapes.js"), str(path)],
            )
        ]
        for result in outputs:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        native, javascript = (result.stdout for result in outputs)
        self.assertEqual(native, javascript)
        verify_output(native, requests, "native")
        return native.splitlines()

    def check(self, name: str, body: str) -> dict[str, subprocess.CompletedProcess]:
        path = self.output.parent / f"{name}.bend"
        path.write_text(
            "import Base\nimport ../../src/shape/core.bend as Shape\n\n" + body
        )
        return {
            checker: subprocess.run(
                bend(ROOT / ".tools", path, checker),
                cwd=ROOT,
                env=environment(ROOT / ".tools"),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            for checker in ("--check-only", "--verdict")
        }

    def test_complete_exhaustive_runtime_coverage(self):
        self.assertEqual(
            self.coverage,
            {
                "exhaustive_shapes": 237,
                "exhaustive_broadcast_pairs": 56169,
                "adversarial_shapes": 21,
                "additional_broadcast_pairs": 9,
                "transport_rejections": 4,
                "requests": 56436,
                "flatten_checks": 2854,
                "unflatten_checks": 1722,
                "projection_checks": 333846,
                "flat_projection_checks": 196492,
            },
        )
        self.assertEqual(len(self.requests), self.coverage["requests"])
        for lane in ("native", "javascript"):
            actual = (self.output / f"shape-{lane}-run.stdout").read_text()
            verify_output(actual, self.requests, lane)

    def test_actual_capacity_edge_index_and_trailing_zero_behavior(self):
        requests = [
            {
                "action": "shape",
                "dimensions": [16, 16, 16, 256],
                "coordinates": [[15, 15, 15, 255]],
                "indices": [CAPACITY - 1, CAPACITY],
            },
            {
                "action": "shape",
                "dimensions": [MAX_U32, MAX_U32, MAX_U32, 0],
                "coordinates": [[0, 0, 0, 0]],
                "indices": [0],
            },
        ]
        self.assertEqual(
            self.evaluate(requests),
            [
                f"shape:16,16,16,256:4:{CAPACITY}",
                f"index:{CAPACITY - 1}",
                "coordinate:15,15,15,255",
                "error:5",
                f"shape:{MAX_U32},{MAX_U32},{MAX_U32},0:4:0",
                "error:4",
                "error:5",
            ],
        )

    def test_safety_decisions_cannot_be_counterfeited(self):
        statements = {
            "rank": (
                "Shape.ConstructResult<[1,1,1,1,1]>",
                "Shape.construct.ranked([1,1,1,1,1], Shape.BTrue{{==}})",
            ),
            "scale": (
                "Shape.Capacity<[4194304n, 1n]>",
                "Shape.capacity.scale(4194304n, [1n], 1n, {==}, Shape.BTrue{{==}})",
            ),
            "bound": (
                "Shape.Capacity<[2n]>",
                "Shape.capacity.bound([2n], 2n, {==}, Shape.BFalse{{==}})",
            ),
        }
        for name, (result_type, expression) in statements.items():
            checks = self.check(
                f"forged-{name}", f"def main() -> {result_type}:\n  {expression}\n"
            )
            for checker, result in checks.items():
                with self.subTest(decision=name, checker=checker):
                    diagnostic = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 1, diagnostic)
                    self.assertIn("SOME PROOFS FAIL", diagnostic)
                    self.assertIn("True{}", diagnostic)
                    self.assertIn("False{}", diagnostic)

    def test_broadcast_certificate_is_inhabited_and_projects(self):
        source = (
            "def alignment() -> "
            "Shape.Broadcast<[3n, 1n], [2n, 1n, 4n], [2n, 3n, 4n]>:\n"
            "  Shape.BAlign{1n, 0n, Shape.PRightZero{{==}}, "
            "Shape.AStep{Shape.AxisLeftOne{{==}, {==}}, "
            "Shape.AStep{Shape.AxisRightOne{{==}, {==}}, "
            "Shape.AStep{Shape.AxisLeftOne{{==}, {==}}, Unit{}}}}}\n\n"
            "def coordinate() -> Shape.Coordinate([2n, 3n, 4n]):\n"
            "  Shape.CStep{Shape.FNext{Shape.FZero{}}, "
            "Shape.CStep{Shape.FNext{Shape.FNext{Shape.FZero{}}}, "
            "Shape.CStep{Shape.FNext{Shape.FNext{Shape.FNext{Shape.FZero{}}}}, "
            "Unit{}}}}\n\n"
            "def left() -> {Shape.indices([3n, 1n], Shape.project_left([3n, 1n], "
            "[2n, 1n, 4n], [2n, 3n, 4n], alignment(), coordinate())) == [2n, 0n] "
            ": List<&2, Nat>}:\n  {==}\n\n"
            "def right() -> {Shape.indices([2n, 1n, 4n], Shape.project_right("
            "[3n, 1n], [2n, 1n, 4n], [2n, 3n, 4n], alignment(), coordinate())) "
            "== RIGHT : List<&2, Nat>}:\n  {==}\n"
        )
        accepted = self.check(
            "broadcast-example",
            source.replace("RIGHT", "[1n, 0n, 3n]"),
        )
        rejected = self.check(
            "broadcast-wrong",
            source.replace("RIGHT", "[1n, 1n, 3n]"),
        )
        for checker in accepted:
            with self.subTest(checker=checker):
                result = accepted[checker]
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("ALL PROOFS CHECK", result.stdout)
                diagnostic = rejected[checker].stdout + rejected[checker].stderr
                self.assertIn(
                    "SOME PROOFS FAIL\nError:\n- expected : [1n, 0n, 3n]\n"
                    "- observed : [1n, 1n, 3n]",
                    diagnostic,
                )

    def test_invalid_broadcast_projections_return_checked_errors(self):
        requests = [
            {
                "action": "broadcast",
                "left": [2, 1],
                "right": [1, 3],
                "coordinates": [[1, 2], [1], [2, 0]],
                "indices": [0, 5, 6, MAX_U32],
            },
            {
                "action": "broadcast",
                "left": [0, 3],
                "right": [1, 3],
                "coordinates": [[0, 0]],
                "indices": [0],
            },
            {
                "action": "broadcast",
                "left": [0, 3],
                "right": [2, 3],
                "coordinates": [],
                "indices": [],
            },
        ]
        self.assertEqual(
            self.evaluate(requests),
            [
                "broadcast:2,3:2:6",
                "coordinate:1,0 coordinate:0,2",
                "error:3 error:3",
                "error:4 error:4",
                "index:0 index:0",
                "index:1 index:2",
                "error:5",
                "error:5",
                "broadcast:0,3:2:0",
                "error:4 error:4",
                "error:5",
                "error:6",
            ],
        )

    def test_proof_and_both_runtime_lanes_all_passed(self):
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(
            [command["name"] for command in report["commands"]],
            [
                "shape-source-check",
                "shape-kernel-check",
                "shape-export",
                "shape-direct-kernel",
                "pytorch-absent",
                "shape-native-build",
                "shape-native-run",
                "shape-javascript-build",
                "shape-javascript-run",
            ],
        )
        self.assertTrue(all(command["passed"] for command in report["commands"]))
        self.assertTrue((self.output / "PROOF.bendtt").is_file())
        self.assertIn(
            "ALL PROOFS CHECK", (self.output / "shape-kernel-check.stdout").read_text()
        )


if __name__ == "__main__":
    unittest.main()
