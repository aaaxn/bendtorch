import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.common import ROOT, GateError
from tools.s3 import PROOF_CONTROLS, SUBGATES
from tools.s3_backward import (
    DANGLING_OPERAND,
    NEGATIVE_ZERO,
    OTHER_NAN,
    QUIET_NAN,
    RUNTIME_CONTROLS,
    UNKNOWN_NODE,
    cases,
    leaf,
    op,
    parse,
    reads,
    reference,
    run,
    run_stage,
    verify,
)
from tools.s3_broadcast import is_nan, word

FIXTURE = ROOT / "oracle/s3/fixtures/s3-backward-torch-v1.json"
FIXTURE_SHA256 = "8c5692d6fef448459e75293f5f7ef562671630c9aa017ff79c2d09bbc0c5622a"
# Requests that PyTorch cannot express or reports differently; the S3 backward
# page lists each one.
UNKNOWN_IDS = {"unknown_operand", "unknown_output_empty", "unknown_output"}
FORGED = {"dangling_self", "dangling_newer", "dangling_mul", "forged_sound"}
# Native sub keeps the first NaN operand's payload; PyTorch keeps the second's.
SUB_PAYLOAD = {"nan_payloads_sub"}


def case(steps: list[dict]) -> dict:
    return {"id": "case", "steps": steps}


def grads(lines: list[str]) -> list[str]:
    return [
        line.removeprefix("S3:grad:") for line in lines if line.startswith("S3:grad:")
    ]


UNUSED = [leaf(3.0), leaf(2.0), leaf(5.0), op("mul", 0, 1), op("add", 2, 2)]


class BackwardReferenceTests(unittest.TestCase):
    def test_seed_is_the_selected_output_not_the_newest_node(self):
        lines = reference(case([*UNUSED, run(3), *reads(5)]))
        self.assertEqual(
            grads(lines),
            [str(word(2.0)), str(word(3.0)), "none", str(word(1.0)), "none"],
        )

    def test_fan_out_adds_in_arrival_order(self):
        selected = {row["id"]: row for row in cases()}
        # x's consumers x*a, x*b, x*c send a, b, c; the newest (x*c) arrives
        # first, so the total is (c + b) + a.
        loses = grads(reference(selected["fan_out_loses_one"]))
        keeps = grads(reference(selected["fan_out_keeps_one"]))
        self.assertEqual(loses[0], str(word(0.0)))
        self.assertEqual(keeps[0], str(word(1.0)))

    def test_repeated_backward_accumulates(self):
        lines = reference(case([*UNUSED, run(3), run(3), *reads(5)]))
        self.assertEqual(grads(lines)[:2], [str(word(4.0)), str(word(6.0))])

    def test_zero_grad_then_negative_zero_moves_in(self):
        lines = reference(
            case(
                [
                    leaf(4.0),
                    leaf(NEGATIVE_ZERO),
                    op("mul", 0, 1),
                    ZERO,
                    run(2),
                    *reads(3),
                ]
            )
        )
        self.assertEqual(grads(lines)[0], str(NEGATIVE_ZERO))

    def test_unknown_ids_and_dangling_operands_are_typed(self):
        lines = reference(case([leaf(1.0), op("add", 0, 4), run(3), *reads(2)]))
        self.assertEqual(lines.count(f"S3:error:{UNKNOWN_NODE}"), 2)
        self.assertIn(f"S3:value:error:{UNKNOWN_NODE}", lines)
        forged = {"op": "forge", "node": {"op": "neg", "a": 0}, "word": word(1.0)}
        lines = reference(case([forged, run(0)]))
        self.assertEqual(lines[-1], f"S3:error:{DANGLING_OPERAND}")


ZERO = {"op": "zero_grad"}


class NanPayloadTests(unittest.TestCase):
    # Observed on native Bend (clang 19 -O3, x86-64) and PyTorch 2.14.1+cpu.
    def test_add_and_mul_keep_the_second_nan(self):
        for name in ("add", "mul"):
            lines = reference(
                case([leaf(QUIET_NAN), leaf(OTHER_NAN), op(name, 0, 1), *reads(3)])
            )
            self.assertEqual(lines[-2], f"S3:value:{OTHER_NAN}")

    def test_sub_keeps_the_first_nan(self):
        lines = reference(
            case([leaf(QUIET_NAN), leaf(OTHER_NAN), op("sub", 0, 1), *reads(3)])
        )
        self.assertEqual(lines[-2], f"S3:value:{QUIET_NAN}")

    def test_a_signaling_nan_is_quieted(self):
        lines = reference(
            case([leaf(1.0), leaf(0x7FA00001), op("mul", 0, 1), *reads(3)])
        )
        self.assertEqual(lines[-2], f"S3:value:{0x7FE00001}")


class BackwardVerificationTests(unittest.TestCase):
    def frame(self, lines: list[str]) -> str:
        return "\n".join(["S3:begin:case", *lines, "S3:end"]) + "\n"

    def test_reference_output_is_accepted(self):
        row = case([*UNUSED, run(3), *reads(5)])
        verify([row], parse(self.frame(reference(row)), ["case"]), "native")

    def test_a_newest_node_seed_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        wrong = reference(case([*UNUSED, run(4), *reads(5)]))
        with self.assertRaises(GateError):
            verify([row], parse(self.frame(wrong), ["case"]), "native")

    def test_native_nan_payload_must_match(self):
        row = case([leaf(0x7FC01234), op("neg", 0), run(1), *reads(2)])
        expected = reference(row)
        changed = [line.replace(str(0x7FC01234), str(0x7FC00000)) for line in expected]
        self.assertNotEqual(changed, expected)
        with self.assertRaises(GateError):
            verify([row], parse(self.frame(changed), ["case"]), "native")
        verify([row], parse(self.frame(changed), ["case"]), "javascript")

    def test_a_missing_line_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        with self.assertRaises(GateError):
            verify([row], parse(self.frame(reference(row)[:-1]), ["case"]), "native")


class BackwardTorchOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = FIXTURE.read_bytes()
        cls.digest = hashlib.sha256(data).hexdigest()
        cls.fixture = json.loads(data)

    def test_fixture_is_pinned_and_covers_current_requests(self):
        self.assertEqual(self.digest, FIXTURE_SHA256)
        self.assertEqual(self.fixture["oracle"]["torch_version"], "2.14.1+cpu")
        requests = json.dumps(cases(), sort_keys=True).encode()
        self.assertEqual(self.fixture["requests"], hashlib.sha256(requests).hexdigest())
        self.assertEqual(
            [row["id"] for row in self.fixture["backward"]],
            [row["id"] for row in cases()],
        )

    def test_reference_matches_pytorch_word_for_word(self):
        exact = 0
        for row, observed in zip(cases(), self.fixture["backward"], strict=True):
            identifier = row["id"]
            with self.subTest(case=identifier):
                if identifier in FORGED:
                    self.assertEqual(observed["unrepresentable"], "forge")
                    continue
                expected = reference(row)
                if identifier in SUB_PAYLOAD:
                    differ = [
                        (seen, wanted)
                        for seen, wanted in zip(
                            observed["lines"], expected, strict=True
                        )
                        if seen != wanted
                    ]
                    self.assertEqual(
                        differ, [(f"S3:value:{OTHER_NAN}", f"S3:value:{QUIET_NAN}")]
                    )
                    continue
                if identifier in UNKNOWN_IDS:
                    self.assertEqual(len(observed["lines"]), len(expected))
                    for seen, wanted in zip(observed["lines"], expected, strict=True):
                        if seen != wanted:
                            self.assertIn("rejected:IndexError", seen)
                            self.assertTrue(wanted.endswith(f"error:{UNKNOWN_NODE}"))
                    continue
                self.assertEqual(observed["lines"], expected)
                exact += 1
        self.assertEqual(
            exact, len(cases()) - len(UNKNOWN_IDS) - len(FORGED) - len(SUB_PAYLOAD)
        )

    def test_nonfinite_cases_are_compared_with_pytorch_not_duals(self):
        rows = {row["id"]: row for row in self.fixture["backward"]}
        # x * y at x = inf, y = 0: the value is NaN, but reverse mode gives
        # dx = 1 * y = 0 and dy = 1 * x = inf. Forward duals would give NaN.
        lines = rows["inf_times_zero"]["lines"]
        self.assertEqual(lines[-5], f"S3:grad:{word(0.0)}")
        self.assertEqual(lines[-3], f"S3:grad:{word(float('inf'))}")
        self.assertTrue(is_nan(int(lines[-2].removeprefix("S3:value:"))))


class StageRegistrationTests(unittest.TestCase):
    def test_s3_gate_registers_backward_and_its_laws(self):
        self.assertIn("backward", SUBGATES)
        required = json.loads((ROOT / "verification/s3.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(sum("backward_2d_LAWS" in name for name in required), 35)
        controls = [name for name in PROOF_CONTROLS if name.startswith("backward-")]
        self.assertEqual(len(controls), 9)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_runtime_controls_cover_the_audit_findings(self):
        source = (ROOT / "src/autograd/backward.bend").read_text()
        self.assertEqual(
            list(RUNTIME_CONTROLS),
            [
                "seed-newest-node",
                "lookup-aliases-newest",
                "fan-out-drops-later-contribution",
                "mul-vjp-swaps-operands",
            ],
        )
        for original, mutated in RUNTIME_CONTROLS.values():
            self.assertEqual(source.count(original), 1)
            self.assertNotEqual(original, mutated)


class BackwardRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(prefix="s3-backward-") as directory:
            summary = run_stage(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))
        self.assertEqual(summary["runtime_controls"], list(RUNTIME_CONTROLS))


if __name__ == "__main__":
    unittest.main()
