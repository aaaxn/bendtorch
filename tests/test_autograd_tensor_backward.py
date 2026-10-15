import json
import tempfile
import unittest
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_backward import DANGLING_OPERAND, NEGATIVE_ZERO, UNKNOWN_NODE
from tools.autograd_broadcast import word
from tools.autograd_tensor_backward import (
    MATMUL_INNER,
    MATMUL_RANK,
    RUNTIME_CONTROLS,
    ZERO_GRAD,
    cases,
    forge,
    leaf,
    op,
    parse,
    reads,
    reference,
    run,
    run_gate,
    verify,
)
from tools.common import ROOT, GateError


def case(steps: list[dict]) -> dict:
    return {"id": "case", "steps": steps}


def tensors(lines: list[str], prefix: str) -> list[list[int] | None]:
    found: list[list[int] | None] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == f"autograd:{prefix}:none":
            found.append(None)
        elif line == f"autograd:{prefix}":
            count = int(lines[index + 2].removeprefix("tensor:count:"))
            words = lines[index + 3 : index + 3 + count]
            found.append([int(row.removeprefix("tensor:word:")) for row in words])
            index += 2 + count
        index += 1
    return found


def grads(lines: list[str]) -> list[list[int] | None]:
    return tensors(lines, "grad")


def values(*numbers: float) -> list[int]:
    return [word(number) for number in numbers]


GRID = leaf([2, 3], values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
ROW = leaf([3], values(0.5, -1.0, 2.0))
UNUSED = [
    leaf([], values(3.0)),
    leaf([], values(2.0)),
    leaf([3], values(1.0, 2.0, 3.0)),
    op("mul", 0, 1),
    op("add", 2, 2),
]


class TensorBackwardReferenceTests(unittest.TestCase):
    def test_seed_is_the_selected_output_not_the_newest_node(self):
        found = grads(reference(case([*UNUSED, run(3), *reads(5)])))
        self.assertEqual(found, [values(2.0), values(3.0), None, values(1.0), None])

    def test_a_non_scalar_output_is_seeded_with_ones(self):
        found = grads(reference(case([GRID, ROW, op("mul", 0, 1), run(2), *reads(3)])))
        self.assertEqual(found[2], values(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(found[0], values(0.5, -1.0, 2.0, 0.5, -1.0, 2.0))
        self.assertEqual(found[1], values(5.0, 7.0, 9.0))

    def test_broadcast_contributions_are_reduced_to_the_operand_shape(self):
        column = leaf([2, 1], values(-2.0, 0.25))
        steps = [GRID, column, op("sub", 0, 1), op("sum", 2), run(3), *reads(4)]
        found = grads(reference(case(steps)))
        self.assertEqual(found[0], values(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(found[1], values(-3.0, -3.0))

    def test_fan_out_adds_in_arrival_order(self):
        selected = {row["id"]: row for row in cases()}
        # x's consumers x*a, x*b, x*c send a, b, c; the newest (x*c) arrives
        # first, so each position of x holds (c + b) + a: (-1e8 + 1e8) + 1 = 1 and
        # (1 + 1e8) - 1e8 = 0.
        found = grads(reference(selected["fan_out_arrival_order"]))
        self.assertEqual(found[0], values(1.0, 0.0))

    def test_sub_negates_before_it_reduces(self):
        steps = [
            leaf([2], [NEGATIVE_ZERO, 0]),
            op("sub", 0, 0),
            run(1),
            *reads(2),
        ]
        # g - g at one leaf: 1.0 then -1.0 arrive, and the total is +0.0.
        self.assertEqual(grads(reference(case(steps)))[0], [0, 0])

    def test_matmul_vjps(self):
        left = leaf([2, 3], values(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
        right = leaf([3, 2], values(2.0, 1.0, -1.0, 4.0, 0.5, -3.0))
        found = grads(
            reference(case([left, right, op("matmul", 0, 1), run(2), *reads(3)]))
        )
        # g @ right^T with g all ones: each row sums a row of right.
        self.assertEqual(found[0], values(3.0, 3.0, -2.5, 3.0, 3.0, -2.5))
        # left^T @ g: each row sums a column of left.
        self.assertEqual(found[1], values(4.0, 4.0, -1.75, -1.75, -0.5, -0.5))

    def test_sum_vjp_broadcasts_the_gradient(self):
        steps = [GRID, op("sum", 0), op("mul", 1, 1), run(2), *reads(3)]
        found = grads(reference(case(steps)))
        self.assertEqual(found[0], values(*[42.0] * 6))

    def test_repeated_backward_accumulates(self):
        found = grads(reference(case([*UNUSED, run(3), run(3), *reads(5)])))
        self.assertEqual(found[:2], [values(4.0), values(6.0)])

    def test_zero_grad_then_negative_zero_moves_in(self):
        steps = [
            leaf([2], values(4.0, -1.0)),
            leaf([2], [NEGATIVE_ZERO, 0]),
            op("mul", 0, 1),
            ZERO_GRAD,
            run(2),
            *reads(3),
        ]
        self.assertEqual(grads(reference(case(steps)))[0], [NEGATIVE_ZERO, 0])

    def test_unknown_ids_and_dangling_operands_are_typed(self):
        lines = reference(case([ROW, op("add", 0, 4), run(3), *reads(2)]))
        self.assertEqual(lines.count(f"autograd:error:{UNKNOWN_NODE}"), 2)
        self.assertIn(f"autograd:value:error:{UNKNOWN_NODE}", lines)
        forged = forge({"op": "sum", "a": 0}, [], values(1.0))
        lines = reference(case([forged, run(0)]))
        self.assertEqual(lines[-1], f"autograd:error:{DANGLING_OPERAND}")

    def test_matmul_errors_are_typed(self):
        lines = reference(case([GRID, ROW, op("matmul", 0, 1), op("matmul", 0, 0)]))
        self.assertEqual(
            lines[2:],
            [f"autograd:error:{MATMUL_RANK}", f"autograd:error:{MATMUL_INNER}"],
        )


class TensorBackwardVerificationTests(unittest.TestCase):
    def frame(self, lines: list[str]) -> str:
        return "\n".join(["autograd:begin:case", *lines, "autograd:end"]) + "\n"

    def test_reference_output_is_accepted(self):
        row = case([*UNUSED, run(3), *reads(5)])
        verify([row], parse(self.frame(reference(row)), ["case"]), "native")

    def test_a_newest_node_seed_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        wrong = reference(case([*UNUSED, run(4), *reads(5)]))
        with self.assertRaises(GateError):
            verify([row], parse(self.frame(wrong), ["case"]), "native")

    def test_a_nan_only_needs_to_be_nan(self):
        row = case([leaf([1], [0x7FC01234]), op("sum", 0), run(1), *reads(2)])
        expected = reference(row)
        changed = [line.replace(str(0x7FC01234), str(0x7FC00000)) for line in expected]
        self.assertNotEqual(changed, expected)
        verify([row], parse(self.frame(changed), ["case"]), "native")

    def test_a_changed_word_or_missing_line_is_rejected(self):
        row = case([*UNUSED, run(3), *reads(5)])
        expected = reference(row)
        changed = [line.replace(str(word(3.0)), str(word(-3.0))) for line in expected]
        for wrong in (changed, expected[:-1]):
            with self.assertRaises(GateError):
                verify([row], parse(self.frame(wrong), ["case"]), "native")


class GateRegistrationTests(unittest.TestCase):
    def test_autograd_gate_registers_tensor_backward_and_its_laws(self):
        self.assertIn("tensor_backward", SUBGATES)
        required = json.loads((ROOT / "verification/autograd.json").read_text())[
            "required_exports"
        ]
        self.assertEqual(
            sum(name.startswith("tensor_2d_backward_2d_LAWS.") for name in required),
            16,
        )
        controls = [
            name for name in PROOF_CONTROLS if name.startswith("tensor-backward-")
        ]
        self.assertEqual(len(controls), 9)
        self.assertIn("negate-left-keeps-sign", PROOF_CONTROLS)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_runtime_controls_apply_once(self):
        source = (ROOT / "src/autograd/tensor-backward.bend").read_text()
        self.assertEqual(len(RUNTIME_CONTROLS), 6)
        for original, mutated in RUNTIME_CONTROLS.values():
            self.assertEqual(source.count(original), 1)
            self.assertNotEqual(original, mutated)


class TensorBackwardRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(
            prefix="autograd-tensor-backward-"
        ) as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))
        self.assertEqual(summary["runtime_controls"], list(RUNTIME_CONTROLS))


if __name__ == "__main__":
    unittest.main()
