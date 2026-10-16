import hashlib
import json
import math
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tools.autograd import PROOF_CONTROLS, SUBGATES
from tools.autograd_backward import DANGLING_OPERAND, NEGATIVE_ZERO, UNKNOWN_NODE
from tools.autograd_broadcast import is_nan, word
from tools.autograd_tensor_backward import (
    MATMUL_INNER,
    MATMUL_RANK,
    NATIVE_CONTROLS,
    NOT_LEAF,
    RUNTIME_CONTROLS,
    ZERO_GRAD,
    cases,
    descend_word,
    forge,
    leaf,
    op,
    parse,
    reads,
    reference,
    run,
    run_gate,
    same_row,
    verify,
)
from tools.common import ROOT, GateError

FIXTURE = ROOT / "oracle/autograd/fixtures/tensor-backward-torch.json"
FIXTURE_SHA256 = "0e8df4a870e876edf04f7993db59a9b288e14056e4eabdfb4cce60ba2caa1cf1"
# The tape answers UnknownNode where the replay indexes past its Python list
# (IndexError), and the tensor layer's error codes where PyTorch raises
# RuntimeError: shapes that do not broadcast (6), a matmul rank (105) or inner
# dimension (106) mismatch, and a leaf with the wrong number of values (10).
REJECTIONS = {
    "unknown_operands",
    "unknown_output_empty",
    "unknown_output",
    "incompatible_shapes",
    "invalid_leaf",
    "linear_inner_mismatch",
    "linear_bias_mismatch",
    "linear_bias_next",
    "linear_bias_unknown",
    "matmul_nt_rank",
    "relu_unknown_operand",
    "sgd_unknown_param",
}
RUNTIME_ERROR_CODES = {6, MATMUL_RANK, MATMUL_INNER, 10}
FORGED = {"dangling_self", "dangling_newer", "dangling_missing"}
# NaN * NaN: the payload is not specified by the tensor layer.
NAN_PAYLOAD = {"specials", "relu_edges", "relu_nan_gradient", "sgd_edges"}
# The forward value differs, not the backward: rounding_09 sums six words in
# ascending order where PyTorch's vectorized sum takes another order, and
# rounding_14 has a matmul with inner dimension 1 whose only product is -0.0:
# the ordered dot adds it to +0.0 and gives +0.0, PyTorch's gemm keeps -0.0.
# relu_13 has the same inner-dimension-1 matmul, whose products 0.5 * -1e-45
# and 0.5 * -0.0 are -0.0.
FORWARD_ORDER = {"rounding_09", "rounding_14", "relu_13"}
# PyTorch's CPU kernel for param.add_(grad, alpha=-lr) rounds p + (-lr) * g
# once (a fused multiply-add) on AVX2 and AVX512; the tape rounds the product,
# then the difference. These scripts match a reference that rounds once.
FUSED_UPDATE = {"sgd_two_layers", "sgd_rounding"}
# torch.optim.SGD accepts a non-leaf that retains its gradient, as every
# non-leaf of the replay does; the tape refuses it with NotLeaf.
NON_LEAF = {"sgd_not_leaf"}


def f32(number: int) -> float:
    return struct.unpack("<f", struct.pack("<I", number))[0]


def rounded(exact: Fraction) -> int:
    negative = exact < 0
    magnitude = -exact if negative else exact
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    unit = Fraction(2) ** (max(exponent, -126) - 23)
    steps = magnitude / unit
    whole = steps.numerator // steps.denominator
    rest = steps - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    if whole * unit >= Fraction(2) ** 128 - Fraction(2) ** 103:
        bits = 0x7F800000
    else:
        bits = struct.unpack("<I", struct.pack("<f", float(whole * unit)))[0]
    return bits | 0x80000000 if negative else bits


# p - lr * g rounded once, from the exact value, for finite words; a NaN or an
# infinity propagates as with two roundings.
def fused_word(lr: int, p: int, g: int) -> int:
    rate, held, gradient = f32(lr), f32(p), f32(g)
    if not all(math.isfinite(number) for number in (rate, held, gradient)):
        return descend_word(lr, p, g)
    product = Fraction(rate) * Fraction(gradient)
    exact = Fraction(held) - product
    if exact != 0:
        return rounded(exact)
    negative = math.copysign(1.0, held) < 0 and math.copysign(1.0, rate * gradient) > 0
    return NEGATIVE_ZERO if held == 0 and product == 0 and negative else 0


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

    def test_linear_vjps_read_the_weight_in_place(self):
        x = leaf([2, 3], values(1.0, -2.0, 0.5, 3.0, 0.25, -1.0))
        weight = leaf([2, 3], values(2.0, 1.0, -1.0, 4.0, 0.5, -3.0))
        bias = leaf([2], values(0.5, -1.0))
        steps = [x, weight, bias, {"op": "linear", "a": 0, "b": 1, "bias": 2}]
        lines = reference(case([*steps, run(4), *reads(5)]))
        found = grads(lines)
        # g @ W with g all ones: each row of x's gradient sums W's rows.
        self.assertEqual(found[0], values(6.0, 1.5, -4.0, 6.0, 1.5, -4.0))
        # g^T @ x: each row of W's gradient sums x's rows.
        self.assertEqual(found[1], values(4.0, -1.75, -0.5, 4.0, -1.75, -0.5))
        # The bias gradient sums g over the batch, column by column.
        self.assertEqual(found[2], values(2.0, 2.0))
        # x @ W^T + b: row 0 is [2 - 2 - 0.5 + 0.5, 4 - 1 - 1.5 - 1], and
        # row 1 is [6 + 0.25 + 1 + 0.5, 12 + 0.125 + 3 - 1].
        self.assertEqual(tensors(lines, "value")[4], values(0.0, 0.5, 7.75, 14.125))

    def test_a_refused_linear_leaves_the_tape_as_it_was(self):
        x = leaf([1, 1], values(2.0))
        weight = leaf([1, 1], values(3.0))
        bias = {"op": "linear", "a": 0, "b": 1, "bias": 2}
        # bias 2 is the id the product would take: it is unknown.
        lines = reference(case([x, weight, bias, leaf([], values(5.0))]))
        self.assertEqual(
            lines,
            [
                "autograd:id:0",
                "autograd:id:1",
                f"autograd:error:{UNKNOWN_NODE}",
                "autograd:id:2",
            ],
        )
        # A bias that does not broadcast to the [2, 3] product drops it.
        column = leaf([2, 1], values(2.0, 4.0))
        rows = leaf([3, 1], values(3.0, 5.0, 7.0))
        short = leaf([2], values(1.0, 2.0))
        steps = [column, rows, short, bias, leaf([], values(5.0))]
        self.assertEqual(
            reference(case(steps))[3:], ["autograd:error:6", "autograd:id:3"]
        )

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


class TensorBackwardTorchOracleTests(unittest.TestCase):
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
                seen = observed["lines"]
                self.assertEqual(len(seen), len(expected))
                differ = [(a, b) for a, b in zip(seen, expected, strict=True) if a != b]
                if identifier in REJECTIONS:
                    for a, b in differ:
                        code = int(b.rsplit(":", 1)[1])
                        if a.endswith("IndexError"):
                            self.assertEqual(code, UNKNOWN_NODE)
                        else:
                            self.assertEqual(a, "autograd:rejected:RuntimeError")
                            self.assertIn(code, RUNTIME_ERROR_CODES)
                        self.assertEqual(
                            a.split(":")[1],
                            b.split(":")[1].replace("error", "rejected"),
                        )
                    continue
                if identifier in NAN_PAYLOAD:
                    self.assertTrue(differ)
                    for a, b in differ:
                        self.assertTrue(is_nan(int(a.removeprefix("tensor:word:"))))
                        self.assertTrue(is_nan(int(b.removeprefix("tensor:word:"))))
                    continue
                if identifier in FUSED_UPDATE:
                    self.assertTrue(differ)
                    fused = reference(row, update=fused_word)
                    self.assertEqual(len(seen), len(fused))
                    for a, b in zip(seen, fused, strict=True):
                        self.assertTrue(same_row(a, b), (a, b))
                    continue
                if identifier in NON_LEAF:
                    self.assertEqual(
                        differ[0], ("autograd:step", f"autograd:error:{NOT_LEAF}")
                    )
                    for a, b in differ[1:]:
                        self.assertTrue(a.startswith("tensor:word:"))
                        self.assertTrue(b.startswith("tensor:word:"))
                    continue
                if identifier in FORWARD_ORDER:
                    self.assertTrue(differ)
                    for a, b in differ:
                        self.assertTrue(a.startswith("tensor:word:"))
                        self.assertTrue(b.startswith("tensor:word:"))
                    self.assertEqual(grads(seen), grads(expected))
                    continue
                self.assertEqual(seen, expected)
                exact += 1
        excluded = (
            REJECTIONS | FORGED | NAN_PAYLOAD | FORWARD_ORDER | FUSED_UPDATE | NON_LEAF
        )
        self.assertEqual(exact, len(cases()) - len(excluded))

    def test_arrival_order_and_broadcast_folds_match_pytorch(self):
        rows = {row["id"]: row for row in self.fixture["backward"]}
        selected = {row["id"]: row for row in cases()}
        for identifier in ("fan_out_arrival_order", "broadcast_fold_order", "linear"):
            self.assertEqual(rows[identifier]["lines"], reference(selected[identifier]))
        self.assertEqual(
            grads(rows["fan_out_arrival_order"]["lines"])[0], values(1.0, 0.0)
        )


def canonicalized(lines: list[str], index: int) -> list[str]:
    changed = list(lines)
    changed[index] = f"tensor:word:{CANONICAL}"
    return changed


CANONICAL = 0x7FC00000


class ReluExactWordTests(unittest.TestCase):
    """The native lane holds relu's words exact, NaNs included; any NaN for any
    NaN is still accepted elsewhere and on the JavaScript lane."""

    def selected(self, identifier: str) -> dict:
        return next(row for row in cases() if row["id"] == identifier)

    def word_index(self, lines: list[str], header: str, nth: int, word: int) -> int:
        start = [i for i, line in enumerate(lines) if line == header][nth] + 3
        return lines.index(f"tensor:word:{word}", start)

    def test_reference_lines_pass_on_both_lanes(self):
        for identifier in ("relu_edges", "relu_nan_gradient", "relu_two_layers"):
            row = self.selected(identifier)
            observed = [{"id": identifier, "lines": reference(row)}]
            verify([row], observed, "native", native=True)
            verify([row], observed, "javascript")

    def test_a_canonical_nan_from_relu_fails_only_the_native_check(self):
        row = self.selected("relu_edges")
        lines = reference(row)
        # Node 1 is relu(x): its word for the negative signaling NaN.
        index = self.word_index(lines, "autograd:value", 1, 0xFF800001)
        observed = [{"id": row["id"], "lines": canonicalized(lines, index)}]
        verify([row], observed, "javascript")
        with self.assertRaisesRegex(GateError, "relu of node 0 is not exact"):
            verify([row], observed, "native", native=True)

    def test_a_canonical_nan_in_the_passed_gradient_fails_the_native_check(self):
        row = self.selected("relu_nan_gradient")
        lines = reference(row)
        # Node 0's gradient where x = 1.0: g's own word, a negative NaN.
        index = self.word_index(lines, "autograd:grad", 0, 0xFFC00001)
        observed = [{"id": row["id"], "lines": canonicalized(lines, index)}]
        verify([row], observed, "javascript")
        with self.assertRaisesRegex(GateError, "relu gradient of node 0 is not exact"):
            verify([row], observed, "native", native=True)


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
        self.assertEqual(
            [name for name in required if name.startswith("tensor_2d_vjp_2d_LAWS.")],
            [
                "tensor_2d_vjp_2d_LAWS.tensor_product_replayed",
                "tensor_2d_vjp_2d_LAWS.tensor_add_vjp_values",
                "tensor_2d_vjp_2d_LAWS.tensor_sub_vjp_values",
                "tensor_2d_vjp_2d_LAWS.tensor_mul_vjp_values",
                "tensor_2d_vjp_2d_LAWS.tensor_matmul_vjp_values",
                "tensor_2d_vjp_2d_LAWS.tensor_matmul_nt_vjp_values",
                "tensor_2d_vjp_2d_LAWS.nt_reads_transpose",
                "tensor_2d_vjp_2d_LAWS.tn_reads_transpose",
                "tensor_2d_vjp_2d_LAWS.tensor_relu_vjp_values",
            ],
        )
        controls = [
            name for name in PROOF_CONTROLS if name.startswith("tensor-backward-")
        ]
        self.assertEqual(len(controls), 9)
        self.assertIn("negate-left-keeps-sign", PROOF_CONTROLS)
        for path, original, _ in PROOF_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)

    def test_native_controls_apply_once(self):
        self.assertEqual(
            set(NATIVE_CONTROLS),
            {"relu-canonicalizes-nan", "relu-backward-canonicalizes-nan"},
        )
        for path, original, mutated in NATIVE_CONTROLS.values():
            self.assertEqual((ROOT / path).read_text().count(original), 1)
            self.assertIn("quiet(2143289344)", mutated)

    def test_runtime_controls_apply_once(self):
        self.assertEqual(len(RUNTIME_CONTROLS), 17)
        for control in RUNTIME_CONTROLS.values():
            path, original, mutated = (
                control
                if len(control) == 3
                else ("src/autograd/tensor-backward.bend", *control)
            )
            self.assertEqual((ROOT / path).read_text().count(original), 1)
            self.assertNotEqual(original, mutated)


class TensorBackwardRuntimeTests(unittest.TestCase):
    def test_native_and_javascript_lanes(self):
        with tempfile.TemporaryDirectory(
            prefix="autograd-tensor-backward-"
        ) as directory:
            summary = run_gate(ROOT / ".tools", Path(directory) / "run")
        self.assertEqual(summary["native"], summary["javascript"])
        self.assertEqual(summary["native"]["cases"], len(cases()))
        self.assertEqual(
            summary["runtime_controls"], [*RUNTIME_CONTROLS, *NATIVE_CONTROLS]
        )


if __name__ == "__main__":
    unittest.main()
