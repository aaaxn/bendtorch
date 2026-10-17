import json
import unittest
from fractions import Fraction

from tools.autograd_broadcast import word
from tools.autograd_reduce import value
from tools.autograd_tensor_backward import cases, reference
from tools.autograd_tensor_bound import (
    UNDERFLOW,
    Bounded,
    Line,
    bounded_add,
    bounded_dot,
    bounded_exp,
    bounded_mul,
    bounded_reference,
    bounded_total,
    bounded_word,
    compare,
    gamma,
    lift,
    within,
    word_of,
)
from tools.common import ROOT, GateError

FIXTURE = ROOT / "oracle/autograd/fixtures/tensor-backward-torch.json"
# The scripts the level-2 comparison reads; the other families of
# tests/test_autograd_tensor_backward.py (refusals, forged tapes, NaN payloads,
# the fused SGD update, the non-leaf step) are compared there on their own.
OTHER_FAMILIES = {
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
    "mse_refused",
    "sgd_unknown_param",
    "sgd_rate_negative",
    "cross_entropy_refused",
    "sum_dims_refused",
    "batched_refusals",
    "dangling_self",
    "dangling_newer",
    "dangling_missing",
    "specials",
    "relu_edges",
    "relu_nan_gradient",
    "mse_empty",
    "cross_entropy_infinities",
    "cross_entropy_nan",
    "cross_entropy_empty",
    "cross_entropy_empty_wide",
    "cross_entropy_one_class_scaled",
    "cross_entropy_two_class_scaled",
    "sgd_edges",
    "sgd_rate_nan",
    "batched_special_words",
    "sgd_two_layers",
    "sgd_rounding",
    "sgd_not_leaf",
    "batched_output_capacity",
    "batched_left_gradient_capacity",
    "batched_right_gradient_capacity",
}


def next_up(number: int) -> int:
    if number == 0x80000000:
        return 1
    return number - 1 if number & 0x80000000 else number + 1


def line(number: int, bound: Bounded | None = None) -> str:
    text = f"tensor:word:{number}"
    return text if bound is None else Line(text, bound)


class BoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads(FIXTURE.read_bytes())
        cls.rows = {row["id"]: row for row in cases()}
        cls.seen = {row["id"]: row.get("lines") for row in fixture["backward"]}
        cls.compared = [
            identifier for identifier in cls.rows if identifier not in OTHER_FAMILIES
        ]

    def test_gamma_and_the_sum_bound(self):
        self.assertEqual(gamma(0), 0)
        self.assertEqual(gamma(1), Fraction(1, 2**24 - 1))
        # 2^24 + 1 + 1 rounds to 2^24 in this order and to 2^24 + 2 in the
        # other; both lie within gamma(2) * (2^24 + 2) of the exact 2^24 + 2.
        terms = [word(2.0**24), word(1.0), word(1.0)]
        total = bounded_total(terms)
        self.assertEqual(total.exact, 2**24 + 2)
        self.assertEqual(total.error, gamma(2) * (2**24 + 2))
        self.assertFalse(total.fixed)
        for number in (word(2.0**24), word(2.0**24 + 2)):
            self.assertTrue(within(number, total))
        self.assertFalse(within(word(2.0**24 + 8), total))

    def test_no_bound_where_another_order_could_overflow(self):
        # M + -M + M is M in this order, and (M + M) + -M overflows: no bound.
        largest = 0x7F7FFFFF
        total = bounded_total([largest, largest | 0x80000000, largest])
        self.assertEqual(int(total), largest)
        self.assertFalse(total.bounded)
        self.assertFalse(within(largest, total))
        self.assertFalse(bounded_dot([(largest, word(1.0))] * 2, largest).bounded)
        # A word under a bound that is not fixed, at the edge of the range.
        rounding = bounded_mul(bounded_total([largest, word(0.0)]), word(1.0))
        self.assertFalse(rounding.bounded)
        # A smaller sum keeps its bound.
        self.assertTrue(bounded_total([word(2.0**126), word(-(2.0**126))]).bounded)
        # An empty sum or dot is the fixed +0.0.
        self.assertEqual((int(bounded_total([])), bounded_total([]).fixed), (0, True))
        self.assertEqual(int(bounded_dot([], 0)), 0)
        self.assertTrue(bounded_dot([], 0).fixed)

    def test_one_rounding_words_are_fixed_and_sums_are_not(self):
        two, three = lift(word(2.0)), lift(word(3.0))
        self.assertTrue(bounded_mul(two, three).fixed)
        self.assertTrue(bounded_add(two, three).fixed)
        self.assertFalse(bounded_total([two, three]).fixed)
        self.assertTrue(bounded_total([]).fixed)
        self.assertFalse(bounded_exp(two).bounded)

    def test_the_replay_prints_the_reference_and_holds_it_within_its_bound(self):
        kinds = {"fixed": 0, "bounded": 0, "unbounded": 0}
        for identifier in self.compared:
            lines = bounded_reference(self.rows[identifier])
            self.assertEqual(lines, reference(self.rows[identifier]))
            for text in lines:
                if word_of(text) is None:
                    continue
                bound = bounded_word(text)
                if bound.fixed:
                    kinds["fixed"] += 1
                elif not bound.bounded:
                    kinds["unbounded"] += 1
                else:
                    kinds["bounded"] += 1
                    self.assertTrue(within(int(bound), bound), (identifier, text))
        self.assertEqual(kinds, {"fixed": 9455, "bounded": 3853, "unbounded": 431})

    def bounded_line(self, identifier: str, index: int) -> tuple[list[str], Bounded]:
        expected = bounded_reference(self.rows[identifier])
        return expected, bounded_word(expected[index])

    def test_a_word_past_the_bound_is_rejected(self):
        # loss_03, line 151: the mse_loss word, a sum of squares over a count;
        # PyTorch's word is 1 ulp from the reference's, and the bound 57 ulp.
        expected, bound = self.bounded_line("loss_03", 151)
        seen = list(self.seen["loss_03"])
        self.assertNotEqual(seen[151], expected[151])
        top = bound.exact + bound.error
        inside = int(bound)
        while Fraction(value(next_up(inside))) <= top:
            inside = next_up(inside)
        outside = next_up(inside)
        self.assertGreater(Fraction(value(outside)), top)
        seen[151] = line(inside)
        compare("loss_03", seen, expected)
        seen[151] = line(outside)
        with self.assertRaises(GateError):
            compare("loss_03", seen, expected)
        # The same word on the reference's side, as a wrong Bend result.
        wrong = list(expected)
        wrong[151] = line(outside, bound)
        with self.assertRaises(GateError):
            compare("loss_03", self.seen["loss_03"], wrong)

    def test_a_fixed_word_one_ulp_off_is_rejected(self):
        expected = bounded_reference(self.rows["loss_03"])
        index = next(
            index
            for index, text in enumerate(expected)
            if text.startswith("tensor:word:")
            and bounded_word(text).fixed
            and value(bounded_word(text)) != 0
            and expected[index] == self.seen["loss_03"][index]
        )
        seen = list(self.seen["loss_03"])
        seen[index] = line(next_up(bounded_word(expected[index])))
        with self.assertRaises(GateError):
            compare("loss_03", seen, expected)

    def test_a_zero_takes_either_sign_only_where_the_bound_reaches_zero(self):
        reaches = Bounded(0, -UNDERFLOW, UNDERFLOW, True, False)
        short = Bounded(0, Fraction(1, 2**140), UNDERFLOW, True, False)
        self.assertTrue(within(0x80000000, reaches))
        self.assertFalse(within(0x80000000, short))
        expected = [line(0, short)]
        with self.assertRaises(GateError):
            compare("zero", [line(0x80000000)], expected)
        compare("zero", [line(0x80000000)], [line(0, reaches)])
        # A fixed zero keeps its sign.
        with self.assertRaises(GateError):
            compare("zero", [line(0x80000000)], [line(0, lift(0))])

    def test_a_pin_is_refused_on_a_word_that_has_a_bound(self):
        expected, bound = self.bounded_line("loss_03", 151)
        pair = (int(self.seen["loss_03"][151].removeprefix("tensor:word:")), bound)
        with self.assertRaises(GateError):
            compare("loss_03", self.seen["loss_03"], expected, {151: pair})
        # And a pin must be used.
        expected = bounded_reference(self.rows["cross_entropy"])
        with self.assertRaises(GateError):
            compare("cross_entropy", self.seen["cross_entropy"], expected, {})
        with self.assertRaises(GateError):
            compare(
                "cross_entropy",
                self.seen["cross_entropy"],
                expected,
                {21: (0x3D6CC5F4, 0x3D6CC5F5), 22: (0, 0)},
            )


if __name__ == "__main__":
    unittest.main()
