"""The rounding-error bound of each word of the tensor backward reference.

AGENTS.md's numeric policy compares the tensor backward with PyTorch at level
2: a word that only operations with one rounding build from the leaves must
be PyTorch's word, bit for bit; a word built on a sum or a matmul must lie
within a bound of the exact value of the script's arithmetic, because PyTorch
adds in another order. This module computes, for every word the reference
prints, that exact value, the bound and whether the word is fixed, and
compares PyTorch's lines with the reference's under that rule.

The bounds hold under these conditions, and a word is marked bounded only
where the replay can show them: binary32 arithmetic that rounds to nearest,
ties to even, with gradual underflow (subnormals kept, so an addition is exact
in the subnormal range); no intermediate overflow in any order, which the
replay requires as `(1 + gamma(n)) * sum |terms| <= LARGEST` for every sum
and dot product and `(1 + u) * (|exact| + spread) <= LARGEST` for every other
rounding that is not fixed; and `n * u < 1` for every count `n`. A word for
which a condition fails, or which is not finite, has no bound and goes to the
special-word checks. An empty sum and a dot product of no terms are the fixed
word `+0.0` in every order, compared bit for bit.

It replays the reference itself (`autograd_tensor_backward.reference`) with
its arithmetic functions replaced by ones that return the same word together
with that information, so the replay runs the same operations in the same
order; `bounded_reference` checks that it prints the reference's lines.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction

from tools import autograd_tensor_backward as tape
from tools.autograd_backward import add_word, mul_word, neg_word, sub_word
from tools.autograd_reduce import value
from tools.common import GateError

# The unit roundoff of binary32, and the largest absolute error of rounding a
# product or a quotient into the subnormal range (half the smallest subnormal).
UNIT = Fraction(1, 2**24)
UNDERFLOW = Fraction(1, 2**150)
ZERO = Fraction(0)
# The largest finite binary32 value, 0x7F7FFFFF.
LARGEST = (2 - Fraction(1, 2**23)) * 2**127

div_word = tape.div_word
exp_word = tape.exp_word
log_word = tape.log_word
max_word = tape.max_word
relu_word = tape.relu_word
relu_backward_word = tape.relu_backward_word
snapshot = tape.snapshot


# gamma(n) = n*u / (1 - n*u) bounds the relative error of n roundings.
def gamma(count: int) -> Fraction:
    return count * UNIT / (1 - count * UNIT)


# Whether no partial result of `count` roundings in any order can overflow,
# when the terms' magnitudes, with their spreads, add up to `magnitude`.
def in_range(count: int, magnitude: Fraction) -> bool:
    return count * UNIT < 1 and (1 + gamma(count)) * magnitude <= LARGEST


class Bounded(int):
    """An F32 word with what every run that follows the policy agrees on.

    `exact` is the value of the script's arithmetic on its leaf words, done
    without rounding. `error` bounds the distance from `exact` of the word that
    any run gives which takes the reference's operations and adds each sum or
    matmul in any order, with or without fused multiply-adds, under the
    module's conditions. `bounded` is False when the word is not finite, when
    no bound is stated (exp, log) or when a run in another order could
    overflow; `exact` and `error` mean nothing then. `fixed` says that every such run
    gives this very word: no sum, matmul, exp or log comes before it.
    """

    exact: Fraction
    error: Fraction
    bounded: bool
    fixed: bool

    def __new__(
        cls, number: int, exact: Fraction, error: Fraction, bounded: bool, fixed: bool
    ) -> "Bounded":
        self = super().__new__(cls, number)
        self.bounded = bounded and math.isfinite(value(number))
        self.exact, self.error = (exact, error) if self.bounded else (ZERO, ZERO)
        self.fixed = fixed
        return self


class Line(str):
    """A printed word line of the reference, with its word's bound."""

    word: Bounded

    def __new__(cls, text: str, number: Bounded) -> "Line":
        self = super().__new__(cls, text)
        self.word = number
        return self


def lift(number: int) -> Bounded:
    if isinstance(number, Bounded):
        return number
    finite = math.isfinite(value(number))
    exact = Fraction(value(number)) if finite else ZERO
    return Bounded(number, exact, ZERO, finite, True)


def unbounded(number: int, fixed: bool) -> Bounded:
    return Bounded(number, ZERO, ZERO, False, fixed)


# The word of one rounding of a result whose operands are off by `spread`.
def rounded(
    number: int,
    fixed: bool,
    exact: Fraction,
    spread: Fraction,
    underflow: Fraction = ZERO,
) -> Bounded:
    error = spread + UNIT * (abs(exact) + spread) + underflow
    safe = fixed or in_range(1, abs(exact) + spread)
    return Bounded(number, exact, error, safe, fixed)


# An addition rounds once and is exact in the subnormal range.
def bounded_add(left: int, right: int) -> Bounded:
    a, b = lift(left), lift(right)
    number, fixed = add_word(a, b), a.fixed and b.fixed
    if not (a.bounded and b.bounded):
        return unbounded(number, fixed)
    return rounded(number, fixed, a.exact + b.exact, a.error + b.error)


def bounded_sub(left: int, right: int) -> Bounded:
    a, b = lift(left), lift(right)
    number, fixed = sub_word(a, b), a.fixed and b.fixed
    if not (a.bounded and b.bounded):
        return unbounded(number, fixed)
    return rounded(number, fixed, a.exact - b.exact, a.error + b.error)


def bounded_neg(number: int) -> Bounded:
    a = lift(number)
    return Bounded(neg_word(a), -a.exact, a.error, a.bounded, a.fixed)


def product_spread(a: Bounded, b: Bounded) -> Fraction:
    return abs(a.exact) * b.error + abs(b.exact) * a.error + a.error * b.error


def bounded_mul(left: int, right: int) -> Bounded:
    a, b = lift(left), lift(right)
    number, fixed = mul_word(a, b), a.fixed and b.fixed
    if not (a.bounded and b.bounded):
        return unbounded(number, fixed)
    exact = a.exact * b.exact
    return rounded(number, fixed, exact, product_spread(a, b), UNDERFLOW)


def bounded_div(left: int, right: int) -> Bounded:
    a, b = lift(left), lift(right)
    number, fixed = div_word(a, b), a.fixed and b.fixed
    if not (a.bounded and b.bounded) or abs(b.exact) <= b.error:
        return unbounded(number, fixed)
    exact = a.exact / b.exact
    spread = (abs(b.exact) * a.error + abs(a.exact) * b.error) / (
        abs(b.exact) * (abs(b.exact) - b.error)
    )
    return rounded(number, fixed, exact, spread, UNDERFLOW)


# PyTorch's exp and log are vectorized approximations whose error no bound
# here states, so their words, and every word built on them, are unbounded.
def bounded_exp(number: int) -> Bounded:
    return unbounded(exp_word(number), False)


def bounded_log(number: int) -> Bounded:
    return unbounded(log_word(number), False)


# A maximum selects one operand, so it is off by at most the larger spread.
def bounded_max(held: int, number: int) -> Bounded:
    a, b = lift(held), lift(number)
    word, fixed = max_word(a, b), a.fixed and b.fixed
    exact, error = max(a.exact, b.exact), max(a.error, b.error)
    return Bounded(word, exact, error, a.bounded and b.bounded, fixed)


# relu is 1-Lipschitz and rounds nothing.
def bounded_relu(number: int) -> Bounded:
    a = lift(number)
    return Bounded(relu_word(a), max(a.exact, ZERO), a.error, a.bounded, a.fixed)


# The relu gradient passes g where x > 0 and gives +0.0 elsewhere. When x is
# not fixed, the bound holds only if no run can put x on the other side.
def bounded_relu_backward(x: int, g: int) -> Bounded:
    a, b = lift(x), lift(g)
    word = relu_backward_word(a, b)
    if a.fixed:
        return lift(word) if tape.at_most_zero(a) else b
    if not a.bounded:
        return unbounded(word, False)
    if a.exact - a.error > 0:
        return b
    if a.exact + a.error <= 0:
        return lift(word)
    return unbounded(word, False)


# A sum of n terms in any order rounds at most n - 1 times on any path:
# |fl(sum) - sum| <= gamma(n - 1) * sum |terms| (Higham, Accuracy and Stability
# of Numerical Algorithms, 2nd ed., section 4.2), plus the terms' own spread.
# The reference's word is its fold from +0.0, left to right.
def bounded_total(words: list[int]) -> Bounded:
    number = 0
    for word in words:
        number = add_word(number, int(word))
    terms = [lift(word) for word in words]
    if not terms:
        return lift(number)
    spread = sum((term.error for term in terms), ZERO)
    magnitude = sum((abs(term.exact) for term in terms), ZERO) + spread
    exact = sum((term.exact for term in terms), ZERO)
    count = len(terms) - 1
    bounded = all(term.bounded for term in terms) and in_range(count, magnitude)
    error = spread + gamma(count) * magnitude if bounded else ZERO
    return Bounded(number, exact, error, bounded, False)


# A dot product of K terms in any order, fused or not:
# |fl(x . y) - x . y| <= gamma(K) * sum |x_k y_k| (Higham, section 3.1), plus
# K * eta / (1 - K u) for products that round into the subnormal range.
def bounded_dot(pairs: list[tuple[int, int]], number: int) -> Bounded:
    if not pairs:
        return lift(number)
    factors = [(lift(a), lift(b)) for a, b in pairs]
    count = len(factors)
    spread = sum((product_spread(a, b) for a, b in factors), ZERO)
    magnitude = sum((abs(a.exact * b.exact) for a, b in factors), ZERO)
    exact = sum((a.exact * b.exact for a, b in factors), ZERO)
    bounded = all(a.bounded and b.bounded for a, b in factors)
    if not (bounded and in_range(count, magnitude + spread)):
        return unbounded(number, False)
    underflow = count * UNDERFLOW / (1 - count * UNIT)
    error = spread + gamma(count) * (magnitude + spread) + underflow
    return Bounded(number, exact, error, True, False)


# The reference's matmul, each cell the ordered fold of its products from +0.0.
def bounded_matmul(left: tuple, right: tuple) -> tuple | int:
    if len(left[0]) != 2 or len(right[0]) != 2:
        return tape.MATMUL_RANK
    (rows, inner), (other, columns) = left[0], right[0]
    if inner != other:
        return tape.MATMUL_INNER
    words = []
    for row in range(rows):
        for column in range(columns):
            pairs = [
                (left[1][row * inner + k], right[1][k * columns + column])
                for k in range(inner)
            ]
            number = 0
            for a, b in pairs:
                number = add_word(number, mul_word(int(a), int(b)))
            words.append(bounded_dot(pairs, number))
    return ([rows, columns], words)


def bounded_sum_to_size(tensor: tuple, target: list[int]) -> tuple:
    shape, words = tensor
    if shape == target:
        return (target, list(words))
    fibers: list[list[int]] = [[] for _ in range(math.prod(target))]
    for flat in range(math.prod(shape)):
        fibers[tape.source_index(target, shape, flat)].append(words[flat])
    return (target, [bounded_total(fiber) for fiber in fibers])


def bounded_snapshot(tensor: tuple) -> list[str]:
    head, count, *rows = snapshot(tensor)
    return [
        head,
        count,
        *[Line(row, lift(word)) for row, word in zip(rows, tensor[1], strict=True)],
    ]


REPLACED = {
    "add_word": bounded_add,
    "sub_word": bounded_sub,
    "neg_word": bounded_neg,
    "mul_word": bounded_mul,
    "div_word": bounded_div,
    "exp_word": bounded_exp,
    "log_word": bounded_log,
    "max_word": bounded_max,
    "relu_word": bounded_relu,
    "relu_backward_word": bounded_relu_backward,
    "total": bounded_total,
    "matmul": bounded_matmul,
    "sum_to_size": bounded_sum_to_size,
    "snapshot": bounded_snapshot,
}


@contextmanager
def replaced(functions: dict) -> Iterator[None]:
    saved = {name: getattr(tape, name) for name in functions}
    try:
        for name, function in functions.items():
            setattr(tape, name, function)
        yield
    finally:
        for name, function in saved.items():
            setattr(tape, name, function)


# The reference's lines for one script, each word line carrying its bound.
# A test passes other functions to plant a wrong result.
def bounded_lines(case: dict, functions: dict = REPLACED) -> list[str]:
    with replaced(functions):
        return tape.reference(case)


def bounded_reference(case: dict) -> list[str]:
    lines = bounded_lines(case)
    if lines != tape.reference(case):
        raise GateError(f"{case['id']}: the bounded replay printed other lines")
    return lines


def word_of(line: str) -> int | None:
    text = line.removeprefix("tensor:word:")
    return int(text) if text != line and text.isdigit() else None


def bounded_word(line: str) -> Bounded:
    if isinstance(line, Line):
        return line.word
    number = word_of(line)
    if number is None:
        raise GateError(f"Not a word line: {line}")
    return lift(number)


# Whether a word lies within a bound. The two zeros have the same value, so a
# zero passes with either sign when the bound reaches zero: the sign of a zero
# sum depends on where its fold starts.
def within(number: int, bound: Bounded) -> bool:
    if not bound.bounded or not math.isfinite(value(number)):
        return False
    return abs(Fraction(value(number)) - bound.exact) <= bound.error


# Level 2 for one script. Each line PyTorch printed must equal the
# reference's line, or both must be words that the reference does not fix and
# that lie within its bound. `pins` lists, by line index, the (PyTorch,
# reference) word pairs allowed where no bound is stated. Returns how many
# word lines passed within the bound and on a pin.
def compare(
    identifier: str,
    seen: list[str],
    expected: list[str],
    pins: dict[int, tuple[int, int]] | None = None,
) -> tuple[int, int]:
    pins = pins or {}
    if len(seen) != len(expected):
        raise GateError(f"{identifier}: {len(seen)} lines, expected {len(expected)}")
    bounded = pinned = 0
    for index, (a, b) in enumerate(zip(seen, expected, strict=True)):
        if a == b:
            if index in pins:
                raise GateError(f"{identifier}:{index}: a pinned word agrees")
            continue
        number, own = word_of(a), word_of(b)
        if number is None or own is None:
            raise GateError(f"{identifier}:{index}: {a} differs from {b}")
        bound = bounded_word(b)
        if index in pins:
            if bound.bounded or pins[index] != (number, own):
                raise GateError(f"{identifier}:{index}: not the pinned pair")
            pinned += 1
            continue
        if bound.fixed:
            raise GateError(f"{identifier}:{index}: {number} is not the fixed {own}")
        if not bound.bounded:
            raise GateError(f"{identifier}:{index}: {number} has no bound and no pin")
        if not within(number, bound) or not within(own, bound):
            raise GateError(f"{identifier}:{index}: {number} is outside the bound")
        bounded += 1
    if pinned != len(pins):
        raise GateError(f"{identifier}: a pinned line is missing")
    return bounded, pinned
