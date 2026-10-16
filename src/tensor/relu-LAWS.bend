import Base
import ./elementwise.bend as Elementwise

# ReLU and its gradient as elementwise operations, and the two F32
# comparisons with zero they need, pinned on the IEEE 754 bits.

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

# The edge words: +0.0, -0.0, the smallest and largest denormals, the
# smallest normal, 1.0, the largest finite, the infinities, the signaling
# NaN 0x7F800001, the quiet NaN 0x7FC00000 and the all-ones NaN 0x7FFFFFFF,
# each followed by its negative.
def edges() -> List<&2, U32>:
  [0, 2147483648, 1, 2147483649, 8388607, 2155872255, 8388608, 2155872256, 1065353216, 3212836864,
    2139095039, 4286578687, 2139095040, 4286578688, 2139095041, 4286578689, 2143289344, 4290772992,
    2147483647, 4294967295]

def below_zero_at(words: List<&2, U32>) -> List<&2, Bool>:
  match words:
    case Nil{}:
      Nil{}
    case Con{word, rest}:
      Elementwise.below_zero(from_bits(word)) <> below_zero_at(rest)

def at_most_zero_at(words: List<&2, U32>) -> List<&2, Bool>:
  match words:
    case Nil{}:
      Nil{}
    case Con{word, rest}:
      Elementwise.at_most_zero(from_bits(word)) <> at_most_zero_at(rest)

# The output words of `operation` with each word as the left operand and
# `right` as the right one.
def scalar_at(+operation: Elementwise.Operation, +right: U32, words: List<&2, U32>) -> List<&2, U32>:
  match words:
    case Nil{}:
      Nil{}
    case Con{word, rest}:
      Elementwise.bits(Elementwise.scalar(operation, from_bits(word), from_bits(right))) <> scalar_at(operation, right, rest)

# `below_zero` is IEEE 754 `x < 0`: true exactly on the words above -0.0
# (0x80000000) up to -inf (0xFF800000). +0.0, -0.0 and every NaN, of either
# sign, are not below zero.
law below_zero_words:
  for +x: F32
  {Elementwise.below_zero(x) ==
    Bool.and(U32.is_lt(2147483648, Elementwise.bits(x)), U32.is_le(Elementwise.bits(x), 4286578688)) : Bool}

# `at_most_zero` is IEEE 754 `x <= 0`: true exactly on +0.0 (0x00000000) and
# the words from -0.0 (0x80000000) up to -inf (0xFF800000). No NaN is at most
# zero.
law at_most_zero_words:
  for +x: F32
  {Elementwise.at_most_zero(x) ==
    Bool.or(U32.is_eq(Elementwise.bits(x), 0),
      Bool.and(U32.is_le(2147483648, Elementwise.bits(x)), U32.is_le(Elementwise.bits(x), 4286578688))) : Bool}

# The two comparisons at the edge words, as PyTorch's `x < 0` and `x <= 0`
# answer them.
law below_zero_edges:
  {below_zero_at(edges()) ==
    [False{}, False{}, False{}, True{}, False{}, True{}, False{}, True{}, False{}, True{},
      False{}, True{}, False{}, True{}, False{}, False{}, False{}, False{}, False{}, False{}] : List<&2, Bool>}

law at_most_zero_edges:
  {at_most_zero_at(edges()) ==
    [True{}, True{}, False{}, True{}, False{}, True{}, False{}, True{}, False{}, True{},
      False{}, True{}, False{}, True{}, False{}, False{}, False{}, False{}, False{}, False{}] : List<&2, Bool>}

# ReluLeft is PyTorch's relu, clamp_min(x, 0), on the left word: +0.0 where
# the word is below zero, and the word itself otherwise, so -0.0 stays -0.0
# and a NaN keeps its sign and payload. The right word is ignored: the
# forward is the pointwise product with a rank-0 +0.0.
law scalar_relu_left_below:
  for +left: F32
  for +right: F32
  for below: {Elementwise.below_zero(left) == True{} : Bool}
  {Elementwise.scalar(Elementwise.ReluLeft{}, left, right) == 0.0 : F32}

law scalar_relu_left_kept:
  for +left: F32
  for +right: F32
  for kept: {Elementwise.below_zero(left) == False{} : Bool}
  {Elementwise.scalar(Elementwise.ReluLeft{}, left, right) == left : F32}

# ReluBackward is PyTorch's relu gradient, threshold_backward(g, relu(x), 0),
# with x on the left and g on the right: +0.0 where x is at most zero, so the
# gradient at +0.0 and -0.0 is +0.0, and g's own word otherwise, NaN x
# included.
law scalar_relu_backward_zeroed:
  for +left: F32
  for +right: F32
  for zeroed: {Elementwise.at_most_zero(left) == True{} : Bool}
  {Elementwise.scalar(Elementwise.ReluBackward{}, left, right) == 0.0 : F32}

law scalar_relu_backward_passed:
  for +left: F32
  for +right: F32
  for passed: {Elementwise.at_most_zero(left) == False{} : Bool}
  {Elementwise.scalar(Elementwise.ReluBackward{}, left, right) == right : F32}

# The words PyTorch 2.14.1's torch.relu returns at the edge words.
law relu_edges:
  {scalar_at(Elementwise.ReluLeft{}, 0, edges()) ==
    [0, 2147483648, 1, 0, 8388607, 0, 8388608, 0, 1065353216, 0,
      2139095039, 0, 2139095040, 0, 2139095041, 4286578689, 2143289344, 4290772992,
      2147483647, 4294967295] : List<&2, U32>}

# The gradient words PyTorch 2.14.1 gives torch.relu at the edge words, for
# g = 2.5 (0x40200000).
law relu_backward_edges:
  {scalar_at(Elementwise.ReluBackward{}, 1075838976, edges()) ==
    [0, 0, 1075838976, 0, 1075838976, 0, 1075838976, 0, 1075838976, 0,
      1075838976, 0, 1075838976, 0, 1075838976, 1075838976, 1075838976, 1075838976,
      1075838976, 1075838976] : List<&2, U32>}
