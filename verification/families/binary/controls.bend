import Base
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/elementwise.bend as Elementwise
import ./consumers.bend as Cases

def mul_word() -> Maybe<&2, F32>:
  Some{F32.mul(5.0, 10.0)}

def swapped_mul_word() -> Maybe<&2, F32>:
  Some{F32.mul(10.0, 5.0)}

def sub_word() -> Maybe<&2, F32>:
  Some{F32.sub(7.0, 3.0)}

def reversed_sub_word() -> Maybe<&2, F32>:
  Some{F32.sub(3.0, 7.0)}

def negated_word() -> Maybe<&2, F32>:
  Some{F32.neg(3.0)}

def unsigned_word() -> Maybe<&2, F32>:
  Some{3.0}

def kept_word() -> Maybe<&2, F32>:
  Some{5.0}

def other_word() -> Maybe<&2, F32>:
  Some{3.0}

def chain_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))}

def single_product() -> Maybe<&2, F32>:
  Some{F32.add(0.0, F32.mul(5.0, 10.0))}

def output_of(-pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32, -word: F32,
  +left_index: U32, -left_word: F32, +right_index: U32, -right_word: F32,
  evidence: Cases.BinaryCase<pair, index, word, left_index, left_word, right_index, right_word>) ->
  {Cases.output_point(pair, index) == Some{word} : Maybe<&2, F32>}:
  Cases.BinaryCase{value, left_kept, right_kept} = evidence
  value

def left_of(-pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32, -word: F32,
  +left_index: U32, -left_word: F32, +right_index: U32, -right_word: F32,
  evidence: Cases.BinaryCase<pair, index, word, left_index, left_word, right_index, right_word>) ->
  {Cases.left_point(pair, left_index) == Some{left_word} : Maybe<&2, F32>}:
  Cases.BinaryCase{value, left_kept, right_kept} = evidence
  left_kept

# The kernel reports a false twin's mismatch against the type the body has.
# Each value witness restates the cell as the literal it evaluates to.
def stated(+claim: Maybe<&2, F32>, +value: Maybe<&2, F32>, equal: {claim == value : Maybe<&2, F32>}) ->
  {claim == value : Maybe<&2, F32>}:
  equal

def mul_witness() ->
  {Cases.output_point(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}), 1) == mul_word() : Maybe<&2, F32>}:
  stated(Cases.output_point(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}), 1),
    Some{F32.mul(5.0, 10.0)},
    output_of(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}),
      1, F32.mul(5.0, 10.0), 1, 5.0, 0, 10.0, Cases.mul_broadcast()))

def sub_witness() ->
  {Cases.output_point(Cases.result(Elementwise.Sub{}, Cases.one_shape(), [7.0], Cases.two_shape(), [2.0, 3.0], {==}, {==}), 1) == sub_word() : Maybe<&2, F32>}:
  stated(Cases.output_point(Cases.result(Elementwise.Sub{}, Cases.one_shape(), [7.0], Cases.two_shape(), [2.0, 3.0], {==}, {==}), 1),
    Some{F32.sub(7.0, 3.0)},
    output_of(Cases.result(Elementwise.Sub{}, Cases.one_shape(), [7.0], Cases.two_shape(), [2.0, 3.0], {==}, {==}),
      1, F32.sub(7.0, 3.0), 0, 7.0, 1, 3.0, Cases.sub_broadcast()))

def negate_witness() ->
  {Cases.output_point(Cases.result(Elementwise.NegateLeft{}, Cases.two_shape(), Cases.gradient(), Cases.scalar_shape(), [0.0], {==}, {==}), 0) == negated_word() : Maybe<&2, F32>}:
  stated(Cases.output_point(Cases.result(Elementwise.NegateLeft{}, Cases.two_shape(), Cases.gradient(), Cases.scalar_shape(), [0.0], {==}, {==}), 0),
    Some{F32.neg(3.0)},
    output_of(Cases.result(Elementwise.NegateLeft{}, Cases.two_shape(), Cases.gradient(), Cases.scalar_shape(), [0.0], {==}, {==}),
      0, F32.neg(3.0), 1, 5.0, 0, 0.0, Cases.negate_scalar()))

def kept_witness() ->
  {Cases.left_point(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}), 1) == kept_word() : Maybe<&2, F32>}:
  stated(Cases.left_point(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}), 1),
    Some{5.0},
    left_of(Cases.result(Elementwise.Mul{}, Cases.two_shape(), Cases.gradient(), Cases.one_shape(), [10.0], {==}, {==}),
      1, F32.mul(5.0, 10.0), 1, 5.0, 0, 10.0, Cases.mul_broadcast()))

def chain_witness() -> {Cases.chained(0) == chain_sum() : Maybe<&2, F32>}:
  stated(Cases.chained(0), Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))}, Cases.binary_then_reduce())
