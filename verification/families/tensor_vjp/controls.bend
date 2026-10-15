import Base
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/autograd/tensor-backward.bend as Backward
import ../../../src/autograd/tensor-backward-LAWS.bend as TensorBackward
import ../../../src/autograd/tensor-vjp-LAWS.bend as Laws
import ./consumers.bend as Cases

def kept_word() -> Maybe<&2, F32>:
  Some{5.0}

def folded_word() -> Maybe<&2, F32>:
  Some{F32.add(0.0, 5.0)}

def added_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, 3.0), 5.0)}

def reversed_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, 5.0), 3.0)}

def negated_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))}

def unsigned_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, 3.0), 5.0)}

def right_product() -> Maybe<&2, F32>:
  Some{F32.mul(5.0, 10.0)}

def left_product() -> Maybe<&2, F32>:
  Some{F32.mul(5.0, 7.0)}

def left_products() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))}

def right_products() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))}

def first_point(-pass: Backward.Pass, +first: U32, -first_word: Maybe<&2, F32>, +second: U32, -second_word: Maybe<&2, F32>,
  evidence: Cases.PairCase<pass, first, first_word, second, second_word>) ->
  {TensorBackward.sole_point(Laws.first_of(pass), first) == first_word : Maybe<&2, F32>}:
  Cases.PairCase{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  first_value

def second_point(-pass: Backward.Pass, +first: U32, -first_word: Maybe<&2, F32>, +second: U32, -second_word: Maybe<&2, F32>,
  evidence: Cases.PairCase<pass, first, first_word, second, second_word>) ->
  {TensorBackward.sole_point(Laws.second_of(pass), second) == second_word : Maybe<&2, F32>}:
  Cases.PairCase{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  second_value

# The kernel reports a false twin's mismatch against the type the body has.
# Each value witness restates the cell as the literal it evaluates to.
def stated(+claim: Maybe<&2, F32>, +value: Maybe<&2, F32>, equal: {claim == value : Maybe<&2, F32>}) ->
  {claim == value : Maybe<&2, F32>}:
  equal

def add_kept_witness() ->
  {TensorBackward.sole_point(Laws.first_of(Cases.added(Cases.gradient(), {==})), 1) == kept_word() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.first_of(Cases.added(Cases.gradient(), {==})), 1), Some{5.0},
    first_point(Cases.added(Cases.gradient(), {==}), 1, Some{5.0}, 0, Some{F32.add(F32.add(0.0, 3.0), 5.0)}, Cases.add_values()))

def add_sum_witness() ->
  {TensorBackward.sole_point(Laws.second_of(Cases.added(Cases.gradient(), {==})), 0) == added_sum() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.second_of(Cases.added(Cases.gradient(), {==})), 0), Some{F32.add(F32.add(0.0, 3.0), 5.0)},
    second_point(Cases.added(Cases.gradient(), {==}), 1, Some{5.0}, 0, Some{F32.add(F32.add(0.0, 3.0), 5.0)}, Cases.add_values()))

def sub_witness() ->
  {TensorBackward.sole_point(Laws.second_of(Cases.subtracted(Cases.gradient(), {==})), 0) == negated_sum() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.second_of(Cases.subtracted(Cases.gradient(), {==})), 0),
    Some{F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))},
    second_point(Cases.subtracted(Cases.gradient(), {==}), 0, Some{3.0}, 0, Some{F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))},
      Cases.sub_values()))

def mul_witness() ->
  {TensorBackward.sole_point(Laws.first_of(Cases.multiplied(Cases.gradient(), {==})), 1) == right_product() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.first_of(Cases.multiplied(Cases.gradient(), {==})), 1), Some{F32.mul(5.0, 10.0)},
    first_point(Cases.multiplied(Cases.gradient(), {==}), 1, Some{F32.mul(5.0, 10.0)}, 0,
      Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))}, Cases.mul_values()))

def mul_sum_witness() ->
  {TensorBackward.sole_point(Laws.second_of(Cases.multiplied(Cases.gradient(), {==})), 0) == left_products() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.second_of(Cases.multiplied(Cases.gradient(), {==})), 0),
    Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))},
    second_point(Cases.multiplied(Cases.gradient(), {==}), 1, Some{F32.mul(5.0, 10.0)}, 0,
      Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))}, Cases.mul_values()))

def front_product() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0))}

def untransposed_product() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 20.0)), F32.mul(5.0, 50.0))}

def back_product() -> Maybe<&2, F32>:
  Some{F32.add(0.0, F32.mul(7.0, 3.0))}

def swapped_product() -> Maybe<&2, F32>:
  Some{F32.add(0.0, F32.mul(3.0, 7.0))}

def matmul_first_point(-pass: Backward.Pass, +first: U32, -first_word: F32, +second: U32, -second_word: F32,
  evidence: Cases.MatmulCase<pass, first, first_word, second, second_word>) ->
  {TensorBackward.sole_point(Laws.first_of(pass), first) == Some{first_word} : Maybe<&2, F32>}:
  Cases.MatmulCase{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  first_value

def matmul_second_point(-pass: Backward.Pass, +first: U32, -first_word: F32, +second: U32, -second_word: F32,
  evidence: Cases.MatmulCase<pass, first, first_word, second, second_word>) ->
  {TensorBackward.sole_point(Laws.second_of(pass), second) == Some{second_word} : Maybe<&2, F32>}:
  Cases.MatmulCase{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  second_value

def matmul_front_witness() ->
  {TensorBackward.sole_point(Laws.first_of(Cases.matmul_pass(Cases.gradient(), {==})), 1) == front_product() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.first_of(Cases.matmul_pass(Cases.gradient(), {==})), 1),
    Some{F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0))},
    matmul_first_point(Cases.matmul_pass(Cases.gradient(), {==}), 1, F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0)),
      2, F32.add(0.0, F32.mul(7.0, 3.0)), Cases.matmul_values()))

def matmul_back_witness() ->
  {TensorBackward.sole_point(Laws.second_of(Cases.matmul_pass(Cases.gradient(), {==})), 2) == back_product() : Maybe<&2, F32>}:
  stated(TensorBackward.sole_point(Laws.second_of(Cases.matmul_pass(Cases.gradient(), {==})), 2), Some{F32.add(0.0, F32.mul(7.0, 3.0))},
    matmul_second_point(Cases.matmul_pass(Cases.gradient(), {==}), 1, F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0)),
      2, F32.add(0.0, F32.mul(7.0, 3.0)), Cases.matmul_values()))
