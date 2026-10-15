import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/elementwise.bend as Elementwise
import ../../../src/tensor/reduction.bend as Reduction
import ../../../src/tensor/from-values-contract.bend as FromValues
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/autograd/tensor-backward.bend as Backward
import ../../../src/autograd/tensor-backward-LAWS.bend as TensorBackward
import ../../../src/autograd/tensor-vjp-LAWS.bend as Laws
import ../../../src/autograd/broadcast-pointwise-PROOF.bend as BroadcastProof
import ../../../src/autograd/reduce-PROOF.bend as ReduceProof
import ../../../src/autograd/reduce-fold-PROOF.bend as FoldProof
import ../../../src/autograd/reduce-to-fold-PROOF.bend as ReduceToProof
import ../../../src/autograd/sum-to-size-fold-PROOF.bend as SumProof
import ../../../src/autograd/tensor-backward-PROOF.bend as TensorBackwardProof
import ../../../src/autograd/tensor-vjp-PROOF.bend as Proof

def one_shape() -> Shape.Shape:
  Shape.SShape{[1], 1n, {==}, Unit{}, Unit{}}

def two_shape() -> Shape.Shape:
  Shape.SShape{[2], 2n, {==}, Unit{}, Unit{}}

def plan_of(r: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  done: {Result.is_done(&2, &2, Shape.Error, Shape.BroadcastPlan, r) == True{} : Bool}) -> Shape.BroadcastPlan:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Shape.BroadcastPlan, Arith.clash(done))

def two_two() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), two_shape()), {==})

def one_two() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), two_shape()), {==})

def two_one() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), one_shape()), {==})

def two_scalar() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), Reduction.scalar_shape()), {==})

# The plans that reduce a product or -g back to an operand: the product's
# output shape is the plan's own term, so each plan is taken from it.
def two_by_one() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), Shape.output_shape(two_one())), {==})

def one_by_two() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), Shape.output_shape(two_two())), {==})

def one_by_negated() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), Shape.output_shape(two_scalar())), {==})

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def gradient() -> List<&2, F32>:
  [3.0, 5.0]

def left_values() -> List<&2, F32>:
  [2.0, 7.0]

def right_values() -> List<&2, F32>:
  [10.0]

# The VJP of node 2 = a + b, a - b or a * b with a = node 0 of shape [2] and
# b = node 1 of shape [1], for the gradient [3.0, 5.0] of shape [2].
def added(+values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(two_shape(), 0.0, writes(values))) == 2n : Nat}) -> Backward.Pass:
  Backward.vjp(Backward.Add{0n, 1n, two_shape(), one_shape()}, Backward.Unsaved{}, input(two_shape(), values, fits))

def subtracted(+values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(two_shape(), 0.0, writes(values))) == 2n : Nat}) -> Backward.Pass:
  Backward.vjp(Backward.Sub{0n, 1n, two_shape(), one_shape()}, Backward.Unsaved{}, input(two_shape(), values, fits))

def multiplied(+values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(two_shape(), 0.0, writes(values))) == 2n : Nat}) -> Backward.Pass:
  Backward.vjp(Backward.Mul{0n, 1n, two_shape(), one_shape()},
    Backward.Operands{input(two_shape(), left_values(), {==}), input(one_shape(), right_values(), {==})},
    input(two_shape(), values, fits))

type PairCase<-pass: Backward.Pass, -first: U32, -first_word: Maybe<&2, F32>, -second: U32, -second_word: Maybe<&2, F32>> is Data:
  PairCase{
    first_target: {TensorBackward.sole_target(Laws.first_of(pass)) == Some{0n} : Maybe<&2, Nat>},
    first_dimensions: {TensorBackward.sole_dimensions(Laws.first_of(pass)) == Some{[2]} : Maybe<&2, List<&2, U32>>},
    first_value: {TensorBackward.sole_point(Laws.first_of(pass), first) == first_word : Maybe<&2, F32>},
    second_target: {TensorBackward.sole_target(Laws.second_of(pass)) == Some{1n} : Maybe<&2, Nat>},
    second_dimensions: {TensorBackward.sole_dimensions(Laws.second_of(pass)) == Some{[1]} : Maybe<&2, List<&2, U32>>},
    second_value: {TensorBackward.sole_point(Laws.second_of(pass), second) == second_word : Maybe<&2, F32>}}

def open_pair(-pass: Backward.Pass, -left_cell: U32 -> Maybe<&2, F32>, -right_cell: U32 -> Maybe<&2, F32>,
  evidence: Laws.PairContract<0n, two_shape(), left_cell, 1n, one_shape(), right_cell, pass>,
  +first: U32, first_inside: {Nat.is_lt(U32.to_nat(first), 2n) == True{} : Bool},
  +second: U32, second_inside: {Nat.is_lt(U32.to_nat(second), 1n) == True{} : Bool}) ->
  PairCase<pass, first, left_cell(first), second, right_cell(second)>:
  Laws.PairContract{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  PairCase{first_target, first_dimensions, first_value(first, first_inside), second_target, second_dimensions,
    second_value(second, second_inside)}

def add_evidence() -> Laws.PairContract<0n, two_shape(),
  index => Laws.reduced(two_shape(), two_shape(), two_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
  1n, one_shape(), index => Laws.reduced(one_shape(), two_shape(), one_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
  added(gradient(), {==})>:
  Laws.tensor_add_vjp_values(0n, 1n, two_shape(), one_shape(), two_shape(), two_two(), {==}, {==}, one_two(), {==}, {==},
    Backward.Unsaved{}, 0.0, writes(gradient()), {==})

# d(a+b)/da keeps g: the operand has g's dimensions, so cell 1 is word 1
# itself; d(a+b)/db adds both words of g in order from +0.0.
def add_values() -> PairCase<added(gradient(), {==}), 1, Some{5.0}, 0, Some{F32.add(F32.add(0.0, 3.0), 5.0)}>:
  open_pair(added(gradient(), {==}), index => Laws.reduced(two_shape(), two_shape(), two_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
    index => Laws.reduced(one_shape(), two_shape(), one_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
    add_evidence(), 1, {==}, 0, {==})

def negatives() -> List<&2, F32>:
  Laws.cells(Elementwise.NegateLeft{}, two_scalar(), two_shape(), 0.0, writes(gradient()), Reduction.scalar_shape(), 0.0, Nil{}, 2n)

def sub_evidence() -> Laws.PairContract<0n, two_shape(),
  index => Laws.reduced(two_shape(), two_shape(), two_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
  1n, one_shape(), index => Laws.reduced(one_shape(), Shape.output_shape(two_scalar()), one_by_negated(), negatives(), index),
  subtracted(gradient(), {==})>:
  Laws.tensor_sub_vjp_values(0n, 1n, two_shape(), one_shape(), two_shape(), two_two(), {==}, {==}, two_scalar(), {==},
    one_by_negated(), {==}, {==}, Backward.Unsaved{}, 0.0, writes(gradient()), {==})

# d(a-b)/db negates each word of g before it adds them.
def sub_values() -> PairCase<subtracted(gradient(), {==}), 0, Some{3.0}, 0, Some{F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))}>:
  open_pair(subtracted(gradient(), {==}),
    index => Laws.reduced(two_shape(), two_shape(), two_two(), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), index),
    index => Laws.reduced(one_shape(), Shape.output_shape(two_scalar()), one_by_negated(), negatives(), index),
    sub_evidence(), 0, {==}, 0, {==})

def by_right() -> List<&2, F32>:
  Laws.cells(Elementwise.Mul{}, two_one(), two_shape(), 0.0, writes(gradient()), one_shape(), 0.0, writes(right_values()), 2n)

def by_left() -> List<&2, F32>:
  Laws.cells(Elementwise.Mul{}, two_two(), two_shape(), 0.0, writes(gradient()), two_shape(), 0.0, writes(left_values()), 2n)

def mul_evidence() -> Laws.PairContract<0n, two_shape(),
  index => Laws.reduced(two_shape(), Shape.output_shape(two_one()), two_by_one(), by_right(), index),
  1n, one_shape(), index => Laws.reduced(one_shape(), Shape.output_shape(two_two()), one_by_two(), by_left(), index),
  multiplied(gradient(), {==})>:
  Laws.tensor_mul_vjp_values(0n, 1n, two_shape(), one_shape(), two_shape(), two_one(), {==}, two_by_one(), {==}, {==},
    two_two(), {==}, one_by_two(), {==}, {==}, 0.0, writes(gradient()), {==}, 0.0, writes(left_values()), {==},
    0.0, writes(right_values()), {==})

# d(a*b)/da is g * b: cell 1 is 5.0 * 10.0; d(a*b)/db is g * a summed:
# 3.0 * 2.0 + 5.0 * 7.0 from +0.0.
def mul_values() -> PairCase<multiplied(gradient(), {==}), 1, Some{F32.mul(5.0, 10.0)}, 0,
  Some{F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))}>:
  open_pair(multiplied(gradient(), {==}),
    index => Laws.reduced(two_shape(), Shape.output_shape(two_one()), two_by_one(), by_right(), index),
    index => Laws.reduced(one_shape(), Shape.output_shape(two_two()), one_by_two(), by_left(), index),
    mul_evidence(), 1, {==}, 0, {==})
