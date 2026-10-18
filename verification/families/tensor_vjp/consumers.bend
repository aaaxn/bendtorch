import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/elementwise.bend as Elementwise
import ../../../src/tensor/reduction.bend as Reduction
import ../../../src/tensor/layout.bend as Layout
import ../../../src/tensor/matmul.bend as Matrix
import ../../../src/tensor/dot_fold.bend as DotFold
import ../../../src/tensor/from_values_contract.bend as FromValues
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/autograd/tensor_backward.bend as Backward
import ../../../src/autograd/tensor_backward_LAWS.bend as TensorBackward
import ../../../src/autograd/tensor_vjp_LAWS.bend as Laws
import ../../../src/autograd/broadcast_pointwise_PROOF.bend as BroadcastProof
import ../../../src/autograd/reduce_PROOF.bend as ReduceProof
import ../../../src/autograd/reduce_fold_PROOF.bend as FoldProof
import ../../../src/autograd/reduce_to_fold_PROOF.bend as ReduceToProof
import ../../../src/autograd/sum_to_size_fold_PROOF.bend as SumProof
import ../../../src/autograd/tensor_backward_PROOF.bend as TensorBackwardProof
import ../../../src/autograd/tensor_vjp_PROOF.bend as Proof

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

# Matmul: node 2 = a @ b with a = node 0 of shape [1, 2] holding [2.0, 7.0]
# and b = node 1 of shape [2, 2] holding [10.0, 20.0, 30.0, 50.0], for the
# gradient [3.0, 5.0] of shape [1, 2]. b is square, so reading it without the
# transpose gives other words.
def row_shape() -> Shape.Shape:
  Shape.SShape{[1, 2], 2n, {==}, Unit{}, Unit{}}

def square_shape() -> Shape.Shape:
  Shape.SShape{[2, 2], 4n, {==}, Unit{}, Unit{}}

def square_values() -> List<&2, F32>:
  [10.0, 20.0, 30.0, 50.0]

def layout_of(+shape: Shape.Shape, r: Result<&2, &2, Layout.Error, Layout.Plan<shape, Laws.swapped(shape)>>,
  done: {Result.is_done(&2, &2, Layout.Error, Layout.Plan<shape, Laws.swapped(shape)>, r) == True{} : Bool}) ->
  Layout.Plan<shape, Laws.swapped(shape)>:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Layout.Plan<shape, Laws.swapped(shape)>, Arith.clash(done))

def product_of(+left: Shape.Shape, +right: Shape.Shape, r: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>,
  done: {Result.is_done(&2, &2, Matrix.Error, Matrix.Plan<left, right>, r) == True{} : Bool}) -> Matrix.Plan<left, right>:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Matrix.Plan<left, right>, Arith.clash(done))

def square_flip() -> Layout.Plan<square_shape(), Laws.swapped(square_shape())>:
  layout_of(square_shape(), Layout.permutation(square_shape(), Laws.swapped(square_shape())), {==})

def row_flip() -> Layout.Plan<row_shape(), Laws.swapped(row_shape())>:
  layout_of(row_shape(), Layout.permutation(row_shape(), Laws.swapped(row_shape())), {==})

def front() -> Matrix.Plan<row_shape(), Laws.transposed(square_shape(), square_flip())>:
  product_of(row_shape(), Laws.transposed(square_shape(), square_flip()),
    Matrix.plan(row_shape(), Laws.transposed(square_shape(), square_flip())), {==})

def back() -> Matrix.Plan<Laws.transposed(row_shape(), row_flip()), row_shape()>:
  product_of(Laws.transposed(row_shape(), row_flip()), row_shape(),
    Matrix.plan(Laws.transposed(row_shape(), row_flip()), row_shape()), {==})

def matmul_pass(+values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(row_shape(), 0.0, writes(values))) == 2n : Nat}) -> Backward.Pass:
  Backward.vjp(Backward.Matmul{0n, 1n, row_shape(), square_shape()},
    Backward.Operands{input(row_shape(), left_values(), {==}), input(square_shape(), square_values(), {==})},
    input(row_shape(), values, fits))

def front_cell(+row: Nat, +column: Nat) -> F32:
  DotFold.dot_fold(Laws.right_transposed_factors(row, column, 2n, 2n, row_shape(), 0.0, writes(gradient()),
    square_shape(), 0.0, writes(square_values()), Laws.swapped(square_shape()), 2n), 0.0)

def back_cell(+row: Nat, +column: Nat) -> F32:
  DotFold.dot_fold(Laws.left_transposed_factors(row, column, 1n, 2n, row_shape(), 0.0, writes(left_values()),
    Laws.swapped(row_shape()), row_shape(), 0.0, writes(gradient()), 1n), 0.0)

type MatmulCase<-pass: Backward.Pass, -first: U32, -first_word: F32, -second: U32, -second_word: F32> is Data:
  MatmulCase{
    first_target: {TensorBackward.sole_target(Laws.first_of(pass)) == Some{0n} : Maybe<&2, Nat>},
    first_dimensions: {TensorBackward.sole_dimensions(Laws.first_of(pass)) == Some{[1, 2]} : Maybe<&2, List<&2, U32>>},
    first_value: {TensorBackward.sole_point(Laws.first_of(pass), first) == Some{first_word} : Maybe<&2, F32>},
    second_target: {TensorBackward.sole_target(Laws.second_of(pass)) == Some{1n} : Maybe<&2, Nat>},
    second_dimensions: {TensorBackward.sole_dimensions(Laws.second_of(pass)) == Some{[2, 2]} : Maybe<&2, List<&2, U32>>},
    second_value: {TensorBackward.sole_point(Laws.second_of(pass), second) == Some{second_word} : Maybe<&2, F32>}}

def matmul_evidence() -> Laws.MatmulPairContract<0n, row_shape(), 1n, 2n, row => column => front_cell(row, column),
  1n, square_shape(), 2n, 2n, row => column => back_cell(row, column), matmul_pass(gradient(), {==})>:
  Laws.tensor_matmul_vjp_values(0n, 1n, row_shape(), square_shape(), row_shape(), {==}, square_flip(), {==}, front(), {==}, {==},
    {==}, row_flip(), {==}, back(), {==}, {==}, 0.0, writes(gradient()), {==}, 0.0, writes(left_values()), {==},
    0.0, writes(square_values()), {==})

def open_matmul(-pass: Backward.Pass, -left_cell: Nat -> Nat -> F32, -right_cell: Nat -> Nat -> F32,
  evidence: Laws.MatmulPairContract<0n, row_shape(), 1n, 2n, left_cell, 1n, square_shape(), 2n, 2n, right_cell, pass>,
  +first_row: Nat, first_row_below: {Nat.is_lt(first_row, 1n) == True{} : Bool},
  +first_column: Nat, first_column_below: {Nat.is_lt(first_column, 2n) == True{} : Bool},
  +second_row: Nat, second_row_below: {Nat.is_lt(second_row, 2n) == True{} : Bool},
  +second_column: Nat, second_column_below: {Nat.is_lt(second_column, 2n) == True{} : Bool}) ->
  MatmulCase<pass, U32.from_nat(Nat.add(Nat.mul(first_row, 2n), first_column)), left_cell(first_row, first_column),
    U32.from_nat(Nat.add(Nat.mul(second_row, 2n), second_column)), right_cell(second_row, second_column)>:
  Laws.MatmulPairContract{first_target, first_dimensions, first_value, second_target, second_dimensions, second_value} = evidence
  MatmulCase{first_target, first_dimensions, first_value(first_row, first_row_below, first_column, first_column_below),
    second_target, second_dimensions, second_value(second_row, second_row_below, second_column, second_column_below)}

# d(a@b)/da is g @ b^T: cell (0, 1) folds g[0] * b[1, 0] and g[1] * b[1, 1]
# from +0.0, the second row of b; d(a@b)/db is a^T @ g: cell (1, 0) is
# a[1] * g[0] from +0.0.
def matmul_values() -> MatmulCase<matmul_pass(gradient(), {==}), 1, F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0)),
  2, F32.add(0.0, F32.mul(7.0, 3.0))>:
  open_matmul(matmul_pass(gradient(), {==}), row => column => front_cell(row, column), row => column => back_cell(row, column),
    matmul_evidence(), 0n, {==}, 1n, {==}, 1n, {==}, 0n, {==})

# MatmulNT, Linear's product: node 2 = a @ b^T with a = node 0 of shape
# [1, 2] holding [2.0, 7.0] and b = node 1 of shape [2, 2] holding
# [10.0, 20.0, 30.0, 50.0], PyTorch's [out, in], for the gradient [3.0, 5.0].
# d/da is g @ b, read in place: cell (0, 1) folds g[0] * b[0, 1] and
# g[1] * b[1, 1] from +0.0, the second column of b; d/db is g^T @ a: cell
# (1, 0) is g[1] * a[0] from +0.0.
def linear_back_of(+left: Shape.Shape, +right: Shape.Shape,
  r: Result<&2, &2, Matrix.Error, Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, left, right>>,
  done: {Result.is_done(&2, &2, Matrix.Error, Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, left, right>, r) == True{} : Bool}) ->
  Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, left, right>:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, left, right>, Arith.clash(done))

def linear_front() -> Matrix.Plan<row_shape(), square_shape()>:
  product_of(row_shape(), square_shape(), Matrix.plan(row_shape(), square_shape()), {==})

def linear_back() -> Matrix.Product<Matrix.Swapped{}, Matrix.Swapped{}, row_shape(), row_shape()>:
  linear_back_of(row_shape(), row_shape(), Matrix.product_plan(Matrix.Swapped{}, Matrix.Swapped{}, row_shape(), row_shape()), {==})

def linear_pass(+values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(row_shape(), 0.0, writes(values))) == 2n : Nat}) -> Backward.Pass:
  Backward.vjp(Backward.MatmulNT{0n, 1n, row_shape(), square_shape()},
    Backward.Operands{input(row_shape(), left_values(), {==}), input(square_shape(), square_values(), {==})},
    input(row_shape(), values, fits))

def linear_front_cell(+row: Nat, +column: Nat) -> F32:
  DotFold.dot_fold(Contracts.factors(row, column, 2n, 2n, row_shape(), 0.0, writes(gradient()),
    square_shape(), 0.0, writes(square_values()), 2n), 0.0)

def linear_back_cell(+row: Nat, +column: Nat) -> F32:
  DotFold.dot_fold(Contracts.product_factors(row, column, Matrix.Swapped{}, Matrix.Swapped{}, 1n, 2n, 2n,
    row_shape(), 0.0, writes(gradient()), row_shape(), 0.0, writes(left_values()), 1n), 0.0)

def linear_evidence() -> Laws.MatmulPairContract<0n, row_shape(), 1n, 2n, row => column => linear_front_cell(row, column),
  1n, square_shape(), 2n, 2n, row => column => linear_back_cell(row, column), linear_pass(gradient(), {==})>:
  Laws.tensor_matmul_nt_vjp_values(0n, 1n, row_shape(), square_shape(), row_shape(), linear_front(), {==}, {==},
    linear_back(), {==}, {==}, 0.0, writes(gradient()), {==}, 0.0, writes(left_values()), {==},
    0.0, writes(square_values()), {==})

def linear_values() -> MatmulCase<linear_pass(gradient(), {==}), 1, F32.add(F32.add(0.0, F32.mul(3.0, 20.0)), F32.mul(5.0, 50.0)),
  2, F32.add(0.0, F32.mul(5.0, 2.0))>:
  open_matmul(linear_pass(gradient(), {==}), row => column => linear_front_cell(row, column),
    row => column => linear_back_cell(row, column), linear_evidence(), 0n, {==}, 1n, {==}, 1n, {==}, 0n, {==})
