import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/storage/main.bend as Storage
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/reduction.bend as Reduction
import ../../../src/tensor/matmul-LAWS.bend as MatmulLaws
import ../../../src/tensor/from-values-contract.bend as FromValues
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/PROOF.bend as ContractsProof
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/fold.bend as Fold
import ../../../src/tensor/contracts/fold-records.bend as Records
import ../../../src/tensor/contracts/sum.bend as Sum
import ../../../src/tensor/contracts/dot.bend as Dot

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def tall_shape() -> Shape.Shape:
  Shape.SShape{[3, 2], 6n, {==}, Unit{}, Unit{}}

def empty_grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 0], 0n, {==}, Unit{}, Unit{}}

def empty_wide_shape() -> Shape.Shape:
  Shape.SShape{[0, 3], 0n, {==}, Unit{}, Unit{}}

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def sum_result(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  Sum.sum(input(shape, values, fits))

def dot_result(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  Dot.dot(Dot.Cell{row, column, inner, columns}, input(left, left_values, left_fits), input(right, right_values, right_fits))

def SumValues(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) -> Type:
  Records.FoldContract<shape, Reduction.sequential_sum(Contracts.words(shape, 0.0, writes(values), Shape.logical_numel(Shape.dimensions(shape))), 0.0),
    other => Replayed.value(shape, 0.0, writes(values), other),
    sum_result(shape, values, fits)>

def DotValues(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) -> Type:
  Records.PairFoldContract<left, right,
    MatmulLaws.dot_fold(Contracts.factors(row, column, inner, columns, left, 0.0, writes(left_values), right, 0.0, writes(right_values), inner), 0.0),
    other => Replayed.value(left, 0.0, writes(left_values), other),
    other => Replayed.value(right, 0.0, writes(right_values), other),
    dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits)>

type SumCase<-result: Core.Dynamic & Result<&2, &1, Core.Error, F32>, -total: F32, -index: U32, -word: F32> is Data:
  SumCase{
    value_exact: {Fold.value(result) == Some{total} : Maybe<&2, F32>},
    kept_exact: {Fold.input_point(result, index) == Some{word} : Maybe<&2, F32>}}

type DotCase<-result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>, -total: F32,
  -left_index: U32, -left_value: F32, -right_index: U32, -right_value: F32> is Data:
  DotCase{
    value_exact: {Fold.pair_value(result) == Some{total} : Maybe<&2, F32>},
    left_exact: {Fold.left_point(result, left_index) == Some{left_value} : Maybe<&2, F32>},
    right_exact: {Fold.right_point(result, right_index) == Some{right_value} : Maybe<&2, F32>}}

def open_sum(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: SumValues(shape, values, fits),
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  SumCase<sum_result(shape, values, fits),
    Reduction.sequential_sum(Contracts.words(shape, 0.0, writes(values), Shape.logical_numel(Shape.dimensions(shape))), 0.0),
    index, Replayed.value(shape, 0.0, writes(values), index)>:
  Records.FoldContract{value, kept} = evidence
  SumCase{value, kept(index, inside)}

def expose_sum(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  SumCase<sum_result(shape, values, fits),
    Reduction.sequential_sum(Contracts.words(shape, 0.0, writes(values), Shape.logical_numel(Shape.dimensions(shape))), 0.0),
    index, Replayed.value(shape, 0.0, writes(values), index)>:
  open_sum(shape, values, fits, Contracts.sum_ordered(shape, 0.0, writes(values), fits), index, inside)

def open_empty_sum(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: SumValues(shape, values, fits)) ->
  {Fold.value(sum_result(shape, values, fits)) ==
    Some{Reduction.sequential_sum(Contracts.words(shape, 0.0, writes(values), Shape.logical_numel(Shape.dimensions(shape))), 0.0)} : Maybe<&2, F32>}:
  Records.FoldContract{value, kept} = evidence
  value

def open_dot(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  evidence: DotValues(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits),
  +left_index: U32, left_inside: {Nat.is_lt(U32.to_nat(left_index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_index: U32, right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}) ->
  DotCase<dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits),
    MatmulLaws.dot_fold(Contracts.factors(row, column, inner, columns, left, 0.0, writes(left_values), right, 0.0, writes(right_values), inner), 0.0),
    left_index, Replayed.value(left, 0.0, writes(left_values), left_index),
    right_index, Replayed.value(right, 0.0, writes(right_values), right_index)>:
  Records.PairFoldContract{value, left_kept, right_kept} = evidence
  DotCase{value, left_kept(left_index, left_inside), right_kept(right_index, right_inside)}

def expose_dot(+rows: Nat, +inner: Nat, +columns: Nat, +row: Nat, row_below: {Nat.is_lt(row, rows) == True{} : Bool},
  +column: Nat, column_below: {Nat.is_lt(column, columns) == True{} : Bool},
  +left: Shape.Shape, left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>}, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>}, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  +left_index: U32, left_inside: {Nat.is_lt(U32.to_nat(left_index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_index: U32, right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool}) ->
  DotCase<dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits),
    MatmulLaws.dot_fold(Contracts.factors(row, column, inner, columns, left, 0.0, writes(left_values), right, 0.0, writes(right_values), inner), 0.0),
    left_index, Replayed.value(left, 0.0, writes(left_values), left_index),
    right_index, Replayed.value(right, 0.0, writes(right_values), right_index)>:
  open_dot(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits,
    Contracts.dot_ordered(rows, inner, columns, row, row_below, column, column_below,
      left, left_extents, 0.0, writes(left_values), left_fits, right, right_extents, 0.0, writes(right_values), right_fits),
    left_index, left_inside, right_index, right_inside)

def open_empty_dot(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  evidence: DotValues(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits)) ->
  {Fold.pair_value(dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits)) ==
    Some{MatmulLaws.dot_fold(Contracts.factors(row, column, inner, columns, left, 0.0, writes(left_values), right, 0.0, writes(right_values), inner), 0.0)} :
    Maybe<&2, F32>}:
  Records.PairFoldContract{value, left_kept, right_kept} = evidence
  value

def three_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0]

def grid_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def tall_values() -> List<&2, F32>:
  [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

def sum_three() -> SumCase<sum_result(three_shape(), three_values(), {==}),
  F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0), 1, 2.0>:
  expose_sum(three_shape(), three_values(), {==}, 1, {==})

def sum_grid() -> SumCase<sum_result(grid_shape(), grid_values(), {==}),
  F32.add(F32.add(F32.add(F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0), 4.0), 5.0), 6.0), 4, 5.0>:
  expose_sum(grid_shape(), grid_values(), {==}, 4, {==})

def sum_scalar() -> SumCase<sum_result(scalar_shape(), [7.0], {==}), F32.add(0.0, 7.0), 0, 7.0>:
  expose_sum(scalar_shape(), [7.0], {==}, 0, {==})

def sum_empty() -> {Fold.value(sum_result(empty_grid_shape(), [], {==})) == Some{0.0} : Maybe<&2, F32>}:
  open_empty_sum(empty_grid_shape(), [], {==}, Contracts.sum_ordered(empty_grid_shape(), 0.0, writes([]), {==}))

def dot_lower_left() -> DotCase<dot_result(1n, 0n, 3n, 2n, grid_shape(), grid_values(), {==}, tall_shape(), tall_values(), {==}),
  F32.add(F32.add(F32.add(0.0, F32.mul(4.0, 10.0)), F32.mul(5.0, 30.0)), F32.mul(6.0, 50.0)), 5, 6.0, 2, 30.0>:
  expose_dot(2n, 3n, 2n, 1n, {==}, 0n, {==}, grid_shape(), {==}, grid_values(), {==}, tall_shape(), {==}, tall_values(), {==}, 5, {==}, 2, {==})

def dot_upper_right() -> DotCase<dot_result(0n, 1n, 3n, 2n, grid_shape(), grid_values(), {==}, tall_shape(), tall_values(), {==}),
  F32.add(F32.add(F32.add(0.0, F32.mul(1.0, 20.0)), F32.mul(2.0, 40.0)), F32.mul(3.0, 60.0)), 0, 1.0, 5, 60.0>:
  expose_dot(2n, 3n, 2n, 0n, {==}, 1n, {==}, grid_shape(), {==}, grid_values(), {==}, tall_shape(), {==}, tall_values(), {==}, 0, {==}, 5, {==})

def dot_empty() -> {Fold.pair_value(dot_result(1n, 2n, 0n, 3n, empty_grid_shape(), [], {==}, empty_wide_shape(), [], {==})) == Some{0.0} : Maybe<&2, F32>}:
  open_empty_dot(1n, 2n, 0n, 3n, empty_grid_shape(), [], {==}, empty_wide_shape(), [], {==},
    Contracts.dot_ordered(2n, 0n, 3n, 1n, {==}, 2n, {==}, empty_grid_shape(), {==}, 0.0, writes([]), {==},
      empty_wide_shape(), {==}, 0.0, writes([]), {==}))

# Failure. The family laws require every index in bounds, so these consumers
# run the generic loops directly with an index map that leaves the input:
# `doubled` reads index 4 of a three-word input at step 2, after two
# successful steps.
def doubled(plan: Unit, index: Nat) -> Nat:
  Nat.mul(2n, index)

def failure(result: Core.Dynamic & Result<&2, &1, Core.Error, F32>) -> Maybe<&2, Core.Error>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      Some{error}
    case Done{word}:
      None{}

def pair_failure(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>) -> Maybe<&2, Core.Error>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      Some{error}
    case Done{word}:
      None{}

def out_of_bounds() -> Maybe<&2, Core.Error>:
  Some{Core.StorageError{Storage.OutOfBounds{}}}

def failed_sum() -> Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  Fold.run(~Unit, ~Sum.add, ~Unit, ~doubled, ~Sum.every, Unit{}, Unit{}, 3n, input(three_shape(), three_values(), {==}))

def failed_left() -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  Fold.pair_run(~Unit, ~Dot.multiply_add, ~Unit, ~doubled, ~Sum.position, Unit{}, Unit{}, 3n,
    input(three_shape(), three_values(), {==}), input(three_shape(), [10.0, 20.0, 30.0], {==}))

def failed_right() -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  Fold.pair_run(~Unit, ~Dot.multiply_add, ~Unit, ~Sum.position, ~doubled, Unit{}, Unit{}, 3n,
    input(three_shape(), three_values(), {==}), input(three_shape(), [10.0, 20.0, 30.0], {==}))

type FailCase<-error: Maybe<&2, Core.Error>, -first: Maybe<&2, F32>, -last: Maybe<&2, F32>> is Data:
  FailCase{
    error_exact: {error == out_of_bounds() : Maybe<&2, Core.Error>},
    first_kept: {first == Some{1.0} : Maybe<&2, F32>},
    last_kept: {last == Some{3.0} : Maybe<&2, F32>}}

type PairFailCase<-error: Maybe<&2, Core.Error>, -left_first: Maybe<&2, F32>, -left_last: Maybe<&2, F32>,
  -right_first: Maybe<&2, F32>, -right_last: Maybe<&2, F32>> is Data:
  PairFailCase{
    error_exact: {error == out_of_bounds() : Maybe<&2, Core.Error>},
    left_first_kept: {left_first == Some{1.0} : Maybe<&2, F32>},
    left_last_kept: {left_last == Some{3.0} : Maybe<&2, F32>},
    right_first_kept: {right_first == Some{10.0} : Maybe<&2, F32>},
    right_last_kept: {right_last == Some{30.0} : Maybe<&2, F32>}}

def sum_fails() -> FailCase<failure(failed_sum()), Fold.input_point(failed_sum(), 0), Fold.input_point(failed_sum(), 2)>:
  FailCase{{==}, {==}, {==}}

def pair_fails_left() -> PairFailCase<pair_failure(failed_left()), Fold.left_point(failed_left(), 0), Fold.left_point(failed_left(), 2),
  Fold.right_point(failed_left(), 0), Fold.right_point(failed_left(), 2)>:
  PairFailCase{{==}, {==}, {==}, {==}, {==}}

def pair_fails_right() -> PairFailCase<pair_failure(failed_right()), Fold.left_point(failed_right(), 0), Fold.left_point(failed_right(), 2),
  Fold.right_point(failed_right(), 0), Fold.right_point(failed_right(), 2)>:
  PairFailCase{{==}, {==}, {==}, {==}, {==}}
