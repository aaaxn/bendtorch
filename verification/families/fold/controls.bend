import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/reduction.bend as Reduction
import ../../../src/tensor/matmul.bend as Matrix
import ../../../src/tensor/dot-fold.bend as DotFold
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/PROOF.bend as ContractsProof
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/fold.bend as Fold
import ../../../src/tensor/contracts/fold-records.bend as Records
import ./consumers.bend as Cases

def expected_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)}

def reversed_sum() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, 3.0), 2.0), 1.0)}

def expected_start() -> Maybe<&2, F32>:
  Some{F32.add(0.0, 7.0)}

def unstarted() -> Maybe<&2, F32>:
  Some{7.0}

def expected_dot() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, F32.mul(4.0, 10.0)), F32.mul(5.0, 30.0)), F32.mul(6.0, 50.0))}

def untransposed_dot() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, F32.mul(4.0, 10.0)), F32.mul(5.0, 20.0)), F32.mul(6.0, 30.0))}

def expected_kept() -> Maybe<&2, F32>:
  Some{6.0}

def swapped_kept() -> Maybe<&2, F32>:
  Some{60.0}

def total(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, Cases.writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Cases.SumValues(shape, values, fits)) ->
  {Fold.value(Cases.sum_result(shape, values, fits)) ==
    Some{Reduction.sequential_sum(Contracts.words(shape, 0.0, Cases.writes(values), Shape.logical_numel(Shape.dimensions(shape))), 0.0)} : Maybe<&2, F32>}:
  Records.FoldContract{value, kept} = evidence
  value

def sum_witness() ->
  {Fold.value(Cases.sum_result(Cases.three_shape(), Cases.three_values(), {==})) == expected_sum() : Maybe<&2, F32>}:
  total(Cases.three_shape(), Cases.three_values(), {==},
    Contracts.sum_ordered(Cases.three_shape(), 0.0, Cases.writes(Cases.three_values()), {==}))

def start_witness() ->
  {Fold.value(Cases.sum_result(Cases.scalar_shape(), [7.0], {==})) == expected_start() : Maybe<&2, F32>}:
  total(Cases.scalar_shape(), [7.0], {==}, Contracts.sum_ordered(Cases.scalar_shape(), 0.0, Cases.writes([7.0]), {==}))

def dot_total(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, Cases.writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, Cases.writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  evidence: Cases.DotValues(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits)) ->
  {Fold.pair_value(Cases.dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits)) ==
    Some{DotFold.dot_fold(Contracts.factors(row, column, inner, columns, left, 0.0, Cases.writes(left_values), right, 0.0, Cases.writes(right_values), inner), 0.0)} :
    Maybe<&2, F32>}:
  Records.PairFoldContract{value, left_kept, right_kept} = evidence
  value

def left_word(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, Cases.writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right: Shape.Shape, +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, Cases.writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  evidence: Cases.DotValues(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits),
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool}) ->
  {Fold.left_point(Cases.dot_result(row, column, inner, columns, left, left_values, left_fits, right, right_values, right_fits), index) ==
    Some{Replayed.value(left, 0.0, Cases.writes(left_values), index)} : Maybe<&2, F32>}:
  Records.PairFoldContract{value, left_kept, right_kept} = evidence
  left_kept(index, inside)

def dot_witness() ->
  {Fold.pair_value(Cases.dot_result(1n, 0n, 3n, 2n, Cases.grid_shape(), Cases.grid_values(), {==},
    Cases.tall_shape(), Cases.tall_values(), {==})) == expected_dot() : Maybe<&2, F32>}:
  dot_total(1n, 0n, 3n, 2n, Cases.grid_shape(), Cases.grid_values(), {==}, Cases.tall_shape(), Cases.tall_values(), {==},
    Contracts.dot_ordered(2n, 3n, 2n, 1n, {==}, 0n, {==}, Cases.grid_shape(), {==}, 0.0, Cases.writes(Cases.grid_values()), {==},
      Cases.tall_shape(), {==}, 0.0, Cases.writes(Cases.tall_values()), {==}))

def kept_witness() ->
  {Fold.left_point(Cases.dot_result(1n, 0n, 3n, 2n, Cases.grid_shape(), Cases.grid_values(), {==},
    Cases.tall_shape(), Cases.tall_values(), {==}), 5) == expected_kept() : Maybe<&2, F32>}:
  left_word(1n, 0n, 3n, 2n, Cases.grid_shape(), Cases.grid_values(), {==}, Cases.tall_shape(), Cases.tall_values(), {==},
    Contracts.dot_ordered(2n, 3n, 2n, 1n, {==}, 0n, {==}, Cases.grid_shape(), {==}, 0.0, Cases.writes(Cases.grid_values()), {==},
      Cases.tall_shape(), {==}, 0.0, Cases.writes(Cases.tall_values()), {==}), 5, {==})

def matmul_total(+left: Shape.Shape, +right: Shape.Shape, +plan: Matrix.Plan<left, right>, +left_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, Cases.writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  +right_values: List<&2, F32>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, Cases.writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  evidence: Cases.MatmulValues(left, right, plan, left_values, left_fits, right_values, right_fits),
  +row: Nat, row_below: {Nat.is_lt(row, Contracts.matrix_rows(left, right, plan)) == True{} : Bool},
  +column: Nat, column_below: {Nat.is_lt(column, Contracts.matrix_columns(left, right, plan)) == True{} : Bool},
  -index: U32, at: {U32.from_nat(Nat.add(Nat.mul(row, Contracts.matrix_columns(left, right, plan)), column)) == index : U32}) ->
  {Records.product_point(Cases.product(left, left_values, left_fits, right, right_values, right_fits), index) ==
    Some{DotFold.dot_fold(Contracts.factors(row, column, Contracts.matrix_inner(left, right, plan),
      Contracts.matrix_columns(left, right, plan), left, 0.0, Cases.writes(left_values), right, 0.0, Cases.writes(right_values),
      Contracts.matrix_inner(left, right, plan)), 0.0)} : Maybe<&2, F32>}:
  Records.MatmulContract{successful, extents, value, left_kept, right_kept} = evidence
  %at : {Records.product_point(Cases.product(left, left_values, left_fits, right, right_values, right_fits), _) ==
    Some{DotFold.dot_fold(Contracts.factors(row, column, Contracts.matrix_inner(left, right, plan),
      Contracts.matrix_columns(left, right, plan), left, 0.0, Cases.writes(left_values), right, 0.0, Cases.writes(right_values),
      Contracts.matrix_inner(left, right, plan)), 0.0)} : Maybe<&2, F32>}
  value(row, row_below, column, column_below)

def matmul_witness() ->
  {Records.product_point(Cases.product(Cases.grid_shape(), Cases.grid_values(), {==},
    Cases.tall_shape(), Cases.tall_values(), {==}), 2) == expected_dot() : Maybe<&2, F32>}:
  matmul_total(Cases.grid_shape(), Cases.tall_shape(), Cases.grid_tall(), Cases.grid_values(), {==}, Cases.tall_values(), {==},
    Contracts.matmul_ordered(Cases.grid_shape(), Cases.tall_shape(), Cases.grid_tall(), {==}, 0.0, Cases.writes(Cases.grid_values()), {==},
      0.0, Cases.writes(Cases.tall_values()), {==}), 1n, {==}, 0n, {==}, 2, {==})
