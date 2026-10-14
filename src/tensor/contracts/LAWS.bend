import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../elementwise.bend as Elementwise
import ../from-values-contract.bend as FromValues
import ../reduction.bend as Reduction
import ../matmul.bend as Matrix
import ../dot-fold.bend as DotFold
import ./replayed.bend as Replayed
import ./broadcast.bend as Broadcast
import ./pointwise-records.bend as Records
import ./sum.bend as Sum
import ./dot.bend as Dot
import ./fold-records.bend as FoldRecords

# Replayed tensors. `Core.new(shape, fill)` is the replayed tensor with an
# empty history, `Core.from_values` the one that writes each value once in
# order, and a read on either returns the replayed word.

law new_replayed:
  for +shape: Shape.Shape
  for +fill: F32
  for -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type
  for h: @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, Nil{})) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Done{Core.TTensor{Replayed.storage(shape, fill, Nil{}), fits}})
  P(Core.new(shape, fill))

law from_values_replayed:
  for +shape: Shape.Shape
  for +values: List<&2, F32>
  for counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type
  for h: @-fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, FromValues.writes(values, 0n))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Done{Core.TTensor{Replayed.storage(shape, 0.0, FromValues.writes(values, 0n)), fits}})
  P(Core.from_values(shape, values))

law replayed_read:
  for +shape: Shape.Shape
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}

# Pointwise family, broadcast instance: for every operation, broadcast plan
# and pair of replayed inputs, the output has the plan's output dimensions,
# holds `Elementwise.scalar(operation, left[ia(i)], right[ib(i)])` at every
# in-bounds index i, and both returned inputs keep every word.
law broadcast_pointwise:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), left_fill, left_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), right_fill, right_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}
  Records.PointwiseContract<Shape.output_shape(plan), Shape.left_shape(plan), Shape.right_shape(plan),
    index => Broadcast.cell(operation, plan, left_fill, left_ops, right_fill, right_ops, index),
    other => Replayed.value(Shape.left_shape(plan), left_fill, left_ops, other),
    other => Replayed.value(Shape.right_shape(plan), right_fill, right_ops, other),
    Broadcast.binary(operation, plan,
      Core.Packed{Shape.left_shape(plan), Core.TTensor{Replayed.storage(Shape.left_shape(plan), left_fill, left_ops), left_fits}},
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), right_fill, right_ops), right_fits}})>

def words(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, F32>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, F32, words(shape, fill, ops, previous), [Replayed.value(shape, fill, ops, U32.from_nat(previous))])

# Ordered-fold family, sum instance: the sum of a replayed input succeeds with
# `Reduction.sequential_sum` of all its words in index order, starting at
# +0.0, and the returned input keeps every word.
law sum_ordered:
  for +shape: Shape.Shape
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  FoldRecords.FoldContract<shape, Reduction.sequential_sum(words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), 0.0),
    other => Replayed.value(shape, fill, ops, other),
    Sum.sum(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>

# The first `count` factor pairs of output cell (row, column) of a row-major
# product of a rows x inner and an inner x columns matrix, in increasing k:
# left[row, k] and right[k, column].
def factors(+row: Nat, +column: Nat, +inner: Nat, +columns: Nat,
  +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, DotFold.Factors>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, DotFold.Factors,
        factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        [DotFold.Factors{Replayed.value(left, left_fill, left_ops, U32.from_nat(Nat.add(Nat.mul(row, inner), previous))),
          Replayed.value(right, right_fill, right_ops, U32.from_nat(Nat.add(Nat.mul(previous, columns), column)))}])

# Ordered-fold family, dot instance: for every in-bounds output cell of a
# product of a rows x inner and an inner x columns replayed matrix, the dot
# product succeeds with `DotFold.dot_fold` of the cell's factor pairs in
# increasing k, starting at +0.0, and both returned inputs keep every word.
law dot_ordered:
  for +rows: Nat
  for +inner: Nat
  for +columns: Nat
  for +row: Nat
  for row_below: {Nat.is_lt(row, rows) == True{} : Bool}
  for +column: Nat
  for column_below: {Nat.is_lt(column, columns) == True{} : Bool}
  for +left: Shape.Shape
  for left_extents: {Shape.extents(left) == [rows, inner] : List<&2, Nat>}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right: Shape.Shape
  for right_extents: {Shape.extents(right) == [inner, columns] : List<&2, Nat>}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  FoldRecords.PairFoldContract<left, right,
    DotFold.dot_fold(factors(row, column, inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, inner), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Dot.dot(Dot.Cell{row, column, inner, columns},
      Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>

def matrix_rows(-left: Shape.Shape, -right: Shape.Shape, plan: Matrix.Plan<left, right>) -> Nat:
  Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  U32.to_nat(rows)

def matrix_inner(-left: Shape.Shape, -right: Shape.Shape, plan: Matrix.Plan<left, right>) -> Nat:
  Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  U32.to_nat(inner)

def matrix_columns(-left: Shape.Shape, -right: Shape.Shape, plan: Matrix.Plan<left, right>) -> Nat:
  Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  U32.to_nat(columns)

# Ordered-fold family, matmul: for every plan of a product of two replayed
# matrices, matmul succeeds with a rows x columns output whose cell
# (row, column) holds `DotFold.dot_fold` of the cell's factor pairs in
# increasing k, starting at +0.0, the value `dot_ordered` states for
# `Dot.dot`, and both returned inputs keep every word.
law matmul_ordered:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +plan: Matrix.Plan<left, right>
  for planned: {Matrix.plan(left, right) == Done{plan} : Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>}
  for +left_fill: F32
  for +left_ops: List<&2, StorageLaws.Op>
  for -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat}
  for +right_fill: F32
  for +right_ops: List<&2, StorageLaws.Op>
  for -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}
  FoldRecords.MatmulContract<left, right, matrix_rows(left, right, plan), matrix_columns(left, right, plan),
    row => column => DotFold.dot_fold(factors(row, column, matrix_inner(left, right, plan), matrix_columns(left, right, plan),
      left, left_fill, left_ops, right, right_fill, right_ops, matrix_inner(left, right, plan)), 0.0),
    other => Replayed.value(left, left_fill, left_ops, other),
    other => Replayed.value(right, right_fill, right_ops, other),
    Matrix.matmul(Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>
