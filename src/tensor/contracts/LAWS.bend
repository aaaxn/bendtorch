import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../elementwise.bend as Elementwise
import ../from-values-contract.bend as FromValues
import ./replayed.bend as Replayed
import ./broadcast.bend as Broadcast
import ./pointwise-records.bend as Records

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
