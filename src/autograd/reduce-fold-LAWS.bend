import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/fold.bend as Fold
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold.bend as ReduceFold

# What a reduce cell promises, stated only through the result: it gives
# `expected`, and the returned contribution still holds every word.
type FiberContract<-input: Shape.Shape, -expected: Maybe<&2, F32>, -word: U32 -> F32,
  -result: Core.Dynamic & Result<&2, &1, Core.Error, F32>> is Type:
  FiberContract{
    value: {Fold.value(result) == expected : Maybe<&2, F32>},
    kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Fold.input_point(result, other) == Some{word(other)} : Maybe<&2, F32>}}

# Ordered-fold family, reduce instance: for a plan whose output has the
# contribution's shape, every target index and every replayed contribution,
# the reduce cell gives `ReduceLaws.fiber` over the contribution's words in
# index order, and returns the contribution with every word kept.
law reduce_fold_fiber:
  for +plan: Shape.BroadcastPlan
  for same: {Shape.extents(Shape.output_shape(plan)) == Shape.extents(Shape.right_shape(plan)) : List<&2, Nat>}
  for +index: U32
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), fill, ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}
  FiberContract<Shape.right_shape(plan),
    ReduceLaws.fiber(plan,
      Contracts.words(Shape.right_shape(plan), fill, ops, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
      index, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
    other => Replayed.value(Shape.right_shape(plan), fill, ops, other),
    ReduceFold.cell(plan, index,
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), fill, ops), fits}})>
