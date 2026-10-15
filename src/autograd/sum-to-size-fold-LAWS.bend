import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold-LAWS.bend as ReduceFold
import ./sum-to-size.bend as SumToSize

# sum_to_size through the reduce_to contract, nothing reduced: when the
# target has the contribution's dimensions, a replayed contribution comes back
# as its clone. The call succeeds with the target's dimensions, every cell
# holds the contribution's own word (-0.0 stays -0.0, as in PyTorch's
# autograd), and the returned contribution keeps every word.
law sum_to_size_clone:
  for +target: Shape.Shape
  for +shape: Shape.Shape
  for same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  ReduceFold.ReduceToContract<target, shape,
    index => Some{Replayed.value(shape, fill, ops, index)},
    other => Replayed.value(shape, fill, ops, other),
    SumToSize.sum_to_size(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>

# sum_to_size through the reduce_to contract, axes reduced: when the
# dimensions differ and the target broadcasts to exactly the contribution's
# dimensions, a replayed contribution gives the ReduceToContract of
# reduce_to_fiber: every target cell holds ReduceLaws.fiber over the
# contribution's words in index order from +0.0.
law sum_to_size_fiber:
  for +target: Shape.Shape
  for +shape: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for differ: {Shape.dimensions(shape) != Shape.dimensions(target) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  ReduceFold.ReduceToContract<target, shape,
    index => ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
      index, Shape.logical_numel(Shape.dimensions(shape))),
    other => Replayed.value(shape, fill, ops, other),
    SumToSize.sum_to_size(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>
