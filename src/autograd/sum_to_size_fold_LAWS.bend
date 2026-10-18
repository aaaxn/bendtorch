import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce.bend as Reduce
import ./reduce_LAWS.bend as ReduceLaws
import ./reduce_fold_LAWS.bend as ReduceFold
import ./sum_to_size.bend as SumToSize

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

# What sum_to_size gives at target cell index for a contribution whose words,
# in index order, are `values`, and which has count words. When same is True
# (the target has the contribution's dimensions) nothing is reduced and the
# cell holds its own word; otherwise it holds ReduceLaws.fiber, the words that
# land on the cell added in index order from +0.0.
def summed(+same: Bool, +plan: Shape.BroadcastPlan, +count: Nat, +values: List<&2, F32>, +index: U32) -> Maybe<&2, F32>:
  match same:
    case True{}:
      ReduceLaws.word_at(values, U32.to_nat(index))
    case False{}:
      ReduceLaws.fiber(plan, values, index, count)

# What the transport form of sum_to_size_replayed asks of P: P holds for the
# returned contribution, replayed with any history `kept` that keeps every
# word, next to an output replayed with any history, with the target's
# dimensions, whose every cell holds `summed`.
def ReducedWith(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +fill: F32, +values: List<&2, F32>, +same: Bool,
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @kept_words: {Contracts.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>} ->
  @-kept_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, kept)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @+output: Shape.Shape ->
  @dimensions: {Shape.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} ->
  @+output_fill: F32 ->
  @+output_ops: List<&2, StorageLaws.Op> ->
  @values_held: (@+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
    {Some{Replayed.value(output, output_fill, output_ops, index)} ==
      summed(same, plan, Shape.logical_numel(Shape.dimensions(shape)), values, index) : Maybe<&2, F32>}) ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(output, output_fill, output_ops)) ==
    Shape.logical_numel(Shape.dimensions(output)) : Nat} ->
    P((Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}},
      Done{Core.Packed{output, Core.TTensor{Replayed.storage(output, output_fill, output_ops), output_fits}}}))

# sum_to_size of a replayed contribution is replayed, in the transport form
# broadcast_to_replayed uses: when the target broadcasts to exactly the
# contribution's dimensions and `values` lists the contribution's words, then
# for every P, if P holds for every result that hands back the contribution
# with the same words and gives an output with the target's dimensions whose
# cells hold `summed`, P holds for the result. The output is the clone when
# nothing is reduced and reduce_to's output otherwise; both are replayed
# tensors, so a later family law applies to either half.
law sum_to_size_replayed:
  for +target: Shape.Shape
  for +shape: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +values: List<&2, F32>
  for words: {Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>}
  for -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type
  for h: ReducedWith(target, shape, plan, fill, values, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), P)
  P(SumToSize.sum_to_size(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target))
