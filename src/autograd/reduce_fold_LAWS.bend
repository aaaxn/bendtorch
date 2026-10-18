import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/fold.bend as Fold
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce.bend as Reduce
import ./reduce_LAWS.bend as ReduceLaws
import ./reduce_fold.bend as ReduceFold

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

def output_dimensions(result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Maybe<&2, List<&2, U32>>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{Core.Packed{shape, tensor}}:
      Some{Shape.dimensions(shape)}

def output_point(result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      Core.dynamic_read_value(Core.read_dynamic(output, index))

def input_point(result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(input, index))

# What reduce_to promises, stated only through the result: it succeeds with
# the target's dimensions, every in-bounds target cell holds `expected`, and
# the returned contribution still holds every word.
type ReduceToContract<-target: Shape.Shape, -input: Shape.Shape, -expected: U32 -> Maybe<&2, F32>, -word: U32 -> F32,
  -result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Type:
  ReduceToContract{
    successful: {ReduceLaws.succeeded(result) == True{} : Bool},
    dimensions: {output_dimensions(result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      {output_point(result, index) == expected(index) : Maybe<&2, F32>},
    kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {input_point(result, other) == Some{word(other)} : Maybe<&2, F32>}}

# reduce_to through the ordered-fold family: when the target broadcasts to
# exactly the contribution's dimensions, reducing a replayed contribution
# succeeds with the target's dimensions, every target cell holds
# `ReduceLaws.fiber` over the contribution's words in index order, the value
# `reduce_fold_fiber` states for `ReduceFold.cell`, and the returned
# contribution keeps every word.
law reduce_to_fiber:
  for +target: Shape.Shape
  for +shape: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  ReduceToContract<target, shape,
    index => ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
      index, Shape.logical_numel(Shape.dimensions(shape))),
    other => Replayed.value(shape, fill, ops, other),
    Reduce.reduce_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>
