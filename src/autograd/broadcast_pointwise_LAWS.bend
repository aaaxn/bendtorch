import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ./broadcast.bend as Broadcast

def succeeded(result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Bool:
  (input, outcome) = result
  Result.is_done(&2, &1, Broadcast.Error, Core.Dynamic, outcome)

def output_dimensions(result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Maybe<&2, List<&2, U32>>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{Core.Packed{shape, tensor}}:
      Some{Shape.dimensions(shape)}

def output_point(result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      Core.dynamic_read_value(Core.read_dynamic(output, index))

def input_point(result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(input, index))

# What broadcast_to promises, stated only through the result: it succeeds
# with the target's dimensions, every in-bounds output index i holds cell(i),
# and the returned input still holds every word.
type BroadcastToContract<-target: Shape.Shape, -input: Shape.Shape, -cell: Nat -> F32, -word: U32 -> F32,
  -result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>> is Type:
  BroadcastToContract{
    successful: {succeeded(result) == True{} : Bool},
    dimensions: {output_dimensions(result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      {output_point(result, index) == Some{cell(U32.to_nat(index))} : Maybe<&2, F32>},
    kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {input_point(result, other) == Some{word(other)} : Maybe<&2, F32>}}

# The words of broadcast_to's output in index order: word k is the input
# word at left_map(plan, k), its right-aligned broadcast projection.
def cells(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, F32>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, F32, cells(plan, shape, fill, ops, previous),
        [Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, previous)))])

# What the transport form of broadcast_to_replayed asks of P: P holds for the
# input, replayed with any history that keeps its words, next to an output of
# the plan's shape, replayed over +0.0 with any history whose words are
# `cells`.
def BroadcastWith(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -P: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @kept_words: {Contracts.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) ==
    Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) : List<&2, F32>} ->
  @-input_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, kept)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @+output_ops: List<&2, StorageLaws.Op> ->
  @values: {Contracts.words(Shape.output_shape(plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) ==
    cells(plan, shape, fill, ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : List<&2, F32>} ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(plan), 0.0, output_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    P((Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), input_fits}},
      Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0, output_ops), output_fits}}}))

# Pointwise family, broadcast_to instance: when the input shape broadcasts to
# exactly the target, broadcasting a replayed input succeeds with the target's
# dimensions, output index i holds the input word at left_map(plan, i), its
# right-aligned broadcast projection, and the returned input keeps every word.
law broadcast_to_pointwise:
  for +shape: Shape.Shape
  for +target: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  BroadcastToContract<target, shape,
    index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    other => Replayed.value(shape, fill, ops, other),
    Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>

# The broadcast_to result is replayed: for every P, if P holds for every
# result that hands back the input with its words and gives an output of the
# plan's shape whose words are `cells`, then P holds for broadcast_to. The
# output word at k is the input word at left_map(plan, k), whatever split of
# the output into parallel halves the build takes, and the output is a
# replayed tensor, so a later law about a replayed contribution, such as
# reduce_to_fiber, applies to it.
law broadcast_to_replayed:
  for +shape: Shape.Shape
  for +target: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for -P: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic> -> Type
  for h: BroadcastWith(plan, shape, fill, ops, P)
  P(Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target))
