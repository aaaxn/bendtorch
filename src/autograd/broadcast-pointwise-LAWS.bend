import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
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

# The reads broadcast_to makes of its input: index left_map(plan, k) at step k.
def reads(+plan: Shape.BroadcastPlan, +count: Nat) -> List<&2, StorageLaws.Op>:
  Pointwise.reads(~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map, plan, count)

# The writes of the pointwise family's broadcast instance with KeepLeft, the
# input on the left and the +0.0 tensor of the target on the right: cell k
# holds scalar(KeepLeft, input[left_map(plan, k)], 0.0).
def writes(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +target: Shape.Shape,
  +count: Nat) -> List<&2, StorageLaws.Op>:
  Pointwise.writes(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~PointwiseBroadcast.left_map,
    ~PointwiseBroadcast.right_map, count, Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{})

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

# The broadcast_to result is replayed: for every P, if P holds for the input
# with the history ops ++ reads and the output over +0.0 with the history
# writes, for any length proofs, then P holds for the result. So a later law
# about a replayed contribution, such as reduce_to_fiber, applies to the
# output.
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
  for h: @-input_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill,
      List.append(&2, StorageLaws.Op, ops, reads(plan, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))))) ==
      Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-output_fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(plan), 0.0,
      writes(plan, shape, fill, ops, target, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))))) ==
      Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    P((Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill,
      List.append(&2, StorageLaws.Op, ops, reads(plan, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))))), input_fits}},
      Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0,
        writes(plan, shape, fill, ops, target, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))))), output_fits}}}))
  P(Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target))
