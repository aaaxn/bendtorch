import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/elementwise_totality.bend as Totality
import ../tensor/projection_LAWS.bend as Projection
import ./broadcast.bend as Broadcast

def owner(pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Core.Dynamic:
  (input, result) = pair
  input

def succeeded(pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Bool:
  (input, result) = pair
  Result.is_done(&2, &1, Broadcast.Error, Core.Dynamic, result)

# The input value that the specification assigns to one output position: the
# input offset of its right-aligned broadcast projection.
def source_value(input: Core.Dynamic, offsets: Maybe<&1, U32 & U32>) -> Maybe<&2, F32>:
  match offsets:
    case None{}:
      None{}
    case Some{pair}:
      (input_offset, target_offset) = pair
      Core.checked_observe_dynamic(input, input_offset)

def planned_source(input: Core.Dynamic, +index: U32, plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Maybe<&2, F32>:
  match plan:
    case Fail{error}:
      None{}
    case Done{+plan}:
      source_value(input, Projection.offset_spec(plan, U32.to_nat(index)))

def MappingWith(-input: Core.Dynamic, +target: Shape.Shape, plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Type:
  (retained, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      {Totality.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} &
        (@+index: U32 -> @inside: Shape.LT(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) ->
          {Core.observe_dynamic(output, index) == planned_source(input, index, plan) : Maybe<&2, F32>})

def Mapping(input: Core.Dynamic, +target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Type:
  MappingWith(input, target, Shape.broadcast_checked(Totality.dynamic_shape(input), target), pair)

type Retained<-input: Core.Dynamic, -pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>> is Data:
  Retained{-kept: Totality.SameSource(input, owner(pair))}

type MappingModel<-input: Core.Dynamic, -target: Shape.Shape, -pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>> is Data:
  MappingModel{-mapping: Mapping(input, target, pair)}

type Success<-pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>> is Data:
  Success{-successful: {succeeded(pair) == True{} : Bool}}

law keep_left_scalar:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.KeepLeft{}, left, right) == left : F32}

# The input owner comes back with the same dimensions and observations.
law broadcast_retains_input:
  for input: Core.Dynamic
  for +target: Shape.Shape
  Retained<input, Broadcast.broadcast_to(input, target)>

# On success the output has the target's dimensions, and each in-bounds output
# position observes the input at the offset that the independent broadcast
# projection specification assigns to it.
law broadcast_mapping:
  for input: Core.Dynamic
  for +target: Shape.Shape
  MappingModel<input, target, Broadcast.broadcast_to(input, target)>

# A request whose right-aligned broadcast with the input shape is the target
# itself always succeeds.
law broadcast_valid_success:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for +alignment: Shape.Broadcast<Shape.extents(Totality.dynamic_shape(input)), Shape.extents(target), Shape.extents(target)>
  Success<Broadcast.broadcast_to(input, target)>

# When the broadcast planner rejects the input and target shapes, broadcast_to
# fails with exactly that planner error and returns the input owner. For example [2] to
# [0] fails with IncompatibleBroadcast instead of returning an empty tensor.
law broadcast_shape_error:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for error: Shape.Error
  for rejected: {Shape.broadcast_checked(Totality.dynamic_shape(input), target) == Fail{error} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  {Broadcast.broadcast_to(input, target) == (input, Fail{Broadcast.TensorError{Core.ShapeError{error}}}) : Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>}

# When the planned broadcast is not the target itself (for example [3] to
# [1]), broadcast_to fails with NotBroadcastable and returns the input owner.
law broadcast_not_target:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(Totality.dynamic_shape(input), target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for differ: {Shape.dimensions(Shape.output_shape(plan)) != Shape.dimensions(target) : List<&2, U32>}
  {Broadcast.broadcast_to(input, target) == (input, Fail{Broadcast.NotBroadcastable{}}) : Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>}
