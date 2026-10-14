import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise-totality.bend as Totality
import ../tensor/projection-LAWS.bend as Projection
import ./reduce.bend as Reduce

def owner(pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Core.Dynamic:
  (input, result) = pair
  input

def succeeded(pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Bool:
  (input, result) = pair
  Result.is_done(&2, &1, Reduce.Error, Core.Dynamic, result)

def word_at(values: List<&2, F32>, +position: Nat) -> Maybe<&2, F32>:
  match values position:
    case Nil{} _:
      None{}
    case value <> rest 0n:
      Some{value}
    case value <> rest 1n+next:
      word_at(rest, next)

def add_word(total: Maybe<&2, F32>, value: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match total value:
    case None{} _:
      None{}
    case Some{sum} None{}:
      None{}
    case Some{sum} Some{word}:
      Some{F32.add(sum, word)}

def matched(total: Maybe<&2, F32>, value: Maybe<&2, F32>, same: Bool) -> Maybe<&2, F32>:
  match same:
    case False{}:
      total
    case True{}:
      add_word(total, value)

# One contribution position: if its projection lands on the requested target
# position, add the source word at the position's source offset.
def contribute(+values: List<&2, F32>, +target_index: U32, total: Maybe<&2, F32>, offsets: Maybe<&1, U32 & U32>) -> Maybe<&2, F32>:
  match offsets:
    case None{}:
      None{}
    case Some{pair}:
      (target_offset, source_offset) = pair
      matched(total, word_at(values, U32.to_nat(source_offset)), Nat.is_eq(U32.to_nat(target_offset), U32.to_nat(target_index)))

# The specified accumulation: start at +0.0 and visit contribution positions
# 0, 1, ..., count - 1 in ascending order; each position whose independent
# broadcast projection (Projection.offset_spec, with the target as the left
# operand) lands on target_index adds its word with one F32 add.
def fiber(+plan: Shape.BroadcastPlan, +values: List<&2, F32>, +target_index: U32, +count: Nat) -> Maybe<&2, F32>:
  match count:
    case 0n:
      Some{0.0}
    case 1n+position:
      contribute(values, target_index, fiber(plan, values, target_index, position), Projection.offset_spec(plan, position))

def planned_fiber(+values: List<&2, F32>, +target_index: U32, +count: Nat,
  plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Maybe<&2, F32>:
  match plan:
    case Fail{error}:
      None{}
    case Done{+plan}:
      fiber(plan, values, target_index, count)

# values lists the contribution's words: the checked observation at every
# U32 position is the word at that position, so values has exactly the
# contribution's logical length.
def Listed(-input: Core.Dynamic, +values: List<&2, F32>) -> Type:
  @+source: U32 -> {Core.checked_observe_dynamic(input, source) == word_at(values, U32.to_nat(source)) : Maybe<&2, F32>}

def MappingWith(-input: Core.Dynamic, +target: Shape.Shape, plan: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Type:
  (retained, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      {Totality.dimensions(output) == Shape.dimensions(target) : List<&2, U32>} &
        (@+values: List<&2, F32> -> @-listed: Listed(input, values) ->
          @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
            {Core.checked_observe_dynamic(output, index) ==
              planned_fiber(values, index, Shape.logical_numel(Totality.dimensions(input)), plan) : Maybe<&2, F32>})

def Mapping(input: Core.Dynamic, +target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Type:
  MappingWith(input, target, Shape.broadcast_checked(target, Totality.dynamic_shape(input)), pair)

def ZeroValues(+target: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Type:
  (retained, result) = pair
  match result:
    case Fail{error}:
      Unit
    case Done{output}:
      @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
        {Core.checked_observe_dynamic(output, index) == Some{0.0} : Maybe<&2, F32>}

type Retained<-input: Core.Dynamic, -pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Data:
  Retained{-kept: Totality.SameSource(input, owner(pair))}

type MappingModel<-input: Core.Dynamic, -target: Shape.Shape,
  -pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Data:
  MappingModel{-mapping: Mapping(input, target, pair)}

type Success<-pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Data:
  Success{-successful: {succeeded(pair) == True{} : Bool}}

type Zeros<-target: Shape.Shape, -pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>> is Data:
  Zeros{-zeros: ZeroValues(target, pair)}

# The contribution owner comes back with the same dimensions and observations.
law reduce_retains_input:
  for input: Core.Dynamic
  for +target: Shape.Shape
  Retained<input, Reduce.reduce_to(input, target)>

# On success the output has the target's dimensions, and for every list of
# words that lists the contribution, each in-bounds target position holds the
# ordered fiber sum of those words.
law reduce_mapping:
  for input: Core.Dynamic
  for +target: Shape.Shape
  MappingModel<input, target, Reduce.reduce_to(input, target)>

# A target whose right-aligned broadcast with the contribution shape is the
# contribution shape itself always succeeds.
law reduce_valid_success:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for +alignment: Shape.Broadcast<Shape.extents(target), Shape.extents(Totality.dynamic_shape(input)), Shape.extents(Totality.dynamic_shape(input))>
  Success<Reduce.reduce_to(input, target)>

# Reducing an empty contribution gives zeros at every target position, not an
# empty tensor: for example [0, 3] reduced to [1, 3] is three zeros.
law reduce_empty_zeros:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for empty: {Shape.logical_numel(Totality.dimensions(input)) == 0n : Nat}
  Zeros<target, Reduce.reduce_to(input, target)>

# When the S1 planner rejects the target and contribution shapes, reduce_to
# fails with exactly that S1 error and returns the contribution owner. For
# example [2] to [0] fails with IncompatibleBroadcast instead of returning an
# empty tensor.
law reduce_shape_error:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for error: Shape.Error
  for rejected: {Shape.broadcast_checked(target, Totality.dynamic_shape(input)) == Fail{error} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  {Reduce.reduce_to(input, target) == (input, Fail{Reduce.TensorError{Core.ShapeError{error}}}) : Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}

# When the planned broadcast of the target is not the contribution's shape
# (for example a [2, 3] target for a [3] contribution), reduce_to fails with
# NotReducible and returns the contribution owner.
law reduce_not_reducible:
  for input: Core.Dynamic
  for +target: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(target, Totality.dynamic_shape(input)) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for differ: {Shape.dimensions(Shape.output_shape(plan)) != Totality.dimensions(input) : List<&2, U32>}
  {Reduce.reduce_to(input, target) == (input, Fail{Reduce.NotReducible{}}) : Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>}
