import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./transform-observation.bend as Observation

type RetainedFailure<-input: Shape.Shape, -fill: F32, -request: Observation.Request, -code: U32> is Type:
  RetainedFailure{
    outcome: {Observation.failure_code(input, request, Core.new(input, fill)) == Some{code} : Maybe<&2, U32>},
    retained_dimensions: {Observation.kept_dimensions(input, request, Core.new(input, fill)) == Some{Shape.dimensions(input)} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.kept_length(input, request, Core.new(input, fill)) == Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
    retained_word_count: {Observation.kept_count(input, request, Core.new(input, fill)) == Some{Shape.count(input)} : Maybe<&2, U32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Observation.kept_value(input, request, other, Core.new(input, fill)) == Some{fill} : Maybe<&2, F32>}}

type ReshapeContract<-input: Shape.Shape, -fill: F32, -target: Shape.Shape> is Type:
  ReshapeContract{
    outcome: {Observation.reshape_code(input, Shape.dimensions(target), Core.new(input, fill)) == Some{0} : Maybe<&2, U32>},
    dimensions: {Observation.reshape_dimensions(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
    count: {Observation.reshape_length(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.logical_numel(Shape.dimensions(target))} : Maybe<&2, Nat>},
    word_count: {Observation.reshape_count(input, Shape.dimensions(target), Core.new(input, fill)) == Some{Shape.count(target)} : Maybe<&2, U32>},
    value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      {Observation.reshape_value(input, Shape.dimensions(target), other, Core.new(input, fill)) == Some{fill} : Maybe<&2, F32>}}

type ReshapeWrite<-input: Shape.Shape, -fill: F32, -target: Shape.Shape, -index: U32, -value: F32> is Type:
  ReshapeWrite{
    written: {Observation.reshape_written(input, index, value, Shape.dimensions(target), index, Core.new(input, fill)) == Some{value} : Maybe<&2, F32>},
    frame: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      {Observation.reshape_written(input, index, value, Shape.dimensions(target), other, Core.new(input, fill)) == Some{fill} : Maybe<&2, F32>}}
