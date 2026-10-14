import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./observation.bend as Observation

type CloneContract<-shape: Shape.Shape, -fill: F32> is Type:
  CloneContract{
    first_dimensions: {Observation.clone_first_dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    second_dimensions: {Observation.clone_second_dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    first_count: {Observation.clone_first_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    second_count: {Observation.clone_second_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    first_word_count: {Observation.clone_first_count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    second_word_count: {Observation.clone_second_count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    first_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.clone_first_value(shape, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>},
    second_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.clone_second_value(shape, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type CloneWrite<-shape: Shape.Shape, -fill: F32, -index: U32, -value: F32> is Type:
  CloneWrite{
    written: {Observation.clone_written_first(shape, index, value, index, Core.new(shape, fill)) == Some{value} : Maybe<&2, F32>},
    frame: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      {Observation.clone_written_first(shape, index, value, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>},
    second: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.clone_written_second(shape, index, value, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type MetadataContract<-shape: Shape.Shape, -fill: F32> is Type:
  MetadataContract{
    length: {Observation.length_value(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    retained_dimensions: {Observation.length_owner_dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.length_owner_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    retained_word_count: {Observation.length_owner_count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    shape_dimensions: {Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    word_count: {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.length_owner_value(shape, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>},
    shape_retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.shape_owner_value(shape, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>},
    count_retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.count_owner_value(shape, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}
