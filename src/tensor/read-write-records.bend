import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./observation.bend as Observation

type ReadContract<-shape: Shape.Shape, -fill: F32, -index: U32, -code: U32> is Type:
  ReadContract{
    outcome: {Observation.read_code(shape, index, Core.new(shape, fill)) == Some{code} : Maybe<&2, U32>},
    retained_dimensions: {Observation.after_dimensions(shape, index, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.after_length(shape, index, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    retained_word_count: {Observation.after_count(shape, index, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.after_value(shape, index, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type ReadInside<-shape: Shape.Shape, -fill: F32, -index: U32> is Type:
  ReadInside{
    read: ReadContract<shape, fill, index, 0>,
    value: {Observation.point(shape, index, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type WriteContract<-shape: Shape.Shape, -fill: F32, -index: U32, -value: F32, -code: U32> is Type:
  WriteContract{
    outcome: {Observation.write_code(shape, index, value, Core.new(shape, fill)) == Some{code} : Maybe<&2, U32>},
    retained_dimensions: {Observation.written_dimensions(shape, index, value, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.written_length(shape, index, value, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    retained_word_count: {Observation.written_count(shape, index, value, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>}}

type WriteInside<-shape: Shape.Shape, -fill: F32, -index: U32, -value: F32> is Type:
  WriteInside{
    write: WriteContract<shape, fill, index, value, 0>,
    written: {Observation.written_value(shape, index, value, index, Core.new(shape, fill)) == Some{value} : Maybe<&2, F32>},
    frame: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      {Observation.written_value(shape, index, value, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type WriteOutside<-shape: Shape.Shape, -fill: F32, -index: U32, -value: F32> is Type:
  WriteOutside{
    write: WriteContract<shape, fill, index, value, 8>,
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.written_value(shape, index, value, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

type CoordinateInside<-shape: Shape.Shape, -fill: F32, -coordinates: List<&2, U32>, -index: U32, -value: F32> is Type:
  CoordinateInside{
    read: {Observation.coordinate_value(shape, index, value, coordinates, Core.new(shape, fill)) == Some{value} : Maybe<&2, F32>},
    written: {Observation.coordinate_written(shape, coordinates, value, index, Core.new(shape, fill)) == Some{value} : Maybe<&2, F32>},
    frame: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      @different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool} ->
      {Observation.coordinate_written(shape, coordinates, value, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}
