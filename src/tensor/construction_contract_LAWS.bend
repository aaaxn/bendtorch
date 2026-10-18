import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./allocation_spec.bend as Allocation
import ./observation.bend as Observation
import ./construction_metadata.bend as Metadata

type NewReads<-shape: Shape.Shape, -fill: F32, -index: U32> is Type:
  NewReads{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool},
    dimensions: {Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    count: {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    word_count: {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    value: {Observation.point(shape, index, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>},
    retained_dimensions: {Observation.after_dimensions(shape, index, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.after_length(shape, index, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    retained_word_count: {Observation.after_count(shape, index, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.after_value(shape, index, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

law new_metadata:
  for +shape: Shape.Shape
  for +fill: F32
  Metadata.NewMetadata<shape, fill>

law new_reads:
  for +shape: Shape.Shape
  for +fill: F32
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  NewReads<shape, fill, index>
