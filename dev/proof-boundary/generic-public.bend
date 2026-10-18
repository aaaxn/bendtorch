import Base
import ../../src/shape/core.bend as Shape
import ../../src/tensor/core.bend as Core
import ../../src/tensor/allocation_spec.bend as Allocation
import ./observations.bend as Observation
import ./generic-read.bend as Reading
import ./generic-metadata.bend as Metadata
import ./retention.bend as Retention

type PublicRead<-shape: Shape.Shape, -fill: F32, -index: U32> is Type:
  PublicRead{read: Reading.Read<shape, fill, index>, retained: Retention.Retained<shape, fill, index>}

def produce(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> PublicRead<shape, fill, index>:
  PublicRead{Reading.produce(shape, fill, index, inside), Retention.produce(shape, fill, index)}

type Claims<-shape: Shape.Shape, -fill: F32, -index: U32> is Type:
  Claims{
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

def expose(-shape: Shape.Shape, -fill: F32, -index: U32, evidence: PublicRead<shape, fill, index>) -> Claims<shape, fill, index>:
  PublicRead{read, retained} = evidence
  Reading.Read{construction, value} = read
  Metadata.Construction{successful, dimensions, count, word_count} = construction
  Retention.Retained{retained_count, retained_dimensions, retained_word_count, retained_value} = retained
  Claims{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value}

def consume(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Claims<shape, fill, index>:
  expose(shape, fill, index, produce(shape, fill, index, inside))
