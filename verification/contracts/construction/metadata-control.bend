import Base
import ../../../src/tensor/construction_metadata.bend as Metadata
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/allocation_spec.bend as Allocation
import ../../../src/tensor/observation.bend as Observation

def expose(-shape: Shape.Shape, -fill: F32, evidence: Metadata.NewMetadata<shape, fill>) ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool} &
  ({Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>} &
    ({Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>} &
      {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>})):
  Metadata.NewMetadata{successful, dimensions, count, word_count} = evidence
  (successful, (dimensions, (count, word_count)))

def witness(+shape: Shape.Shape, +fill: F32) ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool} &
  ({Observation.dimensions(shape, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>} &
    ({Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>} &
      {Observation.count(shape, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>})):
  expose(shape, fill, Metadata.construction(shape, fill))
