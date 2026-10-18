import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./allocation_spec.bend as Allocation
import ./observation.bend as Observation

type FromValuesContract<-shape: Shape.Shape, -values: List<&2, F32>> is Type:
  FromValuesContract{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.from_values(shape, values)) == True{} : Bool},
    dimensions: {Observation.dimensions(shape, Core.from_values(shape, values)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    count: {Allocation.tensor_length(shape, Core.from_values(shape, values)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    word_count: {Observation.count(shape, Core.from_values(shape, values)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.point(shape, index, Core.from_values(shape, values)) == List.get(&2, F32, values, U32.to_nat(index)) : Maybe<&2, F32>}}
