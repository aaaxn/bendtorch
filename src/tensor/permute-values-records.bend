import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./axis-spec.bend as Axis
import ./transpose-spec.bend as Transpose
import ./observation.bend as Values
import ./transform-observation.bend as Observation

type PermuteContract<-input: Shape.Shape, -values: List<&2, F32>, -axes: List<&2, U32>> is Type:
  PermuteContract{
    outcome: {Observation.failure_code(input, Observation.Permute{axes}, Core.from_values(input, values)) == Some{0} : Maybe<&2, U32>},
    dimensions: {Observation.applied_dimensions(input, Observation.Permute{axes}, Core.from_values(input, values)) ==
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} : Maybe<&2, List<&2, U32>>},
    count: {Observation.applied_length(input, Observation.Permute{axes}, Core.from_values(input, values)) ==
      Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
    word_count: {Observation.applied_count(input, Observation.Permute{axes}, Core.from_values(input, values)) == Some{Shape.count(input)} : Maybe<&2, U32>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Observation.applied_value(input, Observation.Permute{axes}, index, Core.from_values(input, values)) ==
        Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Observation.kept_value(input, Observation.Permute{axes}, other, Core.from_values(input, values)) ==
        List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>}}

type TransposeContract<-input: Shape.Shape, -values: List<&2, F32>, -a: U32, -b: U32> is Type:
  TransposeContract{
    outcome: {Observation.failure_code(input, Observation.Transpose{a, b}, Core.from_values(input, values)) == Some{0} : Maybe<&2, U32>},
    dimensions: {Observation.applied_dimensions(input, Observation.Transpose{a, b}, Core.from_values(input, values)) ==
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))))} : Maybe<&2, List<&2, U32>>},
    count: {Observation.applied_length(input, Observation.Transpose{a, b}, Core.from_values(input, values)) ==
      Some{Shape.logical_numel(Shape.dimensions(input))} : Maybe<&2, Nat>},
    word_count: {Observation.applied_count(input, Observation.Transpose{a, b}, Core.from_values(input, values)) == Some{Shape.count(input)} : Maybe<&2, U32>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Observation.applied_value(input, Observation.Transpose{a, b}, index, Core.from_values(input, values)) ==
        Values.listed(values, Values.permute_source(input, Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)), index)) : Maybe<&2, F32>},
    retained_value: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Observation.kept_value(input, Observation.Transpose{a, b}, other, Core.from_values(input, values)) ==
        List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>}}
