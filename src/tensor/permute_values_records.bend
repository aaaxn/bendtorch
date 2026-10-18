import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./axis_spec.bend as Axis
import ./transpose_spec.bend as Transpose
import ./observation.bend as Values
import ./transform_observation.bend as Observation
import ../storage/LAWS.bend as StorageLaws
import ./lifecycle.bend as Lifecycle
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./contracts/replayed.bend as Replayed
import ./contracts/LAWS.bend as Contracts

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

# The words of a permuted copy in index order: word k is the source's word at
# `permute_source(input, axes, k)`.
def permuted_words(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +count: Nat) -> List<&2, F32>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      List.append(&2, F32, permuted_words(input, fill, ops, axes, previous),
        [Replayed.value(input, fill, ops, U32.from_nat(Values.permute_source_index(input, axes, previous)))])

# What the transport form of permute_replayed and transpose_replayed asks of
# P: P holds for the returned source, replayed with any history that keeps its
# words, next to the output, of the plan's shape, replayed over +0.0 with any
# history whose words are `permuted_words`.
def PermutedWith(+input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +axes: List<&2, U32>, +plan: Layout.Plan<input, axes>,
  -P: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @kept_words: {Contracts.words(input, fill, kept, Shape.logical_numel(Shape.dimensions(input))) ==
    Contracts.words(input, fill, ops, Shape.logical_numel(Shape.dimensions(input))) : List<&2, F32>} ->
  @-kept_fits: {Lifecycle.logical_length(Replayed.storage(input, fill, kept)) == Shape.logical_numel(Shape.dimensions(input)) : Nat} ->
  @+output_ops: List<&2, StorageLaws.Op> ->
  @moved: {Contracts.words(Layout.output(input, axes, plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) ==
    permuted_words(input, fill, ops, axes, Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan)))) : List<&2, F32>} ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(Layout.output(input, axes, plan), 0.0, output_ops)) ==
    Shape.logical_numel(Shape.dimensions(Layout.output(input, axes, plan))) : Nat} ->
    P((Core.Packed{input, Core.TTensor{Replayed.storage(input, fill, kept), kept_fits}},
      Done{Core.Packed{Layout.output(input, axes, plan),
        Core.TTensor{Replayed.storage(Layout.output(input, axes, plan), 0.0, output_ops), output_fits}}}))
