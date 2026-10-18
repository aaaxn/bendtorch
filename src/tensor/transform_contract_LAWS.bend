import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./layout.bend as Layout
import ./transpose_spec.bend as Transpose
import ./transform_observation.bend as Observation
import ./transform_records.bend as Records

law reshape_contract:
  for +input: Shape.Shape
  for +fill: F32
  for +target: Shape.Shape
  for same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}
  Records.ReshapeContract<input, fill, target>

law reshape_write:
  for +input: Shape.Shape
  for +fill: F32
  for +target: Shape.Shape
  for +index: U32
  for +value: F32
  for inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}
  for same: {Shape.logical_numel(Shape.dimensions(input)) == Shape.logical_numel(Shape.dimensions(target)) : Nat}
  Records.ReshapeWrite<input, fill, target, index, value>

law reshape_count_mismatch:
  for +input: Shape.Shape
  for +fill: F32
  for +target: Shape.Shape
  for different: {Nat.is_eq(Shape.logical_numel(Shape.dimensions(input)), Shape.logical_numel(Shape.dimensions(target))) == False{} : Bool}
  {Observation.reshape_code(input, Shape.dimensions(target), Core.new(input, fill)) == Some{15} : Maybe<&2, U32>}

law permute_rank_mismatch:
  for +input: Shape.Shape
  for +fill: F32
  for +axes: List<&2, U32>
  for wrong: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == False{} : Bool}
  Records.RetainedFailure<input, fill, Observation.Permute{axes}, 12>

law permute_axis_out_of_bounds:
  for +input: Shape.Shape
  for +fill: F32
  for +axes: List<&2, U32>
  for ranked: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool}
  for outside: {Layout.bounded(axes, Shape.rank(input)) == False{} : Bool}
  Records.RetainedFailure<input, fill, Observation.Permute{axes}, 13>

law permute_duplicate_axis:
  for +input: Shape.Shape
  for +fill: F32
  for +axes: List<&2, U32>
  for ranked: {Nat.is_eq(Shape.rank_dimensions(axes), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool}
  for bounded: {Layout.bounded(axes, Shape.rank(input)) == True{} : Bool}
  for duplicate: {Layout.unique(axes) == False{} : Bool}
  Records.RetainedFailure<input, fill, Observation.Permute{axes}, 14>

law transpose_out_of_bounds:
  for +input: Shape.Shape
  for +fill: F32
  for +a: U32
  for +b: U32
  for outside: {Transpose.valid(input, a, b) == False{} : Bool}
  Records.RetainedFailure<input, fill, Observation.Transpose{a, b}, 13>
