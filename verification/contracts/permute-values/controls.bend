import Base
import ../../../src/tensor/permute-values-contract-LAWS.bend as Contract
import ../../../src/tensor/permute-values-records.bend as Records
import ../../../src/tensor/permute-values-contract-PROOF.bend as ContractProof
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/transpose-spec.bend as Transpose
import ../../../src/tensor/observation.bend as Values
import ../../../src/tensor/transform-observation.bend as Observation
import ./consumers.bend as Cases

def expected_rotated() -> Maybe<&2, F32>:
  Some{5.0}

def expected_transposed() -> Maybe<&2, F32>:
  Some{4.0}

def expected_kept() -> Maybe<&2, F32>:
  Some{3.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def permuted(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, +index: U32,
  evidence: Records.PermuteContract<input, values, axes>,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Observation.applied_value(input, Observation.Permute{axes}, index, Core.from_values(input, values)) ==
    Values.listed(values, Values.permute_source(input, axes, index)) : Maybe<&2, F32>}:
  Records.PermuteContract{outcome, shaped, count, word_count, value, retained_value} = evidence
  value(index, inside)

def transposed(+input: Shape.Shape, +values: List<&2, F32>, +a: U32, +b: U32, +index: U32,
  evidence: Records.TransposeContract<input, values, a, b>,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Observation.applied_value(input, Observation.Transpose{a, b}, index, Core.from_values(input, values)) ==
    Values.listed(values, Values.permute_source(input, Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)), index)) : Maybe<&2, F32>}:
  Records.TransposeContract{outcome, shaped, count, word_count, value, retained_value} = evidence
  value(index, inside)

def retained(+input: Shape.Shape, +values: List<&2, F32>, +a: U32, +b: U32, +other: U32,
  evidence: Records.TransposeContract<input, values, a, b>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Observation.kept_value(input, Observation.Transpose{a, b}, other, Core.from_values(input, values)) ==
    List.get(&2, F32, values, U32.to_nat(other)) : Maybe<&2, F32>}:
  Records.TransposeContract{outcome, shaped, count, word_count, value, retained_value} = evidence
  retained_value(other, inside)

def cube_rotate_witness() ->
  {Observation.applied_value(Cases.cube_shape(), Observation.Permute{Cases.rotation()}, 1, Core.from_values(Cases.cube_shape(), Cases.cube_values())) ==
    expected_rotated() : Maybe<&2, F32>}:
  permuted(Cases.cube_shape(), Cases.cube_values(), Cases.rotation(), 1,
    Contract.permute_contract(Cases.cube_shape(), Cases.cube_values(), {==}, Cases.rotation(), {==}), {==})

def grid_transpose_witness() ->
  {Observation.applied_value(Cases.grid_shape(), Observation.Transpose{0, 1}, 1, Core.from_values(Cases.grid_shape(), Cases.grid_values())) ==
    expected_transposed() : Maybe<&2, F32>}:
  transposed(Cases.grid_shape(), Cases.grid_values(), 0, 1, 1,
    Contract.transpose_contract(Cases.grid_shape(), Cases.grid_values(), {==}, 0, 1, {==}), {==})

def grid_kept_witness() ->
  {Observation.kept_value(Cases.grid_shape(), Observation.Transpose{0, 1}, 2, Core.from_values(Cases.grid_shape(), Cases.grid_values())) ==
    expected_kept() : Maybe<&2, F32>}:
  retained(Cases.grid_shape(), Cases.grid_values(), 0, 1, 2,
    Contract.transpose_contract(Cases.grid_shape(), Cases.grid_values(), {==}, 0, 1, {==}), {==})
