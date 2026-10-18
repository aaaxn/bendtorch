import Base
import ../../../src/tensor/transform_contract_LAWS.bend as Contract
import ../../../src/tensor/transform_records.bend as Records
import ../../../src/tensor/transform_contract_PROOF.bend as ContractProof
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/transform_observation.bend as Observation
import ./consumers.bend as Cases

def expected_fill() -> Maybe<&2, F32>:
  Some{7.0}

def expected_written() -> Maybe<&2, F32>:
  Some{9.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def reshaped_value(+other: U32, evidence: Records.ReshapeContract<Cases.grid_shape(), 7.0, Cases.tall_shape()>,
  inside: {Nat.is_lt(U32.to_nat(other), 6n) == True{} : Bool}) ->
  {Observation.reshape_value(Cases.grid_shape(), [3, 2], other, Core.new(Cases.grid_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.ReshapeContract{code, dimensions, count, word_count, value} = evidence
  value(other, inside)

def written_value(-index: U32, evidence: Records.ReshapeWrite<Cases.grid_shape(), 7.0, Cases.tall_shape(), index, 9.0>) ->
  {Observation.reshape_written(Cases.grid_shape(), index, 9.0, [3, 2], index, Core.new(Cases.grid_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  Records.ReshapeWrite{written, frame} = evidence
  written

def retained_value(-request: Observation.Request, -code: U32, +other: U32, evidence: Records.RetainedFailure<Cases.cube_shape(), 7.0, request, code>,
  inside: {Nat.is_lt(U32.to_nat(other), 24n) == True{} : Bool}) ->
  {Observation.kept_value(Cases.cube_shape(), request, other, Core.new(Cases.cube_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.RetainedFailure{outcome, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  retained_value(other, inside)

def transposed_value(+other: U32, evidence: Records.RetainedFailure<Cases.grid_shape(), 7.0, Observation.Transpose{0, 2}, 13>,
  inside: {Nat.is_lt(U32.to_nat(other), 6n) == True{} : Bool}) ->
  {Observation.kept_value(Cases.grid_shape(), Observation.Transpose{0, 2}, other, Core.new(Cases.grid_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.RetainedFailure{outcome, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  retained_value(other, inside)

def reshape_first_witness() ->
  {Observation.reshape_value(Cases.grid_shape(), [3, 2], 0, Core.new(Cases.grid_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  reshaped_value(0, Contract.reshape_contract(Cases.grid_shape(), 7.0, Cases.tall_shape(), {==}), {==})

def reshape_written_witness() ->
  {Observation.reshape_written(Cases.grid_shape(), 4, 9.0, [3, 2], 4, Core.new(Cases.grid_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  written_value(4, Contract.reshape_write(Cases.grid_shape(), 7.0, Cases.tall_shape(), 4, 9.0, {==}, {==}))

def permute_kept_witness() ->
  {Observation.kept_value(Cases.cube_shape(), Observation.Permute{[2, 0, 2]}, 0, Core.new(Cases.cube_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  retained_value(Observation.Permute{[2, 0, 2]}, 14, 0, Contract.permute_duplicate_axis(Cases.cube_shape(), 7.0, [2, 0, 2], {==}, {==}, {==}), {==})

def transpose_kept_witness() ->
  {Observation.kept_value(Cases.grid_shape(), Observation.Transpose{0, 2}, 0, Core.new(Cases.grid_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  transposed_value(0, Contract.transpose_out_of_bounds(Cases.grid_shape(), 7.0, 0, 2, {==}), {==})
