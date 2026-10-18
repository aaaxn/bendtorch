import Base
import ../../../src/tensor/clone_metadata_contract_LAWS.bend as Contract
import ../../../src/tensor/clone_metadata_records.bend as Records
import ../../../src/tensor/clone_metadata_contract_PROOF.bend as ContractProof
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/observation.bend as Observation
import ./consumers.bend as Cases

def expected_fill() -> Maybe<&2, F32>:
  Some{7.0}

def expected_written() -> Maybe<&2, F32>:
  Some{9.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def second_value(+other: U32, evidence: Records.CloneContract<Cases.three_shape(), 7.0>,
  inside: {Nat.is_lt(U32.to_nat(other), 3n) == True{} : Bool}) ->
  {Observation.clone_second_value(Cases.three_shape(), other, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.CloneContract{first_dimensions, second_dimensions, first_count, second_count, first_word_count, second_word_count, first_value, second_value} = evidence
  second_value(other, inside)

def written_value(-index: U32, evidence: Records.CloneWrite<Cases.three_shape(), 7.0, index, 9.0>) ->
  {Observation.clone_written_first(Cases.three_shape(), index, 9.0, index, Core.new(Cases.three_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  Records.CloneWrite{written, frame, second} = evidence
  written

def untouched_value(-index: U32, +other: U32, evidence: Records.CloneWrite<Cases.three_shape(), 7.0, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), 3n) == True{} : Bool}) ->
  {Observation.clone_written_second(Cases.three_shape(), index, 9.0, other, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.CloneWrite{written, frame, second} = evidence
  second(other, inside)

def owner_value(+other: U32, evidence: Records.MetadataContract<Cases.three_shape(), 7.0>,
  inside: {Nat.is_lt(U32.to_nat(other), 3n) == True{} : Bool}) ->
  {Observation.length_owner_value(Cases.three_shape(), other, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.MetadataContract{length, retained_dimensions, retained_count, retained_word_count, shape_dimensions, word_count,
    retained_value, shape_retained_value, count_retained_value} = evidence
  retained_value(other, inside)

def second_last_witness() ->
  {Observation.clone_second_value(Cases.three_shape(), 2, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  second_value(2, Contract.clone_contract(Cases.three_shape(), 7.0), {==})

def written_one_witness() ->
  {Observation.clone_written_first(Cases.three_shape(), 1, 9.0, 1, Core.new(Cases.three_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  written_value(1, Contract.clone_write(Cases.three_shape(), 7.0, 1, 9.0, {==}))

def alias_one_witness() ->
  {Observation.clone_written_second(Cases.three_shape(), 1, 9.0, 1, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  untouched_value(1, 1, Contract.clone_write(Cases.three_shape(), 7.0, 1, 9.0, {==}), {==})

def length_owner_zero_witness() ->
  {Observation.length_owner_value(Cases.three_shape(), 0, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  owner_value(0, Contract.metadata_contract(Cases.three_shape(), 7.0), {==})
