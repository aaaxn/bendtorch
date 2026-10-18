import Base
import ../../../src/tensor/construction_contract_LAWS.bend as Contract
import ../../../src/tensor/construction_contract_PROOF.bend as ContractProof
import ../../../src/tensor/core.bend as Core
import ./consumers.bend as Cases
import ../../../src/tensor/observation.bend as Observation

def expected_value() -> Maybe<&2, F32>:
  Some{7.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def wrong_fill_value() -> F32:
  0.0

def scalar_value(evidence: Contract.NewReads<Cases.scalar_shape(), 7.0, 0>) ->
  {Observation.point(Cases.scalar_shape(), 0, Core.new(Cases.scalar_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  Contract.NewReads{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  value

def three_value(-index: U32, evidence: Contract.NewReads<Cases.three_shape(), 7.0, index>) ->
  {Observation.point(Cases.three_shape(), index, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  Contract.NewReads{successful, dimensions, count, word_count, value, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  value

def scalar_witness() ->
  {Observation.point(Cases.scalar_shape(), 0, Core.new(Cases.scalar_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  scalar_value(Contract.new_reads(Cases.scalar_shape(), 7.0, 0, {==}))

def three_zero_witness() ->
  {Observation.point(Cases.three_shape(), 0, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(0, Contract.new_reads(Cases.three_shape(), 7.0, 0, {==}))

def three_one_witness() ->
  {Observation.point(Cases.three_shape(), 1, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(1, Contract.new_reads(Cases.three_shape(), 7.0, 1, {==}))

def three_two_witness() ->
  {Observation.point(Cases.three_shape(), 2, Core.new(Cases.three_shape(), 7.0)) == expected_value() : Maybe<&2, F32>}:
  three_value(2, Contract.new_reads(Cases.three_shape(), 7.0, 2, {==}))
