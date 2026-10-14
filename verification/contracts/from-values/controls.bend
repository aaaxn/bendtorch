import Base
import ../../../src/tensor/from-values-contract-LAWS.bend as Contract
import ../../../src/tensor/from-values-records.bend as Records
import ../../../src/tensor/from-values-contract-PROOF.bend as ContractProof
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/observation.bend as Observation
import ./consumers.bend as Cases

def expected_first() -> Maybe<&2, F32>:
  Some{1.0}

def expected_second() -> Maybe<&2, F32>:
  Some{2.0}

def expected_last() -> Maybe<&2, F32>:
  Some{6.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def three_value(+index: U32, evidence: Records.FromValuesContract<Cases.three_shape(), Cases.three_values()>,
  inside: {Nat.is_lt(U32.to_nat(index), 3n) == True{} : Bool}) ->
  {Observation.point(Cases.three_shape(), index, Core.from_values(Cases.three_shape(), Cases.three_values())) ==
    List.get(&2, F32, Cases.three_values(), U32.to_nat(index)) : Maybe<&2, F32>}:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  value(index, inside)

def grid_value(+index: U32, evidence: Records.FromValuesContract<Cases.grid_shape(), Cases.grid_values()>,
  inside: {Nat.is_lt(U32.to_nat(index), 6n) == True{} : Bool}) ->
  {Observation.point(Cases.grid_shape(), index, Core.from_values(Cases.grid_shape(), Cases.grid_values())) ==
    List.get(&2, F32, Cases.grid_values(), U32.to_nat(index)) : Maybe<&2, F32>}:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  value(index, inside)

def three_first_witness() ->
  {Observation.point(Cases.three_shape(), 0, Core.from_values(Cases.three_shape(), Cases.three_values())) == expected_first() : Maybe<&2, F32>}:
  three_value(0, Contract.from_values_contract(Cases.three_shape(), Cases.three_values(), {==}), {==})

def three_second_witness() ->
  {Observation.point(Cases.three_shape(), 1, Core.from_values(Cases.three_shape(), Cases.three_values())) == expected_second() : Maybe<&2, F32>}:
  three_value(1, Contract.from_values_contract(Cases.three_shape(), Cases.three_values(), {==}), {==})

def grid_last_witness() ->
  {Observation.point(Cases.grid_shape(), 5, Core.from_values(Cases.grid_shape(), Cases.grid_values())) == expected_last() : Maybe<&2, F32>}:
  grid_value(5, Contract.from_values_contract(Cases.grid_shape(), Cases.grid_values(), {==}), {==})
