import Base
import ../../../src/tensor/read-write-contract-LAWS.bend as Contract
import ../../../src/tensor/read-write-records.bend as Records
import ../../../src/tensor/read-write-contract-PROOF.bend as ContractProof
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/observation.bend as Observation
import ./consumers.bend as Cases

def expected_fill() -> Maybe<&2, F32>:
  Some{7.0}

def expected_written() -> Maybe<&2, F32>:
  Some{9.0}

def wrong_value() -> Maybe<&2, F32>:
  Some{8.0}

def read_value(-index: U32, evidence: Records.ReadInside<Cases.three_shape(), 7.0, index>) ->
  {Observation.point(Cases.three_shape(), index, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.ReadInside{read, value} = evidence
  value

def written_value(-index: U32, evidence: Records.WriteInside<Cases.three_shape(), 7.0, index, 9.0>) ->
  {Observation.written_value(Cases.three_shape(), index, 9.0, index, Core.new(Cases.three_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  Records.WriteInside{write, written, frame} = evidence
  written

def neighbour_value(-index: U32, +other: U32, evidence: Records.WriteInside<Cases.three_shape(), 7.0, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), 3n) == True{} : Bool}, different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) ->
  {Observation.written_value(Cases.three_shape(), index, 9.0, other, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  Records.WriteInside{write, written, frame} = evidence
  frame(other, inside, different)

def read_one_witness() ->
  {Observation.point(Cases.three_shape(), 1, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  read_value(1, Contract.read_inside(Cases.three_shape(), 7.0, 1, {==}))

def written_one_witness() ->
  {Observation.written_value(Cases.three_shape(), 1, 9.0, 1, Core.new(Cases.three_shape(), 7.0)) == expected_written() : Maybe<&2, F32>}:
  written_value(1, Contract.write_inside(Cases.three_shape(), 7.0, 1, 9.0, {==}))

def neighbour_two_witness() ->
  {Observation.written_value(Cases.three_shape(), 1, 9.0, 2, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  neighbour_value(1, 2, Contract.write_inside(Cases.three_shape(), 7.0, 1, 9.0, {==}), {==}, {==})

def neighbour_zero_witness() ->
  {Observation.written_value(Cases.three_shape(), 1, 9.0, 0, Core.new(Cases.three_shape(), 7.0)) == expected_fill() : Maybe<&2, F32>}:
  neighbour_value(1, 0, Contract.write_inside(Cases.three_shape(), 7.0, 1, 9.0, {==}), {==}, {==})
