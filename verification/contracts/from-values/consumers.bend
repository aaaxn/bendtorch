import Base
import ../../../src/tensor/from_values_contract_LAWS.bend as Contract
import ../../../src/tensor/from_values_records.bend as Records
import ../../../src/tensor/from_values_contract_PROOF.bend as ContractProof
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/allocation_spec.bend as Allocation
import ../../../src/tensor/observation.bend as Observation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def three_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0]

def grid_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

type ValuesCase<-shape: Shape.Shape, -values: List<&2, F32>, -index: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32, -expected: F32> is Data:
  ValuesCase{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.from_values(shape, values)) == True{} : Bool},
    dimensions_exact: {Observation.dimensions(shape, Core.from_values(shape, values)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    count_exact: {Allocation.tensor_length(shape, Core.from_values(shape, values)) == Some{count} : Maybe<&2, Nat>},
    word_count_exact: {Observation.count(shape, Core.from_values(shape, values)) == Some{word} : Maybe<&2, U32>},
    value_exact: {Observation.point(shape, index, Core.from_values(shape, values)) == Some{expected} : Maybe<&2, F32>}}

def scalar_expose(evidence: Records.FromValuesContract<scalar_shape(), [5.0]>) -> ValuesCase<scalar_shape(), [5.0], 0, [], 1n, 1, 5.0>:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  ValuesCase{successful, dimensions, count, word_count, value(0, {==})}

def scalar() -> ValuesCase<scalar_shape(), [5.0], 0, [], 1n, 1, 5.0>:
  scalar_expose(Contract.from_values_contract(scalar_shape(), [5.0], {==}))

def three_expose(+index: U32, -expected: F32, evidence: Records.FromValuesContract<three_shape(), three_values()>,
  inside: {Nat.is_lt(U32.to_nat(index), 3n) == True{} : Bool},
  found: {List.get(&2, F32, three_values(), U32.to_nat(index)) == Some{expected} : Maybe<&2, F32>}) ->
  ValuesCase<three_shape(), three_values(), index, [3], 3n, 3, expected>:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  ValuesCase{successful, dimensions, count, word_count,
    Equal.trans(Maybe<&2, F32>, Observation.point(three_shape(), index, Core.from_values(three_shape(), three_values())),
      List.get(&2, F32, three_values(), U32.to_nat(index)), Some{expected}, value(index, inside), found)}

def three_zero() -> ValuesCase<three_shape(), three_values(), 0, [3], 3n, 3, 1.0>:
  three_expose(0, 1.0, Contract.from_values_contract(three_shape(), three_values(), {==}), {==}, {==})

def three_one() -> ValuesCase<three_shape(), three_values(), 1, [3], 3n, 3, 2.0>:
  three_expose(1, 2.0, Contract.from_values_contract(three_shape(), three_values(), {==}), {==}, {==})

def three_two() -> ValuesCase<three_shape(), three_values(), 2, [3], 3n, 3, 3.0>:
  three_expose(2, 3.0, Contract.from_values_contract(three_shape(), three_values(), {==}), {==}, {==})

def grid_expose(+index: U32, -expected: F32, evidence: Records.FromValuesContract<grid_shape(), grid_values()>,
  inside: {Nat.is_lt(U32.to_nat(index), 6n) == True{} : Bool},
  found: {List.get(&2, F32, grid_values(), U32.to_nat(index)) == Some{expected} : Maybe<&2, F32>}) ->
  ValuesCase<grid_shape(), grid_values(), index, [2, 3], 6n, 6, expected>:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  ValuesCase{successful, dimensions, count, word_count,
    Equal.trans(Maybe<&2, F32>, Observation.point(grid_shape(), index, Core.from_values(grid_shape(), grid_values())),
      List.get(&2, F32, grid_values(), U32.to_nat(index)), Some{expected}, value(index, inside), found)}

def grid_first() -> ValuesCase<grid_shape(), grid_values(), 0, [2, 3], 6n, 6, 1.0>:
  grid_expose(0, 1.0, Contract.from_values_contract(grid_shape(), grid_values(), {==}), {==}, {==})

def grid_row() -> ValuesCase<grid_shape(), grid_values(), 3, [2, 3], 6n, 6, 4.0>:
  grid_expose(3, 4.0, Contract.from_values_contract(grid_shape(), grid_values(), {==}), {==}, {==})

def grid_last() -> ValuesCase<grid_shape(), grid_values(), 5, [2, 3], 6n, 6, 6.0>:
  grid_expose(5, 6.0, Contract.from_values_contract(grid_shape(), grid_values(), {==}), {==}, {==})

type EmptyCase is Data:
  EmptyCase{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<empty_shape()>, Core.from_values(empty_shape(), [])) == True{} : Bool},
    dimensions_exact: {Observation.dimensions(empty_shape(), Core.from_values(empty_shape(), [])) == Some{[0]} : Maybe<&2, List<&2, U32>>},
    count_exact: {Allocation.tensor_length(empty_shape(), Core.from_values(empty_shape(), [])) == Some{0n} : Maybe<&2, Nat>},
    word_count_exact: {Observation.count(empty_shape(), Core.from_values(empty_shape(), [])) == Some{0} : Maybe<&2, U32>}}

def empty_expose(evidence: Records.FromValuesContract<empty_shape(), []>) -> EmptyCase:
  Records.FromValuesContract{successful, dimensions, count, word_count, value} = evidence
  EmptyCase{successful, dimensions, count, word_count}

def empty() -> EmptyCase:
  empty_expose(Contract.from_values_contract(empty_shape(), [], {==}))
