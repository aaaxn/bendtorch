import Base
import ../../../src/tensor/transform_contract_LAWS.bend as Contract
import ../../../src/tensor/transform_records.bend as Records
import ../../../src/tensor/transform_contract_PROOF.bend as ContractProof
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/transform_observation.bend as Observation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def one_shape() -> Shape.Shape:
  Shape.SShape{[1], 1n, {==}, Unit{}, Unit{}}

def six_shape() -> Shape.Shape:
  Shape.SShape{[6], 6n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def tall_shape() -> Shape.Shape:
  Shape.SShape{[3, 2], 6n, {==}, Unit{}, Unit{}}

def cube_shape() -> Shape.Shape:
  Shape.SShape{[2, 3, 4], 24n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def empty_grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 0], 0n, {==}, Unit{}, Unit{}}

type ReshapeCase<-input: Shape.Shape, -target: Shape.Shape, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  ReshapeCase{
    code: {Observation.reshape_code(input, Shape.dimensions(target), Core.new(input, 7.0)) == Some{0} : Maybe<&2, U32>},
    dimensions: {Observation.reshape_dimensions(input, Shape.dimensions(target), Core.new(input, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    count: {Observation.reshape_length(input, Shape.dimensions(target), Core.new(input, 7.0)) == Some{count} : Maybe<&2, Nat>},
    word_count: {Observation.reshape_count(input, Shape.dimensions(target), Core.new(input, 7.0)) == Some{word} : Maybe<&2, U32>},
    value: {Observation.reshape_value(input, Shape.dimensions(target), other, Core.new(input, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def reshape_expose(+input: Shape.Shape, +target: Shape.Shape, +other: U32, evidence: Records.ReshapeContract<input, 7.0, target>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  ReshapeCase<input, target, other, Shape.dimensions(target), Shape.logical_numel(Shape.dimensions(target)), Shape.count(target)>:
  Records.ReshapeContract{code, dimensions, count, word_count, value} = evidence
  ReshapeCase{code, dimensions, count, word_count, value(other, inside)}

def grid_to_tall() -> ReshapeCase<grid_shape(), tall_shape(), 5, [3, 2], 6n, 6>:
  reshape_expose(grid_shape(), tall_shape(), 5, Contract.reshape_contract(grid_shape(), 7.0, tall_shape(), {==}), {==})

def grid_to_flat() -> ReshapeCase<grid_shape(), six_shape(), 0, [6], 6n, 6>:
  reshape_expose(grid_shape(), six_shape(), 0, Contract.reshape_contract(grid_shape(), 7.0, six_shape(), {==}), {==})

def scalar_to_one() -> ReshapeCase<scalar_shape(), one_shape(), 0, [1], 1n, 1>:
  reshape_expose(scalar_shape(), one_shape(), 0, Contract.reshape_contract(scalar_shape(), 7.0, one_shape(), {==}), {==})

type EmptyReshape is Data:
  EmptyReshape{
    code: {Observation.reshape_code(empty_shape(), [2, 0], Core.new(empty_shape(), 7.0)) == Some{0} : Maybe<&2, U32>},
    dimensions: {Observation.reshape_dimensions(empty_shape(), [2, 0], Core.new(empty_shape(), 7.0)) == Some{[2, 0]} : Maybe<&2, List<&2, U32>>},
    count: {Observation.reshape_length(empty_shape(), [2, 0], Core.new(empty_shape(), 7.0)) == Some{0n} : Maybe<&2, Nat>}}

def empty_expose(evidence: Records.ReshapeContract<empty_shape(), 7.0, empty_grid_shape()>) -> EmptyReshape:
  Records.ReshapeContract{code, dimensions, count, word_count, value} = evidence
  EmptyReshape{code, dimensions, count}

def empty_reshape() -> EmptyReshape:
  empty_expose(Contract.reshape_contract(empty_shape(), 7.0, empty_grid_shape(), {==}))

type WriteCase<-input: Shape.Shape, -target: Shape.Shape, -index: U32, -other: U32> is Data:
  WriteCase{
    written: {Observation.reshape_written(input, index, 9.0, Shape.dimensions(target), index, Core.new(input, 7.0)) == Some{9.0} : Maybe<&2, F32>},
    neighbour: {Observation.reshape_written(input, index, 9.0, Shape.dimensions(target), other, Core.new(input, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def write_expose(+input: Shape.Shape, +target: Shape.Shape, -index: U32, +other: U32,
  evidence: Records.ReshapeWrite<input, 7.0, target, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) -> WriteCase<input, target, index, other>:
  Records.ReshapeWrite{written, frame} = evidence
  WriteCase{written, frame(other, inside, different)}

def grid_write_row() -> WriteCase<grid_shape(), tall_shape(), 1, 2>:
  write_expose(grid_shape(), tall_shape(), 1, 2, Contract.reshape_write(grid_shape(), 7.0, tall_shape(), 1, 9.0, {==}, {==}), {==}, {==})

def grid_write_last() -> WriteCase<grid_shape(), six_shape(), 5, 4>:
  write_expose(grid_shape(), six_shape(), 5, 4, Contract.reshape_write(grid_shape(), 7.0, six_shape(), 5, 9.0, {==}, {==}), {==}, {==})

def grid_mismatch() -> {Observation.reshape_code(grid_shape(), [2, 2], Core.new(grid_shape(), 7.0)) == Some{15} : Maybe<&2, U32>}:
  Contract.reshape_count_mismatch(grid_shape(), 7.0, Shape.SShape{[2, 2], 4n, {==}, Unit{}, Unit{}}, {==})

def empty_mismatch() -> {Observation.reshape_code(empty_shape(), [1], Core.new(empty_shape(), 7.0)) == Some{15} : Maybe<&2, U32>}:
  Contract.reshape_count_mismatch(empty_shape(), 7.0, one_shape(), {==})

type FailureCase<-input: Shape.Shape, -request: Observation.Request, -code: U32, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  FailureCase{
    code: {Observation.failure_code(input, request, Core.new(input, 7.0)) == Some{code} : Maybe<&2, U32>},
    retained_dimensions: {Observation.kept_dimensions(input, request, Core.new(input, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.kept_length(input, request, Core.new(input, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_word_count: {Observation.kept_count(input, request, Core.new(input, 7.0)) == Some{word} : Maybe<&2, U32>},
    retained_value: {Observation.kept_value(input, request, other, Core.new(input, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def failure_expose(+input: Shape.Shape, -request: Observation.Request, -code: U32, +other: U32,
  evidence: Records.RetainedFailure<input, 7.0, request, code>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  FailureCase<input, request, code, other, Shape.dimensions(input), Shape.logical_numel(Shape.dimensions(input)), Shape.count(input)>:
  Records.RetainedFailure{code, retained_dimensions, retained_count, retained_word_count, retained_value} = evidence
  FailureCase{code, retained_dimensions, retained_count, retained_word_count, retained_value(other, inside)}

def grid_short_axes() -> FailureCase<grid_shape(), Observation.Permute{[0]}, 12, 5, [2, 3], 6n, 6>:
  failure_expose(grid_shape(), Observation.Permute{[0]}, 12, 5, Contract.permute_rank_mismatch(grid_shape(), 7.0, [0], {==}), {==})

def cube_axis_outside() -> FailureCase<cube_shape(), Observation.Permute{[2, 0, 3]}, 13, 23, [2, 3, 4], 24n, 24>:
  failure_expose(cube_shape(), Observation.Permute{[2, 0, 3]}, 13, 23,
    Contract.permute_axis_out_of_bounds(cube_shape(), 7.0, [2, 0, 3], {==}, {==}), {==})

def cube_duplicate_axis() -> FailureCase<cube_shape(), Observation.Permute{[2, 0, 2]}, 14, 0, [2, 3, 4], 24n, 24>:
  failure_expose(cube_shape(), Observation.Permute{[2, 0, 2]}, 14, 0,
    Contract.permute_duplicate_axis(cube_shape(), 7.0, [2, 0, 2], {==}, {==}, {==}), {==})

def grid_transpose_outside() -> FailureCase<grid_shape(), Observation.Transpose{0, 2}, 13, 3, [2, 3], 6n, 6>:
  failure_expose(grid_shape(), Observation.Transpose{0, 2}, 13, 3, Contract.transpose_out_of_bounds(grid_shape(), 7.0, 0, 2, {==}), {==})

def scalar_transpose_outside() -> FailureCase<scalar_shape(), Observation.Transpose{1, 0}, 13, 0, [], 1n, 1>:
  failure_expose(scalar_shape(), Observation.Transpose{1, 0}, 13, 0, Contract.transpose_out_of_bounds(scalar_shape(), 7.0, 1, 0, {==}), {==})
