import Base
import ../../../src/tensor/permute_values_contract_LAWS.bend as Contract
import ../../../src/tensor/permute_values_records.bend as Records
import ../../../src/tensor/permute_values_contract_PROOF.bend as ContractProof
import ../../../src/tensor/contracts/PROOF.bend as ContractsProof
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/axis_spec.bend as Axis
import ../../../src/tensor/transpose_spec.bend as Transpose
import ../../../src/tensor/observation.bend as Values
import ../../../src/tensor/transform_observation.bend as Observation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def cube_shape() -> Shape.Shape:
  Shape.SShape{[2, 2, 2], 8n, {==}, Unit{}, Unit{}}

def grid_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def cube_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# Output axis k is input axis rotation[k]. A rotation of three axes is not its
# own inverse, so the forward and the inverse coordinate maps read different
# words.
def rotation() -> List<&2, U32>:
  [1, 2, 0]

type Case<-input: Shape.Shape, -values: List<&2, F32>, -request: Observation.Request, -dimensions: List<&2, U32>,
  -index: U32, -expected: F32, -other: U32, -kept: F32> is Data:
  Case{
    successful: {Observation.failure_code(input, request, Core.from_values(input, values)) == Some{0} : Maybe<&2, U32>},
    dimensions_exact: {Observation.applied_dimensions(input, request, Core.from_values(input, values)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    value_exact: {Observation.applied_value(input, request, index, Core.from_values(input, values)) == Some{expected} : Maybe<&2, F32>},
    kept_exact: {Observation.kept_value(input, request, other, Core.from_values(input, values)) == Some{kept} : Maybe<&2, F32>}}

def permute_expose(+input: Shape.Shape, +values: List<&2, F32>, +axes: List<&2, U32>, -dimensions: List<&2, U32>,
  +index: U32, -expected: F32, +other: U32, -kept: F32, evidence: Records.PermuteContract<input, values, axes>,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  retained: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  ordered: {Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))} == Some{dimensions} : Maybe<&2, List<&2, U32>>},
  found: {Values.listed(values, Values.permute_source(input, axes, index)) == Some{expected} : Maybe<&2, F32>},
  stored: {List.get(&2, F32, values, U32.to_nat(other)) == Some{kept} : Maybe<&2, F32>}) ->
  Case<input, values, Observation.Permute{axes}, dimensions, index, expected, other, kept>:
  Records.PermuteContract{outcome, shaped, count, word_count, value, retained_value} = evidence
  Case{outcome,
    Equal.trans(Maybe<&2, List<&2, U32>>, Observation.applied_dimensions(input, Observation.Permute{axes}, Core.from_values(input, values)),
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, Some{dimensions}, shaped, ordered),
    Equal.trans(Maybe<&2, F32>, Observation.applied_value(input, Observation.Permute{axes}, index, Core.from_values(input, values)),
      Values.listed(values, Values.permute_source(input, axes, index)), Some{expected}, value(index, inside), found),
    Equal.trans(Maybe<&2, F32>, Observation.kept_value(input, Observation.Permute{axes}, other, Core.from_values(input, values)),
      List.get(&2, F32, values, U32.to_nat(other)), Some{kept}, retained_value(other, retained), stored)}

def transpose_expose(+input: Shape.Shape, +values: List<&2, F32>, +a: U32, +b: U32, -dimensions: List<&2, U32>,
  +index: U32, -expected: F32, +other: U32, -kept: F32, evidence: Records.TransposeContract<input, values, a, b>,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  retained: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool},
  ordered: {Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))))} ==
    Some{dimensions} : Maybe<&2, List<&2, U32>>},
  found: {Values.listed(values, Values.permute_source(input, Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b)), index)) ==
    Some{expected} : Maybe<&2, F32>},
  stored: {List.get(&2, F32, values, U32.to_nat(other)) == Some{kept} : Maybe<&2, F32>}) ->
  Case<input, values, Observation.Transpose{a, b}, dimensions, index, expected, other, kept>:
  Records.TransposeContract{outcome, shaped, count, word_count, value, retained_value} = evidence
  -axes = Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(input)), U32.to_nat(a), U32.to_nat(b))
  Case{outcome,
    Equal.trans(Maybe<&2, List<&2, U32>>, Observation.applied_dimensions(input, Observation.Transpose{a, b}, Core.from_values(input, values)),
      Some{Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))}, Some{dimensions}, shaped, ordered),
    Equal.trans(Maybe<&2, F32>, Observation.applied_value(input, Observation.Transpose{a, b}, index, Core.from_values(input, values)),
      Values.listed(values, Values.permute_source(input, axes, index)), Some{expected}, value(index, inside), found),
    Equal.trans(Maybe<&2, F32>, Observation.kept_value(input, Observation.Transpose{a, b}, other, Core.from_values(input, values)),
      List.get(&2, F32, values, U32.to_nat(other)), Some{kept}, retained_value(other, retained), stored)}

def scalar_permute() -> Case<scalar_shape(), [5.0], Observation.Permute{[]}, [], 0, 5.0, 0, 5.0>:
  permute_expose(scalar_shape(), [5.0], [], [], 0, 5.0, 0, 5.0,
    Contract.permute_contract(scalar_shape(), [5.0], {==}, [], {==}), {==}, {==}, {==}, {==}, {==})

def grid_permute_first() -> Case<grid_shape(), grid_values(), Observation.Permute{[1, 0]}, [3, 2], 0, 1.0, 5, 6.0>:
  permute_expose(grid_shape(), grid_values(), [1, 0], [3, 2], 0, 1.0, 5, 6.0,
    Contract.permute_contract(grid_shape(), grid_values(), {==}, [1, 0], {==}), {==}, {==}, {==}, {==}, {==})

def grid_permute_second() -> Case<grid_shape(), grid_values(), Observation.Permute{[1, 0]}, [3, 2], 1, 4.0, 1, 2.0>:
  permute_expose(grid_shape(), grid_values(), [1, 0], [3, 2], 1, 4.0, 1, 2.0,
    Contract.permute_contract(grid_shape(), grid_values(), {==}, [1, 0], {==}), {==}, {==}, {==}, {==}, {==})

def grid_permute_row() -> Case<grid_shape(), grid_values(), Observation.Permute{[1, 0]}, [3, 2], 4, 3.0, 3, 4.0>:
  permute_expose(grid_shape(), grid_values(), [1, 0], [3, 2], 4, 3.0, 3, 4.0,
    Contract.permute_contract(grid_shape(), grid_values(), {==}, [1, 0], {==}), {==}, {==}, {==}, {==}, {==})

def cube_rotate_one() -> Case<cube_shape(), cube_values(), Observation.Permute{rotation()}, [2, 2, 2], 1, 5.0, 1, 2.0>:
  permute_expose(cube_shape(), cube_values(), rotation(), [2, 2, 2], 1, 5.0, 1, 2.0,
    Contract.permute_contract(cube_shape(), cube_values(), {==}, rotation(), {==}), {==}, {==}, {==}, {==}, {==})

def cube_rotate_two() -> Case<cube_shape(), cube_values(), Observation.Permute{rotation()}, [2, 2, 2], 2, 2.0, 4, 5.0>:
  permute_expose(cube_shape(), cube_values(), rotation(), [2, 2, 2], 2, 2.0, 4, 5.0,
    Contract.permute_contract(cube_shape(), cube_values(), {==}, rotation(), {==}), {==}, {==}, {==}, {==}, {==})

def cube_rotate_six() -> Case<cube_shape(), cube_values(), Observation.Permute{rotation()}, [2, 2, 2], 6, 4.0, 7, 8.0>:
  permute_expose(cube_shape(), cube_values(), rotation(), [2, 2, 2], 6, 4.0, 7, 8.0,
    Contract.permute_contract(cube_shape(), cube_values(), {==}, rotation(), {==}), {==}, {==}, {==}, {==}, {==})

def grid_transpose() -> Case<grid_shape(), grid_values(), Observation.Transpose{0, 1}, [3, 2], 1, 4.0, 2, 3.0>:
  transpose_expose(grid_shape(), grid_values(), 0, 1, [3, 2], 1, 4.0, 2, 3.0,
    Contract.transpose_contract(grid_shape(), grid_values(), {==}, 0, 1, {==}), {==}, {==}, {==}, {==}, {==})

def grid_transpose_reversed() -> Case<grid_shape(), grid_values(), Observation.Transpose{1, 0}, [3, 2], 3, 5.0, 0, 1.0>:
  transpose_expose(grid_shape(), grid_values(), 1, 0, [3, 2], 3, 5.0, 0, 1.0,
    Contract.transpose_contract(grid_shape(), grid_values(), {==}, 1, 0, {==}), {==}, {==}, {==}, {==}, {==})

def cube_transpose_outer() -> Case<cube_shape(), cube_values(), Observation.Transpose{0, 2}, [2, 2, 2], 1, 5.0, 6, 7.0>:
  transpose_expose(cube_shape(), cube_values(), 0, 2, [2, 2, 2], 1, 5.0, 6, 7.0,
    Contract.transpose_contract(cube_shape(), cube_values(), {==}, 0, 2, {==}), {==}, {==}, {==}, {==}, {==})

def scalar_transpose() -> Case<scalar_shape(), [5.0], Observation.Transpose{0, 0}, [], 0, 5.0, 0, 5.0>:
  transpose_expose(scalar_shape(), [5.0], 0, 0, [], 0, 5.0, 0, 5.0,
    Contract.transpose_contract(scalar_shape(), [5.0], {==}, 0, 0, {==}), {==}, {==}, {==}, {==}, {==})
