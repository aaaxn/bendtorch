import Base
import ../../../src/tensor/read_write_contract_LAWS.bend as Contract
import ../../../src/tensor/read_write_records.bend as Records
import ../../../src/tensor/read_write_contract_PROOF.bend as ContractProof
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/observation.bend as Observation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

type ReadCase<-shape: Shape.Shape, -index: U32, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  ReadCase{
    code: {Observation.read_code(shape, index, Core.new(shape, 7.0)) == Some{0} : Maybe<&2, U32>},
    value: {Observation.point(shape, index, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    retained_dimensions: {Observation.after_dimensions(shape, index, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.after_length(shape, index, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_word_count: {Observation.after_count(shape, index, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    retained_value: {Observation.after_value(shape, index, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def read_expose(+shape: Shape.Shape, -index: U32, +other: U32, evidence: Records.ReadInside<shape, 7.0, index>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  ReadCase<shape, index, other, Shape.dimensions(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.count(shape)>:
  Records.ReadInside{read, value} = evidence
  Records.ReadContract{code, retained_dimensions, retained_count, retained_word_count, retained_value} = read
  ReadCase{code, value, retained_dimensions, retained_count, retained_word_count, retained_value(other, inside)}

def scalar_read() -> ReadCase<scalar_shape(), 0, 0, [], 1n, 1>:
  read_expose(scalar_shape(), 0, 0, Contract.read_inside(scalar_shape(), 7.0, 0, {==}), {==})

def three_read_zero() -> ReadCase<three_shape(), 0, 2, [3], 3n, 3>:
  read_expose(three_shape(), 0, 2, Contract.read_inside(three_shape(), 7.0, 0, {==}), {==})

def three_read_one() -> ReadCase<three_shape(), 1, 0, [3], 3n, 3>:
  read_expose(three_shape(), 1, 0, Contract.read_inside(three_shape(), 7.0, 1, {==}), {==})

def three_read_two() -> ReadCase<three_shape(), 2, 1, [3], 3n, 3>:
  read_expose(three_shape(), 2, 1, Contract.read_inside(three_shape(), 7.0, 2, {==}), {==})

def grid_read_last() -> ReadCase<grid_shape(), 5, 0, [2, 3], 6n, 6>:
  read_expose(grid_shape(), 5, 0, Contract.read_inside(grid_shape(), 7.0, 5, {==}), {==})

type WriteCase<-shape: Shape.Shape, -index: U32, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  WriteCase{
    code: {Observation.write_code(shape, index, 9.0, Core.new(shape, 7.0)) == Some{0} : Maybe<&2, U32>},
    written: {Observation.written_value(shape, index, 9.0, index, Core.new(shape, 7.0)) == Some{9.0} : Maybe<&2, F32>},
    neighbour: {Observation.written_value(shape, index, 9.0, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    retained_dimensions: {Observation.written_dimensions(shape, index, 9.0, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.written_length(shape, index, 9.0, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_word_count: {Observation.written_count(shape, index, 9.0, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>}}

def write_expose(+shape: Shape.Shape, -index: U32, +other: U32, evidence: Records.WriteInside<shape, 7.0, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) ->
  WriteCase<shape, index, other, Shape.dimensions(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.count(shape)>:
  Records.WriteInside{write, written, frame} = evidence
  Records.WriteContract{code, retained_dimensions, retained_count, retained_word_count} = write
  WriteCase{code, written, frame(other, inside, different), retained_dimensions, retained_count, retained_word_count}

def three_write_zero() -> WriteCase<three_shape(), 0, 1, [3], 3n, 3>:
  write_expose(three_shape(), 0, 1, Contract.write_inside(three_shape(), 7.0, 0, 9.0, {==}), {==}, {==})

def three_write_one() -> WriteCase<three_shape(), 1, 2, [3], 3n, 3>:
  write_expose(three_shape(), 1, 2, Contract.write_inside(three_shape(), 7.0, 1, 9.0, {==}), {==}, {==})

def three_write_two() -> WriteCase<three_shape(), 2, 1, [3], 3n, 3>:
  write_expose(three_shape(), 2, 1, Contract.write_inside(three_shape(), 7.0, 2, 9.0, {==}), {==}, {==})

type OutsideCase<-shape: Shape.Shape, -index: U32, -other: U32> is Data:
  OutsideCase{
    read_code: {Observation.read_code(shape, index, Core.new(shape, 7.0)) == Some{8} : Maybe<&2, U32>},
    read_kept: {Observation.after_value(shape, index, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    write_code: {Observation.write_code(shape, index, 9.0, Core.new(shape, 7.0)) == Some{8} : Maybe<&2, U32>},
    write_kept: {Observation.written_value(shape, index, 9.0, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def outside_expose(+shape: Shape.Shape, -index: U32, +other: U32, read: Records.ReadContract<shape, 7.0, index, 8>,
  write: Records.WriteOutside<shape, 7.0, index, 9.0>,
  +inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> OutsideCase<shape, index, other>:
  Records.ReadContract{read_code, read_dimensions, read_count, read_word_count, read_kept} = read
  Records.WriteOutside{written, write_kept} = write
  Records.WriteContract{write_code, write_dimensions, write_count, write_word_count} = written
  OutsideCase{read_code, read_kept(other, inside), write_code, write_kept(other, inside)}

def three_outside() -> OutsideCase<three_shape(), 3, 2>:
  outside_expose(three_shape(), 3, 2, Contract.read_outside(three_shape(), 7.0, 3, {==}),
    Contract.write_outside(three_shape(), 7.0, 3, 9.0, {==}), {==})

def scalar_outside() -> OutsideCase<scalar_shape(), 4294967295, 0>:
  outside_expose(scalar_shape(), 4294967295, 0, Contract.read_outside(scalar_shape(), 7.0, 4294967295, {==}),
    Contract.write_outside(scalar_shape(), 7.0, 4294967295, 9.0, {==}), {==})

type EmptyCase<-shape: Shape.Shape> is Data:
  EmptyCase{
    read_code: {Observation.read_code(shape, 0, Core.new(shape, 7.0)) == Some{8} : Maybe<&2, U32>},
    read_dimensions: {Observation.after_dimensions(shape, 0, Core.new(shape, 7.0)) == Some{[0]} : Maybe<&2, List<&2, U32>>},
    write_code: {Observation.write_code(shape, 0, 9.0, Core.new(shape, 7.0)) == Some{8} : Maybe<&2, U32>},
    write_count: {Observation.written_length(shape, 0, 9.0, Core.new(shape, 7.0)) == Some{0n} : Maybe<&2, Nat>}}

def empty_expose(read: Records.ReadContract<empty_shape(), 7.0, 0, 8>, write: Records.WriteOutside<empty_shape(), 7.0, 0, 9.0>) ->
  EmptyCase<empty_shape()>:
  Records.ReadContract{read_code, read_dimensions, read_count, read_word_count, read_kept} = read
  Records.WriteOutside{written, write_kept} = write
  Records.WriteContract{write_code, write_dimensions, write_count, write_word_count} = written
  EmptyCase{read_code, read_dimensions, write_code, write_count}

def empty() -> EmptyCase<empty_shape()>:
  empty_expose(Contract.read_outside(empty_shape(), 7.0, 0, {==}), Contract.write_outside(empty_shape(), 7.0, 0, 9.0, {==}))

type CornerCase<-coordinates: List<&2, U32>, -index: U32, -other: U32> is Data:
  CornerCase{
    read: {Observation.coordinate_value(grid_shape(), index, 9.0, coordinates, Core.new(grid_shape(), 7.0)) == Some{9.0} : Maybe<&2, F32>},
    written: {Observation.coordinate_written(grid_shape(), coordinates, 9.0, index, Core.new(grid_shape(), 7.0)) == Some{9.0} : Maybe<&2, F32>},
    neighbour: {Observation.coordinate_written(grid_shape(), coordinates, 9.0, other, Core.new(grid_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def corner_expose(-coordinates: List<&2, U32>, -index: U32, +other: U32, evidence: Records.CoordinateInside<grid_shape(), 7.0, coordinates, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), 6n) == True{} : Bool}, different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) ->
  CornerCase<coordinates, index, other>:
  Records.CoordinateInside{read, written, frame} = evidence
  CornerCase{read, written, frame(other, inside, different)}

def corner_first() -> CornerCase<[0, 0], 0, 1>:
  corner_expose([0, 0], 0, 1, Contract.coordinate_inside(grid_shape(), 7.0, [0, 0], 0, 9.0, {==}, {==}), {==}, {==})

def corner_row() -> CornerCase<[0, 2], 2, 3>:
  corner_expose([0, 2], 2, 3, Contract.coordinate_inside(grid_shape(), 7.0, [0, 2], 2, 9.0, {==}, {==}), {==}, {==})

def corner_column() -> CornerCase<[1, 0], 3, 2>:
  corner_expose([1, 0], 3, 2, Contract.coordinate_inside(grid_shape(), 7.0, [1, 0], 3, 9.0, {==}, {==}), {==}, {==})

def corner_last() -> CornerCase<[1, 2], 5, 4>:
  corner_expose([1, 2], 5, 4, Contract.coordinate_inside(grid_shape(), 7.0, [1, 2], 5, 9.0, {==}, {==}), {==}, {==})

def dynamic_read() -> {Observation.dynamic_point(three_shape(), 2, Core.new(three_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  Contract.dynamic_read_inside(three_shape(), 7.0, 2, {==})
