import Base
import ../../../src/tensor/clone-metadata-contract-LAWS.bend as Contract
import ../../../src/tensor/clone-metadata-records.bend as Records
import ../../../src/tensor/clone-metadata-contract-PROOF.bend as ContractProof
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

type CloneCase<-shape: Shape.Shape, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  CloneCase{
    first_dimensions: {Observation.clone_first_dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    second_dimensions: {Observation.clone_second_dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    first_count: {Observation.clone_first_length(shape, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    second_count: {Observation.clone_second_length(shape, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    first_word_count: {Observation.clone_first_count(shape, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    second_word_count: {Observation.clone_second_count(shape, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    first_value: {Observation.clone_first_value(shape, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    second_value: {Observation.clone_second_value(shape, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def clone_expose(+shape: Shape.Shape, +other: U32, evidence: Records.CloneContract<shape, 7.0>,
  +inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  CloneCase<shape, other, Shape.dimensions(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.count(shape)>:
  Records.CloneContract{first_dimensions, second_dimensions, first_count, second_count, first_word_count, second_word_count, first_value, second_value} = evidence
  CloneCase{first_dimensions, second_dimensions, first_count, second_count, first_word_count, second_word_count,
    first_value(other, inside), second_value(other, inside)}

def scalar_clone() -> CloneCase<scalar_shape(), 0, [], 1n, 1>:
  clone_expose(scalar_shape(), 0, Contract.clone_contract(scalar_shape(), 7.0), {==})

def three_clone_last() -> CloneCase<three_shape(), 2, [3], 3n, 3>:
  clone_expose(three_shape(), 2, Contract.clone_contract(three_shape(), 7.0), {==})

def grid_clone_first() -> CloneCase<grid_shape(), 0, [2, 3], 6n, 6>:
  clone_expose(grid_shape(), 0, Contract.clone_contract(grid_shape(), 7.0), {==})

def grid_clone_last() -> CloneCase<grid_shape(), 5, [2, 3], 6n, 6>:
  clone_expose(grid_shape(), 5, Contract.clone_contract(grid_shape(), 7.0), {==})

type EmptyClone is Data:
  EmptyClone{
    first_dimensions: {Observation.clone_first_dimensions(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{[0]} : Maybe<&2, List<&2, U32>>},
    second_dimensions: {Observation.clone_second_dimensions(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{[0]} : Maybe<&2, List<&2, U32>>},
    first_count: {Observation.clone_first_length(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{0n} : Maybe<&2, Nat>},
    second_count: {Observation.clone_second_length(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{0n} : Maybe<&2, Nat>}}

def empty_expose(evidence: Records.CloneContract<empty_shape(), 7.0>) -> EmptyClone:
  Records.CloneContract{first_dimensions, second_dimensions, first_count, second_count, first_word_count, second_word_count, first_value, second_value} = evidence
  EmptyClone{first_dimensions, second_dimensions, first_count, second_count}

def empty_clone() -> EmptyClone:
  empty_expose(Contract.clone_contract(empty_shape(), 7.0))

type WriteCase<-shape: Shape.Shape, -index: U32, -other: U32> is Data:
  WriteCase{
    written: {Observation.clone_written_first(shape, index, 9.0, index, Core.new(shape, 7.0)) == Some{9.0} : Maybe<&2, F32>},
    neighbour: {Observation.clone_written_first(shape, index, 9.0, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    second_at_index: {Observation.clone_written_second(shape, index, 9.0, index, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    second_at_other: {Observation.clone_written_second(shape, index, 9.0, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def second_value(-shape: Shape.Shape, -index: U32, +other: U32, evidence: Records.CloneWrite<shape, 7.0, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.clone_written_second(shape, index, 9.0, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  Records.CloneWrite{written, frame, second} = evidence
  second(other, inside)

def write_expose(+shape: Shape.Shape, +index: U32, +other: U32, evidence: Records.CloneWrite<shape, 7.0, index, 9.0>,
  again: Records.CloneWrite<shape, 7.0, index, 9.0>,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  +inside_other: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) -> WriteCase<shape, index, other>:
  Records.CloneWrite{written, frame, second} = evidence
  WriteCase{written, frame(other, inside_other, different), second_value(shape, index, index, again, inside), second(other, inside_other)}

def three_write_one() -> WriteCase<three_shape(), 1, 2>:
  write_expose(three_shape(), 1, 2, Contract.clone_write(three_shape(), 7.0, 1, 9.0, {==}),
    Contract.clone_write(three_shape(), 7.0, 1, 9.0, {==}), {==}, {==}, {==})

def grid_write_last() -> WriteCase<grid_shape(), 5, 0>:
  write_expose(grid_shape(), 5, 0, Contract.clone_write(grid_shape(), 7.0, 5, 9.0, {==}),
    Contract.clone_write(grid_shape(), 7.0, 5, 9.0, {==}), {==}, {==}, {==})

def second_untouched(evidence: Records.CloneWrite<scalar_shape(), 7.0, 0, 9.0>) ->
  {Observation.clone_written_second(scalar_shape(), 0, 9.0, 0, Core.new(scalar_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  Records.CloneWrite{written, frame, second} = evidence
  second(0, {==})

def scalar_write() -> {Observation.clone_written_second(scalar_shape(), 0, 9.0, 0, Core.new(scalar_shape(), 7.0)) == Some{7.0} : Maybe<&2, F32>}:
  second_untouched(Contract.clone_write(scalar_shape(), 7.0, 0, 9.0, {==}))

type MetadataCase<-shape: Shape.Shape, -other: U32, -dimensions: List<&2, U32>, -count: Nat, -word: U32> is Data:
  MetadataCase{
    length: {Observation.length_value(shape, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_dimensions: {Observation.length_owner_dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    retained_count: {Observation.length_owner_length(shape, Core.new(shape, 7.0)) == Some{count} : Maybe<&2, Nat>},
    retained_word_count: {Observation.length_owner_count(shape, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    shape_dimensions: {Observation.dimensions(shape, Core.new(shape, 7.0)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    word_count: {Observation.count(shape, Core.new(shape, 7.0)) == Some{word} : Maybe<&2, U32>},
    retained_value: {Observation.length_owner_value(shape, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    shape_retained_value: {Observation.shape_owner_value(shape, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>},
    count_retained_value: {Observation.count_owner_value(shape, other, Core.new(shape, 7.0)) == Some{7.0} : Maybe<&2, F32>}}

def metadata_expose(+shape: Shape.Shape, +other: U32, evidence: Records.MetadataContract<shape, 7.0>,
  +inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  MetadataCase<shape, other, Shape.dimensions(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.count(shape)>:
  Records.MetadataContract{length, retained_dimensions, retained_count, retained_word_count, shape_dimensions, word_count,
    retained_value, shape_retained_value, count_retained_value} = evidence
  MetadataCase{length, retained_dimensions, retained_count, retained_word_count, shape_dimensions, word_count,
    retained_value(other, inside), shape_retained_value(other, inside), count_retained_value(other, inside)}

def scalar_metadata() -> MetadataCase<scalar_shape(), 0, [], 1n, 1>:
  metadata_expose(scalar_shape(), 0, Contract.metadata_contract(scalar_shape(), 7.0), {==})

def grid_metadata() -> MetadataCase<grid_shape(), 4, [2, 3], 6n, 6>:
  metadata_expose(grid_shape(), 4, Contract.metadata_contract(grid_shape(), 7.0), {==})

type EmptyMetadata is Data:
  EmptyMetadata{
    length: {Observation.length_value(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{0n} : Maybe<&2, Nat>},
    shape_dimensions: {Observation.dimensions(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{[0]} : Maybe<&2, List<&2, U32>>},
    word_count: {Observation.count(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{0} : Maybe<&2, U32>}}

def empty_metadata_expose(evidence: Records.MetadataContract<empty_shape(), 7.0>) -> EmptyMetadata:
  Records.MetadataContract{length, retained_dimensions, retained_count, retained_word_count, shape_dimensions, word_count,
    retained_value, shape_retained_value, count_retained_value} = evidence
  EmptyMetadata{length, shape_dimensions, word_count}

def empty_metadata() -> EmptyMetadata:
  empty_metadata_expose(Contract.metadata_contract(empty_shape(), 7.0))
