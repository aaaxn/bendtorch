import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./allocation-spec.bend as Allocation
import ./allocation-PROOF.bend as AllocationProof
import ./word-conversion.bend as Conversion
import ./observation.bend as Observation
import ./read-write-records.bend as Laws
import ./read-contract.bend as Reads

def write_value_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>, -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.written_value_pair(shape, other, Core.write_result(shape, original, valid, index, value, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, pair), other)) : Maybe<&2, F32>}:
  (storage, result) = pair
  Core.read_result_value(shape, storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>, (storage, result), Storage.set(original, index, value), same) :
        {Lifecycle.retained_write(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.set_length(original, index, value), valid),
    other, Storage.get(storage, other), {==})

def written_pick(+shape: Shape.Shape, +index: U32, +value: F32, +other: U32, +fill: F32) -> F32:
  Bool.pick(F32, Bool.and(Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))), Nat.is_eq(U32.to_nat(index), U32.to_nat(other))), value, fill)

def write_value_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.written_value(shape, index, value, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{written_pick(shape, index, value, other, fill)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Observation.written_value_pair(shape, other, Core.write_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==})),
    Lifecycle.read_value(Storage.get(StorageLaws.written(StorageLaws.created(Shape.count(shape), fill), index, value), other)), Some{written_pick(shape, index, value, other, fill)},
    write_value_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, other, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==}),
    Reads.stored_value(StorageLaws.written(StorageLaws.created(Shape.count(shape), fill), index, value), other, written_pick(shape, index, value, other, fill),
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.Write{index, value}], other, Conversion.shape_count_within(shape), Reads.word_inside(shape, other, inside))))

def write_code_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>, -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.write_code_pair(shape, Core.write_result(shape, original, valid, index, value, pair, same)) ==
    Observation.write_status(Core.storage_write_error(StorageLaws.outcome(Result<&2, &2, Storage.Error, Unit>, pair))) : U32}:
  (storage, result) = pair
  {==}

def write_code_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, -expected: Result<&2, &2, Storage.Error, Unit>,
  status: {StorageLaws.status(StorageLaws.created(Shape.count(shape), fill), index, value) == expected : Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.write_code(shape, index, value, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Observation.write_status(Core.storage_write_error(expected))} : Maybe<&2, U32>}:
  Equal.cong(U32, Maybe<&2, U32>, code => Some{code},
    Observation.write_code_pair(shape, Core.write_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==})),
    Observation.write_status(Core.storage_write_error(expected)),
    Equal.trans(U32, Observation.write_code_pair(shape, Core.write_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==})),
      Observation.write_status(Core.storage_write_error(StorageLaws.status(StorageLaws.created(Shape.count(shape), fill), index, value))), Observation.write_status(Core.storage_write_error(expected)),
      write_code_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==}),
      Equal.cong(Result<&2, &2, Storage.Error, Unit>, U32, result => Observation.write_status(Core.storage_write_error(result)), StorageLaws.status(StorageLaws.created(Shape.count(shape), fill), index, value), expected, status)))

def written_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  {Observation.written_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (tensor, result) = pair
  {==}

def written_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  {Observation.written_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (tensor, result) = pair
  {==}

def write_length_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>, -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.written_length_pair(shape, Core.write_result(shape, original, valid, index, value, pair, same)) == Lifecycle.retained_write(pair) : Nat}:
  (storage, result) = pair
  {==}

def write_length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32) ->
  {Observation.written_length(shape, index, value, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number},
    Observation.written_length_pair(shape, Core.write_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.written_length_pair(shape, Core.write_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==})),
      Lifecycle.retained_write(Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value)), Shape.logical_numel(Shape.dimensions(shape)),
      write_length_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value), {==}),
      Equal.trans(Nat, Lifecycle.retained_write(Storage.set(StorageLaws.created(Shape.count(shape), fill), index, value)), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
        Lifecycle.set_length(StorageLaws.created(Shape.count(shape), fill), index, value), Reads.created_exact(shape, fill))))

def pick_hit(+shape: Shape.Shape, +index: U32, +value: F32, +fill: F32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {written_pick(shape, index, value, index, fill) == value : F32}:
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))), True{}, Reads.word_inside(shape, index, inside)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(index), U32.to_nat(index))), value, fill) == value : F32}
  %Equal.sym(Bool, Nat.is_eq(U32.to_nat(index), U32.to_nat(index)), True{}, Arith.eq_refl(U32.to_nat(index))) :
    {Bool.pick(F32, _, value, fill) == value : F32}
  {==}

def pick_frame(+shape: Shape.Shape, +index: U32, +value: F32, +other: U32, +fill: F32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  different: {Nat.is_eq(U32.to_nat(index), U32.to_nat(other)) == False{} : Bool}) ->
  {written_pick(shape, index, value, other, fill) == fill : F32}:
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))), True{}, Reads.word_inside(shape, index, inside)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(index), U32.to_nat(other))), value, fill) == fill : F32}
  %Equal.sym(Bool, Nat.is_eq(U32.to_nat(index), U32.to_nat(other)), False{}, different) :
    {Bool.pick(F32, _, value, fill) == fill : F32}
  {==}

def pick_outside(+shape: Shape.Shape, +index: U32, +value: F32, +other: U32, +fill: F32, outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) ->
  {written_pick(shape, index, value, other, fill) == fill : F32}:
  %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))), False{}, Reads.word_outside(shape, index, outside)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(U32.to_nat(index), U32.to_nat(other))), value, fill) == fill : F32}
  {==}

def written_value(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32, +other: U32, -expected: F32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}, picked: {written_pick(shape, index, value, other, fill) == expected : F32}) ->
  {Observation.written_value(shape, index, value, other, Core.new(shape, fill)) == Some{expected} : Maybe<&2, F32>}:
  Reads.new_lift(shape, fill, result => {Observation.written_value(shape, index, value, other, result) == Some{expected} : Maybe<&2, F32>},
    valid => Equal.trans(Maybe<&2, F32>, Observation.written_value(shape, index, value, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}),
      Some{written_pick(shape, index, value, other, fill)}, Some{expected},
      write_value_done(shape, fill, valid, index, value, other, inside),
      Equal.cong(F32, Maybe<&2, F32>, number => Some{number}, written_pick(shape, index, value, other, fill), expected, picked)))

def write_contract(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32, -expected: Result<&2, &2, Storage.Error, Unit>,
  status: {StorageLaws.status(StorageLaws.created(Shape.count(shape), fill), index, value) == expected : Result<&2, &2, Storage.Error, Unit>}) ->
  Laws.WriteContract<shape, fill, index, value, Observation.write_status(Core.storage_write_error(expected))>:
  Laws.WriteContract{
    Reads.new_lift(shape, fill, result => {Observation.write_code(shape, index, value, result) == Some{Observation.write_status(Core.storage_write_error(expected))} : Maybe<&2, U32>},
      valid => write_code_done(shape, fill, valid, index, value, expected, status)),
    Reads.new_lift(shape, fill, result => {Observation.written_dimensions(shape, index, value, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => written_dimensions_pair(shape, Core.write_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, index, value))),
    Reads.new_lift(shape, fill, result => {Observation.written_length(shape, index, value, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => write_length_done(shape, fill, valid, index, value)),
    Reads.new_lift(shape, fill, result => {Observation.written_count(shape, index, value, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => written_count_pair(shape, Core.write_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, index, value)))}

def located_index(-shape: Shape.Shape, +index: Shape.FlatIndex<shape>, +expected: U32,
  located: {Observation.coordinate_index(shape, Done{index}) == Some{expected} : Maybe<&2, U32>}) ->
  {Shape.index_value(shape, index) == expected : U32}:
  Equal.cong(Maybe<&2, U32>, U32, word => Maybe.default(&2, U32, word, 0), Some{Shape.index_value(shape, index)}, Some{expected}, located)

def coordinate_read_flat(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  +index: U32, located: {Observation.coordinate_index(shape, result) == Some{index} : Maybe<&2, U32>}) ->
  {Core.read_coordinate_result(shape, tensor, result) == Core.read_flat(shape, tensor, index) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({(tensor, Fail{Core.ShapeError{error}}) == Core.read_flat(shape, tensor, index) : Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>},
        Arith.clash(Equal.cong(Maybe<&2, U32>, Bool, word => Maybe.is_some(&2, U32, word), None{}, Some{index}, located)))
    case Done{+flat}:
      Equal.cong(U32, Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, word => Core.read_flat(shape, tensor, word),
        Shape.index_value(shape, flat), index, located_index(shape, flat, index, located))

def coordinate_write_flat(+shape: Shape.Shape, -tensor: Core.Tensor<shape>, -value: F32,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>,
  +index: U32, located: {Observation.coordinate_index(shape, result) == Some{index} : Maybe<&2, U32>}) ->
  {Core.write_coordinate_result(shape, tensor, value, result) == Core.write_flat(shape, tensor, index, value) : Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>}:
  match result:
    case Fail{error}:
      Empty.absurd({(tensor, Fail{Core.ShapeError{error}}) == Core.write_flat(shape, tensor, index, value) : Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>},
        Arith.clash(Equal.cong(Maybe<&2, U32>, Bool, word => Maybe.is_some(&2, U32, word), None{}, Some{index}, located)))
    case Done{+flat}:
      Equal.cong(U32, Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, word => Core.write_flat(shape, tensor, word, value),
        Shape.index_value(shape, flat), index, located_index(shape, flat, index, located))

def coordinate_value_pair(+shape: Shape.Shape, +coordinates: List<&2, U32>, +index: U32,
  located: {Observation.coordinate_index(shape, Shape.flatten_checked(shape, coordinates)) == Some{index} : Maybe<&2, U32>},
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  {Observation.coordinate_value_pair(shape, coordinates, pair) == Observation.written_value_pair(shape, index, pair) : Maybe<&2, F32>}:
  (tensor, result) = pair
  Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>, Maybe<&2, F32>, read => Core.read_value(shape, read),
    Core.read_coordinate(shape, tensor, coordinates), Core.read_flat(shape, tensor, index),
    coordinate_read_flat(shape, tensor, Shape.flatten_checked(shape, coordinates), index, located))

def coordinate_value_result(+shape: Shape.Shape, +coordinates: List<&2, U32>, +index: U32, +value: F32,
  located: {Observation.coordinate_index(shape, Shape.flatten_checked(shape, coordinates)) == Some{index} : Maybe<&2, U32>},
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  {Observation.coordinate_value(shape, index, value, coordinates, result) == Observation.written_value(shape, index, value, index, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{tensor}:
      coordinate_value_pair(shape, coordinates, index, located, Core.write_flat(shape, tensor, index, value))

def coordinate_written_result(+shape: Shape.Shape, +coordinates: List<&2, U32>, +index: U32, +value: F32, +other: U32,
  located: {Observation.coordinate_index(shape, Shape.flatten_checked(shape, coordinates)) == Some{index} : Maybe<&2, U32>},
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  {Observation.coordinate_written(shape, coordinates, value, other, result) == Observation.written_value(shape, index, value, other, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{tensor}:
      Equal.cong(Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>, Maybe<&2, F32>, pair => Observation.written_value_pair(shape, other, pair),
        Core.write_coordinate(shape, tensor, coordinates, value), Core.write_flat(shape, tensor, index, value),
        coordinate_write_flat(shape, tensor, value, Shape.flatten_checked(shape, coordinates), index, located))

def write_inside(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Laws.WriteInside<shape, fill, index, value>:
  +inside = inside
  Laws.WriteInside{
    write_contract(shape, fill, index, value, Done{Unit{}},
      StorageLaws.replay_set(Shape.count(shape), fill, [], index, value, Conversion.shape_count_within(shape), Reads.word_inside(shape, index, inside))),
    written_value(shape, fill, index, value, index, value, inside, pick_hit(shape, index, value, fill, inside)),
    other => inside_other => different =>
      written_value(shape, fill, index, value, other, fill, inside_other, pick_frame(shape, index, value, other, fill, inside, different))}

def write_outside(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32,
  outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) -> Laws.WriteOutside<shape, fill, index, value>:
  +outside = outside
  Laws.WriteOutside{
    write_contract(shape, fill, index, value, Fail{Storage.OutOfBounds{}},
      StorageLaws.set_outside(StorageLaws.created(Shape.count(shape), fill), index, value, Reads.size_outside(shape, fill, index, outside))),
    other => inside =>
      written_value(shape, fill, index, value, other, fill, inside, pick_outside(shape, index, value, other, fill, outside))}

def coordinate_inside(+shape: Shape.Shape, +fill: F32, +coordinates: List<&2, U32>, +index: U32, +value: F32,
  located: {Observation.coordinate_index(shape, Shape.flatten_checked(shape, coordinates)) == Some{index} : Maybe<&2, U32>},
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  Laws.CoordinateInside<shape, fill, coordinates, index, value>:
  +located = located
  +inside = inside
  Laws.CoordinateInside{
    Equal.trans(Maybe<&2, F32>, Observation.coordinate_value(shape, index, value, coordinates, Core.new(shape, fill)),
      Observation.written_value(shape, index, value, index, Core.new(shape, fill)), Some{value},
      coordinate_value_result(shape, coordinates, index, value, located, Core.new(shape, fill)),
      written_value(shape, fill, index, value, index, value, inside, pick_hit(shape, index, value, fill, inside))),
    Equal.trans(Maybe<&2, F32>, Observation.coordinate_written(shape, coordinates, value, index, Core.new(shape, fill)),
      Observation.written_value(shape, index, value, index, Core.new(shape, fill)), Some{value},
      coordinate_written_result(shape, coordinates, index, value, index, located, Core.new(shape, fill)),
      written_value(shape, fill, index, value, index, value, inside, pick_hit(shape, index, value, fill, inside))),
    other => inside_other => different =>
      Equal.trans(Maybe<&2, F32>, Observation.coordinate_written(shape, coordinates, value, other, Core.new(shape, fill)),
        Observation.written_value(shape, index, value, other, Core.new(shape, fill)), Some{fill},
        coordinate_written_result(shape, coordinates, index, value, other, located, Core.new(shape, fill)),
        written_value(shape, fill, index, value, other, fill, inside_other, pick_frame(shape, index, value, other, fill, inside, different)))}
