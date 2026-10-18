import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./word_conversion.bend as Conversion
import ./observation.bend as Observation
import ./read_contract.bend as Reads
import ./write_contract.bend as Writes
import ./clone_metadata_records.bend as Laws

def first_value_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_first_value_pair(shape, other, Core.clone_result(shape, original, valid, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Storage.Storage, pair), other)) : Maybe<&2, F32>}:
  (first, second) = pair
  Core.read_result_value(shape, first,
    Equal.trans(Nat, Lifecycle.logical_length(first), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.first_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_first_length(original), valid),
    other, Storage.get(first, other), {==})

def untouched_value(-shape: Shape.Shape, +other: U32, written: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  storage: Storage.Storage, -valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.untouched(shape, other, written, Core.TTensor{storage, valid}) == Lifecycle.read_value(Storage.get(storage, other)) : Maybe<&2, F32>}:
  (tensor, result) = written
  Core.read_result_value(shape, storage, valid, other, Storage.get(storage, other), {==})

def second_value_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_second_value_pair(shape, other, Core.clone_result(shape, original, valid, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.outcome(Storage.Storage, pair), other)) : Maybe<&2, F32>}:
  (first, second) = pair
  Core.read_result_value(shape, second,
    Equal.trans(Nat, Lifecycle.logical_length(second), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.second_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_second_length(original), valid),
    other, Storage.get(second, other), {==})

def first_written_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_written_first_pair(shape, index, value, other, Core.clone_result(shape, original, valid, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.written(StorageLaws.owner(Storage.Storage, pair), index, value), other)) : Maybe<&2, F32>}:
  (first, second) = pair
  Writes.write_value_projection(shape, first,
    Equal.trans(Nat, Lifecycle.logical_length(first), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.first_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_first_length(original), valid),
    index, value, other, Storage.set(first, index, value), {==})

def second_written_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32,
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_written_second_pair(shape, index, value, other, Core.clone_result(shape, original, valid, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.outcome(Storage.Storage, pair), other)) : Maybe<&2, F32>}:
  (first, second) = pair
  untouched_value(shape, other, Core.write_flat(shape, Core.TTensor{first, Equal.trans(Nat, Lifecycle.logical_length(first), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.first_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_first_length(original), valid)}, index, value), second,
    Equal.trans(Nat, Lifecycle.logical_length(second), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.second_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_second_length(original), valid))

def first_length_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_first_length_pair(shape, Core.clone_result(shape, original, valid, pair, same)) == Lifecycle.first_length(pair) : Nat}:
  (first, second) = pair
  {==}

def second_length_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage}) ->
  {Observation.clone_second_length_pair(shape, Core.clone_result(shape, original, valid, pair, same)) == Lifecycle.second_length(pair) : Nat}:
  (first, second) = pair
  {==}

def first_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) ->
  {Observation.clone_first_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (first, second) = pair
  {==}

def second_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) ->
  {Observation.clone_second_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (first, second) = pair
  {==}

def first_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) ->
  {Observation.clone_first_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (first, second) = pair
  {==}

def second_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) ->
  {Observation.clone_second_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (first, second) = pair
  {==}

def first_value_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.clone_first_value(shape, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.clone_first_value_pair(shape, other, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})),
    Lifecycle.read_value(Storage.get(StorageLaws.first(StorageLaws.created(Shape.count(shape), fill)), other)), Some{fill},
    first_value_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, other, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
    Reads.stored_value(StorageLaws.first(StorageLaws.created(Shape.count(shape), fill)), other, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.First{}], other, Conversion.shape_count_within(shape), Reads.word_inside(shape, other, inside))))

def second_value_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.clone_second_value(shape, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.clone_second_value_pair(shape, other, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})),
    Lifecycle.read_value(Storage.get(StorageLaws.second(StorageLaws.created(Shape.count(shape), fill)), other)), Some{fill},
    second_value_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, other, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
    Reads.stored_value(StorageLaws.second(StorageLaws.created(Shape.count(shape), fill)), other, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.Second{}], other, Conversion.shape_count_within(shape), Reads.word_inside(shape, other, inside))))

def first_written_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.clone_written_first(shape, index, value, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Writes.written_pick(shape, index, value, other, fill)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.clone_written_first_pair(shape, index, value, other, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})),
    Lifecycle.read_value(Storage.get(StorageLaws.written(StorageLaws.first(StorageLaws.created(Shape.count(shape), fill)), index, value), other)),
    Some{Writes.written_pick(shape, index, value, other, fill)},
    first_written_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, other, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
    Reads.stored_value(StorageLaws.written(StorageLaws.first(StorageLaws.created(Shape.count(shape), fill)), index, value), other, Writes.written_pick(shape, index, value, other, fill),
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.First{}, StorageLaws.Write{index, value}], other, Conversion.shape_count_within(shape),
        Reads.word_inside(shape, other, inside))))

def second_written_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.clone_written_second(shape, index, value, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.clone_written_second_pair(shape, index, value, other, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})),
    Lifecycle.read_value(Storage.get(StorageLaws.second(StorageLaws.created(Shape.count(shape), fill)), other)), Some{fill},
    second_written_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, value, other, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
    Reads.stored_value(StorageLaws.second(StorageLaws.created(Shape.count(shape), fill)), other, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.Second{}], other, Conversion.shape_count_within(shape), Reads.word_inside(shape, other, inside))))

def first_length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.clone_first_length(shape, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Observation.clone_first_length_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.clone_first_length_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Lifecycle.first_length(Storage.clone(StorageLaws.created(Shape.count(shape), fill))), Shape.logical_numel(Shape.dimensions(shape)),
      first_length_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
      Equal.trans(Nat, Lifecycle.first_length(Storage.clone(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
        Lifecycle.clone_first_length(StorageLaws.created(Shape.count(shape), fill)), Reads.created_exact(shape, fill))))

def second_length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.clone_second_length(shape, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Observation.clone_second_length_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.clone_second_length_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Lifecycle.second_length(Storage.clone(StorageLaws.created(Shape.count(shape), fill))), Shape.logical_numel(Shape.dimensions(shape)),
      second_length_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, Storage.clone(StorageLaws.created(Shape.count(shape), fill)), {==}),
      Equal.trans(Nat, Lifecycle.second_length(Storage.clone(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
        Lifecycle.clone_second_length(StorageLaws.created(Shape.count(shape), fill)), Reads.created_exact(shape, fill))))

def clone_written(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32, +other: U32, -expected: F32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}, picked: {Writes.written_pick(shape, index, value, other, fill) == expected : F32}) ->
  {Observation.clone_written_first(shape, index, value, other, Core.new(shape, fill)) == Some{expected} : Maybe<&2, F32>}:
  Reads.new_lift(shape, fill, result => {Observation.clone_written_first(shape, index, value, other, result) == Some{expected} : Maybe<&2, F32>},
    valid => Equal.trans(Maybe<&2, F32>, Observation.clone_written_first(shape, index, value, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}),
      Some{Writes.written_pick(shape, index, value, other, fill)}, Some{expected},
      first_written_done(shape, fill, valid, index, value, other, inside),
      Equal.cong(F32, Maybe<&2, F32>, number => Some{number}, Writes.written_pick(shape, index, value, other, fill), expected, picked)))

def clone_contract(+shape: Shape.Shape, +fill: F32) -> Laws.CloneContract<shape, fill>:
  Laws.CloneContract{
    Reads.new_lift(shape, fill, result => {Observation.clone_first_dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => first_dimensions_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    Reads.new_lift(shape, fill, result => {Observation.clone_second_dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => second_dimensions_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    Reads.new_lift(shape, fill, result => {Observation.clone_first_length(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => first_length_done(shape, fill, valid)),
    Reads.new_lift(shape, fill, result => {Observation.clone_second_length(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => second_length_done(shape, fill, valid)),
    Reads.new_lift(shape, fill, result => {Observation.clone_first_count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => first_count_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    Reads.new_lift(shape, fill, result => {Observation.clone_second_count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => second_count_pair(shape, Core.clone(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    other => inside => Reads.new_lift(shape, fill, result => {Observation.clone_first_value(shape, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => first_value_done(shape, fill, valid, other, inside)),
    other => inside => Reads.new_lift(shape, fill, result => {Observation.clone_second_value(shape, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => second_value_done(shape, fill, valid, other, inside))}

def clone_write(+shape: Shape.Shape, +fill: F32, +index: U32, +value: F32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Laws.CloneWrite<shape, fill, index, value>:
  +inside = inside
  Laws.CloneWrite{
    clone_written(shape, fill, index, value, index, value, inside, Writes.pick_hit(shape, index, value, fill, inside)),
    other => inside_other => different =>
      clone_written(shape, fill, index, value, other, fill, inside_other, Writes.pick_frame(shape, index, value, other, fill, inside, different)),
    other => inside_other => Reads.new_lift(shape, fill, result => {Observation.clone_written_second(shape, index, value, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => second_written_done(shape, fill, valid, index, value, other, inside_other))}
