import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./word-conversion.bend as Conversion
import ./observation.bend as Observation
import ./read-contract.bend as Reads
import ./clone-metadata-records.bend as Laws

def owner_value_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32,
  pair: Storage.Storage & Nat, -same: {pair == Storage.length(original) : Storage.Storage & Nat}) ->
  {Observation.length_owner_value_pair(shape, other, Core.length_result(shape, original, valid, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Nat, pair), other)) : Maybe<&2, F32>}:
  (storage, number) = pair
  Core.read_result_value(shape, storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Nat, (storage, number), Storage.length(original), same) :
        {Lifecycle.length_owner(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.length_preserved(original), valid),
    other, Storage.get(storage, other), {==})

def number_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, pair: Storage.Storage & Nat, -same: {pair == Storage.length(original) : Storage.Storage & Nat}) ->
  {Observation.length_number(shape, Core.length_result(shape, original, valid, pair, same)) == Lifecycle.length_value(pair) : Nat}:
  (storage, number) = pair
  {==}

def owner_length_projection(-shape: Shape.Shape, -original: Storage.Storage, -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, pair: Storage.Storage & Nat, -same: {pair == Storage.length(original) : Storage.Storage & Nat}) ->
  {Observation.length_owner_length_pair(shape, Core.length_result(shape, original, valid, pair, same)) == Lifecycle.length_owner(pair) : Nat}:
  (storage, number) = pair
  {==}

def owner_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) ->
  {Observation.length_owner_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (tensor, number) = pair
  {==}

def owner_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) ->
  {Observation.length_owner_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (tensor, number) = pair
  {==}

def length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.length_value(shape, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Observation.length_number(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.length_number(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Lifecycle.length_value(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Shape.logical_numel(Shape.dimensions(shape)),
      number_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, Storage.length(StorageLaws.created(Shape.count(shape), fill)), {==}),
      Equal.trans(Nat, Lifecycle.length_value(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.length_owner(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Shape.logical_numel(Shape.dimensions(shape)),
        Equal.sym(Nat, Lifecycle.length_owner(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.length_value(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.length_exact(StorageLaws.created(Shape.count(shape), fill))),
        Equal.trans(Nat, Lifecycle.length_owner(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
          Lifecycle.length_preserved(StorageLaws.created(Shape.count(shape), fill)), Reads.created_exact(shape, fill)))))

def owner_length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.length_owner_length(shape, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, Observation.length_owner_length_pair(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.length_owner_length_pair(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})), Lifecycle.length_owner(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Shape.logical_numel(Shape.dimensions(shape)),
      owner_length_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, Storage.length(StorageLaws.created(Shape.count(shape), fill)), {==}),
      Equal.trans(Nat, Lifecycle.length_owner(Storage.length(StorageLaws.created(Shape.count(shape), fill))), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
        Lifecycle.length_preserved(StorageLaws.created(Shape.count(shape), fill)), Reads.created_exact(shape, fill))))

def owner_value_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.length_owner_value(shape, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.length_owner_value_pair(shape, other, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid})),
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Nat, Storage.length(StorageLaws.created(Shape.count(shape), fill))), other)), Some{fill},
    owner_value_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, other, Storage.length(StorageLaws.created(Shape.count(shape), fill)), {==}),
    Reads.stored_value(StorageLaws.owner(Nat, Storage.length(StorageLaws.created(Shape.count(shape), fill))), other, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.Length{}], other, Conversion.shape_count_within(shape), Reads.word_inside(shape, other, inside))))

def dynamic_value_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.dynamic_owner_value(other, Core.Packed{shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.dynamic_owner_value(other, Core.Packed{shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}),
    Core.read_value(shape, Core.read_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, other)), Some{fill},
    Core.dynamic_result_value(shape, Core.read_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, other)),
    Reads.read_point_done(shape, fill, valid, other, inside))

def metadata_contract(+shape: Shape.Shape, +fill: F32) -> Laws.MetadataContract<shape, fill>:
  Laws.MetadataContract{
    Reads.new_lift(shape, fill, result => {Observation.length_value(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => length_done(shape, fill, valid)),
    Reads.new_lift(shape, fill, result => {Observation.length_owner_dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => owner_dimensions_pair(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    Reads.new_lift(shape, fill, result => {Observation.length_owner_length(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => owner_length_done(shape, fill, valid)),
    Reads.new_lift(shape, fill, result => {Observation.length_owner_count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => owner_count_pair(shape, Core.length(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}))),
    Reads.new_lift(shape, fill, result => {Observation.dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => {==}),
    Reads.new_lift(shape, fill, result => {Observation.count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => {==}),
    other => inside => Reads.new_lift(shape, fill, result => {Observation.length_owner_value(shape, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => owner_value_done(shape, fill, valid, other, inside)),
    other => inside => Reads.new_lift(shape, fill, result => {Observation.shape_owner_value(shape, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => dynamic_value_done(shape, fill, valid, other, inside)),
    other => inside => Reads.new_lift(shape, fill, result => {Observation.count_owner_value(shape, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => dynamic_value_done(shape, fill, valid, other, inside))}
