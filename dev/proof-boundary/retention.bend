import Base
import ../../src/shape/core.bend as Shape
import ../../src/shape/nat_proofs.bend as Nats
import ../../src/storage/main.bend as Storage
import ../../src/storage/LAWS.bend as Laws
import ../../src/storage/api.bend as Api
import ../../src/storage/arith.bend as Arith
import ../../src/tensor/core.bend as Core
import ../../src/tensor/lifecycle.bend as Lifecycle
import ../../src/tensor/allocation-spec.bend as Allocation
import ../../src/tensor/allocation-PROOF.bend as AllocationProof
import ../../src/tensor/word-conversion.bend as Conversion
import ./observations.bend as Observation
import ./generic-read.bend as Reading
import ./generic-metadata.bend as Metadata

def size_is_length(storage: Storage.Storage) -> {Laws.size(storage) == Lifecycle.logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{depth, length, array, balanced, valid}:
      {==}

def retained_is_kept(pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) ->
  {Lifecycle.retained_read(pair) == Lifecycle.logical_length(Laws.owner(Result<&2, &2, Storage.Error, F32>, pair)) : Nat}:
  (storage, result) = pair
  {==}

def stored_after_value(+shape: Shape.Shape, +fill: F32, +index: U32, +other: U32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Lifecycle.read_value(Storage.get(Laws.kept(Laws.created(Shape.count(shape), fill), index), other)) == Some{fill} : Maybe<&2, F32>}:
  +within = Conversion.shape_count_within(shape)
  Equal.trans(Maybe<&2, F32>,
    Lifecycle.read_value(Storage.get(Laws.kept(Laws.created(Shape.count(shape), fill), index), other)),
    Reading.storage_value(Laws.read(Laws.kept(Laws.created(Shape.count(shape), fill), index), other)), Some{fill},
    Reading.storage_pair_value(Storage.get(Laws.kept(Laws.created(Shape.count(shape), fill), index), other)),
    Equal.cong(Result<&2, &2, Storage.Error, F32>, Maybe<&2, F32>, Reading.storage_value,
      Laws.read(Laws.kept(Laws.created(Shape.count(shape), fill), index), other), Done{fill},
      Api.new_read(U32.to_nat(Shape.count(shape)), fill, U32.is_le(Shape.count(shape), 1048576), [Laws.Read{index}], other,
        Equal.sym(Bool, U32.is_le(Shape.count(shape), 1048576), True{}, within), Api.cap_from(Shape.count(shape), within),
        Reading.word_inside(shape, other, inside))))

def stored_after_length(+shape: Shape.Shape, +fill: F32, +index: U32) ->
  {Lifecycle.retained_read(Storage.get(Laws.created(Shape.count(shape), fill), index)) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  +within = Conversion.shape_count_within(shape)
  Equal.trans(Nat, Lifecycle.retained_read(Storage.get(Laws.created(Shape.count(shape), fill), index)),
    Laws.size(Laws.kept(Laws.created(Shape.count(shape), fill), index)), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Lifecycle.retained_read(Storage.get(Laws.created(Shape.count(shape), fill), index)),
      Lifecycle.logical_length(Laws.kept(Laws.created(Shape.count(shape), fill), index)),
      Laws.size(Laws.kept(Laws.created(Shape.count(shape), fill), index)),
      retained_is_kept(Storage.get(Laws.created(Shape.count(shape), fill), index)),
      Equal.sym(Nat, Laws.size(Laws.kept(Laws.created(Shape.count(shape), fill), index)),
        Lifecycle.logical_length(Laws.kept(Laws.created(Shape.count(shape), fill), index)),
        size_is_length(Laws.kept(Laws.created(Shape.count(shape), fill), index)))),
    Equal.trans(Nat, Laws.size(Laws.kept(Laws.created(Shape.count(shape), fill), index)),
      U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)),
      Api.new_size(U32.to_nat(Shape.count(shape)), fill, U32.is_le(Shape.count(shape), 1048576), [Laws.Read{index}],
        Equal.sym(Bool, U32.is_le(Shape.count(shape), 1048576), True{}, within), Api.cap_from(Shape.count(shape), within)),
      Reading.count_exact(shape)))

def next_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +other: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.after_value_pair(shape, other, Core.read_result(shape, original, valid, index, pair, same)) ==
    Lifecycle.read_value(Storage.get(Laws.owner(Result<&2, &2, Storage.Error, F32>, pair), other)) : Maybe<&2, F32>}:
  (storage, result) = pair
  Core.read_result_value(shape, storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, (storage, result), Storage.get(original, index), same) :
        {Lifecycle.retained_read(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.get_length(original, index), valid),
    other, Storage.get(storage, other), {==})

def length_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.after_length_pair(shape, Core.read_result(shape, original, valid, index, pair, same)) == Lifecycle.retained_read(pair) : Nat}:
  (storage, result) = pair
  {==}

def attached_value(+shape: Shape.Shape, +fill: F32, +index: U32, +other: U32, +length: Nat,
  decision: Shape.BoolCase<Nat.is_eq(length, Shape.elements(shape))>, storage: Storage.Storage,
  -exact: {Lifecycle.logical_length(storage) == length : Nat}, counted: {length == Shape.elements(shape) : Nat},
  kept: {Lifecycle.read_value(Storage.get(Laws.kept(storage, index), other)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.after_value(shape, index, other, Core.attach_decided(shape, storage, length, exact, decision)) == Some{fill} : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd({None{} == Some{fill} : Maybe<&2, F32>}, Nats.contradiction(Nat.is_eq(length, Shape.elements(shape)),
        AllocationProof.natural_equal_true(length, Shape.elements(shape), counted), no))
    case Shape.BTrue{yes}:
      Equal.trans(Maybe<&2, F32>,
        Observation.after_value(shape, index, other, Core.attach_decided(shape, storage, length, exact, Shape.BTrue{yes})),
        Lifecycle.read_value(Storage.get(Laws.kept(storage, index), other)), Some{fill},
        next_projection(shape, storage,
          Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
            Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
              Lifecycle.natural_equal(length, Shape.elements(shape), yes), Shape.elements_exact(shape))),
          index, other, Storage.get(storage, index), {==}), kept)

def attach_value(+shape: Shape.Shape, +fill: F32, +index: U32, +other: U32, storage: Storage.Storage,
  counted: {Lifecycle.logical_length(storage) == Shape.elements(shape) : Nat},
  kept: {Lifecycle.read_value(Storage.get(Laws.kept(storage, index), other)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.after_value(shape, index, other, Core.attach(shape, storage)) == Some{fill} : Maybe<&2, F32>}:
  match storage:
    case Storage.Empty{}:
      attached_value(shape, fill, index, other, 0n, Shape.bool_case(Nat.is_eq(0n, Shape.elements(shape))), Storage.Empty{}, {==}, counted, kept)
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      attached_value(shape, fill, index, other, length, Shape.bool_case(Nat.is_eq(length, Shape.elements(shape))),
        Storage.Owned{depth, length, array, balanced, valid}, {==}, counted, kept)

def result_value(+shape: Shape.Shape, +fill: F32, +index: U32, +other: U32,
  result: Result<&2, &1, Storage.Error, Storage.Storage>,
  counted: {Allocation.storage_length(result) == Some{Shape.elements(shape)} : Maybe<&2, Nat>},
  kept: {Lifecycle.read_value(Storage.get(Laws.kept(Laws.created.of(result), index), other)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.after_value(shape, index, other, Core.new_result(shape, result)) == Some{fill} : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{fill} : Maybe<&2, F32>}, Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool,
        value => Maybe.is_some(&2, Nat, value), None{}, Some{Shape.elements(shape)}, counted)))
    case Done{storage}:
      attach_value(shape, fill, index, other, storage,
        Equal.cong(Maybe<&2, Nat>, Nat, value => Maybe.default(&2, Nat, value, 0n),
          Some{Lifecycle.logical_length(storage)}, Some{Shape.elements(shape)}, counted), kept)

def value(+shape: Shape.Shape, +fill: F32, +index: U32, +other: U32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.after_value(shape, index, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}:
  result_value(shape, fill, index, other, Storage.new(Shape.count(shape), fill),
    Equal.trans(Maybe<&2, Nat>, Allocation.storage_length(Storage.new(Shape.count(shape), fill)), Some{U32.to_nat(Shape.count(shape))}, Some{Shape.elements(shape)},
      AllocationProof.storage_new(Shape.count(shape), fill, Conversion.shape_count_within(shape)),
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape))),
    stored_after_value(shape, fill, index, other, inside))

def attached_length(+shape: Shape.Shape, +index: U32, +length: Nat,
  decision: Shape.BoolCase<Nat.is_eq(length, Shape.elements(shape))>, storage: Storage.Storage,
  -exact: {Lifecycle.logical_length(storage) == length : Nat}, counted: {length == Shape.elements(shape) : Nat},
  kept: {Lifecycle.retained_read(Storage.get(storage, index)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.after_length(shape, index, Core.attach_decided(shape, storage, length, exact, decision)) ==
    Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd({None{} == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
        Nats.contradiction(Nat.is_eq(length, Shape.elements(shape)), AllocationProof.natural_equal_true(length, Shape.elements(shape), counted), no))
    case Shape.BTrue{yes}:
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number},
        Observation.after_length_pair(shape, Core.read_flat(shape,
          Core.TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
            Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
              Lifecycle.natural_equal(length, Shape.elements(shape), yes), Shape.elements_exact(shape)))}, index)),
        Shape.logical_numel(Shape.dimensions(shape)),
        Equal.trans(Nat,
          Observation.after_length_pair(shape, Core.read_flat(shape,
            Core.TTensor{storage, Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
              Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
                Lifecycle.natural_equal(length, Shape.elements(shape), yes), Shape.elements_exact(shape)))}, index)),
          Lifecycle.retained_read(Storage.get(storage, index)), Shape.logical_numel(Shape.dimensions(shape)),
          length_projection(shape, storage,
            Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
              Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
                Lifecycle.natural_equal(length, Shape.elements(shape), yes), Shape.elements_exact(shape))),
            index, Storage.get(storage, index), {==}), kept))

def attach_length(+shape: Shape.Shape, +index: U32, storage: Storage.Storage,
  counted: {Lifecycle.logical_length(storage) == Shape.elements(shape) : Nat},
  kept: {Lifecycle.retained_read(Storage.get(storage, index)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.after_length(shape, index, Core.attach(shape, storage)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match storage:
    case Storage.Empty{}:
      attached_length(shape, index, 0n, Shape.bool_case(Nat.is_eq(0n, Shape.elements(shape))), Storage.Empty{}, {==}, counted, kept)
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      attached_length(shape, index, length, Shape.bool_case(Nat.is_eq(length, Shape.elements(shape))),
        Storage.Owned{depth, length, array, balanced, valid}, {==}, counted, kept)

def result_length(+shape: Shape.Shape, +index: U32, result: Result<&2, &1, Storage.Error, Storage.Storage>,
  counted: {Allocation.storage_length(result) == Some{Shape.elements(shape)} : Maybe<&2, Nat>},
  kept: {Lifecycle.retained_read(Storage.get(Laws.created.of(result), index)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.after_length(shape, index, Core.new_result(shape, result)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}, Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool,
        value => Maybe.is_some(&2, Nat, value), None{}, Some{Shape.elements(shape)}, counted)))
    case Done{storage}:
      attach_length(shape, index, storage,
        Equal.cong(Maybe<&2, Nat>, Nat, value => Maybe.default(&2, Nat, value, 0n),
          Some{Lifecycle.logical_length(storage)}, Some{Shape.elements(shape)}, counted), kept)

def length(+shape: Shape.Shape, +fill: F32, +index: U32) ->
  {Observation.after_length(shape, index, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  result_length(shape, index, Storage.new(Shape.count(shape), fill),
    Equal.trans(Maybe<&2, Nat>, Allocation.storage_length(Storage.new(Shape.count(shape), fill)), Some{U32.to_nat(Shape.count(shape))}, Some{Shape.elements(shape)},
      AllocationProof.storage_new(Shape.count(shape), fill, Conversion.shape_count_within(shape)),
      Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number}, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape))),
    stored_after_length(shape, fill, index))

def dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Observation.after_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (tensor, result) = pair
  {==}

def count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Observation.after_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (tensor, result) = pair
  {==}

def dimensions_result(+shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}) ->
  {Observation.after_dimensions(shape, index, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}, Arith.clash(successful))
    case Done{tensor}:
      dimensions_pair(shape, Core.read_flat(shape, tensor, index))

def count_result(+shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}) ->
  {Observation.after_count(shape, index, result) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.count(shape)} : Maybe<&2, U32>}, Arith.clash(successful))
    case Done{tensor}:
      count_pair(shape, Core.read_flat(shape, tensor, index))

type Retained<-shape: Shape.Shape, -fill: F32, -index: U32> is Type:
  Retained{
    count: {Observation.after_length(shape, index, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
    dimensions: {Observation.after_dimensions(shape, index, Core.new(shape, fill)) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    word_count: {Observation.after_count(shape, index, Core.new(shape, fill)) == Some{Shape.count(shape)} : Maybe<&2, U32>},
    readable: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Observation.after_value(shape, index, other, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

def construction_success(-shape: Shape.Shape, -fill: F32, evidence: Metadata.Construction<shape, fill>) ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool}:
  Metadata.Construction{successful, dimensions, count, word_count} = evidence
  successful

def produce(+shape: Shape.Shape, +fill: F32, +index: U32) -> Retained<shape, fill, index>:
  +successful = Metadata.construction(shape, fill)
  Retained{length(shape, fill, index),
    dimensions_result(shape, index, Core.new(shape, fill), construction_success(shape, fill, successful)),
    count_result(shape, index, Core.new(shape, fill), construction_success(shape, fill, successful)),
    other => inside => value(shape, fill, index, other, inside)}
