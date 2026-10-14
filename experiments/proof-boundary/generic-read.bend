import Base
import ../../src/shape/core.bend as Shape
import ../../src/shape/nat_proofs.bend as Nats
import ../../src/storage/main.bend as Storage
import ../../src/storage/LAWS.bend as StorageLaws
import ../../src/storage/api.bend as StorageApi
import ../../src/storage/arith.bend as Arith
import ../../src/tensor/core.bend as Core
import ../../src/tensor/lifecycle.bend as Lifecycle
import ../../src/tensor/allocation-spec.bend as Allocation
import ../../src/tensor/allocation-PROOF.bend as AllocationProof
import ../../src/tensor/word-conversion.bend as Conversion
import ./observations.bend as Observation
import ./generic-metadata.bend as Metadata

def storage_value(result: Result<&2, &2, Storage.Error, F32>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{value}:
      Some{value}

def storage_pair_value(pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) ->
  {Lifecycle.read_value(pair) == storage_value(StorageLaws.outcome(Result<&2, &2, Storage.Error, F32>, pair)) : Maybe<&2, F32>}:
  (storage, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{value}:
      {==}

def count_exact(+shape: Shape.Shape) ->
  {U32.to_nat(Shape.count(shape)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
    Conversion.shape_count_exact(shape), Shape.elements_exact(shape))

def word_inside(+shape: Shape.Shape, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), count_exact(shape)) :
    {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def storage_new_value(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Lifecycle.read_value(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)) == Some{fill} : Maybe<&2, F32>}:
  +within = Conversion.shape_count_within(shape)
  Equal.trans(Maybe<&2, F32>, Lifecycle.read_value(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)),
    storage_value(StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index)), Some{fill},
    storage_pair_value(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)),
    Equal.cong(Result<&2, &2, Storage.Error, F32>, Maybe<&2, F32>, storage_value,
      StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index), Done{fill},
      StorageApi.new_read(U32.to_nat(Shape.count(shape)), fill, U32.is_le(Shape.count(shape), 1048576), [], index,
        Equal.sym(Bool, U32.is_le(Shape.count(shape), 1048576), True{}, within),
        StorageApi.cap_from(Shape.count(shape), within), word_inside(shape, index, inside))))

def attached_point(+shape: Shape.Shape, +fill: F32, +index: U32, +length: Nat,
  decision: Shape.BoolCase<Nat.is_eq(length, Shape.elements(shape))>, storage: Storage.Storage,
  -exact: {Lifecycle.logical_length(storage) == length : Nat},
  counted: {length == Shape.elements(shape) : Nat},
  value: {Lifecycle.read_value(Storage.get(storage, index)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.point(shape, index, Core.attach_decided(shape, storage, length, exact, decision)) == Some{fill} : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd({None{} == Some{fill} : Maybe<&2, F32>},
        Nats.contradiction(Nat.is_eq(length, Shape.elements(shape)),
          AllocationProof.natural_equal_true(length, Shape.elements(shape), counted), no))
    case Shape.BTrue{yes}:
      Equal.trans(Maybe<&2, F32>,
        Observation.point(shape, index, Core.attach_decided(shape, storage, length, exact, Shape.BTrue{yes})),
        Lifecycle.read_value(Storage.get(storage, index)), Some{fill},
        Core.read_result_value(shape, storage,
          Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
            Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)),
              Lifecycle.natural_equal(length, Shape.elements(shape), yes), Shape.elements_exact(shape))),
          index, Storage.get(storage, index), {==}), value)

def attach_point(+shape: Shape.Shape, +fill: F32, +index: U32, storage: Storage.Storage,
  counted: {Lifecycle.logical_length(storage) == Shape.elements(shape) : Nat},
  value: {Lifecycle.read_value(Storage.get(storage, index)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.point(shape, index, Core.attach(shape, storage)) == Some{fill} : Maybe<&2, F32>}:
  match storage:
    case Storage.Empty{}:
      attached_point(shape, fill, index, 0n, Shape.bool_case(Nat.is_eq(0n, Shape.elements(shape))),
        Storage.Empty{}, {==}, counted, value)
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      attached_point(shape, fill, index, length, Shape.bool_case(Nat.is_eq(length, Shape.elements(shape))),
        Storage.Owned{depth, length, array, balanced, valid}, {==}, counted, value)

def result_point(+shape: Shape.Shape, +fill: F32, +index: U32,
  result: Result<&2, &1, Storage.Error, Storage.Storage>,
  counted: {Allocation.storage_length(result) == Some{Shape.elements(shape)} : Maybe<&2, Nat>},
  value: {Lifecycle.read_value(Storage.get(StorageLaws.created.of(result), index)) == Some{fill} : Maybe<&2, F32>}) ->
  {Observation.point(shape, index, Core.new_result(shape, result)) == Some{fill} : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{fill} : Maybe<&2, F32>},
        Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool, value => Maybe.is_some(&2, Nat, value),
          None{}, Some{Shape.elements(shape)}, counted)))
    case Done{storage}:
      attach_point(shape, fill, index, storage,
        Equal.cong(Maybe<&2, Nat>, Nat, value => Maybe.default(&2, Nat, value, 0n),
          Some{Lifecycle.logical_length(storage)}, Some{Shape.elements(shape)}, counted), value)

type Read<-shape: Shape.Shape, -fill: F32, -index: U32> is Data:
  Read{construction: Metadata.Construction<shape, fill>,
    value: {Observation.point(shape, index, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}}

def produce(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Read<shape, fill, index>:
  Read{Metadata.construction(shape, fill),
    result_point(shape, fill, index, Storage.new(Shape.count(shape), fill),
      Equal.trans(Maybe<&2, Nat>, Allocation.storage_length(Storage.new(Shape.count(shape), fill)),
        Some{U32.to_nat(Shape.count(shape))}, Some{Shape.elements(shape)},
        AllocationProof.storage_new(Shape.count(shape), fill, Conversion.shape_count_within(shape)),
        Equal.cong(Nat, Maybe<&2, Nat>, value => Some{value}, U32.to_nat(Shape.count(shape)), Shape.elements(shape),
          Conversion.shape_count_exact(shape))),
      storage_new_value(shape, fill, index, inside))}
