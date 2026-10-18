import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/api.bend as Api
import ../storage/PROOF.bend as StorageProof
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./allocation_spec.bend as Allocation
import ./allocation_LAWS.bend as Laws
import ./word_conversion.bend as Conversion
import ./word_conversion_PROOF.bend as ConversionProof

def allocate_length(+depth: Nat, +length: Nat, +fill: F32, valid: Bool,
  -decision: {valid == Storage.bounds(depth, length) : Bool}, accepted: {valid == True{} : Bool}) ->
  {Allocation.storage_length(Storage.allocate(depth, length, fill, valid, decision)) == Some{length} : Maybe<&2, Nat>}:
  match valid:
    case False{}:
      Empty.absurd({None{} == Some{length} : Maybe<&2, Nat>}, Arith.clash(accepted))
    case True{}:
      {==}

def allocate_at_length(+length: Nat, +fill: F32, metadata: Nat & Nat,
  valid: {Storage.bounds(Pair.fst(Nat, Nat, metadata), length) == True{} : Bool}) ->
  {Allocation.storage_length(Storage.allocate_at(length, fill, metadata)) == Some{length} : Maybe<&2, Nat>}:
  (+depth, capacity) = metadata
  allocate_length(depth, length, fill, Storage.bounds(depth, length), {==}, valid)

def new_checked_length(+length: Nat, +fill: F32, allowed: Bool, accepted: {allowed == True{} : Bool},
  limit: {Nat.is_le(length, Model.capacity(20n)) == True{} : Bool}) ->
  {Allocation.storage_length(Storage.new_checked(length, fill, allowed)) == Some{length} : Maybe<&2, Nat>}:
  match length allowed:
    case 0n False{}:
      {==}
    case 0n True{}:
      {==}
    case 1n+p False{}:
      Empty.absurd({None{} == Some{1n+p} : Maybe<&2, Nat>}, Arith.clash(accepted))
    case 1n+p True{}:
      allocate_at_length(1n+p, fill, Storage.depth_seek(20n, 1n+p, 0n, 1n, Nat.is_le(1n+p, 1n)), Api.seek(1n+p, limit, {==}))

def storage_new(+word: U32, +fill: F32, +within: {U32.is_le(word, 1048576) == True{} : Bool}) ->
  {Allocation.storage_length(Storage.new(word, fill)) == Some{U32.to_nat(word)} : Maybe<&2, Nat>}:
  new_checked_length(U32.to_nat(word), fill, U32.is_le(word, 1048576), within, Api.cap_from(word, within))

def storage_zero(+fill: F32) -> {Storage.new(0, fill) == Done{Storage.Empty{}} : Result<&2, &1, Storage.Error, Storage.Storage>}:
  {==}

def natural_equal_true(+a: Nat, +b: Nat, equal: {a == b : Nat}) -> {True{} == Nat.is_eq(a, b) : Bool}:
  Equal.sym(Bool, Nat.is_eq(a, b), True{},
    Equal.trans(Bool, Nat.is_eq(a, b), Nat.is_eq(b, b), True{},
      Equal.cong(Nat, Bool, n => Nat.is_eq(n, b), a, b, equal), Arith.eq_refl(b)))

def attach_decided_length(+shape: Shape.Shape, +length: Nat, decision: Shape.BoolCase<Nat.is_eq(length, Shape.elements(shape))>,
  storage: Storage.Storage, exact: {Lifecycle.logical_length(storage) == length : Nat}, accepted: {True{} == Nat.is_eq(length, Shape.elements(shape)) : Bool}) ->
  {Allocation.tensor_length(shape, Core.attach_decided(shape, storage, length, exact, decision)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd({None{} == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}, Nats.contradiction(Nat.is_eq(length, Shape.elements(shape)), accepted, equal))
    case Shape.BTrue{equal}:
      Equal.cong(Nat, Maybe<&2, Nat>, n => Some{n}, Lifecycle.logical_length(storage), Shape.logical_numel(Shape.dimensions(shape)),
        Equal.trans(Nat, Lifecycle.logical_length(storage), length, Shape.logical_numel(Shape.dimensions(shape)), exact,
          Equal.trans(Nat, length, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Lifecycle.natural_equal(length, Shape.elements(shape), equal), Shape.elements_exact(shape))))

def attach_length(+shape: Shape.Shape, storage: Storage.Storage, exact: {Lifecycle.logical_length(storage) == Shape.elements(shape) : Nat}) ->
  {Allocation.tensor_length(shape, Core.attach(shape, storage)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match storage:
    case Storage.Empty{}:
      attach_decided_length(shape, 0n, Shape.bool_case(Nat.is_eq(0n, Shape.elements(shape))), Storage.Empty{}, {==}, natural_equal_true(0n, Shape.elements(shape), exact))
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      attach_decided_length(shape, length, Shape.bool_case(Nat.is_eq(length, Shape.elements(shape))), Storage.Owned{depth, length, array, balanced, valid}, {==}, natural_equal_true(length, Shape.elements(shape), exact))

def new_result_length(+shape: Shape.Shape, result: Result<&2, &1, Storage.Error, Storage.Storage>,
  exact: {Allocation.storage_length(result) == Some{Shape.elements(shape)} : Maybe<&2, Nat>}) ->
  {Allocation.tensor_length(shape, Core.new_result(shape, result)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  match result:
    case Fail{error}:
      Empty.absurd({None{} == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
        Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool, value => Maybe.is_some(&2, Nat, value), None{}, Some{Shape.elements(shape)}, exact)))
    case Done{storage}:
      attach_length(shape, storage,
        Equal.cong(Maybe<&2, Nat>, Nat, value => Maybe.default(&2, Nat, value, 0n), Some{Lifecycle.logical_length(storage)}, Some{Shape.elements(shape)}, exact))

def tensor_done(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>,
  exact: {Allocation.tensor_length(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}) -> Allocation.TensorDone<shape, result>:
  match result:
    case Fail{error}:
      Empty.absurd(Allocation.TensorDone<shape, Fail{error}>,
        Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool, value => Maybe.is_some(&2, Nat, value), None{}, Some{Shape.logical_numel(Shape.dimensions(shape))}, exact)))
    case Done{tensor}:
      Allocation.TensorDone{tensor, {==}}

def core_new(+shape: Shape.Shape, +fill: F32) ->
  {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  new_result_length(shape, Storage.new(Shape.count(shape), fill),
    Equal.trans(Maybe<&2, Nat>, Allocation.storage_length(Storage.new(Shape.count(shape), fill)), Some{U32.to_nat(Shape.count(shape))}, Some{Shape.elements(shape)},
      storage_new(Shape.count(shape), fill, Conversion.shape_count_within(shape)),
      Equal.cong(Nat, Maybe<&2, Nat>, n => Some{n}, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape))))

def core_new_done(+shape: Shape.Shape, +fill: F32) -> Allocation.TensorDone<shape, Core.new(shape, fill)>:
  tensor_done(shape, Core.new(shape, fill), core_new(shape, fill))

def Laws.storage_new_complete(word, fill, within):
  storage_new(word, fill, within)

def Laws.storage_new_zero(fill):
  storage_zero(fill)

def Laws.storage_new_depth_bounds(word, within, positive):
  Api.seek(U32.to_nat(word), Api.cap_from(word, within),
    Equal.trans(Bool, Nat.is_lt(0n, U32.to_nat(word)), U32.is_lt(0, word), True{}, Arith.u32_lt(0, word), positive))

def Laws.storage_seek_bounds(fuel, length, depth, capacity, done, decided, sized, left, limit, positive):
  Api.seek_bounds(fuel, length, depth, capacity, done, decided, sized, left, limit, positive)

def Laws.core_new_complete(shape, fill):
  core_new(shape, fill)

def Laws.core_new_done(shape, fill):
  core_new_done(shape, fill)
