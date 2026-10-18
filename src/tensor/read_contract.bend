import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./allocation_spec.bend as Allocation
import ./allocation_PROOF.bend as AllocationProof
import ./word_conversion.bend as Conversion
import ./observation.bend as Observation
import ./read_write_records.bend as Laws

def count_exact(+shape: Shape.Shape) -> {U32.to_nat(Shape.count(shape)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Conversion.shape_count_exact(shape), Shape.elements_exact(shape))

def word_inside(+shape: Shape.Shape, +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), count_exact(shape)) : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def word_outside(+shape: Shape.Shape, +index: U32, outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == False{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), count_exact(shape)) : {Nat.is_lt(U32.to_nat(index), _) == False{} : Bool}
  outside

def created_size(+shape: Shape.Shape, +fill: F32) -> {StorageLaws.size(StorageLaws.created(Shape.count(shape), fill)) == U32.to_nat(Shape.count(shape)) : Nat}:
  StorageLaws.replay_size(Shape.count(shape), fill, [], Conversion.shape_count_within(shape))

def size_is_length(storage: Storage.Storage) -> {StorageLaws.size(storage) == Lifecycle.logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{depth, length, array, balanced, valid}:
      {==}

def created_exact(+shape: Shape.Shape, +fill: F32) -> {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), StorageLaws.size(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.sym(Nat, StorageLaws.size(StorageLaws.created(Shape.count(shape), fill)), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), size_is_length(StorageLaws.created(Shape.count(shape), fill))),
    Equal.trans(Nat, StorageLaws.size(StorageLaws.created(Shape.count(shape), fill)), U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), created_size(shape, fill), count_exact(shape)))

def size_outside(+shape: Shape.Shape, +fill: F32, +index: U32, outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), StorageLaws.size(StorageLaws.created(Shape.count(shape), fill))) == False{} : Bool}:
  %Equal.sym(Nat, StorageLaws.size(StorageLaws.created(Shape.count(shape), fill)), U32.to_nat(Shape.count(shape)), created_size(shape, fill)) : {Nat.is_lt(U32.to_nat(index), _) == False{} : Bool}
  word_outside(shape, index, outside)

def attached(+shape: Shape.Shape, -storage: Storage.Storage, +n: Nat, -exact: {Lifecycle.logical_length(storage) == n : Nat},
  decision: Shape.BoolCase<Nat.is_eq(n, Shape.elements(shape))>, counted: {n == Shape.elements(shape) : Nat},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  h: @-valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> P(Done{Core.TTensor{storage, valid}})) ->
  P(Core.attach_decided(shape, storage, n, exact, decision)):
  match decision:
    case Shape.BFalse{no}:
      Empty.absurd(P(Core.attach_decided(shape, storage, n, exact, Shape.BFalse{no})),
        Nats.contradiction(Nat.is_eq(n, Shape.elements(shape)), AllocationProof.natural_equal_true(n, Shape.elements(shape), counted), no))
    case Shape.BTrue{yes}:
      h(Equal.trans(Nat, Lifecycle.logical_length(storage), n, Shape.logical_numel(Shape.dimensions(shape)), exact,
        Equal.trans(Nat, n, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Lifecycle.natural_equal(n, Shape.elements(shape), yes), Shape.elements_exact(shape))))

def attach_lift(+shape: Shape.Shape, storage: Storage.Storage, counted: {Lifecycle.logical_length(storage) == Shape.elements(shape) : Nat},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  h: @-valid: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> P(Done{Core.TTensor{storage, valid}})) ->
  P(Core.attach(shape, storage)):
  match storage:
    case Storage.Empty{}:
      attached(shape, Storage.Empty{}, 0n, {==}, Shape.bool_case(Nat.is_eq(0n, Shape.elements(shape))), counted, P, h)
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      attached(shape, Storage.Owned{depth, length, array, balanced, valid}, length, {==},
        Shape.bool_case(Nat.is_eq(length, Shape.elements(shape))), counted, P, h)

def result_lift(+shape: Shape.Shape, result: Result<&2, &1, Storage.Error, Storage.Storage>,
  counted: {Allocation.storage_length(result) == Some{Shape.elements(shape)} : Maybe<&2, Nat>},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  h: @-valid: {Lifecycle.logical_length(StorageLaws.created.of(result)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Done{Core.TTensor{StorageLaws.created.of(result), valid}})) ->
  P(Core.new_result(shape, result)):
  match result:
    case Fail{error}:
      Empty.absurd(P(Core.new_result(shape, Fail{error})),
        Arith.clash(Equal.cong(Maybe<&2, Nat>, Bool, value => Maybe.is_some(&2, Nat, value), None{}, Some{Shape.elements(shape)}, counted)))
    case Done{storage}:
      attach_lift(shape, storage,
        Equal.cong(Maybe<&2, Nat>, Nat, value => Maybe.default(&2, Nat, value, 0n),
          Some{Lifecycle.logical_length(storage)}, Some{Shape.elements(shape)}, counted), P, h)

def new_lift(+shape: Shape.Shape, +fill: F32, -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  h: @-valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> P(Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}})) -> P(Core.new(shape, fill)):
  result_lift(shape, Storage.new(Shape.count(shape), fill),
    Equal.trans(Maybe<&2, Nat>, Allocation.storage_length(Storage.new(Shape.count(shape), fill)), Some{U32.to_nat(Shape.count(shape))}, Some{Shape.elements(shape)},
      AllocationProof.storage_new(Shape.count(shape), fill, Conversion.shape_count_within(shape)),
      Equal.cong(Nat, Maybe<&2, Nat>, value => Some{value}, U32.to_nat(Shape.count(shape)), Shape.elements(shape), Conversion.shape_count_exact(shape))),
    P, h)

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

def stored_value(storage: Storage.Storage, +other: U32, -expected: F32,
  read: {StorageLaws.read(storage, other) == Done{expected} : Result<&2, &2, Storage.Error, F32>}) ->
  {Lifecycle.read_value(Storage.get(storage, other)) == Some{expected} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Lifecycle.read_value(Storage.get(storage, other)), storage_value(StorageLaws.read(storage, other)), Some{expected},
    storage_pair_value(Storage.get(storage, other)),
    Equal.cong(Result<&2, &2, Storage.Error, F32>, Maybe<&2, F32>, storage_value, StorageLaws.read(storage, other), Done{expected}, read))

def read_code_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>, -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.read_code_pair(shape, Core.read_result(shape, original, valid, index, pair, same)) ==
    Observation.read_status(Core.storage_read_error(StorageLaws.outcome(Result<&2, &2, Storage.Error, F32>, pair))) : U32}:
  (storage, result) = pair
  {==}

def read_code_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, -expected: Result<&2, &2, Storage.Error, F32>,
  read: {StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index) == expected : Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.read_code(shape, index, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Observation.read_status(Core.storage_read_error(expected))} : Maybe<&2, U32>}:
  Equal.cong(U32, Maybe<&2, U32>, code => Some{code},
    Observation.read_code_pair(shape, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})),
    Observation.read_status(Core.storage_read_error(expected)),
    Equal.trans(U32, Observation.read_code_pair(shape, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})),
      Observation.read_status(Core.storage_read_error(StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index))), Observation.read_status(Core.storage_read_error(expected)),
      read_code_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==}),
      Equal.cong(Result<&2, &2, Storage.Error, F32>, U32, result => Observation.read_status(Core.storage_read_error(result)), StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index), expected, read)))

def read_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Observation.after_dimensions_pair(shape, pair) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}:
  (tensor, result) = pair
  {==}

def read_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Observation.after_count_pair(shape, pair) == Some{Shape.count(shape)} : Maybe<&2, U32>}:
  (tensor, result) = pair
  {==}

def read_length_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>, -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.after_length_pair(shape, Core.read_result(shape, original, valid, index, pair, same)) == Lifecycle.retained_read(pair) : Nat}:
  (storage, result) = pair
  {==}

def read_length_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32) ->
  {Observation.after_length(shape, index, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}:
  Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number},
    Observation.after_length_pair(shape, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.trans(Nat, Observation.after_length_pair(shape, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})),
      Lifecycle.retained_read(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)), Shape.logical_numel(Shape.dimensions(shape)),
      read_length_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==}),
      Equal.trans(Nat, Lifecycle.retained_read(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)), Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)), Shape.logical_numel(Shape.dimensions(shape)),
        Lifecycle.get_length(StorageLaws.created(Shape.count(shape), fill), index), created_exact(shape, fill))))

def read_value_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +other: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>, -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {Observation.after_value_pair(shape, other, Core.read_result(shape, original, valid, index, pair, same)) ==
    Lifecycle.read_value(Storage.get(StorageLaws.owner(Result<&2, &2, Storage.Error, F32>, pair), other)) : Maybe<&2, F32>}:
  (storage, result) = pair
  Core.read_result_value(shape, storage,
    Equal.trans(Nat, Lifecycle.logical_length(storage), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, F32>, (storage, result), Storage.get(original, index), same) :
        {Lifecycle.retained_read(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.get_length(original, index), valid),
    other, Storage.get(storage, other), {==})

def read_after_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.after_value(shape, index, other, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Observation.after_value_pair(shape, other, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})),
    Lifecycle.read_value(Storage.get(StorageLaws.kept(StorageLaws.created(Shape.count(shape), fill), index), other)), Some{fill},
    read_value_projection(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, other, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==}),
    stored_value(StorageLaws.kept(StorageLaws.created(Shape.count(shape), fill), index), other, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [StorageLaws.Read{index}], other, Conversion.shape_count_within(shape), word_inside(shape, other, inside))))

def read_point_done(+shape: Shape.Shape, +fill: F32, -valid: {Lifecycle.logical_length(StorageLaws.created(Shape.count(shape), fill)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.point(shape, index, Done{Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}}) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Core.read_value(shape, Core.read_result(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==})),
    Lifecycle.read_value(Storage.get(StorageLaws.created(Shape.count(shape), fill), index)), Some{fill},
    Core.read_result_value(shape, StorageLaws.created(Shape.count(shape), fill), valid, index, Storage.get(StorageLaws.created(Shape.count(shape), fill), index), {==}),
    stored_value(StorageLaws.created(Shape.count(shape), fill), index, fill,
      StorageLaws.replay_get(Shape.count(shape), fill, [], index, Conversion.shape_count_within(shape), word_inside(shape, index, inside))))

def read_contract(+shape: Shape.Shape, +fill: F32, +index: U32, -expected: Result<&2, &2, Storage.Error, F32>,
  read: {StorageLaws.read(StorageLaws.created(Shape.count(shape), fill), index) == expected : Result<&2, &2, Storage.Error, F32>}) ->
  Laws.ReadContract<shape, fill, index, Observation.read_status(Core.storage_read_error(expected))>:
  Laws.ReadContract{
    new_lift(shape, fill, result => {Observation.read_code(shape, index, result) == Some{Observation.read_status(Core.storage_read_error(expected))} : Maybe<&2, U32>},
      valid => read_code_done(shape, fill, valid, index, expected, read)),
    new_lift(shape, fill, result => {Observation.after_dimensions(shape, index, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      valid => read_dimensions_pair(shape, Core.read_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, index))),
    new_lift(shape, fill, result => {Observation.after_length(shape, index, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      valid => read_length_done(shape, fill, valid, index)),
    new_lift(shape, fill, result => {Observation.after_count(shape, index, result) == Some{Shape.count(shape)} : Maybe<&2, U32>},
      valid => read_count_pair(shape, Core.read_flat(shape, Core.TTensor{StorageLaws.created(Shape.count(shape), fill), valid}, index))),
    other => inside => new_lift(shape, fill, result => {Observation.after_value(shape, index, other, result) == Some{fill} : Maybe<&2, F32>},
      valid => read_after_done(shape, fill, valid, index, other, inside))}

def dynamic_point_result(+shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  {Observation.dynamic_point(shape, index, result) == Observation.point(shape, index, result) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{tensor}:
      Core.dynamic_result_value(shape, Core.read_flat(shape, tensor, index))

def read_inside(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) -> Laws.ReadInside<shape, fill, index>:
  +inside = inside
  Laws.ReadInside{
    read_contract(shape, fill, index, Done{fill},
      StorageLaws.replay_get(Shape.count(shape), fill, [], index, Conversion.shape_count_within(shape), word_inside(shape, index, inside))),
    new_lift(shape, fill, result => {Observation.point(shape, index, result) == Some{fill} : Maybe<&2, F32>},
      valid => read_point_done(shape, fill, valid, index, inside))}

def read_outside(+shape: Shape.Shape, +fill: F32, +index: U32,
  outside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == False{} : Bool}) -> Laws.ReadContract<shape, fill, index, 8>:
  read_contract(shape, fill, index, Fail{Storage.OutOfBounds{}},
    StorageLaws.get_outside(StorageLaws.created(Shape.count(shape), fill), index, size_outside(shape, fill, index, outside)))

def dynamic_read_inside(+shape: Shape.Shape, +fill: F32, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Observation.dynamic_point(shape, index, Core.new(shape, fill)) == Some{fill} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Observation.dynamic_point(shape, index, Core.new(shape, fill)), Observation.point(shape, index, Core.new(shape, fill)), Some{fill},
    dynamic_point_result(shape, index, Core.new(shape, fill)),
    new_lift(shape, fill, result => {Observation.point(shape, index, result) == Some{fill} : Maybe<&2, F32>},
      valid => read_point_done(shape, fill, valid, index, inside)))
