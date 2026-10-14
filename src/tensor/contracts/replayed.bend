import Base
import ../../shape/core.bend as Shape
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../word-conversion.bend as Conversion
import ../read-contract.bend as Reads
import ../from-values-contract.bend as FromValues

# A replayed tensor holds the storage that `Storage.new(count, fill)` creates,
# changed by a list of storage operations in order. `Core.new(shape, fill)` is
# the empty history, `Core.from_values` writes each value once, and every
# contract-family output is a write history over a `+0.0` fill.
def storage(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>) -> Storage.Storage:
  StorageLaws.replay(ops, StorageLaws.created(Shape.count(shape), fill))

def value(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) -> F32:
  StorageLaws.expected(ops, U32.to_nat(Shape.count(shape)), index, fill)

def snoc(+ops: List<&2, StorageLaws.Op>, +op: StorageLaws.Op) -> List<&2, StorageLaws.Op>:
  List.append(&2, StorageLaws.Op, ops, [op])

def stored(-shape: Shape.Shape, tensor: Core.Tensor<shape>) -> Storage.Storage:
  Core.TTensor{storage, valid} = tensor
  storage

def read_stored(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Storage.Storage:
  (tensor, result) = pair
  stored(shape, tensor)

def write_stored(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Storage.Storage:
  (tensor, result) = pair
  stored(shape, tensor)

def write_status(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Bool:
  (tensor, result) = pair
  Result.is_done(&2, &2, Core.Error, Unit, result)

def some_value(+word: F32, +other: F32, same: {Some{word} == Some{other} : Maybe<&2, F32>}) -> {word == other : F32}:
  Equal.cong(Maybe<&2, F32>, F32, found => Maybe.default(&2, F32, found, word), Some{word}, Some{other}, same)

def fits(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>) ->
  {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, Lifecycle.logical_length(storage(shape, fill, ops)), StorageLaws.size(storage(shape, fill, ops)),
    Shape.logical_numel(Shape.dimensions(shape)),
    Equal.sym(Nat, StorageLaws.size(storage(shape, fill, ops)), Lifecycle.logical_length(storage(shape, fill, ops)),
      Reads.size_is_length(storage(shape, fill, ops))),
    Equal.trans(Nat, StorageLaws.size(storage(shape, fill, ops)), U32.to_nat(Shape.count(shape)),
      Shape.logical_numel(Shape.dimensions(shape)),
      StorageLaws.replay_size(Shape.count(shape), fill, ops, Conversion.shape_count_within(shape)), Reads.count_exact(shape)))

def replay_snoc(+ops: List<&2, StorageLaws.Op>, +op: StorageLaws.Op, +shape: Shape.Shape, +fill: F32) ->
  {storage(shape, fill, snoc(ops, op)) == StorageLaws.step(storage(shape, fill, ops), op) : Storage.Storage}:
  FromValues.replay_snoc(ops, op, StorageLaws.created(Shape.count(shape), fill))

def read_owner_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>,
  -same: {pair == Storage.get(original, index) : Storage.Storage & Result<&2, &2, Storage.Error, F32>}) ->
  {read_stored(shape, Core.read_result(shape, original, valid, index, pair, same)) ==
    StorageLaws.owner(Result<&2, &2, Storage.Error, F32>, pair) : Storage.Storage}:
  (storage, result) = pair
  {==}

def read_moved(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32) ->
  {read_stored(shape, Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index)) ==
    storage(shape, fill, snoc(ops, StorageLaws.Read{index})) : Storage.Storage}:
  Equal.trans(Storage.Storage, read_stored(shape, Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index)),
    StorageLaws.kept(storage(shape, fill, ops), index), storage(shape, fill, snoc(ops, StorageLaws.Read{index})),
    read_owner_projection(shape, storage(shape, fill, ops), valid, index, Storage.get(storage(shape, fill, ops), index), {==}),
    Equal.sym(Storage.Storage, storage(shape, fill, snoc(ops, StorageLaws.Read{index})), StorageLaws.kept(storage(shape, fill, ops), index),
      replay_snoc(ops, StorageLaws.Read{index}, shape, fill)))

def read_found(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Core.read_value(shape, Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index)) ==
    Some{value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>, Core.read_value(shape, Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index)),
    Lifecycle.read_value(Storage.get(storage(shape, fill, ops), index)), Some{value(shape, fill, ops, index)},
    Core.read_result_value(shape, storage(shape, fill, ops), valid, index, Storage.get(storage(shape, fill, ops), index), {==}),
    Reads.stored_value(storage(shape, fill, ops), index, value(shape, fill, ops, index),
      StorageLaws.replay_get(Shape.count(shape), fill, ops, index, Conversion.shape_count_within(shape),
        Reads.word_inside(shape, index, inside))))

def read_shift(+shape: Shape.Shape, -source: Storage.Storage, -target: Storage.Storage, -word: F32, -found: F32,
  moved: {target == source : Storage.Storage}, same: {found == word : F32},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32> -> Type,
  h: @-fits: {Lifecycle.logical_length(target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{target, fits}, Done{found}))) ->
  @-fits: {Lifecycle.logical_length(source) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{source, fits}, Done{word})):
  %moved : (@-fits: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{_, fits}, Done{word})))
  %same : (@-fits: {Lifecycle.logical_length(target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{target, fits}, Done{_})))
  h

def read_pair_lift(+shape: Shape.Shape, -kept: Storage.Storage, +found: F32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>,
  moved: {read_stored(shape, pair) == kept : Storage.Storage},
  read: {Core.read_value(shape, pair) == Some{found} : Maybe<&2, F32>},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32> -> Type,
  h: @-fits: {Lifecycle.logical_length(kept) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{kept, fits}, Done{found}))) -> P(pair):
  match pair:
    case (Core.TTensor{storage, valid}, Fail{error}):
      Empty.absurd(P((Core.TTensor{storage, valid}, Fail{error})),
        Arith.clash(Equal.cong(Maybe<&2, F32>, Bool, word => Maybe.is_some(&2, F32, word), None{}, Some{found}, read)))
    case (Core.TTensor{storage, valid}, Done{+word}):
      read_shift(shape, storage, kept, word, found, Equal.sym(Storage.Storage, storage, kept, moved), some_value(found, word, Equal.sym(Maybe<&2, F32>, Some{word}, Some{found}, read)),
        P, h)(valid)

def read_lift(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32> -> Type,
  h: @-fits: {Lifecycle.logical_length(storage(shape, fill, snoc(ops, StorageLaws.Read{index}))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{storage(shape, fill, snoc(ops, StorageLaws.Read{index})), fits}, Done{value(shape, fill, ops, index)}))) ->
  P(Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index)):
  read_pair_lift(shape, storage(shape, fill, snoc(ops, StorageLaws.Read{index})), value(shape, fill, ops, index),
    Core.read_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index),
    read_moved(shape, fill, ops, valid, index), read_found(shape, fill, ops, valid, index, inside), P, h)

def write_owner_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +word: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, index, word) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {write_stored(shape, Core.write_result(shape, original, valid, index, word, pair, same)) ==
    StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, pair) : Storage.Storage}:
  (storage, result) = pair
  {==}

def write_status_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +word: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, index, word) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {write_status(shape, Core.write_result(shape, original, valid, index, word, pair, same)) ==
    Result.is_done(&2, &2, Storage.Error, Unit, StorageLaws.outcome(Result<&2, &2, Storage.Error, Unit>, pair)) : Bool}:
  (storage, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{unit}:
      {==}

def write_moved(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, +word: F32) ->
  {write_stored(shape, Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word)) ==
    storage(shape, fill, snoc(ops, StorageLaws.Write{index, word})) : Storage.Storage}:
  Equal.trans(Storage.Storage, write_stored(shape, Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word)),
    StorageLaws.written(storage(shape, fill, ops), index, word), storage(shape, fill, snoc(ops, StorageLaws.Write{index, word})),
    write_owner_projection(shape, storage(shape, fill, ops), valid, index, word, Storage.set(storage(shape, fill, ops), index, word), {==}),
    Equal.sym(Storage.Storage, storage(shape, fill, snoc(ops, StorageLaws.Write{index, word})), StorageLaws.written(storage(shape, fill, ops), index, word),
      replay_snoc(ops, StorageLaws.Write{index, word}, shape, fill)))

def write_done(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, +word: F32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {write_status(shape, Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word)) == True{} : Bool}:
  Equal.trans(Bool, write_status(shape, Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word)),
    Result.is_done(&2, &2, Storage.Error, Unit, StorageLaws.status(storage(shape, fill, ops), index, word)), True{},
    write_status_projection(shape, storage(shape, fill, ops), valid, index, word, Storage.set(storage(shape, fill, ops), index, word), {==}),
    Equal.cong(Result<&2, &2, Storage.Error, Unit>, Bool, Result.is_done(&2, &2, Storage.Error, Unit),
      StorageLaws.status(storage(shape, fill, ops), index, word), Done{Unit{}},
      StorageLaws.replay_set(Shape.count(shape), fill, ops, index, word, Conversion.shape_count_within(shape),
        Reads.word_inside(shape, index, inside))))

def write_shift(+shape: Shape.Shape, -source: Storage.Storage, -target: Storage.Storage,
  moved: {target == source : Storage.Storage},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit> -> Type,
  h: @-fits: {Lifecycle.logical_length(target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{target, fits}, Done{Unit{}}))) ->
  @-fits: {Lifecycle.logical_length(source) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{source, fits}, Done{Unit{}})):
  %moved : (@-fits: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{_, fits}, Done{Unit{}})))
  h

def write_pair_lift(+shape: Shape.Shape, -kept: Storage.Storage,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  moved: {write_stored(shape, pair) == kept : Storage.Storage},
  done: {write_status(shape, pair) == True{} : Bool},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit> -> Type,
  h: @-fits: {Lifecycle.logical_length(kept) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{kept, fits}, Done{Unit{}}))) -> P(pair):
  match pair:
    case (Core.TTensor{storage, valid}, Fail{error}):
      Empty.absurd(P((Core.TTensor{storage, valid}, Fail{error})), Arith.clash(done))
    case (Core.TTensor{storage, valid}, Done{Unit{}}):
      write_shift(shape, storage, kept, Equal.sym(Storage.Storage, storage, kept, moved), P, h)(valid)

def write_lift(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, +word: F32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -P: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit> -> Type,
  h: @-fits: {Lifecycle.logical_length(storage(shape, fill, snoc(ops, StorageLaws.Write{index, word}))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{storage(shape, fill, snoc(ops, StorageLaws.Write{index, word})), fits}, Done{Unit{}}))) ->
  P(Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word)):
  write_pair_lift(shape, storage(shape, fill, snoc(ops, StorageLaws.Write{index, word})),
    Core.write_flat(shape, Core.TTensor{storage(shape, fill, ops), valid}, index, word),
    write_moved(shape, fill, ops, valid, index, word), write_done(shape, fill, ops, valid, index, word, inside), P, h)

def new_lift(+shape: Shape.Shape, +fill: F32, -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  h: @-fits: {Lifecycle.logical_length(storage(shape, fill, Nil{})) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Done{Core.TTensor{storage(shape, fill, Nil{}), fits}})) -> P(Core.new(shape, fill)):
  Reads.new_lift(shape, fill, P, h)

def expected_snoc(+ops: List<&2, StorageLaws.Op>, +op: StorageLaws.Op, +n: Nat, +index: U32, +word: F32) ->
  {StorageLaws.expected(snoc(ops, op), n, index, word) == StorageLaws.latest(n, op, index, StorageLaws.expected(ops, n, index, word)) : F32}:
  match ops:
    case Nil{}:
      {==}
    case +first <> +rest:
      expected_snoc(rest, op, n, index, StorageLaws.latest(n, first, index, word))

def value_after_read(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +read: U32, +index: U32) ->
  {value(shape, fill, snoc(ops, StorageLaws.Read{read}), index) == value(shape, fill, ops, index) : F32}:
  expected_snoc(ops, StorageLaws.Read{read}, U32.to_nat(Shape.count(shape)), index, fill)
