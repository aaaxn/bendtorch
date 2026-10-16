import Base
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ../storage/route.bend as Route

def logical_length(storage: Storage.Storage) -> Nat:
  match storage:
    case Storage.Empty{}:
      0n
    case Storage.Owned{d, n, array, balanced, valid}:
      n

def retained_read(result: Storage.Storage & Result<&2, &2, Storage.Error, F32>) -> Nat:
  (storage, value) = result
  logical_length(storage)

def retained_write(result: Storage.Storage & Result<&2, &2, Storage.Error, Unit>) -> Nat:
  (storage, value) = result
  logical_length(storage)

def finish_read_length(+d: Nat, +n: Nat, -original: Array<F32>, +i: U32,
  result: Array<F32> & F32, -exact: {result == Array.get(F32, original, i) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {retained_read(Storage.finish_read(d, n, original, i, result, exact, balanced, valid)) == n : Nat}:
  (array, value) = result
  {==}

def get_checked_length(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {retained_read(Storage.get_checked(d, n, array, i, allowed, decision, balanced, valid)) == n : Nat}:
  match allowed:
    case False{}:
      {==}
    case True{}:
      finish_read_length(d, n, array, i, Array.get(F32, array, i), {==}, balanced, valid)

def get_length(storage: Storage.Storage, +i: U32) ->
  {retained_read(Storage.get(storage, i)) == logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      get_checked_length(d, n, array, i, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid)

def set_checked_length(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {retained_write(Storage.set_checked(d, n, array, i, value, allowed, decision, balanced, valid)) == n : Nat}:
  match allowed:
    case False{}:
      {==}
    case True{}:
      {==}

def set_length(storage: Storage.Storage, +i: U32, +value: F32) ->
  {retained_write(Storage.set(storage, i, value)) == logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      set_checked_length(d, n, array, i, value, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid)

def first_length(pair: Storage.Storage & Storage.Storage) -> Nat:
  (first, second) = pair
  logical_length(first)

def second_length(pair: Storage.Storage & Storage.Storage) -> Nat:
  (first, second) = pair
  logical_length(second)

def finish_clone_first(+d: Nat, +n: Nat, -original: Array<F32>, pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {first_length(Storage.finish_clone(d, n, original, pair, exact, balanced, valid)) == n : Nat}:
  (first, second) = pair
  {==}

def finish_clone_second(+d: Nat, +n: Nat, -original: Array<F32>, pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, original), -valid: {Storage.bounds(d, n) == True{} : Bool}) ->
  {second_length(Storage.finish_clone(d, n, original, pair, exact, balanced, valid)) == n : Nat}:
  (first, second) = pair
  {==}

def clone_first_length(storage: Storage.Storage) ->
  {first_length(Storage.clone(storage)) == logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      finish_clone_first(d, n, array, Array.clone(F32, array), {==}, balanced, valid)

def clone_second_length(storage: Storage.Storage) ->
  {second_length(Storage.clone(storage)) == logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{+d, +n, array, balanced, valid}:
      finish_clone_second(d, n, array, Array.clone(F32, array), {==}, balanced, valid)

def length_owner(pair: Storage.Storage & Nat) -> Nat:
  (storage, n) = pair
  logical_length(storage)

def length_value(pair: Storage.Storage & Nat) -> Nat:
  (storage, n) = pair
  n

def length_exact(storage: Storage.Storage) ->
  {length_owner(Storage.length(storage)) == length_value(Storage.length(storage)) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{d, n, array, balanced, valid}:
      {==}

def length_preserved(storage: Storage.Storage) ->
  {length_owner(Storage.length(storage)) == logical_length(storage) : Nat}:
  match storage:
    case Storage.Empty{}:
      {==}
    case Storage.Owned{d, n, array, balanced, valid}:
      {==}

def IsTrue(value: Bool) -> Data:
  match value:
    case True{}:
      Unit
    case False{}:
      Empty

def false_true(equal: {True{} == False{} : Bool}) -> Empty:
  %equal : IsTrue(_)
  Unit{}

def natural_equal(+a: Nat, +b: Nat, equal: {True{} == Nat.is_eq(a, b) : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, false_true(equal))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, false_true(equal))
    case 1n+p 1n+q:
      Equal.cong(Nat, Nat, x => 1n+x, p, q, natural_equal(p, q, equal))

def observe(storage: Storage.Storage, +index: U32) -> Maybe<&2, F32>:
  match storage:
    case Storage.Empty{}:
      None{}
    case Storage.Owned{+d, n, array, balanced, valid}:
      Model.observe(F32, d, array, Refinement.route(d, index))

def read_value(pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) -> Maybe<&2, F32>:
  (storage, result) = pair
  match result:
    case Fail{error}:
      None{}
    case Done{value}:
      Some{value}

def observe_index(+depth: Nat, -length: Nat, array: Array<F32>, +index: U32, matched: Bool) -> Maybe<&2, F32>:
  match matched:
    case False{}:
      None{}
    case True{}:
      Model.observe(F32, depth, array, Refinement.route(depth, index))

def observe_allowed(+depth: Nat, +length: Nat, array: Array<F32>, +index: U32, allowed: Bool) -> Maybe<&2, F32>:
  match allowed:
    case False{}:
      None{}
    case True{}:
      observe_index(depth, length, array, index, Storage.index_matches(depth, length, index))

def checked_observe(storage: Storage.Storage, +index: U32) -> Maybe<&2, F32>:
  match storage:
    case Storage.Empty{}:
      None{}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      observe_allowed(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length))

def finish_read_value(-depth: Nat, -length: Nat, -original: Array<F32>, -index: U32,
  pair: Array<F32> & F32, -exact: {pair == Array.get(F32, original, index) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, depth, original), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {read_value(Storage.finish_read(depth, length, original, index, pair, exact, balanced, valid)) ==
    Some{Refinement.second(F32, pair)} : Maybe<&2, F32>}:
  (array, value) = pair
  {==}

type ReadCheckedModel<-depth: Nat, -length: Nat, -array: Array<F32>, -index: U32, -allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}> is Data:
  ReadCheckedModel{-meaning: {read_value(Storage.get_checked(depth, length, array, index, allowed, decision, balanced, valid)) ==
    observe_allowed(depth, length, array, index, allowed) : Maybe<&2, F32>}}

def read_checked_model(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  ReadCheckedModel<depth, length, array, index, allowed, decision, balanced, valid>:
  match allowed:
    case False{}:
      ReadCheckedModel{{==}}
    case True{}:
      ReadCheckedModel{
        %Route.index_complete(depth, length, index, valid,
          Equal.sym(Bool, True{}, Nat.is_lt(U32.to_nat(index), length), decision)) :
          {read_value(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid)) ==
            observe_index(depth, length, array, index, _) : Maybe<&2, F32>}
        Equal.trans(Maybe<&2, F32>,
          read_value(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid)),
          Some{Refinement.second(F32, Array.get(F32, array, index))}, Model.observe(F32, depth, array, Refinement.route(depth, index)),
          finish_read_value(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid),
          Refinement.law_get_refines(F32, depth, array, index, balanced))}

type StorageReadModel<-original: Storage.Storage, -index: U32> is Data:
  StorageReadModel{-meaning: {read_value(Storage.get(original, index)) == checked_observe(original, index) : Maybe<&2, F32>}}

def read_checked_storage(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  certificate: ReadCheckedModel<depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid>) ->
  StorageReadModel<Storage.Owned{depth, length, array, balanced, valid}, index>:
  ReadCheckedModel{meaning} = certificate
  StorageReadModel{meaning}

def storage_read_model(original: Storage.Storage, +index: U32) -> StorageReadModel<original, index>:
  match original:
    case Storage.Empty{}:
      StorageReadModel{{==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      read_checked_storage(depth, length, array, index, balanced, valid,
        read_checked_model(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid))

def checked_observe_array(+depth: Nat, +length: Nat, array: Array<F32>, +index: U32) -> Maybe<&2, F32>:
  observe_allowed(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length))

def retained_observe(pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>, index: U32) -> Maybe<&2, F32>:
  (storage, result) = pair
  checked_observe(storage, index)

def finish_read_observe(+depth: Nat, +length: Nat, -original: Array<F32>, -index: U32, +other: U32,
  pair: Array<F32> & F32, -exact: {pair == Array.get(F32, original, index) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, depth, original), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  {retained_observe(Storage.finish_read(depth, length, original, index, pair, exact, balanced, valid), other) ==
    checked_observe_array(depth, length, Refinement.first(F32, pair), other) : Maybe<&2, F32>}:
  (array, value) = pair
  {==}

type ReadCheckedKeeps<-depth: Nat, -length: Nat, -array: Array<F32>, -index: U32, -allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}> is Data:
  ReadCheckedKeeps{-kept: @+other: U32 ->
    {retained_observe(Storage.get_checked(depth, length, array, index, allowed, decision, balanced, valid), other) ==
      checked_observe_array(depth, length, array, other) : Maybe<&2, F32>}}

def read_checked_keeps(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(index), length) : Bool},
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool}) ->
  ReadCheckedKeeps<depth, length, array, index, allowed, decision, balanced, valid>:
  match allowed:
    case False{}:
      ReadCheckedKeeps{other => {==}}
    case True{}:
      ReadCheckedKeeps{other => Equal.trans(Maybe<&2, F32>,
        retained_observe(Storage.finish_read(depth, length, array, index, Array.get(F32, array, index), {==}, balanced, valid), other),
        checked_observe_array(depth, length, Refinement.first(F32, Array.get(F32, array, index)), other),
        checked_observe_array(depth, length, array, other),
        finish_read_observe(depth, length, array, index, other, Array.get(F32, array, index), {==}, balanced, valid),
        Equal.cong(Array<F32>, Maybe<&2, F32>, a => checked_observe_array(depth, length, a, other),
          Refinement.first(F32, Array.get(F32, array, index)), array, Refinement.law_get_retains(F32, depth, array, index, balanced)))}

type StorageReadKeeps<-original: Storage.Storage, -index: U32> is Data:
  StorageReadKeeps{-kept: @+other: U32 ->
    {retained_observe(Storage.get(original, index), other) == checked_observe(original, other) : Maybe<&2, F32>}}

def read_checked_storage_keeps(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  -balanced: Model.Balanced(F32, depth, array), -valid: {Storage.bounds(depth, length) == True{} : Bool},
  certificate: ReadCheckedKeeps<depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid>) ->
  StorageReadKeeps<Storage.Owned{depth, length, array, balanced, valid}, index>:
  ReadCheckedKeeps{kept} = certificate
  StorageReadKeeps{kept}

def storage_read_keeps(original: Storage.Storage, +index: U32) -> StorageReadKeeps<original, index>:
  match original:
    case Storage.Empty{}:
      StorageReadKeeps{other => {==}}
    case Storage.Owned{+depth, +length, array, balanced, valid}:
      read_checked_storage_keeps(depth, length, array, index, balanced, valid,
        read_checked_keeps(depth, length, array, index, Nat.is_lt(U32.to_nat(index), length), {==}, balanced, valid))

def and_left_true(+left: Bool, +right: Bool, equal: {Bool.and(left, right) == True{} : Bool}) -> {left == True{} : Bool}:
  match left:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, false_true(Equal.sym(Bool, False{}, True{}, equal)))
    case True{}:
      {==}

def and_right_true(+left: Bool, +right: Bool, equal: {Bool.and(left, right) == True{} : Bool}) -> {right == True{} : Bool}:
  match left:
    case False{}:
      Empty.absurd({right == True{} : Bool}, false_true(Equal.sym(Bool, False{}, True{}, equal)))
    case True{}:
      equal

def checked_index_allowed(+depth: Nat, +length: Nat, +index: U32,
  +checked: {Storage.index_matches(depth, length, index) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}:
  %natural_equal(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index),
    Equal.sym(Bool, Nat.is_eq(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index)), True{},
      and_left_true(Nat.is_eq(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index)),
        Nat.is_lt(Model.offset(depth, Refinement.route(depth, index)), length), checked))) :
    {Nat.is_lt(_, length) == True{} : Bool}
  and_right_true(Nat.is_eq(Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index)),
    Nat.is_lt(Model.offset(depth, Refinement.route(depth, index)), length), checked)

type ArrayObservationModel<-depth: Nat, -length: Nat, -array: Array<F32>, -index: U32,
  -checked: {Storage.index_matches(depth, length, index) == True{} : Bool}> is Data:
  ArrayObservationModel{-equal: {checked_observe_array(depth, length, array, index) ==
    Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}}

def array_observation_model(+depth: Nat, +length: Nat, -array: Array<F32>, +index: U32,
  +checked: {Storage.index_matches(depth, length, index) == True{} : Bool}) -> ArrayObservationModel<depth, length, array, index, checked>:
  ArrayObservationModel{
    %Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), length), True{}, checked_index_allowed(depth, length, index, checked)) :
      {observe_allowed(depth, length, array, index, _) == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}
    %Equal.sym(Bool, Storage.index_matches(depth, length, index), True{}, checked) :
      {observe_index(depth, length, array, index, _) == Model.observe(F32, depth, array, Refinement.route(depth, index)) : Maybe<&2, F32>}
    {==}}
