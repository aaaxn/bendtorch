import Base
import ./model.bend as Model
import ./array.bend as ArrayCore
import ./refinement.bend as Refinement
import ./main.bend as Storage

law write_read:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +path: Model.Path(d)
  for value: T
  for valid: Model.Balanced(T, d, array)
  {Model.observe(T, d, ArrayCore.write_path(T, d, array, path, value), path) == Some{value} : Maybe<&2, T>}

law update_frame:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +i: Model.Path(d)
  for +j: Model.Path(d)
  for +value: T
  for valid: Model.Balanced(T, d, array)
  {Model.observe(T, d, ArrayCore.write_path(T, d, array, i, value), j) == Model.select(T, Model.same(d, i, j), value, Model.observe(T, d, array, j)) : Maybe<&2, T>}

law read_refines:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +path: Model.Path(d)
  for valid: Model.Balanced(T, d, array)
  {ArrayCore.observation(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid)) == Model.observe(T, d, array, path) : Maybe<&2, T>}

law read_retains:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +path: Model.Path(d)
  for valid: Model.Balanced(T, d, array)
  {ArrayCore.retained(T, d, array, path, ArrayCore.read_path(T, d, array, path, valid)) == array : Array<T>}

law primitive_update_frame:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +i: U32
  for +other: Model.Path(d)
  for +value: T
  for valid: Model.Balanced(T, d, array)
  {Model.observe(T, d, Array.set(T, array, i, value), other) == Model.select(T, Model.same(d, Refinement.route(d, i), other), value, Model.observe(T, d, array, other)) : Maybe<&2, T>}

law path_self_same:
  for +d: Nat
  for +path: Model.Path(d)
  {Model.same(d, path, path) == True{} : Bool}

law primitive_write_read:
  for -T: Data
  for +d: Nat
  for array: Array<T>
  for +i: U32
  for +value: T
  for valid: Model.Balanced(T, d, array)
  {Model.observe(T, d, Array.set(T, array, i, value), Refinement.route(d, i)) == Some{value} : Maybe<&2, T>}

def outcome(-A: Type, pair: Storage.Storage & A) -> A:
  (storage, result) = pair
  result

def size(storage: Storage.Storage) -> Nat:
  outcome(Nat, Storage.length(storage))

def owner(-A: Type, pair: Storage.Storage & A) -> Storage.Storage:
  (storage, result) = pair
  storage

def read(storage: Storage.Storage, +i: U32) -> Result<&2, &2, Storage.Error, F32>:
  outcome(Result<&2, &2, Storage.Error, F32>, Storage.get(storage, i))

def kept(storage: Storage.Storage, +i: U32) -> Storage.Storage:
  owner(Result<&2, &2, Storage.Error, F32>, Storage.get(storage, i))

def written(storage: Storage.Storage, +i: U32, value: F32) -> Storage.Storage:
  owner(Result<&2, &2, Storage.Error, Unit>, Storage.set(storage, i, value))

def status(storage: Storage.Storage, +i: U32, value: F32) -> Result<&2, &2, Storage.Error, Unit>:
  outcome(Result<&2, &2, Storage.Error, Unit>, Storage.set(storage, i, value))

def created.of(result: Result<&2, &1, Storage.Error, Storage.Storage>) -> Storage.Storage:
  match result:
    case Fail{error}:
      Storage.Empty{}
    case Done{storage}:
      storage

def created(+n: U32, value: F32) -> Storage.Storage:
  created.of(Storage.new(n, value))

def refused.of(result: Result<&2, &1, Storage.Error, Storage.Storage>) -> Maybe<&2, Storage.Error>:
  match result:
    case Fail{error}:
      Some{error}
    case Done{storage}:
      None{}

def refused(+n: U32, value: F32) -> Maybe<&2, Storage.Error>:
  refused.of(Storage.new(n, value))

def first(storage: Storage.Storage) -> Storage.Storage:
  owner(Storage.Storage, Storage.clone(storage))

def second(storage: Storage.Storage) -> Storage.Storage:
  outcome(Storage.Storage, Storage.clone(storage))

type Op is Data:
  Write{index: U32, value: F32}
  Read{index: U32}
  First{}
  Second{}
  Length{}

def step(storage: Storage.Storage, op: Op) -> Storage.Storage:
  match op:
    case Write{+i, value}:
      written(storage, i, value)
    case Read{+i}:
      kept(storage, i)
    case First{}:
      first(storage)
    case Second{}:
      second(storage)
    case Length{}:
      owner(Nat, Storage.length(storage))

def replay(ops: List<&2, Op>, storage: Storage.Storage) -> Storage.Storage:
  match ops:
    case Nil{}:
      storage
    case op <> rest:
      replay(rest, step(storage, op))

def latest(+n: Nat, op: Op, +j: U32, value: F32) -> F32:
  match op:
    case Write{+i, new}:
      Bool.pick(F32, Bool.and(Nat.is_lt(U32.to_nat(i), n), Nat.is_eq(U32.to_nat(i), U32.to_nat(j))), new, value)
    case Read{i}:
      value
    case First{}:
      value
    case Second{}:
      value
    case Length{}:
      value

def expected(ops: List<&2, Op>, +n: Nat, +j: U32, value: F32) -> F32:
  match ops:
    case Nil{}:
      value
    case op <> rest:
      expected(rest, n, j, latest(n, op, j, value))

law new_too_large:
  for +n: U32
  for value: F32
  for +over: {U32.is_le(n, 1048576) == False{} : Bool}
  {refused(n, value) == Some{Storage.TooLarge{}} : Maybe<&2, Storage.Error>}

law new_fits:
  for +n: U32
  for value: F32
  for +fits: {U32.is_le(n, 1048576) == True{} : Bool}
  {refused(n, value) == None{} : Maybe<&2, Storage.Error>}

law new_empty:
  for value: F32
  {Storage.new(0, value) == Done{Storage.Empty{}} : Result<&2, &1, Storage.Error, Storage.Storage>}

law get_outside:
  for storage: Storage.Storage
  for +i: U32
  for outside: {Nat.is_lt(U32.to_nat(i), size(storage)) == False{} : Bool}
  {read(storage, i) == Fail{Storage.OutOfBounds{}} : Result<&2, &2, Storage.Error, F32>}

law set_outside:
  for storage: Storage.Storage
  for +i: U32
  for value: F32
  for outside: {Nat.is_lt(U32.to_nat(i), size(storage)) == False{} : Bool}
  {status(storage, i, value) == Fail{Storage.OutOfBounds{}} : Result<&2, &2, Storage.Error, Unit>}

law replay_size:
  for +n: U32
  for +value: F32
  for +ops: List<&2, Op>
  for +fits: {U32.is_le(n, 1048576) == True{} : Bool}
  {size(replay(ops, created(n, value))) == U32.to_nat(n) : Nat}

law replay_get:
  for +n: U32
  for +value: F32
  for +ops: List<&2, Op>
  for +j: U32
  for +fits: {U32.is_le(n, 1048576) == True{} : Bool}
  for inside: {Nat.is_lt(U32.to_nat(j), U32.to_nat(n)) == True{} : Bool}
  {read(replay(ops, created(n, value)), j) == Done{expected(ops, U32.to_nat(n), j, value)} : Result<&2, &2, Storage.Error, F32>}

law replay_set:
  for +n: U32
  for +value: F32
  for +ops: List<&2, Op>
  for +j: U32
  for +w: F32
  for +fits: {U32.is_le(n, 1048576) == True{} : Bool}
  for inside: {Nat.is_lt(U32.to_nat(j), U32.to_nat(n)) == True{} : Bool}
  {status(replay(ops, created(n, value)), j, w) == Done{Unit{}} : Result<&2, &2, Storage.Error, Unit>}
