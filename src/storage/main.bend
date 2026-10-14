import Base
import ./model.bend as Model
import ./array.bend as ArrayModel
import ./refinement.bend as Refinement
import ./certificates.bend as Certificates

type Error is Data:
  TooLarge{}
  OutOfBounds{}
  InvalidIndex{}

def bounds(+d: Nat, +n: Nat) -> Bool:
  Bool.and(Nat.is_le(d, 20n), Bool.and(Nat.is_lt(0n, n), Nat.is_le(n, Model.capacity(d))))

type Storage is Type:
  Empty{}
  Owned{depth: Nat, length: Nat, array: Array<F32>,
    -balanced: Model.Balanced(F32, depth, array),
    -bounds: {bounds(depth, length) == True{} : Bool}}

def depth_seek(fuel: Nat, +n: Nat, +depth: Nat, +capacity: Nat, done: Bool) -> Nat & Nat:
  match fuel done:
    case 0n _:
      (depth, capacity)
    case 1n+p True{}:
      (depth, capacity)
    case 1n+p False{}:
      +next = Nat.add(capacity, capacity)
      depth_seek(p, n, 1n+depth, next, Nat.is_le(n, next))

def allocate(+d: Nat, +n: Nat, +value: F32, valid: Bool,
  -decision: {valid == bounds(d, n) : Bool}) -> Result<&2, &1, Error, Storage>:
  match valid:
    case False{}:
      Fail{TooLarge{}}
    case True{}:
      Done{Owned{d, n, Array.new(F32, d, value), ArrayModel.new_balanced(F32, d, value),
        Equal.sym(Bool, True{}, bounds(d, n), decision)}}

def allocate_at(+n: Nat, value: F32, metadata: Nat & Nat) -> Result<&2, &1, Error, Storage>:
  (d0, capacity) = metadata
  +d = d0
  allocate(d, n, value, bounds(d, n), {==})

def new_checked(+n: Nat, value: F32, allowed: Bool) -> Result<&2, &1, Error, Storage>:
  match n allowed:
    case 0n _:
      Done{Empty{}}
    case 1n+p False{}:
      Fail{TooLarge{}}
    case 1n+p True{}:
      allocate_at(n, value, depth_seek(20n, n, 0n, 1n, Nat.is_le(n, 1n)))

def new(+n: U32, value: F32) -> Result<&2, &1, Error, Storage>:
  new_checked(U32.to_nat(n), value, U32.is_le(n, 1048576))

def index_matches(+d: Nat, +n: Nat, +i: U32) -> Bool:
  +offset = Model.offset(d, Refinement.route(d, i))
  Bool.and(Nat.is_eq(offset, U32.to_nat(i)), Nat.is_lt(offset, n))

def finish_read(+d: Nat, n: Nat, -original: Array<F32>, -i: U32,
  result: Array<F32> & F32, -exact: {result == Array.get(F32, original, i) : Array<F32> & F32},
  -balanced: Model.Balanced(F32, d, original), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, F32>:
  (array, value) = result
  (Owned{d, n, array, Certificates.get_result(F32, d, original, i, (array, value), exact, balanced), valid}, Done{value})

def read_index(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, matched: Bool,
  -decision: {matched == index_matches(d, n, i) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, F32>:
  match matched:
    case False{}:
      (Owned{d, n, array, balanced, valid}, Fail{InvalidIndex{}})
    case True{}:
      finish_read(d, n, array, i, Array.get(F32, array, i), {==}, balanced, valid)

def get_checked(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, F32>:
  match allowed:
    case False{}:
      (Owned{d, n, array, balanced, valid}, Fail{OutOfBounds{}})
    case True{}:
      read_index(d, n, array, i, index_matches(d, n, i), {==}, balanced, valid)

def get(storage: Storage, +i: U32) -> Storage & Result<&2, &2, Error, F32>:
  match storage:
    case Empty{}:
      (Empty{}, Fail{OutOfBounds{}})
    case Owned{+d, +n, array, balanced, valid}:
      get_checked(d, n, array, i, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid)

def finish_write(+d: Nat, n: Nat, -original: Array<F32>, -i: U32, -value: F32,
  array: Array<F32>, -exact: {array == Array.set(F32, original, i, value) : Array<F32>},
  -balanced: Model.Balanced(F32, d, original), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, Unit>:
  (Owned{d, n, array, Certificates.set_result(F32, d, original, i, value, array, exact, balanced), valid}, Done{Unit{}})

def write_index(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, +value: F32, matched: Bool,
  -decision: {matched == index_matches(d, n, i) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, Unit>:
  match matched:
    case False{}:
      (Owned{d, n, array, balanced, valid}, Fail{InvalidIndex{}})
    case True{}:
      finish_write(d, n, array, i, value, Array.set(F32, array, i, value), {==}, balanced, valid)

def set_checked(+d: Nat, +n: Nat, array: Array<F32>, +i: U32, value: F32, allowed: Bool,
  -decision: {allowed == Nat.is_lt(U32.to_nat(i), n) : Bool},
  -balanced: Model.Balanced(F32, d, array), -valid: {bounds(d, n) == True{} : Bool}) ->
  Storage & Result<&2, &2, Error, Unit>:
  match allowed:
    case False{}:
      (Owned{d, n, array, balanced, valid}, Fail{OutOfBounds{}})
    case True{}:
      write_index(d, n, array, i, value, index_matches(d, n, i), {==}, balanced, valid)

def set(storage: Storage, +i: U32, value: F32) -> Storage & Result<&2, &2, Error, Unit>:
  match storage:
    case Empty{}:
      (Empty{}, Fail{OutOfBounds{}})
    case Owned{+d, +n, array, balanced, valid}:
      set_checked(d, n, array, i, value, Nat.is_lt(U32.to_nat(i), n), {==}, balanced, valid)

def finish_clone(+d: Nat, +n: Nat, -original: Array<F32>, pair: Array<F32> & Array<F32>,
  -exact: {pair == Array.clone(F32, original) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, original), -valid: {bounds(d, n) == True{} : Bool}) -> Storage & Storage:
  (first, second) = pair
  (Owned{d, n, first, Certificates.clone_first(F32, d, original, (first, second), exact, balanced), valid},
    Owned{d, n, second, Certificates.clone_second(F32, d, original, (first, second), exact, balanced), valid})

def clone(storage: Storage) -> Storage & Storage:
  match storage:
    case Empty{}:
      (Empty{}, Empty{})
    case Owned{+d, +n, array, balanced, valid}:
      finish_clone(d, n, array, Array.clone(F32, array), {==}, balanced, valid)

def length(storage: Storage) -> Storage & Nat:
  match storage:
    case Empty{}:
      (Empty{}, 0n)
    case Owned{+d, +n, array, balanced, valid}:
      (Owned{d, n, array, balanced, valid}, n)
