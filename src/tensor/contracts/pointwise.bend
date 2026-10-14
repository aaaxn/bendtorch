import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ./replayed.bend as Replayed

# Pointwise family: out[i] = f(operation, left[ia(plan, i)], right[ib(plan, i)])
# for every flat output index i, written in increasing i. `f` is the scalar
# function and `ia`, `ib` the index maps.

type State<-shape: Shape.Shape> is Type:
  State{left: Core.Dynamic, right: Core.Dynamic, output: Core.Tensor<shape>}

def write_step(-shape: Shape.Shape,
  rec: State<shape> -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  left: Core.Dynamic, right: Core.Dynamic, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  (output, result) = pair
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{unit}:
      rec(State{left, right, output})

def right_step(~O: Data, ~f: O -> F32 -> F32 -> F32, +operation: O, -shape: Shape.Shape, +index: U32,
  rec: State<shape> -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  left: Core.Dynamic, output: Core.Tensor<shape>, x: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  (right, result) = pair
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{y}:
      write_step(shape, rec, left, right, Core.write_flat(shape, output, index, f(operation, x, y)))

def left_step(~O: Data, ~f: O -> F32 -> F32 -> F32, +operation: O, -shape: Shape.Shape, +index: U32, +right_index: U32,
  rec: State<shape> -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  right: Core.Dynamic, output: Core.Tensor<shape>, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  (left, result) = pair
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{x}:
      right_step(~O, ~f, operation, shape, index, rec, left, output, x, Core.read_dynamic(right, right_index))

def step(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, -shape: Shape.Shape, +cursor: Nat,
  rec: State<shape> -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  state: State<shape>) -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  State{left, right, output} = state
  left_step(~O, ~f, operation, shape, U32.from_nat(cursor), U32.from_nat(ib(plan, cursor)), rec, right, output,
    Core.read_dynamic(left, U32.from_nat(ia(plan, cursor))))

def loop(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  fuel: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +cursor: Nat, state: State<shape>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match fuel:
    case 0n:
      State{left, right, output} = state
      ((left, right), Done{Core.pack(shape, output)})
    case 1n+rest:
      step(~O, ~f, ~P, ~ia, ~ib, operation, plan, shape, cursor,
        state => loop(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, shape, 1n+cursor, state), state)

def allocated(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{output}:
      loop(~O, ~f, ~P, ~ia, ~ib, Shape.elements(shape), operation, plan, shape, 0n, State{left, right, output})

# Allocates a `+0.0` output of `shape` and writes every cell. Both inputs come
# back to the caller, on success and on failure.
def run(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  allocated(~O, ~f, ~P, ~ia, ~ib, operation, plan, shape, left, right, Core.new(shape, 0.0))

# True when `map(plan, i) < bound` for every `i < count`.
def inside(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +bound: Nat) -> Bool:
  match count:
    case 0n:
      True{}
    case 1n+previous:
      Bool.and(inside(~P, ~map, plan, previous, bound), Nat.is_lt(map(plan, previous), bound))

# The word the family writes at output index `index`.
def cell(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +index: Nat) -> F32:
  f(operation, Replayed.value(left, left_fill, left_ops, U32.from_nat(ia(plan, index))),
    Replayed.value(right, right_fill, right_ops, U32.from_nat(ib(plan, index))))

# The reads the loop makes on one input before output index `count`.
def reads(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      Replayed.snoc(reads(~P, ~map, plan, previous), StorageLaws.Read{U32.from_nat(map(plan, previous))})

# The writes the loop makes on the output before output index `count`.
def writes(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +count: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      Replayed.snoc(writes(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops),
        StorageLaws.Write{U32.from_nat(previous),
          cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous)})

def output_point(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      Core.dynamic_read_value(Core.read_dynamic(output, index))

def output_dimensions(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Maybe<&2, List<&2, U32>>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{Core.Packed{shape, tensor}}:
      Some{Shape.dimensions(shape)}

def succeeded(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  (inputs, outcome) = result
  Result.is_done(&2, &1, Core.Error, Core.Dynamic, outcome)

def left_point(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  ((left, right), outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(left, index))

def right_point(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  ((left, right), outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(right, index))
