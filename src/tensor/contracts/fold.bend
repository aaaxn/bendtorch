import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ./replayed.bend as Replayed

# Ordered-fold family: start at +0.0 and, for k = 0, 1, ..., count - 1 in
# increasing order, read input[ia(plan, k)] and, where keep(plan, k) holds,
# fold it in with acc = g(operation, acc, input[ia(plan, k)]). The pair form
# reads left[ia(plan, k)] and right[ib(plan, k)] and folds every pair with
# acc = g(operation, acc, left word, right word).

# The accumulator after one word: `g` folds the word in where `kept` holds,
# and the accumulator stays as it is elsewhere.
def chosen(~O: Data, ~g: O -> F32 -> F32 -> F32, +operation: O, kept: Bool, accumulator: F32, word: F32) -> F32:
  match kept:
    case False{}:
      accumulator
    case True{}:
      g(operation, accumulator, word)

def step(~O: Data, ~g: O -> F32 -> F32 -> F32, +operation: O, +kept: Bool,
  rec: Core.Dynamic -> F32 -> Core.Dynamic & Result<&2, &1, Core.Error, F32>,
  accumulator: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{error})
    case Done{x}:
      rec(input, chosen(~O, ~g, operation, kept, accumulator, x))

def loop(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~keep: P -> Nat -> Bool,
  fuel: Nat, +operation: O, +plan: P, +cursor: Nat, input: Core.Dynamic, accumulator: F32) ->
  Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  match fuel:
    case 0n:
      (input, Done{accumulator})
    case 1n+rest:
      step(~O, ~g, operation, keep(plan, cursor), input => value => loop(~O, ~g, ~P, ~ia, ~keep, rest, operation, plan, 1n+cursor, input, value),
        accumulator, Core.read_dynamic(input, U32.from_nat(ia(plan, cursor))))

# Folds `count` words of the input. The input comes back to the caller, on
# success and on failure.
def run(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~keep: P -> Nat -> Bool,
  +operation: O, +plan: P, +count: Nat, input: Core.Dynamic) -> Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  loop(~O, ~g, ~P, ~ia, ~keep, count, operation, plan, 0n, input, 0.0)

def pair_right(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, +operation: O,
  rec: Core.Dynamic -> Core.Dynamic -> F32 -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>,
  left: Core.Dynamic, accumulator: F32, x: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  (right, result) = pair
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{y}:
      rec(left, right, g(operation, accumulator, x, y))

def pair_left(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, +operation: O, +right_index: U32,
  rec: Core.Dynamic -> Core.Dynamic -> F32 -> (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>,
  right: Core.Dynamic, accumulator: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  (left, result) = pair
  match result:
    case Fail{error}:
      ((left, right), Fail{error})
    case Done{x}:
      pair_right(~O, ~g, operation, rec, left, accumulator, x, Core.read_dynamic(right, right_index))

def pair_loop(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  fuel: Nat, +operation: O, +plan: P, +cursor: Nat, left: Core.Dynamic, right: Core.Dynamic, accumulator: F32) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  match fuel:
    case 0n:
      ((left, right), Done{accumulator})
    case 1n+rest:
      pair_left(~O, ~g, operation, U32.from_nat(ib(plan, cursor)),
        left => right => value => pair_loop(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, 1n+cursor, left, right, value),
        right, accumulator, Core.read_dynamic(left, U32.from_nat(ia(plan, cursor))))

# Folds `count` pairs of words. Both inputs come back to the caller, on
# success and on failure.
def pair_run(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +count: Nat, left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>:
  pair_loop(~O, ~g, ~P, ~ia, ~ib, count, operation, plan, 0n, left, right, 0.0)

# The fold of the first `count` words that `ia` picks from a replayed input.
def total(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~keep: P -> Nat -> Bool,
  +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> F32:
  match count:
    case 0n:
      0.0
    case 1n+previous:
      chosen(~O, ~g, operation, keep(plan, previous), total(~O, ~g, ~P, ~ia, ~keep, operation, plan, shape, fill, ops, previous),
        Replayed.value(shape, fill, ops, U32.from_nat(ia(plan, previous))))

# The fold of the first `count` pairs of words that `ia` and `ib` pick.
def pair_total(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> F32:
  match count:
    case 0n:
      0.0
    case 1n+previous:
      g(operation, pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        Replayed.value(left, left_fill, left_ops, U32.from_nat(ia(plan, previous))),
        Replayed.value(right, right_fill, right_ops, U32.from_nat(ib(plan, previous))))

def value(result: Core.Dynamic & Result<&2, &1, Core.Error, F32>) -> Maybe<&2, F32>:
  (input, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{word}:
      Some{word}

def input_point(result: Core.Dynamic & Result<&2, &1, Core.Error, F32>, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(input, index))

def pair_value(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>) -> Maybe<&2, F32>:
  (inputs, outcome) = result
  match outcome:
    case Fail{error}:
      None{}
    case Done{word}:
      Some{word}

def left_point(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>, +index: U32) -> Maybe<&2, F32>:
  ((left, right), outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(left, index))

def right_point(result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>, +index: U32) -> Maybe<&2, F32>:
  ((left, right), outcome) = result
  Core.dynamic_read_value(Core.read_dynamic(right, index))
