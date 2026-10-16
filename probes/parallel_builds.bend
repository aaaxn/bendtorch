import Base
import ../src/shape/core.bend as Shape
import ../src/tensor/core.bend as Core
import ../src/tensor/matmul.bend as Matrix
import ../src/tensor/elementwise.bend as Elementwise
import ../src/autograd/broadcast.bend as Broadcast
import ../src/autograd/reduce.bend as Reduce
import ../src/autograd/reduce-fold.bend as ReduceFold

# The parallel builds next to the sequential definitions, on inputs large
# enough to fork. For each product, reduction and broadcast the program prints how many levels the
# public build forks, how many output words differ from the sequential
# definition, and a rolling hash of the public words. tools/tensor_parallel.py
# runs it at several thread counts and requires every count to print the same
# bytes, with no differing word.

# Multiples of 1/64: every product and partial sum of the small products below
# is exact in F32, so these cells come out the same in any term order.
def exact(+index: Nat, +step: Nat) -> F32:
  F32.sub(F32.div(F32.from_nat(Nat.mod(Nat.mul(index, step), 251n)), 64.0), 1.875)

# Sevenths, every fourth one 4096 times larger: products and partial sums
# round, large terms cancel and small ones get lost, so adding a cell's terms
# in another order changes its bits. On the rounding product below, adding
# them in decreasing k changes 3245 of its 4096 cells.
def rough(+index: Nat, +step: Nat) -> F32:
  +n = Nat.mod(Nat.mul(index, step), 251n)
  +value = F32.div(F32.sub(F32.from_nat(n), 125.0), 7.0)
  Bool.pick(F32, Nat.is_eq(Nat.mod(n, 4n), 0n), F32.mul(value, 4096.0), value)

def word(+rounding: Bool, +index: Nat, +step: Nat) -> F32:
  Bool.pick(F32, rounding, rough(index, step), exact(index, step))

def words(fuel: Nat, +rounding: Bool, +index: Nat, +step: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      words(rest, rounding, 1n+index, step, word(rounding, index, step) <> acc)

def reversed(values: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match values:
    case Nil{}:
      acc
    case value <> rest:
      reversed(rest, value <> acc)

def matrix(+rounding: Bool, +rows: U32, +columns: U32, +step: Nat) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([rows, columns],
    reversed(words(Nat.mul(U32.to_nat(rows), U32.to_nat(columns)), rounding, 0n, step, Nil{}), Nil{}))

# The sequential definition: Matrix.multiply, the cell loop the laws state.
def sequential_result(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>) -> Maybe<&1, Core.Dynamic>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (a, b, out) = values
      Some{Core.pack(output, out)}

def sequential_planned(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Matrix.Error, Matrix.Plan<left, right>>) -> Maybe<&1, Core.Dynamic>:
  match result:
    case Fail{error}:
      None{}
    case Done{+plan}:
      sequential_result(left, right, Matrix.output_shape(left, right, plan), Matrix.multiply(left, right, plan, a, b))

def sequential(left: Core.Dynamic, right: Core.Dynamic) -> Maybe<&1, Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  sequential_planned(left_shape, right_shape, a, b, Matrix.plan(left_shape, right_shape))

def public(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (a, b, out) = values
      Some{out}

type Tally is Data:
  Tally{differing: U32, hash: U32}

def tally_found(rec: Core.Dynamic -> Core.Dynamic -> Tally -> Tally, +differing: U32, +hash: U32,
  mine: Core.Dynamic, kept: Core.Dynamic, found: Result<&2, &2, Core.Error, F32>, expected: Result<&2, &2, Core.Error, F32>) -> Tally:
  match found expected:
    case Done{x} Done{y}:
      +bits = F32.bits(x)
      rec(mine, kept, Tally{Bool.pick(U32, U32.is_eq(bits, F32.bits(y)), differing, (differing + 1 : U32)),
        (U32.mul(hash, 16777619) .^. bits : U32)})
    case _ _:
      rec(mine, kept, Tally{(differing + 1 : U32), hash})

def tally_word(rec: Core.Dynamic -> Core.Dynamic -> Tally -> Tally, tally: Tally,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>, theirs: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Tally:
  Tally{+differing, +hash} = tally
  (mine, found) = pair
  (kept, expected) = theirs
  tally_found(rec, differing, hash, mine, kept, found, expected)

def compare(fuel: Nat, +index: U32, mine: Core.Dynamic, theirs: Core.Dynamic, tally: Tally) -> Tally:
  match fuel:
    case 0n:
      tally
    case 1n+rest:
      tally_word(m => t => next => compare(rest, (index + 1 : U32), m, t, next), tally,
        Core.read_dynamic(mine, index), Core.read_dynamic(theirs, index))

def show(+name: String, +splits: Nat, tally: Tally) -> IO(Unit):
  Tally{differing, hash} = tally
  IO.print(name ++ " splits " ++ U32.show(U32.from_nat(splits)) ++ " differing " ++ U32.show(differing) ++ " hash " ++ U32.show(hash))

def report(+name: String, +splits: Nat, +cells: Nat, mine: Maybe<&1, Core.Dynamic>, theirs: Maybe<&1, Core.Dynamic>) -> IO(Unit):
  match mine theirs:
    case Some{m} Some{t}:
      show(name, splits, compare(cells, 0, m, t, Tally{0, 2166136261}))
    case _ _:
      IO.print(name ++ " failed")

def product_inputs(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>,
  c: Result<&2, &1, Core.Error, Core.Dynamic>, d: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b c d:
    case Done{a} Done{b} Done{c} Done{d}:
      report(name, Matrix.product_splits(depth, U32.to_nat(inner)), Nat.mul(U32.to_nat(rows), U32.to_nat(columns)),
        public(Matrix.matmul(a, b)), sequential(c, d))
    case _ _ _ _:
      IO.print(name ++ " inputs failed")

def product(+name: String, +rounding: Bool, +rows: U32, +inner: U32, +columns: U32, +depth: Nat) -> IO(Unit):
  product_inputs(name, rows, inner, columns, depth,
    matrix(rounding, rows, inner, 7n), matrix(rounding, inner, columns, 11n),
    matrix(rounding, rows, inner, 7n), matrix(rounding, inner, columns, 11n))

# The sequential definition of reduce_to: `ReduceFold.cell`, the masked fold
# that `reduce_fold_fiber` states, one target cell at a time over every
# contribution position.
def reduce_tally_found(rec: Core.Dynamic -> Core.Dynamic -> Tally -> Tally, +differing: U32, +hash: U32,
  mine: Core.Dynamic, input: Core.Dynamic, found: Result<&2, &2, Core.Error, F32>, expected: Result<&2, &1, Core.Error, F32>) -> Tally:
  match found expected:
    case Done{x} Done{y}:
      +bits = F32.bits(x)
      rec(mine, input, Tally{Bool.pick(U32, U32.is_eq(bits, F32.bits(y)), differing, (differing + 1 : U32)),
        (U32.mul(hash, 16777619) .^. bits : U32)})
    case _ _:
      rec(mine, input, Tally{(differing + 1 : U32), hash})

def reduce_tally_word(rec: Core.Dynamic -> Core.Dynamic -> Tally -> Tally, tally: Tally,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>, theirs: Core.Dynamic & Result<&2, &1, Core.Error, F32>) -> Tally:
  Tally{+differing, +hash} = tally
  (mine, found) = pair
  (input, expected) = theirs
  reduce_tally_found(rec, differing, hash, mine, input, found, expected)

def reduce_compare(fuel: Nat, +plan: Shape.BroadcastPlan, +index: U32, mine: Core.Dynamic, input: Core.Dynamic, tally: Tally) -> Tally:
  match fuel:
    case 0n:
      tally
    case 1n+rest:
      reduce_tally_word(m => i => next => reduce_compare(rest, plan, (index + 1 : U32), m, i, next), tally,
        Core.read_dynamic(mine, index), ReduceFold.cell(plan, index, input))

def reduce_report(+name: String, +cells: Nat, +splits: Nat, +plan: Shape.BroadcastPlan,
  result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> IO(Unit):
  (input, outcome) = result
  match outcome:
    case Done{out}:
      show(name, splits, reduce_compare(cells, plan, 0, out, input, Tally{0, 2166136261}))
    case Fail{error}:
      IO.print(name ++ " failed")

def reduce_planned(+name: String, +cells: Nat, +depth: Nat, input: Core.Dynamic, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> IO(Unit):
  match result:
    case Done{+plan}:
      reduce_report(name, cells, Reduce.splits(plan, depth, cells), plan, Reduce.reduce_to(input, target))
    case Fail{error}:
      IO.print(name ++ " failed")

def reduce_target(+name: String, +cells: Nat, +depth: Nat, -dims: List<&2, U32>,
  result: Shape.ConstructResult<dims>, input: Core.Dynamic) -> IO(Unit):
  match result:
    case Shape.SBuilt{+target, preserved}:
      Core.Packed{+shape, tensor} = input
      reduce_planned(name, cells, depth, Core.Packed{shape, tensor}, target, Shape.broadcast_checked(target, shape))
    case Shape.SError{error}:
      IO.print(name ++ " failed")

def reduce_input(+name: String, +cells: Nat, +depth: Nat, +target: List<&2, U32>, input: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match input:
    case Done{input}:
      reduce_target(name, cells, depth, target, Shape.construct(target), input)
    case Fail{error}:
      IO.print(name ++ " inputs failed")

def count(dims: List<&2, U32>) -> Nat:
  match dims:
    case Nil{}:
      1n
    case dim <> rest:
      Nat.mul(U32.to_nat(dim), count(rest))

# A reduction of a contribution of shape `dims` to `target`, whose `cells`
# cells sit in a storage of depth `depth`.
def reduction(+name: String, +rounding: Bool, +dims: List<&2, U32>, +target: List<&2, U32>, +cells: Nat, +depth: Nat) -> IO(Unit):
  reduce_input(name, cells, depth, target, Core.construct(dims, reversed(words(count(dims), rounding, 0n, 7n, Nil{}), Nil{})))

def broadcast_public(pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Done{out}:
      Some{out}
    case Fail{error}:
      None{}

def broadcast_loop_result(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (inputs, result) = pair
  match result:
    case Done{out}:
      Some{out}
    case Fail{error}:
      None{}

# The sequential definition of broadcast_to: the checked elementwise loop with
# KeepLeft, the input on the left and the +0.0 tensor of the target on the
# right, which broadcast_to ran before it built its output in parallel.
def broadcast_loop_zeroed(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  zeros: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Maybe<&1, Core.Dynamic>:
  match zeros:
    case Done{zeros}:
      broadcast_loop_result(Elementwise.allocated(Elementwise.KeepLeft{}, plan, input, Core.Packed{target, zeros},
        Core.new(Shape.output_shape(plan), 0.0)))
    case Fail{error}:
      None{}

def broadcast_loop_planned(input: Core.Dynamic, +target: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Maybe<&1, Core.Dynamic>:
  match result:
    case Done{+plan}:
      broadcast_loop_zeroed(input, target, plan, Core.new(target, 0.0))
    case Fail{error}:
      None{}

def broadcast_loop(input: Core.Dynamic, +target: Shape.Shape) -> Maybe<&1, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  broadcast_loop_planned(Core.Packed{shape, tensor}, target, Shape.broadcast_checked(shape, target))

def broadcast_inputs(+name: String, +cells: Nat, +depth: Nat, -dims: List<&2, U32>, result: Shape.ConstructResult<dims>,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result a b:
    case Shape.SBuilt{+target, preserved} Done{a} Done{b}:
      report(name, Elementwise.splits(depth), cells, broadcast_public(Broadcast.broadcast_to(a, target)), broadcast_loop(b, target))
    case _ _ _:
      IO.print(name ++ " inputs failed")

# broadcast_to of an input of shape `dims` to `target`, whose `cells` cells
# sit in a storage of depth `depth`.
def broadcast(+name: String, +dims: List<&2, U32>, +target: List<&2, U32>, +cells: Nat, +depth: Nat) -> IO(Unit):
  broadcast_inputs(name, cells, depth, target, Shape.construct(target),
    Core.construct(dims, reversed(words(count(dims), True{}, 0n, 7n, Nil{}), Nil{})),
    Core.construct(dims, reversed(words(count(dims), True{}, 0n, 7n, Nil{}), Nil{})))

def main() -> IO(Unit):
  do IO<Unit>:
    product("matmul-128x64x64", False{}, 128, 64, 64, 13n)
    product("matmul-96x40x80", False{}, 96, 40, 80, 13n)
    product("matmul-2x3x2", False{}, 2, 3, 2, 2n)
    product("matmul-64x48x64-rounding", True{}, 64, 48, 64, 12n)
    reduction("reduce-256x128-to-1x128-rounding", True{}, [256, 128], [1, 128], 128n, 7n)
    reduction("reduce-256x64-to-256x1-rounding", True{}, [256, 64], [256, 1], 256n, 8n)
    reduction("reduce-64x16x32-to-16x1-rounding", True{}, [64, 16, 32], [16, 1], 16n, 4n)
    reduction("reduce-64x16x32-to-64x1x1-rounding", True{}, [64, 16, 32], [64, 1, 1], 64n, 6n)
    reduction("reduce-6x5x4-to-5x1", False{}, [6, 5, 4], [5, 1], 5n, 3n)
    broadcast("broadcast-256-to-256x256", [256], [256, 256], 65536n, 16n)
    broadcast("broadcast-300x1-to-300x400", [300, 1], [300, 400], 120000n, 17n)
    broadcast("broadcast-scalar-to-64x1024", [], [64, 1024], 65536n, 16n)
    broadcast("broadcast-1x128x1-to-4x128x160", [1, 128, 1], [4, 128, 160], 81920n, 17n)
    broadcast("broadcast-3x1x5-to-7x3x4x5", [3, 1, 5], [7, 3, 4, 5], 420n, 9n)
