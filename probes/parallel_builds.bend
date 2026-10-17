import Base
import ../src/shape/core.bend as Shape
import ../src/tensor/core.bend as Core
import ../src/tensor/matmul.bend as Matrix
import ../src/tensor/layout.bend as Layout
import ../src/tensor/transforms.bend as Transforms
import ../src/tensor/permute-build.bend as Permute
import ../src/tensor/elementwise.bend as Elementwise
import ../src/tensor/reduction.bend as Reduction
import ../src/autograd/reduce.bend as Reduce
import ../src/autograd/reduce-fold.bend as ReduceFold

# The parallel builds next to the sequential definitions, on inputs large
# enough to fork. For each product, elementwise operation, sum, reduction
# and permute the program prints how many levels the
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

# matmul_nt and matmul_tn, which build their cells with product_tree and run
# eight cells of one row as one loop, next to Matrix.multiply over the
# transposed copy of the operand they read in place. Each output has more
# than 2^15 cells in a storage of 2^16, so it forks and pads, and a row of a
# column count that is not a multiple of eight puts some eight-cell subtrees
# across two rows, where the tree builds them cell by cell. Their inputs
# round: adding each cell's terms in decreasing k changes 26537 of the
# 33000 cells of the first and 25263 of the 33847 of the second.
def nt_reference(c: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (kept, flipped) = pair
  match flipped:
    case Fail{error}:
      None{}
    case Done{t}:
      sequential(c, t)

def tn_reference(d: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (kept, flipped) = pair
  match flipped:
    case Fail{error}:
      None{}
    case Done{t}:
      sequential(t, d)

def nt_inputs(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>,
  c: Result<&2, &1, Core.Error, Core.Dynamic>, d: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b c d:
    case Done{a} Done{b} Done{c} Done{d}:
      report(name, Matrix.product_splits(depth, U32.to_nat(inner)), Nat.mul(U32.to_nat(rows), U32.to_nat(columns)),
        public(Matrix.matmul_nt(a, b)), nt_reference(c, Transforms.transpose_dynamic(d, 0, 1)))
    case _ _ _ _:
      IO.print(name ++ " inputs failed")

def tn_inputs(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>,
  c: Result<&2, &1, Core.Error, Core.Dynamic>, d: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b c d:
    case Done{a} Done{b} Done{c} Done{d}:
      report(name, Matrix.product_splits(depth, U32.to_nat(inner)), Nat.mul(U32.to_nat(rows), U32.to_nat(columns)),
        public(Matrix.matmul_tn(a, b)), tn_reference(d, Transforms.transpose_dynamic(c, 0, 1)))
    case _ _ _ _:
      IO.print(name ++ " inputs failed")

# x @ W^T for x of [rows, inner] and W of [columns, inner].
def nt_product(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat) -> IO(Unit):
  nt_inputs(name, rows, inner, columns, depth,
    matrix(True{}, rows, inner, 7n), matrix(True{}, columns, inner, 11n),
    matrix(True{}, rows, inner, 7n), matrix(True{}, columns, inner, 11n))

# g^T @ x for g of [inner, rows] and x of [inner, columns].
def tn_product(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat) -> IO(Unit):
  tn_inputs(name, rows, inner, columns, depth,
    matrix(True{}, inner, rows, 7n), matrix(True{}, inner, columns, 11n),
    matrix(True{}, inner, rows, 7n), matrix(True{}, inner, columns, 11n))

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

def transformed(-input: Shape.Shape, pair: Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (source, result) = pair
  match result:
    case Fail{error}:
      None{}
    case Done{out}:
      Some{out}

# Elementwise builds next to the one-thread checked loop that
# `Elementwise.allocated` runs: it reads each input at the indices
# `Elementwise.input_indices` projects from the flat output index, with no
# broadcast walk and no input slot. The inputs exceed 2^15 cells, so the
# build forks, and their counts are not powers of two, so the output storage
# has padding cells past the count.
def elementwise_outcome(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (inputs, result) = pair
  match result:
    case Fail{error}:
      None{}
    case Done{out}:
      Some{out}

# The sequential definition of a permute: `Transforms.loop_allocated`, the
# checked copy loop the transform laws state.
def loop_planned(+input: Shape.Shape, +axes: List<&2, U32>, source: Core.Tensor<input>,
  planned: Result<&2, &2, Layout.Error, Layout.Plan<input, axes>>) -> Maybe<&1, Core.Dynamic>:
  match planned:
    case Fail{error}:
      None{}
    case Done{+plan}:
      transformed(input, Transforms.loop_allocated(input, axes, plan, source, Core.new(Layout.output(input, axes, plan), 0.0)))

def looped(dynamic: Core.Dynamic, +axes: List<&2, U32>) -> Maybe<&1, Core.Dynamic>:
  Core.Packed{+input, source} = dynamic
  loop_planned(input, axes, source, Layout.permutation(input, axes))

def permuted(pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Maybe<&1, Core.Dynamic>:
  (source, result) = pair
  match result:
    case Fail{error}:
      None{}
    case Done{out}:
      Some{out}

def permute_inputs(+name: String, +axes: List<&2, U32>, +cells: Nat, +depth: Nat,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b:
    case Done{a} Done{b}:
      report(name, Permute.splits(depth), cells, permuted(Transforms.permute_dynamic(a, axes)), looped(b, axes))
    case _ _:
      IO.print(name ++ " inputs failed")

# A permute of a tensor of shape `dims` by `axes`, whose `cells` cells sit in
# a storage of depth `depth`.
def permute(+name: String, +dims: List<&2, U32>, +axes: List<&2, U32>, +cells: Nat, +depth: Nat) -> IO(Unit):
  permute_inputs(name, axes, cells, depth,
    Core.construct(dims, reversed(words(count(dims), True{}, 0n, 7n, Nil{}), Nil{})),
    Core.construct(dims, reversed(words(count(dims), True{}, 0n, 7n, Nil{}), Nil{})))

def elementwise_sequential(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Maybe<&1, Core.Dynamic>:
  match result:
    case Fail{error}:
      None{}
    case Done{+plan}:
      elementwise_outcome(Elementwise.allocated(operation, plan, left, right, Core.new(Shape.output_shape(plan), 0.0)))

def elementwise_built(+name: String, +operation: Elementwise.Operation, +cells: Nat, +depth: Nat,
  a: Core.Dynamic, b: Core.Dynamic, c: Core.Dynamic, d: Core.Dynamic) -> IO(Unit):
  Core.Packed{+left, x} = c
  Core.Packed{+right, y} = d
  report(name, Elementwise.splits(depth), cells, elementwise_outcome(Elementwise.binary(operation, a, b)),
    elementwise_sequential(operation, Core.Packed{left, x}, Core.Packed{right, y}, Shape.broadcast_checked(left, right)))

def elementwise_inputs(+name: String, +operation: Elementwise.Operation, +cells: Nat, +depth: Nat,
  a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>,
  c: Result<&2, &1, Core.Error, Core.Dynamic>, d: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b c d:
    case Done{a} Done{b} Done{c} Done{d}:
      elementwise_built(name, operation, cells, depth, a, b, c, d)
    case _ _ _ _:
      IO.print(name ++ " inputs failed")

def tensor(+dims: List<&2, U32>, +step: Nat) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct(dims, reversed(words(count(dims), True{}, 0n, step, Nil{}), Nil{}))

# `operation` of a `left` and a `right` tensor of rounding words, whose
# output of `cells` cells sits in a storage of depth `depth`.
def elementwise(+name: String, +operation: Elementwise.Operation, +left: List<&2, U32>, +right: List<&2, U32>,
  +cells: Nat, +depth: Nat) -> IO(Unit):
  elementwise_inputs(name, operation, cells, depth, tensor(left, 7n), tensor(right, 11n), tensor(left, 7n), tensor(right, 11n))

# `Reduction.sum` next to `Reduction.natural_sum_loop`, the checked loop its
# laws state, on rounding words: adding them in another order changes the
# sum's bits. The sum never forks.
def sum_tally(found: Result<&2, &2, Core.Error, F32>, +expected: F32) -> Tally:
  match found:
    case Done{x}:
      +bits = F32.bits(x)
      Tally{Bool.pick(U32, U32.is_eq(bits, F32.bits(expected)), 0, 1), (U32.mul(2166136261, 16777619) .^. bits : U32)}
    case Fail{error}:
      Tally{1, 2166136261}

def sum_scalar(+expected: F32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Tally:
  (scalar, found) = pair
  sum_tally(found, expected)

def sum_public(+expected: F32, result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> Maybe<&2, Tally>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      (input, scalar) = pair
      Some{sum_scalar(expected, Core.read_dynamic(scalar, 0))}

def sum_sequential(-shape: Shape.Shape, mine: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>,
  theirs: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) -> Maybe<&2, Tally>:
  match theirs:
    case Fail{error}:
      None{}
    case Done{pair}:
      (tensor, +expected) = pair
      sum_public(expected, mine)

def sum_shown(+name: String, tally: Maybe<&2, Tally>) -> IO(Unit):
  match tally:
    case Some{tally}:
      show(name, 0n, tally)
    case None{}:
      IO.print(name ++ " failed")

def sum_built(+name: String, a: Core.Dynamic, b: Core.Dynamic) -> IO(Unit):
  Core.Packed{+shape, x} = b
  sum_shown(name, sum_sequential(shape, Reduction.sum_dynamic(a),
    Reduction.natural_sum_loop(Shape.elements(shape), shape, x, 0n, 0.0)))

def sum_inputs(+name: String, a: Result<&2, &1, Core.Error, Core.Dynamic>, b: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match a b:
    case Done{a} Done{b}:
      sum_built(name, a, b)
    case _ _:
      IO.print(name ++ " inputs failed")

def sum(+name: String, +dims: List<&2, U32>) -> IO(Unit):
  sum_inputs(name, tensor(dims, 7n), tensor(dims, 7n))

def main() -> IO(Unit):
  do IO<Unit>:
    product("matmul-128x64x64", False{}, 128, 64, 64, 13n)
    product("matmul-96x40x80", False{}, 96, 40, 80, 13n)
    product("matmul-2x3x2", False{}, 2, 3, 2, 2n)
    product("matmul-64x48x64-rounding", True{}, 64, 48, 64, 12n)
    nt_product("matmul_nt-200x40x165-rounding", 200, 40, 165, 16n)
    tn_product("matmul_tn-181x24x187-rounding", 181, 24, 187, 16n)
    reduction("reduce-256x128-to-1x128-rounding", True{}, [256, 128], [1, 128], 128n, 7n)
    reduction("reduce-256x64-to-256x1-rounding", True{}, [256, 64], [256, 1], 256n, 8n)
    reduction("reduce-64x16x32-to-16x1-rounding", True{}, [64, 16, 32], [16, 1], 16n, 4n)
    reduction("reduce-64x16x32-to-64x1x1-rounding", True{}, [64, 16, 32], [64, 1, 1], 64n, 6n)
    reduction("reduce-6x5x4-to-5x1", False{}, [6, 5, 4], [5, 1], 5n, 3n)
    permute("transpose-256x257", [256, 257], [1, 0], 65792n, 17n)
    permute("permute-16x64x40-201", [16, 64, 40], [2, 0, 1], 40960n, 16n)
    permute("permute-3x5x7-120", [3, 5, 7], [1, 2, 0], 105n, 7n)
    reduction("reduce-130x256-to-1x256-rounding", True{}, [130, 256], [1, 256], 256n, 8n)
    reduction("reduce-256x130-to-256x1-rounding", True{}, [256, 130], [256, 1], 256n, 8n)
    reduction("reduce-65536-to-scalar-rounding", True{}, [65536], [], 1n, 0n)
    elementwise("mul-100000-own-rounding", Elementwise.Mul{}, [100000], [100000], 100000n, 17n)
    elementwise("add-100000-scalar-rounding", Elementwise.Add{}, [100000], [], 100000n, 17n)
    elementwise("sub-scalar-100000-rounding", Elementwise.Sub{}, [], [100000], 100000n, 17n)
    elementwise("mul-320x320-row-rounding", Elementwise.Mul{}, [320, 320], [320], 102400n, 17n)
    elementwise("sub-320-320x320-rounding", Elementwise.Sub{}, [320], [320, 320], 102400n, 17n)
    elementwise("add-320x1-1x320-rounding", Elementwise.Add{}, [320, 1], [1, 320], 102400n, 17n)
    sum("sum-100000-rounding", [100000])
