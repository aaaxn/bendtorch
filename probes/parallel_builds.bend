import Base
import ../src/shape/core.bend as Shape
import ../src/tensor/core.bend as Core
import ../src/tensor/matmul.bend as Matrix

# The parallel builds next to the sequential definitions, on inputs large
# enough to fork. For each product the program prints how many levels the
# public build forks, how many output words differ from the sequential
# definition, and a rolling hash of the public words. tools/tensor_parallel.py
# runs it at several thread counts and requires every count to print the same
# bytes, with no differing word.

def word(+index: Nat, +step: Nat) -> F32:
  F32.sub(F32.div(F32.from_nat(Nat.mod(Nat.mul(index, step), 251n)), 64.0), 1.875)

def words(fuel: Nat, +index: Nat, +step: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      words(rest, 1n+index, step, word(index, step) <> acc)

def reversed(values: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match values:
    case Nil{}:
      acc
    case value <> rest:
      reversed(rest, value <> acc)

def matrix(+rows: U32, +columns: U32, +step: Nat) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([rows, columns], reversed(words(Nat.mul(U32.to_nat(rows), U32.to_nat(columns)), 0n, step, Nil{}), Nil{}))

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
      report(name, Matrix.splits(depth, U32.to_nat(inner)), Nat.mul(U32.to_nat(rows), U32.to_nat(columns)),
        public(Matrix.matmul(a, b)), sequential(c, d))
    case _ _ _ _:
      IO.print(name ++ " inputs failed")

def product(+name: String, +rows: U32, +inner: U32, +columns: U32, +depth: Nat) -> IO(Unit):
  product_inputs(name, rows, inner, columns, depth,
    matrix(rows, inner, 7n), matrix(inner, columns, 11n), matrix(rows, inner, 7n), matrix(inner, columns, 11n))

def main() -> IO(Unit):
  do IO<Unit>:
    product("matmul-128x64x64", 128, 64, 64, 13n)
    product("matmul-96x40x80", 96, 40, 80, 13n)
    product("matmul-2x3x2", 2, 3, 2, 2n)
