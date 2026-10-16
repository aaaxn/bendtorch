import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./elementwise.bend as Elementwise
import ./parallel.bend as Parallel

type Error is Data:
  TensorError{error: Core.Error}
  ShapeError{error: Shape.Error}
  RankMismatch{}
  InnerDimensionMismatch{}

type Plan<-left: Shape.Shape, -right: Shape.Shape> is Data:
  Plan{output: Shape.Shape, rows: U32, inner: U32, right_inner: U32, columns: U32,
    -left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    -right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    -output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def output_shape(-left: Shape.Shape, -right: Shape.Shape, plan: Plan<left, right>) -> Shape.Shape:
  Plan{shape, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  shape

def plan_output(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) -> Result<&2, &2, Error, Plan<left, right>>:
  match result:
    case Shape.SError{error}:
      Fail{ShapeError{error}}
    case Shape.SBuilt{shape, preserved}:
      Done{Plan{shape, rows, inner, right_inner, columns, le, re, preserved, compatible}}

def plan_compatible(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{InnerDimensionMismatch{}}
    case Shape.BTrue{equal}:
      plan_output(left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal),
        Shape.construct([rows, columns]))

def plan_right(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, dimensions: List<&2, U32>,
  -same: {dimensions == Shape.dimensions(right) : List<&2, U32>}) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match dimensions:
    case +right_inner <> +columns <> Nil{}:
      plan_compatible(left, right, rows, inner, right_inner, columns, le,
        Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same),
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))
    case rest:
      Fail{RankMismatch{}}

def plan_left(+left: Shape.Shape, +right: Shape.Shape, dimensions: List<&2, U32>,
  -same: {dimensions == Shape.dimensions(left) : List<&2, U32>}) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match dimensions:
    case +rows <> +inner <> Nil{}:
      plan_right(left, right, rows, inner,
        Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same),
        Shape.dimensions(right), {==})
    case rest:
      Fail{RankMismatch{}}

def plan(+left: Shape.Shape, +right: Shape.Shape) -> Result<&2, &2, Error, Plan<left, right>>:
  plan_left(left, right, Shape.dimensions(left), {==})

def right_read(-left: Shape.Shape, -right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  a: Core.Tensor<left>, left_value: F32, accumulator: F32,
  pair: Core.Tensor<right> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  (b, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{right_value}:
      rec(a, b, F32.add(accumulator, F32.mul(left_value, right_value)))

def left_read(-left: Shape.Shape, +right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  b: Core.Tensor<right>, coordinates: List<&2, U32>, accumulator: F32,
  pair: Core.Tensor<left> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  (a, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{value}:
      right_read(left, right, rec, a, value, accumulator, Core.read_coordinate(right, b, coordinates))

def dot_decided(+left: Shape.Shape, +right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  a: Core.Tensor<left>, b: Core.Tensor<right>, +row: U32, +column: U32, +cursor: Nat, +index: U32, accumulator: F32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{TensorError{Core.StorageError{Storage.InvalidIndex{}}}}
    case Shape.BTrue{equal}:
      left_read(left, right, rec, b, [index, column], accumulator, Core.read_coordinate(left, a, [row, index]))

def natural_dot(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  +row: U32, +column: U32, +cursor: Nat, accumulator: F32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  match fuel:
    case 0n:
      Done{(a, b, accumulator)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      dot_decided(left, right,
        a => b => value => natural_dot(tail, left, right, a, b, row, column, 1n+cursor, value),
        a, b, row, column, cursor, index, accumulator, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def dot(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  +row: U32, +column: U32, +inner_index: U32, accumulator: F32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  natural_dot(fuel, left, right, a, b, row, column, U32.to_nat(inner_index), accumulator)

def wrote(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  a: Core.Tensor<left>, b: Core.Tensor<right>,
  pair: Core.Tensor<output> & Result<&2, &2, Core.Error, Unit>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  (out, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{unit}:
      rec(a, b, out)

def dotted(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  out: Core.Tensor<output>, index: U32,
  result: Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{values}:
      (a, b, value) = values
      wrote(left, right, output, rec, a, b, Core.write_flat(output, out, index, value))

def product_coordinates(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32, coordinates: List<&2, U32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match coordinates:
    case +row <> +column <> Nil{}:
      dotted(left, right, output, rec, out, index, natural_dot(U32.to_nat(inner), left, right, a, b, row, column, 0n, 0.0))
    case rest:
      Fail{RankMismatch{}}

def product_unflattened(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{ShapeError{error}}
    case Done{coordinate}:
      product_coordinates(left, right, output, rec, inner, a, b, out, index, Shape.coordinate_values(output, coordinate))

def product_decided(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{TensorError{Core.StorageError{Storage.InvalidIndex{}}}}
    case Shape.BTrue{equal}:
      product_unflattened(left, right, output, rec, inner, a, b, out, index, Shape.unflatten_checked(output, index))

def natural_product_loop(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +cursor: Nat) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match fuel:
    case 0n:
      Done{(a, b, out)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      product_decided(left, right, output,
        a => b => out => natural_product_loop(tail, left, right, output, inner, columns, a, b, out, 1n+cursor),
        inner, columns, a, b, out, cursor, index, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def product_loop(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, U32.to_nat(index))

def created(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{out}:
      natural_product_loop(Shape.elements(output), left, right, output,
        inner, columns, a, b, out, 0n)

def multiply(+left: Shape.Shape, +right: Shape.Shape, +plan: Plan<left, right>,
  a: Core.Tensor<left>, b: Core.Tensor<right>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output_shape(left, right, plan)>>:
  match plan:
    case Plan{+output, rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      created(left, right, output, inner, columns, a, b, Core.new(output, 0.0))

# The parallel product. Output cell `position` of a rows x columns product
# folds row `position / columns` of the left array with column
# `position % columns` of the right array: it adds left[row, k] * right[k,
# column] for k = 0, 1, ..., inner - 1, starting from +0.0, the same F32 terms
# in the same order as `natural_dot`. The cells are built like
# `Elementwise.build` builds its output: the top `splits` levels of the
# output tree fork, each half with its own clone of both input arrays, and a
# thread builds the cells below the forks in index order.

# Output cell `position` of a product with `columns` columns sits at row
# `position / columns` and column `position % columns`, as the digits of the
# output shape spell them.
def row_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(position, Nat.mul(columns, 1n))

def column_of(+columns: Nat, +position: Nat) -> Nat:
  Nat.div(Nat.mod(position, Nat.mul(columns, 1n)), 1n)

def left_index(+inner: Nat, +row: Nat, +k: Nat) -> U32:
  U32.from_nat(Nat.add(Nat.mul(row, inner), k))

def right_index(+columns: Nat, +column: Nat, +k: Nat) -> U32:
  U32.from_nat(Nat.add(Nat.mul(k, columns), column))

def term_right(rec: F32 -> Array<F32> -> Array<F32> -> F32 & (Array<F32> & Array<F32>), accumulator: F32, x: F32,
  la: Array<F32>, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<F32>):
  (ra, y) = pair
  rec(F32.add(accumulator, F32.mul(x, y)), la, ra)

def term_left(rec: F32 -> Array<F32> -> Array<F32> -> F32 & (Array<F32> & Array<F32>), accumulator: F32, ra: Array<F32>,
  +right_at: U32, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<F32>):
  (la, x) = pair
  term_right(rec, accumulator, x, la, Array.get(F32, ra, right_at))

# The terms k = cursor .. cursor + fuel - 1 of cell (row, column), added to
# `accumulator`. Both arrays come back with the total.
def cell_dot(fuel: Nat, +inner: Nat, +columns: Nat, +row: Nat, +column: Nat, +cursor: Nat, accumulator: F32,
  la: Array<F32>, ra: Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  match fuel:
    case 0n:
      (accumulator, (la, ra))
    case 1n+rest:
      term_left(value => l => r => cell_dot(rest, inner, columns, row, column, 1n+cursor, value, l, r),
        accumulator, ra, right_index(columns, column, cursor), Array.get(F32, la, left_index(inner, row, cursor)))

# Cell `position`: all `inner` terms, from +0.0.
def dot_of(+inner: Nat, +columns: Nat, +position: Nat, la: Array<F32>, ra: Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  cell_dot(inner, inner, columns, row_of(columns, position), column_of(columns, position), 0n, 0.0, la, ra)

# One output cell: its total when the cell is below `count`, else +0.0.
def cell(+inner: Nat, +columns: Nat, +position: Nat, la: Array<F32>, ra: Array<F32>, inside: Bool) ->
  F32 & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (0.0, (la, ra))
    case True{}:
      dot_of(inner, columns, position, la, ra)

# What every cell of a build needs: the inner extent and the column count.
type Dims is Data:
  Dims{inner: Nat, columns: Nat}

# Each thread below the forks reads its own copy.
def dims(+given: Dims) -> Dims:
  Dims{+inner, +columns} = given
  Dims{inner, columns}

# The cell function `Parallel.build` runs: its state is the pair of input
# arrays, split for a fork as `Elementwise.split` splits them.
def cell_at(+dims: Dims, +position: Nat, inputs: Array<F32> & Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  Dims{+inner, +columns} = dims
  (la, ra) = inputs
  cell(inner, columns, position, la, ra, inside)

def build(+depth: Nat, +splits: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Array<F32>:
  Parallel.build(~Dims, ~Dims, ~Array<F32>, ~Array<F32>, ~dims, ~Elementwise.split, ~cell_at, Dims{inner, columns}, depth, splits, count, base, (la, ra))

def halvings_step(rec: Nat -> Nat, +number: Nat, small: Bool) -> Nat:
  match small:
    case True{}:
      0n
    case False{}:
      1n+rec(Nat.div(number, 2n))

# floor(log2(number)) for a positive number, 0 for 0. `fuel` bounds the
# halvings; `number` itself is enough.
def halvings(fuel: Nat, +number: Nat) -> Nat:
  match fuel:
    case 0n:
      0n
    case 1n+rest:
      halvings_step(n => halvings(rest, n), number, Nat.is_lt(number, 2n))

def least(+a: Nat, +b: Nat, smaller: Bool) -> Nat:
  match smaller:
    case True{}:
      a
    case False{}:
      b

def at_most(+a: Nat, +b: Nat) -> Nat:
  least(a, b, Nat.is_lt(a, b))

# Split by work, not by cell count: a cell costs one term per inner index.
# The build forks until a subtree holds about 2^13 terms, and at most six
# levels, 64 subtrees, each with its own copy of both inputs: an output of
# 2^depth cells of `inner` terms forks depth + log2(inner) - 13 levels. On
# medusa2 at 8 threads a product of 2^15 terms or more ran faster with every
# extra level up to six, and forking cost nothing measurable on one thread.
def splits(+depth: Nat, +inner: Nat) -> Nat:
  at_most(at_most(Nat.sub(Nat.add(depth, halvings(inner, inner)), 13n), 6n), depth)

# The parallel build returns a balanced tree of its depth, whatever the split.
def build_balanced(+depth: Nat, +splits: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Model.Balanced(F32, depth, build(depth, splits, inner, columns, count, base, la, ra)):
  Parallel.balanced(~Dims, ~Dims, ~Array<F32>, ~Array<F32>, ~dims, ~Elementwise.split, ~cell_at, Dims{inner, columns}, depth, splits, count, base, (la, ra))

# Both inputs come back to the caller as the kept halves of their clones,
# and the build reads the other halves.
def parallel_output(+output: Shape.Shape, +inner: Nat, +columns: Nat, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  left: Core.Dynamic & Array<F32>, right: Core.Dynamic & Array<F32>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  (kept_left, la) = left
  (kept_right, ra) = right
  Done{(kept_left, kept_right, Core.pack(output, Core.TTensor{Storage.Owned{depth, length,
    build(depth, splits(depth, inner), inner, columns, length, 0n, la, ra),
    build_balanced(depth, splits(depth, inner), inner, columns, length, 0n, la, ra),
    valid}, exact}))}

def parallel_allocated(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: Nat, +columns: Nat,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Done{(Core.pack(left, a), Core.pack(right, b), Core.pack(output, Core.TTensor{Storage.Empty{}, exact}))}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          parallel_output(output, inner, columns, depth, length, valid, exact,
            Elementwise.take(Core.pack(left, a)), Elementwise.take(Core.pack(right, b)))

def parallel_multiply(+left: Shape.Shape, +right: Shape.Shape, +plan: Plan<left, right>,
  a: Core.Tensor<left>, b: Core.Tensor<right>) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match plan:
    case Plan{+output, rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      parallel_allocated(left, right, output, U32.to_nat(inner), U32.to_nat(columns), a, b, Core.new(output, 0.0))

def parallel_planned(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Error, Plan<left, right>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+plan}:
      parallel_multiply(left, right, plan, a, b)

# `matmul` builds its output in parallel. `multiply`, `natural_product_loop`
# and `natural_dot` above are the sequential definition of the same cells,
# which their own laws state.
def matmul(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  parallel_planned(left_shape, right_shape, a, b, plan(left_shape, right_shape))

# Products that read an operand in place. `x @ W^T` and `g^T @ x` read a
# transposed operand; instead of building the transpose they read the
# operand at the transposed index. An operand of a product is laid out in
# one of two orders: `Kept`, as `[outer, inner]`, where term k of output row
# (or column) `outer` sits at outer * inner + k; or `Swapped`, as
# `[inner, outer]`, where it sits at k * outers + outer. `matmul` is the
# product of a `Kept` left and a `Swapped` right operand, `matmul_nt` of two
# `Kept` operands and `matmul_tn` of two `Swapped` ones. Cell (row, column)
# adds the inner terms in increasing k from +0.0, as `natural_dot` does, and
# the cells are built over the output with `Parallel.build`, as `matmul`
# builds them.
type Order is Data:
  Kept{}
  Swapped{}

def read_offset(+order: Order, +inner: Nat, +outers: Nat, +outer: Nat, +k: Nat) -> Nat:
  match order:
    case Kept{}:
      Nat.add(Nat.mul(outer, inner), k)
    case Swapped{}:
      Nat.add(Nat.mul(k, outers), outer)

def read_index(+order: Order, +inner: Nat, +outers: Nat, +outer: Nat, +k: Nat) -> U32:
  U32.from_nat(read_offset(order, inner, outers, outer, k))

# The terms k = cursor .. cursor + fuel of cell (row, column) of a rows x
# columns product, fuel + 1 of them, added to `accumulator`; `left` and
# `right` are the reads of term `cursor`, already made. Each step reads the
# next term's words, so the loop is one tail call with no closure: it
# compiles to a flat loop.
def product_run(fuel: Nat, +lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat, +row: Nat, +column: Nat,
  +cursor: Nat, accumulator: F32, left: Array<F32> & F32, right: Array<F32> & F32) -> F32 & (Array<F32> & Array<F32>):
  match fuel left right:
    case 0n Tuple{la, x} Tuple{ra, y}:
      (F32.add(accumulator, F32.mul(x, y)), (la, ra))
    case 1n+rest Tuple{la, x} Tuple{ra, y}:
      product_run(rest, lo, ro, inner, rows, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(x, y)),
        Array.get(F32, la, read_index(lo, inner, rows, row, 1n+cursor)), Array.get(F32, ra, read_index(ro, inner, columns, column, 1n+cursor)))

# The terms k = cursor .. cursor + fuel - 1 of cell (row, column) of a
# rows x columns product, added to `accumulator`.
def product_dot(fuel: Nat, +lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat, +row: Nat, +column: Nat,
  +cursor: Nat, accumulator: F32, la: Array<F32>, ra: Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  match fuel:
    case 0n:
      (accumulator, (la, ra))
    case 1n+rest:
      product_run(rest, lo, ro, inner, rows, columns, row, column, cursor, accumulator,
        Array.get(F32, la, read_index(lo, inner, rows, row, cursor)), Array.get(F32, ra, read_index(ro, inner, columns, column, cursor)))

def product_dot_of(+lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat, +position: Nat,
  la: Array<F32>, ra: Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  product_dot(inner, lo, ro, inner, rows, columns, row_of(columns, position), column_of(columns, position), 0n, 0.0, la, ra)

def product_cell(+lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat, +position: Nat,
  la: Array<F32>, ra: Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (0.0, (la, ra))
    case True{}:
      product_dot_of(lo, ro, inner, rows, columns, position, la, ra)

# What every cell of a product build reads.
type Layout is Data:
  Layout{left: Order, right: Order, inner: Nat, rows: Nat, columns: Nat}

def layout(+given: Layout) -> Layout:
  Layout{+lo, +ro, +inner, +rows, +columns} = given
  Layout{lo, ro, inner, rows, columns}

def product_cell_at(+layout: Layout, +position: Nat, inputs: Array<F32> & Array<F32>, inside: Bool) ->
  F32 & (Array<F32> & Array<F32>):
  Layout{+lo, +ro, +inner, +rows, +columns} = layout
  (la, ra) = inputs
  product_cell(lo, ro, inner, rows, columns, position, la, ra, inside)

def product_build(+depth: Nat, +splits: Nat, +lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat,
  +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) -> Array<F32>:
  Parallel.build(~Layout, ~Layout, ~Array<F32>, ~Array<F32>, ~layout, ~Elementwise.split, ~product_cell_at,
    Layout{lo, ro, inner, rows, columns}, depth, splits, count, base, (la, ra))

def product_build_balanced(+depth: Nat, +splits: Nat, +lo: Order, +ro: Order, +inner: Nat, +rows: Nat, +columns: Nat,
  +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Model.Balanced(F32, depth, product_build(depth, splits, lo, ro, inner, rows, columns, count, base, la, ra)):
  Parallel.balanced(~Layout, ~Layout, ~Array<F32>, ~Array<F32>, ~layout, ~Elementwise.split, ~product_cell_at,
    Layout{lo, ro, inner, rows, columns}, depth, splits, count, base, (la, ra))

# The dimensions of an operand of `outer` rows (or columns) and `inner`
# terms laid out in `order`.
def arranged(+order: Order, +outer: U32, +inner: U32) -> List<&2, U32>:
  match order:
    case Kept{}:
      [outer, inner]
    case Swapped{}:
      [inner, outer]

type Product<-lo: Order, -ro: Order, -left: Shape.Shape, -right: Shape.Shape> is Data:
  Product{output: Shape.Shape, rows: U32, inner: U32, right_inner: U32, columns: U32,
    -left_dimensions: {Shape.dimensions(left) == arranged(lo, rows, inner) : List<&2, U32>},
    -right_dimensions: {Shape.dimensions(right) == arranged(ro, columns, right_inner) : List<&2, U32>},
    -output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def product_shape(-lo: Order, -ro: Order, -left: Shape.Shape, -right: Shape.Shape, plan: Product<lo, ro, left, right>) -> Shape.Shape:
  Product{shape, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  shape

# A rank-2 operand split into its outer and inner extents.
type Split<-order: Order, -dims: List<&2, U32>> is Data:
  Split{outer: U32, inner: U32, same: {dims == arranged(order, outer, inner) : List<&2, U32>}}

def split_dims(+order: Order, dims: List<&2, U32>) -> Result<&2, &2, Error, Split<order, dims>>:
  match order:
    case Kept{}:
      match dims:
        case +a <> +b <> Nil{}:
          Done{Split{a, b, {==}}}
        case rest:
          Fail{RankMismatch{}}
    case Swapped{}:
      match dims:
        case +a <> +b <> Nil{}:
          Done{Split{b, a, {==}}}
        case rest:
          Fail{RankMismatch{}}

def product_output(-lo: Order, -ro: Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == arranged(lo, rows, inner) : List<&2, U32>},
  -re: {Shape.dimensions(right) == arranged(ro, columns, right_inner) : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) -> Result<&2, &2, Error, Product<lo, ro, left, right>>:
  match result:
    case Shape.SError{error}:
      Fail{ShapeError{error}}
    case Shape.SBuilt{shape, preserved}:
      Done{Product{shape, rows, inner, right_inner, columns, le, re, preserved, compatible}}

def product_compatible(-lo: Order, -ro: Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == arranged(lo, rows, inner) : List<&2, U32>},
  -re: {Shape.dimensions(right) == arranged(ro, columns, right_inner) : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Result<&2, &2, Error, Product<lo, ro, left, right>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{InnerDimensionMismatch{}}
    case Shape.BTrue{equal}:
      product_output(lo, ro, left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal),
        Shape.construct([rows, columns]))

def product_right(-lo: Order, -ro: Order, -left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == arranged(lo, rows, inner) : List<&2, U32>},
  result: Result<&2, &2, Error, Split<ro, Shape.dimensions(right)>>) -> Result<&2, &2, Error, Product<lo, ro, left, right>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{Split{+columns, +right_inner, same}}:
      product_compatible(lo, ro, left, right, rows, inner, right_inner, columns, le, same,
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))

def product_left(-lo: Order, +ro: Order, -left: Shape.Shape, +right: Shape.Shape,
  result: Result<&2, &2, Error, Split<lo, Shape.dimensions(left)>>) -> Result<&2, &2, Error, Product<lo, ro, left, right>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{Split{+rows, +inner, same}}:
      product_right(lo, ro, left, right, rows, inner, same, split_dims(ro, Shape.dimensions(right)))

# The plan of a product: both operands rank 2, laid out in their orders,
# with equal inner extents.
def product_plan(+lo: Order, +ro: Order, +left: Shape.Shape, +right: Shape.Shape) -> Result<&2, &2, Error, Product<lo, ro, left, right>>:
  product_left(lo, ro, left, right, split_dims(lo, Shape.dimensions(left)))

def product_parallel_output(+lo: Order, +ro: Order, +output: Shape.Shape, +inner: Nat, +rows: Nat, +columns: Nat,
  +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  left: Core.Dynamic & Array<F32>, right: Core.Dynamic & Array<F32>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  (kept_left, la) = left
  (kept_right, ra) = right
  Done{(kept_left, kept_right, Core.pack(output, Core.TTensor{Storage.Owned{depth, length,
    product_build(depth, splits(depth, inner), lo, ro, inner, rows, columns, length, 0n, la, ra),
    product_build_balanced(depth, splits(depth, inner), lo, ro, inner, rows, columns, length, 0n, la, ra),
    valid}, exact}))}

def product_allocated(+lo: Order, +ro: Order, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +inner: Nat, +rows: Nat, +columns: Nat,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          Done{(Core.pack(left, a), Core.pack(right, b), Core.pack(output, Core.TTensor{Storage.Empty{}, exact}))}
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          product_parallel_output(lo, ro, output, inner, rows, columns, depth, length, valid, exact,
            Elementwise.take(Core.pack(left, a)), Elementwise.take(Core.pack(right, b)))

def product_multiply(+lo: Order, +ro: Order, +left: Shape.Shape, +right: Shape.Shape, +plan: Product<lo, ro, left, right>,
  a: Core.Tensor<left>, b: Core.Tensor<right>) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match plan:
    case Product{+output, +rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      product_allocated(lo, ro, left, right, output, U32.to_nat(inner), U32.to_nat(rows), U32.to_nat(columns), a, b,
        Core.new(output, 0.0))

def product_planned(+lo: Order, +ro: Order, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Error, Product<lo, ro, left, right>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+plan}:
      product_multiply(lo, ro, left, right, plan, a, b)

# The product of two rank-2 operands laid out in orders lo and ro. Both
# inputs come back to the caller as the kept halves of their clones.
def product(+lo: Order, +ro: Order, left: Core.Dynamic, right: Core.Dynamic) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  product_planned(lo, ro, left_shape, right_shape, a, b, product_plan(lo, ro, left_shape, right_shape))

# x @ W^T for x of [rows, inner] and W of [columns, inner]: cell (row,
# column) adds x[row, k] * W[column, k], W read in place.
def matmul_nt(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  product(Kept{}, Kept{}, left, right)

# g^T @ x for g of [inner, rows] and x of [inner, columns]: cell (row,
# column) adds g[k, row] * x[k, column], g read in place.
def matmul_tn(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  product(Swapped{}, Swapped{}, left, right)
