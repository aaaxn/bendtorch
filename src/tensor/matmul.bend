import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./elementwise.bend as Elementwise

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

def packed(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  result: Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{values}:
      (a, b, out) = values
      Done{(Core.pack(left, a), Core.pack(right, b), Core.pack(output, out))}

def planned(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Error, Plan<left, right>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+plan}:
      packed(left, right, output_shape(left, right, plan), multiply(left, right, plan, a, b))

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

def cell_leaf(pair: F32 & (Array<F32> & Array<F32>)) -> Array<F32> & (Array<F32> & Array<F32>):
  (value, inputs) = pair
  (ALeaf{value}, inputs)

# One output cell: its total when the cell is below `count`, else +0.0.
def cell(+inner: Nat, +columns: Nat, +position: Nat, la: Array<F32>, ra: Array<F32>, inside: Bool) ->
  Array<F32> & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (ALeaf{0.0}, (la, ra))
    case True{}:
      cell_leaf(cell_dot(inner, inner, columns, row_of(columns, position), column_of(columns, position), 0n, 0.0, la, ra))

# The cells `base .. base + 2^depth` as a balanced tree, built in index order
# on one thread.
def tree(+depth: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Array<F32> & (Array<F32> & Array<F32>):
  match depth:
    case 0n:
      cell(inner, columns, base, la, ra, Nat.is_lt(base, count))
    case 1n+lower:
      Elementwise.tree_node(l => r => tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r),
        tree(lower, inner, columns, count, base, la, ra))

def build(+depth: Nat, +splits: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Array<F32>:
  match depth splits:
    case 1n+lower 1n+rest:
      +high = Elementwise.upper(lower, base)
      Elementwise.fork(Array.clone(F32, la), Array.clone(F32, ra),
        l => r => build(lower, rest, inner, columns, count, base, l, r),
        l => r => build(lower, rest, inner, columns, count, high, l, r))
    case _ _:
      Pair.fst(Array<F32>, Array<F32> & Array<F32>, tree(depth, inner, columns, count, base, la, ra))

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
# The build forks until a subtree holds about 2^17 terms, and at most six
# levels: 64 subtrees, each with its own copy of both inputs. An output of
# 2^depth cells of `inner` terms forks depth + log2(inner) - 17 levels.
def splits(+depth: Nat, +inner: Nat) -> Nat:
  at_most(at_most(Nat.sub(Nat.add(depth, halvings(inner, inner)), 17n), 6n), depth)

# A run of the build next to its proof of balance, as `Elementwise.Certified`.
def cell_leaf_certified(pair: F32 & (Array<F32> & Array<F32>)) -> Elementwise.Certified<0n, cell_leaf(pair)>:
  (value, inputs) = pair
  Elementwise.Certified{cell_leaf((value, inputs)), {==}, Unit{}}

def cell_certified(+inner: Nat, +columns: Nat, +position: Nat, la: Array<F32>, ra: Array<F32>, inside: Bool) ->
  Elementwise.Certified<0n, cell(inner, columns, position, la, ra, inside)>:
  match inside:
    case False{}:
      Elementwise.Certified{cell(inner, columns, position, la, ra, False{}), {==}, Unit{}}
    case True{}:
      cell_leaf_certified(cell_dot(inner, inner, columns, row_of(columns, position), column_of(columns, position), 0n, 0.0, la, ra))

def node_joined(+lower: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>, a: Array<F32>,
  -l: Array<F32>, -r: Array<F32>,
  same: {(a, (l, r)) == tree(lower, inner, columns, count, base, la, ra) : Array<F32> & (Array<F32> & Array<F32>)},
  balanced: Model.Balanced(F32, lower, a),
  high: Elementwise.Certified<lower, tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r)>) ->
  Elementwise.Certified<1n+lower, tree(1n+lower, inner, columns, count, base, la, ra)>:
  Elementwise.Certified{result, high_same, high_balanced} = high
  (b, rest) = result
  Elementwise.Certified{(ANode{a, b}, rest),
    %same : {(ANode{a, b}, rest) == Elementwise.tree_node(x => y => tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), x, y), _) : Array<F32> & (Array<F32> & Array<F32>)}
    %high_same : {(ANode{a, b}, rest) == Elementwise.tree_join(a, _) : Array<F32> & (Array<F32> & Array<F32>)}
    {==},
    (balanced, high_balanced)}

def node_split(+lower: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>,
  result: Array<F32> & (Array<F32> & Array<F32>),
  same: {result == tree(lower, inner, columns, count, base, la, ra) : Array<F32> & (Array<F32> & Array<F32>)},
  balanced: Model.Balanced(F32, lower, Pair.fst(Array<F32>, Array<F32> & Array<F32>, result)),
  next: @l: Array<F32> -> @r: Array<F32> -> Elementwise.Certified<lower, tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r)>) ->
  Elementwise.Certified<1n+lower, tree(1n+lower, inner, columns, count, base, la, ra)>:
  (a, inputs) = result
  (l, r) = inputs
  node_joined(lower, inner, columns, count, base, la, ra, a, l, r, same, balanced, next(l, r))

def tree_node_certified(+lower: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>,
  low: Elementwise.Certified<lower, tree(lower, inner, columns, count, base, la, ra)>,
  next: @l: Array<F32> -> @r: Array<F32> -> Elementwise.Certified<lower, tree(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r)>) ->
  Elementwise.Certified<1n+lower, tree(1n+lower, inner, columns, count, base, la, ra)>:
  Elementwise.Certified{result, same, balanced} = low
  node_split(lower, inner, columns, count, base, la, ra, result, same, balanced, next)

def tree_certified(+depth: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, la: Array<F32>, ra: Array<F32>) ->
  Elementwise.Certified<depth, tree(depth, inner, columns, count, base, la, ra)>:
  match depth:
    case 0n:
      cell_certified(inner, columns, base, la, ra, Nat.is_lt(base, count))
    case 1n+lower:
      tree_node_certified(lower, inner, columns, count, base, la, ra,
        tree_certified(lower, inner, columns, count, base, la, ra),
        l => r => tree_certified(lower, inner, columns, count, Nat.add(Model.capacity(lower), base), l, r))

def tree_balanced(+depth: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat, -la: Array<F32>, -ra: Array<F32>,
  certified: Elementwise.Certified<depth, tree(depth, inner, columns, count, base, la, ra)>) ->
  Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32> & Array<F32>, tree(depth, inner, columns, count, base, la, ra))):
  Elementwise.Certified{result, same, balanced} = certified
  %same : Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32> & Array<F32>, _))
  balanced

# The parallel build returns a balanced tree of its depth, whatever the split.
def build_balanced(+depth: Nat, +splits: Nat, +inner: Nat, +columns: Nat, +count: Nat, +base: Nat,
  la: Array<F32>, ra: Array<F32>) -> Model.Balanced(F32, depth, build(depth, splits, inner, columns, count, base, la, ra)):
  match depth splits:
    case 1n+lower 1n+rest:
      +high = Elementwise.upper(lower, base)
      Elementwise.fork_balanced(lower, Array.clone(F32, la), Array.clone(F32, ra),
        l => r => build(lower, rest, inner, columns, count, base, l, r),
        l => r => build(lower, rest, inner, columns, count, high, l, r),
        l => r => build_balanced(lower, rest, inner, columns, count, base, l, r),
        l => r => build_balanced(lower, rest, inner, columns, count, high, l, r))
    case 0n _:
      tree_balanced(0n, inner, columns, count, base, la, ra, tree_certified(0n, inner, columns, count, base, la, ra))
    case 1n+lower 0n:
      tree_balanced(1n+lower, inner, columns, count, base, la, ra, tree_certified(1n+lower, inner, columns, count, base, la, ra))

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

def parallel_matmul(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  parallel_planned(left_shape, right_shape, a, b, plan(left_shape, right_shape))

def matmul(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  planned(left_shape, right_shape, a, b, plan(left_shape, right_shape))
