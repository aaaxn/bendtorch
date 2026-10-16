import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/layout.bend as Layout
import ../tensor/matmul.bend as Matrix

# Reduction of a broadcast contribution back to a target shape. The target
# must broadcast into the contribution's shape. Target cell t holds the
# contribution words whose right-aligned projection lands on t, added one
# F32 add at a time in increasing contribution position, starting from
# +0.0. Cells are independent, so the output is built in parallel below.

type Error is Data:
  TensorError{error: Core.Error}
  NotReducible{}

# The parallel reduction. Target cell `target` sums the contribution words
# whose broadcast projection lands on it, in increasing contribution
# position from +0.0, the same F32 adds in the same order as the loop above.
# It walks only those words: over the index rows of the plan, one row per
# contribution axis, a row whose left stride is 0 (an axis the target
# reduces) visits each of its digits in increasing order, and any other row
# takes the one digit that `target` names there. The cells are built like
# `Matrix.build` builds its output: the top `splits` levels of the output
# tree fork, each half with its own clone of the contribution array.

def leaf_add(accumulator: F32, pair: Array<F32> & F32) -> F32 & Array<F32>:
  (kept, word) = pair
  (F32.add(accumulator, word), kept)

# How many digits a row walks for a target cell: every digit of an axis the
# target reduces (left stride 0), else one.
def span(+left: Nat, +extent: Nat) -> Nat:
  match left:
    case 0n:
      extent
    case 1n+less:
      1n

def first_span(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>) -> Nat:
  match rows extents:
    case Shape.Row{stride, +left, right} <> rest +extent <> others:
      span(left, extent)
    case rows extents:
      0n

# The first digit a row walks: 0 on a reduced axis (`Nat.div` by 0 is 0),
# else the digit `target` names there.
def first_digit(+rows: List<&2, Shape.Row>, +target: Nat) -> Nat:
  match rows:
    case Nil{}:
      0n
    case Shape.Row{stride, +left, right} <> rest:
      Nat.div(target, left)

def walk_next(rec: F32 -> Array<F32> -> F32 & Array<F32>, pair: F32 & Array<F32>) -> F32 & Array<F32>:
  (accumulator, array) = pair
  rec(accumulator, array)

# The words of the fiber of `target` below these rows, added to
# `accumulator` in increasing position. `extents` lists the contribution
# extent of each row and `base` is the contribution offset the rows above
# chose. The first row walks `fuel` digits from `digit`, each at offset
# base + right * digit, and the rows below get `target % left` (all of
# `target` on a reduced axis, as `Nat.mod` by 0 keeps it).
def fiber(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, fuel: Nat, +target: Nat, +base: Nat, +digit: Nat,
  accumulator: F32, array: Array<F32>) -> F32 & Array<F32>:
  match rows extents fuel:
    case Nil{} _ _:
      leaf_add(accumulator, Array.get(F32, array, U32.from_nat(base)))
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n:
      (accumulator, array)
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining:
      +inner = Nat.mod(target, left)
      walk_next(value => kept => fiber(rows, extents, remaining, target, base, 1n+digit, value, kept),
        fiber(rest, others, first_span(rest, others), inner, Nat.add(base, Nat.mul(right, digit)), first_digit(rest, inner),
          accumulator, array))
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _:
      (accumulator, array)

# Cell `target` of the reduction: its whole fiber from +0.0.
def fiber_of(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +target: Nat, array: Array<F32>) -> F32 & Array<F32>:
  fiber(rows, extents, first_span(rows, extents), target, 0n, first_digit(rows, target), 0.0, array)

def cell_leaf(pair: F32 & Array<F32>) -> Array<F32> & Array<F32>:
  (value, array) = pair
  (ALeaf{value}, array)

# One target cell: its fiber sum from +0.0 when the cell is below `count`,
# else +0.0.
def cell(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +position: Nat, array: Array<F32>, inside: Bool) ->
  Array<F32> & Array<F32>:
  match inside:
    case False{}:
      (ALeaf{0.0}, array)
    case True{}:
      cell_leaf(fiber_of(rows, extents, position, array))

def tree_join(a: Array<F32>, high: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (b, array) = high
  (ANode{a, b}, array)

def tree_node(rec: Array<F32> -> Array<F32> & Array<F32>, low: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (a, array) = low
  tree_join(a, rec(array))

# The cells `base .. base + 2^depth` as a balanced tree, built in index order
# on one thread.
def tree(+depth: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, array: Array<F32>) ->
  Array<F32> & Array<F32>:
  match depth:
    case 0n:
      cell(rows, extents, base, array, Nat.is_lt(base, count))
    case 1n+lower:
      tree_node(a => tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), a),
        tree(lower, rows, extents, count, base, array))

def fork(pair: Array<F32> & Array<F32>, low: Array<F32> -> Array<F32>, high: Array<F32> -> Array<F32>) -> Array<F32>:
  (first, second) = pair
  a b = low(first) high(second)
  ANode{a, b}

def build(+depth: Nat, +splits: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>) -> Array<F32>:
  match depth splits:
    case 1n+lower 1n+rest:
      +high = Nat.add(Model.capacity(lower), base)
      fork(Array.clone(F32, array),
        a => build(lower, rest, rows, extents, count, base, a),
        a => build(lower, rest, rows, extents, count, high, a))
    case _ _:
      Pair.fst(Array<F32>, Array<F32>, tree(depth, rows, extents, count, base, array))

# A run of the build next to its proof of balance, as `Elementwise.Certified`.
type Certified<-depth: Nat, -expected: Array<F32> & Array<F32>> is Type:
  Certified{result: Array<F32> & Array<F32>, same: {result == expected : Array<F32> & Array<F32>},
    balanced: Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32>, result))}

def cell_leaf_certified(pair: F32 & Array<F32>) -> Certified<0n, cell_leaf(pair)>:
  (value, array) = pair
  Certified{cell_leaf((value, array)), {==}, Unit{}}

def cell_certified(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +position: Nat, array: Array<F32>, inside: Bool) ->
  Certified<0n, cell(rows, extents, position, array, inside)>:
  match inside:
    case False{}:
      Certified{cell(rows, extents, position, array, False{}), {==}, Unit{}}
    case True{}:
      cell_leaf_certified(fiber_of(rows, extents, position, array))

def node_joined(+lower: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, -array: Array<F32>,
  a: Array<F32>, -kept: Array<F32>,
  same: {(a, kept) == tree(lower, rows, extents, count, base, array) : Array<F32> & Array<F32>},
  balanced: Model.Balanced(F32, lower, a),
  high: Certified<lower, tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), kept)>) ->
  Certified<1n+lower, tree(1n+lower, rows, extents, count, base, array)>:
  Certified{result, high_same, high_balanced} = high
  (b, rest) = result
  Certified{(ANode{a, b}, rest),
    %same : {(ANode{a, b}, rest) == tree_node(x => tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), x), _) : Array<F32> & Array<F32>}
    %high_same : {(ANode{a, b}, rest) == tree_join(a, _) : Array<F32> & Array<F32>}
    {==},
    (balanced, high_balanced)}

def node_split(+lower: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, -array: Array<F32>,
  result: Array<F32> & Array<F32>,
  same: {result == tree(lower, rows, extents, count, base, array) : Array<F32> & Array<F32>},
  balanced: Model.Balanced(F32, lower, Pair.fst(Array<F32>, Array<F32>, result)),
  next: @kept: Array<F32> -> Certified<lower, tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), kept)>) ->
  Certified<1n+lower, tree(1n+lower, rows, extents, count, base, array)>:
  (a, kept) = result
  node_joined(lower, rows, extents, count, base, array, a, kept, same, balanced, next(kept))

def tree_node_certified(+lower: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, -array: Array<F32>,
  low: Certified<lower, tree(lower, rows, extents, count, base, array)>,
  next: @kept: Array<F32> -> Certified<lower, tree(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), kept)>) ->
  Certified<1n+lower, tree(1n+lower, rows, extents, count, base, array)>:
  Certified{result, same, balanced} = low
  node_split(lower, rows, extents, count, base, array, result, same, balanced, next)

def tree_certified(+depth: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, array: Array<F32>) ->
  Certified<depth, tree(depth, rows, extents, count, base, array)>:
  match depth:
    case 0n:
      cell_certified(rows, extents, base, array, Nat.is_lt(base, count))
    case 1n+lower:
      tree_node_certified(lower, rows, extents, count, base, array,
        tree_certified(lower, rows, extents, count, base, array),
        kept => tree_certified(lower, rows, extents, count, Nat.add(Model.capacity(lower), base), kept))

def tree_balanced(+depth: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat, -array: Array<F32>,
  certified: Certified<depth, tree(depth, rows, extents, count, base, array)>) ->
  Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32>, tree(depth, rows, extents, count, base, array))):
  Certified{result, same, balanced} = certified
  %same : Model.Balanced(F32, depth, Pair.fst(Array<F32>, Array<F32>, _))
  balanced

def fork_balanced(+lower: Nat, pair: Array<F32> & Array<F32>, -low: Array<F32> -> Array<F32>, -high: Array<F32> -> Array<F32>,
  low_valid: @a: Array<F32> -> Model.Balanced(F32, lower, low(a)),
  high_valid: @a: Array<F32> -> Model.Balanced(F32, lower, high(a))) ->
  Model.Balanced(F32, 1n+lower, fork(pair, low, high)):
  (first, second) = pair
  (low_valid(first), high_valid(second))

# The parallel build returns a balanced tree of its depth, whatever the split.
def build_balanced(+depth: Nat, +splits: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>) -> Model.Balanced(F32, depth, build(depth, splits, rows, extents, count, base, array)):
  match depth splits:
    case 1n+lower 1n+rest:
      +high = Nat.add(Model.capacity(lower), base)
      fork_balanced(lower, Array.clone(F32, array),
        a => build(lower, rest, rows, extents, count, base, a),
        a => build(lower, rest, rows, extents, count, high, a),
        a => build_balanced(lower, rest, rows, extents, count, base, a),
        a => build_balanced(lower, rest, rows, extents, count, high, a))
    case 0n _:
      tree_balanced(0n, rows, extents, count, base, array, tree_certified(0n, rows, extents, count, base, array))
    case 1n+lower 0n:
      tree_balanced(1n+lower, rows, extents, count, base, array, tree_certified(1n+lower, rows, extents, count, base, array))

# Split by work, as `Matrix.splits`: a target cell costs one term per
# contribution word that lands on it.
def terms(+plan: Shape.BroadcastPlan, +length: Nat) -> Nat:
  Nat.div(Shape.elements(Shape.output_shape(plan)), length)

def parallel_rows(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  taken: Core.Dynamic & Array<F32>, rows: List<&2, Shape.Row>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (kept, array) = taken
  +extents = Shape.extents(Shape.output_shape(plan))
  +splits = Matrix.splits(depth, terms(plan, length))
  (kept, Done{Core.Packed{target, Core.TTensor{Storage.Owned{depth, length,
    build(depth, splits, rows, extents, length, 0n, array),
    build_balanced(depth, splits, rows, extents, length, 0n, array),
    valid}, exact}}})

# An empty contribution leaves every target cell at +0.0.
def parallel_indexed(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, output: Core.Tensor<target>,
  +depth: Nat, +length: Nat, -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match map:
    case Shape.Unindexed{zero}:
      (input, Done{Core.Packed{target, output}})
    case Shape.Indexed{rows, same}:
      parallel_rows(plan, target, depth, length, valid, exact, Elementwise.take(input), rows)

def parallel_allocated(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  result: Result<&2, &1, Core.Error, Core.Tensor<target>>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (input, Fail{TensorError{error}})
    case Done{Core.TTensor{storage, exact}}:
      match storage:
        case Storage.Empty{}:
          (input, Done{Core.Packed{target, Core.TTensor{Storage.Empty{}, exact}}})
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          parallel_indexed(input, target, plan, Core.TTensor{Storage.Owned{depth, length, array, balanced, valid}, exact},
            depth, length, valid, exact, Shape.index_map(plan))

def fitted(input: Core.Dynamic, +target: Shape.Shape, +plan: Shape.BroadcastPlan, same: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match same:
    case False{}:
      (input, Fail{NotReducible{}})
    case True{}:
      parallel_allocated(input, target, plan, Core.new(target, 0.0))

def planned(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +target: Shape.Shape,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      (Core.Packed{shape, tensor}, Fail{TensorError{Core.ShapeError{error}}})
    case Done{+plan}:
      fitted(Core.Packed{shape, tensor}, target, plan,
        Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)))

# The contribution owner comes back unchanged next to the result. A target
# that does not broadcast with the contribution fails with the planner's
# broadcast error; a target whose broadcast with the contribution is larger
# than the contribution (for example [2, 3] for a [3] contribution) fails with
# NotReducible.
def reduce_to(input: Core.Dynamic, +target: Shape.Shape) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  planned(shape, tensor, target, Shape.broadcast_checked(target, shape))

def reduce_to_dimensions_result(input: Core.Dynamic, -dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      (input, Fail{TensorError{Core.ShapeError{error}}})
    case Shape.SBuilt{+target, preserved}:
      reduce_to(input, target)

def reduce_to_dimensions(input: Core.Dynamic, +dimensions: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  reduce_to_dimensions_result(input, dimensions, Shape.construct(dimensions))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case NotReducible{}:
      101
