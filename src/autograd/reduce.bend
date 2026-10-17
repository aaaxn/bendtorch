import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/layout.bend as Layout
import ../tensor/matmul.bend as Matrix
import ../tensor/parallel.bend as Parallel

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

# Whether every row here takes one digit (span 1), so that the fiber below
# a digit of the row above is one word.
def single_rows(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>) -> Bool:
  match rows extents:
    case Nil{} _:
      True{}
    case Shape.Row{stride, +left, right} <> +rest +extent <> +others:
      Bool.and(Nat.is_eq(span(left, extent), 1n), single_rows(rest, others))
    case Shape.Row{stride, left, right} <> rest Nil{}:
      False{}

# `single_rows` of the rows below the first one.
def single_below(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>) -> Bool:
  match rows extents:
    case Shape.Row{stride, left, right} <> +rest +extent <> +others:
      single_rows(rest, others)
    case _ _:
      False{}

# The offset of the one word that rows of span 1 reach from `base`.
def single_offset(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +target: Nat, +base: Nat) -> Nat:
  match rows extents:
    case Shape.Row{stride, +left, +right} <> +rest +extent <> +others:
      single_offset(rest, others, Nat.mod(target, left), Nat.add(base, Nat.mul(right, Nat.div(target, left))))
    case _ _:
      base

# The contribution offset of digit d of a row whose right stride is `step`.
def step_offset(+start: Nat, +step: Nat, +d: Nat) -> Nat:
  Nat.add(start, Nat.mul(step, d))

def strided_offset(+step: Nat, +base: Nat, +digit: Nat, +lift: Nat) -> Nat:
  Nat.add(Nat.add(base, Nat.mul(step, digit)), lift)

# The fuel + 1 words at offsets base + step * d + lift for d = digit, digit +
# 1, ..., added to `accumulator` in that order; `pair` is the read of the
# first. Each step reads the next word and hands it on as an argument.
def strided_run(fuel: Nat, +step: Nat, +base: Nat, +digit: Nat, +lift: Nat, accumulator: F32, pair: Array<F32> & F32) ->
  F32 & Array<F32>:
  match fuel pair:
    case 0n Tuple{array, word}:
      (F32.add(accumulator, word), array)
    case 1n+remaining Tuple{array, word}:
      strided_run(remaining, step, base, 1n+digit, lift, F32.add(accumulator, word),
        Array.get(F32, array, U32.from_nat(strided_offset(step, base, 1n+digit, lift))))

def strided(fuel: Nat, +step: Nat, +base: Nat, +digit: Nat, +lift: Nat, accumulator: F32, array: Array<F32>) -> F32 & Array<F32>:
  match fuel:
    case 0n:
      (accumulator, array)
    case 1n+remaining:
      strided_run(remaining, step, base, digit, lift, accumulator, Array.get(F32, array, U32.from_nat(strided_offset(step, base, digit, lift))))

# `fiber`, the same words added in the same order, except that a row whose
# rows below all take one digit (`single`, which is `single_below` of the
# rows; a later digit of the same row passes False) walks its digits in one
# strided loop: below digit d it reads the one word at base + right * d +
# single_offset(rest, others, target % left, 0), with no closure per word.
# Column sums, row sums and batch sums over one axis take that loop.
def fiber_fast(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, fuel: Nat, single: Bool, +target: Nat, +base: Nat, +digit: Nat,
  accumulator: F32, array: Array<F32>) -> F32 & Array<F32>:
  match rows extents fuel single:
    case Nil{} _ _ single:
      leaf_add(accumulator, Array.get(F32, array, U32.from_nat(base)))
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others fuel True{}:
      strided(fuel, right, base, digit, single_offset(rest, others, Nat.mod(target, left), 0n), accumulator, array)
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n False{}:
      (accumulator, array)
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining False{}:
      +inner = Nat.mod(target, left)
      walk_next(value => kept => fiber_fast(rows, extents, remaining, False{}, target, base, 1n+digit, value, kept),
        fiber_fast(rest, others, first_span(rest, others), single_below(rest, others), inner, step_offset(base, right, digit),
          first_digit(rest, inner), accumulator, array))
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _ single:
      (accumulator, array)

# Cell `target` of the reduction: its whole fiber from +0.0.
def fiber_of(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +target: Nat, array: Array<F32>) -> F32 & Array<F32>:
  fiber_fast(rows, extents, first_span(rows, extents), single_below(rows, extents), target, 0n, first_digit(rows, target), 0.0, array)

# One target cell: its fiber sum from +0.0 when the cell is below `count`,
# else +0.0.
def cell(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +position: Nat, array: Array<F32>, inside: Bool) ->
  F32 & Array<F32>:
  match inside:
    case False{}:
      (0.0, array)
    case True{}:
      fiber_of(rows, extents, position, array)

# What every cell of a build needs: the index rows of the plan and the
# contribution extent of each row.
type Fibers is Data:
  Fibers{rows: List<&2, Shape.Row>, extents: List<&2, Nat>}

# Each thread below the forks reads its own copy.
def fibers(+given: Fibers) -> Fibers:
  Fibers{+rows, +extents} = given
  Fibers{rows, extents}

# The cell function `Parallel.build` runs: its state is the contribution
# array, next to a Unit that fills the build's second state slot.
def keep(pair: F32 & Array<F32>, none: Unit) -> F32 & (Array<F32> & Unit):
  (value, array) = pair
  (value, (array, none))

def cell_at(+fibers: Fibers, +position: Nat, state: Array<F32> & Unit, inside: Bool) -> F32 & (Array<F32> & Unit):
  Fibers{+rows, +extents} = fibers
  (array, none) = state
  keep(cell(rows, extents, position, array, inside), none)

# Each half of a fork gets its own clone of the contribution array.
def halves(pair: Array<F32> & Array<F32>) -> (Array<F32> & Unit) & (Array<F32> & Unit):
  (first, second) = pair
  ((first, Unit{}), (second, Unit{}))

def split(state: Array<F32> & Unit) -> (Array<F32> & Unit) & (Array<F32> & Unit):
  (array, none) = state
  halves(Array.clone(F32, array))

def build(+depth: Nat, +splits: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>) -> Array<F32>:
  Parallel.build(~Fibers, ~Fibers, ~Array<F32>, ~Unit, ~fibers, ~split, ~cell_at, Fibers{rows, extents}, depth, splits, count, base,
    (array, Unit{}))

# The parallel build returns a balanced tree of its depth, whatever the split.
def build_balanced(+depth: Nat, +splits: Nat, +rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +count: Nat, +base: Nat,
  array: Array<F32>) -> Model.Balanced(F32, depth, build(depth, splits, rows, extents, count, base, array)):
  Parallel.balanced(~Fibers, ~Fibers, ~Array<F32>, ~Unit, ~fibers, ~split, ~cell_at, Fibers{rows, extents}, depth, splits, count, base,
    (array, Unit{}))

# A target cell costs one term per contribution word that lands on it.
def terms(+plan: Shape.BroadcastPlan, +length: Nat) -> Nat:
  Nat.div(Shape.elements(Shape.output_shape(plan)), length)

# Whether some axis below these rows has extent above 1 and is kept by the
# target (left stride above 0).
def kept_after(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>) -> Bool:
  match rows extents:
    case Shape.Row{stride, +left, right} <> +rest +extent <> +others:
      Bool.or(Bool.and(Nat.is_lt(1n, extent), Bool.not(Nat.is_eq(left, 0n))), kept_after(rest, others))
    case _ _:
      False{}

# Whether every fiber is one contiguous run of the contribution: no axis the
# target keeps comes after an axis it reduces, counting only axes of extent
# above 1. Row sums such as [n, m] to [n, 1] are contiguous; column and batch
# sums such as [n, m] to [1, m] or [b, n, m] to [n, m] are strided.
def contiguous(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>) -> Bool:
  match rows extents:
    case Shape.Row{stride, +left, right} <> +rest +extent <> +others:
      Bool.pick(Bool, Bool.and(Nat.is_lt(1n, extent), Nat.is_eq(left, 0n)), Bool.not(kept_after(rest, others)),
        contiguous(rest, others))
    case _ _:
      True{}

# How many top levels of the target tree fork. Every fork clones the whole
# contribution. A target whose fibers are contiguous splits by work, as
# `Matrix.splits` does with the terms of a cell; a target with strided fibers
# never forks, since there the forked build ran slower at 8 and 24 threads
# than one thread (docs/reduce-to.md). The proofs hold for any split.
def splits(+plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat) -> Nat:
  +extents = Shape.extents(Shape.output_shape(plan))
  +rows = Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), extents)
  Bool.pick(Nat, contiguous(rows, extents), Matrix.splits(depth, terms(plan, length)), 0n)

def parallel_rows(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(target)) : Nat},
  taken: Core.Dynamic & Array<F32>, rows: List<&2, Shape.Row>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (kept, array) = taken
  +extents = Shape.extents(Shape.output_shape(plan))
  +splits = splits(plan, depth, length)
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
