import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/certificates.bend as Certificates
import ./core.bend as Core
import ./layout.bend as Layout
import ./lifecycle.bend as Lifecycle
import ./transforms.bend as Transforms
import ./parallel.bend as Parallel
import ./elementwise.bend as Elementwise

# PyTorch's basic slicing along one axis, as a copy (there are no views
# yet): `input[..., start:stop:step, ...]` with the slice at axis `dim`
# (aten::slice), `narrow(dim, start, length)` and `select(dim, index)`, which
# drops the axis. A multi-axis slice `t[a:b, c:d]` is one slice per axis, as
# PyTorch's indexing applies them.
#
# The axis is checked first: a rank-0 input has no axis to slice, and an axis
# at or past the rank is AxisOutOfBounds (13), permute's code. Then a step of
# 0 is StepNotPositive (19), as PyTorch refuses every step <= 0. The start and
# stop are clamped to the axis as PyTorch clamps them (`bounds`), so a stop
# past the extent ends at the extent and a stop before the start gives an
# empty axis. `narrow` refuses a start past the extent or a start + length
# past it, and `select` an index at or past the extent, with OutOfRange (20).
#
# Every index is a U32, so none is negative: the signed positions of
# PyTorch (-1 is the last element) are turned into these unsigned ones before
# `bounds`, the one place a start and a stop are normalized.
#
# Output cell (o, k, r) of the row-major view [outer, length, inner] around
# the axis holds input cell (o, start + k * step, r) of [outer, extent,
# inner]. The cells are built by `Parallel.build`, so no word depends on the
# split or the thread count. The backward, PyTorch's slice_backward, scatters
# a gradient of the output's shape into +0.0 words of the input's shape: input
# cell (o, i, r) holds gradient cell (o, k, r) where i = start + k * step for
# some k below the length, and +0.0 everywhere else.

type Error is Data:
  TensorError{error: Core.Error}
  LayoutError{error: Layout.Error}
  StepNotPositive{}
  OutOfRange{}

# A slice's start and stop after PyTorch's clamp to an axis of extent n:
# start' = min(start, n), stop' = max(start', min(stop, n)).
type Bounds is Data:
  Bounds{start: Nat, stop: Nat}

def bounds(+extent: Nat, +start: Nat, +stop: Nat) -> Bounds:
  +low = Nat.min(start, extent)
  Bounds{low, Nat.max(low, Nat.min(stop, extent))}

# How many of 0, step, 2 step, ... lie below `span`: none for an empty span,
# else 1 + (span - 1) / step, a ceiling division that never forms a number
# past span. It takes one division, not a step per index.
def counted(+span: Nat, +step: Nat) -> Nat:
  match span:
    case 0n:
      0n
    case 1n+rest:
      1n+Nat.div(rest, step)

# The length of the sliced axis: the number of indices start', start' +
# step, ... below stop'.
def length(+clamped: Bounds, +step: Nat) -> Nat:
  Bounds{+low, +high} = clamped
  counted(Nat.sub(high, low), step)

# The row-major view of the input around the axis: the element count before
# it, its extent and the element count after it.
type View is Data:
  View{outer: Nat, extent: Nat, inner: Nat}

def view_at(dimensions: List<&2, U32>, +dim: Nat, +outer: Nat) -> View:
  match dimensions dim:
    case extent <> rest 0n:
      View{outer, U32.to_nat(extent), Shape.logical_numel(rest)}
    case extent <> rest 1n+next:
      view_at(rest, next, Nat.mul(outer, U32.to_nat(extent)))
    case Nil{} _:
      View{outer, 0n, 1n}

def view(dimensions: List<&2, U32>, +dim: Nat) -> View:
  view_at(dimensions, dim, 1n)

# The extent of axis `dim`, 0 past the rank. It reads that one dimension and
# forms no product.
def axis_extent(dimensions: List<&2, U32>, +dim: Nat) -> Nat:
  match dimensions dim:
    case extent <> rest 0n:
      U32.to_nat(extent)
    case extent <> rest 1n+next:
      axis_extent(rest, next)
    case Nil{} _:
      0n

def extent_at(dimensions: List<&2, U32>, +dim: U32) -> Nat:
  axis_extent(dimensions, U32.to_nat(dim))

# Whether some dimension is 0, so the input has no element.
def holds_zero(dimensions: List<&2, U32>) -> Bool:
  match dimensions:
    case Nil{}:
      False{}
    case extent <> rest:
      Bool.or(Nat.is_eq(U32.to_nat(extent), 0n), holds_zero(rest))

# The view a plan uses. An input with no element gets no cell in either
# build, so its outer and inner counts are left at 0 rather than formed: the
# product of the other extents can pass 2^48 - 1, the largest Nat the
# runtime holds, as [2^24, 2^24, 0] at axis 2 does.
def sized(+dimensions: List<&2, U32>, +dim: Nat, empty: Bool) -> View:
  match empty:
    case True{}:
      View{0n, axis_extent(dimensions, dim), 0n}
    case False{}:
      view(dimensions, dim)

# The input dimensions with the extent at the axis replaced.
def resized(dimensions: List<&2, U32>, +dim: Nat, +extent: U32) -> List<&2, U32>:
  match dimensions dim:
    case head <> rest 0n:
      extent <> rest
    case head <> rest 1n+next:
      head <> resized(rest, next, extent)
    case Nil{} _:
      Nil{}

# The input dimensions without the axis, as `select` returns them.
def dropped(dimensions: List<&2, U32>, +dim: Nat) -> List<&2, U32>:
  match dimensions dim:
    case head <> rest 0n:
      rest
    case head <> rest 1n+next:
      head <> dropped(rest, next)
    case Nil{} _:
      Nil{}

# What every cell of a slice and of its backward reads: the view, the
# clamped start, the step and the length of the sliced axis.
type Plan is Data:
  Plan{outer: Nat, extent: Nat, inner: Nat, start: Nat, step: Nat, length: Nat}

def planned_view(+outer: Nat, +extent: Nat, +inner: Nat, +clamped: Bounds, +step: Nat) -> Plan:
  Bounds{+low, +high} = clamped
  Plan{outer, extent, inner, low, step, length(clamped, step)}

def viewed(+given: View, +start: Nat, +stop: Nat, +step: Nat) -> Plan:
  View{+outer, +extent, +inner} = given
  planned_view(outer, extent, inner, bounds(extent, start, stop), step)

def plan_of(+dimensions: List<&2, U32>, +dim: U32, +start: U32, +stop: U32, +step: U32) -> Plan:
  viewed(sized(dimensions, U32.to_nat(dim), holds_zero(dimensions)), U32.to_nat(start), U32.to_nat(stop), U32.to_nat(step))

def plan_length(+plan: Plan) -> Nat:
  Plan{outer, extent, inner, start, step, length} = plan
  length

# PyTorch's checks, in its order: the axis (a rank-0 input has none), then
# the step.
def refused(inside: Bool, positive: Bool) -> Maybe<&2, Error>:
  match inside positive:
    case False{} _:
      Some{LayoutError{Layout.AxisOutOfBounds{}}}
    case True{} False{}:
      Some{StepNotPositive{}}
    case True{} True{}:
      None{}

def refusal(dimensions: List<&2, U32>, +dim: U32, +step: U32) -> Maybe<&2, Error>:
  refused(Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(dimensions)), Nat.is_lt(0n, U32.to_nat(step)))

# A slice the checks accept: its plan and its output shape.
type Planned is Data:
  Planned{plan: Plan, output: Shape.Shape}

def shaped(+plan: Plan, -dimensions: List<&2, U32>, built: Shape.ConstructResult<dimensions>) -> Result<&2, &2, Error, Planned>:
  match built:
    case Shape.SError{error}:
      Fail{TensorError{Core.ShapeError{error}}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Planned{plan, shape}}

def checked(+dimensions: List<&2, U32>, +dim: U32, +start: U32, +stop: U32, +step: U32, refused: Maybe<&2, Error>) ->
  Result<&2, &2, Error, Planned>:
  match refused:
    case Some{error}:
      Fail{error}
    case None{}:
      +plan = plan_of(dimensions, dim, start, stop, step)
      +output = resized(dimensions, U32.to_nat(dim), U32.from_nat(plan_length(plan)))
      shaped(plan, output, Shape.construct(output))

# The plan of input[..., start:stop:step, ...] at axis `dim`, or its refusal.
def plan(+input: Shape.Shape, +dim: U32, +start: U32, +stop: U32, +step: U32) -> Result<&2, &2, Error, Planned>:
  checked(Shape.dimensions(input), dim, start, stop, step, refusal(Shape.dimensions(input), dim, step))

# The cells of the forward build.

# The input index output cell `position` reads: position is (o * length + k)
# * inner + r, and it reads (o * extent + start + k * step) * inner + r.
def source(+plan: Plan, +position: Nat) -> Nat:
  Plan{+outer, +extent, +inner, +start, +step, +length} = plan
  +row = Nat.div(position, inner)
  +o = Nat.div(row, length)
  Nat.add(Nat.mul(Nat.add(Nat.mul(o, extent), Nat.add(start, Nat.mul(Nat.mod(row, length), step))), inner), Nat.mod(position, inner))

def same(+plan: Plan) -> Plan:
  plan

def take(b: Array<F32>, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<F32>):
  (a, x) = pair
  (x, (a, b))

def read(+plan: Plan, +position: Nat, state: Array<F32> & Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  (a, b) = state
  take(b, Array.get(F32, a, U32.from_nat(source(plan, position))))

# The state is the input array and an empty array that fills the second
# slot of the state pair; a cell past the count is +0.0 padding.
def cell(+plan: Plan, +position: Nat, state: Array<F32> & Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (0.0, state)
    case True{}:
      read(plan, position, state)

# A subtree below the forks holds 2^15 cells, as in the other builds.
def splits(+depth: Nat) -> Nat:
  Nat.sub(depth, 15n)

def build(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, a: Array<F32>) -> Array<F32>:
  Parallel.build(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, ~cell, plan, depth, splits, count, base,
    (a, Array.new(F32, 0n, 0.0)))

def balanced(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, a: Array<F32>) ->
  Model.Balanced(F32, depth, build(depth, splits, plan, count, base, a)):
  Parallel.balanced(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, ~cell, plan, depth, splits, count, base,
    (a, Array.new(F32, 0n, 0.0)))

# The cells of the backward build, over the input's shape.

# The gradient index input cell `position` reads, or None where no output
# cell came from it: position is (o * extent + i) * inner + r, and it reads
# (o * length + k) * inner + r when i = start + k * step with k below the
# length, for k = (i - start) / step.
def picked(+plan: Plan, +row: Nat, +within: Nat, +i: Nat, +k: Nat, hit: Bool) -> Maybe<&2, Nat>:
  Plan{+outer, +extent, +inner, +start, +step, +length} = plan
  match hit:
    case False{}:
      None{}
    case True{}:
      Some{Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(row, extent), length), k), inner), within)}

def hits(+start: Nat, +step: Nat, +length: Nat, +i: Nat, +k: Nat) -> Bool:
  Bool.and(Nat.is_le(start, i), Bool.and(Nat.is_lt(k, length), Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i)))

def target(+plan: Plan, +position: Nat) -> Maybe<&2, Nat>:
  Plan{+outer, +extent, +inner, +start, +step, +length} = plan
  +row = Nat.div(position, inner)
  +i = Nat.mod(row, extent)
  +k = Nat.div(Nat.sub(i, start), step)
  picked(plan, row, Nat.mod(position, inner), i, k, hits(start, step, length, i, k))

def fetched(+index: Nat, state: Array<F32> & Array<F32>) -> F32 & (Array<F32> & Array<F32>):
  (a, b) = state
  take(b, Array.get(F32, a, U32.from_nat(index)))

def scattered(state: Array<F32> & Array<F32>, at: Maybe<&2, Nat>) -> F32 & (Array<F32> & Array<F32>):
  match at:
    case None{}:
      (0.0, state)
    case Some{+index}:
      fetched(index, state)

def scatter_cell(+plan: Plan, +position: Nat, state: Array<F32> & Array<F32>, inside: Bool) -> F32 & (Array<F32> & Array<F32>):
  match inside:
    case False{}:
      (0.0, state)
    case True{}:
      scattered(state, target(plan, position))

def scatter_build(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, g: Array<F32>) -> Array<F32>:
  Parallel.build(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, ~scatter_cell, plan, depth, splits, count, base,
    (g, Array.new(F32, 0n, 0.0)))

def scatter_balanced(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, g: Array<F32>) ->
  Model.Balanced(F32, depth, scatter_build(depth, splits, plan, count, base, g)):
  Parallel.balanced(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, ~scatter_cell, plan, depth, splits, count, base,
    (g, Array.new(F32, 0n, 0.0)))

# Running a build on a tensor.

# Which build fills the output: the forward copy or the backward scatter.
type Pass is Data:
  Forward{}
  Backward{}

def built_array(kind: Pass, +depth: Nat, +plan: Plan, +count: Nat, a: Array<F32>) -> Array<F32>:
  match kind:
    case Forward{}:
      build(depth, splits(depth), plan, count, 0n, a)
    case Backward{}:
      scatter_build(depth, splits(depth), plan, count, 0n, a)

def built_balanced(kind: Pass, +depth: Nat, +plan: Plan, +count: Nat, a: Array<F32>) ->
  Model.Balanced(F32, depth, built_array(kind, depth, plan, count, a)):
  match kind:
    case Forward{}:
      balanced(depth, splits(depth), plan, count, 0n, a)
    case Backward{}:
      scatter_balanced(depth, splits(depth), plan, count, 0n, a)

# The build from an owned source: the source comes back as the first half of
# its clone, and the build reads the second half.
def built_cloned(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  +d: Nat, +n: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, array), -ok: {Storage.bounds(d, n) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Storage.Owned{d, n, array, balanced, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (kept, copy) = pair
  (Core.Packed{input, Core.TTensor{Storage.Owned{d, n, kept, Certificates.clone_first(F32, d, array, (kept, copy), same, balanced), ok}, fits}},
    Done{Core.Packed{output, Core.TTensor{Storage.Owned{depth, length, built_array(kind, depth, plan, length, copy),
      built_balanced(kind, depth, plan, length, copy), valid}, exact}}})

# An empty source: the forward copy never has cells then, and the backward
# of an empty gradient reads nothing, so the build reads an empty array.
def built_empty(+kind: Pass, +output: Shape.Shape, +plan: Plan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat}) -> Result<&2, &1, Error, Core.Dynamic>:
  match kind:
    case Forward{}:
      Fail{TensorError{Core.StorageError{Storage.OutOfBounds{}}}}
    case Backward{}:
      Done{Core.Packed{output, Core.TTensor{Storage.Owned{depth, length, built_array(Backward{}, depth, plan, length, Array.new(F32, 0n, 0.0)),
        built_balanced(Backward{}, depth, plan, length, Array.new(F32, 0n, 0.0)), valid}, exact}}}

def built_owned(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  source: Core.Tensor<input>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.TTensor{storage, fits} = source
  match storage:
    case Storage.Empty{}:
      (Core.Packed{input, Core.TTensor{Storage.Empty{}, fits}}, built_empty(kind, output, plan, depth, length, valid, exact))
    case Storage.Owned{+d, +n, array, balanced, ok}:
      built_cloned(kind, input, output, plan, depth, length, valid, exact, d, n, array, Array.clone(F32, array), {==}, balanced, ok, fits)

# Every output cell at once, into the storage that `Core.new` allocates. An
# output with no cell keeps that storage, which is empty.
def built(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, source: Core.Tensor<input>,
  allocated: Result<&2, &1, Core.Error, Core.Tensor<output>>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match allocated:
    case Fail{error}:
      (Core.Packed{input, source}, Fail{TensorError{error}})
    case Done{target}:
      Core.TTensor{storage, exact} = target
      match storage:
        case Storage.Empty{}:
          (Core.Packed{input, source}, Done{Core.Packed{output, Core.TTensor{Storage.Empty{}, exact}}})
        case Storage.Owned{+depth, +length, array, balanced, valid}:
          built_owned(kind, input, output, plan, depth, length, valid, exact, source)

def sliced(+shape: Shape.Shape, tensor: Core.Tensor<shape>, planned: Result<&2, &2, Error, Planned>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match planned:
    case Fail{error}:
      (Core.Packed{shape, tensor}, Fail{error})
    case Done{Planned{+plan, +output}}:
      built(Forward{}, shape, output, plan, tensor, Core.new(output, 0.0))

# PyTorch's input[..., start:stop:step, ...] with the slice at axis `dim`.
# The input owner always comes back unchanged next to the result.
def slice(input: Core.Dynamic, +dim: U32, +start: U32, +stop: U32, +step: U32) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  sliced(shape, tensor, plan(shape, dim, start, stop, step))

# PyTorch's slice_backward: the gradient g of a planned slice's output,
# scattered into +0.0 words of the input's shape. g must hold as many words
# as the slice's output; g comes back unchanged next to the result.
def scatter(g: Core.Dynamic, +input: Shape.Shape, +plan: Plan) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = g
  built(Backward{}, shape, input, plan, tensor, Core.new(input, 0.0))

def transformed(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{Transforms.LayoutError{error}}:
      Fail{LayoutError{error}}
    case Fail{Transforms.TensorError{error}}:
      Fail{TensorError{error}}
    case Done{output}:
      Done{output}

def reshaped(+dimensions: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{error})
    case Done{output}:
      (input, transformed(Transforms.reshape_dynamic(output, dimensions)))

def narrow_ranged(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +dim: U32, +start: U32, +length: U32, fits: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match fits:
    case False{}:
      (Core.Packed{shape, tensor}, Fail{OutOfRange{}})
    case True{}:
      slice(Core.Packed{shape, tensor}, dim, start, U32.add(start, length), 1)

def narrow_checked(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +dim: U32, +start: U32, +length: U32,
  refused: Maybe<&2, Error>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match refused:
    case Some{error}:
      (Core.Packed{shape, tensor}, Fail{error})
    case None{}:
      narrow_ranged(shape, tensor, dim, start, length,
        Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), extent_at(Shape.dimensions(shape), dim)))

# PyTorch's input.narrow(dim, start, length): the slice start:start + length
# at axis `dim`, refused when start + length passes the extent.
def narrow(input: Core.Dynamic, +dim: U32, +start: U32, +length: U32) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  narrow_checked(shape, tensor, dim, start, length, refusal(Shape.dimensions(shape), dim, 1))

def select_ranged(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +dim: U32, +index: U32, below: Bool) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match below:
    case False{}:
      (Core.Packed{shape, tensor}, Fail{OutOfRange{}})
    case True{}:
      reshaped(dropped(Shape.dimensions(shape), U32.to_nat(dim)), slice(Core.Packed{shape, tensor}, dim, index, U32.add(index, 1), 1))

def select_checked(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +dim: U32, +index: U32, refused: Maybe<&2, Error>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match refused:
    case Some{error}:
      (Core.Packed{shape, tensor}, Fail{error})
    case None{}:
      select_ranged(shape, tensor, dim, index, Nat.is_lt(U32.to_nat(index), extent_at(Shape.dimensions(shape), dim)))

# PyTorch's input.select(dim, index): the slice index:index + 1 at axis
# `dim`, then a reshape that drops the axis, refused when the index is at or
# past the extent.
def select(input: Core.Dynamic, +dim: U32, +index: U32) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  select_checked(shape, tensor, dim, index, refusal(Shape.dimensions(shape), dim, 1))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case LayoutError{error}:
      Transforms.error_code(Transforms.LayoutError{error})
    case StepNotPositive{}:
      19
    case OutOfRange{}:
      20
