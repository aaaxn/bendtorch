import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/model.bend as Model
import ../storage/certificates.bend as Certificates
import ./core.bend as Core
import ./layout.bend as Layout
import ./lifecycle.bend as Lifecycle
import ./parallel.bend as Parallel
import ./slice.bend as Slice
import ./signed.bend as Signed

# PyTorch's index_select(dim, index) and gather(dim, index), as copies (there
# are no views yet), and their backward, index_add and scatter_add into a
# +0.0 gradient of the input's shape.
#
# There are no integer tensors (every tensor is F32), so an index is a list
# of U32 entries, never F32 words: index_select takes a vector of entries,
# and gather an Index, the shape of PyTorch's index tensor and its entries
# in row-major order. PyTorch refuses a negative entry in both, so
# a U32 entry loses nothing; an entry at or past the extent is OutOfRange
# (20). The axis is a signed position, wrapped among the rank: one outside
# -rank..rank-1, and every axis of a rank-0 input, is AxisOutOfBounds (13).
#
# index_select's output is the input with the extent at the axis replaced by
# the entry count: output cell (o, k, r) of the row-major view [outer, length,
# inner] holds input cell (o, index[k], r) of [outer, extent, inner].
# gather's index has the input's rank (else IndexRankMismatch, 21) and the
# input's extent at every other axis (else IndexShapeMismatch, 22); its
# entry count must be its shape's element count (else ValueCountMismatch,
# 10). Its output has the index's dimensions, and output
# cell (o, k, r) holds input cell (o, index[o, k, r], r).
#
# Both builds run on Parallel.build, so no word depends on the split or the
# thread count. The state of a cell is the source array and the index array.
# The backward fills input cell (o, i, r) with the gradient words of cells
# (o, k, r) whose entry is i, added in increasing k to +0.0, the order of
# PyTorch's one-thread index_add and scatter_add, and +0.0 where no entry is
# i.

type Error is Data:
  TensorError{error: Core.Error}
  LayoutError{error: Layout.Error}
  OutOfRange{}
  IndexRankMismatch{}
  IndexShapeMismatch{}

# PyTorch's index tensor for gather: its shape, a Shape as every tensor's
# (rank at most 4, at most 2^20 elements), and its entries in row-major
# order.
type Index is Data:
  Index{shape: Shape.Shape, entries: List<&2, U32>}

def built_index(+entries: List<&2, U32>, -dimensions: List<&2, U32>, built: Shape.ConstructResult<dimensions>) -> Result<&2, &2, Error, Index>:
  match built:
    case Shape.SError{error}:
      Fail{TensorError{Core.ShapeError{error}}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Index{shape, entries}}

# An Index from an index tensor's dimensions and entries, refused as a tensor
# of those dimensions is: past rank 4 (RankExceeded, 1) or 2^20 elements
# (CapacityExceeded, 2). Its entry count is checked by gather.
def index(+dimensions: List<&2, U32>, +entries: List<&2, U32>) -> Result<&2, &2, Error, Index>:
  built_index(entries, dimensions, Shape.construct(dimensions))

# Which entry a cell reads: entry k of an index_select, the entry at the
# output position itself for a gather.
type Lookup is Data:
  Selected{}
  Gathered{}

# What every cell of both builds reads: the kind, the view [outer, extent,
# inner] of the input around the axis, and the length of the output's axis.
type Plan is Data:
  Plan{kind: Lookup, outer: Nat, extent: Nat, inner: Nat, length: Nat}

def plan_length(+plan: Plan) -> Nat:
  Plan{kind, outer, extent, inner, length} = plan
  length

def count(entries: List<&2, U32>) -> Nat:
  match entries:
    case Nil{}:
      0n
    case head <> rest:
      1n+count(rest)

# Whether every entry is below the extent.
def below(entries: List<&2, U32>, +extent: Nat) -> Bool:
  match entries:
    case Nil{}:
      True{}
    case head <> rest:
      Bool.and(Nat.is_lt(U32.to_nat(head), extent), below(rest, extent))

# Whether two lists of dimensions are equal.
def equal(left: List<&2, U32>, right: List<&2, U32>) -> Bool:
  match left right:
    case Nil{} Nil{}:
      True{}
    case x <> xs y <> ys:
      Bool.and(Nat.is_eq(U32.to_nat(x), U32.to_nat(y)), equal(xs, ys))
    case _ _:
      False{}

# Whether the index's dimensions are the input's at every axis but `dim`.
def matches(dimensions: List<&2, U32>, index: List<&2, U32>, +dim: Nat) -> Bool:
  match dimensions index dim:
    case extent <> rest other <> more 0n:
      equal(rest, more)
    case extent <> rest other <> more 1n+next:
      Bool.and(Nat.is_eq(U32.to_nat(extent), U32.to_nat(other)), matches(rest, more, next))
    case _ _ _:
      False{}

# The plan of a gather or an index_select of `length` entries at an axis in
# range.
def viewed(+kind: Lookup, +given: Slice.View, +length: Nat) -> Plan:
  Slice.View{+outer, +extent, +inner} = given
  Plan{kind, outer, extent, inner, length}

def planned_at(+kind: Lookup, +dimensions: List<&2, U32>, +axis: Nat, +length: Nat) -> Plan:
  viewed(kind, Slice.sized(dimensions, axis, Slice.holds_zero(dimensions)), length)

# A plan the checks accept, and its output shape.
type Planned is Data:
  Planned{plan: Plan, output: Shape.Shape}

def shaped(+plan: Plan, -dimensions: List<&2, U32>, built: Shape.ConstructResult<dimensions>) -> Result<&2, &2, Error, Planned>:
  match built:
    case Shape.SError{error}:
      Fail{TensorError{Core.ShapeError{error}}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Planned{plan, shape}}

# The output keeps every axis of the input but `axis`, whose extent becomes
# the length.
def accepted(+kind: Lookup, +dimensions: List<&2, U32>, +axis: Nat, +length: Nat) -> Result<&2, &2, Error, Planned>:
  +output = Slice.resized(dimensions, axis, U32.from_nat(length))
  shaped(planned_at(kind, dimensions, axis, length), output, Shape.construct(output))

def ranged(+kind: Lookup, +dimensions: List<&2, U32>, +axis: Nat, +entries: List<&2, U32>, +length: Nat, inside: Bool) ->
  Result<&2, &2, Error, Planned>:
  match inside:
    case False{}:
      Fail{OutOfRange{}}
    case True{}:
      accepted(kind, dimensions, axis, length)

# The axis check, then the entries against the extent.
def selecting(+dimensions: List<&2, U32>, +axis: Nat, +entries: List<&2, U32>, in_rank: Bool) -> Result<&2, &2, Error, Planned>:
  match in_rank:
    case False{}:
      Fail{LayoutError{Layout.AxisOutOfBounds{}}}
    case True{}:
      ranged(Selected{}, dimensions, axis, entries, count(entries), below(entries, Slice.axis_extent(dimensions, axis)))

# The plan of index_select(dim, index), or its refusal.
def select_plan(+input: Shape.Shape, dim: Signed.Position, +entries: List<&2, U32>) -> Result<&2, &2, Error, Planned>:
  +dimensions = Shape.dimensions(input)
  +axis = U32.to_nat(Signed.wrap(Shape.rank(input), dim))
  selecting(dimensions, axis, entries, Nat.is_lt(axis, Shape.rank_dimensions(dimensions)))

def gather_shaped(+dimensions: List<&2, U32>, +axis: Nat, +index: List<&2, U32>, +entries: List<&2, U32>, same: Bool) ->
  Result<&2, &2, Error, Planned>:
  match same:
    case False{}:
      Fail{IndexShapeMismatch{}}
    case True{}:
      ranged(Gathered{}, dimensions, axis, entries, Slice.axis_extent(index, axis), below(entries, Slice.axis_extent(dimensions, axis)))

def gather_ranked(+dimensions: List<&2, U32>, +axis: Nat, +index: List<&2, U32>, +entries: List<&2, U32>, same: Bool) ->
  Result<&2, &2, Error, Planned>:
  match same:
    case False{}:
      Fail{IndexRankMismatch{}}
    case True{}:
      gather_shaped(dimensions, axis, index, entries, matches(dimensions, index, axis))

def gather_axis(+dimensions: List<&2, U32>, +axis: Nat, +index: List<&2, U32>, +entries: List<&2, U32>, in_rank: Bool) ->
  Result<&2, &2, Error, Planned>:
  match in_rank:
    case False{}:
      Fail{LayoutError{Layout.AxisOutOfBounds{}}}
    case True{}:
      gather_ranked(dimensions, axis, index, entries, Nat.is_eq(Shape.rank_dimensions(index), Shape.rank_dimensions(dimensions)))

def gather_counted(+dimensions: List<&2, U32>, +axis: Nat, +index: List<&2, U32>, +entries: List<&2, U32>, exact: Bool) ->
  Result<&2, &2, Error, Planned>:
  match exact:
    case False{}:
      Fail{TensorError{Core.ValueCountMismatch{}}}
    case True{}:
      gather_axis(dimensions, axis, index, entries, Nat.is_lt(axis, Shape.rank_dimensions(dimensions)))

# The plan of gather(dim, index), or its refusal, in this order: an entry
# count other than the index shape's element count, the axis, the index's
# rank, its extents at the other axes, then its entries. The element count
# is the Shape's own, so no product of its extents is formed here.
def gather_plan(+input: Shape.Shape, dim: Signed.Position, +index: Index) -> Result<&2, &2, Error, Planned>:
  Index{+shape, +entries} = index
  +axis = U32.to_nat(Signed.wrap(Shape.rank(input), dim))
  gather_counted(Shape.dimensions(input), axis, Shape.dimensions(shape), entries,
    Nat.is_eq(count(entries), Shape.elements(shape)))

# The entries as a balanced U32 array that a cell reads in a few steps: the
# smallest depth that holds them, as Storage allocates.
def written(entries: List<&2, U32>, array: Array<U32>, +at: U32) -> Array<U32>:
  match entries:
    case Nil{}:
      array
    case head <> rest:
      written(rest, Array.set(U32, array, at, head), U32.add(at, 1))

def index_depth(pair: Nat & Nat) -> Nat:
  (depth, capacity) = pair
  depth

def index_array(+entries: List<&2, U32>) -> Array<U32>:
  +n = count(entries)
  written(entries, Array.new(U32, index_depth(Storage.depth_seek(20n, n, 0n, 1n, Nat.is_le(n, 1n))), 0), 0)

# The cells of the forward build.

# The position of cell (o, k, r) of the view [outer, rows, inner].
def position(+rows: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(o, rows), k), inner), r)

# The entry output cell (o, k, r) reads: k for an index_select, the cell's
# own position for a gather.
def entry_of(+kind: Lookup, +length: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  match kind:
    case Selected{}:
      k
    case Gathered{}:
      position(length, inner, o, k, r)

# Output cell `at` is (o, k, r) of [outer, length, inner]; with entry j, it
# reads input cell (o, j, r) of [outer, extent, inner].
def entry(+plan: Plan, +at: Nat) -> Nat:
  Plan{+kind, +outer, +extent, +inner, +length} = plan
  +row = Nat.div(at, inner)
  entry_of(kind, length, inner, Nat.div(row, length), Nat.mod(row, length), Nat.mod(at, inner))

def source(+plan: Plan, +at: Nat, +j: Nat) -> Nat:
  Plan{+kind, +outer, +extent, +inner, +length} = plan
  position(extent, inner, Nat.div(Nat.div(at, inner), length), j, Nat.mod(at, inner))

def same(+plan: Plan) -> Plan:
  plan

def take(index: Array<U32>, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<U32>):
  (a, x) = pair
  (x, (a, index))

def looked(+plan: Plan, +at: Nat, a: Array<F32>, pair: Array<U32> & U32) -> F32 & (Array<F32> & Array<U32>):
  (index, +j) = pair
  take(index, Array.get(F32, a, U32.from_nat(source(plan, at, U32.to_nat(j)))))

def read(+plan: Plan, +at: Nat, state: Array<F32> & Array<U32>) -> F32 & (Array<F32> & Array<U32>):
  (a, index) = state
  looked(plan, at, a, Array.get(U32, index, U32.from_nat(entry(plan, at))))

# A cell past the count is +0.0 padding.
def cell(+plan: Plan, +at: Nat, state: Array<F32> & Array<U32>, inside: Bool) -> F32 & (Array<F32> & Array<U32>):
  match inside:
    case False{}:
      (0.0, state)
    case True{}:
      read(plan, at, state)

def halves(a: Array<F32> & Array<F32>, b: Array<U32> & Array<U32>) ->
  (Array<F32> & Array<U32>) & (Array<F32> & Array<U32>):
  (a1, a2) = a
  (b1, b2) = b
  ((a1, b1), (a2, b2))

# Each half of a fork gets its own copy of both arrays.
def split(state: Array<F32> & Array<U32>) -> (Array<F32> & Array<U32>) & (Array<F32> & Array<U32>):
  (a, b) = state
  halves(Array.clone(F32, a), Array.clone(U32, b))

def build(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, a: Array<F32>, index: Array<U32>) -> Array<F32>:
  Parallel.build(~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~cell, plan, depth, splits, count, base, (a, index))

def balanced(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, a: Array<F32>, index: Array<U32>) ->
  Model.Balanced(F32, depth, build(depth, splits, plan, count, base, a, index)):
  Parallel.balanced(~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~cell, plan, depth, splits, count, base, (a, index))

# The cells of the backward build, over the input's shape.

def accumulate(acc: F32, index: Array<U32>, pair: Array<F32> & F32) -> F32 & (Array<F32> & Array<U32>):
  (g, x) = pair
  (F32.add(acc, x), (g, index))

def summed(hit: Bool, +at: Nat, acc: F32, g: Array<F32>, index: Array<U32>) -> F32 & (Array<F32> & Array<U32>):
  match hit:
    case False{}:
      (acc, (g, index))
    case True{}:
      accumulate(acc, index, Array.get(F32, g, U32.from_nat(at)))

def compared(+i: Nat, +at: Nat, acc: F32, g: Array<F32>, pair: Array<U32> & U32) -> F32 & (Array<F32> & Array<U32>):
  (index, +j) = pair
  summed(Nat.is_eq(U32.to_nat(j), i), at, acc, g, index)

# One gradient cell (o, k, r): its word is added when its entry is i.
def stepped(+i: Nat, +at: Nat, +slot: Nat, acc: F32, state: Array<F32> & Array<U32>) -> F32 & (Array<F32> & Array<U32>):
  (g, index) = state
  compared(i, at, acc, g, Array.get(U32, index, U32.from_nat(slot)))

# The gradient cells (o, k, r), (o, k + 1, r), ... for `left` more k, added
# to acc in increasing k.
# Gradient cell (o, k, r) and the index slot it reads.
def cell_at(+plan: Plan, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  Plan{+kind, +outer, +extent, +inner, +length} = plan
  position(length, inner, o, k, r)

def slot_at(+plan: Plan, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  Plan{+kind, +outer, +extent, +inner, +length} = plan
  entry_of(kind, length, inner, o, k, r)

def stepped_pair(+i: Nat, +at: Nat, +slot: Nat, pair: F32 & (Array<F32> & Array<U32>)) -> F32 & (Array<F32> & Array<U32>):
  (acc, state) = pair
  stepped(i, at, slot, acc, state)

def added(+plan: Plan, +o: Nat, +i: Nat, +r: Nat, left: Nat, +k: Nat, pair: F32 & (Array<F32> & Array<U32>)) ->
  F32 & (Array<F32> & Array<U32>):
  match left:
    case 0n:
      pair
    case 1n+rest:
      added(plan, o, i, r, rest, 1n+k, stepped_pair(i, cell_at(plan, o, k, r), slot_at(plan, o, k, r), pair))

# Input cell `at` is (o, i, r) of [outer, extent, inner].
def scattered(+plan: Plan, +at: Nat, state: Array<F32> & Array<U32>) -> F32 & (Array<F32> & Array<U32>):
  Plan{+kind, +outer, +extent, +inner, +length} = plan
  +row = Nat.div(at, inner)
  added(plan, Nat.div(row, extent), Nat.mod(row, extent), Nat.mod(at, inner), length, 0n, (0.0, state))

def scatter_cell(+plan: Plan, +at: Nat, state: Array<F32> & Array<U32>, inside: Bool) -> F32 & (Array<F32> & Array<U32>):
  match inside:
    case False{}:
      (0.0, state)
    case True{}:
      scattered(plan, at, state)

def scatter_build(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, g: Array<F32>, index: Array<U32>) -> Array<F32>:
  Parallel.build(~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~scatter_cell, plan, depth, splits, count, base, (g, index))

def scatter_balanced(+depth: Nat, +splits: Nat, +plan: Plan, +count: Nat, +base: Nat, g: Array<F32>, index: Array<U32>) ->
  Model.Balanced(F32, depth, scatter_build(depth, splits, plan, count, base, g, index)):
  Parallel.balanced(~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~scatter_cell, plan, depth, splits, count, base, (g, index))

# Running a build on a tensor.

type Pass is Data:
  Forward{}
  Backward{}

def built_array(kind: Pass, +depth: Nat, +plan: Plan, +count: Nat, a: Array<F32>, index: Array<U32>) -> Array<F32>:
  match kind:
    case Forward{}:
      build(depth, Slice.splits(depth), plan, count, 0n, a, index)
    case Backward{}:
      scatter_build(depth, Slice.splits(depth), plan, count, 0n, a, index)

def built_balanced(kind: Pass, +depth: Nat, +plan: Plan, +count: Nat, a: Array<F32>, index: Array<U32>) ->
  Model.Balanced(F32, depth, built_array(kind, depth, plan, count, a, index)):
  match kind:
    case Forward{}:
      balanced(depth, Slice.splits(depth), plan, count, 0n, a, index)
    case Backward{}:
      scatter_balanced(depth, Slice.splits(depth), plan, count, 0n, a, index)

def built_cloned(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, +entries: List<&2, U32>, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  +d: Nat, +n: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>, -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>},
  -balanced: Model.Balanced(F32, d, array), -ok: {Storage.bounds(d, n) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Storage.Owned{d, n, array, balanced, ok}) == Shape.logical_numel(Shape.dimensions(input)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  (kept, copy) = pair
  (Core.Packed{input, Core.TTensor{Storage.Owned{d, n, kept, Certificates.clone_first(F32, d, array, (kept, copy), same, balanced), ok}, fits}},
    Done{Core.Packed{output, Core.TTensor{Storage.Owned{depth, length, built_array(kind, depth, plan, length, copy, index_array(entries)),
      built_balanced(kind, depth, plan, length, copy, index_array(entries)), valid}, exact}}})

# An empty source: the forward copy never has cells then, and the backward
# of an empty gradient reads nothing.
def built_empty(+kind: Pass, +output: Shape.Shape, +plan: Plan, +entries: List<&2, U32>, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat}) -> Result<&2, &1, Error, Core.Dynamic>:
  match kind:
    case Forward{}:
      Fail{TensorError{Core.StorageError{Storage.OutOfBounds{}}}}
    case Backward{}:
      Done{Core.Packed{output, Core.TTensor{Storage.Owned{depth, length,
        built_array(Backward{}, depth, plan, length, Array.new(F32, 0n, 0.0), index_array(entries)),
        built_balanced(Backward{}, depth, plan, length, Array.new(F32, 0n, 0.0), index_array(entries)), valid}, exact}}}

def built_owned(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, +entries: List<&2, U32>, +depth: Nat, +length: Nat,
  -valid: {Storage.bounds(depth, length) == True{} : Bool},
  -exact: {length == Shape.logical_numel(Shape.dimensions(output)) : Nat},
  source: Core.Tensor<input>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.TTensor{storage, fits} = source
  match storage:
    case Storage.Empty{}:
      (Core.Packed{input, Core.TTensor{Storage.Empty{}, fits}}, built_empty(kind, output, plan, entries, depth, length, valid, exact))
    case Storage.Owned{+d, +n, array, balanced, ok}:
      built_cloned(kind, input, output, plan, entries, depth, length, valid, exact, d, n, array, Array.clone(F32, array), {==}, balanced, ok, fits)

# Every output cell at once, into the storage that `Core.new` allocates. An
# output with no cell keeps that storage, which is empty.
def built(+kind: Pass, +input: Shape.Shape, +output: Shape.Shape, +plan: Plan, +entries: List<&2, U32>, source: Core.Tensor<input>,
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
          built_owned(kind, input, output, plan, entries, depth, length, valid, exact, source)

def copied(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +entries: List<&2, U32>, planned: Result<&2, &2, Error, Planned>) ->
  Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  match planned:
    case Fail{error}:
      (Core.Packed{shape, tensor}, Fail{error})
    case Done{Planned{+plan, +output}}:
      built(Forward{}, shape, output, plan, entries, tensor, Core.new(output, 0.0))

# PyTorch's input.index_select(dim, index). The input owner always comes back
# unchanged next to the result.
def index_select(input: Core.Dynamic, dim: Signed.Position, +entries: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  copied(shape, tensor, entries, select_plan(shape, dim, entries))

# PyTorch's input.gather(dim, index).
def gather(input: Core.Dynamic, dim: Signed.Position, +index: Index) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  Index{+index_shape, +entries} = index
  copied(shape, tensor, entries, gather_plan(shape, dim, Index{index_shape, entries}))

# The backward of a planned index_select or gather: the gradient g of its
# output, added into +0.0 words of the input's shape at the entries, PyTorch's
# index_add and scatter_add. g comes back unchanged next to the result.
def scatter(g: Core.Dynamic, +input: Shape.Shape, +plan: Plan, +entries: List<&2, U32>) -> Core.Dynamic & Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = g
  built(Backward{}, shape, input, plan, entries, tensor, Core.new(input, 0.0))

def error_code(error: Error) -> U32:
  match error:
    case TensorError{error}:
      Core.error_code(error)
    case LayoutError{error}:
      Slice.error_code(Slice.LayoutError{error})
    case OutOfRange{}:
      20
    case IndexRankMismatch{}:
      21
    case IndexShapeMismatch{}:
      22
