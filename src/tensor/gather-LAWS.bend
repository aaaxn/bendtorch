import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./layout.bend as Layout
import ./elementwise-totality.bend as Totality
import ./slice.bend as Slice
import ./signed.bend as Signed
import ./gather.bend as Gather

# The laws of index_select and gather (gather.bend), written from PyTorch's
# definitions: which requests they refuse and in which order, the plan and
# the output shape of the ones they take, what every cell of the forward
# copy holds, and what every cell of the backward, index_add and
# scatter_add into +0.0, holds.

type Holds<-P: Type> is Data:
  Holds{-proof: P}

# The axis a request names: its signed dim wrapped among the input's rank.
def axis(+input: Shape.Shape, dim: Signed.Position) -> Nat:
  U32.to_nat(Signed.wrap(Shape.rank(input), dim))

def rank(+input: Shape.Shape) -> Nat:
  Shape.rank_dimensions(Shape.dimensions(input))

def extent(+input: Shape.Shape, +at: Nat) -> Nat:
  Slice.axis_extent(Shape.dimensions(input), at)

# How many entries an index holds.
def length(entries: List<&2, U32>) -> Nat:
  match entries:
    case Nil{}:
      0n
    case head <> rest:
      1n+length(rest)

# Whether every entry lies below the extent.
def inside(entries: List<&2, U32>, +bound: Nat) -> Bool:
  match entries:
    case Nil{}:
      True{}
    case head <> rest:
      Bool.and(Nat.is_lt(U32.to_nat(head), bound), inside(rest, bound))

# Whether two lists of extents are equal, extent by extent.
def equal(left: List<&2, U32>, right: List<&2, U32>) -> Bool:
  match left right:
    case Nil{} Nil{}:
      True{}
    case x <> xs y <> ys:
      Bool.and(Nat.is_eq(U32.to_nat(x), U32.to_nat(y)), equal(xs, ys))
    case _ _:
      False{}

# Whether a gather index has the input's extent at every axis but `at`.
def others(dimensions: List<&2, U32>, index: List<&2, U32>, +at: Nat) -> Bool:
  match dimensions index at:
    case x <> xs y <> ys 0n:
      equal(xs, ys)
    case x <> xs y <> ys 1n+next:
      Bool.and(Nat.is_eq(U32.to_nat(x), U32.to_nat(y)), others(xs, ys, next))
    case _ _ _:
      False{}

# The plan of a request the checks take: the row-major view of the input
# around the axis, slice.bend's, and the length of the output's axis ...
def viewed(+kind: Gather.Lookup, +given: Slice.View, +count: Nat) -> Gather.Plan:
  Slice.View{+outer, +rows, +inner} = given
  Gather.Plan{kind, outer, rows, inner, count}

# ... and its output, the input's dimensions with that length at the axis,
# or the refusal of the shape's own checks (rank and capacity).
def output(-dimensions: List<&2, U32>, +plan: Gather.Plan, built: Shape.ConstructResult<dimensions>) ->
  Result<&2, &2, Gather.Error, Gather.Planned>:
  match built:
    case Shape.SError{error}:
      Fail{Gather.TensorError{Core.ShapeError{error}}}
    case Shape.SBuilt{+shape, preserved}:
      Done{Gather.Planned{plan, shape}}

def taken(+kind: Gather.Lookup, +dimensions: List<&2, U32>, +at: Nat, +count: Nat) -> Result<&2, &2, Gather.Error, Gather.Planned>:
  output(Slice.resized(dimensions, at, U32.from_nat(count)),
    viewed(kind, Slice.sized(dimensions, at, Slice.holds_zero(dimensions)), count),
    Shape.construct(Slice.resized(dimensions, at, U32.from_nat(count))))

# index_select(dim, index): an axis outside -rank..rank-1 is refused first,
# then an entry at or past the extent; any other request is planned.
law index_select_axis_refused:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +entries: List<&2, U32>
  for outside: {Nat.is_lt(axis(input, dim), rank(input)) == False{} : Bool}
  {Gather.select_plan(input, dim, entries) == Fail{Gather.LayoutError{Layout.AxisOutOfBounds{}}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law index_select_out_of_range:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +entries: List<&2, U32>
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for past: {inside(entries, extent(input, axis(input, dim))) == False{} : Bool}
  {Gather.select_plan(input, dim, entries) == Fail{Gather.OutOfRange{}} : Result<&2, &2, Gather.Error, Gather.Planned>}

law index_select_planned:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +entries: List<&2, U32>
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for within: {inside(entries, extent(input, axis(input, dim))) == True{} : Bool}
  {Gather.select_plan(input, dim, entries) == taken(Gather.Selected{}, Shape.dimensions(input), axis(input, dim), length(entries)) :
    Result<&2, &2, Gather.Error, Gather.Planned>}

# gather(dim, index): an entry count other than the product of the index's
# dimensions is refused first, then the axis, an index of another rank, one
# with another extent at another axis, and an entry at or past the extent;
# any other request is planned, with the index's extent at the axis.
law gather_count_refused:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for miscounted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == False{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) == Fail{Gather.TensorError{Core.ValueCountMismatch{}}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law gather_axis_refused:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for counted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == True{} : Bool}
  for outside: {Nat.is_lt(axis(input, dim), rank(input)) == False{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) == Fail{Gather.LayoutError{Layout.AxisOutOfBounds{}}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law gather_rank_refused:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for counted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == True{} : Bool}
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for ranked: {Nat.is_eq(Shape.rank_dimensions(dimensions), rank(input)) == False{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) == Fail{Gather.IndexRankMismatch{}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law gather_shape_refused:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for counted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == True{} : Bool}
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for ranked: {Nat.is_eq(Shape.rank_dimensions(dimensions), rank(input)) == True{} : Bool}
  for unmatched: {others(Shape.dimensions(input), dimensions, axis(input, dim)) == False{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) == Fail{Gather.IndexShapeMismatch{}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law gather_out_of_range:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for counted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == True{} : Bool}
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for ranked: {Nat.is_eq(Shape.rank_dimensions(dimensions), rank(input)) == True{} : Bool}
  for matched: {others(Shape.dimensions(input), dimensions, axis(input, dim)) == True{} : Bool}
  for past: {inside(entries, extent(input, axis(input, dim))) == False{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) == Fail{Gather.OutOfRange{}} :
    Result<&2, &2, Gather.Error, Gather.Planned>}

law gather_planned:
  for +input: Shape.Shape
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for counted: {Nat.is_eq(length(entries), Shape.logical_numel(dimensions)) == True{} : Bool}
  for in_rank: {Nat.is_lt(axis(input, dim), rank(input)) == True{} : Bool}
  for ranked: {Nat.is_eq(Shape.rank_dimensions(dimensions), rank(input)) == True{} : Bool}
  for matched: {others(Shape.dimensions(input), dimensions, axis(input, dim)) == True{} : Bool}
  for within: {inside(entries, extent(input, axis(input, dim))) == True{} : Bool}
  {Gather.gather_plan(input, dim, Gather.Index{dimensions, entries}) ==
    taken(Gather.Gathered{}, Shape.dimensions(input), axis(input, dim), Slice.axis_extent(dimensions, axis(input, dim))) :
    Result<&2, &2, Gather.Error, Gather.Planned>}

# A refused plan fails the copy with its error and hands the input back.
law index_select_refused:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +entries: List<&2, U32>
  for error: Gather.Error
  for refused: {Gather.select_plan(Totality.dynamic_shape(input), dim, entries) == Fail{error} : Result<&2, &2, Gather.Error, Gather.Planned>}
  {Gather.index_select(input, dim, entries) == (input, Fail{error}) : Core.Dynamic & Result<&2, &1, Gather.Error, Core.Dynamic>}

law gather_refused:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +dimensions: List<&2, U32>
  for +entries: List<&2, U32>
  for error: Gather.Error
  for refused: {Gather.gather_plan(Totality.dynamic_shape(input), dim, Gather.Index{dimensions, entries}) == Fail{error} :
    Result<&2, &2, Gather.Error, Gather.Planned>}
  {Gather.gather(input, dim, Gather.Index{dimensions, entries}) == (input, Fail{error}) : Core.Dynamic & Result<&2, &1, Gather.Error, Core.Dynamic>}

# The cells of the builds, in the coordinates of the views: cell (o, k, r)
# of [outer, rows, inner] sits at (o * rows + k) * inner + r.
def position(+rows: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(o, rows), k), inner), r)

# The entry output cell (o, k, r) of [outer, length, inner] reads: entry k
# of an index_select's index, the entry at the cell itself of a gather's.
def entry_at(+kind: Gather.Lookup, +count: Nat, +inner: Nat, +o: Nat, +k: Nat, +r: Nat) -> Nat:
  match kind:
    case Gather.Selected{}:
      k
    case Gather.Gathered{}:
      position(count, inner, o, k, r)

# For a balanced input array A, a balanced index array I and every split,
# the slot of the forward build that holds output cell (o, k, r), below
# the count, holds input cell (o, j, r), where j is the entry the cell
# reads: the word PyTorch's index_select and gather put there.
law gather_build_cells:
  for +depth: Nat
  for +splits: Nat
  for +kind: Gather.Lookup
  for +outer: Nat
  for +rows: Nat
  for +inner: Nat
  for +count: Nat
  for +cells: Nat
  for +base: Nat
  for +da: Nat
  for -A: Array<F32>
  for a: Array<F32>
  for same: {a == A : Array<F32>}
  for valid: Model.Balanced(F32, da, A)
  for +di: Nat
  for -I: Array<U32>
  for index: Array<U32>
  for same_index: {index == I : Array<U32>}
  for index_valid: Model.Balanced(U32, di, I)
  for +path: Model.Path(depth)
  for +o: Nat
  for +k: Nat
  for +r: Nat
  for k_below: {Nat.is_lt(k, count) == True{} : Bool}
  for r_below: {Nat.is_lt(r, inner) == True{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == position(count, inner, o, k, r) : Nat}
  for in_count: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), cells) == True{} : Bool}
  for +j: U32
  for read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(entry_at(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>}
  Holds<{Model.observe(F32, depth, Gather.build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, a, index), path) ==
    Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(position(rows, inner, o, U32.to_nat(j), r)))) : Maybe<&2, F32>}>

# The backward of input cell (o, i, r): from gradient cell (o, k, r) on, for
# `left` more k, a cell whose entry is i adds its word to the sum, in
# increasing k; a cell whose entry is another skips. `value` is the sum at
# the end.
def Added(+dg: Nat, -G: Array<F32>, +di: Nat, -I: Array<U32>, +kind: Gather.Lookup, +count: Nat, +inner: Nat, +o: Nat, +i: Nat, +r: Nat,
  left: Nat, +k: Nat, +acc: F32, +value: F32) -> Type:
  match left:
    case 0n:
      {value == acc : F32}
    case 1n+rest:
      AddStep<dg, G, di, I, kind, count, inner, o, i, r, rest, k, acc, value>

type AddStep<-dg: Nat, -G: Array<F32>, -di: Nat, -I: Array<U32>, -kind: Gather.Lookup, -count: Nat, -inner: Nat, -o: Nat, -i: Nat, -r: Nat,
  -rest: Nat, -k: Nat, -acc: F32, -value: F32> is Data:
  Skips{-j: U32,
    -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(entry_at(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>},
    -other: {Nat.is_eq(U32.to_nat(j), i) == False{} : Bool},
    -next: Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, acc, value)}
  Adds{-j: U32,
    -read: {Model.observe(U32, di, I, Refinement.route(di, U32.from_nat(entry_at(kind, count, inner, o, k, r)))) == Some{j} : Maybe<&2, U32>},
    -same: {Nat.is_eq(U32.to_nat(j), i) == True{} : Bool},
    -x: F32,
    -word: {Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(position(count, inner, o, k, r)))) == Some{x} : Maybe<&2, F32>},
    -next: Added(dg, G, di, I, kind, count, inner, o, i, r, rest, 1n+k, F32.add(acc, x), value)}

# A slot that holds `value`, the sum Added names from +0.0 over every k.
type Summed<-depth: Nat, -out: Array<F32>, -path: Model.Path(depth), -dg: Nat, -G: Array<F32>, -di: Nat, -I: Array<U32>,
  -kind: Gather.Lookup, -count: Nat, -inner: Nat, -o: Nat, -i: Nat, -r: Nat> is Data:
  Summed{-value: F32, -observed: {Model.observe(F32, depth, out, path) == Some{value} : Maybe<&2, F32>},
    -added: Added(dg, G, di, I, kind, count, inner, o, i, r, count, 0n, 0.0, value)}

# For a balanced gradient array G, a balanced index array I and every split,
# the slot of the backward build that holds input cell (o, i, r), below the
# count, holds the gradient words of the cells (o, k, r) whose entry is i,
# added to +0.0 in increasing k, and +0.0 where no entry is i: PyTorch's
# one-thread index_add and scatter_add.
law gather_scatter_cells:
  for +depth: Nat
  for +splits: Nat
  for +kind: Gather.Lookup
  for +outer: Nat
  for +rows: Nat
  for +inner: Nat
  for +count: Nat
  for +cells: Nat
  for +base: Nat
  for +dg: Nat
  for -G: Array<F32>
  for g: Array<F32>
  for same: {g == G : Array<F32>}
  for valid: Model.Balanced(F32, dg, G)
  for +di: Nat
  for -I: Array<U32>
  for index: Array<U32>
  for same_index: {index == I : Array<U32>}
  for index_valid: Model.Balanced(U32, di, I)
  for +path: Model.Path(depth)
  for +o: Nat
  for +i: Nat
  for +r: Nat
  for i_below: {Nat.is_lt(i, rows) == True{} : Bool}
  for r_below: {Nat.is_lt(r, inner) == True{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == position(rows, inner, o, i, r) : Nat}
  for in_count: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), cells) == True{} : Bool}
  Summed<depth, Gather.scatter_build(depth, splits, Gather.Plan{kind, outer, rows, inner, count}, cells, base, g, index), path,
    dg, G, di, I, kind, count, inner, o, i, r>
