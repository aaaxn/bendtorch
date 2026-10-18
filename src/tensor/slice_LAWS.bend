import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/refinement.bend as Refinement
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./elementwise_totality.bend as Totality
import ./slice.bend as Slice

# The laws of basic slicing (slice.bend), written from PyTorch's definition
# of a slice: which start and stop it clamps to, how long the sliced axis
# is, how the row-major view [outer, extent, inner] sits around the axis,
# which calls it refuses, and what every cell of the forward copy and of the
# backward scatter holds.

type Holds<-P: Type> is Data:
  Holds{-proof: P}

def low(+clamped: Slice.Bounds) -> Nat:
  Slice.Bounds{start, stop} = clamped
  start

def high(+clamped: Slice.Bounds) -> Nat:
  Slice.Bounds{start, stop} = clamped
  stop

# PyTorch's clamp of a slice to an axis of extent n. A start at most n is
# kept, a start past n becomes n.
law slice_start_kept:
  for +extent: Nat
  for +start: Nat
  for +stop: Nat
  for inside: {Nat.is_le(start, extent) == True{} : Bool}
  {low(Slice.bounds(extent, start, stop)) == start : Nat}

law slice_start_clamped:
  for +extent: Nat
  for +start: Nat
  for +stop: Nat
  for past: {Nat.is_le(extent, start) == True{} : Bool}
  {low(Slice.bounds(extent, start, stop)) == extent : Nat}

# A stop from the start to n is kept, a stop past n becomes n, and a stop
# at or before the start ends the slice at its start: an empty axis.
law slice_stop_kept:
  for +extent: Nat
  for +start: Nat
  for +stop: Nat
  for +after: {Nat.is_le(start, stop) == True{} : Bool}
  for +inside: {Nat.is_le(stop, extent) == True{} : Bool}
  {high(Slice.bounds(extent, start, stop)) == stop : Nat}

law slice_stop_clamped:
  for +extent: Nat
  for +start: Nat
  for +stop: Nat
  for past: {Nat.is_le(extent, stop) == True{} : Bool}
  {high(Slice.bounds(extent, start, stop)) == extent : Nat}

law slice_stop_empty:
  for +extent: Nat
  for +start: Nat
  for +stop: Nat
  for before: {Nat.is_le(stop, start) == True{} : Bool}
  {high(Slice.bounds(extent, start, stop)) == low(Slice.bounds(extent, start, stop)) : Nat}

# The length of the sliced axis is the number of indices start, start +
# step, start + 2 step, ... below the stop: each index below the length
# lies below the stop, and, for a positive step, the index at the length
# does not.
law slice_length_below:
  for +start: Nat
  for +stop: Nat
  for +step: Nat
  for +k: Nat
  for below: {Nat.is_lt(k, Slice.length(Slice.Bounds{start, stop}, step)) == True{} : Bool}
  {Nat.is_lt(Nat.add(start, Nat.mul(k, step)), stop) == True{} : Bool}

law slice_length_reaches:
  for +start: Nat
  for +stop: Nat
  for +step: Nat
  for positive: {Nat.is_lt(0n, step) == True{} : Bool}
  {Nat.is_le(stop, Nat.add(start, Nat.mul(Slice.length(Slice.Bounds{start, stop}, step), step))) == True{} : Bool}

# The row-major view around axis `dim` of dimensions prefix ++ [extent] ++
# suffix, with `dim` the rank of the prefix: the element count of the
# prefix, the extent, and the element count of the suffix.
law slice_view_around:
  for +prefix: List<&2, U32>
  for +extent: U32
  for +suffix: List<&2, U32>
  {Slice.view(List.append(&2, U32, prefix, extent <> suffix), Shape.rank_dimensions(prefix)) ==
    Slice.View{Shape.logical_numel(prefix), U32.to_nat(extent), Shape.logical_numel(suffix)} : Slice.View}

# The sliced output keeps every other axis and takes the new extent at the
# axis; `select` drops the axis.
law slice_resized_around:
  for +prefix: List<&2, U32>
  for +extent: U32
  for +suffix: List<&2, U32>
  for +other: U32
  {Slice.resized(List.append(&2, U32, prefix, extent <> suffix), Shape.rank_dimensions(prefix), other) ==
    List.append(&2, U32, prefix, other <> suffix) : List<&2, U32>}

law select_dropped_around:
  for +prefix: List<&2, U32>
  for +extent: U32
  for +suffix: List<&2, U32>
  {Slice.dropped(List.append(&2, U32, prefix, extent <> suffix), Shape.rank_dimensions(prefix)) ==
    List.append(&2, U32, prefix, suffix) : List<&2, U32>}

# What a plan holds along its axis (the extent, the clamped start, the step
# and the length) and around it (the element counts before and after it).
def axis_part(+plan: Slice.Plan) -> Nat & Nat & Nat & Nat:
  Slice.Plan{outer, extent, inner, start, step, length} = plan
  (extent, start, step, length)

def view_part(+plan: Slice.Plan) -> Nat & Nat:
  Slice.Plan{outer, extent, inner, start, step, length} = plan
  (outer, inner)

# PyTorch's clamp of a slice position to an axis of extent n: the position
# if it is at most n, else n.
def clamp(+position: Nat, +extent: Nat) -> Nat:
  Bool.pick(Nat, Nat.is_le(position, extent), position, extent)

# The stop after the clamp: the clamped stop, or the clamped start where the
# clamped stop falls before it, so the axis is empty.
def clamped_stop(+start: Nat, +stop: Nat, +extent: Nat) -> Nat:
  Bool.pick(Nat, Nat.is_le(clamp(start, extent), clamp(stop, extent)), clamp(stop, extent), clamp(start, extent))

# What a request start:stop:step plans along an axis of extent n: the
# extent, the clamped start, the step, and the length of the clamped range.
def axis_spec(+extent: Nat, +start: Nat, +stop: Nat, +step: Nat) -> Nat & Nat & Nat & Nat:
  (extent, clamp(start, extent), step,
    Slice.length(Slice.Bounds{clamp(start, extent), clamped_stop(start, stop, extent)}, step))

# The plan of a request at axis `dim` of prefix ++ [extent] ++ suffix, with
# `dim` the rank of the prefix, holds the request's own start, stop and step,
# clamped to that extent, whatever the other extents, an empty input's
# included.
law slice_plan_axis:
  for +prefix: List<&2, U32>
  for +extent: U32
  for +suffix: List<&2, U32>
  for +dim: U32
  for at: {U32.to_nat(dim) == Shape.rank_dimensions(prefix) : Nat}
  for +start: U32
  for +stop: U32
  for +step: U32
  {axis_part(Slice.plan_of(List.append(&2, U32, prefix, extent <> suffix), dim, start, stop, step)) ==
    axis_spec(U32.to_nat(extent), U32.to_nat(start), U32.to_nat(stop), U32.to_nat(step)) : Nat & Nat & Nat & Nat}

# For an input with an element, the plan's view is the element counts of the
# prefix and of the suffix, the row-major view around the axis.
law slice_plan_view:
  for +prefix: List<&2, U32>
  for +extent: U32
  for +suffix: List<&2, U32>
  for +dim: U32
  for at: {U32.to_nat(dim) == Shape.rank_dimensions(prefix) : Nat}
  for +start: U32
  for +stop: U32
  for +step: U32
  for nonempty: {Nat.is_lt(0n, Shape.logical_numel(List.append(&2, U32, prefix, extent <> suffix))) == True{} : Bool}
  {view_part(Slice.plan_of(List.append(&2, U32, prefix, extent <> suffix), dim, start, stop, step)) ==
    (Shape.logical_numel(prefix), Shape.logical_numel(suffix)) : Nat & Nat}

# An input with no element gets a plan with no view: neither build has a
# cell then, so the plan forms no product of the other extents, which can
# pass the largest Nat the runtime holds although no element exists.
law slice_plan_empty_view:
  for +dimensions: List<&2, U32>
  for +dim: U32
  for +start: U32
  for +stop: U32
  for +step: U32
  for empty: {Shape.logical_numel(dimensions) == 0n : Nat}
  {view_part(Slice.plan_of(dimensions, dim, start, stop, step)) == (0n, 0n) : Nat & Nat}

# PyTorch's checks, in its order. An axis at or past the rank (every axis
# of a rank-0 input) is refused first, then a step of 0.
law slice_axis_refused:
  for +input: Shape.Shape
  for +dim: U32
  for +start: U32
  for +stop: U32
  for +step: U32
  for outside: {Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(input))) == False{} : Bool}
  {Slice.plan(input, dim, start, stop, step) == Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}} :
    Result<&2, &2, Slice.Error, Slice.Planned>}

law slice_step_refused:
  for +input: Shape.Shape
  for +dim: U32
  for +start: U32
  for +stop: U32
  for +step: U32
  for inside: {Nat.is_lt(U32.to_nat(dim), Shape.rank_dimensions(Shape.dimensions(input))) == True{} : Bool}
  for zero: {U32.to_nat(step) == 0n : Nat}
  {Slice.plan(input, dim, start, stop, step) == Fail{Slice.StepNotPositive{}} : Result<&2, &2, Slice.Error, Slice.Planned>}

# What an accepted slice plans: the view around the axis, the start and
# stop clamped to the extent, the step, the length of the sliced axis, and
# the output's dimensions, the input's with that length at the axis.
def PlannedAs(+dimensions: List<&2, U32>, +dim: U32, +start: U32, +stop: U32, +step: U32,
  result: Result<&2, &2, Slice.Error, Slice.Planned>) -> Type:
  match result:
    case Fail{error}:
      Unit
    case Done{Slice.Planned{plan, output}}:
      {plan == Slice.plan_of(dimensions, dim, start, stop, step) : Slice.Plan} &
        {Shape.dimensions(output) == Slice.resized(dimensions, U32.to_nat(dim), U32.from_nat(Slice.plan_length(plan))) : List<&2, U32>}

law slice_planned:
  for +input: Shape.Shape
  for +dim: U32
  for +start: U32
  for +stop: U32
  for +step: U32
  PlannedAs(Shape.dimensions(input), dim, start, stop, step, Slice.plan(input, dim, start, stop, step))

# A refused plan fails the slice with its error and hands the input back.
law slice_refused:
  for input: Core.Dynamic
  for +dim: U32
  for +start: U32
  for +stop: U32
  for +step: U32
  for error: Slice.Error
  for refused: {Slice.plan(Totality.dynamic_shape(input), dim, start, stop, step) == Fail{error} : Result<&2, &2, Slice.Error, Slice.Planned>}
  {Slice.slice(input, dim, start, stop, step) == (input, Fail{error}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

def rank_of(input: Core.Dynamic) -> Nat:
  Shape.rank_dimensions(Shape.dimensions(Totality.dynamic_shape(input)))

def extent_of(input: Core.Dynamic, +dim: U32) -> Nat:
  Slice.extent_at(Shape.dimensions(Totality.dynamic_shape(input)), dim)

# PyTorch's narrow(dim, start, length) is the slice start:start + length
# with step 1, once the axis is in range and start + length is at most the
# extent; a start + length past the extent is refused with OutOfRange, an
# axis past the rank as a slice refuses it.
law narrow_slices:
  for input: Core.Dynamic
  for +dim: U32
  for +start: U32
  for +length: U32
  for inside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == True{} : Bool}
  for fits: {Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), extent_of(input, dim)) == True{} : Bool}
  {Slice.narrow(input, dim, start, length) == Slice.slice(input, dim, start, U32.add(start, length), 1) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law narrow_out_of_range:
  for input: Core.Dynamic
  for +dim: U32
  for +start: U32
  for +length: U32
  for inside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == True{} : Bool}
  for past: {Nat.is_le(Nat.add(U32.to_nat(start), U32.to_nat(length)), extent_of(input, dim)) == False{} : Bool}
  {Slice.narrow(input, dim, start, length) == (input, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law narrow_axis_refused:
  for input: Core.Dynamic
  for +dim: U32
  for +start: U32
  for +length: U32
  for outside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == False{} : Bool}
  {Slice.narrow(input, dim, start, length) == (input, Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}}) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

# A slice's result with its output reshaped to `dimensions`.
def reshape_as_slice(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, Slice.Error, Core.Dynamic>:
  match result:
    case Fail{Transforms.LayoutError{error}}:
      Fail{Slice.LayoutError{error}}
    case Fail{Transforms.TensorError{error}}:
      Fail{Slice.TensorError{error}}
    case Done{output}:
      Done{output}

def reshaped(+dimensions: List<&2, U32>, pair: Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (input, Fail{error})
    case Done{output}:
      (input, reshape_as_slice(Transforms.reshape_dynamic(output, dimensions)))

# PyTorch's select(dim, index) is the slice index:index + 1 with step 1,
# reshaped to the input's dimensions without the axis, once the axis is in
# range and the index below the extent; an index at or past the extent is
# refused with OutOfRange, an axis past the rank as a slice refuses it.
law select_slices:
  for input: Core.Dynamic
  for +dim: U32
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == True{} : Bool}
  for below: {Nat.is_lt(U32.to_nat(index), extent_of(input, dim)) == True{} : Bool}
  {Slice.select(input, dim, index) ==
    reshaped(Slice.dropped(Shape.dimensions(Totality.dynamic_shape(input)), U32.to_nat(dim)), Slice.slice(input, dim, index, U32.add(index, 1), 1)) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law select_out_of_range:
  for input: Core.Dynamic
  for +dim: U32
  for +index: U32
  for inside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == True{} : Bool}
  for past: {Nat.is_lt(U32.to_nat(index), extent_of(input, dim)) == False{} : Bool}
  {Slice.select(input, dim, index) == (input, Fail{Slice.OutOfRange{}}) : Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law select_axis_refused:
  for input: Core.Dynamic
  for +dim: U32
  for +index: U32
  for outside: {Nat.is_lt(U32.to_nat(dim), rank_of(input)) == False{} : Bool}
  {Slice.select(input, dim, index) == (input, Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}}) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

# The cells of the builds, in the coordinates of the views: output cell
# (o, k, r) of [outer, length, inner] sits at row-major position
# (o * length + k) * inner + r, and input cell (o, i, r) of [outer, extent,
# inner] at (o * extent + i) * inner + r.
def position(+rows: Nat, +inner: Nat, +o: Nat, +i: Nat, +r: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(o, rows), i), inner), r)

# For a balanced input array A and every split, the slot of the forward
# build that holds output cell (o, k, r), below the count, holds input cell
# (o, start + k * step, r): the word PyTorch's slice puts there.
law slice_build_cells:
  for +depth: Nat
  for +splits: Nat
  for +outer: Nat
  for +extent: Nat
  for +inner: Nat
  for +start: Nat
  for +step: Nat
  for +length: Nat
  for +count: Nat
  for +base: Nat
  for +da: Nat
  for -A: Array<F32>
  for a: Array<F32>
  for same: {a == A : Array<F32>}
  for valid: Model.Balanced(F32, da, A)
  for +path: Model.Path(depth)
  for +o: Nat
  for +k: Nat
  for +r: Nat
  for k_below: {Nat.is_lt(k, length) == True{} : Bool}
  for r_below: {Nat.is_lt(r, inner) == True{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == position(length, inner, o, k, r) : Nat}
  for inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool}
  Holds<{Model.observe(F32, depth, Slice.build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, a), path) ==
    Model.observe(F32, da, A, Refinement.route(da, U32.from_nat(position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r)))) :
    Maybe<&2, F32>}>

# For a balanced gradient array G and every split, the slot of the backward
# build that holds input cell (o, start + k * step, r), with k below the
# length and a positive step, holds gradient cell (o, k, r) ...
law slice_scatter_hit:
  for +depth: Nat
  for +splits: Nat
  for +outer: Nat
  for +extent: Nat
  for +inner: Nat
  for +start: Nat
  for +step: Nat
  for +length: Nat
  for +count: Nat
  for +base: Nat
  for +dg: Nat
  for -G: Array<F32>
  for g: Array<F32>
  for same: {g == G : Array<F32>}
  for valid: Model.Balanced(F32, dg, G)
  for +path: Model.Path(depth)
  for +o: Nat
  for +k: Nat
  for +r: Nat
  for positive: {Nat.is_lt(0n, step) == True{} : Bool}
  for k_below: {Nat.is_lt(k, length) == True{} : Bool}
  for i_below: {Nat.is_lt(Nat.add(start, Nat.mul(k, step)), extent) == True{} : Bool}
  for r_below: {Nat.is_lt(r, inner) == True{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == position(extent, inner, o, Nat.add(start, Nat.mul(k, step)), r) : Nat}
  for inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool}
  Holds<{Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
    Model.observe(F32, dg, G, Refinement.route(dg, U32.from_nat(position(length, inner, o, k, r)))) : Maybe<&2, F32>}>

# ... and the slot of input cell (o, i, r) where no k below the length has
# i = start + k * step holds +0.0, as PyTorch's slice_backward leaves it.
law slice_scatter_miss:
  for +depth: Nat
  for +splits: Nat
  for +outer: Nat
  for +extent: Nat
  for +inner: Nat
  for +start: Nat
  for +step: Nat
  for +length: Nat
  for +count: Nat
  for +base: Nat
  for +dg: Nat
  for -G: Array<F32>
  for g: Array<F32>
  for same: {g == G : Array<F32>}
  for valid: Model.Balanced(F32, dg, G)
  for +path: Model.Path(depth)
  for +o: Nat
  for +i: Nat
  for +r: Nat
  for i_below: {Nat.is_lt(i, extent) == True{} : Bool}
  for r_below: {Nat.is_lt(r, inner) == True{} : Bool}
  for missed: @+k: Nat -> @below: {Nat.is_lt(k, length) == True{} : Bool} ->
    {Nat.is_eq(Nat.add(start, Nat.mul(k, step)), i) == False{} : Bool}
  for at: {Nat.add(Model.offset(depth, path), base) == position(extent, inner, o, i, r) : Nat}
  for inside: {Nat.is_lt(Nat.add(Model.offset(depth, path), base), count) == True{} : Bool}
  Holds<{Model.observe(F32, depth, Slice.scatter_build(depth, splits, Slice.Plan{outer, extent, inner, start, step, length}, count, base, g), path) ==
    Some{0.0} : Maybe<&2, F32>}>
