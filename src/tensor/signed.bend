import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./transpose-spec.bend as Transpose
import ./slice.bend as Slice

# PyTorch's signed axes, indices and reshape sizes, parsed into the unsigned
# ones that the tensor operations check. A Position counts from either end, as
# a PyTorch int does: First{k} is k, and Last{k} is -1 - k, so Last{0} is -1,
# the last axis or element. Each integer has exactly one Position.
type Position is Data:
  First{k: U32}
  Last{k: U32}

# A reshape size: an exact size, or -1, the one size PyTorch infers.
type Extent is Data:
  Exact{size: U32}
  Inferred{}

# A signed reshape's refusals: the unsigned reshape's, more than one Inferred
# size, and an Inferred size that any value would fit, on an empty input whose
# other sizes multiply to 0.
type Error is Data:
  TransformError{error: Transforms.Error}
  RepeatedInferred{}
  AmbiguousInferred{}

# PyTorch's wrap of a position among `limit` axes or elements (maybe_wrap_dim):
# a position d in -limit..limit-1 becomes d + limit when d is negative and d
# otherwise. A position outside that range stays outside it, at or past
# `limit`, so the operation refuses it with its own check, code and order:
# First{k} stays k, and an out-of-range Last{k} becomes `limit`.
def wrap(+limit: U32, position: Position) -> U32:
  match position:
    case First{k}:
      k
    case Last{+k}:
      Bool.pick(U32, U32.is_lt(k, limit), U32.sub(U32.sub(limit, 1), k), limit)

def wrap_all(+limit: U32, positions: List<&2, Position>) -> List<&2, U32>:
  match positions:
    case Nil{}:
      Nil{}
    case head <> tail:
      wrap(limit, head) <> wrap_all(limit, tail)

# Each coordinate wrapped among its own axis's extent. A coordinate past the
# rank has no extent and is wrapped among 0: the read refuses the coordinate
# count before any range.
def wrap_each(dimensions: List<&2, U32>, positions: List<&2, Position>) -> List<&2, U32>:
  match dimensions positions:
    case Nil{} positions:
      wrap_all(0, positions)
    case extent <> rest Nil{}:
      Nil{}
    case extent <> rest head <> tail:
      wrap(extent, head) <> wrap_each(rest, tail)

# Whether a position lies in PyTorch's range -limit..limit-1. Last{k} is
# -1 - k, at least -limit exactly when k < limit.
def inside(+limit: U32, position: Position) -> Bool:
  match position:
    case First{k}:
      U32.is_lt(k, limit)
    case Last{k}:
      U32.is_lt(k, limit)

def all_inside(+limit: U32, positions: List<&2, Position>) -> Bool:
  match positions:
    case Nil{}:
      True{}
    case head <> tail:
      Bool.and(inside(limit, head), all_inside(limit, tail))

# The axes of the input for transpose and sum: its rank, and 1 for a rank-0
# input, whose only axis is 0 (or -1).
def axis_limit(+input: Shape.Shape) -> U32:
  U32.from_nat(Transpose.axis_limit(Shape.rank_dimensions(Shape.dimensions(input))))

# How many sizes are Inferred.
def inferred_count(extents: List<&2, Extent>) -> Nat:
  match extents:
    case Nil{}:
      0n
    case Exact{size} <> tail:
      inferred_count(tail)
    case Inferred{} <> tail:
      1n+inferred_count(tail)

# The product of the exact sizes, in Nat: what the laws state the inference
# against. The inference never forms it, since it can pass the native Nat
# limit of 2^48 on an empty input: [-1, 2^24, 2^24] multiplies to 2^48.
def exact_product(extents: List<&2, Extent>) -> Nat:
  match extents:
    case Nil{}:
      1n
    case Exact{+size} <> tail:
      Nat.mul(U32.to_nat(size), exact_product(tail))
    case Inferred{} <> tail:
      exact_product(tail)

# The sizes, each Inferred one replaced by `size`.
def filled(extents: List<&2, Extent>, +size: U32) -> List<&2, U32>:
  match extents:
    case Nil{}:
      Nil{}
    case Exact{extent} <> tail:
      extent <> filled(tail, size)
    case Inferred{} <> tail:
      size <> filled(tail, size)

# The exact product against a bound, the input's element count: ZeroProduct
# when it is 0, Within{p} when it is p > 0 at most the bound, and Beyond when
# it is past the bound. capped only multiplies a size and a product that are
# both at most the bound, and a tensor holds at most 2^20 elements, so no
# number it forms passes 2^40.
type Product is Data:
  ZeroProduct{}
  Within{product: Nat}
  Beyond{}

def bounded(product: Nat, below: Bool) -> Product:
  match below:
    case True{}:
      Within{product}
    case False{}:
      Beyond{}

def cap(+bound: Nat, +product: Nat) -> Product:
  bounded(product, Nat.is_le(product, bound))

def scaled(+bound: Nat, +size: Nat, +product: Nat, past: Bool) -> Product:
  match past:
    case True{}:
      Beyond{}
    case False{}:
      cap(bound, Nat.mul(size, product))

# A nonzero size times the product of the sizes after it. A size past the
# bound makes any nonzero product pass it, so it is never multiplied.
def times(+bound: Nat, size: Nat, rest: Product) -> Product:
  match size:
    case 0n:
      ZeroProduct{}
    case 1n+(+less):
      match rest:
        case ZeroProduct{}:
          ZeroProduct{}
        case Within{+product}:
          scaled(bound, 1n+less, product, Nat.is_lt(bound, 1n+less))
        case Beyond{}:
          Beyond{}

def capped(+bound: Nat, extents: List<&2, Extent>) -> Product:
  match extents:
    case Nil{}:
      cap(bound, 1n)
    case Exact{+size} <> tail:
      times(bound, U32.to_nat(size), capped(bound, tail))
    case Inferred{} <> tail:
      capped(bound, tail)

def count_mismatch() -> Error:
  TransformError{Transforms.LayoutError{Layout.ElementCountMismatch{}}}

# An Inferred size whose other sizes multiply to 0: ambiguous on an empty
# input, a count mismatch on any other.
def unfilled(count: Nat) -> Result<&2, &2, Error, List<&2, U32>>:
  match count:
    case 0n:
      Fail{AmbiguousInferred{}}
    case 1n+rest:
      Fail{count_mismatch()}

def divided(+extents: List<&2, Extent>, split: Nat & Nat) -> Result<&2, &2, Error, List<&2, U32>>:
  (quotient, remainder) = split
  match remainder:
    case 0n:
      Done{filled(extents, U32.from_nat(quotient))}
    case 1n+rest:
      Fail{count_mismatch()}

# A product past the count: the inferred size is 0 on an empty input, and no
# size fits any other.
def beyond(count: Nat, +extents: List<&2, Extent>) -> Result<&2, &2, Error, List<&2, U32>>:
  match count:
    case 0n:
      Done{filled(extents, 0)}
    case 1n+rest:
      Fail{count_mismatch()}

def inferred_one(+count: Nat, +extents: List<&2, Extent>, product: Product) -> Result<&2, &2, Error, List<&2, U32>>:
  match product:
    case ZeroProduct{}:
      unfilled(count)
    case Within{product}:
      divided(extents, Nat.divmod(count, product))
    case Beyond{}:
      beyond(count, extents)

def inferred_from(+count: Nat, +extents: List<&2, Extent>, inferred: Nat) -> Result<&2, &2, Error, List<&2, U32>>:
  match inferred:
    case 0n:
      Done{filled(extents, 0)}
    case 1n+more:
      match more:
        case 0n:
          inferred_one(count, extents, capped(count, extents))
        case 1n+rest:
          Fail{RepeatedInferred{}}

# PyTorch's infer_size: the sizes of a reshape of `count` elements. With no
# Inferred size, the sizes themselves; with one, it is count / product of
# the others when that divides exactly, the product capped at the count;
# more than one is refused first.
def inferred_dimensions(+count: Nat, +extents: List<&2, Extent>) -> Result<&2, &2, Error, List<&2, U32>>:
  inferred_from(count, extents, inferred_count(extents))

def transformed(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Result<&2, &1, Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{TransformError{error}}
    case Done{output}:
      Done{output}

def reshape_inferred(input: Shape.Shape, tensor: Core.Tensor<input>,
  inferred: Result<&2, &2, Error, List<&2, U32>>) -> Result<&2, &1, Error, Core.Dynamic>:
  match inferred:
    case Fail{error}:
      Fail{error}
    case Done{+dimensions}:
      transformed(Transforms.reshape(input, tensor, dimensions))

# PyTorch's reshape with -1: the sizes inferred, then the unsigned reshape.
def reshape(+input: Shape.Shape, tensor: Core.Tensor<input>, +extents: List<&2, Extent>) -> Result<&2, &1, Error, Core.Dynamic>:
  reshape_inferred(input, tensor, inferred_dimensions(Shape.elements(input), extents))

def reshape_dynamic(dynamic: Core.Dynamic, extents: List<&2, Extent>) -> Result<&2, &1, Error, Core.Dynamic>:
  Core.Packed{+input, tensor} = dynamic
  reshape(input, tensor, extents)

# PyTorch's permute_copy with signed axes, wrapped among the input's rank.
def permute_copy(+input: Shape.Shape, source: Core.Tensor<input>, axes: List<&2, Position>) ->
  Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>:
  Transforms.permute_copy(input, source, wrap_all(Shape.rank(input), axes))

def permute_dynamic(dynamic: Core.Dynamic, axes: List<&2, Position>) -> Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>:
  Core.Packed{+input, source} = dynamic
  Transforms.dynamic_result(input, permute_copy(input, source, axes))

# PyTorch's transpose_copy with signed axes, wrapped among axis_limit.
def transpose_copy(+input: Shape.Shape, source: Core.Tensor<input>, a: Position, b: Position) ->
  Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>:
  Transforms.transpose_copy(input, source, wrap(axis_limit(input), a), wrap(axis_limit(input), b))

def transpose_dynamic(dynamic: Core.Dynamic, a: Position, b: Position) -> Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>:
  Core.Packed{+input, source} = dynamic
  Transforms.dynamic_result(input, transpose_copy(input, source, a, b))

# A[coordinates] with signed coordinates, each wrapped among its axis's
# extent; and the write A[coordinates] = value.
def read_coordinate(+shape: Shape.Shape, tensor: Core.Tensor<shape>, coordinates: List<&2, Position>) ->
  Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>:
  Core.read_coordinate(shape, tensor, wrap_each(Shape.dimensions(shape), coordinates))

def write_coordinate(+shape: Shape.Shape, tensor: Core.Tensor<shape>, coordinates: List<&2, Position>, value: F32) ->
  Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>:
  Core.write_coordinate(shape, tensor, wrap_each(Shape.dimensions(shape), coordinates), value)

# The element at a signed flat index, PyTorch's A.flatten()[index], wrapped
# among the element count; and the write at it.
def read_flat(+shape: Shape.Shape, tensor: Core.Tensor<shape>, index: Position) ->
  Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>:
  Core.read_flat(shape, tensor, wrap(Shape.count(shape), index))

def write_flat(+shape: Shape.Shape, tensor: Core.Tensor<shape>, index: Position, +value: F32) ->
  Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>:
  Core.write_flat(shape, tensor, wrap(Shape.count(shape), index), value)

def read_dynamic(dynamic: Core.Dynamic, index: Position) -> Core.Dynamic & Result<&2, &2, Core.Error, F32>:
  Core.Packed{+shape, tensor} = dynamic
  Core.read_dynamic_result(shape, read_flat(shape, tensor, index))

def write_dynamic(dynamic: Core.Dynamic, index: Position, value: F32) -> Core.Dynamic & Result<&2, &2, Core.Error, Unit>:
  Core.Packed{+shape, tensor} = dynamic
  Core.write_dynamic_result(shape, write_flat(shape, tensor, index, value))

# Whether a position lies before -limit: Last{k}, the position -1 - k, with
# k at or past limit. No First does.
def before(+limit: U32, position: Position) -> Bool:
  match position:
    case First{k}:
      False{}
    case Last{k}:
      Bool.not(U32.is_lt(k, limit))

# PyTorch's slice positions (aten::slice, get_slice_bounds) before its clamp
# to the axis: a negative position d becomes d + limit, the wrap, and one
# still below 0 becomes 0. First{k} stays k; Slice.bounds then clamps both
# ends to the extent.
def bound(+limit: U32, +position: Position) -> U32:
  Bool.pick(U32, before(limit, position), 0, wrap(limit, position))

# The extent of axis `axis`, 0 past the rank.
def extent(dimensions: List<&2, U32>, +axis: U32) -> U32:
  U32.from_nat(Slice.extent_at(dimensions, axis))

# PyTorch's input[..., start:stop:step, ...] with the slice at a signed axis
# `dim`, wrapped among the rank, and signed start and stop: the unsigned
# slice of the bounds. A rank-0 input has no axis, so every dim is refused.
def slice(input: Core.Dynamic, dim: Position, start: Position, stop: Position, +step: U32) ->
  Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  +axis = wrap(Shape.rank(shape), dim)
  +n = extent(Shape.dimensions(shape), axis)
  Slice.slice(Core.Packed{shape, tensor}, axis, bound(n, start), bound(n, stop), step)

# PyTorch's input.select(dim, index) at a signed axis and a signed index,
# wrapped among the extent: an index outside -extent..extent-1 stays at or
# past the extent, so the unsigned select refuses it with OutOfRange after
# its axis check.
def select(input: Core.Dynamic, dim: Position, index: Position) -> Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  +axis = wrap(Shape.rank(shape), dim)
  Slice.select(Core.Packed{shape, tensor}, axis, wrap(extent(Shape.dimensions(shape), axis), index))

# A narrow's start before -extent: the axis check first, as the unsigned
# narrow does it, then OutOfRange.
def narrow_before(input: Core.Dynamic, refused: Maybe<&2, Slice.Error>) -> Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  match refused:
    case Some{error}:
      (input, Fail{error})
    case None{}:
      (input, Fail{Slice.OutOfRange{}})

def narrow_started(+shape: Shape.Shape, tensor: Core.Tensor<shape>, +axis: U32, +start: U32, +length: U32, early: Bool) ->
  Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  match early:
    case False{}:
      Slice.narrow(Core.Packed{shape, tensor}, axis, start, length)
    case True{}:
      narrow_before(Core.Packed{shape, tensor}, Slice.refusal(Shape.dimensions(shape), axis, 1))

# PyTorch's input.narrow(dim, start, length) at a signed axis with a signed
# start. PyTorch takes a start in -extent..extent: Last{k} with k below the
# extent is extent - 1 - k, and one before -extent is OutOfRange. A First
# start past the extent is refused by the unsigned narrow, which checks
# start + length against the extent, with the same code. A length is a
# U32, so PyTorch's negative lengths cannot be written.
def narrow(input: Core.Dynamic, dim: Position, +start: Position, +length: U32) ->
  Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>:
  Core.Packed{+shape, tensor} = input
  +axis = wrap(Shape.rank(shape), dim)
  +n = extent(Shape.dimensions(shape), axis)
  narrow_started(shape, tensor, axis, wrap(n, start), length, before(n, start))

def error_code(error: Error) -> U32:
  match error:
    case TransformError{error}:
      Transforms.error_code(error)
    case RepeatedInferred{}:
      17
    case AmbiguousInferred{}:
      18
