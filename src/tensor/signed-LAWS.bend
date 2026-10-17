import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./transpose-spec.bend as Transpose
import ./elementwise-totality.bend as Totality
import ./signed.bend as Signed
import ./slice.bend as Slice

# PyTorch's signed axes, indices and -1 reshape sizes (signed.bend). These
# laws state the wrap against PyTorch's definition, the range it accepts, the
# reshape size it infers and the cases it refuses, and that each signed
# operation is the unsigned one on the wrapped arguments; the unsigned
# operations' laws then give the result and the refusal codes.

# Exact sizes, the reshape request of the unsigned reshape.
def exacts(dimensions: List<&2, U32>) -> List<&2, Signed.Extent>:
  match dimensions:
    case Nil{}:
      Nil{}
    case extent <> rest:
      Signed.Exact{extent} <> exacts(rest)

# A position counted from the start is kept.
law signed_first_kept:
  for +limit: U32
  for +k: U32
  {Signed.wrap(limit, Signed.First{k}) == k : U32}

# Last{k} is the position d = -1 - k. In range, it wraps to d + limit:
# the wrapped position plus k + 1 is limit, in exact arithmetic.
law signed_last_inside:
  for +limit: U32
  for +k: U32
  for inside: {U32.is_lt(k, limit) == True{} : Bool}
  {Nat.add(U32.to_nat(Signed.wrap(limit, Signed.Last{k})), 1n+U32.to_nat(k)) == U32.to_nat(limit) : Nat}

# Below -limit, it becomes limit itself, which every check refuses.
law signed_last_outside:
  for +limit: U32
  for +k: U32
  for outside: {U32.is_lt(k, limit) == False{} : Bool}
  {Signed.wrap(limit, Signed.Last{k}) == limit : U32}

# A wrapped position is below limit exactly when the position is in
# PyTorch's range -limit..limit-1.
law signed_wrap_inside:
  for +limit: U32
  for +position: Signed.Position
  {U32.is_lt(Signed.wrap(limit, position), limit) == Signed.inside(limit, position) : Bool}

# So a list of wrapped axes passes the operations' range check
# (Layout.bounded) exactly when every signed axis is in range.
law signed_axes_bounded:
  for +limit: U32
  for +positions: List<&2, Signed.Position>
  {Layout.bounded(Signed.wrap_all(limit, positions), limit) == Signed.all_inside(limit, positions) : Bool}

# Axes are wrapped one by one, in order, among the same limit.
law signed_wrap_all_empty:
  for +limit: U32
  {Signed.wrap_all(limit, []) == [] : List<&2, U32>}

law signed_wrap_all_step:
  for +limit: U32
  for +head: Signed.Position
  for +tail: List<&2, Signed.Position>
  {Signed.wrap_all(limit, head <> tail) == Signed.wrap(limit, head) <> Signed.wrap_all(limit, tail) : List<&2, U32>}

# Coordinates are wrapped one by one, each among its own axis's extent; a
# coordinate past the rank among 0.
law signed_wrap_each_step:
  for +extent: U32
  for +rest: List<&2, U32>
  for +head: Signed.Position
  for +tail: List<&2, Signed.Position>
  {Signed.wrap_each(extent <> rest, head <> tail) == Signed.wrap(extent, head) <> Signed.wrap_each(rest, tail) : List<&2, U32>}

law signed_wrap_each_end:
  for +dimensions: List<&2, U32>
  {Signed.wrap_each(dimensions, []) == [] : List<&2, U32>}

law signed_wrap_each_past_rank:
  for +positions: List<&2, Signed.Position>
  {Signed.wrap_each([], positions) == Signed.wrap_all(0, positions) : List<&2, U32>}

# Transpose wraps among the rank, and among 1 for a rank-0 input.
law signed_axis_limit:
  for +input: Shape.Shape
  {Signed.axis_limit(input) == U32.from_nat(Transpose.axis_limit(Shape.rank_dimensions(Shape.dimensions(input)))) : U32}

# Each signed operation is the unsigned one on the wrapped arguments.
law signed_permute_wraps:
  for +input: Shape.Shape
  for source: Core.Tensor<input>
  for +axes: List<&2, Signed.Position>
  {Signed.permute_copy(input, source, axes) == Transforms.permute_copy(input, source, Signed.wrap_all(Shape.rank(input), axes)) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}

law signed_permute_dynamic_wraps:
  for input: Core.Dynamic
  for +axes: List<&2, Signed.Position>
  {Signed.permute_dynamic(input, axes) == Transforms.permute_dynamic(input, Signed.wrap_all(Shape.rank(Totality.dynamic_shape(input)), axes)) :
    Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>}

law signed_transpose_wraps:
  for +input: Shape.Shape
  for source: Core.Tensor<input>
  for +a: Signed.Position
  for +b: Signed.Position
  {Signed.transpose_copy(input, source, a, b) ==
    Transforms.transpose_copy(input, source, Signed.wrap(Signed.axis_limit(input), a), Signed.wrap(Signed.axis_limit(input), b)) :
    Core.Tensor<input> & Result<&2, &1, Transforms.Error, Core.Dynamic>}

law signed_transpose_dynamic_wraps:
  for input: Core.Dynamic
  for +a: Signed.Position
  for +b: Signed.Position
  {Signed.transpose_dynamic(input, a, b) ==
    Transforms.transpose_dynamic(input, Signed.wrap(Signed.axis_limit(Totality.dynamic_shape(input)), a),
      Signed.wrap(Signed.axis_limit(Totality.dynamic_shape(input)), b)) :
    Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>}

law signed_read_coordinate_wraps:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +coordinates: List<&2, Signed.Position>
  {Signed.read_coordinate(shape, tensor, coordinates) == Core.read_coordinate(shape, tensor, Signed.wrap_each(Shape.dimensions(shape), coordinates)) :
    Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}

law signed_write_coordinate_wraps:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +coordinates: List<&2, Signed.Position>
  for value: F32
  {Signed.write_coordinate(shape, tensor, coordinates, value) ==
    Core.write_coordinate(shape, tensor, Signed.wrap_each(Shape.dimensions(shape), coordinates), value) :
    Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>}

law signed_read_flat_wraps:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +index: Signed.Position
  {Signed.read_flat(shape, tensor, index) == Core.read_flat(shape, tensor, Signed.wrap(Shape.count(shape), index)) :
    Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>}

law signed_write_flat_wraps:
  for +shape: Shape.Shape
  for tensor: Core.Tensor<shape>
  for +index: Signed.Position
  for +value: F32
  {Signed.write_flat(shape, tensor, index, value) == Core.write_flat(shape, tensor, Signed.wrap(Shape.count(shape), index), value) :
    Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>}

law signed_read_dynamic_wraps:
  for input: Core.Dynamic
  for +index: Signed.Position
  {Signed.read_dynamic(input, index) == Core.read_dynamic(input, Signed.wrap(Shape.count(Totality.dynamic_shape(input)), index)) :
    Core.Dynamic & Result<&2, &2, Core.Error, F32>}

law signed_write_dynamic_wraps:
  for input: Core.Dynamic
  for +index: Signed.Position
  for value: F32
  {Signed.write_dynamic(input, index, value) == Core.write_dynamic(input, Signed.wrap(Shape.count(Totality.dynamic_shape(input)), index), value) :
    Core.Dynamic & Result<&2, &2, Core.Error, Unit>}

# A reshape infers its sizes from the input's element count, then is the
# unsigned reshape.
law signed_reshape_infers:
  for +input: Shape.Shape
  for tensor: Core.Tensor<input>
  for +extents: List<&2, Signed.Extent>
  {Signed.reshape(input, tensor, extents) ==
    Signed.reshape_inferred(input, tensor, Signed.inferred_dimensions(Shape.elements(input), extents)) :
    Result<&2, &1, Signed.Error, Core.Dynamic>}

law signed_reshape_dynamic_infers:
  for +input: Shape.Shape
  for tensor: Core.Tensor<input>
  for +extents: List<&2, Signed.Extent>
  {Signed.reshape_dynamic(Core.Packed{input, tensor}, extents) == Signed.reshape(input, tensor, extents) :
    Result<&2, &1, Signed.Error, Core.Dynamic>}

# Exact sizes reshape exactly as the unsigned reshape does, its errors as
# TransformError.
law signed_reshape_unsigned:
  for +input: Shape.Shape
  for tensor: Core.Tensor<input>
  for +dimensions: List<&2, U32>
  {Signed.reshape(input, tensor, exacts(dimensions)) == Signed.transformed(Transforms.reshape(input, tensor, dimensions)) :
    Result<&2, &1, Signed.Error, Core.Dynamic>}

# With no Inferred size, the sizes are the request's.
law signed_inferred_none:
  for +count: Nat
  for +extents: List<&2, Signed.Extent>
  for none: {Signed.inferred_count(extents) == 0n : Nat}
  {Signed.inferred_dimensions(count, extents) == Done{Signed.filled(extents, 0)} : Result<&2, &2, Signed.Error, List<&2, U32>>}

# With one Inferred size and the other sizes multiplying to a product p > 0,
# the Inferred size is the q with q * p = count.
law signed_inferred_size:
  for +count: Nat
  for +extents: List<&2, Signed.Extent>
  for one: {Signed.inferred_count(extents) == 1n : Nat}
  for +rest: Nat
  for product: {Signed.exact_product(extents) == 1n+rest : Nat}
  for +quotient: Nat
  for exact: {Nat.mul(quotient, 1n+rest) == count : Nat}
  {Signed.inferred_dimensions(count, extents) == Done{Signed.filled(extents, U32.from_nat(quotient))} :
    Result<&2, &2, Signed.Error, List<&2, U32>>}

# And then the filled sizes hold q * p elements.
law signed_filled_count:
  for +extents: List<&2, Signed.Extent>
  for one: {Signed.inferred_count(extents) == 1n : Nat}
  for +size: U32
  {Shape.logical_numel(Signed.filled(extents, size)) == Nat.mul(U32.to_nat(size), Signed.exact_product(extents)) : Nat}

# p > 0 that does not divide count: ElementCountMismatch, as PyTorch's
# "shape is invalid for input of size".
law signed_inferred_indivisible:
  for +count: Nat
  for +extents: List<&2, Signed.Extent>
  for one: {Signed.inferred_count(extents) == 1n : Nat}
  for +rest: Nat
  for product: {Signed.exact_product(extents) == 1n+rest : Nat}
  for +quotient: Nat
  for +remainder: Nat
  for split: {Nat.divmod(count, 1n+rest) == (quotient, 1n+remainder) : Nat & Nat}
  {Signed.inferred_dimensions(count, extents) == Fail{Signed.TransformError{Transforms.LayoutError{Layout.ElementCountMismatch{}}}} :
    Result<&2, &2, Signed.Error, List<&2, U32>>}

# p = 0: ambiguous on an empty input, ElementCountMismatch on any other.
law signed_inferred_ambiguous:
  for +extents: List<&2, Signed.Extent>
  for one: {Signed.inferred_count(extents) == 1n : Nat}
  for product: {Signed.exact_product(extents) == 0n : Nat}
  {Signed.inferred_dimensions(0n, extents) == Fail{Signed.AmbiguousInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}

law signed_inferred_zero_product:
  for +count: Nat
  for +extents: List<&2, Signed.Extent>
  for one: {Signed.inferred_count(extents) == 1n : Nat}
  for product: {Signed.exact_product(extents) == 0n : Nat}
  {Signed.inferred_dimensions(1n+count, extents) == Fail{Signed.TransformError{Transforms.LayoutError{Layout.ElementCountMismatch{}}}} :
    Result<&2, &2, Signed.Error, List<&2, U32>>}

# The class of a product against a bound, in Signed.Product, stated from
# its definition and not through signed.bend's helpers: 0 is ZeroProduct, a
# nonzero product at most the bound is Within{product}, and one past it is
# Beyond.
def classified(product: Nat, below: Bool) -> Signed.Product:
  match below:
    case True{}:
      Signed.Within{product}
    case False{}:
      Signed.Beyond{}

def classify(+bound: Nat, product: Nat) -> Signed.Product:
  match product:
    case 0n:
      Signed.ZeroProduct{}
    case 1n+(+rest):
      classified(1n+rest, Nat.is_le(1n+rest, bound))

# The inference never forms the exact sizes' product: capped classifies it
# against the input's element count, ZeroProduct, Within{p} with p at most
# the count, or Beyond, multiplying only numbers at most the count. So the
# inferred size is worked out from numbers at most 2^20, however large the
# exact sizes, as [-1, 2^24, 2^24] of an empty input, whose product 2^48 is
# past the native Nat limit.
law signed_capped_classifies:
  for +bound: Nat
  for +extents: List<&2, Signed.Extent>
  {Signed.capped(bound, extents) == classify(bound, Signed.exact_product(extents)) : Signed.Product}

law signed_capped_within:
  for +bound: Nat
  for +extents: List<&2, Signed.Extent>
  for +product: Nat
  for within: {Signed.capped(bound, extents) == Signed.Within{product} : Signed.Product}
  {Nat.is_le(product, bound) == True{} : Bool}

# More than one Inferred size is refused before anything else.
law signed_inferred_repeated:
  for +count: Nat
  for +extents: List<&2, Signed.Extent>
  for +more: Nat
  for repeated: {Signed.inferred_count(extents) == 2n+more : Nat}
  {Signed.inferred_dimensions(count, extents) == Fail{Signed.RepeatedInferred{}} : Result<&2, &2, Signed.Error, List<&2, U32>>}

# Filling puts the size at each Inferred place and keeps each exact size.
law signed_filled_exact:
  for +extent: U32
  for +tail: List<&2, Signed.Extent>
  for +size: U32
  {Signed.filled(Signed.Exact{extent} <> tail, size) == extent <> Signed.filled(tail, size) : List<&2, U32>}

law signed_filled_inferred:
  for +tail: List<&2, Signed.Extent>
  for +size: U32
  {Signed.filled(Signed.Inferred{} <> tail, size) == size <> Signed.filled(tail, size) : List<&2, U32>}

# The new error codes: RepeatedInferred 17, AmbiguousInferred 18; the
# unsigned reshape's errors keep their codes.
law signed_error_codes:
  for error: Transforms.Error
  {(Signed.error_code(Signed.TransformError{error}), Signed.error_code(Signed.RepeatedInferred{}), Signed.error_code(Signed.AmbiguousInferred{})) ==
    (Transforms.error_code(error), 17, 18) : U32 & U32 & U32}

# Signed slicing (Signed.slice, narrow and select). The axis is wrapped among
# the input's rank, as every other signed axis; the positions along it among
# its extent, each as PyTorch takes it.
def axis_of(input: Core.Dynamic, dim: Signed.Position) -> U32:
  Signed.wrap(Shape.rank(Totality.dynamic_shape(input)), dim)

def extent_of(input: Core.Dynamic, +axis: U32) -> U32:
  U32.from_nat(Slice.extent_at(Shape.dimensions(Totality.dynamic_shape(input)), axis))

def rank_of(input: Core.Dynamic) -> Nat:
  Shape.rank_dimensions(Shape.dimensions(Totality.dynamic_shape(input)))

# A slice's start and stop, PyTorch's get_slice_bounds before its clamp to
# the axis: a position counted from the start is kept; Last{k}, the position
# d = -1 - k, becomes d + limit in range, and 0 below -limit.
law signed_bound_first:
  for +limit: U32
  for +k: U32
  {Signed.bound(limit, Signed.First{k}) == k : U32}

law signed_bound_last_inside:
  for +limit: U32
  for +k: U32
  for inside: {U32.is_lt(k, limit) == True{} : Bool}
  {Nat.add(U32.to_nat(Signed.bound(limit, Signed.Last{k})), 1n+U32.to_nat(k)) == U32.to_nat(limit) : Nat}

law signed_bound_last_outside:
  for +limit: U32
  for +k: U32
  for outside: {U32.is_lt(k, limit) == False{} : Bool}
  {Signed.bound(limit, Signed.Last{k}) == 0 : U32}

# A signed slice is the unsigned slice at the wrapped axis of the bounds of
# its start and stop among that axis's extent; slice-LAWS.bend's clamp then
# gives PyTorch's.
law signed_slice_bounds:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for start: Signed.Position
  for stop: Signed.Position
  for +step: U32
  {Signed.slice(input, dim, start, stop, step) ==
    Slice.slice(input, axis_of(input, dim), Signed.bound(extent_of(input, axis_of(input, dim)), start),
      Signed.bound(extent_of(input, axis_of(input, dim)), stop), step) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

# A signed select is the unsigned select at the wrapped axis of the index
# wrapped among that axis's extent: an index outside -extent..extent-1 stays
# at or past the extent, which the unsigned select refuses.
law signed_select_wraps:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for index: Signed.Position
  {Signed.select(input, dim, index) ==
    Slice.select(input, axis_of(input, dim), Signed.wrap(extent_of(input, axis_of(input, dim)), index)) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

# A signed narrow takes a start in -extent..extent, PyTorch's range. A start
# counted from the start is the unsigned narrow's, which refuses one past
# the extent itself; Last{k} below the extent is wrapped among it ...
law signed_narrow_first:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +k: U32
  for +length: U32
  {Signed.narrow(input, dim, Signed.First{k}, length) == Slice.narrow(input, axis_of(input, dim), k, length) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law signed_narrow_last_inside:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +k: U32
  for +length: U32
  for inside: {U32.is_lt(k, extent_of(input, axis_of(input, dim))) == True{} : Bool}
  {Signed.narrow(input, dim, Signed.Last{k}, length) ==
    Slice.narrow(input, axis_of(input, dim), Signed.wrap(extent_of(input, axis_of(input, dim)), Signed.Last{k}), length) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

# ... and a start before -extent is refused: OutOfRange once the axis is in
# range, and AxisOutOfBounds first, as the unsigned narrow checks it.
law signed_narrow_before:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +k: U32
  for +length: U32
  for in_rank: {Nat.is_lt(U32.to_nat(axis_of(input, dim)), rank_of(input)) == True{} : Bool}
  for before: {U32.is_lt(k, extent_of(input, axis_of(input, dim))) == False{} : Bool}
  {Signed.narrow(input, dim, Signed.Last{k}, length) == (input, Fail{Slice.OutOfRange{}}) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}

law signed_narrow_before_axis:
  for input: Core.Dynamic
  for +dim: Signed.Position
  for +k: U32
  for +length: U32
  for past_rank: {Nat.is_lt(U32.to_nat(axis_of(input, dim)), rank_of(input)) == False{} : Bool}
  for before: {U32.is_lt(k, extent_of(input, axis_of(input, dim))) == False{} : Bool}
  {Signed.narrow(input, dim, Signed.Last{k}, length) == (input, Fail{Slice.LayoutError{Layout.AxisOutOfBounds{}}}) :
    Core.Dynamic & Result<&2, &1, Slice.Error, Core.Dynamic>}
