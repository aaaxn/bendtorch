import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./layout.bend as Layout
import ./transforms.bend as Transforms
import ./transpose-spec.bend as Transpose
import ./elementwise-totality.bend as Totality
import ./signed.bend as Signed

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
