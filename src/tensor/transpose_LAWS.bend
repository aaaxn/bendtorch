import Base
import ../shape/core.bend as Shape
import ./axis_spec.bend as Axis
import ./layout.bend as Layout
import ./transpose_spec.bend as Transpose
import ./observation.bend as Values

law transpose_axes_requested:
  for +shape: Shape.Shape
  for +a: U32
  for +b: U32
  for +inside: {True{} == Transpose.valid(shape, a, b) : Bool}
  {Layout.transpose_axes(Shape.dimensions(shape), a, b, 0) ==
    Transpose.requested_axes(Shape.rank_dimensions(Shape.dimensions(shape)), U32.to_nat(a), U32.to_nat(b)) : List<&2, U32>}

law transpose_axes_valid:
  for +shape: Shape.Shape
  for +a: U32
  for +b: U32
  for +inside: {True{} == Transpose.valid(shape, a, b) : Bool}
  {True{} == Layout.bijective(shape, Layout.transpose_axes(Shape.dimensions(shape), a, b, 0)) : Bool}

law transpose_axes_self_inverse:
  for +shape: Shape.Shape
  for +a: U32
  for +b: U32
  for +inside: {True{} == Transpose.valid(shape, a, b) : Bool}
  {Axis.inverse_axes(Shape.rank_dimensions(Shape.dimensions(shape)), Shape.naturals(Layout.transpose_axes(Shape.dimensions(shape), a, b, 0))) ==
    Shape.naturals(Layout.transpose_axes(Shape.dimensions(shape), a, b, 0)) : List<&2, Nat>}

# The index map of a rank-2 transpose reads the transposed cell. For an input
# of dimensions [rows, columns], output cell [j, l] of the [columns, rows]
# transpose sits at j * rows + l, and `permute_source` with axes [1, 0] sends
# it to input index l * columns + j, the input cell [l, j].
law permute_source_transposed:
  for +input: Shape.Shape
  for +rows: U32
  for +columns: U32
  for matrix: {Shape.dimensions(input) == [rows, columns] : List<&2, U32>}
  for +j: Nat
  for j_below: {Nat.is_lt(j, U32.to_nat(columns)) == True{} : Bool}
  for +l: Nat
  for l_below: {Nat.is_lt(l, U32.to_nat(rows)) == True{} : Bool}
  {Values.permute_source(input, [1, 0], U32.from_nat(Nat.add(Nat.mul(j, U32.to_nat(rows)), l))) ==
    Some{U32.from_nat(Nat.add(Nat.mul(l, U32.to_nat(columns)), j))} : Maybe<&2, U32>}
