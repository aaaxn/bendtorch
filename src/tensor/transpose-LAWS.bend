import Base
import ../shape/core.bend as Shape
import ./axis-spec.bend as Axis
import ./layout.bend as Layout
import ./transpose-spec.bend as Transpose

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
