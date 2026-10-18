import Base
import ../../shape/core.bend as Shape
import ../core.bend as Core
import ./pointwise.bend as Pointwise

# What a pointwise result promises, stated only through the result: success,
# the output dimensions, cell(i) at every in-bounds output index, and the
# words both returned inputs still hold. `cell`, `left_word` and `right_word`
# are the specification; an instance fills them in.
type PointwiseContract<-shape: Shape.Shape, -left: Shape.Shape, -right: Shape.Shape,
  -cell: Nat -> F32, -left_word: U32 -> F32, -right_word: U32 -> F32,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>> is Type:
  PointwiseContract{
    successful: {Pointwise.succeeded(result) == True{} : Bool},
    dimensions: {Pointwise.output_dimensions(result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool} ->
      {Pointwise.output_point(result, index) == Some{cell(U32.to_nat(index))} : Maybe<&2, F32>},
    left_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool} ->
      {Pointwise.left_point(result, other) == Some{left_word(other)} : Maybe<&2, F32>},
    right_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool} ->
      {Pointwise.right_point(result, other) == Some{right_word(other)} : Maybe<&2, F32>}}
