import Base
import ../../shape/core.bend as Shape
import ../core.bend as Core
import ./fold.bend as Fold

# What a fold result promises, stated only through the result: it succeeds
# with `total`, and the returned input still holds every word. `total` and
# `word` are the specification; an instance fills them in.
type FoldContract<-input: Shape.Shape, -total: F32, -word: U32 -> F32,
  -result: Core.Dynamic & Result<&2, &1, Core.Error, F32>> is Type:
  FoldContract{
    value: {Fold.value(result) == Some{total} : Maybe<&2, F32>},
    kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool} ->
      {Fold.input_point(result, other) == Some{word(other)} : Maybe<&2, F32>}}

# The same for a pair fold: the total, and every word of both returned inputs.
type PairFoldContract<-left: Shape.Shape, -right: Shape.Shape, -total: F32, -left_word: U32 -> F32, -right_word: U32 -> F32,
  -result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32>> is Type:
  PairFoldContract{
    value: {Fold.pair_value(result) == Some{total} : Maybe<&2, F32>},
    left_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool} ->
      {Fold.left_point(result, other) == Some{left_word(other)} : Maybe<&2, F32>},
    right_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool} ->
      {Fold.right_point(result, other) == Some{right_word(other)} : Maybe<&2, F32>}}
