import Base
import ../../shape/core.bend as Shape
import ../core.bend as Core
import ../matmul.bend as Matrix
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

def product_succeeded(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Bool:
  Result.is_done(&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, result)

def product_extents(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Maybe<&2, List<&2, Nat>>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (left, right, product) = values
      Core.Packed{shape, tensor} = product
      Some{Shape.extents(shape)}

def product_point(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (left, right, product) = values
      Core.dynamic_read_value(Core.read_dynamic(product, index))

def product_left_point(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (left, right, product) = values
      Core.dynamic_read_value(Core.read_dynamic(left, index))

def product_right_point(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{values}:
      (left, right, product) = values
      Core.dynamic_read_value(Core.read_dynamic(right, index))

# What a matrix product promises, stated only through the result: it
# succeeds with a rows x columns output whose cell (row, column), at index
# row * columns + column, holds `cell(row, column)`, and both returned inputs
# still hold every word.
type MatmulContract<-left: Shape.Shape, -right: Shape.Shape, -rows: Nat, -columns: Nat, -cell: Nat -> Nat -> F32,
  -left_word: U32 -> F32, -right_word: U32 -> F32,
  -result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>> is Type:
  MatmulContract{
    successful: {product_succeeded(result) == True{} : Bool},
    extents: {product_extents(result) == Some{[rows, columns]} : Maybe<&2, List<&2, Nat>>},
    value: @+row: Nat -> @row_below: {Nat.is_lt(row, rows) == True{} : Bool} ->
      @+column: Nat -> @column_below: {Nat.is_lt(column, columns) == True{} : Bool} ->
      {product_point(result, U32.from_nat(Nat.add(Nat.mul(row, columns), column))) == Some{cell(row, column)} : Maybe<&2, F32>},
    left_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool} ->
      {product_left_point(result, other) == Some{left_word(other)} : Maybe<&2, F32>},
    right_kept: @+other: U32 -> @inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool} ->
      {product_right_point(result, other) == Some{right_word(other)} : Maybe<&2, F32>}}
