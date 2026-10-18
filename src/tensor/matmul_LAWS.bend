import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./matmul.bend as Matrix
import ./math_totality.bend as Totality
import ./dot_fold.bend as DotFold

def coordinate_observe_result(+shape: Shape.Shape, tensor: Core.Tensor<shape>,
  result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{index}:
      Core.checked_observe(shape, tensor, Shape.index_value(shape, index))

def coordinate_observe(+shape: Shape.Shape, tensor: Core.Tensor<shape>, coordinates: List<&2, U32>) -> Maybe<&2, F32>:
  coordinate_observe_result(shape, tensor, Shape.flatten_checked(shape, coordinates))

def LogicalDot(+fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  +row: U32, +column: U32, +cursor: Nat, values: List<&2, DotFold.Factors>) -> Data:
  match fuel values:
    case 0n Nil{}:
      Unit
    case 1n+tail DotFold.Factors{lv, rv} <> rest:
      DotStep<tail, left, right, a, b, row, column, cursor, lv, rv, rest>
    case 0n DotFold.Factors{lv, rv} <> rest:
      Empty
    case 1n+tail Nil{}:
      Empty

type DotStep<-fuel: Nat, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -row: U32, -column: U32, -cursor: Nat, -lv: F32, -rv: F32, -values: List<&2, DotFold.Factors>> is Data:
  DotStep{-exact: {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat},
    -left_observed: {coordinate_observe(left, a, [row, U32.from_nat(cursor)]) == Some{lv} : Maybe<&2, F32>},
    -right_observed: {coordinate_observe(right, b, [U32.from_nat(cursor), column]) == Some{rv} : Maybe<&2, F32>},
    -remaining: LogicalDot(fuel, left, right, a, b, row, column, 1n+cursor, values)}

type DotMeaning<-fuel: Nat, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -row: U32, -column: U32, -cursor: Nat, -accumulator: F32, -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -value: F32> is Data:
  DotMeaning{-values: List<&2, DotFold.Factors>, -logical: LogicalDot(fuel, left, right, a, b, row, column, cursor, values),
    -fold: {value == DotFold.dot_fold(values, accumulator) : F32},
    -left_kept: @+index: U32 -> {Core.checked_observe(left, ra, index) == Core.checked_observe(left, a, index) : Maybe<&2, F32>},
    -right_kept: @+index: U32 -> {Core.checked_observe(right, rb, index) == Core.checked_observe(right, b, index) : Maybe<&2, F32>}}

def DotResultMeaning(-fuel: Nat, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -row: U32, -column: U32, -cursor: Nat, -accumulator: F32,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (ra, rb, value) = pair
      DotMeaning<fuel, left, right, a, b, row, column, cursor, accumulator, ra, rb, value>

type SourceDot<-fuel: Nat, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -row: U32, -column: U32, -cursor: Nat, -accumulator: F32,
  -result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>> is Data:
  SourceDot{-meaning: @unit: Unit -> DotResultMeaning(fuel, left, right, a, b, row, column, cursor, accumulator, result)}

type ProductCell<-inner: U32, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -output: Shape.Shape, -out: Core.Tensor<output>, -position: Nat> is Data:
  ProductCell{-row: U32, -column: U32, -coordinate: Shape.CheckedCoordinate<output>,
    -decoded: {Shape.unflatten_checked(output, U32.from_nat(position)) == Done{coordinate} : Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>},
    -coordinates: {Shape.coordinate_values(output, coordinate) == [row, column] : List<&2, U32>},
    -word_exact: {U32.to_nat(U32.from_nat(position)) == position : Nat},
    -factors: List<&2, DotFold.Factors>, -logical: LogicalDot(U32.to_nat(inner), left, right, a, b, row, column, 0n, factors),
    -value: F32, -fold: {value == DotFold.dot_fold(factors, 0.0) : F32},
    -checked: Core.ObservationChecked(Core.Packed{output, out}, U32.from_nat(position)),
    -observed: {Core.checked_observe(output, out, U32.from_nat(position)) == Some{value} : Maybe<&2, F32>}}

type ProductPrefix<-inner: U32, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -output: Shape.Shape, -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -out: Core.Tensor<output>, -cursor: Nat> is Data:
  ProductPrefix{-left_kept: @+other: U32 -> {Core.checked_observe(left, ra, other) == Core.checked_observe(left, a, other) : Maybe<&2, F32>},
    -right_kept: @+other: U32 -> {Core.checked_observe(right, rb, other) == Core.checked_observe(right, b, other) : Maybe<&2, F32>},
    -cells: @+position: Nat -> @earlier: {Nat.is_lt(position, cursor) == True{} : Bool} -> ProductCell<inner, left, right, a, b, output, out, position>}

def ProductMeaning(-inner: U32, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -output: Shape.Shape, -end: Nat, result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (ra, rb, out) = pair
      ProductPrefix<inner, left, right, a, b, output, ra, rb, out, end>

type SourceProduct<-inner: U32, -left: Shape.Shape, -right: Shape.Shape, -a: Core.Tensor<left>, -b: Core.Tensor<right>,
  -output: Shape.Shape, -end: Nat, -result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>> is Data:
  SourceProduct{-meaning: @unit: Unit -> ProductMeaning(inner, left, right, a, b, output, end, result)}

law successful_product_interpretation:
  for +fuel: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +output: Shape.Shape
  for +inner: U32
  for +columns: U32
  for -a: Core.Tensor<left>
  for -b: Core.Tensor<right>
  for ra: Core.Tensor<left>
  for rb: Core.Tensor<right>
  for out: Core.Tensor<output>
  for +cursor: Nat
  for prefix: ProductPrefix<inner, left, right, a, b, output, ra, rb, out, cursor>
  SourceProduct<inner, left, right, a, b, output, Nat.add(fuel, cursor),
    Matrix.natural_product_loop(fuel, left, right, output, inner, columns, ra, rb, out, cursor)>

def plan_inner(-left: Shape.Shape, -right: Shape.Shape, plan: Matrix.Plan<left, right>) -> U32:
  Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  inner

law successful_multiply_interpretation:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +plan: Matrix.Plan<left, right>
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  SourceProduct<plan_inner(left, right, plan), left, right, a, b, Matrix.output_shape(left, right, plan),
    Shape.logical_numel(Shape.dimensions(Matrix.output_shape(left, right, plan))), Matrix.multiply(left, right, plan, a, b)>

type MatmulSuccess<-original_a: Core.Dynamic, -original_b: Core.Dynamic, -retained_a: Core.Dynamic, -retained_b: Core.Dynamic, -product: Core.Dynamic> is Data:
  MatmulSuccess{-left: Shape.Shape, -right: Shape.Shape, -plan: Matrix.Plan<left, right>,
    -a: Core.Tensor<left>, -b: Core.Tensor<right>, -ra: Core.Tensor<left>, -rb: Core.Tensor<right>, -out: Core.Tensor<Matrix.output_shape(left, right, plan)>,
    -input_left: {original_a == Core.Packed{left, a} : Core.Dynamic}, -input_right: {original_b == Core.Packed{right, b} : Core.Dynamic},
    -output_left: {retained_a == Core.Packed{left, ra} : Core.Dynamic}, -output_right: {retained_b == Core.Packed{right, rb} : Core.Dynamic},
    -output: {product == Core.Packed{Matrix.output_shape(left, right, plan), out} : Core.Dynamic},
    -source: ProductPrefix<plan_inner(left, right, plan), left, right, a, b, Matrix.output_shape(left, right, plan), ra, rb, out,
      Shape.logical_numel(Shape.dimensions(Matrix.output_shape(left, right, plan)))>}

def MatmulMeaning(-a: Core.Dynamic, -b: Core.Dynamic, result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (ra, rb, product) = pair
      MatmulSuccess<a, b, ra, rb, product>

type SourceMatmul<-a: Core.Dynamic, -b: Core.Dynamic, -result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>> is Data:
  SourceMatmul{-meaning: @unit: Unit -> MatmulMeaning(a, b, result)}

law successful_matmul_interpretation:
  for left: Core.Dynamic
  for right: Core.Dynamic
  SourceMatmul<left, right, Matrix.matmul(left, right)>

type ZeroPrefix<-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -out: Core.Tensor<output>, -end: Nat> is Data:
  ZeroPrefix{-observed: @+position: Nat -> @earlier: {Nat.is_lt(position, end) == True{} : Bool} ->
    {Core.checked_observe(output, out, U32.from_nat(position)) == Some{0.0} : Maybe<&2, F32>}}

def ZeroMeaning(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -end: Nat,
  result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>) -> Data:
  match result:
    case Fail{error}:
      Unit
    case Done{pair}:
      (a, b, out) = pair
      ZeroPrefix<left, right, output, out, end>

type ZeroResult<-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape, -end: Nat,
  -result: Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>> is Data:
  ZeroResult{-meaning: @unit: Unit -> ZeroMeaning(left, right, output, end, result)}

law zero_inner_multiply:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for +output: Shape.Shape
  for +rows: U32
  for +right_inner: U32
  for +columns: U32
  for +le: {Shape.dimensions(left) == [rows, 0] : List<&2, U32>}
  for +re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}
  for +oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}
  for +compatible: {0n == U32.to_nat(right_inner) : Nat}
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  ZeroResult<left, right, output, Shape.logical_numel(Shape.dimensions(output)),
    Matrix.multiply(left, right, Matrix.Plan{output, rows, 0, right_inner, columns, le, re, oe, compatible}, a, b)>

law successful_dot_interpretation:
  for +fuel: Nat
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for +row: U32
  for +column: U32
  for +cursor: Nat
  for +accumulator: F32
  SourceDot<fuel, left, right, a, b, row, column, cursor, accumulator,
    Matrix.natural_dot(fuel, left, right, a, b, row, column, cursor, accumulator)>

law zero_inner_dot:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for +row: U32
  for +column: U32
  {Matrix.natural_dot(0n, left, right, a, b, row, column, 0n, 0.0) == Done{(a, b, 0.0)} :
    Result<&2, &1, Matrix.Error, Core.Tensor<left> & Core.Tensor<right> & F32>}

law output_dimensions:
  for -left: Shape.Shape
  for -right: Shape.Shape
  for +output: Shape.Shape
  for rows: U32
  for inner: U32
  for right_inner: U32
  for columns: U32
  for le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}
  for re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}
  for oe: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>}
  for compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}
  {Shape.dimensions(Matrix.output_shape(left, right, Matrix.Plan{output, rows, inner, right_inner, columns, le, re, oe, compatible})) ==
    [rows, columns] : List<&2, U32>}

type TotalMatmul<-left: Core.Dynamic, -right: Core.Dynamic,
  -result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>> is Data:
  TotalMatmul{-done: Totality.Success<Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic, result>,
    -interpretation: SourceMatmul<left, right, result>}

law matmul_valid_success:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for valid: Totality.MatrixInputs<left, right>
  TotalMatmul<Core.Packed{left, a}, Core.Packed{right, b}, Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b})>

# A left operand that is not rank 2 fails with RankMismatch, whatever the
# right operand is: there is no 1-D promotion and no batching.
law matmul_left_rank_error:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for not_matrix: {Shape.rank_dimensions(Shape.dimensions(left)) != 2n : Nat}
  {Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.RankMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# A right operand that is not rank 2 fails with RankMismatch, whatever the
# left operand is.
law matmul_right_rank_error:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for not_matrix: {Shape.rank_dimensions(Shape.dimensions(right)) != 2n : Nat}
  {Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.RankMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}

# Two rank-2 operands whose inner extents differ fail with
# InnerDimensionMismatch, never with a rank or capacity error: [2, 3] @ [4, 2]
# is an inner mismatch.
law matmul_inner_error:
  for +left: Shape.Shape
  for +right: Shape.Shape
  for a: Core.Tensor<left>
  for b: Core.Tensor<right>
  for +rows: U32
  for +inner: U32
  for +right_inner: U32
  for +columns: U32
  for left_matrix: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}
  for right_matrix: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>}
  for mismatch: {U32.to_nat(inner) != U32.to_nat(right_inner) : Nat}
  {Matrix.matmul(Core.Packed{left, a}, Core.Packed{right, b}) == Fail{Matrix.InnerDimensionMismatch{}} : Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>}
