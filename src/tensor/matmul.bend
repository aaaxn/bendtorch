import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle

type Error is Data:
  TensorError{error: Core.Error}
  ShapeError{error: Shape.Error}
  RankMismatch{}
  InnerDimensionMismatch{}

type Plan<-left: Shape.Shape, -right: Shape.Shape> is Data:
  Plan{output: Shape.Shape, rows: U32, inner: U32, right_inner: U32, columns: U32,
    -left_dimensions: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
    -right_dimensions: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
    -output_dimensions: {Shape.dimensions(output) == [rows, columns] : List<&2, U32>},
    -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat}}

def output_shape(-left: Shape.Shape, -right: Shape.Shape, plan: Plan<left, right>) -> Shape.Shape:
  Plan{shape, rows, inner, right_inner, columns, le, re, oe, compatible} = plan
  shape

def plan_output(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  -compatible: {U32.to_nat(inner) == U32.to_nat(right_inner) : Nat},
  result: Shape.ConstructResult<[rows, columns]>) -> Result<&2, &2, Error, Plan<left, right>>:
  match result:
    case Shape.SError{error}:
      Fail{ShapeError{error}}
    case Shape.SBuilt{shape, preserved}:
      Done{Plan{shape, rows, inner, right_inner, columns, le, re, preserved, compatible}}

def plan_compatible(-left: Shape.Shape, -right: Shape.Shape, +rows: U32, +inner: U32,
  +right_inner: U32, +columns: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>},
  -re: {Shape.dimensions(right) == [right_inner, columns] : List<&2, U32>},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))>) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{InnerDimensionMismatch{}}
    case Shape.BTrue{equal}:
      plan_output(left, right, rows, inner, right_inner, columns, le, re,
        Lifecycle.natural_equal(U32.to_nat(inner), U32.to_nat(right_inner), equal),
        Shape.construct([rows, columns]))

def plan_right(-left: Shape.Shape, +right: Shape.Shape, +rows: U32, +inner: U32,
  -le: {Shape.dimensions(left) == [rows, inner] : List<&2, U32>}, dimensions: List<&2, U32>,
  -same: {dimensions == Shape.dimensions(right) : List<&2, U32>}) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match dimensions:
    case +right_inner <> +columns <> Nil{}:
      plan_compatible(left, right, rows, inner, right_inner, columns, le,
        Equal.sym(List<&2, U32>, [right_inner, columns], Shape.dimensions(right), same),
        Shape.bool_case(Nat.is_eq(U32.to_nat(inner), U32.to_nat(right_inner))))
    case rest:
      Fail{RankMismatch{}}

def plan_left(+left: Shape.Shape, +right: Shape.Shape, dimensions: List<&2, U32>,
  -same: {dimensions == Shape.dimensions(left) : List<&2, U32>}) ->
  Result<&2, &2, Error, Plan<left, right>>:
  match dimensions:
    case +rows <> +inner <> Nil{}:
      plan_right(left, right, rows, inner,
        Equal.sym(List<&2, U32>, [rows, inner], Shape.dimensions(left), same),
        Shape.dimensions(right), {==})
    case rest:
      Fail{RankMismatch{}}

def plan(+left: Shape.Shape, +right: Shape.Shape) -> Result<&2, &2, Error, Plan<left, right>>:
  plan_left(left, right, Shape.dimensions(left), {==})

def right_read(-left: Shape.Shape, -right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  a: Core.Tensor<left>, left_value: F32, accumulator: F32,
  pair: Core.Tensor<right> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  (b, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{right_value}:
      rec(a, b, F32.add(accumulator, F32.mul(left_value, right_value)))

def left_read(-left: Shape.Shape, +right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  b: Core.Tensor<right>, coordinates: List<&2, U32>, accumulator: F32,
  pair: Core.Tensor<left> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  (a, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{value}:
      right_read(left, right, rec, a, value, accumulator, Core.read_coordinate(right, b, coordinates))

def dot_decided(+left: Shape.Shape, +right: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> F32 -> Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>,
  a: Core.Tensor<left>, b: Core.Tensor<right>, +row: U32, +column: U32, +cursor: Nat, +index: U32, accumulator: F32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{TensorError{Core.StorageError{Storage.InvalidIndex{}}}}
    case Shape.BTrue{equal}:
      left_read(left, right, rec, b, [index, column], accumulator, Core.read_coordinate(left, a, [row, index]))

def natural_dot(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  +row: U32, +column: U32, +cursor: Nat, accumulator: F32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  match fuel:
    case 0n:
      Done{(a, b, accumulator)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      dot_decided(left, right,
        a => b => value => natural_dot(tail, left, right, a, b, row, column, 1n+cursor, value),
        a, b, row, column, cursor, index, accumulator, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def dot(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  +row: U32, +column: U32, +inner_index: U32, accumulator: F32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>:
  natural_dot(fuel, left, right, a, b, row, column, U32.to_nat(inner_index), accumulator)

def wrote(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  a: Core.Tensor<left>, b: Core.Tensor<right>,
  pair: Core.Tensor<output> & Result<&2, &2, Core.Error, Unit>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  (out, result) = pair
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{unit}:
      rec(a, b, out)

def dotted(-left: Shape.Shape, -right: Shape.Shape, -output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  out: Core.Tensor<output>, index: U32,
  result: Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & F32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{values}:
      (a, b, value) = values
      wrote(left, right, output, rec, a, b, Core.write_flat(output, out, index, value))

def product_coordinates(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32, coordinates: List<&2, U32>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match coordinates:
    case +row <> +column <> Nil{}:
      dotted(left, right, output, rec, out, index, natural_dot(U32.to_nat(inner), left, right, a, b, row, column, 0n, 0.0))
    case rest:
      Fail{RankMismatch{}}

def product_unflattened(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32,
  result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<output>>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{ShapeError{error}}
    case Done{coordinate}:
      product_coordinates(left, right, output, rec, inner, a, b, out, index, Shape.coordinate_values(output, coordinate))

def product_decided(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  rec: Core.Tensor<left> -> Core.Tensor<right> -> Core.Tensor<output> ->
    Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +cursor: Nat, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{TensorError{Core.StorageError{Storage.InvalidIndex{}}}}
    case Shape.BTrue{equal}:
      product_unflattened(left, right, output, rec, inner, a, b, out, index, Shape.unflatten_checked(output, index))

def natural_product_loop(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +cursor: Nat) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match fuel:
    case 0n:
      Done{(a, b, out)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      product_decided(left, right, output,
        a => b => out => natural_product_loop(tail, left, right, output, inner, columns, a, b, out, 1n+cursor),
        inner, columns, a, b, out, cursor, index, Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def product_loop(fuel: Nat, +left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  +inner: U32, +columns: U32, a: Core.Tensor<left>, b: Core.Tensor<right>, out: Core.Tensor<output>, +index: U32) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  natural_product_loop(fuel, left, right, output, inner, columns, a, b, out, U32.to_nat(index))

def created(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape, +inner: U32, +columns: U32,
  a: Core.Tensor<left>, b: Core.Tensor<right>, result: Result<&2, &1, Core.Error, Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{out}:
      natural_product_loop(Shape.elements(output), left, right, output,
        inner, columns, a, b, out, 0n)

def multiply(+left: Shape.Shape, +right: Shape.Shape, +plan: Plan<left, right>,
  a: Core.Tensor<left>, b: Core.Tensor<right>) ->
  Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output_shape(left, right, plan)>>:
  match plan:
    case Plan{+output, rows, +inner, right_inner, +columns, le, re, oe, compatible}:
      created(left, right, output, inner, columns, a, b, Core.new(output, 0.0))

def packed(+left: Shape.Shape, +right: Shape.Shape, +output: Shape.Shape,
  result: Result<&2, &1, Error, Core.Tensor<left> & Core.Tensor<right> & Core.Tensor<output>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{values}:
      (a, b, out) = values
      Done{(Core.pack(left, a), Core.pack(right, b), Core.pack(output, out))}

def planned(+left: Shape.Shape, +right: Shape.Shape, a: Core.Tensor<left>, b: Core.Tensor<right>,
  result: Result<&2, &2, Error, Plan<left, right>>) ->
  Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{+plan}:
      packed(left, right, output_shape(left, right, plan), multiply(left, right, plan, a, b))

def matmul(left: Core.Dynamic, right: Core.Dynamic) -> Result<&2, &1, Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Core.Packed{+left_shape, a} = left
  Core.Packed{+right_shape, b} = right
  planned(left_shape, right_shape, a, b, plan(left_shape, right_shape))
