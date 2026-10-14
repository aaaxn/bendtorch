import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./matmul.bend as Matrix
import ./reduction.bend as Reduction
import ./transforms.bend as Transforms

type Unary is Data:
  Construct{}
  Clone{}
  Sum{}
  Reshape{dimensions: List<&2, U32>}
  Permute{axes: List<&2, U32>}
  Transpose{a: U32, b: U32}
  Get{coordinate: List<&2, U32>}
  GetFlat{index: U32}

type Binary is Data:
  Add{}
  Sub{}
  Mul{}
  Generic{operation: Elementwise.Operation}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def zero_shape(-dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) ->
  Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Shape.SError{error}:
      Fail{Core.ShapeError{error}}
    case Shape.SBuilt{+shape, preserved}:
      Core.construct_values(shape, Core.new(shape, from_bits(0)))

def zeroes(+dimensions: List<&2, U32>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  zero_shape(dimensions, Shape.construct(dimensions))

def core_error(error: Core.Error) -> IO(Unit):
  IO.print("tensor:error:core:" ++ U32.show(Core.error_code(error)))

def word_step(rec: Core.Dynamic -> IO(Core.Dynamic), pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Core.Dynamic):
  (tensor, result) = pair
  match result:
    case Fail{error}:
      do IO<Core.Dynamic>:
        core_error(error)
        return tensor
    case Done{value}:
      do IO<Core.Dynamic>:
        IO.print("tensor:word:" ++ U32.show(F32.bits(value)))
        rec(tensor)

def words(fuel: Nat, +index: U32, tensor: Core.Dynamic) -> IO(Core.Dynamic):
  match fuel:
    case 0n:
      IO.pure(Core.Dynamic, tensor)
    case 1n+tail:
      word_step(tensor => words(tail, U32.inc(index), tensor), Core.read_dynamic(tensor, index))

def snapshot_shape(pair: Core.Dynamic & Shape.Shape) -> IO(Unit):
  (tensor, +shape) = pair
  do IO<Unit>:
    IO.print("tensor:shape:" ++ List.show(~&2, ~U32, ~U32.show, Shape.dimensions(shape)))
    IO.print("tensor:count:" ++ U32.show(Shape.count(shape)))
    tensor : Core.Dynamic <- words(U32.to_nat(Shape.count(shape)), 0, tensor)
    return Unit{}

def snapshot(tensor: Core.Dynamic) -> IO(Unit):
  snapshot_shape(Core.shape(tensor))

def transform_result(result: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("tensor:error:transform:" ++ U32.show(Transforms.error_code(error)))
    case Done{tensor}:
      snapshot(tensor)

def core_result(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      core_error(error)
    case Done{tensor}:
      snapshot(tensor)

def scalar_result(result: Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  match result:
    case Fail{error}:
      core_error(error)
    case Done{value}:
      do IO<Unit>:
        IO.print("tensor:shape:[]")
        IO.print("tensor:count:1")
        IO.print("tensor:word:" ++ U32.show(F32.bits(value)))

def coordinate_result(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  (value, result) = pair
  scalar_result(result)

def coordinate(tensor: Core.Dynamic, coordinate: List<&2, U32>) -> IO(Unit):
  Core.Packed{+shape, value} = tensor
  coordinate_result(shape, Core.read_coordinate(shape, value, coordinate))

def flat_result(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  (input, result) = pair
  scalar_result(result)

def clone_result(pair: Core.Dynamic & Core.Dynamic) -> IO(Unit):
  (input, copied) = pair
  snapshot(copied)

def retained_transform(pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (input, result) = pair
  transform_result(result)

def binary_result(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  (inputs, output) = pair
  core_result(output)

def retained_binary_result(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  (inputs, output) = pair
  (left, right) = inputs
  do IO<Unit>:
    IO.print("tensor:binary:called")
    IO.print("tensor:retained:left")
    snapshot(left)
    IO.print("tensor:retained:right")
    snapshot(right)
    IO.print("tensor:output")
    core_result(output)

def sum_result(result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      core_error(error)
    case Done{pair}:
      (input, scalar) = pair
      snapshot(scalar)

def unary_value(operation: Unary, tensor: Core.Dynamic) -> IO(Unit):
  match operation:
    case Construct{}:
      snapshot(tensor)
    case Clone{}:
      clone_result(Core.clone_dynamic(tensor))
    case Sum{}:
      sum_result(Reduction.sum_dynamic(tensor))
    case Reshape{dimensions}:
      transform_result(Transforms.reshape_dynamic(tensor, dimensions))
    case Permute{axes}:
      retained_transform(Transforms.permute_dynamic(tensor, axes))
    case Transpose{a, b}:
      retained_transform(Transforms.transpose_dynamic(tensor, a, b))
    case Get{coordinates}:
      coordinate(tensor, coordinates)
    case GetFlat{index}:
      flat_result(Core.read_dynamic(tensor, index))

def unary(operation: Unary, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      core_error(error)
    case Done{tensor}:
      unary_value(operation, tensor)

def binary_value(operation: Binary, left: Core.Dynamic, right: Core.Dynamic) -> IO(Unit):
  match operation:
    case Add{}:
      retained_binary_result(Elementwise.add(left, right))
    case Sub{}:
      retained_binary_result(Elementwise.sub(left, right))
    case Mul{}:
      retained_binary_result(Elementwise.mul(left, right))
    case Generic{operation}:
      retained_binary_result(Elementwise.binary(operation, left, right))

def construction_error(error: Core.Error) -> IO(Unit):
  do IO<Unit>:
    IO.print("tensor:boundary:construction")
    core_error(error)

def binary_right(operation: Binary, left: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      construction_error(error)
    case Done{right}:
      binary_value(operation, left, right)

def binary(operation: Binary, left: Result<&2, &1, Core.Error, Core.Dynamic>,
  right: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match left:
    case Fail{error}:
      construction_error(error)
    case Done{tensor}:
      binary_right(operation, tensor, right)

def matrix_error(error: Matrix.Error) -> IO(Unit):
  match error:
    case Matrix.TensorError{error}:
      core_error(error)
    case Matrix.ShapeError{error}:
      IO.print("tensor:error:shape:" ++ U32.show(Shape.error_code(error)))
    case Matrix.RankMismatch{}:
      IO.print("tensor:error:matrix:rank")
    case Matrix.InnerDimensionMismatch{}:
      IO.print("tensor:error:matrix:inner")

def matrix_output(bias: Maybe<&1, Core.Dynamic>, output: Core.Dynamic) -> IO(Unit):
  match bias:
    case None{}:
      snapshot(output)
    case Some{tensor}:
      binary_result(Elementwise.add(output, tensor))

def matrix_result(bias: Maybe<&1, Core.Dynamic>, result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      matrix_error(error)
    case Done{values}:
      (left, right, output) = values
      matrix_output(bias, output)

def matrix_right(bias: Maybe<&1, Core.Dynamic>, left: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      core_error(error)
    case Done{right}:
      matrix_result(bias, Matrix.matmul(left, right))

def matrix(bias: Maybe<&1, Core.Dynamic>, left: Result<&2, &1, Core.Error, Core.Dynamic>,
  right: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match left:
    case Fail{error}:
      core_error(error)
    case Done{tensor}:
      matrix_right(bias, tensor, right)

def biased(bias: Result<&2, &1, Core.Error, Core.Dynamic>, left: Result<&2, &1, Core.Error, Core.Dynamic>,
  right: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match bias:
    case Fail{error}:
      core_error(error)
    case Done{tensor}:
      matrix(Some{tensor}, left, right)

def shape_result(-dimensions: List<&2, U32>, result: Shape.ConstructResult<dimensions>) -> IO(Unit):
  match result:
    case Shape.SError{error}:
      IO.print("tensor:error:shape:" ++ U32.show(Shape.error_code(error)))
    case Shape.SBuilt{+shape, preserved}:
      do IO<Unit>:
        IO.print("tensor:shape:" ++ List.show(~&2, ~U32, ~U32.show, Shape.dimensions(shape)))
        IO.print("tensor:numel:" ++ U32.show(Shape.count(shape)))

def shape(+dimensions: List<&2, U32>) -> IO(Unit):
  shape_result(dimensions, Shape.construct(dimensions))
