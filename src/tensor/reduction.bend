import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ./core.bend as Core

def scalar_shape() -> Shape.Shape:
  Shape.SShape{Nil{}, 1n, {==}, Unit{}, Unit{}}

def sequential_sum(values: List<&2, F32>, accumulator: F32) -> F32:
  match values:
    case Nil{}:
      accumulator
    case value <> tail:
      sequential_sum(tail, F32.add(accumulator, value))

def read_step(-shape: Shape.Shape,
  rec: Core.Tensor<shape> -> F32 -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>,
  accumulator: F32, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      Fail{error}
    case Done{value}:
      rec(tensor, F32.add(accumulator, value))

def sum_decided(-shape: Shape.Shape,
  rec: Core.Tensor<shape> -> F32 -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>, tensor: Core.Tensor<shape>, +cursor: Nat,
  +index: U32, accumulator: F32, decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  match decision:
    case Shape.BFalse{equal}:
      Fail{Core.StorageError{Storage.InvalidIndex{}}}
    case Shape.BTrue{equal}:
      read_step(shape, rec, accumulator, Core.read_flat(shape, tensor, index))

def natural_sum_loop(fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +cursor: Nat,
  accumulator: F32) -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  match fuel:
    case 0n:
      Done{(tensor, accumulator)}
    case 1n+tail:
      +index = U32.from_nat(cursor)
      sum_decided(shape, tensor => value => natural_sum_loop(tail, shape, tensor, 1n+cursor, value), tensor, cursor, index, accumulator,
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), cursor)))

def sum_loop(fuel: Nat, -shape: Shape.Shape, tensor: Core.Tensor<shape>, +index: U32,
  accumulator: F32) -> Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>:
  natural_sum_loop(fuel, shape, tensor, U32.to_nat(index), accumulator)

def scalar_result(-input_shape: Shape.Shape, tensor: Core.Tensor<input_shape>,
  result: Result<&2, &1, Core.Error, Core.Tensor<scalar_shape()>>) ->
  Result<&2, &1, Core.Error, Core.Tensor<input_shape> & Core.Tensor<scalar_shape()>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{scalar}:
      Done{(tensor, scalar)}

def sum_result(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape> & F32>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      (tensor, value) = pair
      scalar_result(shape, tensor, Core.new(scalar_shape(), value))

def sum(+shape: Shape.Shape, tensor: Core.Tensor<shape>) ->
  Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>:
  sum_result(shape, natural_sum_loop(Shape.elements(shape), shape, tensor, 0n, 0.0))

def pack_result(+shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>) ->
  Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      (tensor, scalar) = pair
      Done{(Core.pack(shape, tensor), Core.pack(scalar_shape(), scalar))}

def sum_dynamic(dynamic: Core.Dynamic) -> Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>:
  Core.Packed{+shape, tensor} = dynamic
  pack_result(shape, sum(shape, tensor))

def sum_observation(-shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape> & Core.Tensor<scalar_shape()>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{pair}:
      (tensor, scalar) = pair
      Core.observe(scalar_shape(), scalar, 0)
