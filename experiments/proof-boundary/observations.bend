import Base
import ../../src/shape/core.bend as Shape
import ../../src/tensor/core.bend as Core

def dimensions_pair(pair: Core.Dynamic & Shape.Shape) -> Maybe<&2, List<&2, U32>>:
  (dynamic, shape) = pair
  Some{Shape.dimensions(shape)}

def count_pair(pair: Core.Dynamic & U32) -> Maybe<&2, U32>:
  (dynamic, count) = pair
  Some{count}

def dimensions(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      dimensions_pair(Core.shape(Core.pack(shape, tensor)))

def count(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      count_pair(Core.count(Core.pack(shape, tensor)))

def point(-shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Core.read_value(shape, Core.read_flat(shape, tensor, index))

def retained_point_pair(-shape: Shape.Shape, +other: U32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Maybe<&2, F32>:
  (tensor, result) = pair
  Core.checked_observe(shape, tensor, other)

def retained_point(-shape: Shape.Shape, +index: U32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      retained_point_pair(shape, other, Core.read_flat(shape, tensor, index))

def original_point(-shape: Shape.Shape, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Core.checked_observe(shape, tensor, other)

def after_value_pair(-shape: Shape.Shape, +other: U32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Maybe<&2, F32>:
  (tensor, result) = pair
  Core.read_value(shape, Core.read_flat(shape, tensor, other))

def after_value(-shape: Shape.Shape, +index: U32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      after_value_pair(shape, other, Core.read_flat(shape, tensor, index))

def after_length_pair(-shape: Shape.Shape,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Nat:
  (tensor, result) = pair
  Core.logical_length(shape, tensor)

def after_length(-shape: Shape.Shape, +index: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{after_length_pair(shape, Core.read_flat(shape, tensor, index))}

def after_dimensions_pair(+shape: Shape.Shape,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Maybe<&2, List<&2, U32>>:
  (tensor, result) = pair
  dimensions_pair(Core.shape(Core.pack(shape, tensor)))

def after_dimensions(+shape: Shape.Shape, +index: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      after_dimensions_pair(shape, Core.read_flat(shape, tensor, index))

def after_count_pair(+shape: Shape.Shape,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Maybe<&2, U32>:
  (tensor, result) = pair
  count_pair(Core.count(Core.pack(shape, tensor)))

def after_count(+shape: Shape.Shape, +index: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      after_count_pair(shape, Core.read_flat(shape, tensor, index))
