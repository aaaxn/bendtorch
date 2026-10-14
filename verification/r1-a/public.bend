import Base
import ../../src/shape/core.bend as Shape
import ../../src/tensor/core.bend as Core
import ../../src/tensor/allocation-spec.bend as Allocation

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def huge_zero_shape(+a: U32, +b: U32, +c: U32) -> Shape.Shape:
  Shape.SShape{[0, a, b, c], 0n, {==}, Unit{}, Unit{}}

def result_dimensions(+shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      (dynamic, returned_shape) = Core.shape(Core.pack(shape, tensor))
      Some{Shape.dimensions(returned_shape)}

def result_count(+shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      (dynamic, count) = Core.count(Core.pack(shape, tensor))
      Some{count}

def tensor_readings(-shape: Shape.Shape, tensor: Core.Tensor<shape>) -> List<&2, Maybe<&2, F32>>:
  (tensor, first) = Core.read_flat(shape, tensor, 0)
  (tensor, second) = Core.read_flat(shape, tensor, 1)
  (tensor, third) = Core.read_flat(shape, tensor, 2)
  [Core.result_value(first), Core.result_value(second), Core.result_value(third)]

def result_readings(-shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, Maybe<&2, F32>>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{tensor_readings(shape, tensor)}

type Facts<-shape: Shape.Shape, -fill: F32, -dimensions: List<&2, U32>, -count: U32,
  -length: Nat, -values: List<&2, Maybe<&2, F32>>> is Data:
  Facts{
    successful: {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, Core.new(shape, fill)) == True{} : Bool},
    dimensions_exact: {result_dimensions(shape, Core.new(shape, fill)) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    count_exact: {result_count(shape, Core.new(shape, fill)) == Some{count} : Maybe<&2, U32>},
    length_exact: {Allocation.tensor_length(shape, Core.new(shape, fill)) == Some{length} : Maybe<&2, Nat>},
    values_exact: {result_readings(shape, Core.new(shape, fill)) == Some{values} : Maybe<&2, List<&2, Maybe<&2, F32>>>}}

def scalar_producer(+fill: F32) -> Facts<scalar_shape(), fill, [], 1, 1n, [Some{fill}, None{}, None{}]>:
  Facts{{==}, {==}, {==}, {==}, {==}}

def three_producer(+fill: F32) -> Facts<three_shape(), fill, [3], 3, 3n, [Some{fill}, Some{fill}, Some{fill}]>:
  Facts{{==}, {==}, {==}, {==}, {==}}

def empty_producer(+fill: F32) -> Facts<empty_shape(), fill, [0], 0, 0n, [None{}, None{}, None{}]>:
  Facts{{==}, {==}, {==}, {==}, {==}}

def huge_zero_producer(+fill: F32, +a: U32, +b: U32, +c: U32) ->
  Facts<huge_zero_shape(a, b, c), fill, [0, a, b, c], 0, 0n, [None{}, None{}, None{}]>:
  Facts{{==}, {==}, {==}, {==}, {==}}

def scalar_consumer() ->
  {result_readings(scalar_shape(), Core.new(scalar_shape(), 7.0)) ==
    Some{[Some{7.0}, None{}, None{}]} : Maybe<&2, List<&2, Maybe<&2, F32>>>}:
  Facts{successful, dimensions, count, length, values} = scalar_producer(7.0)
  values

def three_consumer() ->
  {result_readings(three_shape(), Core.new(three_shape(), 7.0)) ==
    Some{[Some{7.0}, Some{7.0}, Some{7.0}]} : Maybe<&2, List<&2, Maybe<&2, F32>>>}:
  Facts{successful, dimensions, count, length, values} = three_producer(7.0)
  values

def scalar_status_consumer() ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<scalar_shape()>, Core.new(scalar_shape(), 7.0)) == True{} : Bool}:
  Facts{successful, dimensions, count, length, values} = scalar_producer(7.0)
  successful

def empty_consumer() ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<empty_shape()>, Core.new(empty_shape(), 7.0)) == True{} : Bool} &
  ({result_dimensions(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{[0]} : Maybe<&2, List<&2, U32>>} &
    {result_count(empty_shape(), Core.new(empty_shape(), 7.0)) == Some{0} : Maybe<&2, U32>}):
  Facts{successful, dimensions, count, length, values} = empty_producer(7.0)
  (successful, (dimensions, count))

def huge_zero_consumer() ->
  {Result.is_done(&2, &1, Core.Error, Core.Tensor<huge_zero_shape(4294967295, 4294967295, 4294967295)>,
    Core.new(huge_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == True{} : Bool} &
  ({result_dimensions(huge_zero_shape(4294967295, 4294967295, 4294967295),
    Core.new(huge_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == Some{[0, 4294967295, 4294967295, 4294967295]} : Maybe<&2, List<&2, U32>>} &
    {result_count(huge_zero_shape(4294967295, 4294967295, 4294967295),
      Core.new(huge_zero_shape(4294967295, 4294967295, 4294967295), 7.0)) == Some{0} : Maybe<&2, U32>}):
  Facts{successful, dimensions, count, length, values} = huge_zero_producer(7.0, 4294967295, 4294967295, 4294967295)
  (successful, (dimensions, count))
