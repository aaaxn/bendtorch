import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./layout.bend as Layout
import ./axis-spec.bend as Axis

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

def write_done_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Bool:
  (tensor, result) = pair
  Result.is_done(&2, &2, Core.Error, Unit, result)

def write_done(-shape: Shape.Shape, +index: U32, +value: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Bool>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{write_done_pair(shape, Core.write_flat(shape, tensor, index, value))}

def written_value_pair(-shape: Shape.Shape, +other: U32, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  Maybe<&2, F32>:
  (tensor, result) = pair
  Core.read_value(shape, Core.read_flat(shape, tensor, other))

def written_value(-shape: Shape.Shape, +index: U32, +value: F32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      written_value_pair(shape, other, Core.write_flat(shape, tensor, index, value))

def written_length_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Nat:
  (tensor, result) = pair
  Core.logical_length(shape, tensor)

def written_length(-shape: Shape.Shape, +index: U32, +value: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{written_length_pair(shape, Core.write_flat(shape, tensor, index, value))}

def written_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) ->
  Maybe<&2, List<&2, U32>>:
  (tensor, result) = pair
  dimensions_pair(Core.shape(Core.pack(shape, tensor)))

def written_dimensions(+shape: Shape.Shape, +index: U32, +value: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      written_dimensions_pair(shape, Core.write_flat(shape, tensor, index, value))

def written_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Maybe<&2, U32>:
  (tensor, result) = pair
  count_pair(Core.count(Core.pack(shape, tensor)))

def written_count(+shape: Shape.Shape, +index: U32, +value: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      written_count_pair(shape, Core.write_flat(shape, tensor, index, value))

def read_done_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> Bool:
  (tensor, result) = pair
  Result.is_done(&2, &2, Core.Error, F32, result)

def read_done(-shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Bool>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{read_done_pair(shape, Core.read_flat(shape, tensor, index))}

def read_status(result: Result<&2, &2, Core.Error, F32>) -> U32:
  match result:
    case Fail{error}:
      Core.error_code(error)
    case Done{value}:
      0

def read_code_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> U32:
  (tensor, result) = pair
  read_status(result)

def read_code(-shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{read_code_pair(shape, Core.read_flat(shape, tensor, index))}

def write_status(result: Result<&2, &2, Core.Error, Unit>) -> U32:
  match result:
    case Fail{error}:
      Core.error_code(error)
    case Done{unit}:
      0

def write_code_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> U32:
  (tensor, result) = pair
  write_status(result)

def write_code(-shape: Shape.Shape, +index: U32, +value: F32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{write_code_pair(shape, Core.write_flat(shape, tensor, index, value))}

def coordinate_index(-shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{index}:
      Some{Shape.index_value(shape, index)}

def coordinate_value_pair(+shape: Shape.Shape, +coordinates: List<&2, U32>,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Maybe<&2, F32>:
  (tensor, result) = pair
  Core.read_value(shape, Core.read_coordinate(shape, tensor, coordinates))

def coordinate_value(+shape: Shape.Shape, +index: U32, +value: F32, +coordinates: List<&2, U32>,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      coordinate_value_pair(shape, coordinates, Core.write_flat(shape, tensor, index, value))

def coordinate_written(+shape: Shape.Shape, +coordinates: List<&2, U32>, +value: F32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      written_value_pair(shape, other, Core.write_coordinate(shape, tensor, coordinates, value))

def dynamic_point(+shape: Shape.Shape, +index: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, tensor}, index))

def clone_first_value_pair(-shape: Shape.Shape, +other: U32, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.read_value(shape, Core.read_flat(shape, first, other))

def clone_second_value_pair(-shape: Shape.Shape, +other: U32, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, F32>:
  (first, second) = pair
  Core.read_value(shape, Core.read_flat(shape, second, other))

def clone_first_value(-shape: Shape.Shape, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_first_value_pair(shape, other, Core.clone(shape, tensor))

def clone_second_value(-shape: Shape.Shape, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_second_value_pair(shape, other, Core.clone(shape, tensor))

def clone_first_length_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Nat:
  (first, second) = pair
  Core.logical_length(shape, first)

def clone_second_length_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Nat:
  (first, second) = pair
  Core.logical_length(shape, second)

def clone_first_length(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{clone_first_length_pair(shape, Core.clone(shape, tensor))}

def clone_second_length(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{clone_second_length_pair(shape, Core.clone(shape, tensor))}

def clone_first_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, List<&2, U32>>:
  (first, second) = pair
  dimensions_pair(Core.shape(Core.pack(shape, first)))

def clone_second_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, List<&2, U32>>:
  (first, second) = pair
  dimensions_pair(Core.shape(Core.pack(shape, second)))

def clone_first_dimensions(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_first_dimensions_pair(shape, Core.clone(shape, tensor))

def clone_second_dimensions(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_second_dimensions_pair(shape, Core.clone(shape, tensor))

def clone_first_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, U32>:
  (first, second) = pair
  count_pair(Core.count(Core.pack(shape, first)))

def clone_second_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, U32>:
  (first, second) = pair
  count_pair(Core.count(Core.pack(shape, second)))

def clone_first_count(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_first_count_pair(shape, Core.clone(shape, tensor))

def clone_second_count(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_second_count_pair(shape, Core.clone(shape, tensor))

def clone_written_first_pair(-shape: Shape.Shape, +index: U32, +value: F32, +other: U32,
  pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, F32>:
  (first, second) = pair
  written_value_pair(shape, other, Core.write_flat(shape, first, index, value))

def clone_written_first(-shape: Shape.Shape, +index: U32, +value: F32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_written_first_pair(shape, index, value, other, Core.clone(shape, tensor))

def untouched(-shape: Shape.Shape, +other: U32, written: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  second: Core.Tensor<shape>) -> Maybe<&2, F32>:
  (tensor, result) = written
  Core.read_value(shape, Core.read_flat(shape, second, other))

def clone_written_second_pair(-shape: Shape.Shape, +index: U32, +value: F32, +other: U32,
  pair: Core.Tensor<shape> & Core.Tensor<shape>) -> Maybe<&2, F32>:
  (first, second) = pair
  untouched(shape, other, Core.write_flat(shape, first, index, value), second)

def clone_written_second(-shape: Shape.Shape, +index: U32, +value: F32, +other: U32,
  result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      clone_written_second_pair(shape, index, value, other, Core.clone(shape, tensor))

def length_number(-shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) -> Nat:
  (tensor, n) = pair
  n

def length_value(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{length_number(shape, Core.length(shape, tensor))}

def length_owner_value_pair(-shape: Shape.Shape, +other: U32, pair: Core.Tensor<shape> & Nat) -> Maybe<&2, F32>:
  (tensor, n) = pair
  Core.read_value(shape, Core.read_flat(shape, tensor, other))

def length_owner_value(-shape: Shape.Shape, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      length_owner_value_pair(shape, other, Core.length(shape, tensor))

def length_owner_length_pair(-shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) -> Nat:
  (tensor, n) = pair
  Core.logical_length(shape, tensor)

def length_owner_length(-shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, Nat>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{length_owner_length_pair(shape, Core.length(shape, tensor))}

def length_owner_dimensions_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) -> Maybe<&2, List<&2, U32>>:
  (tensor, n) = pair
  dimensions_pair(Core.shape(Core.pack(shape, tensor)))

def length_owner_dimensions(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, List<&2, U32>>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      length_owner_dimensions_pair(shape, Core.length(shape, tensor))

def length_owner_count_pair(+shape: Shape.Shape, pair: Core.Tensor<shape> & Nat) -> Maybe<&2, U32>:
  (tensor, n) = pair
  count_pair(Core.count(Core.pack(shape, tensor)))

def length_owner_count(+shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      length_owner_count_pair(shape, Core.length(shape, tensor))

def dynamic_owner_value(+other: U32, dynamic: Core.Dynamic) -> Maybe<&2, F32>:
  Core.dynamic_read_value(Core.read_dynamic(dynamic, other))

def shape_owner_pair(+other: U32, pair: Core.Dynamic & Shape.Shape) -> Maybe<&2, F32>:
  (dynamic, shape) = pair
  dynamic_owner_value(other, dynamic)

def shape_owner_value(+shape: Shape.Shape, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      shape_owner_pair(other, Core.shape(Core.pack(shape, tensor)))

def count_owner_pair(+other: U32, pair: Core.Dynamic & U32) -> Maybe<&2, F32>:
  (dynamic, count) = pair
  dynamic_owner_value(other, dynamic)

def count_owner_value(+shape: Shape.Shape, +other: U32, result: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Maybe<&2, F32>:
  match result:
    case Fail{error}:
      None{}
    case Done{tensor}:
      count_owner_pair(other, Core.count(Core.pack(shape, tensor)))

def listed(values: List<&2, F32>, found: Maybe<&2, U32>) -> Maybe<&2, F32>:
  match found:
    case None{}:
      None{}
    case Some{index}:
      List.get(&2, F32, values, U32.to_nat(index))

def permute_source(+input: Shape.Shape, +axes: List<&2, U32>, +index: U32) -> Maybe<&2, U32>:
  Layout.reference_source_coordinates(Shape.dimensions(input),
    Axis.scatter(U32, Shape.rank_dimensions(Shape.dimensions(input)), Shape.naturals(axes),
      Shape.words(Shape.digits(Shape.naturals(Axis.order(U32, Shape.dimensions(input), Shape.naturals(axes))), U32.to_nat(index))), 0n))

# The row-major source index that `permute_source` names for output index
# `index`, as a natural (0 where it names none).
def permute_source_index(+input: Shape.Shape, +axes: List<&2, U32>, +index: Nat) -> Nat:
  U32.to_nat(Maybe.default(&2, U32, permute_source(input, axes, U32.from_nat(index)), 0))
