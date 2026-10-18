import Base
import ../shape/core.bend as Shape
import ./core.bend as Core
import ./elementwise.bend as Elementwise

def print_value(result: Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{value}:
      IO.print(String.append("bits:", U32.show(F32.bits(value))))

def print_write(result: Result<&2, &2, Core.Error, Unit>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{unit}:
      IO.print("written")

def observed(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Core.Dynamic):
  (dynamic, result) = pair
  do IO<Core.Dynamic>:
    print_value(result)
    return dynamic

def written(pair: Core.Dynamic & Result<&2, &2, Core.Error, Unit>) -> IO(Core.Dynamic):
  (dynamic, result) = pair
  do IO<Core.Dynamic>:
    print_write(result)
    return dynamic

def print_values_step(rec: Core.Dynamic -> IO(Core.Dynamic), pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Core.Dynamic):
  (dynamic, result) = pair
  do IO<Core.Dynamic>:
    print_value(result)
    rec(dynamic)

def print_values(fuel: Nat, +index: U32, dynamic: Core.Dynamic) -> IO(Core.Dynamic):
  match fuel:
    case 0n:
      IO.pure(Core.Dynamic, dynamic)
    case 1n+remaining:
      print_values_step(dynamic => print_values(remaining, U32.add(index, 1), dynamic), Core.read_dynamic(dynamic, index))

def print_count(pair: Core.Dynamic & U32) -> IO(Core.Dynamic):
  (dynamic, +count) = pair
  do IO<Core.Dynamic>:
    IO.print(String.append("count:", U32.show(count)))
    print_values(U32.to_nat(count), 0, dynamic)

def show(dynamic: Core.Dynamic) -> IO(Core.Dynamic):
  print_count(Core.count(dynamic))

def binary_output(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  (inputs, result) = pair
  (left, right) = inputs
  match result:
    case Fail{error}:
      do IO<Unit>:
        IO.print(String.append("error:", U32.show(Core.error_code(error))))
        left : Core.Dynamic <- show(left)
        right : Core.Dynamic <- show(right)
        return Unit{}
    case Done{output}:
      do IO<Unit>:
        output : Core.Dynamic <- show(output)
        IO.print("retained-left")
        left : Core.Dynamic <- show(left)
        IO.print("retained-right")
        right : Core.Dynamic <- show(right)
        return Unit{}

def binary_right(operation: Elementwise.Operation, left: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{right}:
      binary_output(Elementwise.binary(operation, left, right))

def binary_left(operation: Elementwise.Operation, right_dimensions: List<&2, U32>, right_values: List<&2, F32>,
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{left}:
      binary_right(operation, left, Core.construct(right_dimensions, right_values))

def binary_case(operation: Elementwise.Operation, left_dimensions: List<&2, U32>, left_values: List<&2, F32>,
  right_dimensions: List<&2, U32>, right_values: List<&2, F32>) -> IO(Unit):
  binary_left(operation, right_dimensions, right_values, Core.construct(left_dimensions, left_values))

def construction_result(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{dynamic}:
      do IO<Unit>:
        dynamic : Core.Dynamic <- show(dynamic)
        return Unit{}

def coordinate_observed(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) -> IO(Core.Tensor<shape>):
  (tensor, result) = pair
  do IO<Core.Tensor<shape>>:
    print_value(result)
    return tensor

def invalid_coordinates(dynamic: Core.Dynamic) -> IO(Core.Dynamic):
  Core.Packed{+shape, tensor} = dynamic
  do IO<Core.Dynamic>:
    tensor : Core.Tensor<shape> <- coordinate_observed(shape, Core.read_coordinate(shape, tensor, [1, 0]))
    tensor : Core.Tensor<shape> <- coordinate_observed(shape, Core.read_coordinate(shape, tensor, [3]))
    return Core.pack(shape, tensor)

def clone_case(pair: Core.Dynamic & Core.Dynamic) -> IO(Unit):
  (original, copy) = pair
  do IO<Unit>:
    copy : Core.Dynamic <- written(Core.write_dynamic(copy, 0, 99.0))
    original : Core.Dynamic <- observed(Core.read_dynamic(original, 0))
    copy : Core.Dynamic <- observed(Core.read_dynamic(copy, 0))
    return Unit{}

def access_case(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{dynamic}:
      do IO<Unit>:
        dynamic : Core.Dynamic <- written(Core.write_dynamic(dynamic, 3, 99.0))
        dynamic : Core.Dynamic <- observed(Core.read_dynamic(dynamic, 1))
        dynamic : Core.Dynamic <- written(Core.write_dynamic(dynamic, 1, 7.0))
        dynamic : Core.Dynamic <- observed(Core.read_dynamic(dynamic, 1))
        dynamic : Core.Dynamic <- invalid_coordinates(dynamic)
        dynamic : Core.Dynamic <- observed(Core.read_dynamic(dynamic, 4294967295))
        clone_case(Core.clone_dynamic(dynamic))

def empty_access(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print(String.append("error:", U32.show(Core.error_code(error))))
    case Done{dynamic}:
      do IO<Unit>:
        dynamic : Core.Dynamic <- observed(Core.read_dynamic(dynamic, 0))
        dynamic : Core.Dynamic <- written(Core.write_dynamic(dynamic, 0, 1.0))
        return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("scalar-add")
    binary_case(Elementwise.Add{}, [], [2.0], [], [3.0])
    IO.print("singleton-add")
    binary_case(Elementwise.Add{}, [2, 1], [1.0, 2.0], [1, 3], [10.0, 20.0, 30.0])
    IO.print("nonpower-sub")
    binary_case(Elementwise.Sub{}, [3], [4.0, 5.0, 6.0], [], [2.0])
    IO.print("nonpower-mul")
    binary_case(Elementwise.Mul{}, [3], [1.0, 2.0, 3.0], [1], [2.0])
    IO.print("zero-singleton")
    binary_case(Elementwise.Add{}, [0, 3], [], [1, 3], [1.0, 2.0, 3.0])
    IO.print("singleton-zero")
    binary_case(Elementwise.Mul{}, [1, 3], [1.0, 2.0, 3.0], [0, 3], [])
    IO.print("incompatible")
    binary_case(Elementwise.Add{}, [2], [1.0, 2.0], [3], [1.0, 2.0, 3.0])
    IO.print("rank-four")
    construction_result(Core.construct([1, 1, 1, 3], [1.0, 2.0, 3.0]))
    IO.print("invalid-count")
    construction_result(Core.construct([3], [1.0, 2.0]))
    construction_result(Core.construct([], []))
    construction_result(Core.construct([0], [1.0]))
    IO.print("invalid-rank-capacity")
    construction_result(Core.construct([1, 1, 1, 1, 1], [1.0]))
    construction_result(Core.construct([1048577], []))
    IO.print("access-clone")
    access_case(Core.construct([3], [2.0, 3.0, 4.0]))
    IO.print("empty-access")
    empty_access(Core.construct([0], []))
