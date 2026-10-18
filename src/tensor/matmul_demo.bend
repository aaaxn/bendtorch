import Base
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./matmul.bend as Matrix
import ./reduction.bend as Reduction

def report_read(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Core.Dynamic):
  (tensor, result) = pair
  match result:
    case Fail{error}:
      do IO<Core.Dynamic>:
        IO.print("read-error")
        return tensor
    case Done{value}:
      do IO<Core.Dynamic>:
        IO.print(U32.show(F32.bits(value)))
        return tensor

def report_sum(result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("sum-error")
    case Done{pair}:
      (tensor, scalar) = pair
      do IO<Unit>:
        scalar : Core.Dynamic <- report_read(Core.read_dynamic(scalar, 0))
        tensor : Core.Dynamic <- report_read(Core.read_dynamic(tensor, 0))
        tensor : Core.Dynamic <- report_read(Core.read_dynamic(tensor, 1))
        tensor : Core.Dynamic <- report_read(Core.read_dynamic(tensor, 2))
        tensor : Core.Dynamic <- report_read(Core.read_dynamic(tensor, 3))
        return Unit{}

def biased(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  (inputs, result) = pair
  match result:
    case Fail{error}:
      IO.print("bias-error")
    case Done{tensor}:
      report_sum(Reduction.sum_dynamic(tensor))

def bias_constructed(product: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("bias-construction-error")
    case Done{bias}:
      biased(Elementwise.add(product, bias))

def multiplied(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("matrix-error")
    case Done{values}:
      (a, b, product) = values
      bias_constructed(product, Core.construct([2], [1.0, F32.neg(1.0)]))

def b_constructed(a: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("B-construction-error")
    case Done{b}:
      multiplied(Matrix.matmul(a, b))

def a_constructed(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("A-construction-error")
    case Done{a}:
      b_constructed(a, Core.construct([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))

def report_count(pair: Core.Dynamic & U32) -> IO(Core.Dynamic):
  (tensor, count) = pair
  do IO<Core.Dynamic>:
    IO.print(U32.show(count))
    return tensor

def report_values(fuel: Nat, tensor: Core.Dynamic, +index: U32) -> IO(Core.Dynamic):
  match fuel:
    case 0n:
      do IO<Core.Dynamic>:
        return tensor
    case 1n+tail:
      do IO<Core.Dynamic>:
        tensor : Core.Dynamic <- report_read(Core.read_dynamic(tensor, index))
        tensor : Core.Dynamic <- report_values(tail, tensor, U32.add(index, 1))
        return tensor

def zero_inner_result(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("zero-inner-error")
    case Done{values}:
      (a, b, product) = values
      do IO<Unit>:
        IO.print("zero-inner")
        a : Core.Dynamic <- report_count(Core.count(a))
        b : Core.Dynamic <- report_count(Core.count(b))
        product : Core.Dynamic <- report_count(Core.count(product))
        product : Core.Dynamic <- report_values(6n, product, 0)
        return Unit{}

def empty_output_result(result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("empty-output-error")
    case Done{values}:
      (a, b, product) = values
      do IO<Unit>:
        IO.print("empty-output")
        a : Core.Dynamic <- report_count(Core.count(a))
        b : Core.Dynamic <- report_count(Core.count(b))
        product : Core.Dynamic <- report_count(Core.count(product))
        b : Core.Dynamic <- report_read(Core.read_dynamic(b, 0))
        return Unit{}

def edge_right(report: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> IO(Unit),
  a: Core.Dynamic, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("edge-construction-error")
    case Done{b}:
      report(Matrix.matmul(a, b))

def edge_left(report: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> IO(Unit),
  right_dimensions: List<&2, U32>, right_values: List<&2, F32>, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("edge-construction-error")
    case Done{a}:
      edge_right(report, a, Core.construct(right_dimensions, right_values))

def edge_matrix(report: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic> -> IO(Unit),
  left_dimensions: List<&2, U32>, left_values: List<&2, F32>, right_dimensions: List<&2, U32>, right_values: List<&2, F32>) -> IO(Unit):
  edge_left(report, right_dimensions, right_values, Core.construct(left_dimensions, left_values))

def empty_sum_result(result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("empty-sum-error")
    case Done{pair}:
      (input, scalar) = pair
      do IO<Unit>:
        IO.print("empty-sum")
        scalar : Core.Dynamic <- report_read(Core.read_dynamic(scalar, 0))
        input : Core.Dynamic <- report_count(Core.count(input))
        return Unit{}

def empty_sum_input(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("empty-construction-error")
    case Done{tensor}:
      empty_sum_result(Reduction.sum_dynamic(tensor))

def main() -> IO(Unit):
  do IO<Unit>:
    a_constructed(Core.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    edge_matrix(zero_inner_result, [2, 0], [], [0, 3], [])
    empty_sum_input(Core.construct([0], []))
    edge_matrix(empty_output_result, [0, 2], [], [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return Unit{}
