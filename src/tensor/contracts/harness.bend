import Base
import ../../shape/core.bend as Shape
import ../core.bend as Core
import ../elementwise.bend as Elementwise
import ../harness.bend as Harness
import ./broadcast.bend as Broadcast

# Runs `Broadcast.binary`, the pointwise family's broadcast instance, on the
# tensor corpus's binary requests and prints what `Harness.binary` prints for
# `Elementwise.binary`. A shape that does not broadcast fails before the
# family loop, as in `Elementwise.planned`.

def planned(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic,
  result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  match result:
    case Fail{error}:
      ((left, right), Fail{Core.ShapeError{error}})
    case Done{+plan}:
      Broadcast.binary(operation, plan, left, right)

def binary_value(+operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic) -> IO(Unit):
  Core.Packed{+left_shape, left_tensor} = left
  Core.Packed{+right_shape, right_tensor} = right
  Harness.retained_binary_result(planned(operation, Core.pack(left_shape, left_tensor), Core.pack(right_shape, right_tensor),
    Shape.broadcast_checked(left_shape, right_shape)))

def binary_right(+operation: Elementwise.Operation, left: Core.Dynamic,
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      Harness.construction_error(error)
    case Done{right}:
      binary_value(operation, left, right)

def binary(+operation: Elementwise.Operation, left: Result<&2, &1, Core.Error, Core.Dynamic>,
  right: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match left:
    case Fail{error}:
      Harness.construction_error(error)
    case Done{tensor}:
      binary_right(operation, tensor, right)
