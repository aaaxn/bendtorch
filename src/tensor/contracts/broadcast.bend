import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../elementwise.bend as Elementwise
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise

def left_walk(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  map: Shape.Index<left, right, output>, +index: Nat) -> Nat:
  match map:
    case Shape.Indexed{rows, exact}:
      Shape.walk_left(rows, index)
    case Shape.Unindexed{zero}:
      0n

def right_walk(-left: List<&2, Nat>, -right: List<&2, Nat>, -output: List<&2, Nat>,
  map: Shape.Index<left, right, output>, +index: Nat) -> Nat:
  match map:
    case Shape.Indexed{rows, exact}:
      Shape.walk_right(rows, index)
    case Shape.Unindexed{zero}:
      0n

def left_map(plan: Shape.BroadcastPlan, index: Nat) -> Nat:
  +plan = plan
  left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
    Shape.index_map(plan), index)

def right_map(plan: Shape.BroadcastPlan, index: Nat) -> Nat:
  +plan = plan
  right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)),
    Shape.index_map(plan), index)

def binary(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Dynamic, right: Core.Dynamic) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  Pointwise.run(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~left_map, ~right_map,
    operation, plan, Shape.output_shape(plan), left, right)

def cell(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +index: Nat) -> F32:
  Elementwise.scalar(operation, Replayed.value(Shape.left_shape(plan), left_fill, left_ops, U32.from_nat(left_map(plan, index))),
    Replayed.value(Shape.right_shape(plan), right_fill, right_ops, U32.from_nat(right_map(plan, index))))
