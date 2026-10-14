import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ./core.bend as Core
import ./elementwise.bend as Elementwise

def pair_offsets(left: Maybe<&2, U32>, right: Maybe<&2, U32>) -> Maybe<&1, U32 & U32>:
  match left right:
    case None{} _:
      None{}
    case Some{left} None{}:
      None{}
    case Some{left} Some{right}:
      Some{(left, right)}

def offset_spec(+plan: Shape.BroadcastPlan, +position: Nat) -> Maybe<&1, U32 & U32>:
  +left = Shape.extents(Shape.left_shape(plan))
  +right = Shape.extents(Shape.right_shape(plan))
  +output = Shape.extents(Shape.output_shape(plan))
  +index = ShapeLaws.fin_of(Shape.numel(output), position)
  pair_offsets(
    ShapeLaws.flatten_spec(left, ShapeLaws.project_left_coordinate(left, right, output, Shape.alignment(plan), index)),
    ShapeLaws.flatten_spec(right, ShapeLaws.project_right_coordinate(left, right, output, Shape.alignment(plan), index)))

def indices_result(result: Result<&2, &1, Core.Error, U32 & U32>) -> Maybe<&1, U32 & U32>:
  match result:
    case Fail{error}:
      None{}
    case Done{indices}:
      Some{indices}

def indices_accepted(result: Result<&2, &1, Core.Error, U32 & U32>) -> Bool:
  match result:
    case Fail{error}:
      False{}
    case Done{indices}:
      True{}

def specified_expected(operation: Elementwise.Operation, left: Core.Dynamic, right: Core.Dynamic,
  offsets: Maybe<&1, U32 & U32>) -> Maybe<&2, F32>:
  match offsets:
    case None{}:
      None{}
    case Some{indices}:
      (left_index, right_index) = indices
      Elementwise.expected_left(operation, right, right_index, Core.checked_observe_dynamic(left, left_index))

law actual_flat_projection:
  for +plan: Shape.BroadcastPlan
  for +index: Shape.FlatIndex<Shape.output_shape(plan)>
  {indices_result(Elementwise.flat_input_indices(plan, Done{index})) ==
    offset_spec(plan, ShapeLaws.position(Shape.output_shape(plan), index)) : Maybe<&1, U32 & U32>}

law input_indices_refines:
  for +plan: Shape.BroadcastPlan
  for +index: U32
  {indices_result(Elementwise.input_indices(plan, index)) ==
    offset_spec(plan, U32.to_nat(index)) : Maybe<&1, U32 & U32>}

law input_indices_acceptance:
  for +plan: Shape.BroadcastPlan
  for +index: U32
  {indices_accepted(Elementwise.input_indices(plan, index)) ==
    Nat.is_lt(U32.to_nat(index), Shape.numel(Shape.extents(Shape.output_shape(plan)))) : Bool}

law expected_projection_refines:
  for +operation: Elementwise.Operation
  for +plan: Shape.BroadcastPlan
  for left: Core.Dynamic
  for right: Core.Dynamic
  for +index: U32
  {Elementwise.expected(operation, plan, left, right, index) ==
    specified_expected(operation, left, right, offset_spec(plan, U32.to_nat(index))) : Maybe<&2, F32>}
