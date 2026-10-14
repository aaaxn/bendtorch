import Base
import ../shape/core.bend as Shape
import ../shape/flat.bend as Flat
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../shape/boundary_proofs.bend as Boundary
import ../shape/coordinate_proofs.bend as Coordinates
import ./core.bend as Core
import ./elementwise.bend as Elementwise
import ./projection-LAWS.bend as Laws

def projected_pair(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +index: Shape.FlatIndex<Shape.output_shape(plan)>) ->
  {Some{(Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index)),
    Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index)))} ==
    Laws.offset_spec(plan, ShapeLaws.position(Shape.output_shape(plan), index)) : Maybe<&1, U32 & U32>}:
  +left = Shape.extents(Shape.left_shape(plan))
  +right = Shape.extents(Shape.right_shape(plan))
  +output = Shape.extents(Shape.output_shape(plan))
  +decoded = ShapeLaws.fin_of(Shape.numel(output), ShapeLaws.position(Shape.output_shape(plan), index))
  +left_value = Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index))
  +right_value = Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index))
  %ShapeLaws.project_left_index_refines(plan, map, index) :
    {Some{(left_value, right_value)} == Laws.pair_offsets(_, ShapeLaws.flatten_spec(right,
      ShapeLaws.project_right_coordinate(left, right, output, Shape.alignment(plan), decoded))) : Maybe<&1, U32 & U32>}
  %ShapeLaws.project_right_index_refines(plan, map, index) :
    {Some{(left_value, right_value)} == Laws.pair_offsets(Some{left_value}, _) : Maybe<&1, U32 & U32>}
  {==}

def Laws.actual_flat_projection(plan, index):
  projected_pair(plan, Shape.index_map(plan), index)

def checked_projection(+plan: Shape.BroadcastPlan, +position: Nat,
  decision: Shape.BoolCase<Nat.is_lt(position, Shape.elements(Shape.output_shape(plan)))>) ->
  {Laws.indices_result(Elementwise.flat_input_indices(plan,
    Shape.flat_checked.finish(Shape.output_shape(plan), position, decision))) ==
    Laws.offset_spec(plan, position) : Maybe<&1, U32 & U32>}:
  match decision:
    case Shape.BFalse{equal}:
      %Boundary.fin_of_none(Shape.numel(Shape.extents(Shape.output_shape(plan))), position,
        Coordinates.size_misses(Shape.output_shape(plan), position, equal)) :
        {None{} == Laws.pair_offsets(
          ShapeLaws.flatten_spec(Shape.extents(Shape.left_shape(plan)), ShapeLaws.project_left_coordinate(
            Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
            Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), _)),
          ShapeLaws.flatten_spec(Shape.extents(Shape.right_shape(plan)), ShapeLaws.project_right_coordinate(
            Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
            Shape.extents(Shape.output_shape(plan)), Shape.alignment(plan), _))) : Maybe<&1, U32 & U32>}
      {==}
    case Shape.BTrue{equal}:
      Laws.actual_flat_projection(plan, Shape.FIndex{position, equal})

def Laws.input_indices_refines(plan, index):
  checked_projection(plan, U32.to_nat(index),
    Shape.bool_case(Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan)))))

def checked_acceptance(+plan: Shape.BroadcastPlan, +position: Nat,
  decision: Shape.BoolCase<Nat.is_lt(position, Shape.elements(Shape.output_shape(plan)))>) ->
  {Laws.indices_accepted(Elementwise.flat_input_indices(plan,
    Shape.flat_checked.finish(Shape.output_shape(plan), position, decision))) ==
    Nat.is_lt(position, Shape.numel(Shape.extents(Shape.output_shape(plan)))) : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Coordinates.size_misses(Shape.output_shape(plan), position, equal)
    case Shape.BTrue{equal}:
      Coordinates.size_fits(Shape.output_shape(plan), position, equal)

def Laws.input_indices_acceptance(plan, index):
  checked_acceptance(plan, U32.to_nat(index),
    Shape.bool_case(Nat.is_lt(U32.to_nat(index), Shape.elements(Shape.output_shape(plan)))))

def expected_indices_view(+operation: Elementwise.Operation, -left: Core.Dynamic, -right: Core.Dynamic,
  result: Result<&2, &1, Core.Error, U32 & U32>) ->
  {Elementwise.expected_indices(operation, left, right, result) ==
    Laws.specified_expected(operation, left, right, Laws.indices_result(result)) : Maybe<&2, F32>}:
  match result:
    case Fail{error}:
      {==}
    case Done{indices}:
      (left_index, right_index) = indices
      {==}

def Laws.expected_projection_refines(operation, plan, left, right, index):
  Equal.trans(Maybe<&2, F32>, Elementwise.expected(operation, plan, left, right, index),
    Laws.specified_expected(operation, left, right, Laws.indices_result(Elementwise.input_indices(plan, index))),
    Laws.specified_expected(operation, left, right, Laws.offset_spec(plan, U32.to_nat(index))),
    expected_indices_view(operation, left, right, Elementwise.input_indices(plan, index)),
    Equal.cong(Maybe<&1, U32 & U32>, Maybe<&2, F32>,
      offsets => Laws.specified_expected(operation, left, right, offsets),
      Laws.indices_result(Elementwise.input_indices(plan, index)), Laws.offset_spec(plan, U32.to_nat(index)),
      Laws.input_indices_refines(plan, index)))
