import Base
import ./core.bend as Shape
import ./nat_proofs.bend as Nats
import ./coordinate_proofs.bend as Coordinates
import ./projection_proofs.bend as Projection

def hollow(+output: Shape.Shape, +value: Nat, zero: {0n == Shape.numel(Shape.extents(output)) : Nat}, bound: {True{} == Nat.is_lt(value, Shape.elements(output)) : Bool}) -> Empty:
  +size = Equal.trans(Nat, Shape.elements(output), Shape.numel(Shape.extents(output)), 0n, Shape.elements_exact(output), Equal.sym(Nat, 0n, Shape.numel(Shape.extents(output)), zero))
  %Equal.trans(Bool, True{}, Nat.is_lt(value, 0n), False{}, Equal.trans(Bool, True{}, Nat.is_lt(value, Shape.elements(output)), Nat.is_lt(value, 0n), bound, Equal.cong(Nat, Bool, count => Nat.is_lt(value, count), Shape.elements(output), 0n, size)), Projection.lt_zero_false(value)) : Shape.truth(_)
  Unit{}

def left_bound(+plan: Shape.BroadcastPlan, -rows: List<&2, Shape.Row>, exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>}, +value: Nat, bound: {True{} == Nat.is_lt(value, Shape.elements(Shape.output_shape(plan))) : Bool}) -> {True{} == Nat.is_lt(Shape.walk_left(rows, value), Shape.elements(Shape.left_shape(plan))) : Bool}:
  +left = Shape.extents(Shape.left_shape(plan))
  +right = Shape.extents(Shape.right_shape(plan))
  +output = Shape.extents(Shape.output_shape(plan))
  +input = Shape.left_shape(plan)
  +fits = Coordinates.size_fits(Shape.output_shape(plan), value, bound)
  +index = Coordinates.fin_from(Shape.numel(output), value, fits)
  +projected = Shape.project_left(left, right, output, Shape.alignment(plan), Shape.unflatten(output, index))
  +table = Shape.index_table(left, right, output)
  %Equal.sym(List<&2, Shape.Row>, rows, table, exact) : {True{} == Nat.is_lt(Shape.walk_left(_, value), Shape.elements(input)) : Bool}
  %Coordinates.fin_from_value(Shape.numel(output), value, fits) : {True{} == Nat.is_lt(Shape.walk_left(table, _), Shape.elements(input)) : Bool}
  %Equal.sym(Nat, Shape.walk_left(table, Shape.Fin.value(Shape.numel(output), index)), Shape.row_major(Shape.extents(input), projected), Projection.walk_left_value(left, right, output, Shape.alignment(plan), index)) : {True{} == Nat.is_lt(_, Shape.elements(input)) : Bool}
  %Equal.sym(Nat, Shape.elements(input), Shape.numel(Shape.extents(input)), Shape.elements_exact(input)) : {True{} == Nat.is_lt(Shape.row_major(Shape.extents(input), projected), _) : Bool}
  %Nats.lt_true(Shape.row_major(Shape.extents(input), projected), Shape.numel(Shape.extents(input)), Coordinates.row_major_lt(Shape.extents(input), projected)) : {True{} == _ : Bool}
  {==}

def project_left_index(+plan: Shape.BroadcastPlan, map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, index: Shape.FlatIndex<Shape.output_shape(plan)>) -> Shape.FlatIndex<Shape.left_shape(plan)>:
  match map index:
    case Shape.Indexed{rows, exact} Shape.FIndex{+value, bound}:
      Shape.FIndex{Shape.walk_left(rows, value), left_bound(plan, rows, exact, value, bound)}
    case Shape.Unindexed{zero} Shape.FIndex{+value, bound}:
      Empty.absurd(Shape.FlatIndex<Shape.left_shape(plan)>, hollow(Shape.output_shape(plan), value, zero, bound))

def right_bound(+plan: Shape.BroadcastPlan, -rows: List<&2, Shape.Row>, exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>}, +value: Nat, bound: {True{} == Nat.is_lt(value, Shape.elements(Shape.output_shape(plan))) : Bool}) -> {True{} == Nat.is_lt(Shape.walk_right(rows, value), Shape.elements(Shape.right_shape(plan))) : Bool}:
  +left = Shape.extents(Shape.left_shape(plan))
  +right = Shape.extents(Shape.right_shape(plan))
  +output = Shape.extents(Shape.output_shape(plan))
  +input = Shape.right_shape(plan)
  +fits = Coordinates.size_fits(Shape.output_shape(plan), value, bound)
  +index = Coordinates.fin_from(Shape.numel(output), value, fits)
  +projected = Shape.project_right(left, right, output, Shape.alignment(plan), Shape.unflatten(output, index))
  +table = Shape.index_table(left, right, output)
  %Equal.sym(List<&2, Shape.Row>, rows, table, exact) : {True{} == Nat.is_lt(Shape.walk_right(_, value), Shape.elements(input)) : Bool}
  %Coordinates.fin_from_value(Shape.numel(output), value, fits) : {True{} == Nat.is_lt(Shape.walk_right(table, _), Shape.elements(input)) : Bool}
  %Equal.sym(Nat, Shape.walk_right(table, Shape.Fin.value(Shape.numel(output), index)), Shape.row_major(Shape.extents(input), projected), Projection.walk_right_value(left, right, output, Shape.alignment(plan), index)) : {True{} == Nat.is_lt(_, Shape.elements(input)) : Bool}
  %Equal.sym(Nat, Shape.elements(input), Shape.numel(Shape.extents(input)), Shape.elements_exact(input)) : {True{} == Nat.is_lt(Shape.row_major(Shape.extents(input), projected), _) : Bool}
  %Nats.lt_true(Shape.row_major(Shape.extents(input), projected), Shape.numel(Shape.extents(input)), Coordinates.row_major_lt(Shape.extents(input), projected)) : {True{} == _ : Bool}
  {==}

def project_right_index(+plan: Shape.BroadcastPlan, map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, index: Shape.FlatIndex<Shape.output_shape(plan)>) -> Shape.FlatIndex<Shape.right_shape(plan)>:
  match map index:
    case Shape.Indexed{rows, exact} Shape.FIndex{+value, bound}:
      Shape.FIndex{Shape.walk_right(rows, value), right_bound(plan, rows, exact, value, bound)}
    case Shape.Unindexed{zero} Shape.FIndex{+value, bound}:
      Empty.absurd(Shape.FlatIndex<Shape.right_shape(plan)>, hollow(Shape.output_shape(plan), value, zero, bound))
