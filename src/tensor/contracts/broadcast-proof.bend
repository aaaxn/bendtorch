import Base
import ../../shape/core.bend as Shape
import ../../shape/flat.bend as Flat
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../elementwise.bend as Elementwise
import ../word-conversion.bend as Conversion
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise-proof.bend as Proof
import ./pointwise-records.bend as Records
import ./broadcast.bend as Broadcast

def output_inside(+plan: Shape.BroadcastPlan, +index: Nat,
  below: {Nat.is_lt(index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {True{} == Nat.is_lt(index, Shape.elements(Shape.output_shape(plan))) : Bool}:
  %Equal.sym(Nat, Shape.elements(Shape.output_shape(plan)), Shape.numel(Shape.extents(Shape.output_shape(plan))),
    Shape.elements_exact(Shape.output_shape(plan))) : {True{} == Nat.is_lt(index, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))), True{}, below)

def left_walk_bound(+plan: Shape.BroadcastPlan,
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +index: Nat, below: {Nat.is_lt(index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Nat.is_lt(Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, index),
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool}:
  match map:
    case Shape.Indexed{rows, exact}:
      Equal.sym(Bool, True{}, Nat.is_lt(Shape.walk_left(rows, index), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))),
        %Shape.elements_exact(Shape.left_shape(plan)) : {True{} == Nat.is_lt(Shape.walk_left(rows, index), _) : Bool}
        Flat.left_bound(plan, rows, exact, index, output_inside(plan, index, below)))
    case Shape.Unindexed{zero}:
      Empty.absurd({Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool},
        Arith.lt_zero(index,
          %Equal.sym(Nat, 0n, Shape.numel(Shape.extents(Shape.output_shape(plan))), zero) : {Nat.is_lt(index, _) == True{} : Bool}
          below))

def right_walk_bound(+plan: Shape.BroadcastPlan,
  map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +index: Nat, below: {Nat.is_lt(index, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Nat.is_lt(Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan)), map, index),
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}:
  match map:
    case Shape.Indexed{rows, exact}:
      Equal.sym(Bool, True{}, Nat.is_lt(Shape.walk_right(rows, index), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
        %Shape.elements_exact(Shape.right_shape(plan)) : {True{} == Nat.is_lt(Shape.walk_right(rows, index), _) : Bool}
        Flat.right_bound(plan, rows, exact, index, output_inside(plan, index, below)))
    case Shape.Unindexed{zero}:
      Empty.absurd({Nat.is_lt(0n, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool},
        Arith.lt_zero(index,
          %Equal.sym(Nat, 0n, Shape.numel(Shape.extents(Shape.output_shape(plan))), zero) : {Nat.is_lt(index, _) == True{} : Bool}
          below))

def left_inside(+plan: Shape.BroadcastPlan, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Pointwise.inside(~Shape.BroadcastPlan, ~Broadcast.left_map, plan, count, Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +total = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +bound = Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))
      %Equal.sym(Bool, Pointwise.inside(~Shape.BroadcastPlan, ~Broadcast.left_map, plan, previous, bound), True{},
        left_inside(plan, previous, Conversion.le_predecessor(previous, total, room))) :
        {Bool.and(_, Nat.is_lt(Broadcast.left_map(plan, previous), bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Broadcast.left_map(plan, previous), bound), True{},
        left_walk_bound(plan, Shape.index_map(plan), previous, Conversion.le_successor_lt(previous, total, room))) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def right_inside(+plan: Shape.BroadcastPlan, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Pointwise.inside(~Shape.BroadcastPlan, ~Broadcast.right_map, plan, count, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +total = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +bound = Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))
      %Equal.sym(Bool, Pointwise.inside(~Shape.BroadcastPlan, ~Broadcast.right_map, plan, previous, bound), True{},
        right_inside(plan, previous, Conversion.le_predecessor(previous, total, room))) :
        {Bool.and(_, Nat.is_lt(Broadcast.right_map(plan, previous), bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(Broadcast.right_map(plan, previous), bound), True{},
        right_walk_bound(plan, Shape.index_map(plan), previous, Conversion.le_successor_lt(previous, total, room))) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def broadcast_pointwise(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), left_fill, left_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), right_fill, right_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  Records.PointwiseContract<Shape.output_shape(plan), Shape.left_shape(plan), Shape.right_shape(plan),
    index => Broadcast.cell(operation, plan, left_fill, left_ops, right_fill, right_ops, index),
    other => Replayed.value(Shape.left_shape(plan), left_fill, left_ops, other),
    other => Replayed.value(Shape.right_shape(plan), right_fill, right_ops, other),
    Broadcast.binary(operation, plan,
      Core.Packed{Shape.left_shape(plan), Core.TTensor{Replayed.storage(Shape.left_shape(plan), left_fill, left_ops), left_fits}},
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), right_fill, right_ops), right_fits}})>:
  +total = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
  Proof.contract(~Elementwise.Operation, ~Elementwise.scalar, ~Shape.BroadcastPlan, ~Broadcast.left_map, ~Broadcast.right_map,
    operation, plan, Shape.left_shape(plan), left_fill, left_ops, Shape.right_shape(plan), right_fill, right_ops, Shape.output_shape(plan),
    left_inside(plan, total, Conversion.le_reflexive(total)), right_inside(plan, total, Conversion.le_reflexive(total)), left_fits, right_fits)
