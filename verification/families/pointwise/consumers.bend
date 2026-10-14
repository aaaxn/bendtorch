import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/elementwise.bend as Elementwise
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/from-values-contract.bend as FromValues
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/PROOF.bend as ContractsProof
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/broadcast.bend as Broadcast
import ../../../src/tensor/contracts/pointwise.bend as Pointwise
import ../../../src/tensor/contracts/pointwise-records.bend as Records

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def column_shape() -> Shape.Shape:
  Shape.SShape{[2, 1], 2n, {==}, Unit{}, Unit{}}

def row_shape() -> Shape.Shape:
  Shape.SShape{[1, 3], 3n, {==}, Unit{}, Unit{}}

def empty_grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 0], 0n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def plan_of(r: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  done: {Result.is_done(&2, &2, Shape.Error, Shape.BroadcastPlan, r) == True{} : Bool}) -> Shape.BroadcastPlan:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Shape.BroadcastPlan, Arith.clash(done))

def grid_three() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(grid_shape(), three_shape()), {==})

def column_row() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(column_shape(), row_shape()), {==})

def three_three() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(three_shape(), three_shape()), {==})

def scalar_three() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(scalar_shape(), three_shape()), {==})

def empty_plan() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(empty_grid_shape(), empty_shape()), {==})

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def result(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  Broadcast.binary(operation, plan, input(Shape.left_shape(plan), left, left_fits), input(Shape.right_shape(plan), right, right_fits))

def Values(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) -> Type:
  Records.PointwiseContract<Shape.output_shape(plan), Shape.left_shape(plan), Shape.right_shape(plan),
    index => Broadcast.cell(operation, plan, 0.0, writes(left), 0.0, writes(right), index),
    other => Replayed.value(Shape.left_shape(plan), 0.0, writes(left), other),
    other => Replayed.value(Shape.right_shape(plan), 0.0, writes(right), other),
    result(operation, plan, left, right, left_fits, right_fits)>

type PointCase<-result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, -dimensions: List<&2, U32>,
  -index: U32, -expected: F32, -left_index: U32, -left_value: F32, -right_index: U32, -right_value: F32> is Data:
  PointCase{
    successful: {Pointwise.succeeded(result) == True{} : Bool},
    dimensions_exact: {Pointwise.output_dimensions(result) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    value_exact: {Pointwise.output_point(result, index) == Some{expected} : Maybe<&2, F32>},
    left_exact: {Pointwise.left_point(result, left_index) == Some{left_value} : Maybe<&2, F32>},
    right_exact: {Pointwise.right_point(result, right_index) == Some{right_value} : Maybe<&2, F32>}}

type EmptyCase<-result: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, -dimensions: List<&2, U32>> is Data:
  EmptyCase{
    successful: {Pointwise.succeeded(result) == True{} : Bool},
    dimensions_exact: {Pointwise.output_dimensions(result) == Some{dimensions} : Maybe<&2, List<&2, U32>>}}

def open_case(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Values(operation, plan, left, right, left_fits, right_fits),
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +left_index: U32, left_inside: {Nat.is_lt(U32.to_nat(left_index), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool},
  +right_index: U32, right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}) ->
  PointCase<result(operation, plan, left, right, left_fits, right_fits), Shape.dimensions(Shape.output_shape(plan)),
    index, Broadcast.cell(operation, plan, 0.0, writes(left), 0.0, writes(right), U32.to_nat(index)),
    left_index, Replayed.value(Shape.left_shape(plan), 0.0, writes(left), left_index),
    right_index, Replayed.value(Shape.right_shape(plan), 0.0, writes(right), right_index)>:
  Records.PointwiseContract{successful, dimensions, value, left_kept, right_kept} = evidence
  PointCase{successful, dimensions, value(index, inside), left_kept(left_index, left_inside), right_kept(right_index, right_inside)}

def expose(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +left_index: U32, left_inside: {Nat.is_lt(U32.to_nat(left_index), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool},
  +right_index: U32, right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}) ->
  PointCase<result(operation, plan, left, right, left_fits, right_fits), Shape.dimensions(Shape.output_shape(plan)),
    index, Broadcast.cell(operation, plan, 0.0, writes(left), 0.0, writes(right), U32.to_nat(index)),
    left_index, Replayed.value(Shape.left_shape(plan), 0.0, writes(left), left_index),
    right_index, Replayed.value(Shape.right_shape(plan), 0.0, writes(right), right_index)>:
  open_case(operation, plan, left, right, left_fits, right_fits,
    Contracts.broadcast_pointwise(operation, plan, 0.0, writes(left), left_fits, 0.0, writes(right), right_fits),
    index, inside, left_index, left_inside, right_index, right_inside)

def open_empty(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Values(operation, plan, left, right, left_fits, right_fits)) ->
  EmptyCase<result(operation, plan, left, right, left_fits, right_fits), Shape.dimensions(Shape.output_shape(plan))>:
  Records.PointwiseContract{successful, dimensions, value, left_kept, right_kept} = evidence
  EmptyCase{successful, dimensions}

def expose_empty(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  EmptyCase<result(operation, plan, left, right, left_fits, right_fits), Shape.dimensions(Shape.output_shape(plan))>:
  open_empty(operation, plan, left, right, left_fits, right_fits,
    Contracts.broadcast_pointwise(operation, plan, 0.0, writes(left), left_fits, 0.0, writes(right), right_fits))

def grid_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def three_values() -> List<&2, F32>:
  [10.0, 20.0, 30.0]

def column_values() -> List<&2, F32>:
  [2.0, 3.0]

def sub_grid_first() -> PointCase<result(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}), [2, 3],
  0, F32.sub(1.0, 10.0), 0, 1.0, 0, 10.0>:
  expose(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}, 0, {==}, 0, {==}, 0, {==})

def sub_grid_middle() -> PointCase<result(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}), [2, 3],
  4, F32.sub(5.0, 20.0), 4, 5.0, 1, 20.0>:
  expose(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}, 4, {==}, 4, {==}, 1, {==})

def sub_grid_last() -> PointCase<result(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}), [2, 3],
  5, F32.sub(6.0, 30.0), 5, 6.0, 2, 30.0>:
  expose(Elementwise.Sub{}, grid_three(), grid_values(), three_values(), {==}, {==}, 5, {==}, 5, {==}, 2, {==})

def mul_outer_row() -> PointCase<result(Elementwise.Mul{}, column_row(), column_values(), three_values(), {==}, {==}), [2, 3],
  1, F32.mul(2.0, 20.0), 0, 2.0, 1, 20.0>:
  expose(Elementwise.Mul{}, column_row(), column_values(), three_values(), {==}, {==}, 1, {==}, 0, {==}, 1, {==})

def mul_outer_column() -> PointCase<result(Elementwise.Mul{}, column_row(), column_values(), three_values(), {==}, {==}), [2, 3],
  3, F32.mul(3.0, 10.0), 1, 3.0, 0, 10.0>:
  expose(Elementwise.Mul{}, column_row(), column_values(), three_values(), {==}, {==}, 3, {==}, 1, {==}, 0, {==})

def add_same() -> PointCase<result(Elementwise.Add{}, three_three(), [1.0, 2.0, 3.0], three_values(), {==}, {==}), [3],
  2, F32.add(3.0, 30.0), 1, 2.0, 2, 30.0>:
  expose(Elementwise.Add{}, three_three(), [1.0, 2.0, 3.0], three_values(), {==}, {==}, 2, {==}, 1, {==}, 2, {==})

def keep_scalar() -> PointCase<result(Elementwise.KeepLeft{}, scalar_three(), [7.0], three_values(), {==}, {==}), [3],
  2, 7.0, 0, 7.0, 2, 30.0>:
  expose(Elementwise.KeepLeft{}, scalar_three(), [7.0], three_values(), {==}, {==}, 2, {==}, 0, {==}, 2, {==})

def empty() -> EmptyCase<result(Elementwise.Add{}, empty_plan(), [], [], {==}, {==}), [2, 0]>:
  expose_empty(Elementwise.Add{}, empty_plan(), [], [], {==}, {==})

# The family on tensors that the public producers build. A producer failure
# leaves no inputs to broadcast.
def produced_right(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, left: Core.Tensor<Shape.left_shape(plan)>,
  right: Result<&2, &1, Core.Error, Core.Tensor<Shape.right_shape(plan)>>) ->
  Maybe<&1, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>>:
  match right:
    case Fail{error}:
      None{}
    case Done{tensor}:
      Some{Broadcast.binary(operation, plan, Core.Packed{Shape.left_shape(plan), left}, Core.Packed{Shape.right_shape(plan), tensor})}

def produced(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  left: Result<&2, &1, Core.Error, Core.Tensor<Shape.left_shape(plan)>>,
  right: Result<&2, &1, Core.Error, Core.Tensor<Shape.right_shape(plan)>>) ->
  Maybe<&1, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>>:
  match left:
    case Fail{error}:
      None{}
    case Done{tensor}:
      produced_right(operation, plan, tensor, right)

def produced_point(outcome: Maybe<&1, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>>, +index: U32) -> Maybe<&2, F32>:
  match outcome:
    case None{}:
      None{}
    case Some{result}:
      Pointwise.output_point(result, index)

# What the family promises for a produced pair; nothing when a producer fails.
def Produced(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  outcome: Maybe<&1, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>>) -> Type:
  match outcome:
    case None{}:
      Empty
    case Some{result}:
      Records.PointwiseContract<Shape.output_shape(plan), Shape.left_shape(plan), Shape.right_shape(plan),
        index => Broadcast.cell(operation, plan, left_fill, left_ops, right_fill, right_ops, index),
        other => Replayed.value(Shape.left_shape(plan), left_fill, left_ops, other),
        other => Replayed.value(Shape.right_shape(plan), right_fill, right_ops, other),
        result>

# `from_values_replayed` turns both `Core.from_values` results into replayed
# tensors, and `broadcast_pointwise` gives the contract of their broadcast.
def from_values_pointwise(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: List<&2, F32>, left_counted: {List.length(&2, F32, left) == Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  +right: List<&2, F32>, right_counted: {List.length(&2, F32, right) == Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  Produced(operation, plan, 0.0, writes(left), 0.0, writes(right),
    produced(operation, plan, Core.from_values(Shape.left_shape(plan), left), Core.from_values(Shape.right_shape(plan), right))):
  Contracts.from_values_replayed(Shape.left_shape(plan), left, left_counted,
    r => Produced(operation, plan, 0.0, writes(left), 0.0, writes(right),
      produced(operation, plan, r, Core.from_values(Shape.right_shape(plan), right))),
    left_fits => Contracts.from_values_replayed(Shape.right_shape(plan), right, right_counted,
      r => Produced(operation, plan, 0.0, writes(left), 0.0, writes(right),
        produced(operation, plan, Done{Core.TTensor{Replayed.storage(Shape.left_shape(plan), 0.0, writes(left)), left_fits}}, r)),
      right_fits => Contracts.broadcast_pointwise(operation, plan, 0.0, writes(left), left_fits, 0.0, writes(right), right_fits)))

# The same through `new_replayed`, for two `Core.new` fills.
def new_pointwise(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left_fill: F32, +right_fill: F32) ->
  Produced(operation, plan, left_fill, Nil{}, right_fill, Nil{},
    produced(operation, plan, Core.new(Shape.left_shape(plan), left_fill), Core.new(Shape.right_shape(plan), right_fill))):
  Contracts.new_replayed(Shape.left_shape(plan), left_fill,
    r => Produced(operation, plan, left_fill, Nil{}, right_fill, Nil{},
      produced(operation, plan, r, Core.new(Shape.right_shape(plan), right_fill))),
    left_fits => Contracts.new_replayed(Shape.right_shape(plan), right_fill,
      r => Produced(operation, plan, left_fill, Nil{}, right_fill, Nil{},
        produced(operation, plan, Done{Core.TTensor{Replayed.storage(Shape.left_shape(plan), left_fill, Nil{}), left_fits}}, r)),
      right_fits => Contracts.broadcast_pointwise(operation, plan, left_fill, Nil{}, left_fits, right_fill, Nil{}, right_fits)))

def produced_value(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  outcome: Maybe<&1, (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>>,
  evidence: Produced(operation, plan, left_fill, left_ops, right_fill, right_ops, outcome),
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {produced_point(outcome, index) ==
    Some{Broadcast.cell(operation, plan, left_fill, left_ops, right_fill, right_ops, U32.to_nat(index))} : Maybe<&2, F32>}:
  match outcome:
    case None{}:
      Empty.absurd({None{} == Some{Broadcast.cell(operation, plan, left_fill, left_ops, right_fill, right_ops, U32.to_nat(index))} : Maybe<&2, F32>}, evidence)
    case Some{result}:
      Records.PointwiseContract{successful, dimensions, value, left_kept, right_kept} = evidence
      value(index, inside)

# The kernel evaluates the produced inputs while it checks these, and runs out
# of fuel on the [2, 3] grid, so they use [3] - [3].
def from_values_sub() ->
  {produced_point(produced(Elementwise.Sub{}, three_three(), Core.from_values(three_shape(), [1.0, 2.0, 3.0]),
    Core.from_values(three_shape(), three_values())), 2) == Some{F32.sub(3.0, 30.0)} : Maybe<&2, F32>}:
  produced_value(Elementwise.Sub{}, three_three(), 0.0, writes([1.0, 2.0, 3.0]), 0.0, writes(three_values()),
    produced(Elementwise.Sub{}, three_three(), Core.from_values(three_shape(), [1.0, 2.0, 3.0]), Core.from_values(three_shape(), three_values())),
    from_values_pointwise(Elementwise.Sub{}, three_three(), [1.0, 2.0, 3.0], {==}, three_values(), {==}), 2, {==})

def new_sub() ->
  {produced_point(produced(Elementwise.Sub{}, three_three(), Core.new(three_shape(), 2.0), Core.new(three_shape(), 5.0)), 2) ==
    Some{F32.sub(2.0, 5.0)} : Maybe<&2, F32>}:
  produced_value(Elementwise.Sub{}, three_three(), 2.0, Nil{}, 5.0, Nil{},
    produced(Elementwise.Sub{}, three_three(), Core.new(three_shape(), 2.0), Core.new(three_shape(), 5.0)),
    new_pointwise(Elementwise.Sub{}, three_three(), 2.0, 5.0), 2, {==})
