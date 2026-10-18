import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/elementwise.bend as Elementwise
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/PROOF.bend as ContractsProof
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/broadcast.bend as Broadcast
import ../../../src/tensor/contracts/pointwise.bend as Pointwise
import ../../../src/tensor/contracts/pointwise_records.bend as Records
import ./consumers.bend as Cases

def expected_sub() -> Maybe<&2, F32>:
  Some{F32.sub(5.0, 20.0)}

def swapped_sub() -> Maybe<&2, F32>:
  Some{F32.sub(20.0, 5.0)}

def expected_outer() -> Maybe<&2, F32>:
  Some{F32.mul(3.0, 30.0)}

def unbroadcast_outer() -> Maybe<&2, F32>:
  Some{F32.mul(2.0, 30.0)}

def expected_keep() -> Maybe<&2, F32>:
  Some{7.0}

def wrong_keep() -> Maybe<&2, F32>:
  Some{8.0}

def expected_kept() -> Maybe<&2, F32>:
  Some{5.0}

def overwritten_kept() -> Maybe<&2, F32>:
  Some{F32.sub(5.0, 20.0)}

def point(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, Cases.writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, Cases.writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Cases.Values(operation, plan, left, right, left_fits, right_fits), +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Pointwise.output_point(Cases.result(operation, plan, left, right, left_fits, right_fits), index) ==
    Some{Broadcast.cell(operation, plan, 0.0, Cases.writes(left), 0.0, Cases.writes(right), U32.to_nat(index))} : Maybe<&2, F32>}:
  Records.PointwiseContract{successful, dimensions, value, left_kept, right_kept} = evidence
  value(index, inside)

def left_word(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan, +left: List<&2, F32>, +right: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(Shape.left_shape(plan), 0.0, Cases.writes(left))) ==
    Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan))) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, Cases.writes(right))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Cases.Values(operation, plan, left, right, left_fits, right_fits), +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.left_shape(plan)))) == True{} : Bool}) ->
  {Pointwise.left_point(Cases.result(operation, plan, left, right, left_fits, right_fits), index) ==
    Some{Replayed.value(Shape.left_shape(plan), 0.0, Cases.writes(left), index)} : Maybe<&2, F32>}:
  Records.PointwiseContract{successful, dimensions, value, left_kept, right_kept} = evidence
  left_kept(index, inside)

def sub_witness() ->
  {Pointwise.output_point(Cases.result(Elementwise.Sub{}, Cases.grid_three(), Cases.grid_values(), Cases.three_values(), {==}, {==}), 4) == expected_sub() : Maybe<&2, F32>}:
  point(Elementwise.Sub{}, Cases.grid_three(), Cases.grid_values(), Cases.three_values(), {==}, {==},
    Contracts.broadcast_pointwise(Elementwise.Sub{}, Cases.grid_three(), 0.0, Cases.writes(Cases.grid_values()), {==},
      0.0, Cases.writes(Cases.three_values()), {==}), 4, {==})

def outer_witness() ->
  {Pointwise.output_point(Cases.result(Elementwise.Mul{}, Cases.column_row(), Cases.column_values(), Cases.three_values(), {==}, {==}), 5) == expected_outer() : Maybe<&2, F32>}:
  point(Elementwise.Mul{}, Cases.column_row(), Cases.column_values(), Cases.three_values(), {==}, {==},
    Contracts.broadcast_pointwise(Elementwise.Mul{}, Cases.column_row(), 0.0, Cases.writes(Cases.column_values()), {==},
      0.0, Cases.writes(Cases.three_values()), {==}), 5, {==})

def keep_witness() ->
  {Pointwise.output_point(Cases.result(Elementwise.KeepLeft{}, Cases.scalar_three(), [7.0], Cases.three_values(), {==}, {==}), 1) == expected_keep() : Maybe<&2, F32>}:
  point(Elementwise.KeepLeft{}, Cases.scalar_three(), [7.0], Cases.three_values(), {==}, {==},
    Contracts.broadcast_pointwise(Elementwise.KeepLeft{}, Cases.scalar_three(), 0.0, Cases.writes([7.0]), {==},
      0.0, Cases.writes(Cases.three_values()), {==}), 1, {==})

def kept_witness() ->
  {Pointwise.left_point(Cases.result(Elementwise.Sub{}, Cases.grid_three(), Cases.grid_values(), Cases.three_values(), {==}, {==}), 4) == expected_kept() : Maybe<&2, F32>}:
  left_word(Elementwise.Sub{}, Cases.grid_three(), Cases.grid_values(), Cases.three_values(), {==}, {==},
    Contracts.broadcast_pointwise(Elementwise.Sub{}, Cases.grid_three(), 0.0, Cases.writes(Cases.grid_values()), {==},
      0.0, Cases.writes(Cases.three_values()), {==}), 4, {==})

def expected_produced() -> Maybe<&2, F32>:
  Some{F32.sub(3.0, 30.0)}

def swapped_produced() -> Maybe<&2, F32>:
  Some{F32.sub(30.0, 3.0)}

def produced_witness() ->
  {Cases.produced_point(Cases.produced(Elementwise.Sub{}, Cases.three_three(), Core.from_values(Cases.three_shape(), [1.0, 2.0, 3.0]),
    Core.from_values(Cases.three_shape(), Cases.three_values())), 2) == expected_produced() : Maybe<&2, F32>}:
  Cases.produced_value(Elementwise.Sub{}, Cases.three_three(), 0.0, Cases.writes([1.0, 2.0, 3.0]), 0.0, Cases.writes(Cases.three_values()),
    Cases.produced(Elementwise.Sub{}, Cases.three_three(), Core.from_values(Cases.three_shape(), [1.0, 2.0, 3.0]),
      Core.from_values(Cases.three_shape(), Cases.three_values())),
    Cases.from_values_pointwise(Elementwise.Sub{}, Cases.three_three(), [1.0, 2.0, 3.0], {==}, Cases.three_values(), {==}), 2, {==})
