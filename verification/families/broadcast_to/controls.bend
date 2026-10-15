import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/autograd/broadcast-pointwise-LAWS.bend as Laws
import ../../../src/autograd/broadcast-pointwise-PROOF.bend as Proof
import ./consumers.bend as Cases

def expected_columns() -> Maybe<&2, F32>:
  Some{2.0}

def first_row() -> Maybe<&2, F32>:
  Some{1.0}

def expected_rows() -> Maybe<&2, F32>:
  Some{3.0}

def row_read() -> Maybe<&2, F32>:
  Some{2.0}

def expected_keep() -> Maybe<&2, F32>:
  Some{7.0}

def added_keep() -> Maybe<&2, F32>:
  Some{F32.add(7.0, 0.0)}

def expected_kept() -> Maybe<&2, F32>:
  Some{3.0}

def zero_kept() -> Maybe<&2, F32>:
  Some{0.0}

def expected_chain() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, 7.0), 7.0)}

def single_chain() -> Maybe<&2, F32>:
  Some{F32.add(0.0, 7.0)}

def point(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, Cases.writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Cases.Values(shape, target, plan, values, fits), +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Laws.output_point(Cases.result(shape, target, values, fits), index) == Some{Cases.source(shape, plan, values, U32.to_nat(index))} : Maybe<&2, F32>}:
  Laws.BroadcastToContract{successful, dimensions, value, kept} = evidence
  value(index, inside)

def word(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, Cases.writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Cases.Values(shape, target, plan, values, fits), +other: U32,
  inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Laws.input_point(Cases.result(shape, target, values, fits), other) == Some{Replayed.value(shape, 0.0, Cases.writes(values), other)} : Maybe<&2, F32>}:
  Laws.BroadcastToContract{successful, dimensions, value, kept} = evidence
  kept(other, inside)

# The kernel reports a false twin's mismatch against the type the body has,
# where a law's value is still an unevaluated term. Each value witness
# restates it as the literal it evaluates to.
def stated(+claim: Maybe<&2, F32>, +value: Maybe<&2, F32>, equal: {claim == value : Maybe<&2, F32>}) ->
  {claim == value : Maybe<&2, F32>}:
  equal

def columns_witness() ->
  {Laws.output_point(Cases.result(Cases.column_shape(), Cases.grid_shape(), Cases.column_values(), {==}), 4) == expected_columns() : Maybe<&2, F32>}:
  stated(Laws.output_point(Cases.result(Cases.column_shape(), Cases.grid_shape(), Cases.column_values(), {==}), 4), Some{2.0},
    point(Cases.column_shape(), Cases.grid_shape(), Cases.column_grid(), Cases.column_values(), {==},
      Laws.broadcast_to_pointwise(Cases.column_shape(), Cases.grid_shape(), Cases.column_grid(), {==}, {==}, 0.0,
        Cases.writes(Cases.column_values()), {==}), 4, {==}))

def rows_witness() ->
  {Laws.output_point(Cases.result(Cases.three_shape(), Cases.grid_shape(), Cases.three_values(), {==}), 5) == expected_rows() : Maybe<&2, F32>}:
  stated(Laws.output_point(Cases.result(Cases.three_shape(), Cases.grid_shape(), Cases.three_values(), {==}), 5), Some{3.0},
    point(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), Cases.three_values(), {==},
      Laws.broadcast_to_pointwise(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), {==}, {==}, 0.0,
        Cases.writes(Cases.three_values()), {==}), 5, {==}))

def keep_witness() ->
  {Laws.output_point(Cases.result(Cases.scalar_shape(), Cases.square_shape(), [7.0], {==}), 3) == expected_keep() : Maybe<&2, F32>}:
  stated(Laws.output_point(Cases.result(Cases.scalar_shape(), Cases.square_shape(), [7.0], {==}), 3), Some{7.0},
    point(Cases.scalar_shape(), Cases.square_shape(), Cases.scalar_square(), [7.0], {==},
      Laws.broadcast_to_pointwise(Cases.scalar_shape(), Cases.square_shape(), Cases.scalar_square(), {==}, {==}, 0.0,
        Cases.writes([7.0]), {==}), 3, {==}))

def kept_witness() ->
  {Laws.input_point(Cases.result(Cases.three_shape(), Cases.grid_shape(), Cases.three_values(), {==}), 2) == expected_kept() : Maybe<&2, F32>}:
  stated(Laws.input_point(Cases.result(Cases.three_shape(), Cases.grid_shape(), Cases.three_values(), {==}), 2), Some{3.0},
    word(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), Cases.three_values(), {==},
      Laws.broadcast_to_pointwise(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), {==}, {==}, 0.0,
        Cases.writes(Cases.three_values()), {==}), 2, {==}))

def chain_witness() ->
  {Cases.chained(0) == expected_chain() : Maybe<&2, F32>}:
  stated(Cases.chained(0), Some{F32.add(F32.add(0.0, 7.0), 7.0)}, Cases.chain_reduced(0, {==}))
