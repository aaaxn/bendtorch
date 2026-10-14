import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/fold.bend as Fold
import ../../../src/autograd/reduce-fold-LAWS.bend as Laws
import ../../../src/autograd/reduce-fold-PROOF.bend as FoldProof
import ../../../src/autograd/reduce-to-fold-PROOF.bend as ReduceToProof
import ./consumers.bend as Cases

def expected_column() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(0.0, 2.0), 5.0)}

def unmasked_column() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0), 4.0), 5.0), 6.0)}

def expected_row() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, 4.0), 5.0), 6.0)}

def first_row() -> Maybe<&2, F32>:
  Some{F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)}

def expected_same() -> Maybe<&2, F32>:
  Some{F32.add(0.0, 5.0)}

def uncopied_same() -> Maybe<&2, F32>:
  Some{5.0}

def expected_word() -> Maybe<&2, F32>:
  Some{5.0}

def shifted_word() -> Maybe<&2, F32>:
  Some{6.0}

def total(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, Cases.writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Cases.Values(plan, index, values, fits)) ->
  {Fold.value(Cases.result(plan, index, values, fits)) == Cases.total(plan, index, values) : Maybe<&2, F32>}:
  Laws.FiberContract{value, kept} = evidence
  value

def word(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, Cases.writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Cases.Values(plan, index, values, fits),
  +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}) ->
  {Fold.input_point(Cases.result(plan, index, values, fits), other) ==
    Some{Replayed.value(Shape.right_shape(plan), 0.0, Cases.writes(values), other)} : Maybe<&2, F32>}:
  Laws.FiberContract{value, kept} = evidence
  kept(other, inside)

# The kernel reports a false twin's mismatch against the type the body has, and
# a law's total stays an unevaluated fiber there. Each value witness restates
# it as the literal it evaluates to.
def stated(+claim: Maybe<&2, F32>, +value: Maybe<&2, F32>, equal: {claim == value : Maybe<&2, F32>}) ->
  {claim == value : Maybe<&2, F32>}:
  equal

def column_witness() ->
  {Fold.value(Cases.result(Cases.three_grid(), 1, Cases.grid_values(), {==})) == expected_column() : Maybe<&2, F32>}:
  stated(Fold.value(Cases.result(Cases.three_grid(), 1, Cases.grid_values(), {==})), Some{F32.add(F32.add(0.0, 2.0), 5.0)},
    total(Cases.three_grid(), 1, Cases.grid_values(), {==},
      Laws.reduce_fold_fiber(Cases.three_grid(), {==}, 1, 0.0, Cases.writes(Cases.grid_values()), {==})))

def row_witness() ->
  {Fold.value(Cases.result(Cases.column_grid(), 1, Cases.grid_values(), {==})) == expected_row() : Maybe<&2, F32>}:
  stated(Fold.value(Cases.result(Cases.column_grid(), 1, Cases.grid_values(), {==})), Some{F32.add(F32.add(F32.add(0.0, 4.0), 5.0), 6.0)},
    total(Cases.column_grid(), 1, Cases.grid_values(), {==},
      Laws.reduce_fold_fiber(Cases.column_grid(), {==}, 1, 0.0, Cases.writes(Cases.grid_values()), {==})))

def same_witness() ->
  {Fold.value(Cases.result(Cases.grid_grid(), 4, Cases.grid_values(), {==})) == expected_same() : Maybe<&2, F32>}:
  stated(Fold.value(Cases.result(Cases.grid_grid(), 4, Cases.grid_values(), {==})), Some{F32.add(0.0, 5.0)},
    total(Cases.grid_grid(), 4, Cases.grid_values(), {==},
      Laws.reduce_fold_fiber(Cases.grid_grid(), {==}, 4, 0.0, Cases.writes(Cases.grid_values()), {==})))

def word_witness() ->
  {Fold.input_point(Cases.result(Cases.three_grid(), 1, Cases.grid_values(), {==}), 4) == expected_word() : Maybe<&2, F32>}:
  word(Cases.three_grid(), 1, Cases.grid_values(), {==},
    Laws.reduce_fold_fiber(Cases.three_grid(), {==}, 1, 0.0, Cases.writes(Cases.grid_values()), {==}), 4, {==})

def reduced_total(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, Cases.writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Cases.ReduceToValues(target, shape, plan, values, fits),
  +cell: U32, inside: {Nat.is_lt(U32.to_nat(cell), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Laws.output_point(Cases.reduced(target, shape, values, fits), cell) == Cases.reduce_to_total(plan, shape, values, cell) : Maybe<&2, F32>}:
  Laws.ReduceToContract{successful, dimensions, value, kept} = evidence
  value(cell, inside)

def reduce_to_witness() ->
  {Laws.output_point(Cases.reduced(Cases.three_shape(), Cases.grid_shape(), Cases.grid_values(), {==}), 1) == expected_column() : Maybe<&2, F32>}:
  stated(Laws.output_point(Cases.reduced(Cases.three_shape(), Cases.grid_shape(), Cases.grid_values(), {==}), 1),
    Some{F32.add(F32.add(0.0, 2.0), 5.0)},
    reduced_total(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), Cases.grid_values(), {==},
      Laws.reduce_to_fiber(Cases.three_shape(), Cases.grid_shape(), Cases.three_grid(), {==}, {==}, 0.0, Cases.writes(Cases.grid_values()), {==}),
      1, {==}))
