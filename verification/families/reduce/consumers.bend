import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/layout.bend as Layout
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/from-values-contract.bend as FromValues
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/fold.bend as Fold
import ../../../src/autograd/reduce.bend as Reduce
import ../../../src/autograd/reduce-LAWS.bend as ReduceLaws
import ../../../src/autograd/reduce-fold.bend as ReduceFold
import ../../../src/autograd/reduce-fold-LAWS.bend as Laws
import ../../../src/autograd/reduce-PROOF.bend as ReduceProof
import ../../../src/autograd/reduce-fold-PROOF.bend as FoldProof
import ../../../src/autograd/reduce-to-fold-PROOF.bend as ReduceToProof
import ../../../src/autograd/sum-to-size.bend as SumToSize
import ../../../src/autograd/sum-to-size-fold-LAWS.bend as SumLaws
import ../../../src/autograd/sum-to-size-fold-PROOF.bend as SumProof

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def column_shape() -> Shape.Shape:
  Shape.SShape{[2, 1], 2n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0], 0n, {==}, Unit{}, Unit{}}

def empty_grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 0], 0n, {==}, Unit{}, Unit{}}

def plan_of(r: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  done: {Result.is_done(&2, &2, Shape.Error, Shape.BroadcastPlan, r) == True{} : Bool}) -> Shape.BroadcastPlan:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Shape.BroadcastPlan, Arith.clash(done))

def three_grid() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(three_shape(), grid_shape()), {==})

def column_grid() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(column_shape(), grid_shape()), {==})

def scalar_grid() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(scalar_shape(), grid_shape()), {==})

def grid_grid() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(grid_shape(), grid_shape()), {==})

def empty_plan() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(empty_shape(), empty_grid_shape()), {==})

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def result(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Core.Error, F32>:
  ReduceFold.cell(plan, index, input(Shape.right_shape(plan), values, fits))

def total(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>) -> Maybe<&2, F32>:
  ReduceLaws.fiber(plan,
    Contracts.words(Shape.right_shape(plan), 0.0, writes(values), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
    index, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))))

def Values(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) -> Type:
  Laws.FiberContract<Shape.right_shape(plan), total(plan, index, values),
    other => Replayed.value(Shape.right_shape(plan), 0.0, writes(values), other),
    result(plan, index, values, fits)>

type ReduceCase<-result: Core.Dynamic & Result<&2, &1, Core.Error, F32>, -total: Maybe<&2, F32>, -index: U32, -word: F32> is Data:
  ReduceCase{
    value_exact: {Fold.value(result) == total : Maybe<&2, F32>},
    kept_exact: {Fold.input_point(result, index) == Some{word} : Maybe<&2, F32>}}

def open_reduce(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Values(plan, index, values, fits),
  +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}) ->
  ReduceCase<result(plan, index, values, fits), total(plan, index, values),
    other, Replayed.value(Shape.right_shape(plan), 0.0, writes(values), other)>:
  Laws.FiberContract{value, kept} = evidence
  ReduceCase{value, kept(other, inside)}

def expose_reduce(+plan: Shape.BroadcastPlan,
  same: {Shape.extents(Shape.output_shape(plan)) == Shape.extents(Shape.right_shape(plan)) : List<&2, Nat>},
  +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}) ->
  ReduceCase<result(plan, index, values, fits), total(plan, index, values),
    other, Replayed.value(Shape.right_shape(plan), 0.0, writes(values), other)>:
  open_reduce(plan, index, values, fits, Laws.reduce_fold_fiber(plan, same, index, 0.0, writes(values), fits), other, inside)

def open_empty(+plan: Shape.BroadcastPlan, +index: U32, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), 0.0, writes(values))) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  evidence: Values(plan, index, values, fits)) ->
  {Fold.value(result(plan, index, values, fits)) == total(plan, index, values) : Maybe<&2, F32>}:
  Laws.FiberContract{value, kept} = evidence
  value

def grid_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def reduce_columns() -> ReduceCase<result(three_grid(), 1, grid_values(), {==}),
  Some{F32.add(F32.add(0.0, 2.0), 5.0)}, 4, 5.0>:
  expose_reduce(three_grid(), {==}, 1, grid_values(), {==}, 4, {==})

def reduce_rows() -> ReduceCase<result(column_grid(), 1, grid_values(), {==}),
  Some{F32.add(F32.add(F32.add(0.0, 4.0), 5.0), 6.0)}, 0, 1.0>:
  expose_reduce(column_grid(), {==}, 1, grid_values(), {==}, 0, {==})

def reduce_scalar() -> ReduceCase<result(scalar_grid(), 0, grid_values(), {==}),
  Some{F32.add(F32.add(F32.add(F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0), 4.0), 5.0), 6.0)}, 5, 6.0>:
  expose_reduce(scalar_grid(), {==}, 0, grid_values(), {==}, 5, {==})

def reduce_same() -> ReduceCase<result(grid_grid(), 4, grid_values(), {==}),
  Some{F32.add(0.0, 5.0)}, 2, 3.0>:
  expose_reduce(grid_grid(), {==}, 4, grid_values(), {==}, 2, {==})

def reduce_empty() -> {Fold.value(result(empty_plan(), 0, [], {==})) == Some{0.0} : Maybe<&2, F32>}:
  open_empty(empty_plan(), 0, [], {==}, Laws.reduce_fold_fiber(empty_plan(), {==}, 0, 0.0, writes([]), {==}))

def reduced(+target: Shape.Shape, +shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  Reduce.reduce_to(input(shape, values, fits), target)

type ReduceToCase<-result: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>, -dimensions: List<&2, U32>, -cell: U32,
  -total: Maybe<&2, F32>, -index: U32, -word: F32> is Data:
  ReduceToCase{
    dimensions_exact: {Laws.output_dimensions(result) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    value_exact: {Laws.output_point(result, cell) == total : Maybe<&2, F32>},
    kept_exact: {Laws.input_point(result, index) == Some{word} : Maybe<&2, F32>}}

def reduce_to_total(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +values: List<&2, F32>, +cell: U32) ->
  Maybe<&2, F32>:
  ReduceLaws.fiber(plan, Contracts.words(shape, 0.0, writes(values), Shape.logical_numel(Shape.dimensions(shape))),
    cell, Shape.logical_numel(Shape.dimensions(shape)))

def ReduceToValues(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) -> Type:
  Laws.ReduceToContract<target, shape, cell => reduce_to_total(plan, shape, values, cell),
    other => Replayed.value(shape, 0.0, writes(values), other), reduced(target, shape, values, fits)>

def open_reduce_to(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: ReduceToValues(target, shape, plan, values, fits),
  +cell: U32, cell_inside: {Nat.is_lt(U32.to_nat(cell), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  +other: U32, other_inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  ReduceToCase<reduced(target, shape, values, fits), Shape.dimensions(target), cell, reduce_to_total(plan, shape, values, cell),
    other, Replayed.value(shape, 0.0, writes(values), other)>:
  Laws.ReduceToContract{successful, dimensions, value, kept} = evidence
  ReduceToCase{dimensions, value(cell, cell_inside), kept(other, other_inside)}

def reduce_to_columns() -> ReduceToCase<reduced(three_shape(), grid_shape(), grid_values(), {==}), [3],
  1, Some{F32.add(F32.add(0.0, 2.0), 5.0)}, 4, 5.0>:
  open_reduce_to(three_shape(), grid_shape(), three_grid(), grid_values(), {==},
    Laws.reduce_to_fiber(three_shape(), grid_shape(), three_grid(), {==}, {==}, 0.0, writes(grid_values()), {==}), 1, {==}, 4, {==})

def reduce_to_rows() -> ReduceToCase<reduced(column_shape(), grid_shape(), grid_values(), {==}), [2, 1],
  1, Some{F32.add(F32.add(F32.add(0.0, 4.0), 5.0), 6.0)}, 0, 1.0>:
  open_reduce_to(column_shape(), grid_shape(), column_grid(), grid_values(), {==},
    Laws.reduce_to_fiber(column_shape(), grid_shape(), column_grid(), {==}, {==}, 0.0, writes(grid_values()), {==}), 1, {==}, 0, {==})

def summed(+target: Shape.Shape, +shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>:
  SumToSize.sum_to_size(input(shape, values, fits), target)

def open_summed(+target: Shape.Shape, +shape: Shape.Shape, +values: List<&2, F32>, -expected: U32 -> Maybe<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Laws.ReduceToContract<target, shape, expected, other => Replayed.value(shape, 0.0, writes(values), other),
    summed(target, shape, values, fits)>,
  +cell: U32, cell_inside: {Nat.is_lt(U32.to_nat(cell), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  +other: U32, other_inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  ReduceToCase<summed(target, shape, values, fits), Shape.dimensions(target), cell, expected(cell),
    other, Replayed.value(shape, 0.0, writes(values), other)>:
  Laws.ReduceToContract{successful, dimensions, value, kept} = evidence
  ReduceToCase{dimensions, value(cell, cell_inside), kept(other, other_inside)}

def kept_word(+shape: Shape.Shape, +values: List<&2, F32>, +cell: U32) -> Maybe<&2, F32>:
  Some{Replayed.value(shape, 0.0, writes(values), cell)}

# sum_to_size to the contribution's own dimensions clones it: cell 4 holds
# word 4 itself, not F32.add(0.0, 5.0) as reduce_to gives.
def sum_to_size_same() -> ReduceToCase<summed(grid_shape(), grid_shape(), grid_values(), {==}), [2, 3], 4, Some{5.0}, 2, 3.0>:
  open_summed(grid_shape(), grid_shape(), grid_values(), cell => kept_word(grid_shape(), grid_values(), cell), {==},
    SumLaws.sum_to_size_clone(grid_shape(), grid_shape(), {==}, 0.0, writes(grid_values()), {==}), 4, {==}, 2, {==})

def grid_not_three() -> {Shape.dimensions(grid_shape()) != Shape.dimensions(three_shape()) : List<&2, U32>}:
  equal => Arith.clash(Equal.cong(List<&2, U32>, Bool, dimensions => Layout.words_same(dimensions, [3]), [2, 3], [3], equal))

# sum_to_size to [3] reduces through reduce_to: cell 1 adds column 1.
def sum_to_size_columns() -> ReduceToCase<summed(three_shape(), grid_shape(), grid_values(), {==}), [3],
  1, Some{F32.add(F32.add(0.0, 2.0), 5.0)}, 4, 5.0>:
  open_summed(three_shape(), grid_shape(), grid_values(), cell => reduce_to_total(three_grid(), grid_shape(), grid_values(), cell), {==},
    SumLaws.sum_to_size_fiber(three_shape(), grid_shape(), three_grid(), {==}, {==}, grid_not_three(), 0.0, writes(grid_values()), {==}),
    1, {==}, 4, {==})
