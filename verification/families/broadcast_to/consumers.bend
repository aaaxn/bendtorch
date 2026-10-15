import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/from-values-contract.bend as FromValues
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/broadcast.bend as PointwiseBroadcast
import ../../../src/autograd/broadcast.bend as Broadcast
import ../../../src/autograd/broadcast-pointwise-LAWS.bend as Laws
import ../../../src/autograd/reduce.bend as Reduce
import ../../../src/autograd/reduce-LAWS.bend as ReduceLaws
import ../../../src/autograd/reduce-fold-LAWS.bend as ReduceFold
import ../../../src/autograd/reduce-PROOF.bend as ReduceProof
import ../../../src/autograd/reduce-fold-PROOF.bend as FoldProof
import ../../../src/autograd/reduce-to-fold-PROOF.bend as ReduceToProof
import ../../../src/autograd/broadcast-pointwise-PROOF.bend as Proof

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def one_shape() -> Shape.Shape:
  Shape.SShape{[1], 1n, {==}, Unit{}, Unit{}}

def two_shape() -> Shape.Shape:
  Shape.SShape{[2], 2n, {==}, Unit{}, Unit{}}

def three_shape() -> Shape.Shape:
  Shape.SShape{[3], 3n, {==}, Unit{}, Unit{}}

def column_shape() -> Shape.Shape:
  Shape.SShape{[2, 1], 2n, {==}, Unit{}, Unit{}}

def row_shape() -> Shape.Shape:
  Shape.SShape{[1, 3], 3n, {==}, Unit{}, Unit{}}

def square_shape() -> Shape.Shape:
  Shape.SShape{[2, 2], 4n, {==}, Unit{}, Unit{}}

def grid_shape() -> Shape.Shape:
  Shape.SShape{[2, 3], 6n, {==}, Unit{}, Unit{}}

def empty_shape() -> Shape.Shape:
  Shape.SShape{[0, 3], 0n, {==}, Unit{}, Unit{}}

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

def scalar_square() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(scalar_shape(), square_shape()), {==})

def one_two() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), two_shape()), {==})

def row_empty() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(row_shape(), empty_shape()), {==})

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def result(+shape: Shape.Shape, +target: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>:
  Broadcast.broadcast_to(input(shape, values, fits), target)

def source(+shape: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>, +index: Nat) -> F32:
  Replayed.value(shape, 0.0, writes(values), U32.from_nat(PointwiseBroadcast.left_map(plan, index)))

def Values(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) -> Type:
  Laws.BroadcastToContract<target, shape, index => source(shape, plan, values, index),
    other => Replayed.value(shape, 0.0, writes(values), other), result(shape, target, values, fits)>

type BroadcastCase<-result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>, -dimensions: List<&2, U32>,
  -index: U32, -expected: F32, -other: U32, -word: F32> is Data:
  BroadcastCase{
    successful: {Laws.succeeded(result) == True{} : Bool},
    dimensions_exact: {Laws.output_dimensions(result) == Some{dimensions} : Maybe<&2, List<&2, U32>>},
    value_exact: {Laws.output_point(result, index) == Some{expected} : Maybe<&2, F32>},
    kept_exact: {Laws.input_point(result, other) == Some{word} : Maybe<&2, F32>}}

type EmptyCase<-result: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>, -dimensions: List<&2, U32>> is Data:
  EmptyCase{
    successful: {Laws.succeeded(result) == True{} : Bool},
    dimensions_exact: {Laws.output_dimensions(result) == Some{dimensions} : Maybe<&2, List<&2, U32>>}}

def open_case(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Values(shape, target, plan, values, fits),
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  +other: U32, other_inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  BroadcastCase<result(shape, target, values, fits), Shape.dimensions(target), index, source(shape, plan, values, U32.to_nat(index)),
    other, Replayed.value(shape, 0.0, writes(values), other)>:
  Laws.BroadcastToContract{successful, dimensions, value, kept} = evidence
  BroadcastCase{successful, dimensions, value(index, inside), kept(other, other_inside)}

def expose(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool},
  +other: U32, other_inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  BroadcastCase<result(shape, target, values, fits), Shape.dimensions(target), index, source(shape, plan, values, U32.to_nat(index)),
    other, Replayed.value(shape, 0.0, writes(values), other)>:
  open_case(shape, target, plan, values, fits, Laws.broadcast_to_pointwise(shape, target, plan, planned, same, 0.0, writes(values), fits),
    index, inside, other, other_inside)

def open_empty(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  evidence: Values(shape, target, plan, values, fits)) ->
  EmptyCase<result(shape, target, values, fits), Shape.dimensions(target)>:
  Laws.BroadcastToContract{successful, dimensions, value, kept} = evidence
  EmptyCase{successful, dimensions}

def three_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0]

def column_values() -> List<&2, F32>:
  [1.0, 2.0]

# [3] to [2, 3]: index 4 is row 1, column 1, so it holds input word 1.
def rows_repeat() -> BroadcastCase<result(three_shape(), grid_shape(), three_values(), {==}), [2, 3], 4, 2.0, 2, 3.0>:
  expose(three_shape(), grid_shape(), three_grid(), {==}, {==}, three_values(), {==}, 4, {==}, 2, {==})

# [2, 1] to [2, 3]: index 4 is row 1, so it holds input word 1.
def columns_repeat() -> BroadcastCase<result(column_shape(), grid_shape(), column_values(), {==}), [2, 3], 4, 2.0, 0, 1.0>:
  expose(column_shape(), grid_shape(), column_grid(), {==}, {==}, column_values(), {==}, 4, {==}, 0, {==})

# A scalar fills [2, 2].
def scalar_fills() -> BroadcastCase<result(scalar_shape(), square_shape(), [7.0], {==}), [2, 2], 3, 7.0, 0, 7.0>:
  expose(scalar_shape(), square_shape(), scalar_square(), {==}, {==}, [7.0], {==}, 3, {==}, 0, {==})

# KeepLeft copies the word itself, so -0.0 stays -0.0 instead of becoming
# F32.add(-0.0, 0.0).
def negative_zero_kept() -> BroadcastCase<result(one_shape(), two_shape(), [F32.neg(0.0)], {==}), [2], 1, F32.neg(0.0), 0, F32.neg(0.0)>:
  expose(one_shape(), two_shape(), one_two(), {==}, {==}, [F32.neg(0.0)], {==}, 1, {==}, 0, {==})

# [1, 3] to [0, 3] succeeds with an empty [0, 3] tensor.
def empty() -> EmptyCase<result(row_shape(), empty_shape(), three_values(), {==}), [0, 3]>:
  open_empty(row_shape(), empty_shape(), row_empty(), three_values(), {==},
    Laws.broadcast_to_pointwise(row_shape(), empty_shape(), row_empty(), {==}, {==}, 0.0, writes(three_values()), {==}))

# broadcast_to then reduce_to through laws alone: broadcast_to_replayed makes
# the output a replayed tensor, so reduce_to_fiber applies to it. Reducing
# [7] broadcast to [2] back to [1] adds the two copies from +0.0. BendTT
# evaluates the broadcast history while it checks this consumer, and on a
# [2, 3] grid it runs out of fuel, so the chain uses [2].
def reduced_point(pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>, +target: Shape.Shape, +index: U32) -> Maybe<&2, F32>:
  (input, outcome) = pair
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      ReduceFold.output_point(Reduce.reduce_to(output, target), index)

def chained(+index: U32) -> Maybe<&2, F32>:
  reduced_point(result(one_shape(), two_shape(), [7.0], {==}), one_shape(), index)

def back_plan() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), Shape.output_shape(one_two())), {==})

def broadcast_writes() -> List<&2, StorageLaws.Op>:
  Laws.writes(one_two(), one_shape(), 0.0, writes([7.0]), two_shape(), Shape.logical_numel(Shape.dimensions(Shape.output_shape(one_two()))))

def chain_total(+index: U32) -> Maybe<&2, F32>:
  +output = Shape.output_shape(one_two())
  +count = Shape.logical_numel(Shape.dimensions(output))
  ReduceLaws.fiber(back_plan(), Contracts.words(output, 0.0, broadcast_writes(), count), index, count)

def chain_reduced(+index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(one_shape()))) == True{} : Bool}) ->
  {chained(index) == chain_total(index) : Maybe<&2, F32>}:
  +output = Shape.output_shape(one_two())
  +count = Shape.logical_numel(Shape.dimensions(output))
  Laws.broadcast_to_replayed(one_shape(), two_shape(), one_two(), {==}, {==}, 0.0, writes([7.0]), {==},
    pair => {reduced_point(pair, one_shape(), index) == chain_total(index) : Maybe<&2, F32>},
    input_fits => output_fits =>
      {\{ReduceFold.ReduceToContract: successful => dimensions => value => kept => value(index, inside)} :
        ReduceFold.ReduceToContract<one_shape(), output,
          cell => ReduceLaws.fiber(back_plan(), Contracts.words(output, 0.0, broadcast_writes(), count), cell, count),
          other => Replayed.value(output, 0.0, broadcast_writes(), other),
          Reduce.reduce_to(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, broadcast_writes()), output_fits}}, one_shape())> ->
        {ReduceFold.output_point(Reduce.reduce_to(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, broadcast_writes()), output_fits}},
          one_shape()), index) == chain_total(index) : Maybe<&2, F32>}}(
        ReduceFold.reduce_to_fiber(one_shape(), output, back_plan(), {==}, {==}, 0.0, broadcast_writes(), output_fits)))

def broadcast_then_reduce() -> {chained(0) == Some{F32.add(F32.add(0.0, 7.0), 7.0)} : Maybe<&2, F32>}:
  chain_reduced(0, {==})
