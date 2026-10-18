import Base
import ../../../src/shape/core.bend as Shape
import ../../../src/storage/arith.bend as Arith
import ../../../src/storage/LAWS.bend as StorageLaws
import ../../../src/tensor/core.bend as Core
import ../../../src/tensor/lifecycle.bend as Lifecycle
import ../../../src/tensor/elementwise.bend as Elementwise
import ../../../src/tensor/from_values_contract.bend as FromValues
import ../../../src/tensor/contracts/replayed.bend as Replayed
import ../../../src/tensor/contracts/LAWS.bend as Contracts
import ../../../src/tensor/contracts/pointwise_proof.bend as PointwiseProof
import ../../../src/tensor/contracts/binary_LAWS.bend as Laws
import ../../../src/tensor/contracts/binary_PROOF.bend as Proof
import ../../../src/autograd/reduce.bend as Reduce
import ../../../src/autograd/reduce_LAWS.bend as ReduceLaws
import ../../../src/autograd/reduce_fold_LAWS.bend as ReduceFold
import ../../../src/autograd/reduce_PROOF.bend as ReduceProof
import ../../../src/autograd/reduce_fold_PROOF.bend as FoldProof
import ../../../src/autograd/reduce_to_fold_PROOF.bend as ReduceToProof

def scalar_shape() -> Shape.Shape:
  Shape.SShape{[], 1n, {==}, Unit{}, Unit{}}

def one_shape() -> Shape.Shape:
  Shape.SShape{[1], 1n, {==}, Unit{}, Unit{}}

def two_shape() -> Shape.Shape:
  Shape.SShape{[2], 2n, {==}, Unit{}, Unit{}}

def plan_of(r: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>,
  done: {Result.is_done(&2, &2, Shape.Error, Shape.BroadcastPlan, r) == True{} : Bool}) -> Shape.BroadcastPlan:
  match r:
    case Done{plan}:
      plan
    case Fail{error}:
      Empty.absurd(Shape.BroadcastPlan, Arith.clash(done))

def two_one() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), one_shape()), {==})

def one_two() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), two_shape()), {==})

def two_scalar() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(two_shape(), scalar_shape()), {==})

def writes(values: List<&2, F32>) -> List<&2, StorageLaws.Op>:
  FromValues.writes(values, 0n)

def input(+shape: Shape.Shape, +values: List<&2, F32>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, 0.0, writes(values))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Core.Dynamic:
  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, writes(values)), fits}}

def result(+operation: Elementwise.Operation, +left: Shape.Shape, +left_values: List<&2, F32>, +right: Shape.Shape, +right_values: List<&2, F32>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, writes(left_values))) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, writes(right_values))) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>:
  Elementwise.binary(operation, input(left, left_values, left_fits), input(right, right_values, right_fits))

def output_point(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (inputs, outcome) = pair
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      Core.dynamic_read_value(Core.read_dynamic(output, index))

def left_point(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (inputs, outcome) = pair
  (left, right) = inputs
  Core.dynamic_read_value(Core.read_dynamic(left, index))

def right_point(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +index: U32) -> Maybe<&2, F32>:
  (inputs, outcome) = pair
  (left, right) = inputs
  Core.dynamic_read_value(Core.read_dynamic(right, index))

# The word of a list at a position, +0.0 past its end.
def pick(values: List<&2, F32>, +position: Nat) -> F32:
  match values:
    case Nil{}:
      0.0
    case value <> rest:
      match position:
        case 0n:
          value
        case 1n+later:
          pick(rest, later)

def picked(+left: List<&2, F32>, +right: List<&2, F32>, +position: Nat, same: {left == right : List<&2, F32>}) ->
  {pick(left, position) == pick(right, position) : F32}:
  Equal.cong(List<&2, F32>, F32, values => pick(values, position), left, right, same)

def some_word(-word: F32, -other: F32, same: {word == other : F32}) -> {Some{word} == Some{other} : Maybe<&2, F32>}:
  Equal.cong(F32, Maybe<&2, F32>, value => Some{value}, word, other, same)

# One output cell and one cell of each returned input, at literal indices.
type BinaryCase<-pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>,
  -index: U32, -word: F32, -left_index: U32, -left_word: F32, -right_index: U32, -right_word: F32> is Data:
  BinaryCase{
    value: {output_point(pair, index) == Some{word} : Maybe<&2, F32>},
    left_kept: {left_point(pair, left_index) == Some{left_word} : Maybe<&2, F32>},
    right_kept: {right_point(pair, right_index) == Some{right_word} : Maybe<&2, F32>}}

# What one replayed result of the law gives for a case: the output read at
# `index` is the replayed word, which `values` names in the `cells` list, and
# each returned input read is its replayed word, which its words list names.
def replayed_case(+operation: Elementwise.Operation, +plan: Shape.BroadcastPlan,
  +left: Shape.Shape, +left_values: List<&2, F32>, +right: Shape.Shape, +right_values: List<&2, F32>,
  +left_kept: List<&2, StorageLaws.Op>, +right_kept: List<&2, StorageLaws.Op>, +ops: List<&2, StorageLaws.Op>,
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, 0.0, left_kept)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, 0.0, right_kept)) == Shape.logical_numel(Shape.dimensions(right)) : Nat},
  -output_fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(plan), 0.0, ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat},
  +index: U32, +word: F32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  value: {Replayed.value(Shape.output_shape(plan), 0.0, ops, index) == word : F32},
  +left_index: U32, +left_word: F32, left_inside: {Nat.is_lt(U32.to_nat(left_index), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  left_value: {Replayed.value(left, 0.0, left_kept, left_index) == left_word : F32},
  +right_index: U32, +right_word: F32, right_inside: {Nat.is_lt(U32.to_nat(right_index), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  right_value: {Replayed.value(right, 0.0, right_kept, right_index) == right_word : F32}) ->
  BinaryCase<((Core.Packed{left, Core.TTensor{Replayed.storage(left, 0.0, left_kept), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, 0.0, right_kept), right_fits}}),
      Done{Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0, ops), output_fits}}}),
    index, word, left_index, left_word, right_index, right_word>:
  BinaryCase{
    Equal.trans(Maybe<&2, F32>,
      Core.dynamic_read_value(Core.read_dynamic(Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0, ops), output_fits}}, index)),
      Some{Replayed.value(Shape.output_shape(plan), 0.0, ops, index)}, Some{word},
      PointwiseProof.replayed_point(Shape.output_shape(plan), 0.0, ops, output_fits, index, inside),
      some_word(Replayed.value(Shape.output_shape(plan), 0.0, ops, index), word, value)),
    Equal.trans(Maybe<&2, F32>,
      Core.dynamic_read_value(Core.read_dynamic(Core.Packed{left, Core.TTensor{Replayed.storage(left, 0.0, left_kept), left_fits}}, left_index)),
      Some{Replayed.value(left, 0.0, left_kept, left_index)}, Some{left_word},
      PointwiseProof.replayed_point(left, 0.0, left_kept, left_fits, left_index, left_inside),
      some_word(Replayed.value(left, 0.0, left_kept, left_index), left_word, left_value)),
    Equal.trans(Maybe<&2, F32>,
      Core.dynamic_read_value(Core.read_dynamic(Core.Packed{right, Core.TTensor{Replayed.storage(right, 0.0, right_kept), right_fits}}, right_index)),
      Some{Replayed.value(right, 0.0, right_kept, right_index)}, Some{right_word},
      PointwiseProof.replayed_point(right, 0.0, right_kept, right_fits, right_index, right_inside),
      some_word(Replayed.value(right, 0.0, right_kept, right_index), right_word, right_value))}

def gradient() -> List<&2, F32>:
  [3.0, 5.0]

# [2] * [1]: cell 1 is 5.0 * 10.0, and both inputs come back with their words.
def mul_broadcast() -> BinaryCase<result(Elementwise.Mul{}, two_shape(), gradient(), one_shape(), [10.0], {==}, {==}),
  1, F32.mul(5.0, 10.0), 1, 5.0, 0, 10.0>:
  Laws.binary_replayed(Elementwise.Mul{}, two_shape(), one_shape(), two_one(), {==}, 0.0, writes(gradient()), {==}, 0.0, writes([10.0]), {==},
    pair => BinaryCase<pair, 1, F32.mul(5.0, 10.0), 1, 5.0, 0, 10.0>,
    left_kept => left_words => left_fits => right_kept => right_words => right_fits => ops => values => output_fits =>
      replayed_case(Elementwise.Mul{}, two_one(), two_shape(), gradient(), one_shape(), [10.0], left_kept, right_kept, ops,
        left_fits, right_fits, output_fits,
        1, F32.mul(5.0, 10.0), {==}, picked(Contracts.words(Shape.output_shape(two_one()), 0.0, ops, 2n), Laws.cells(Elementwise.Mul{}, two_one(), two_shape(), 0.0, writes(gradient()), one_shape(), 0.0, writes([10.0]), 2n), 1n, values),
        1, 5.0, {==}, picked(Contracts.words(two_shape(), 0.0, left_kept, 2n), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), 1n, left_words),
        0, 10.0, {==}, picked(Contracts.words(one_shape(), 0.0, right_kept, 1n), Contracts.words(one_shape(), 0.0, writes([10.0]), 1n), 0n, right_words)))

# [1] - [2]: cell 1 is 7.0 - 3.0, the left word broadcast against right word 1.
def sub_broadcast() -> BinaryCase<result(Elementwise.Sub{}, one_shape(), [7.0], two_shape(), [2.0, 3.0], {==}, {==}),
  1, F32.sub(7.0, 3.0), 0, 7.0, 1, 3.0>:
  Laws.binary_replayed(Elementwise.Sub{}, one_shape(), two_shape(), one_two(), {==}, 0.0, writes([7.0]), {==}, 0.0, writes([2.0, 3.0]), {==},
    pair => BinaryCase<pair, 1, F32.sub(7.0, 3.0), 0, 7.0, 1, 3.0>,
    left_kept => left_words => left_fits => right_kept => right_words => right_fits => ops => values => output_fits =>
      replayed_case(Elementwise.Sub{}, one_two(), one_shape(), [7.0], two_shape(), [2.0, 3.0], left_kept, right_kept, ops,
        left_fits, right_fits, output_fits,
        1, F32.sub(7.0, 3.0), {==}, picked(Contracts.words(Shape.output_shape(one_two()), 0.0, ops, 2n), Laws.cells(Elementwise.Sub{}, one_two(), one_shape(), 0.0, writes([7.0]), two_shape(), 0.0, writes([2.0, 3.0]), 2n), 1n, values),
        0, 7.0, {==}, picked(Contracts.words(one_shape(), 0.0, left_kept, 1n), Contracts.words(one_shape(), 0.0, writes([7.0]), 1n), 0n, left_words),
        1, 3.0, {==}, picked(Contracts.words(two_shape(), 0.0, right_kept, 2n), Contracts.words(two_shape(), 0.0, writes([2.0, 3.0]), 2n), 1n, right_words)))

# NegateLeft against a rank-0 +0.0, as the Sub VJP runs it: cell 0 is
# F32.neg(3.0), the sign flipped.
def negate_scalar() -> BinaryCase<result(Elementwise.NegateLeft{}, two_shape(), gradient(), scalar_shape(), [0.0], {==}, {==}),
  0, F32.neg(3.0), 1, 5.0, 0, 0.0>:
  Laws.binary_replayed(Elementwise.NegateLeft{}, two_shape(), scalar_shape(), two_scalar(), {==}, 0.0, writes(gradient()), {==}, 0.0, writes([0.0]), {==},
    pair => BinaryCase<pair, 0, F32.neg(3.0), 1, 5.0, 0, 0.0>,
    left_kept => left_words => left_fits => right_kept => right_words => right_fits => ops => values => output_fits =>
      replayed_case(Elementwise.NegateLeft{}, two_scalar(), two_shape(), gradient(), scalar_shape(), [0.0], left_kept, right_kept, ops,
        left_fits, right_fits, output_fits,
        0, F32.neg(3.0), {==}, picked(Contracts.words(Shape.output_shape(two_scalar()), 0.0, ops, 2n), Laws.cells(Elementwise.NegateLeft{}, two_scalar(), two_shape(), 0.0, writes(gradient()), scalar_shape(), 0.0, writes([0.0]), 2n), 0n, values),
        1, 5.0, {==}, picked(Contracts.words(two_shape(), 0.0, left_kept, 2n), Contracts.words(two_shape(), 0.0, writes(gradient()), 2n), 1n, left_words),
        0, 0.0, {==}, picked(Contracts.words(scalar_shape(), 0.0, right_kept, 1n), Contracts.words(scalar_shape(), 0.0, writes([0.0]), 1n), 0n, right_words)))

# The parallel product then reduce_to through laws alone: binary_replayed
# makes the product a replayed tensor, so reduce_to_fiber applies to it.
# Reducing [3.0, 5.0] * [10.0] back to [1] adds both products from +0.0.
def reduced_point(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>, +target: Shape.Shape, +index: U32) -> Maybe<&2, F32>:
  (inputs, outcome) = pair
  match outcome:
    case Fail{error}:
      None{}
    case Done{output}:
      ReduceFold.output_point(Reduce.reduce_to(output, target), index)

def chained(+index: U32) -> Maybe<&2, F32>:
  reduced_point(result(Elementwise.Mul{}, two_shape(), gradient(), one_shape(), [10.0], {==}, {==}), one_shape(), index)

def back_plan() -> Shape.BroadcastPlan:
  plan_of(Shape.broadcast_checked(one_shape(), Shape.output_shape(two_one())), {==})

def products() -> List<&2, F32>:
  Laws.cells(Elementwise.Mul{}, two_one(), two_shape(), 0.0, writes(gradient()), one_shape(), 0.0, writes([10.0]),
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(two_one()))))

def chain_total(+index: U32) -> Maybe<&2, F32>:
  +count = Shape.logical_numel(Shape.dimensions(Shape.output_shape(two_one())))
  ReduceLaws.fiber(back_plan(), products(), index, count)

def fiber_value(-target: Shape.Shape, -input: Shape.Shape, -expected: U32 -> Maybe<&2, F32>, -word: U32 -> F32,
  -outcome: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>,
  evidence: ReduceFold.ReduceToContract<target, input, expected, word, outcome>,
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {ReduceFold.output_point(outcome, index) == expected(index) : Maybe<&2, F32>}:
  ReduceFold.ReduceToContract{successful, dimensions, value, kept} = evidence
  value(index, inside)

def reduced_value(+ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(two_one()), 0.0, ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(two_one()))) : Nat},
  values: {Contracts.words(Shape.output_shape(two_one()), 0.0, ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(two_one())))) ==
    products() : List<&2, F32>},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), 1n) == True{} : Bool}) ->
  {ReduceFold.output_point(Reduce.reduce_to(Core.Packed{Shape.output_shape(two_one()), Core.TTensor{Replayed.storage(Shape.output_shape(two_one()), 0.0, ops), fits}},
    one_shape()), index) == chain_total(index) : Maybe<&2, F32>}:
  +output = Shape.output_shape(two_one())
  +count = Shape.logical_numel(Shape.dimensions(output))
  Equal.trans(Maybe<&2, F32>,
    ReduceFold.output_point(Reduce.reduce_to(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, ops), fits}}, one_shape()), index),
    ReduceLaws.fiber(back_plan(), Contracts.words(output, 0.0, ops, count), index, count), chain_total(index),
    fiber_value(one_shape(), output, cell => ReduceLaws.fiber(back_plan(), Contracts.words(output, 0.0, ops, count), cell, count),
      other => Replayed.value(output, 0.0, ops, other),
      Reduce.reduce_to(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, ops), fits}}, one_shape()),
      ReduceFold.reduce_to_fiber(one_shape(), output, back_plan(), {==}, {==}, 0.0, ops, fits), index, inside),
    Equal.cong(List<&2, F32>, Maybe<&2, F32>, words => ReduceLaws.fiber(back_plan(), words, index, count),
      Contracts.words(output, 0.0, ops, count), products(), values))

def chain_reduced(+index: U32, inside: {Nat.is_lt(U32.to_nat(index), 1n) == True{} : Bool}) -> {chained(index) == chain_total(index) : Maybe<&2, F32>}:
  Laws.binary_replayed(Elementwise.Mul{}, two_shape(), one_shape(), two_one(), {==}, 0.0, writes(gradient()), {==}, 0.0, writes([10.0]), {==},
    pair => {reduced_point(pair, one_shape(), index) == chain_total(index) : Maybe<&2, F32>},
    left_kept => left_words => left_fits => right_kept => right_words => right_fits => ops => values => output_fits =>
      reduced_value(ops, output_fits, values, index, inside))

def binary_then_reduce() -> {chained(0) == Some{F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))} : Maybe<&2, F32>}:
  chain_reduced(0, {==})
