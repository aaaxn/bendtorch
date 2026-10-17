import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/word-conversion.bend as Conversion
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ../tensor/contracts/binary-LAWS.bend as BinaryLaws
import ../tensor/contracts/binary-proof.bend as Binary
import ../tensor/contracts/binary-PROOF.bend as BinaryProof
import ./broadcast.bend as Broadcast
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold-PROOF.bend as FoldProof
import ./reduce-to-fold-PROOF.bend as ReduceTo
import ./broadcast-pointwise-LAWS.bend as Laws

# The pointwise family's KeepLeft words, with the input on the left and the
# +0.0 tensor of the target on the right, are broadcast_to's `cells`.
def cells_keep(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +target: Shape.Shape,
  +count: Nat) ->
  {BinaryLaws.cells(Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, count) ==
    Laws.cells(plan, shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      Equal.cong(List<&2, F32>, List<&2, F32>,
        before => List.append(&2, F32, before, [Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, previous)))]),
        BinaryLaws.cells(Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, previous), Laws.cells(plan, shape, fill, ops, previous),
        cells_keep(plan, shape, fill, ops, target, previous))

def cells_length(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {List.length(&2, F32, Laws.cells(plan, shape, fill, ops, count)) == count : Nat}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +before = Laws.cells(plan, shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, previous)))
      Equal.trans(Nat, List.length(&2, F32, List.append(&2, F32, before, [last])), 1n+List.length(&2, F32, before), 1n+previous,
        FoldProof.length_snoc(before, last),
        Equal.cong(Nat, Nat, n => 1n+n, List.length(&2, F32, before), previous, cells_length(plan, shape, fill, ops, previous)))

# Word `position` of `cells` is the input word at left_map(plan, position).
def cells_step(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +position: Nat,
  below: {Nat.is_lt(position, 1n+previous) == True{} : Bool}, decision: Shape.BoolCase<Nat.is_eq(position, previous)>,
  earlier: @inside: {Nat.is_lt(position, previous) == True{} : Bool} ->
    {ReduceLaws.word_at(Laws.cells(plan, shape, fill, ops, previous), position) ==
      Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, position)))} : Maybe<&2, F32>}) ->
  {ReduceLaws.word_at(Laws.cells(plan, shape, fill, ops, 1n+previous), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, position)))} : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{equal}:
      +before = Laws.cells(plan, shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, previous)))
      +inside = PointwiseProof.lt_succ_ne(position, previous, below, equal)
      Equal.trans(Maybe<&2, F32>, ReduceLaws.word_at(List.append(&2, F32, before, [last]), position), ReduceLaws.word_at(before, position),
        Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, position)))},
        FoldProof.word_at_front(before, last, position,
          %Equal.sym(Nat, List.length(&2, F32, before), previous, cells_length(plan, shape, fill, ops, previous)) :
            {Nat.is_lt(position, _) == True{} : Bool}
          inside),
        earlier(inside))
    case Shape.BTrue{equal}:
      +before = Laws.cells(plan, shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, previous)))
      %Equal.sym(Nat, position, previous, Arith.eq_true(position, previous, Equal.sym(Bool, True{}, Nat.is_eq(position, previous), equal))) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) ==
          Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, _)))} : Maybe<&2, F32>}
      %cells_length(plan, shape, fill, ops, previous) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) == Some{last} : Maybe<&2, F32>}
      FoldProof.word_at_last(before, last)

def cells_at(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +position: Nat,
  below: {Nat.is_lt(position, count) == True{} : Bool}) ->
  {ReduceLaws.word_at(Laws.cells(plan, shape, fill, ops, count), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, position)))} : Maybe<&2, F32>}:
  match count:
    case 0n:
      Empty.absurd({ReduceLaws.word_at(Nil{}, position) ==
        Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, position)))} : Maybe<&2, F32>},
        Arith.lt_zero(position, below))
    case 1n+previous:
      cells_step(plan, shape, fill, ops, previous, position, below, Shape.bool_case(Nat.is_eq(position, previous)),
        inside => cells_at(plan, shape, fill, ops, previous, position, inside))

# A replayed tensor whose first `count` words are `values` holds, below
# `count`, the word `values` lists there.
def listed_value(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +values: List<&2, F32>, +index: U32,
  words: {Contracts.words(shape, fill, ops, count) == values : List<&2, F32>},
  below: {Nat.is_lt(U32.to_nat(index), count) == True{} : Bool}) ->
  {Some{Replayed.value(shape, fill, ops, index)} == ReduceLaws.word_at(values, U32.to_nat(index)) : Maybe<&2, F32>}:
  %Conversion.word_natural_exact(index) :
    {Some{Replayed.value(shape, fill, ops, _)} == ReduceLaws.word_at(values, U32.to_nat(index)) : Maybe<&2, F32>}
  Equal.trans(Maybe<&2, F32>, Some{Replayed.value(shape, fill, ops, U32.from_nat(U32.to_nat(index)))},
    ReduceLaws.word_at(Contracts.words(shape, fill, ops, count), U32.to_nat(index)), ReduceLaws.word_at(values, U32.to_nat(index)),
    Equal.sym(Maybe<&2, F32>, ReduceLaws.word_at(Contracts.words(shape, fill, ops, count), U32.to_nat(index)),
      Some{Replayed.value(shape, fill, ops, U32.from_nat(U32.to_nat(index)))},
      FoldProof.words_at(shape, fill, ops, count, U32.to_nat(index), below)),
    Equal.cong(List<&2, F32>, Maybe<&2, F32>, listed => ReduceLaws.word_at(listed, U32.to_nat(index)),
      Contracts.words(shape, fill, ops, count), values, words))

# broadcast_to on a replayed input whose shape broadcasts to exactly the
# target: the planner and the target check pass, the zero tensor of the target
# is allocated, and the parallel build runs KeepLeft with the input on the
# left and the zero tensor on the right, whose words are `cells`.
def run_lift(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -P: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic> -> Type,
  h: Laws.BroadcastWith(plan, shape, fill, ops, P)) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)):
  +output = Shape.output_shape(plan)
  +count = Shape.logical_numel(Shape.dimensions(output))
  fits =>
    -input = {Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}} : Core.Dynamic}
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(shape, target), Done{plan}, planned) :
      P(Broadcast.planned(input, target, _))
    %Equal.sym(Bool, Layout.words_same(Shape.dimensions(output), Shape.dimensions(target)), True{}, ReduceTo.fitted_same(target, plan, same)) :
      P(Broadcast.fitted(input, target, plan, _))
    Replayed.new_lift(target, 0.0, result => P(Broadcast.zeroed(input, target, plan, result)),
      zeros_fits => Binary.allocated_replayed(Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, planned,
        pair => P(Broadcast.finish(pair)),
        kept => kept_words => input_fits => zeros_kept => zeros_words => zeros_kept_fits => output_ops => values => output_fits =>
          h(kept, kept_words, input_fits, output_ops,
            Equal.trans(List<&2, F32>, Contracts.words(output, 0.0, output_ops, count),
              BinaryLaws.cells(Elementwise.KeepLeft{}, plan, shape, fill, ops, target, 0.0, Nil{}, count), Laws.cells(plan, shape, fill, ops, count),
              values, cells_keep(plan, shape, fill, ops, target, count)),
            output_fits))(fits, zeros_fits))

def output_below(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +index: U32,
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(target), same) :
    {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
  inside

def broadcast_to_pointwise(+shape: Shape.Shape, +target: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(shape, target) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Laws.BroadcastToContract<target, shape,
    index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    other => Replayed.value(shape, fill, ops, other),
    Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>:
  +output = Shape.output_shape(plan)
  +count = Shape.logical_numel(Shape.dimensions(output))
  +total = Shape.logical_numel(Shape.dimensions(shape))
  run_lift(shape, target, plan, planned, same, fill, ops,
    result => Laws.BroadcastToContract<target, shape,
      index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
      other => Replayed.value(shape, fill, ops, other), result>,
    kept => kept_words => input_fits => output_ops => values => output_fits =>
      Laws.BroadcastToContract{
        {==},
        Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions}, Shape.dimensions(output), Shape.dimensions(target), same),
        index => inside =>
          +below = output_below(plan, target, index, same, inside)
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, output_ops), output_fits}}, index)),
            Some{Replayed.value(output, 0.0, output_ops, index)},
            Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, U32.to_nat(index))))},
            PointwiseProof.replayed_point(output, 0.0, output_ops, output_fits, index, below),
            Equal.trans(Maybe<&2, F32>, Some{Replayed.value(output, 0.0, output_ops, index)},
              ReduceLaws.word_at(Laws.cells(plan, shape, fill, ops, count), U32.to_nat(index)),
              Some{Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, U32.to_nat(index))))},
              listed_value(output, 0.0, output_ops, count, Laws.cells(plan, shape, fill, ops, count), index, values, below),
              cells_at(plan, shape, fill, ops, count, U32.to_nat(index), below))),
        other => inside =>
          +within = {inside : {Nat.is_lt(U32.to_nat(other), total) == True{} : Bool}}
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), input_fits}}, other)),
            Some{Replayed.value(shape, fill, kept, other)}, Some{Replayed.value(shape, fill, ops, other)},
            PointwiseProof.replayed_point(shape, fill, kept, input_fits, other, within),
            Equal.trans(Maybe<&2, F32>, Some{Replayed.value(shape, fill, kept, other)},
              ReduceLaws.word_at(Contracts.words(shape, fill, ops, total), U32.to_nat(other)), Some{Replayed.value(shape, fill, ops, other)},
              listed_value(shape, fill, kept, total, Contracts.words(shape, fill, ops, total), other, kept_words, within),
              Equal.sym(Maybe<&2, F32>, Some{Replayed.value(shape, fill, ops, other)},
                ReduceLaws.word_at(Contracts.words(shape, fill, ops, total), U32.to_nat(other)),
                listed_value(shape, fill, ops, total, Contracts.words(shape, fill, ops, total), other, {==}, within))))})(fits)

def Laws.broadcast_to_pointwise(shape, target, plan, planned, same, fill, ops, fits):
  broadcast_to_pointwise(shape, target, plan, planned, same, fill, ops, fits)

def Laws.broadcast_to_replayed(shape, target, plan, planned, same, fill, ops, fits, P, h):
  run_lift(shape, target, plan, planned, same, fill, ops, P, h)(fits)
