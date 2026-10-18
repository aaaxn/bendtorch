import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/permutation_mapping.bend as Mapping
import ../tensor/word_conversion.bend as Conversion
import ../tensor/transform_PROOF.bend as TransformProof
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/pointwise_proof.bend as PointwiseProof
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/words_proof.bend as Words
import ./reduce.bend as Reduce
import ./reduce_LAWS.bend as ReduceLaws
import ./reduce_fold.bend as Cells
import ./reduce_fold_PROOF.bend as FoldProof
import ./reduce_fold_LAWS.bend as ReduceFold
import ./reduce_to_fold_PROOF.bend as ReduceTo
import ./reduce_replayed_proof.bend as Replay
import ./sum_to_size.bend as SumToSize
import ./sum_to_size_fold_LAWS.bend as Laws

def clone_pair_lift(+shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  pair: Storage.Storage & Storage.Storage, -same: {pair == Storage.clone(original) : Storage.Storage & Storage.Storage},
  -P: Core.Tensor<shape> & Core.Tensor<shape> -> Type,
  h: @-first_fits: {Lifecycle.logical_length(StorageLaws.owner(Storage.Storage, pair)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-second_fits: {Lifecycle.logical_length(StorageLaws.outcome(Storage.Storage, pair)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{StorageLaws.owner(Storage.Storage, pair), first_fits},
      Core.TTensor{StorageLaws.outcome(Storage.Storage, pair), second_fits}))) ->
  P(Core.clone_result(shape, original, valid, pair, same)):
  (first, second) = pair
  h(Equal.trans(Nat, Lifecycle.logical_length(first), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.first_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_first_length(original), valid),
    Equal.trans(Nat, Lifecycle.logical_length(second), Lifecycle.logical_length(original), Shape.logical_numel(Shape.dimensions(shape)),
      %Equal.sym(Storage.Storage & Storage.Storage, (first, second), Storage.clone(original), same) :
        {Lifecycle.second_length(_) == Lifecycle.logical_length(original) : Nat}
      Lifecycle.clone_second_length(original), valid))

def pair_shift(+shape: Shape.Shape, -first: Storage.Storage, -second: Storage.Storage,
  -first_target: Storage.Storage, -second_target: Storage.Storage,
  first_moved: {first_target == first : Storage.Storage}, second_moved: {second_target == second : Storage.Storage},
  -P: Core.Tensor<shape> & Core.Tensor<shape> -> Type,
  h: @-first_fits: {Lifecycle.logical_length(first_target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-second_fits: {Lifecycle.logical_length(second_target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{first_target, first_fits}, Core.TTensor{second_target, second_fits}))) ->
  @-first_fits: {Lifecycle.logical_length(first) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @-second_fits: {Lifecycle.logical_length(second) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{first, first_fits}, Core.TTensor{second, second_fits})):
  %first_moved : (@-first_fits: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-second_fits: {Lifecycle.logical_length(second) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
      P((Core.TTensor{_, first_fits}, Core.TTensor{second, second_fits})))
  %second_moved : (@-first_fits: {Lifecycle.logical_length(first_target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-second_fits: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
      P((Core.TTensor{first_target, first_fits}, Core.TTensor{_, second_fits})))
  h

# Cloning a replayed tensor gives two replayed tensors: the kept one with the
# history ops ++ [First] and the copy with ops ++ [Second]. Neither changes a
# word.
def clone_lift(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -P: Core.Tensor<shape> & Core.Tensor<shape> -> Type,
  h: @-first_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.First{}))) ==
      Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    @-second_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.Second{}))) ==
      Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P((Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.First{})), first_fits},
      Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.Second{})), second_fits}))) ->
  P(Core.clone(shape, Core.TTensor{Replayed.storage(shape, fill, ops), valid})):
  clone_pair_lift(shape, Replayed.storage(shape, fill, ops), valid, Storage.clone(Replayed.storage(shape, fill, ops)), {==}, P,
    pair_shift(shape, StorageLaws.first(Replayed.storage(shape, fill, ops)), StorageLaws.second(Replayed.storage(shape, fill, ops)),
      Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.First{})), Replayed.storage(shape, fill, Replayed.snoc(ops, StorageLaws.Second{})),
      Replayed.replay_snoc(ops, StorageLaws.First{}, shape, fill), Replayed.replay_snoc(ops, StorageLaws.Second{}, shape, fill), P, h))

def same_true(+shape: Shape.Shape, +target: Shape.Shape, same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>}) ->
  {Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) == True{} : Bool}:
  %same : {Layout.words_same(Shape.dimensions(shape), _) == True{} : Bool}
  Mapping.words_same_refl(Shape.dimensions(shape))

def differ_false(+shape: Shape.Shape, +target: Shape.Shape, differ: {Shape.dimensions(shape) != Shape.dimensions(target) : List<&2, U32>},
  decision: Shape.BoolCase<Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target))>) ->
  {Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) == False{} : Bool}:
  match decision:
    case Shape.BTrue{equal}:
      Empty.absurd({Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) == False{} : Bool},
        differ(TransformProof.words_equal(Shape.dimensions(shape), Shape.dimensions(target), equal)))
    case Shape.BFalse{equal}:
      Equal.sym(Bool, False{}, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), equal)

def shape_below(+shape: Shape.Shape, +target: Shape.Shape, +index: U32,
  same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>},
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(shape), Shape.dimensions(target), same) :
    {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(_)) == True{} : Bool}
  inside

# A clone half reads the contribution's word: First and Second change no word.
def clone_point(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +half: StorageLaws.Op,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, Replayed.snoc(ops, half))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  kept: {StorageLaws.latest(U32.to_nat(Shape.count(shape)), half, index, Replayed.value(shape, fill, ops, index)) ==
    Replayed.value(shape, fill, ops, index) : F32}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(ops, half)), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(ops, half)), fits}}, index)),
    Some{Replayed.value(shape, fill, Replayed.snoc(ops, half), index)}, Some{Replayed.value(shape, fill, ops, index)},
    PointwiseProof.replayed_point(shape, fill, Replayed.snoc(ops, half), fits, index, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, fill, Replayed.snoc(ops, half), index),
      Replayed.value(shape, fill, ops, index),
      Equal.trans(F32, Replayed.value(shape, fill, Replayed.snoc(ops, half), index),
        StorageLaws.latest(U32.to_nat(Shape.count(shape)), half, index, Replayed.value(shape, fill, ops, index)),
        Replayed.value(shape, fill, ops, index),
        Replayed.expected_snoc(ops, half, U32.to_nat(Shape.count(shape)), index, fill), kept)))

def sum_to_size_clone(+target: Shape.Shape, +shape: Shape.Shape,
  +same: {Shape.dimensions(shape) == Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  ReduceFold.ReduceToContract<target, shape,
    index => Some{Replayed.value(shape, fill, ops, index)},
    other => Replayed.value(shape, fill, ops, other),
    SumToSize.sum_to_size(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>:
  %Equal.sym(Bool, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), True{}, same_true(shape, target, same)) :
    ReduceFold.ReduceToContract<target, shape,
      index => Some{Replayed.value(shape, fill, ops, index)},
      other => Replayed.value(shape, fill, ops, other),
      SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _)>
  clone_lift(shape, fill, ops, fits,
    pair => ReduceFold.ReduceToContract<target, shape,
      index => Some{Replayed.value(shape, fill, ops, index)},
      other => Replayed.value(shape, fill, ops, other),
      SumToSize.cloned(shape, pair)>,
    first_fits => second_fits =>
      ReduceFold.ReduceToContract{
        {==},
        Equal.cong(List<&2, U32>, Maybe<&2, List<&2, U32>>, dimensions => Some{dimensions}, Shape.dimensions(shape), Shape.dimensions(target), same),
        index => inside =>
          clone_point(shape, fill, ops, StorageLaws.Second{}, second_fits, index, shape_below(shape, target, index, same, inside), {==}),
        other => inside =>
          clone_point(shape, fill, ops, StorageLaws.First{}, first_fits, other, inside, {==})})

def sum_to_size_fiber(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  differ: {Shape.dimensions(shape) != Shape.dimensions(target) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  ReduceFold.ReduceToContract<target, shape,
    index => ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
      index, Shape.logical_numel(Shape.dimensions(shape))),
    other => Replayed.value(shape, fill, ops, other),
    SumToSize.sum_to_size(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>:
  %Equal.sym(Bool, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), False{},
    differ_false(shape, target, differ, Shape.bool_case(Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target))))) :
    ReduceFold.ReduceToContract<target, shape,
      index => ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
        index, Shape.logical_numel(Shape.dimensions(shape))),
      other => Replayed.value(shape, fill, ops, other),
      SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _)>
  ReduceFold.reduce_to_fiber(target, shape, plan, planned, same, fill, ops, fits)

def Laws.sum_to_size_clone(target, shape, same, fill, ops, fits):
  sum_to_size_clone(target, shape, same, fill, ops, fits)

def Laws.sum_to_size_fiber(target, shape, plan, planned, same, differ, fill, ops, fits):
  sum_to_size_fiber(target, shape, plan, planned, same, differ, fill, ops, fits)

# A First or Second half of a clone keeps every word.
def half_keeps(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +half: StorageLaws.Op, +index: U32,
  kept: {StorageLaws.latest(U32.to_nat(Shape.count(shape)), half, index, Replayed.value(shape, fill, ops, index)) ==
    Replayed.value(shape, fill, ops, index) : F32}) ->
  {Replayed.value(shape, fill, Replayed.snoc(ops, half), index) == Replayed.value(shape, fill, ops, index) : F32}:
  Equal.trans(F32, Replayed.value(shape, fill, Replayed.snoc(ops, half), index),
    StorageLaws.latest(U32.to_nat(Shape.count(shape)), half, index, Replayed.value(shape, fill, ops, index)),
    Replayed.value(shape, fill, ops, index),
    Replayed.expected_snoc(ops, half, U32.to_nat(Shape.count(shape)), index, fill), kept)

# The First half of a clone lists the same words.
def words_first(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Contracts.words(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), count) == Contracts.words(shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +position = U32.from_nat(previous)
      Words.words_snoc(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), previous,
        Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), position), Replayed.value(shape, fill, ops, position),
        Contracts.words(shape, fill, ops, previous), words_first(shape, fill, ops, previous),
        half_keeps(shape, fill, ops, StorageLaws.First{}, position, {==}))

# A list equal to the words of a replayed tensor holds each in-bounds word at
# its index.
def value_at(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +values: List<&2, F32>,
  +words: {Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Some{Replayed.value(shape, fill, ops, index)} == ReduceLaws.word_at(values, U32.to_nat(index)) : Maybe<&2, F32>}:
  +count = Shape.logical_numel(Shape.dimensions(shape))
  +position = U32.to_nat(index)
  %Conversion.word_natural_exact(index) : {Some{Replayed.value(shape, fill, ops, _)} == ReduceLaws.word_at(values, position) : Maybe<&2, F32>}
  %words : {Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} == ReduceLaws.word_at(_, position) : Maybe<&2, F32>}
  Equal.sym(Maybe<&2, F32>, ReduceLaws.word_at(Contracts.words(shape, fill, ops, count), position),
    Some{Replayed.value(shape, fill, ops, U32.from_nat(position))}, FoldProof.words_at(shape, fill, ops, count, position, inside))

# Two histories that list the same words agree at every in-bounds index.
def agree_at(+shape: Shape.Shape, +fill: F32, +kept: List<&2, StorageLaws.Op>, +ops: List<&2, StorageLaws.Op>,
  +same: {Contracts.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) ==
    Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) : List<&2, F32>},
  +index: U32, +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Replayed.value(shape, fill, kept, index) == Replayed.value(shape, fill, ops, index) : F32}:
  +values = Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape)))
  Replayed.some_value(Replayed.value(shape, fill, kept, index), Replayed.value(shape, fill, ops, index),
    Equal.trans(Maybe<&2, F32>, Some{Replayed.value(shape, fill, kept, index)}, ReduceLaws.word_at(values, U32.to_nat(index)),
      Some{Replayed.value(shape, fill, ops, index)},
      value_at(shape, fill, kept, values, same, index, inside),
      Equal.sym(Maybe<&2, F32>, Some{Replayed.value(shape, fill, ops, index)}, ReduceLaws.word_at(values, U32.to_nat(index)),
        value_at(shape, fill, ops, values, {==}, index, inside))))

def replayed_clone(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +values: List<&2, F32>,
  +words: {Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>},
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  equal: {True{} == Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)) : Bool},
  h: Laws.ReducedWith(target, shape, plan, fill, values, True{}, P)) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, True{})):
  +count = Shape.logical_numel(Shape.dimensions(shape))
  +dimensions = TransformProof.words_equal(Shape.dimensions(shape), Shape.dimensions(target), equal)
  fits =>
    clone_lift(shape, fill, ops, fits, pair => P(SumToSize.cloned(shape, pair)),
      first_fits => second_fits =>
        h(Replayed.snoc(ops, StorageLaws.First{}),
          Equal.trans(List<&2, F32>, Contracts.words(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), count),
            Contracts.words(shape, fill, ops, count), values, words_first(shape, fill, ops, count), words),
          first_fits, shape, dimensions, fill, Replayed.snoc(ops, StorageLaws.Second{}),
          index => inside =>
            +below = shape_below(shape, target, index, dimensions, inside)
            Equal.trans(Maybe<&2, F32>, Some{Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.Second{}), index)},
              Some{Replayed.value(shape, fill, ops, index)}, ReduceLaws.word_at(values, U32.to_nat(index)),
              Equal.cong(F32, Maybe<&2, F32>, value => Some{value},
                Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.Second{}), index), Replayed.value(shape, fill, ops, index),
                half_keeps(shape, fill, ops, StorageLaws.Second{}, index, {==})),
              value_at(shape, fill, ops, values, words, index, below)),
          second_fits))

def replayed_fiber(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>, +values: List<&2, F32>,
  +words: {Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>},
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  h: Laws.ReducedWith(target, shape, plan, fill, values, False{}, P)) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, False{})):
  +count = Shape.logical_numel(Shape.dimensions(shape))
  ReduceTo.run_lift(target, shape, plan, planned, same, fill, ops, P,
    kept => values_eq => words_eq => kept_fits => out => cells => out_fits =>
      h(kept, Equal.trans(List<&2, F32>, Contracts.words(shape, fill, kept, count), Contracts.words(shape, fill, ops, count), values, words_eq, words),
        kept_fits, target, {==}, 0.0, out,
        index => inside =>
          +inside = inside
          Equal.trans(Maybe<&2, F32>, Some{Replayed.value(target, 0.0, out, index)},
            Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))},
            ReduceLaws.fiber(plan, values, index, count),
            Equal.cong(F32, Maybe<&2, F32>, value => Some{value}, Replayed.value(target, 0.0, out, index),
              Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index)), cells(index)(inside)),
            %words : {Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))} == ReduceLaws.fiber(plan, _, index, count) : Maybe<&2, F32>}
            ReduceTo.cell_fiber(target, shape, plan, planned, same, fill, ops, index, inside)),
        out_fits))

def replayed_decided(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>, +values: List<&2, F32>,
  +words: {Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) == values : List<&2, F32>},
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  decision: Shape.BoolCase<Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target))>) ->
  Laws.ReducedWith(target, shape, plan, fill, values, Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)), P) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target,
      Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target)))):
  match decision:
    case Shape.BTrue{+equal}:
      %equal : (Laws.ReducedWith(target, shape, plan, fill, values, _, P) ->
        @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
          P(SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _)))
      h => replayed_clone(target, shape, plan, fill, ops, values, words, P, equal, h)
    case Shape.BFalse{equal}:
      %equal : (Laws.ReducedWith(target, shape, plan, fill, values, _, P) ->
        @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
          P(SumToSize.decided(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _)))
      h => replayed_fiber(target, shape, plan, planned, same, fill, ops, values, words, P, h)

def Laws.sum_to_size_replayed(target, shape, plan, planned, same, fill, ops, fits, values, words, P, h):
  replayed_decided(target, shape, plan, planned, same, fill, ops, values, words, P,
    Shape.bool_case(Layout.words_same(Shape.dimensions(shape), Shape.dimensions(target))))(h)(fits)
