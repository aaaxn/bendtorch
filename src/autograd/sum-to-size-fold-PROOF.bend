import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/permutation-mapping.bend as Mapping
import ../tensor/transform-PROOF.bend as TransformProof
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold-LAWS.bend as ReduceFold
import ./reduce-to-fold-PROOF.bend as ReduceTo
import ./sum-to-size.bend as SumToSize
import ./sum-to-size-fold-LAWS.bend as Laws

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
