import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/layout.bend as Layout
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/word-conversion.bend as Conversion
import ../tensor/permutation-mapping.bend as Mapping
import ../tensor/projection-LAWS.bend as Projection
import ../tensor/projection-PROOF.bend as ProjectionProof
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/broadcast-proof.bend as BroadcastProof
import ../tensor/contracts/fold.bend as Fold
import ../tensor/contracts/sum.bend as Sum
import ../tensor/contracts/LAWS.bend as Contracts
import ./reduce.bend as Reduce
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold.bend as ReduceFold
import ./reduce-fold-PROOF.bend as FoldProof
import ./reduce-fold-LAWS.bend as Laws
import ../storage/api.bend as Api
import ../storage/array.bend as ArrayCore
import ../tensor/read-contract.bend as Reads
import ../tensor/contracts/binary-storage-proof.bend as Owned
import ./reduce-replayed-proof.bend as Replay

def planned_shapes(left: Shape.Shape, right: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>) -> Shape.Shape & Shape.Shape:
  match result:
    case Fail{error}:
      (left, right)
    case Done{+plan}:
      (Shape.left_shape(plan), Shape.right_shape(plan))

def shapes_construct(+left: Shape.Shape, +right: Shape.Shape, -left_padding: Nat, -right_padding: Nat,
  -minimal: Shape.PadMinimal<left_padding, right_padding>, -output: List<&2, U32>,
  -aligned: Shape.Aligned(Shape.pad(left_padding, Shape.extents(left)), Shape.pad(right_padding, Shape.extents(right)), Shape.naturals(output)),
  +result: Shape.ConstructResult<output>) ->
  {planned_shapes(left, right, Shape.broadcast_checked.construct(left, right, left_padding, right_padding, minimal, output, aligned, result)) ==
    (left, right) : Shape.Shape & Shape.Shape}:
  match result:
    case Shape.SError{error}:
      {==}
    case Shape.SBuilt{shape, preserved}:
      {==}

def shapes_align(+left: Shape.Shape, +right: Shape.Shape, +left_padding: Nat, +right_padding: Nat,
  -minimal: Shape.PadMinimal<left_padding, right_padding>,
  +alignment: Shape.Alignment<Shape.pad_words(left_padding, Shape.dimensions(left)), Shape.pad_words(right_padding, Shape.dimensions(right))>) ->
  {planned_shapes(left, right, Shape.broadcast_checked.align(left, right, left_padding, right_padding, minimal, alignment)) ==
    (left, right) : Shape.Shape & Shape.Shape}:
  match alignment:
    case Shape.Misaligned{}:
      {==}
    case Shape.Aligns{+output, aligned}:
      shapes_construct(left, right, left_padding, right_padding, minimal, output,
        Shape.broadcast_checked.repad(left_padding, right_padding, Shape.dimensions(left), Shape.dimensions(right), output, aligned),
        Shape.construct(output))

def shapes_checked(+left: Shape.Shape, +right: Shape.Shape) ->
  {planned_shapes(left, right, Shape.broadcast_checked(left, right)) == (left, right) : Shape.Shape & Shape.Shape}:
  +left_rank = Shape.rank_dimensions(Shape.dimensions(left))
  +right_rank = Shape.rank_dimensions(Shape.dimensions(right))
  shapes_align(left, right, Nat.sub(right_rank, left_rank), Nat.sub(left_rank, right_rank), Shape.minimal(left_rank, right_rank),
    Shape.align(Shape.pad_words(Nat.sub(right_rank, left_rank), Shape.dimensions(left)),
      Shape.pad_words(Nat.sub(left_rank, right_rank), Shape.dimensions(right))))

def first(pair: Shape.Shape & Shape.Shape) -> Shape.Shape:
  (left, right) = pair
  left

def second(pair: Shape.Shape & Shape.Shape) -> Shape.Shape:
  (left, right) = pair
  right

def plan_shapes(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {(Shape.left_shape(plan), Shape.right_shape(plan)) == (target, shape) : Shape.Shape & Shape.Shape}:
  %planned : {planned_shapes(target, shape, _) == (target, shape) : Shape.Shape & Shape.Shape}
  shapes_checked(target, shape)

def plan_left(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {Shape.left_shape(plan) == target : Shape.Shape}:
  Equal.cong(Shape.Shape & Shape.Shape, Shape.Shape, first, (Shape.left_shape(plan), Shape.right_shape(plan)), (target, shape),
    plan_shapes(target, shape, plan, planned))

def plan_right(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}) ->
  {Shape.right_shape(plan) == shape : Shape.Shape}:
  Equal.cong(Shape.Shape & Shape.Shape, Shape.Shape, second, (Shape.left_shape(plan), Shape.right_shape(plan)), (target, shape),
    plan_shapes(target, shape, plan, planned))

def target_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(ReduceFold.target_offset(plan, position))

def source_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(ReduceFold.contribution_offset(plan, position))

# The target's history after `count` steps of the reduce_to loop: step k reads
# the target cell that position k lands on, then writes that cell plus
# contribution word k back to it.
def target_below(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +left: {Shape.left_shape(plan) == target : Shape.Shape}, +position: Nat,
  below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Nat.is_lt(ReduceFold.target_offset(plan, position), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}:
  %left : {Nat.is_lt(ReduceFold.target_offset(plan, position), Shape.logical_numel(Shape.dimensions(_))) == True{} : Bool}
  BroadcastProof.left_walk_bound(plan, Shape.index_map(plan), position, below)

def source_below(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +right: {Shape.right_shape(plan) == shape : Shape.Shape}, +position: Nat,
  below: {Nat.is_lt(position, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Nat.is_lt(ReduceFold.contribution_offset(plan, position), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %right : {Nat.is_lt(ReduceFold.contribution_offset(plan, position), Shape.logical_numel(Shape.dimensions(_))) == True{} : Bool}
  BroadcastProof.right_walk_bound(plan, Shape.index_map(plan), position, below)

def some_injective(-A: Type, -x: A, -y: A, equal: {Some{x} == Some{y} : Maybe<&1, A>}) -> {x == y : A}:
  Equal.cong(Maybe<&1, A>, A, found => Maybe.default(&1, A, found, x), Some{x}, Some{y}, equal)

def indices_found(+plan: Shape.BroadcastPlan, +cursor: Nat,
  +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Projection.indices_result(Elementwise.input_indices(plan, U32.from_nat(cursor))) ==
    Some{(target_index(plan, cursor), source_index(plan, cursor))} : Maybe<&1, U32 & U32>}:
  Equal.trans(Maybe<&1, U32 & U32>, Projection.indices_result(Elementwise.input_indices(plan, U32.from_nat(cursor))),
    Projection.offset_spec(plan, U32.to_nat(U32.from_nat(cursor))), Some{(target_index(plan, cursor), source_index(plan, cursor))},
    Projection.input_indices_refines(plan, U32.from_nat(cursor)),
    %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, PointwiseProof.word_exact(cursor, Shape.output_shape(plan), below)) :
      {Projection.offset_spec(plan, _) == Some{(target_index(plan, cursor), source_index(plan, cursor))} : Maybe<&1, U32 & U32>}
    FoldProof.offsets_walk(plan, Shape.index_map(plan), cursor, BroadcastProof.output_inside(plan, cursor, below)))

def contribution_count(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  {Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.cong(List<&2, U32>, Nat, Shape.logical_numel, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape), same)

def contribution_room(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  {Nat.is_le(Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))), Shape.logical_numel(Shape.dimensions(shape)),
    contribution_count(shape, plan, same)) : {Nat.is_le(Shape.logical_numel(Shape.dimensions(shape)), _) == True{} : Bool}
  Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(shape)))

def fitted_same(+shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  {Layout.words_same(Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape)) == True{} : Bool}:
  %Equal.sym(List<&2, U32>, Shape.dimensions(Shape.output_shape(plan)), Shape.dimensions(shape), same) :
    {Layout.words_same(_, Shape.dimensions(shape)) == True{} : Bool}
  Mapping.words_same_refl(Shape.dimensions(shape))

# The parallel reduce over a replayed contribution, in the transport form of
# `Replay.Final`.
def run_lift(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  h: Replay.Final(plan, target, shape, fill, ops, Shape.index_map(plan), StorageLaws.created(Shape.count(target), 0.0), P)) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Reduce.reduce_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)):
  +output = Shape.output_shape(plan)
  +left = plan_left(target, shape, plan, planned)
  +right = plan_right(target, shape, plan, planned)
  +count_left : {U32.to_nat(Shape.count(target)) == Shape.numel(Shape.extents(Shape.left_shape(plan))) : Nat} =
    %Equal.sym(Shape.Shape, Shape.left_shape(plan), target, left) :
      {U32.to_nat(Shape.count(target)) == Shape.logical_numel(Shape.dimensions(_)) : Nat}
    Reads.count_exact(target)
  +facts : Replay.Facts<plan, shape, U32.to_nat(Shape.count(target))> =
    Replay.Facts{count_left, Equal.cong(Shape.Shape, List<&2, U32>, Shape.dimensions, Shape.right_shape(plan), shape, right)}
  fits =>
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(target, shape), Done{plan}, planned) :
      P(Reduce.planned(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _))
    %Equal.sym(Bool, Layout.words_same(Shape.dimensions(output), Shape.dimensions(shape)), True{}, fitted_same(shape, plan, same)) :
      P(Reduce.fitted(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan, _))
    Replayed.new_lift(target, 0.0,
      result => P(Reduce.parallel_allocated(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan, result)),
      fo => Owned.created_lift(Shape.count(target), 0.0, Conversion.shape_count_within(target),
        st => (Replay.Final(plan, target, shape, fill, ops, Shape.index_map(plan), st, P) ->
          @-fo: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(target)) : Nat} ->
          P(Reduce.parallel_allocated(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan, Done{Core.TTensor{st, fo}}))),
        zero => final => fo =>
          final(ops)(other => {==})({==})(fits)(Nil{})(
            index => inside => Empty.absurd(
              {Replayed.value(target, 0.0, Nil{}, index) == Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index)) : F32},
              Arith.lt_zero(U32.to_nat(index),
                %Equal.trans(Nat, Shape.logical_numel(Shape.dimensions(target)), U32.to_nat(Shape.count(target)), 0n,
                  Equal.sym(Nat, U32.to_nat(Shape.count(target)), Shape.logical_numel(Shape.dimensions(target)), Reads.count_exact(target)), zero) :
                  {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
                inside)))(fo),
        d => fits_d => f => final => fo =>
          Replay.indexed_lift(plan, target, shape, fill, ops, fits, facts, d, fits_d, f, fo, Shape.index_map(plan), P, final))(h)(fo))

# A target cell of the replayed reduce holds `ReduceLaws.fiber` over the
# contribution's words.
def cell_fiber(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))} ==
    ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index,
      Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}:
  +left = plan_left(target, shape, plan, planned)
  %contribution_count(shape, plan, same) :
    {Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))} ==
      ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index, _) : Maybe<&2, F32>}
  Replay.word_fiber(plan, Shape.index_map(plan), shape, fill, ops, index,
    %Equal.sym(Shape.Shape, Shape.left_shape(plan), target, left) : {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(_))) == True{} : Bool}
    inside,
    Equal.cong(Shape.Shape, List<&2, U32>, Shape.dimensions, Shape.right_shape(plan), shape, plan_right(target, shape, plan, planned)))

def reduce_to_fiber(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Laws.ReduceToContract<target, shape,
    index => ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
      index, Shape.logical_numel(Shape.dimensions(shape))),
    other => Replayed.value(shape, fill, ops, other),
    Reduce.reduce_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)>:
  +count = Shape.logical_numel(Shape.dimensions(shape))
  Laws.ReduceToContract{
    run_lift(target, shape, plan, planned, same, fill, ops, result => {ReduceLaws.succeeded(result) == True{} : Bool},
      kept => values => words => kept_fits => out => cells => out_fits => {==})(fits),
    run_lift(target, shape, plan, planned, same, fill, ops,
      result => {Laws.output_dimensions(result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
      kept => values => words => kept_fits => out => cells => out_fits => {==})(fits),
    index => inside =>
      +inside = inside
      run_lift(target, shape, plan, planned, same, fill, ops,
        result => {Laws.output_point(result, index) ==
          ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, count), index, count) : Maybe<&2, F32>},
        kept => values => words => kept_fits => out => cells => out_fits =>
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{target, Core.TTensor{Replayed.storage(target, 0.0, out), out_fits}}, index)),
            Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))},
            ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, count), index, count),
            Equal.trans(Maybe<&2, F32>,
              Core.dynamic_read_value(Core.read_dynamic(Core.Packed{target, Core.TTensor{Replayed.storage(target, 0.0, out), out_fits}}, index)),
              Some{Replayed.value(target, 0.0, out, index)}, Some{Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index))},
              PointwiseProof.replayed_point(target, 0.0, out, out_fits, index, inside),
              Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(target, 0.0, out, index),
                Replay.cell_word(plan, Shape.index_map(plan), shape, fill, ops, U32.to_nat(index)), cells(index)(inside))),
            cell_fiber(target, shape, plan, planned, same, fill, ops, index, inside)))(fits),
    other => inside =>
      run_lift(target, shape, plan, planned, same, fill, ops,
        result => {Laws.input_point(result, other) == Some{Replayed.value(shape, fill, ops, other)} : Maybe<&2, F32>},
        kept => values => words => kept_fits => out => cells => out_fits =>
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}}, other)),
            Some{Replayed.value(shape, fill, kept, other)}, Some{Replayed.value(shape, fill, ops, other)},
            PointwiseProof.replayed_point(shape, fill, kept, kept_fits, other, inside),
            Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, fill, kept, other), Replayed.value(shape, fill, ops, other),
              values(other))))(fits)}

def Laws.reduce_to_fiber(target, shape, plan, planned, same, fill, ops, fits):
  reduce_to_fiber(target, shape, plan, planned, same, fill, ops, fits)
