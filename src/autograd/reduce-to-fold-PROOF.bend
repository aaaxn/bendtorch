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

def total(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32, +count: Nat) -> F32:
  Fold.total(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, ReduceFold.Cell{plan, index},
    shape, fill, ops, count)

def target_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(ReduceFold.target_offset(plan, position))

def source_index(+plan: Shape.BroadcastPlan, +position: Nat) -> U32:
  U32.from_nat(ReduceFold.contribution_offset(plan, position))

def reads(+plan: Shape.BroadcastPlan, +count: Nat) -> List<&2, StorageLaws.Op>:
  Pointwise.reads(~Shape.BroadcastPlan, ~ReduceFold.contribution_offset, plan, count)

# The target's history after `count` steps of the reduce_to loop: step k reads
# the target cell that position k lands on, then writes that cell plus
# contribution word k back to it.
def history(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      +before = history(plan, target, shape, fill, ops, previous)
      +cell = target_index(plan, previous)
      Replayed.snoc(Replayed.snoc(before, StorageLaws.Read{cell}),
        StorageLaws.Write{cell, F32.add(Replayed.value(target, 0.0, before, cell), Replayed.value(shape, fill, ops, source_index(plan, previous)))})

def input_storage(+plan: Shape.BroadcastPlan, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~Shape.BroadcastPlan, ~ReduceFold.contribution_offset, plan, shape, fill, ops, count)

def output_storage(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +count: Nat) -> Storage.Storage:
  Replayed.storage(target, 0.0, history(plan, target, shape, fill, ops, count))

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

def landed(+offset: Nat, +index: U32, +old: F32, +word: F32, +prior: F32, +before: F32,
  decision: Shape.BoolCase<Nat.is_eq(offset, U32.to_nat(index))>,
  at_index: {Nat.is_eq(offset, U32.to_nat(index)) == True{} : Bool} -> {old == before : F32},
  kept: {prior == before : F32}) ->
  {Bool.pick(F32, Nat.is_eq(offset, U32.to_nat(index)), F32.add(old, word), prior) ==
    Fold.chosen(~Unit, ~Sum.add, Unit{}, Nat.is_eq(offset, U32.to_nat(index)), before, word) : F32}:
  match decision:
    case Shape.BTrue{+equal}:
      %equal : {Bool.pick(F32, _, F32.add(old, word), prior) == Fold.chosen(~Unit, ~Sum.add, Unit{}, _, before, word) : F32}
      Equal.cong(F32, F32, value => F32.add(value, word), old, before, at_index(Equal.sym(Bool, True{}, Nat.is_eq(offset, U32.to_nat(index)), equal)))
    case Shape.BFalse{equal}:
      %equal : {Bool.pick(F32, _, F32.add(old, word), prior) == Fold.chosen(~Unit, ~Sum.add, Unit{}, _, before, word) : F32}
      kept

def same_cell(+target: Shape.Shape, +history: List<&2, StorageLaws.Op>, +offset: Nat, +index: U32,
  same: {Nat.is_eq(offset, U32.to_nat(index)) == True{} : Bool}) ->
  {Replayed.value(target, 0.0, history, U32.from_nat(offset)) == Replayed.value(target, 0.0, history, index) : F32}:
  %Equal.sym(Nat, offset, U32.to_nat(index), Arith.eq_true(offset, U32.to_nat(index), same)) :
    {Replayed.value(target, 0.0, history, U32.from_nat(_)) == Replayed.value(target, 0.0, history, index) : F32}
  Equal.cong(U32, F32, cell => Replayed.value(target, 0.0, history, cell), U32.from_nat(U32.to_nat(index)), index,
    Conversion.word_natural_exact(index))

def history_value(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +left: {Shape.left_shape(plan) == target : Shape.Shape}, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +index: U32) ->
  {Replayed.value(target, 0.0, history(plan, target, shape, fill, ops, count), index) == total(plan, shape, fill, ops, index, count) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+k:
      +limit = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +before = history(plan, target, shape, fill, ops, k)
      +offset = ReduceFold.target_offset(plan, k)
      +cell = U32.from_nat(offset)
      +word = Replayed.value(shape, fill, ops, source_index(plan, k))
      +old = Replayed.value(target, 0.0, before, cell)
      +prior = Replayed.value(target, 0.0, before, index)
      +n = U32.to_nat(Shape.count(target))
      +written = {StorageLaws.Write{cell, F32.add(old, word)} : StorageLaws.Op}
      +offset_below = target_below(plan, target, left, k, Conversion.le_successor_lt(k, limit, room))
      +rec = history_value(plan, target, shape, fill, ops, left, k, Conversion.le_predecessor(k, limit, room), index)
      %Equal.sym(F32, Replayed.value(target, 0.0, Replayed.snoc(Replayed.snoc(before, StorageLaws.Read{cell}), written), index),
        StorageLaws.latest(n, written, index, prior),
        Equal.trans(F32, Replayed.value(target, 0.0, Replayed.snoc(Replayed.snoc(before, StorageLaws.Read{cell}), written), index),
          StorageLaws.latest(n, written, index, Replayed.value(target, 0.0, Replayed.snoc(before, StorageLaws.Read{cell}), index)),
          StorageLaws.latest(n, written, index, prior),
          Replayed.expected_snoc(Replayed.snoc(before, StorageLaws.Read{cell}), written, n, index, 0.0),
          Equal.cong(F32, F32, value => StorageLaws.latest(n, written, index, value),
            Replayed.value(target, 0.0, Replayed.snoc(before, StorageLaws.Read{cell}), index), prior,
            Replayed.value_after_read(target, 0.0, before, cell, index)))) :
        {_ == total(plan, shape, fill, ops, index, 1n+k) : F32}
      %Equal.sym(Nat, U32.to_nat(cell), offset, PointwiseProof.word_exact(offset, target, offset_below)) :
        {Bool.pick(F32, Bool.and(Nat.is_lt(_, n), Nat.is_eq(_, U32.to_nat(index))), F32.add(old, word), prior) ==
          total(plan, shape, fill, ops, index, 1n+k) : F32}
      %Equal.sym(Bool, Nat.is_lt(offset, n), True{}, PointwiseProof.lt_count(offset, target, offset_below)) :
        {Bool.pick(F32, Bool.and(_, Nat.is_eq(offset, U32.to_nat(index))), F32.add(old, word), prior) ==
          total(plan, shape, fill, ops, index, 1n+k) : F32}
      landed(offset, index, old, word, prior, total(plan, shape, fill, ops, index, k),
        Shape.bool_case(Nat.is_eq(offset, U32.to_nat(index))),
        same => Equal.trans(F32, old, prior, total(plan, shape, fill, ops, index, k),
          same_cell(target, before, offset, index, same), rec),
        rec)

def Shaped(+fuel: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +cursor: Nat,
  -input: Storage.Storage, -output: Storage.Storage,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type) -> Type:
  @-input_fits: {Lifecycle.logical_length(input) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output) == Shape.logical_numel(Shape.dimensions(target)) : Nat} ->
    Q(Reduce.loop(fuel, plan, target, cursor,
      Reduce.State{Core.Packed{shape, Core.TTensor{input, input_fits}}, Core.TTensor{output, output_fits}}))

def Start(+fuel: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +cursor: Nat, -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type) -> Type:
  Shaped(fuel, plan, target, shape, cursor, input_storage(plan, shape, fill, ops, cursor),
    output_storage(plan, target, shape, fill, ops, cursor), Q)

def Final(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type) -> Type:
  @-input_fits: {Lifecycle.logical_length(input_storage(plan, shape, fill, ops, count)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_storage(plan, target, shape, fill, ops, count)) ==
    Shape.logical_numel(Shape.dimensions(target)) : Nat} ->
    Q((Core.Packed{shape, Core.TTensor{input_storage(plan, shape, fill, ops, count), input_fits}},
      Done{Core.TTensor{output_storage(plan, target, shape, fill, ops, count), output_fits}}))

def final_at(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -end: Nat, -count: Nat, same: {end == count : Nat},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  h: Final(plan, target, shape, fill, ops, end, Q)) -> Final(plan, target, shape, fill, ops, count, Q):
  %same : Final(plan, target, shape, fill, ops, _, Q)
  h

def shift(+rest: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +cursor: Nat, -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  next: Start(rest, plan, target, shape, fill, ops, 1n+cursor, Q)) ->
  Shaped(rest, plan, target, shape, 1n+cursor,
    Replayed.storage(shape, fill, Replayed.snoc(List.append(&2, StorageLaws.Op, ops, reads(plan, cursor)),
      StorageLaws.Read{source_index(plan, cursor)})),
    Replayed.storage(target, 0.0, Replayed.snoc(Replayed.snoc(history(plan, target, shape, fill, ops, cursor),
      StorageLaws.Read{target_index(plan, cursor)}),
      StorageLaws.Write{target_index(plan, cursor),
        F32.add(Replayed.value(target, 0.0, history(plan, target, shape, fill, ops, cursor), target_index(plan, cursor)),
          Replayed.value(shape, fill, List.append(&2, StorageLaws.Op, ops, reads(plan, cursor)), source_index(plan, cursor)))})),
    Q):
  +read = {StorageLaws.Read{source_index(plan, cursor)} : StorageLaws.Op}
  +seen = List.append(&2, StorageLaws.Op, ops, reads(plan, cursor))
  +before = history(plan, target, shape, fill, ops, cursor)
  +cell = target_index(plan, cursor)
  +old = Replayed.value(target, 0.0, before, cell)
  +x = Replayed.value(shape, fill, seen, source_index(plan, cursor))
  +original = Replayed.value(shape, fill, ops, source_index(plan, cursor))
  -output_after = Replayed.storage(target, 0.0, Replayed.snoc(Replayed.snoc(before, StorageLaws.Read{cell}), StorageLaws.Write{cell, F32.add(old, x)}))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(shape, fill, history),
    List.append(&2, StorageLaws.Op, ops, Replayed.snoc(reads(plan, cursor), read)), Replayed.snoc(seen, read),
    PointwiseProof.append_snoc(ops, reads(plan, cursor), read)) :
    Shaped(rest, plan, target, shape, 1n+cursor, _, output_after, Q)
  %Equal.cong(F32, Storage.Storage,
    word => Replayed.storage(target, 0.0, Replayed.snoc(Replayed.snoc(before, StorageLaws.Read{cell}), StorageLaws.Write{cell, F32.add(old, word)})),
    original, x,
    Equal.sym(F32, x, original,
      PointwiseProof.value_reads(~Shape.BroadcastPlan, ~ReduceFold.contribution_offset, plan, cursor, shape, fill, ops, source_index(plan, cursor)))) :
    Shaped(rest, plan, target, shape, 1n+cursor, input_storage(plan, shape, fill, ops, 1n+cursor), _, Q)
  next

def decided_lift(+rest: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +cursor: Nat,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type, -state: Reduce.State<target>,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)>,
  same: {Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor) == True{} : Bool},
  h: Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
    Reduce.step(plan, target, U32.from_nat(cursor), state)))) ->
  Q(Reduce.decided(plan, target, cursor, U32.from_nat(cursor), state, state => Reduce.loop(rest, plan, target, 1n+cursor, state), decision)):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Q(Reduce.decided(plan, target, cursor, U32.from_nat(cursor), state,
        state => Reduce.loop(rest, plan, target, 1n+cursor, state), Shape.BFalse{equal})),
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor), True{}, equal, same)))
    case Shape.BTrue{equal}:
      h

def some_injective(-A: Type, -x: A, -y: A, equal: {Some{x} == Some{y} : Maybe<&1, A>}) -> {x == y : A}:
  Equal.cong(Maybe<&1, A>, A, found => Maybe.default(&1, A, found, x), Some{x}, Some{y}, equal)

def indices_lift(+rest: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +cursor: Nat, -state: Reduce.State<target>,
  +cell: U32, +source: U32, result: Result<&2, &1, Core.Error, U32 & U32>,
  found: {Projection.indices_result(result) == Some{(cell, source)} : Maybe<&1, U32 & U32>},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  h: Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
    Reduce.step_indices(target, state, Done{(cell, source)})))) ->
  Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state), Reduce.step_indices(target, state, result))):
  match result:
    case Fail{error}:
      Empty.absurd(Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
        Reduce.step_indices(target, state, Fail{error}))),
        Arith.clash(Equal.cong(Maybe<&1, U32 & U32>, Bool, value => Maybe.is_some(&1, U32 & U32, value), None{}, Some{(cell, source)}, found)))
    case Done{indices}:
      %Equal.sym(U32 & U32, indices, (cell, source), some_injective(U32 & U32, indices, (cell, source), found)) :
        Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state), Reduce.step_indices(target, state, Done{_})))
      h

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

def step_lift(+rest: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +cursor: Nat, +below: {Nat.is_lt(cursor, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +left: {Shape.left_shape(plan) == target : Shape.Shape}, +right: {Shape.right_shape(plan) == shape : Shape.Shape},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  next: Start(rest, plan, target, shape, fill, ops, 1n+cursor, Q)) ->
  Start(1n+rest, plan, target, shape, fill, ops, cursor, Q):
  +seen = List.append(&2, StorageLaws.Op, ops, reads(plan, cursor))
  +before = history(plan, target, shape, fill, ops, cursor)
  +cell = target_index(plan, cursor)
  +source = source_index(plan, cursor)
  +x = Replayed.value(shape, fill, seen, source)
  +old = Replayed.value(target, 0.0, before, cell)
  +cell_inside = PointwiseProof.word_inside(ReduceFold.target_offset(plan, cursor), target, target_below(plan, target, left, cursor, below))
  +source_inside = PointwiseProof.word_inside(ReduceFold.contribution_offset(plan, cursor), shape, source_below(plan, shape, right, cursor, below))
  input_fits => output_fits =>
    decided_lift(rest, plan, target, cursor, Q,
      Reduce.State{Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, seen), input_fits}},
        Core.TTensor{Replayed.storage(target, 0.0, before), output_fits}},
      Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
      Mapping.natural_equal_true(U32.to_nat(U32.from_nat(cursor)), cursor, PointwiseProof.word_exact(cursor, Shape.output_shape(plan), below)),
      indices_lift(rest, plan, target, cursor,
        Reduce.State{Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, seen), input_fits}},
          Core.TTensor{Replayed.storage(target, 0.0, before), output_fits}},
        cell, source, Elementwise.input_indices(plan, U32.from_nat(cursor)), indices_found(plan, cursor, below), Q,
        Replayed.read_lift(shape, fill, seen, input_fits, source, source_inside,
          pair => Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
            Reduce.step_input(target, cell, Core.TTensor{Replayed.storage(target, 0.0, before), output_fits}, Core.read_dynamic_result(shape, pair)))),
          input_after => Replayed.read_lift(target, 0.0, before, output_fits, cell, cell_inside,
            pair => Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
              Reduce.step_output(target, cell,
                Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(seen, StorageLaws.Read{source})), input_after}}, x, pair))),
            read_after => Replayed.write_lift(target, 0.0, Replayed.snoc(before, StorageLaws.Read{cell}), read_after, cell, F32.add(old, x), cell_inside,
              pair => Q(Reduce.continue_step(target, state => Reduce.loop(rest, plan, target, 1n+cursor, state),
                Reduce.step_write(target,
                  Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, Replayed.snoc(seen, StorageLaws.Read{source})), input_after}}, pair))),
              write_after => shift(rest, plan, target, shape, fill, ops, cursor, Q, next)(input_after, write_after))))))

def loop_lift(+fuel: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +cursor: Nat, +count: Nat, +steps: {Nat.add(cursor, fuel) == count : Nat},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool},
  +left: {Shape.left_shape(plan) == target : Shape.Shape}, +right: {Shape.right_shape(plan) == shape : Shape.Shape},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  h: Final(plan, target, shape, fill, ops, count, Q)) ->
  Start(fuel, plan, target, shape, fill, ops, cursor, Q):
  match fuel:
    case 0n:
      final_at(plan, target, shape, fill, ops, count, cursor, Equal.sym(Nat, cursor, count, PointwiseProof.steps_done(cursor, count, steps)), Q, h)
    case 1n+rest:
      +limit = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      step_lift(rest, plan, target, shape, fill, ops, cursor,
        Arith.lt_le_trans(cursor, count, limit, PointwiseProof.cursor_below(cursor, rest, count, steps), room), left, right, Q,
        loop_lift(rest, plan, target, shape, fill, ops, 1n+cursor, count, PointwiseProof.steps_next(cursor, rest, count, steps), room,
          left, right, Q, h))

def start_cast(+fuel: Nat, +plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, Core.Tensor<target>> -> Type,
  start: Start(fuel, plan, target, shape, fill, ops, 0n, Q)) ->
  Shaped(fuel, plan, target, shape, 0n, Replayed.storage(shape, fill, ops), Replayed.storage(target, 0.0, Nil{}), Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(shape, fill, history),
    List.append(&2, StorageLaws.Op, ops, Nil{}), ops, PointwiseProof.append_nil(ops)) :
    Shaped(fuel, plan, target, shape, 0n, _, Replayed.storage(target, 0.0, Nil{}), Q)
  start

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

def run_lift(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  h: Final(plan, target, shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape)), pair => P(Reduce.finish(target, pair)))) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Reduce.reduce_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target)):
  +count = Shape.logical_numel(Shape.dimensions(shape))
  +output = Shape.output_shape(plan)
  fits =>
    %Equal.sym(Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, Shape.broadcast_checked(target, shape), Done{plan}, planned) :
      P(Reduce.planned(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, target, _))
    %Equal.sym(Bool, Layout.words_same(Shape.dimensions(output), Shape.dimensions(shape)), True{}, fitted_same(shape, plan, same)) :
      P(Reduce.fitted(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan, _))
    Replayed.new_lift(target, 0.0,
      result => P(Reduce.allocated(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan, result)),
      output_fits => start_cast(Shape.elements(output), plan, target, shape, fill, ops, pair => P(Reduce.finish(target, pair)),
        loop_lift(Shape.elements(output), plan, target, shape, fill, ops, 0n, count,
          Equal.trans(Nat, Shape.elements(output), Shape.logical_numel(Shape.dimensions(output)), count,
            Shape.elements_exact(output), contribution_count(shape, plan, same)),
          contribution_room(shape, plan, same), plan_left(target, shape, plan, planned), plan_right(target, shape, plan, planned),
          pair => P(Reduce.finish(target, pair)), h))(fits, output_fits))

def fiber_equation(+target: Shape.Shape, +shape: Shape.Shape, +plan: Shape.BroadcastPlan,
  +planned: {Shape.broadcast_checked(target, shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  +same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>},
  +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Some{total(plan, shape, fill, ops, index, Shape.logical_numel(Shape.dimensions(shape)))} ==
    ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))),
      index, Shape.logical_numel(Shape.dimensions(shape))) : Maybe<&2, F32>}:
  +count = Shape.logical_numel(Shape.dimensions(shape))
  %plan_right(target, shape, plan, planned) :
    {Some{total(plan, _, fill, ops, index, count)} == ReduceLaws.fiber(plan, Contracts.words(_, fill, ops, Shape.logical_numel(Shape.dimensions(_))), index, count) :
      Maybe<&2, F32>}
  Equal.sym(Maybe<&2, F32>,
    ReduceLaws.fiber(plan, Contracts.words(Shape.right_shape(plan), fill, ops, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))), index, count),
    Some{total(plan, Shape.right_shape(plan), fill, ops, index, count)},
    FoldProof.fiber_total(plan, index, fill, ops, count, contribution_room(shape, plan, same)))

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
  +left = plan_left(target, shape, plan, planned)
  +room = contribution_room(shape, plan, same)
  Laws.ReduceToContract{
    run_lift(target, shape, plan, planned, same, fill, ops, result => {ReduceLaws.succeeded(result) == True{} : Bool},
      input_after => output_after => {==})(fits),
    run_lift(target, shape, plan, planned, same, fill, ops,
      result => {Laws.output_dimensions(result) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
      input_after => output_after => {==})(fits),
    index => inside =>
      +inside = inside
      run_lift(target, shape, plan, planned, same, fill, ops,
        result => {Laws.output_point(result, index) ==
          ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, count), index, count) : Maybe<&2, F32>},
        input_after => output_after =>
          +history = history(plan, target, shape, fill, ops, count)
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{target, Core.TTensor{Replayed.storage(target, 0.0, history), output_after}}, index)),
            Some{total(plan, shape, fill, ops, index, count)},
            ReduceLaws.fiber(plan, Contracts.words(shape, fill, ops, count), index, count),
            Equal.trans(Maybe<&2, F32>,
              Core.dynamic_read_value(Core.read_dynamic(Core.Packed{target, Core.TTensor{Replayed.storage(target, 0.0, history), output_after}}, index)),
              Some{Replayed.value(target, 0.0, history, index)}, Some{total(plan, shape, fill, ops, index, count)},
              PointwiseProof.replayed_point(target, 0.0, history, output_after, index, inside),
              Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(target, 0.0, history, index),
                total(plan, shape, fill, ops, index, count),
                history_value(plan, target, shape, fill, ops, left, count, room, index))),
            fiber_equation(target, shape, plan, planned, same, fill, ops, index)))(fits),
    other => inside =>
      run_lift(target, shape, plan, planned, same, fill, ops,
        result => {Laws.input_point(result, other) == Some{Replayed.value(shape, fill, ops, other)} : Maybe<&2, F32>},
        input_after => output_after =>
          PointwiseProof.input_point(~Shape.BroadcastPlan, ~ReduceFold.contribution_offset, plan, count, shape, fill, ops,
            input_after, other, inside))(fits)}

def Laws.reduce_to_fiber(target, shape, plan, planned, same, fill, ops, fits):
  reduce_to_fiber(target, shape, plan, planned, same, fill, ops, fits)
