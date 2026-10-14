import Base
import ../shape/core.bend as Shape
import ../shape/LAWS.bend as ShapeLaws
import ../shape/PROOF.bend as ShapeProof
import ../storage/LAWS.bend as StorageLaws
import ../storage/arith.bend as Arith
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/word-conversion.bend as Conversion
import ../tensor/projection-LAWS.bend as Projection
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise.bend as Pointwise
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/broadcast.bend as Broadcast
import ../tensor/contracts/broadcast-proof.bend as BroadcastProof
import ../tensor/contracts/fold.bend as Fold
import ../tensor/contracts/fold-records.bend as Records
import ../tensor/contracts/fold-proof.bend as FoldProof
import ../tensor/contracts/sum.bend as Sum
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/PROOF.bend as ContractsProof
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fold.bend as ReduceFold
import ./reduce-fold-LAWS.bend as Laws

def offsets_walk(+plan: Shape.BroadcastPlan,
  +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>,
  +position: Nat, +bound: {True{} == Nat.is_lt(position, Shape.elements(Shape.output_shape(plan))) : Bool}) ->
  {Projection.offset_spec(plan, position) ==
    Some{(U32.from_nat(Broadcast.left_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
        Shape.extents(Shape.output_shape(plan)), map, position)),
      U32.from_nat(Broadcast.right_walk(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)),
        Shape.extents(Shape.output_shape(plan)), map, position)))} : Maybe<&1, U32 & U32>}:
  match map:
    case Shape.Indexed{+rows, +exact}:
      +left = Shape.extents(Shape.left_shape(plan))
      +right = Shape.extents(Shape.right_shape(plan))
      +output = Shape.extents(Shape.output_shape(plan))
      +index = {Shape.FIndex{position, bound} : Shape.FlatIndex<Shape.output_shape(plan)>}
      +slot = ShapeLaws.fin_of(Shape.numel(output), position)
      +x = U32.from_nat(Shape.walk_left(rows, position))
      +y = U32.from_nat(Shape.walk_right(rows, position))
      %ShapeLaws.project_left_index_refines(plan, Shape.Indexed{rows, exact}, index) :
        {Projection.pair_offsets(_, ShapeLaws.flatten_spec(right, ShapeLaws.project_right_coordinate(left, right, output, Shape.alignment(plan), slot))) ==
          Some{(x, y)} : Maybe<&1, U32 & U32>}
      %ShapeLaws.project_right_index_refines(plan, Shape.Indexed{rows, exact}, index) :
        {Projection.pair_offsets(Some{x}, _) == Some{(x, y)} : Maybe<&1, U32 & U32>}
      {==}
    case Shape.Unindexed{zero}:
      +output = Shape.output_shape(plan)
      Empty.absurd({Projection.offset_spec(plan, position) == Some{(U32.from_nat(0n), U32.from_nat(0n))} : Maybe<&1, U32 & U32>},
        Arith.lt_zero(position,
          %Equal.trans(Nat, Shape.elements(output), Shape.numel(Shape.extents(output)), 0n, Shape.elements_exact(output),
            Equal.sym(Nat, 0n, Shape.numel(Shape.extents(output)), zero)) : {Nat.is_lt(position, _) == True{} : Bool}
          Equal.sym(Bool, True{}, Nat.is_lt(position, Shape.elements(output)), bound)))

def length_snoc(+values: List<&2, F32>, +word: F32) ->
  {List.length(&2, F32, List.append(&2, F32, values, [word])) == 1n+List.length(&2, F32, values) : Nat}:
  match values:
    case Nil{}:
      {==}
    case +value <> +rest:
      Equal.cong(Nat, Nat, n => 1n+n, List.length(&2, F32, List.append(&2, F32, rest, [word])), 1n+List.length(&2, F32, rest),
        length_snoc(rest, word))

def words_length(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {List.length(&2, F32, Contracts.words(shape, fill, ops, count)) == count : Nat}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +before = Contracts.words(shape, fill, ops, previous)
      Equal.trans(Nat, List.length(&2, F32, List.append(&2, F32, before, [Replayed.value(shape, fill, ops, U32.from_nat(previous))])),
        1n+List.length(&2, F32, before), 1n+previous,
        length_snoc(before, Replayed.value(shape, fill, ops, U32.from_nat(previous))),
        Equal.cong(Nat, Nat, n => 1n+n, List.length(&2, F32, before), previous, words_length(shape, fill, ops, previous)))

def word_at_front(+values: List<&2, F32>, +word: F32, +position: Nat,
  below: {Nat.is_lt(position, List.length(&2, F32, values)) == True{} : Bool}) ->
  {ReduceLaws.word_at(List.append(&2, F32, values, [word]), position) == ReduceLaws.word_at(values, position) : Maybe<&2, F32>}:
  match values position:
    case Nil{} _:
      Empty.absurd({ReduceLaws.word_at([word], position) == None{} : Maybe<&2, F32>}, Arith.lt_zero(position, below))
    case +value <> +rest 0n:
      {==}
    case +value <> +rest 1n+next:
      word_at_front(rest, word, next, below)

def word_at_last(+values: List<&2, F32>, +word: F32) ->
  {ReduceLaws.word_at(List.append(&2, F32, values, [word]), List.length(&2, F32, values)) == Some{word} : Maybe<&2, F32>}:
  match values:
    case Nil{}:
      {==}
    case +value <> +rest:
      word_at_last(rest, word)

def words_step(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +position: Nat,
  below: {Nat.is_lt(position, 1n+previous) == True{} : Bool}, decision: Shape.BoolCase<Nat.is_eq(position, previous)>,
  earlier: @inside: {Nat.is_lt(position, previous) == True{} : Bool} ->
    {ReduceLaws.word_at(Contracts.words(shape, fill, ops, previous), position) ==
      Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}) ->
  {ReduceLaws.word_at(Contracts.words(shape, fill, ops, 1n+previous), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{equal}:
      +before = Contracts.words(shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(previous))
      +inside = PointwiseProof.lt_succ_ne(position, previous, below, equal)
      Equal.trans(Maybe<&2, F32>, ReduceLaws.word_at(List.append(&2, F32, before, [last]), position), ReduceLaws.word_at(before, position),
        Some{Replayed.value(shape, fill, ops, U32.from_nat(position))},
        word_at_front(before, last, position,
          %Equal.sym(Nat, List.length(&2, F32, before), previous, words_length(shape, fill, ops, previous)) :
            {Nat.is_lt(position, _) == True{} : Bool}
          inside),
        earlier(inside))
    case Shape.BTrue{equal}:
      +before = Contracts.words(shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(previous))
      %Equal.sym(Nat, position, previous, Arith.eq_true(position, previous, Equal.sym(Bool, True{}, Nat.is_eq(position, previous), equal))) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) == Some{Replayed.value(shape, fill, ops, U32.from_nat(_))} : Maybe<&2, F32>}
      %words_length(shape, fill, ops, previous) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) == Some{last} : Maybe<&2, F32>}
      word_at_last(before, last)

def words_at(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +position: Nat,
  below: {Nat.is_lt(position, count) == True{} : Bool}) ->
  {ReduceLaws.word_at(Contracts.words(shape, fill, ops, count), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}:
  match count:
    case 0n:
      Empty.absurd({ReduceLaws.word_at(Nil{}, position) == Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>},
        Arith.lt_zero(position, below))
    case 1n+previous:
      words_step(shape, fill, ops, previous, position, below, Shape.bool_case(Nat.is_eq(position, previous)),
        inside => words_at(shape, fill, ops, previous, position, inside))

def matched_chosen(+same: Bool, +total: F32, +word: F32) ->
  {ReduceLaws.matched(Some{total}, Some{word}, same) == Some{Fold.chosen(~Unit, ~Sum.add, Unit{}, same, total, word)} : Maybe<&2, F32>}:
  match same:
    case False{}:
      {==}
    case True{}:
      {==}

def fiber_total(+plan: Shape.BroadcastPlan, +index: U32, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {ReduceLaws.fiber(plan,
    Contracts.words(Shape.right_shape(plan), fill, ops, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))), index, count) ==
    Some{Fold.total(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, ReduceFold.Cell{plan, index},
      Shape.right_shape(plan), fill, ops, count)} : Maybe<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+k:
      +right = Shape.right_shape(plan)
      +output = Shape.output_shape(plan)
      +limit = Shape.logical_numel(Shape.dimensions(output))
      +values = Contracts.words(right, fill, ops, Shape.logical_numel(Shape.dimensions(right)))
      +cell : ReduceFold.Cell = ReduceFold.Cell{plan, index}
      +lm = ReduceFold.target_offset(plan, k)
      +rm = ReduceFold.contribution_offset(plan, k)
      +before = Fold.total(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, cell, right, fill, ops, k)
      +word = Replayed.value(right, fill, ops, U32.from_nat(rm))
      +lands = Nat.is_eq(lm, U32.to_nat(index))
      +after : Maybe<&2, F32> = Some{Fold.chosen(~Unit, ~Sum.add, Unit{}, lands, before, word)}
      +below = Conversion.le_successor_lt(k, limit, room)
      +rm_below = BroadcastProof.right_walk_bound(plan, Shape.index_map(plan), k, below)
      +lm_below = BroadcastProof.left_walk_bound(plan, Shape.index_map(plan), k, below)
      %Equal.sym(Maybe<&1, U32 & U32>, Projection.offset_spec(plan, k), Some{(U32.from_nat(lm), U32.from_nat(rm))},
        offsets_walk(plan, Shape.index_map(plan), k, BroadcastProof.output_inside(plan, k, below))) :
        {ReduceLaws.contribute(values, index, ReduceLaws.fiber(plan, values, index, k), _) == after : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, ReduceLaws.fiber(plan, values, index, k), Some{before},
        fiber_total(plan, index, fill, ops, k, Conversion.le_predecessor(k, limit, room))) :
        {ReduceLaws.matched(_, ReduceLaws.word_at(values, U32.to_nat(U32.from_nat(rm))), Nat.is_eq(U32.to_nat(U32.from_nat(lm)), U32.to_nat(index))) ==
          after : Maybe<&2, F32>}
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(rm)), rm, PointwiseProof.word_exact(rm, right, rm_below)) :
        {ReduceLaws.matched(Some{before}, ReduceLaws.word_at(values, _), Nat.is_eq(U32.to_nat(U32.from_nat(lm)), U32.to_nat(index))) ==
          after : Maybe<&2, F32>}
      %Equal.sym(Maybe<&2, F32>, ReduceLaws.word_at(values, rm), Some{word},
        words_at(right, fill, ops, Shape.logical_numel(Shape.dimensions(right)), rm, rm_below)) :
        {ReduceLaws.matched(Some{before}, _, Nat.is_eq(U32.to_nat(U32.from_nat(lm)), U32.to_nat(index))) == after : Maybe<&2, F32>}
      %Equal.sym(Nat, U32.to_nat(U32.from_nat(lm)), lm, PointwiseProof.word_exact(lm, Shape.left_shape(plan), lm_below)) :
        {ReduceLaws.matched(Some{before}, Some{word}, Nat.is_eq(_, U32.to_nat(index))) == after : Maybe<&2, F32>}
      matched_chosen(lands, before, word)

def source_inside(+plan: Shape.BroadcastPlan, +index: U32, +count: Nat,
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}) ->
  {Pointwise.inside(~ReduceFold.Cell, ~ReduceFold.source, ReduceFold.Cell{plan, index}, count,
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +cell : ReduceFold.Cell = ReduceFold.Cell{plan, index}
      +limit = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
      +bound = Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))
      %Equal.sym(Bool, Pointwise.inside(~ReduceFold.Cell, ~ReduceFold.source, cell, previous, bound), True{},
        source_inside(plan, index, previous, Conversion.le_predecessor(previous, limit, room))) :
        {Bool.and(_, Nat.is_lt(ReduceFold.source(cell, previous), bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(ReduceFold.source(cell, previous), bound), True{},
        BroadcastProof.right_walk_bound(plan, Shape.index_map(plan), previous, Conversion.le_successor_lt(previous, limit, room))) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def contribution_room(+plan: Shape.BroadcastPlan,
  same: {Shape.extents(Shape.output_shape(plan)) == Shape.extents(Shape.right_shape(plan)) : List<&2, Nat>}) ->
  {Nat.is_le(Shape.elements(Shape.right_shape(plan)), Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) == True{} : Bool}:
  +right = Shape.right_shape(plan)
  +limit = Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))
  %Equal.sym(Nat, Shape.elements(right), Shape.numel(Shape.extents(right)), Shape.elements_exact(right)) :
    {Nat.is_le(_, limit) == True{} : Bool}
  %Equal.cong(List<&2, Nat>, Nat, extents => Shape.numel(extents), Shape.extents(Shape.output_shape(plan)), Shape.extents(right), same) :
    {Nat.is_le(_, limit) == True{} : Bool}
  Conversion.le_reflexive(limit)

def fiber_contract(+plan: Shape.BroadcastPlan, +index: U32, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), fill, ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat},
  +expected: Maybe<&2, F32>, +total: F32, +count: Nat,
  evidence: Records.FoldContract<Shape.right_shape(plan), total,
    other => Replayed.value(Shape.right_shape(plan), fill, ops, other),
    Fold.run(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, ReduceFold.Cell{plan, index}, count,
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), fill, ops), fits}})>,
  equation: {expected == Some{total} : Maybe<&2, F32>}) ->
  Laws.FiberContract<Shape.right_shape(plan), expected,
    other => Replayed.value(Shape.right_shape(plan), fill, ops, other),
    Fold.run(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, ReduceFold.Cell{plan, index}, count,
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), fill, ops), fits}})>:
  Records.FoldContract{value, kept} = evidence
  %Equal.sym(Maybe<&2, F32>, expected, Some{total}, equation) :
    Laws.FiberContract<Shape.right_shape(plan), _,
      other => Replayed.value(Shape.right_shape(plan), fill, ops, other),
      Fold.run(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, ReduceFold.Cell{plan, index}, count,
        Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), fill, ops), fits}})>
  Laws.FiberContract{value, kept}

def reduce_fold_fiber(+plan: Shape.BroadcastPlan,
  same: {Shape.extents(Shape.output_shape(plan)) == Shape.extents(Shape.right_shape(plan)) : List<&2, Nat>},
  +index: U32, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(Shape.right_shape(plan), fill, ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan))) : Nat}) ->
  Laws.FiberContract<Shape.right_shape(plan),
    ReduceLaws.fiber(plan,
      Contracts.words(Shape.right_shape(plan), fill, ops, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
      index, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))),
    other => Replayed.value(Shape.right_shape(plan), fill, ops, other),
    ReduceFold.cell(plan, index,
      Core.Packed{Shape.right_shape(plan), Core.TTensor{Replayed.storage(Shape.right_shape(plan), fill, ops), fits}})>:
  +right = Shape.right_shape(plan)
  +count = Shape.elements(right)
  +values = Contracts.words(right, fill, ops, Shape.logical_numel(Shape.dimensions(right)))
  +cell : ReduceFold.Cell = ReduceFold.Cell{plan, index}
  +total = Fold.total(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, cell, right, fill, ops, count)
  +room = contribution_room(plan, same)
  fiber_contract(plan, index, fill, ops, fits,
    ReduceLaws.fiber(plan, values, index, Shape.logical_numel(Shape.dimensions(right))), total, count,
    FoldProof.contract(~Unit, ~Sum.add, ~ReduceFold.Cell, ~ReduceFold.source, ~ReduceFold.lands, Unit{}, cell, right, fill, ops, count,
      source_inside(plan, index, count, room), fits),
    %Shape.elements_exact(right) : {ReduceLaws.fiber(plan, values, index, _) == Some{total} : Maybe<&2, F32>}
    fiber_total(plan, index, fill, ops, count, room))

def Laws.reduce_fold_fiber(plan, same, index, fill, ops, fits):
  reduce_fold_fiber(plan, same, index, fill, ops, fits)
