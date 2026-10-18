import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../reduction.bend as Reduction
import ../word_conversion.bend as Conversion
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./fold.bend as Fold
import ./fold_records.bend as Records
import ./fold_proof.bend as Proof
import ./sum.bend as Sum
import ./LAWS.bend as Laws

def sequential_snoc(+values: List<&2, F32>, +word: F32, +accumulator: F32) ->
  {Reduction.sequential_sum(List.append(&2, F32, values, [word]), accumulator) ==
    F32.add(Reduction.sequential_sum(values, accumulator), word) : F32}:
  match values:
    case Nil{}:
      {==}
    case +value <> +rest:
      sequential_snoc(rest, word, F32.add(accumulator, value))

def total_words(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Fold.total(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, shape, fill, ops, count) ==
    Reduction.sequential_sum(Laws.words(shape, fill, ops, count), 0.0) : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +word = Replayed.value(shape, fill, ops, U32.from_nat(previous))
      %Equal.sym(F32, Reduction.sequential_sum(List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]), 0.0),
        F32.add(Reduction.sequential_sum(Laws.words(shape, fill, ops, previous), 0.0), word),
        sequential_snoc(Laws.words(shape, fill, ops, previous), word, 0.0)) :
        {F32.add(Fold.total(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, shape, fill, ops, previous), word) == _ : F32}
      Equal.cong(F32, F32, total => F32.add(total, word),
        Fold.total(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, shape, fill, ops, previous),
        Reduction.sequential_sum(Laws.words(shape, fill, ops, previous), 0.0),
        total_words(shape, fill, ops, previous))

def position_inside(+count: Nat, +bound: Nat, +room: {Nat.is_le(count, bound) == True{} : Bool}) ->
  {Pointwise.inside(~Unit, ~Sum.position, Unit{}, count, bound) == True{} : Bool}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      %Equal.sym(Bool, Pointwise.inside(~Unit, ~Sum.position, Unit{}, previous, bound), True{},
        position_inside(previous, bound, Conversion.le_predecessor(previous, bound, room))) :
        {Bool.and(_, Nat.is_lt(previous, bound)) == True{} : Bool}
      %Equal.sym(Bool, Nat.is_lt(previous, bound), True{}, Conversion.le_successor_lt(previous, bound, room)) :
        {Bool.and(True{}, _) == True{} : Bool}
      {==}

def elements_room(+shape: Shape.Shape) ->
  {Nat.is_le(Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %Equal.sym(Nat, Shape.elements(shape), Shape.numel(Shape.extents(shape)), Shape.elements_exact(shape)) :
    {Nat.is_le(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(shape)))

def sum_ordered(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Records.FoldContract<shape, Reduction.sequential_sum(Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), 0.0),
    other => Replayed.value(shape, fill, ops, other),
    Sum.sum(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>:
  +count = Shape.elements(shape)
  %Shape.elements_exact(shape) :
    Records.FoldContract<shape, Reduction.sequential_sum(Laws.words(shape, fill, ops, _), 0.0),
      other => Replayed.value(shape, fill, ops, other),
      Fold.run(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, count,
        Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>
  %total_words(shape, fill, ops, count) :
    Records.FoldContract<shape, _,
      other => Replayed.value(shape, fill, ops, other),
      Fold.run(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, count,
        Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>
  Proof.contract(~Unit, ~Sum.add, ~Unit, ~Sum.position, ~Sum.every, Unit{}, Unit{}, shape, fill, ops, count,
    position_inside(count, Shape.logical_numel(Shape.dimensions(shape)), elements_room(shape)), fits)
