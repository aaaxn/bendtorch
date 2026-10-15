import Base
import ../../shape/core.bend as Shape
import ../../storage/LAWS.bend as StorageLaws
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise-proof.bend as PointwiseProof
import ./LAWS.bend as Laws

# A word appended to two equal word lists, equal words, gives equal lists.
def words_snoc(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +word: F32, +other: F32,
  +before: List<&2, F32>, rest: {Laws.words(shape, fill, ops, previous) == before : List<&2, F32>},
  last: {word == other : F32}) ->
  {List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, before, [other]) : List<&2, F32>}:
  %rest :
    {List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, _, [other]) : List<&2, F32>}
  %last :
    {List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, Laws.words(shape, fill, ops, previous), [_]) : List<&2, F32>}
  {==}

# Reads appended to a history list the same words.
def words_reads(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +reads: Nat, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +count: Nat) ->
  {Laws.words(shape, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, reads)), count) ==
    Laws.words(shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +position = U32.from_nat(previous)
      +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, reads))
      words_snoc(shape, fill, seen, previous, Replayed.value(shape, fill, seen, position), Replayed.value(shape, fill, ops, position),
        Laws.words(shape, fill, ops, previous), words_reads(~P, ~map, plan, reads, shape, fill, ops, previous),
        PointwiseProof.value_reads(~P, ~map, plan, reads, shape, fill, ops, position))

# Equal lists with equal words appended are equal.
def snoc_equal(+before: List<&2, F32>, +after: List<&2, F32>, +word: F32, +other: F32,
  rest: {before == after : List<&2, F32>}, last: {word == other : F32}) ->
  {List.append(&2, F32, before, [word]) == List.append(&2, F32, after, [other]) : List<&2, F32>}:
  %rest : {List.append(&2, F32, before, [word]) == List.append(&2, F32, _, [other]) : List<&2, F32>}
  %last : {List.append(&2, F32, before, [word]) == List.append(&2, F32, before, [_]) : List<&2, F32>}
  {==}
