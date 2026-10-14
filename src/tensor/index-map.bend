import Base
import ../shape/core.bend as Shape
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./word-conversion.bend as Conversion
import ./from-values-contract.bend as FromValues
import ./contracts/replayed.bend as Replayed
import ./contracts/pointwise.bend as Pointwise
import ./contracts/pointwise-proof.bend as PointwiseProof

# Index-map copy: out[i] = input[map(plan, i)] for every output index i, in
# increasing i, on a replayed input. It is the pointwise family with a scalar
# function that keeps its first operand and the same map on both operands, so
# the read and write histories and the final-value theorem are the pointwise
# ones. An instance supplies `map` and runs its own loop; these lemmas carry
# one loop step and the final state.

def keep(operation: Unit, word: F32, other: F32) -> F32:
  word

def source(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +count: Nat) -> Storage.Storage:
  PointwiseProof.input_storage(~P, ~map, plan, input, fill, ops, count)

def writes(~P: Data, ~map: P -> Nat -> Nat, +count: Nat, +plan: P, +input: Shape.Shape, +fill: F32,
  +ops: List<&2, StorageLaws.Op>) -> List<&2, StorageLaws.Op>:
  Pointwise.writes(~Unit, ~keep, ~P, ~map, ~map, count, Unit{}, plan, input, fill, ops, input, fill, ops)

def target(~P: Data, ~map: P -> Nat -> Nat, +count: Nat, +plan: P, +input: Shape.Shape, +fill: F32,
  +ops: List<&2, StorageLaws.Op>, +output: Shape.Shape) -> Storage.Storage:
  Replayed.storage(output, 0.0, writes(~P, ~map, count, plan, input, fill, ops))

def shift(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +output: Shape.Shape, +cursor: Nat, -K: Storage.Storage -> Storage.Storage -> Type,
  next: K(source(~P, ~map, plan, input, fill, ops, 1n+cursor), target(~P, ~map, 1n+cursor, plan, input, fill, ops, output))) ->
  K(Replayed.storage(input, fill, Replayed.snoc(List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, cursor)),
      StorageLaws.Read{U32.from_nat(map(plan, cursor))})),
    Replayed.storage(output, 0.0, Replayed.snoc(writes(~P, ~map, cursor, plan, input, fill, ops),
      StorageLaws.Write{U32.from_nat(cursor),
        Replayed.value(input, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, cursor)), U32.from_nat(map(plan, cursor)))}))):
  +read = {StorageLaws.Read{U32.from_nat(map(plan, cursor))} : StorageLaws.Op}
  +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, cursor))
  +word = U32.from_nat(map(plan, cursor))
  +history = writes(~P, ~map, cursor, plan, input, fill, ops)
  -written = Replayed.storage(output, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(cursor), Replayed.value(input, fill, seen, word)}))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, list => Replayed.storage(input, fill, list),
    List.append(&2, StorageLaws.Op, ops, Replayed.snoc(Pointwise.reads(~P, ~map, plan, cursor), read)), Replayed.snoc(seen, read),
    PointwiseProof.append_snoc(ops, Pointwise.reads(~P, ~map, plan, cursor), read)) :
    K(_, written)
  %Equal.cong(F32, Storage.Storage, value => Replayed.storage(output, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(cursor), value})),
    Replayed.value(input, fill, ops, word), Replayed.value(input, fill, seen, word),
    Equal.sym(F32, Replayed.value(input, fill, seen, word), Replayed.value(input, fill, ops, word),
      PointwiseProof.value_reads(~P, ~map, plan, cursor, input, fill, ops, word))) :
    K(source(~P, ~map, plan, input, fill, ops, 1n+cursor), _)
  next

def copied_point(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +output: Shape.Shape,
  -fits: {Lifecycle.logical_length(target(~P, ~map, Shape.logical_numel(Shape.dimensions(output)), plan, input, fill, ops, output)) ==
    Shape.logical_numel(Shape.dimensions(output)) : Nat},
  +index: U32, +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output,
    Core.TTensor{target(~P, ~map, Shape.logical_numel(Shape.dimensions(output)), plan, input, fill, ops, output), fits}}, index)) ==
    Some{Replayed.value(input, fill, ops, U32.from_nat(map(plan, U32.to_nat(index))))} : Maybe<&2, F32>}:
  +total = Shape.logical_numel(Shape.dimensions(output))
  +history = writes(~P, ~map, total, plan, input, fill, ops)
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{output, Core.TTensor{Replayed.storage(output, 0.0, history), fits}}, index)),
    Some{Replayed.value(output, 0.0, history, index)},
    Some{Replayed.value(input, fill, ops, U32.from_nat(map(plan, U32.to_nat(index))))},
    PointwiseProof.replayed_point(output, 0.0, history, fits, index, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(output, 0.0, history, index),
      Replayed.value(input, fill, ops, U32.from_nat(map(plan, U32.to_nat(index)))),
      PointwiseProof.writes_value(~Unit, ~keep, ~P, ~map, ~map, total, Unit{}, plan, input, fill, ops, input, fill, ops, output, index,
        inside, Conversion.le_reflexive(total))))

def kept(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +input: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(source(~P, ~map, plan, input, fill, ops, count)) == Shape.logical_numel(Shape.dimensions(input)) : Nat},
  +other: U32, inside: {Nat.is_lt(U32.to_nat(other), Shape.logical_numel(Shape.dimensions(input))) == True{} : Bool}) ->
  {Core.read_value(input, Core.read_flat(input, Core.TTensor{source(~P, ~map, plan, input, fill, ops, count), fits}, other)) ==
    Some{Replayed.value(input, fill, ops, other)} : Maybe<&2, F32>}:
  +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, count))
  Equal.trans(Maybe<&2, F32>, Core.read_value(input, Core.read_flat(input, Core.TTensor{Replayed.storage(input, fill, seen), fits}, other)),
    Some{Replayed.value(input, fill, seen, other)}, Some{Replayed.value(input, fill, ops, other)},
    Replayed.read_found(input, fill, seen, fits, other, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(input, fill, seen, other), Replayed.value(input, fill, ops, other),
      PointwiseProof.value_reads(~P, ~map, plan, count, input, fill, ops, other)))

def stored_at(+shape: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Some{Replayed.value(shape, 0.0, FromValues.writes(values, 0n), index)} == List.get(&2, F32, values, U32.to_nat(index)) : Maybe<&2, F32>}:
  FromValues.written_at(values, 0n, Shape.count(shape), index, 0.0, U32.to_nat(index), FromValues.start_room(shape, values, counted), {==},
    %Equal.sym(Nat, List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), counted) :
      {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
    inside)
