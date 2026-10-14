import Base
import ../../shape/core.bend as Shape
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise-proof.bend as PointwiseProof
import ./fold.bend as Fold
import ./fold-records.bend as Records

def Shaped(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +cursor: Nat, -storage: Storage.Storage, +accumulator: F32,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  @-fits: {Lifecycle.logical_length(storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q(Fold.loop(~O, ~g, ~P, ~ia, fuel, operation, plan, cursor, Core.Packed{shape, Core.TTensor{storage, fits}}, accumulator))

def Start(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  Shaped(~O, ~g, ~P, ~ia, fuel, operation, plan, shape, cursor, PointwiseProof.input_storage(~P, ~ia, plan, shape, fill, ops, cursor),
    Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, cursor), Q)

def Final(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  @-fits: {Lifecycle.logical_length(PointwiseProof.input_storage(~P, ~ia, plan, shape, fill, ops, count)) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q((Core.Packed{shape, Core.TTensor{PointwiseProof.input_storage(~P, ~ia, plan, shape, fill, ops, count), fits}},
      Done{Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count)}))

def final_at(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, -end: Nat, -count: Nat,
  same: {end == count : Nat}, -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  h: Final(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, end, Q)) ->
  Final(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, Q):
  %same : Final(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, _, Q)
  h

def shift(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  next: Start(~O, ~g, ~P, ~ia, rest, operation, plan, shape, fill, ops, 1n+cursor, Q)) ->
  Shaped(~O, ~g, ~P, ~ia, rest, operation, plan, shape, 1n+cursor,
    Replayed.storage(shape, fill, Replayed.snoc(List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~ia, plan, cursor)),
      StorageLaws.Read{U32.from_nat(ia(plan, cursor))})),
    g(operation, Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, cursor),
      Replayed.value(shape, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~ia, plan, cursor)), U32.from_nat(ia(plan, cursor)))),
    Q):
  +read = {StorageLaws.Read{U32.from_nat(ia(plan, cursor))} : StorageLaws.Op}
  +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +before = Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, cursor)
  +x = Replayed.value(shape, fill, seen, U32.from_nat(ia(plan, cursor)))
  +original = Replayed.value(shape, fill, ops, U32.from_nat(ia(plan, cursor)))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(shape, fill, history),
    List.append(&2, StorageLaws.Op, ops, Replayed.snoc(Pointwise.reads(~P, ~ia, plan, cursor), read)), Replayed.snoc(seen, read),
    PointwiseProof.append_snoc(ops, Pointwise.reads(~P, ~ia, plan, cursor), read)) :
    Shaped(~O, ~g, ~P, ~ia, rest, operation, plan, shape, 1n+cursor, _, g(operation, before, x), Q)
  %Equal.cong(F32, F32, w => g(operation, before, w), original, x,
    Equal.sym(F32, x, original, PointwiseProof.value_reads(~P, ~ia, plan, cursor, shape, fill, ops, U32.from_nat(ia(plan, cursor))))) :
    Shaped(~O, ~g, ~P, ~ia, rest, operation, plan, shape, 1n+cursor, PointwiseProof.input_storage(~P, ~ia, plan, shape, fill, ops, 1n+cursor), _, Q)
  next

def step_lift(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  inside: {Nat.is_lt(U32.to_nat(U32.from_nat(ia(plan, cursor))), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  next: Start(~O, ~g, ~P, ~ia, rest, operation, plan, shape, fill, ops, 1n+cursor, Q)) ->
  Start(~O, ~g, ~P, ~ia, 1n+rest, operation, plan, shape, fill, ops, cursor, Q):
  +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +before = Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, cursor)
  fits =>
    Replayed.read_lift(shape, fill, seen, fits, U32.from_nat(ia(plan, cursor)), inside,
      pair => Q(Fold.step(~O, ~g, operation,
        input => value => Fold.loop(~O, ~g, ~P, ~ia, rest, operation, plan, 1n+cursor, input, value),
        before, Core.read_dynamic_result(shape, pair))),
      after => shift(~O, ~g, ~P, ~ia, rest, operation, plan, shape, fill, ops, cursor, Q, next)(after))

def loop_lift(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +cursor: Nat, +count: Nat,
  +steps: {Nat.add(cursor, fuel) == count : Nat},
  +inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  h: Final(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, Q)) ->
  Start(~O, ~g, ~P, ~ia, fuel, operation, plan, shape, fill, ops, cursor, Q):
  match fuel:
    case 0n:
      final_at(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, cursor,
        Equal.sym(Nat, cursor, count, PointwiseProof.steps_done(cursor, count, steps)), Q, h)
    case 1n+rest:
      +below = PointwiseProof.cursor_below(cursor, rest, count, steps)
      step_lift(~O, ~g, ~P, ~ia, rest, operation, plan, shape, fill, ops, cursor,
        PointwiseProof.word_inside(ia(plan, cursor), shape,
          PointwiseProof.inside_at(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(shape)), inside, cursor, below)),
        Q, loop_lift(~O, ~g, ~P, ~ia, rest, operation, plan, shape, fill, ops, 1n+cursor, count,
          PointwiseProof.steps_next(cursor, rest, count, steps), inside, Q, h))

def start_cast(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  start: Start(~O, ~g, ~P, ~ia, fuel, operation, plan, shape, fill, ops, 0n, Q)) ->
  Shaped(~O, ~g, ~P, ~ia, fuel, operation, plan, shape, 0n, Replayed.storage(shape, fill, ops), 0.0, Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(shape, fill, history),
    List.append(&2, StorageLaws.Op, ops, Nil{}), ops, PointwiseProof.append_nil(ops)) :
    Shaped(~O, ~g, ~P, ~ia, fuel, operation, plan, shape, 0n, _, 0.0, Q)
  start

# The family result: for a replayed input whose index map stays in bounds,
# the loop returns the input with the reads it made appended to its history,
# and the fold of the words it read.
def run_lift(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  +inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -Q: Core.Dynamic & Result<&2, &1, Core.Error, F32> -> Type,
  h: Final(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, Q)) ->
  Shaped(~O, ~g, ~P, ~ia, count, operation, plan, shape, 0n, Replayed.storage(shape, fill, ops), 0.0, Q):
  start_cast(~O, ~g, ~P, ~ia, count, operation, plan, shape, fill, ops, Q,
    loop_lift(~O, ~g, ~P, ~ia, count, operation, plan, shape, fill, ops, 0n, count, {==}, inside, Q, h))

# The generic fold contract: one proof for every step function and every
# in-bounds index map.
def contract(~O: Data, ~g: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat,
  +operation: O, +plan: P, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat,
  +inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Records.FoldContract<shape, Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count),
    other => Replayed.value(shape, fill, ops, other),
    Fold.run(~O, ~g, ~P, ~ia, operation, plan, count, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>:
  Records.FoldContract{
    run_lift(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, inside,
      result => {Fold.value(result) == Some{Fold.total(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count)} : Maybe<&2, F32>},
      after => {==})(fits),
    other => within =>
      run_lift(~O, ~g, ~P, ~ia, operation, plan, shape, fill, ops, count, inside,
        result => {Fold.input_point(result, other) == Some{Replayed.value(shape, fill, ops, other)} : Maybe<&2, F32>},
        after => PointwiseProof.input_point(~P, ~ia, plan, count, shape, fill, ops, after, other, within))(fits)}

def PairShaped(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +right: Shape.Shape, +cursor: Nat,
  -left_storage: Storage.Storage, -right_storage: Storage.Storage, +accumulator: F32,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_storage) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_storage) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    Q(Fold.pair_loop(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, cursor,
      Core.Packed{left, Core.TTensor{left_storage, left_fits}}, Core.Packed{right, Core.TTensor{right_storage, right_fits}}, accumulator))

def PairStart(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  PairShaped(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, right, cursor,
    PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, cursor),
    PointwiseProof.input_storage(~P, ~ib, plan, right, right_fill, right_ops, cursor),
    Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor), Q)

def PairFinal(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, count)) ==
    Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(PointwiseProof.input_storage(~P, ~ib, plan, right, right_fill, right_ops, count)) ==
    Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    Q(((Core.Packed{left, Core.TTensor{PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, count), left_fits}},
      Core.Packed{right, Core.TTensor{PointwiseProof.input_storage(~P, ~ib, plan, right, right_fill, right_ops, count), right_fits}}),
      Done{Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count)}))

def pair_final_at(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, -end: Nat, -count: Nat,
  same: {end == count : Nat}, -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  h: PairFinal(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, end, Q)) ->
  PairFinal(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, Q):
  %same : PairFinal(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, _, Q)
  h

def pair_shift(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  next: PairStart(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  PairShaped(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, right, 1n+cursor,
    Replayed.storage(left, left_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor)),
      StorageLaws.Read{U32.from_nat(ia(plan, cursor))})),
    Replayed.storage(right, right_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor)),
      StorageLaws.Read{U32.from_nat(ib(plan, cursor))})),
    g(operation, Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor),
      Replayed.value(left, left_fill, List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor)), U32.from_nat(ia(plan, cursor))),
      Replayed.value(right, right_fill, List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor)), U32.from_nat(ib(plan, cursor)))),
    Q):
  +left_read = {StorageLaws.Read{U32.from_nat(ia(plan, cursor))} : StorageLaws.Op}
  +right_read = {StorageLaws.Read{U32.from_nat(ib(plan, cursor))} : StorageLaws.Op}
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor))
  +before = Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +x = Replayed.value(left, left_fill, left_seen, U32.from_nat(ia(plan, cursor)))
  +y = Replayed.value(right, right_fill, right_seen, U32.from_nat(ib(plan, cursor)))
  +left_word = Replayed.value(left, left_fill, left_ops, U32.from_nat(ia(plan, cursor)))
  +right_word = Replayed.value(right, right_fill, right_ops, U32.from_nat(ib(plan, cursor)))
  -right_after = Replayed.storage(right, right_fill, Replayed.snoc(right_seen, right_read))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(left, left_fill, history),
    List.append(&2, StorageLaws.Op, left_ops, Replayed.snoc(Pointwise.reads(~P, ~ia, plan, cursor), left_read)), Replayed.snoc(left_seen, left_read),
    PointwiseProof.append_snoc(left_ops, Pointwise.reads(~P, ~ia, plan, cursor), left_read)) :
    PairShaped(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, right, 1n+cursor, _, right_after, g(operation, before, x, y), Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(right, right_fill, history),
    List.append(&2, StorageLaws.Op, right_ops, Replayed.snoc(Pointwise.reads(~P, ~ib, plan, cursor), right_read)), Replayed.snoc(right_seen, right_read),
    PointwiseProof.append_snoc(right_ops, Pointwise.reads(~P, ~ib, plan, cursor), right_read)) :
    PairShaped(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, right, 1n+cursor,
      PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, 1n+cursor), _, g(operation, before, x, y), Q)
  %Equal.trans(F32, g(operation, before, left_word, right_word), g(operation, before, x, right_word), g(operation, before, x, y),
    Equal.cong(F32, F32, w => g(operation, before, w, right_word), left_word, x,
      Equal.sym(F32, x, left_word, PointwiseProof.value_reads(~P, ~ia, plan, cursor, left, left_fill, left_ops, U32.from_nat(ia(plan, cursor))))),
    Equal.cong(F32, F32, w => g(operation, before, x, w), right_word, y,
      Equal.sym(F32, y, right_word, PointwiseProof.value_reads(~P, ~ib, plan, cursor, right, right_fill, right_ops, U32.from_nat(ib(plan, cursor)))))) :
    PairShaped(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, right, 1n+cursor,
      PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, 1n+cursor),
      PointwiseProof.input_storage(~P, ~ib, plan, right, right_fill, right_ops, 1n+cursor), _, Q)
  next

def pair_step_lift(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  left_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(ia(plan, cursor))), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  right_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(ib(plan, cursor))), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  next: PairStart(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, Q)) ->
  PairStart(~O, ~g, ~P, ~ia, ~ib, 1n+rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor))
  +left_index = U32.from_nat(ia(plan, cursor))
  +right_index = U32.from_nat(ib(plan, cursor))
  +before = Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
  +x = Replayed.value(left, left_fill, left_seen, left_index)
  left_fits => right_fits =>
    Replayed.read_lift(left, left_fill, left_seen, left_fits, left_index, left_inside,
      pair => Q(Fold.pair_left(~O, ~g, operation, right_index,
        left => right => value => Fold.pair_loop(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, 1n+cursor, left, right, value),
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}}, before,
        Core.read_dynamic_result(left, pair))),
      left_after => Replayed.read_lift(right, right_fill, right_seen, right_fits, right_index, right_inside,
        pair => Q(Fold.pair_right(~O, ~g, operation,
          left => right => value => Fold.pair_loop(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, 1n+cursor, left, right, value),
          Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}},
          before, x, Core.read_dynamic_result(right, pair))),
        right_after => pair_shift(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor,
          Q, next)(left_after, right_after)))

def pair_loop_lift(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat, +count: Nat,
  +steps: {Nat.add(cursor, fuel) == count : Nat},
  +left_inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, count, Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  h: PairFinal(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, Q)) ->
  PairStart(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor, Q):
  match fuel:
    case 0n:
      pair_final_at(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, cursor,
        Equal.sym(Nat, cursor, count, PointwiseProof.steps_done(cursor, count, steps)), Q, h)
    case 1n+rest:
      +below = PointwiseProof.cursor_below(cursor, rest, count, steps)
      pair_step_lift(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, cursor,
        PointwiseProof.word_inside(ia(plan, cursor), left,
          PointwiseProof.inside_at(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(left)), left_inside, cursor, below)),
        PointwiseProof.word_inside(ib(plan, cursor), right,
          PointwiseProof.inside_at(~P, ~ib, plan, count, Shape.logical_numel(Shape.dimensions(right)), right_inside, cursor, below)),
        Q, pair_loop_lift(~O, ~g, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, count,
          PointwiseProof.steps_next(cursor, rest, count, steps), left_inside, right_inside, Q, h))

def pair_start_cast(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  start: PairStart(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, Q)) ->
  PairShaped(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, right, 0n,
    Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops), 0.0, Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(left, left_fill, history),
    List.append(&2, StorageLaws.Op, left_ops, Nil{}), left_ops, PointwiseProof.append_nil(left_ops)) :
    PairShaped(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, right, 0n, _, Replayed.storage(right, right_fill, right_ops), 0.0, Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, history => Replayed.storage(right, right_fill, history),
    List.append(&2, StorageLaws.Op, right_ops, Nil{}), right_ops, PointwiseProof.append_nil(right_ops)) :
    PairShaped(~O, ~g, ~P, ~ia, ~ib, fuel, operation, plan, left, right, 0n,
      PointwiseProof.input_storage(~P, ~ia, plan, left, left_fill, left_ops, 0n), _, 0.0, Q)
  start

# The pair result: both inputs come back with the reads appended to their
# histories, and the fold of the pairs of words read.
def pair_run_lift(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  +left_inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, count, Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, F32> -> Type,
  h: PairFinal(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, Q)) ->
  PairShaped(~O, ~g, ~P, ~ia, ~ib, count, operation, plan, left, right, 0n,
    Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops), 0.0, Q):
  pair_start_cast(~O, ~g, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, Q,
    pair_loop_lift(~O, ~g, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, 0n, count,
      {==}, left_inside, right_inside, Q, h))

# The generic pair fold contract: one proof for every step function and
# every pair of in-bounds index maps.
def pair_contract(~O: Data, ~g: O -> F32 -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat,
  +left_inside: {Pointwise.inside(~P, ~ia, plan, count, Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, count, Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.PairFoldContract<left, right,
    Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count),
    other => Replayed.value(left, left_fill, left_ops, other), other => Replayed.value(right, right_fill, right_ops, other),
    Fold.pair_run(~O, ~g, ~P, ~ia, ~ib, operation, plan, count,
      Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  Records.PairFoldContract{
    pair_run_lift(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, left_inside, right_inside,
      result => {Fold.pair_value(result) ==
        Some{Fold.pair_total(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count)} : Maybe<&2, F32>},
      left_after => right_after => {==})(left_fits, right_fits),
    other => within =>
      pair_run_lift(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, left_inside, right_inside,
        result => {Fold.left_point(result, other) == Some{Replayed.value(left, left_fill, left_ops, other)} : Maybe<&2, F32>},
        left_after => right_after => PointwiseProof.input_point(~P, ~ia, plan, count, left, left_fill, left_ops, left_after, other, within))(left_fits, right_fits),
    other => within =>
      pair_run_lift(~O, ~g, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, count, left_inside, right_inside,
        result => {Fold.right_point(result, other) == Some{Replayed.value(right, right_fill, right_ops, other)} : Maybe<&2, F32>},
        left_after => right_after => PointwiseProof.input_point(~P, ~ib, plan, count, right, right_fill, right_ops, right_after, other, within))(left_fits, right_fits)}
