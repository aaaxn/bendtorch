import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/PROOF.bend as StorageProof
import ../storage/api.bend as Api
import ../storage/arith.bend as Arith
import ./core.bend as Core
import ./lifecycle.bend as Lifecycle
import ./allocation-spec.bend as Allocation
import ./allocation-PROOF.bend as AllocationProof
import ./word-conversion.bend as Conversion
import ./observation.bend as Observation
import ./read-contract.bend as Reads
import ./from-values-records.bend as Laws

def writes(values: List<&2, F32>, +cursor: Nat) -> List<&2, StorageLaws.Op>:
  match values:
    case Nil{}:
      Nil{}
    case value <> rest:
      StorageLaws.Write{U32.from_nat(cursor), value} <> writes(rest, 1n+cursor)

def base(+shape: Shape.Shape, +prefix: List<&2, StorageLaws.Op>) -> Storage.Storage:
  StorageLaws.replay(prefix, StorageLaws.created(Shape.count(shape), 0.0))

def Filled(+shape: Shape.Shape, +values: List<&2, F32>, +cursor: Nat, +prefix: List<&2, StorageLaws.Op>,
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type) -> Type:
  @-fits: {Lifecycle.logical_length(StorageLaws.replay(writes(values, cursor), base(shape, prefix))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Done{Core.TTensor{StorageLaws.replay(writes(values, cursor), base(shape, prefix)), fits}})

def storage_of(-shape: Shape.Shape, tensor: Core.Tensor<shape>) -> Storage.Storage:
  Core.TTensor{storage, valid} = tensor
  storage

def written_storage(-shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>) -> Storage.Storage:
  (tensor, result) = pair
  storage_of(shape, tensor)

def room_next(+cursor: Nat, +remaining: Nat, +bound: Nat,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+cursor, remaining), bound) == True{} : Bool}:
  %Arith.add_succ(cursor, remaining) : {Nat.is_le(_, bound) == True{} : Bool}
  room

def cursor_le(+cursor: Nat, +remaining: Nat, +bound: Nat,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), bound) == True{} : Bool}) ->
  {Nat.is_le(cursor, bound) == True{} : Bool}:
  Conversion.le_trans(cursor, Nat.add(cursor, 1n+remaining), bound, Api.le_add_right(cursor, 1n+remaining), room)

def cursor_lt(+cursor: Nat, +remaining: Nat, +bound: Nat,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), bound) == True{} : Bool}) ->
  {Nat.is_lt(cursor, bound) == True{} : Bool}:
  Arith.lt_le_trans(cursor, 1n+cursor, bound, Arith.lt_succ(cursor, cursor, Api.le_refl(cursor)),
    Conversion.le_trans(1n+cursor, Nat.add(1n+cursor, remaining), bound, Api.le_add_right(1n+cursor, remaining),
      room_next(cursor, remaining, bound, room)))

def cursor_exact(+cursor: Nat, +remaining: Nat, +limit: U32,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(cursor)) == cursor : Nat}:
  Conversion.natural_word_exact(cursor, limit, cursor_le(cursor, remaining, U32.to_nat(limit), room))

def cursor_inside(+cursor: Nat, +remaining: Nat, +limit: U32,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), U32.to_nat(limit)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), U32.to_nat(limit)) == True{} : Bool}:
  +room = room
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_exact(cursor, remaining, limit, room)) :
    {Nat.is_lt(_, U32.to_nat(limit)) == True{} : Bool}
  cursor_lt(cursor, remaining, U32.to_nat(limit), room)

def lt_different(+small: Nat, +big: Nat, earlier: {Nat.is_lt(small, big) == True{} : Bool}) ->
  {Nat.is_eq(big, small) == False{} : Bool}:
  match small big:
    case 0n 0n:
      Empty.absurd({True{} == False{} : Bool}, Arith.clash(earlier))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      lt_different(p, q, earlier)

def missed(+cursor: Nat, +remaining: Nat, +limit: U32, +index: U32, +value: F32, +base: F32,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), U32.to_nat(limit)) == True{} : Bool},
  earlier: {Nat.is_lt(U32.to_nat(index), cursor) == True{} : Bool}) ->
  {StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base) == base : F32}:
  +room = room
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_exact(cursor, remaining, limit, room)) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(_, U32.to_nat(limit)), Nat.is_eq(_, U32.to_nat(index))), value, base) == base : F32}
  %Equal.sym(Bool, Nat.is_lt(cursor, U32.to_nat(limit)), True{}, cursor_lt(cursor, remaining, U32.to_nat(limit), room)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(cursor, U32.to_nat(index))), value, base) == base : F32}
  %Equal.sym(Bool, Nat.is_eq(cursor, U32.to_nat(index)), False{}, lt_different(U32.to_nat(index), cursor, earlier)) :
    {Bool.pick(F32, _, value, base) == base : F32}
  {==}

def later_writes(+values: List<&2, F32>, +cursor: Nat, +limit: U32, +index: U32, +base: F32,
  room: {Nat.is_le(Nat.add(cursor, List.length(&2, F32, values)), U32.to_nat(limit)) == True{} : Bool},
  earlier: {Nat.is_lt(U32.to_nat(index), cursor) == True{} : Bool}) ->
  {StorageLaws.expected(writes(values, cursor), U32.to_nat(limit), index, base) == base : F32}:
  match values:
    case Nil{}:
      {==}
    case +value <> +rest:
      +room = room
      +earlier = earlier
      Equal.trans(F32,
        StorageLaws.expected(writes(rest, 1n+cursor), U32.to_nat(limit), index,
          StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base)),
        StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base), base,
        later_writes(rest, 1n+cursor, limit, index,
          StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base),
          room_next(cursor, List.length(&2, F32, rest), U32.to_nat(limit), room),
          Arith.lt_le_trans(U32.to_nat(index), cursor, 1n+cursor, earlier, Arith.le_succ(cursor, cursor, Api.le_refl(cursor)))),
        missed(cursor, List.length(&2, F32, rest), limit, index, value, base, room, earlier))

def hit(+cursor: Nat, +remaining: Nat, +limit: U32, +index: U32, +value: F32, +base: F32,
  room: {Nat.is_le(Nat.add(cursor, 1n+remaining), U32.to_nat(limit)) == True{} : Bool},
  at: {U32.to_nat(index) == cursor : Nat}) ->
  {StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base) == value : F32}:
  +room = room
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(cursor)), cursor, cursor_exact(cursor, remaining, limit, room)) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(_, U32.to_nat(limit)), Nat.is_eq(_, U32.to_nat(index))), value, base) == value : F32}
  %Equal.sym(Nat, U32.to_nat(index), cursor, at) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(cursor, U32.to_nat(limit)), Nat.is_eq(cursor, _)), value, base) == value : F32}
  %Equal.sym(Bool, Nat.is_lt(cursor, U32.to_nat(limit)), True{}, cursor_lt(cursor, remaining, U32.to_nat(limit), room)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(cursor, cursor)), value, base) == value : F32}
  %Equal.sym(Bool, Nat.is_eq(cursor, cursor), True{}, Arith.eq_refl(cursor)) :
    {Bool.pick(F32, _, value, base) == value : F32}
  {==}

def written_at(+values: List<&2, F32>, +cursor: Nat, +limit: U32, +index: U32, +base: F32, +offset: Nat,
  +room: {Nat.is_le(Nat.add(cursor, List.length(&2, F32, values)), U32.to_nat(limit)) == True{} : Bool},
  at: {U32.to_nat(index) == Nat.add(cursor, offset) : Nat},
  below: {Nat.is_lt(offset, List.length(&2, F32, values)) == True{} : Bool}) ->
  {Some{StorageLaws.expected(writes(values, cursor), U32.to_nat(limit), index, base)} == List.get(&2, F32, values, offset) : Maybe<&2, F32>}:
  match values offset:
    case Nil{} _:
      Empty.absurd({Some{base} == None{} : Maybe<&2, F32>}, Arith.lt_zero(offset, below))
    case +value <> +rest 0n:
      +exact = Equal.trans(Nat, U32.to_nat(index), Nat.add(cursor, 0n), cursor, at, Conversion.add_zero(cursor))
      Equal.cong(F32, Maybe<&2, F32>, found => Some{found},
        StorageLaws.expected(writes(rest, 1n+cursor), U32.to_nat(limit), index,
          StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base)), value,
        Equal.trans(F32,
          StorageLaws.expected(writes(rest, 1n+cursor), U32.to_nat(limit), index,
            StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base)),
          StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base), value,
          later_writes(rest, 1n+cursor, limit, index,
            StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base),
            room_next(cursor, List.length(&2, F32, rest), U32.to_nat(limit), room),
            %Equal.sym(Nat, U32.to_nat(index), cursor, exact) : {Nat.is_lt(_, 1n+cursor) == True{} : Bool}
            Arith.lt_succ(cursor, cursor, Api.le_refl(cursor))),
          hit(cursor, List.length(&2, F32, rest), limit, index, value, base, room, exact)))
    case +value <> +rest 1n+next:
      written_at(rest, 1n+cursor, limit, index,
        StorageLaws.latest(U32.to_nat(limit), StorageLaws.Write{U32.from_nat(cursor), value}, index, base), next,
        room_next(cursor, List.length(&2, F32, rest), U32.to_nat(limit), room),
        Equal.trans(Nat, U32.to_nat(index), Nat.add(cursor, 1n+next), 1n+Nat.add(cursor, next), at, Arith.add_succ(cursor, next)),
        below)

def replay_snoc(+ops: List<&2, StorageLaws.Op>, +op: StorageLaws.Op, -storage: Storage.Storage) ->
  {StorageLaws.replay(List.append(&2, StorageLaws.Op, ops, [op]), storage) == StorageLaws.step(StorageLaws.replay(ops, storage), op) : Storage.Storage}:
  match ops:
    case Nil{}:
      {==}
    case +first <> +rest:
      replay_snoc(rest, op, StorageLaws.step(storage, first))

def moved_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {written_storage(shape, Core.write_result(shape, original, valid, index, value, pair, same)) ==
    StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, pair) : Storage.Storage}:
  (storage, result) = pair
  {==}

def done_projection(-shape: Shape.Shape, -original: Storage.Storage,
  -valid: {Lifecycle.logical_length(original) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
  -same: {pair == Storage.set(original, index, value) : Storage.Storage & Result<&2, &2, Storage.Error, Unit>}) ->
  {Observation.write_done_pair(shape, Core.write_result(shape, original, valid, index, value, pair, same)) ==
    Result.is_done(&2, &2, Storage.Error, Unit, StorageLaws.outcome(Result<&2, &2, Storage.Error, Unit>, pair)) : Bool}:
  (storage, result) = pair
  match result:
    case Fail{error}:
      {==}
    case Done{unit}:
      {==}

def write_moved(+shape: Shape.Shape, +prefix: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(base(shape, prefix)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32) ->
  {written_storage(shape, Core.write_flat(shape, Core.TTensor{base(shape, prefix), valid}, index, value)) ==
    StorageLaws.written(base(shape, prefix), index, value) : Storage.Storage}:
  moved_projection(shape, base(shape, prefix), valid, index, value, Storage.set(base(shape, prefix), index, value), {==})

def write_done(+shape: Shape.Shape, +prefix: List<&2, StorageLaws.Op>,
  -valid: {Lifecycle.logical_length(base(shape, prefix)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32, +value: F32,
  inside: {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == True{} : Bool}) ->
  {Observation.write_done_pair(shape, Core.write_flat(shape, Core.TTensor{base(shape, prefix), valid}, index, value)) == True{} : Bool}:
  Equal.trans(Bool, Observation.write_done_pair(shape, Core.write_flat(shape, Core.TTensor{base(shape, prefix), valid}, index, value)),
    Result.is_done(&2, &2, Storage.Error, Unit, StorageLaws.status(base(shape, prefix), index, value)), True{},
    done_projection(shape, base(shape, prefix), valid, index, value, Storage.set(base(shape, prefix), index, value), {==}),
    Equal.cong(Result<&2, &2, Storage.Error, Unit>, Bool, Result.is_done(&2, &2, Storage.Error, Unit),
      StorageLaws.status(base(shape, prefix), index, value), Done{Unit{}},
      StorageLaws.replay_set(Shape.count(shape), 0.0, prefix, index, value, Conversion.shape_count_within(shape), inside)))

def moved_lift(+shape: Shape.Shape, -rest: List<&2, F32>, +cursor: Nat, -source: Storage.Storage, -target: Storage.Storage,
  moved: {source == target : Storage.Storage}, -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  next: @-valid: {Lifecycle.logical_length(source) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(rest, shape, Core.TTensor{source, valid}, 1n+cursor))) ->
  @-valid: {Lifecycle.logical_length(target) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(rest, shape, Core.TTensor{target, valid}, 1n+cursor)):
  %moved : (@-valid: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(rest, shape, Core.TTensor{_, valid}, 1n+cursor)))
  next

def written_step(+shape: Shape.Shape, +value: F32, -rest: List<&2, F32>, +cursor: Nat, +prefix: List<&2, StorageLaws.Op>, +index: U32,
  pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, Unit>,
  done: {Observation.write_done_pair(shape, pair) == True{} : Bool},
  moved: {written_storage(shape, pair) == StorageLaws.written(base(shape, prefix), index, value) : Storage.Storage},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  next: @-valid: {Lifecycle.logical_length(base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}]))) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(rest, shape, Core.TTensor{base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}])), valid}, 1n+cursor))) ->
  P(Core.fill_values_step(shape, tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), pair)):
  match pair:
    case (Core.TTensor{storage, valid}, Fail{error}):
      Empty.absurd(P(Fail{error}), Arith.clash(done))
    case (Core.TTensor{storage, valid}, Done{unit}):
      moved_lift(shape, rest, cursor, base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}])), storage,
        Equal.trans(Storage.Storage, base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}])),
          StorageLaws.written(base(shape, prefix), index, value), storage,
          replay_snoc(prefix, StorageLaws.Write{index, value}, StorageLaws.created(Shape.count(shape), 0.0)),
          Equal.sym(Storage.Storage, storage, StorageLaws.written(base(shape, prefix), index, value), moved)),
        P, next)(valid)

def decided_step(+shape: Shape.Shape, +value: F32, -rest: List<&2, F32>, +cursor: Nat, +prefix: List<&2, StorageLaws.Op>, +index: U32,
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), cursor)>,
  converted: {U32.to_nat(index) == cursor : Nat},
  inside: {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == True{} : Bool},
  -valid: {Lifecycle.logical_length(base(shape, prefix)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type,
  next: @-valid: {Lifecycle.logical_length(base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}]))) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(rest, shape, Core.TTensor{base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{index, value}])), valid}, 1n+cursor))) ->
  P(Core.fill_values_decided(shape, cursor, index, value, Core.TTensor{base(shape, prefix), valid},
    tensor => Core.fill_values(rest, shape, tensor, 1n+cursor), decision)):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(P(Fail{Core.StorageError{Storage.InvalidIndex{}}}),
        Nats.contradiction(Nat.is_eq(U32.to_nat(index), cursor),
          AllocationProof.natural_equal_true(U32.to_nat(index), cursor, converted), equal))
    case Shape.BTrue{equal}:
      written_step(shape, value, rest, cursor, prefix, index,
        Core.write_flat(shape, Core.TTensor{base(shape, prefix), valid}, index, value),
        write_done(shape, prefix, valid, index, value, inside), write_moved(shape, prefix, valid, index, value), P, next)

def filled_next(+shape: Shape.Shape, +value: F32, +rest: List<&2, F32>, +cursor: Nat, +prefix: List<&2, StorageLaws.Op>,
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type, h: Filled(shape, value <> rest, cursor, prefix, P)) ->
  Filled(shape, rest, 1n+cursor, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{U32.from_nat(cursor), value}]), P):
  %Equal.sym(Storage.Storage, base(shape, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{U32.from_nat(cursor), value}])),
    StorageLaws.written(base(shape, prefix), U32.from_nat(cursor), value),
    replay_snoc(prefix, StorageLaws.Write{U32.from_nat(cursor), value}, StorageLaws.created(Shape.count(shape), 0.0))) :
    (@-fits: {Lifecycle.logical_length(StorageLaws.replay(writes(rest, 1n+cursor), _)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
      P(Done{Core.TTensor{StorageLaws.replay(writes(rest, 1n+cursor), _), fits}}))
  h

def fill_lift(+values: List<&2, F32>, +shape: Shape.Shape, +cursor: Nat, +prefix: List<&2, StorageLaws.Op>,
  room: {Nat.is_le(Nat.add(cursor, List.length(&2, F32, values)), U32.to_nat(Shape.count(shape))) == True{} : Bool},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type, h: Filled(shape, values, cursor, prefix, P)) ->
  @-valid: {Lifecycle.logical_length(base(shape, prefix)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    P(Core.fill_values(values, shape, Core.TTensor{base(shape, prefix), valid}, cursor)):
  match values:
    case Nil{}:
      h
    case +value <> +rest:
      +room = room
      valid => decided_step(shape, value, rest, cursor, prefix, U32.from_nat(cursor),
        Shape.bool_case(Nat.is_eq(U32.to_nat(U32.from_nat(cursor)), cursor)),
        cursor_exact(cursor, List.length(&2, F32, rest), Shape.count(shape), room),
        cursor_inside(cursor, List.length(&2, F32, rest), Shape.count(shape), room), valid, P,
        fill_lift(rest, shape, 1n+cursor, List.append(&2, StorageLaws.Op, prefix, [StorageLaws.Write{U32.from_nat(cursor), value}]),
          room_next(cursor, List.length(&2, F32, rest), U32.to_nat(Shape.count(shape)), room), P,
          filled_next(shape, value, rest, cursor, prefix, P, h)))

def start_room(+shape: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Nat.is_le(Nat.add(0n, List.length(&2, F32, values)), U32.to_nat(Shape.count(shape))) == True{} : Bool}:
  %Equal.sym(Nat, List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), counted) :
    {Nat.is_le(_, U32.to_nat(Shape.count(shape))) == True{} : Bool}
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), Reads.count_exact(shape)) :
    {Nat.is_le(Shape.logical_numel(Shape.dimensions(shape)), _) == True{} : Bool}
  Api.le_refl(Shape.logical_numel(Shape.dimensions(shape)))

def counted_lift(+shape: Shape.Shape, +values: List<&2, F32>,
  decision: Shape.BoolCase<Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type, h: Filled(shape, values, 0n, Nil{}, P)) ->
  P(Core.from_values_counted(shape, values, decision)):
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(P(Fail{Core.ValueCountMismatch{}}),
        Nats.contradiction(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape)),
          AllocationProof.natural_equal_true(List.length(&2, F32, values), Shape.elements(shape),
            Equal.trans(Nat, List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements(shape), counted,
              Equal.sym(Nat, Shape.elements(shape), Shape.logical_numel(Shape.dimensions(shape)), Shape.elements_exact(shape)))),
          equal))
    case Shape.BTrue{equal}:
      Reads.new_lift(shape, 0.0, result => P(Core.from_values_new(shape, values, result)),
        fill_lift(values, shape, 0n, Nil{}, start_room(shape, values, counted), P, h))

def from_values_lift(+shape: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -P: Result<&2, &1, Core.Error, Core.Tensor<shape>> -> Type, h: Filled(shape, values, 0n, Nil{}, P)) ->
  P(Core.from_values(shape, values)):
  counted_lift(shape, values, Shape.bool_case(Nat.is_eq(List.length(&2, F32, values), Shape.elements(shape))), counted, P, h)

def filled_length(+shape: Shape.Shape, +values: List<&2, F32>) ->
  {Lifecycle.logical_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}:
  Equal.trans(Nat, Lifecycle.logical_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))),
    StorageLaws.size(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))), Shape.logical_numel(Shape.dimensions(shape)),
    Equal.sym(Nat, StorageLaws.size(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))),
      Lifecycle.logical_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))),
      Reads.size_is_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{})))),
    Equal.trans(Nat, StorageLaws.size(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))),
      U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)),
      StorageLaws.replay_size(Shape.count(shape), 0.0, writes(values, 0n), Conversion.shape_count_within(shape)),
      Reads.count_exact(shape)))

def filled_value(+shape: Shape.Shape, +values: List<&2, F32>,
  +counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -fits: {Lifecycle.logical_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  {Observation.point(shape, index, Done{Core.TTensor{StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), fits}}) ==
    List.get(&2, F32, values, U32.to_nat(index)) : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Observation.point(shape, index, Done{Core.TTensor{StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), fits}}),
    Lifecycle.read_value(Storage.get(StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), index)),
    List.get(&2, F32, values, U32.to_nat(index)),
    Core.read_result_value(shape, StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), fits, index,
      Storage.get(StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), index), {==}),
    Equal.trans(Maybe<&2, F32>,
      Lifecycle.read_value(Storage.get(StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), index)),
      Some{StorageLaws.expected(writes(values, 0n), U32.to_nat(Shape.count(shape)), index, 0.0)},
      List.get(&2, F32, values, U32.to_nat(index)),
      Reads.stored_value(StorageLaws.replay(writes(values, 0n), base(shape, Nil{})), index,
        StorageLaws.expected(writes(values, 0n), U32.to_nat(Shape.count(shape)), index, 0.0),
        StorageLaws.replay_get(Shape.count(shape), 0.0, writes(values, 0n), index, Conversion.shape_count_within(shape),
          Reads.word_inside(shape, index, inside))),
      written_at(values, 0n, Shape.count(shape), index, 0.0, U32.to_nat(index), start_room(shape, values, counted), {==},
        %Equal.sym(Nat, List.length(&2, F32, values), Shape.logical_numel(Shape.dimensions(shape)), counted) :
          {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
        inside)))

def from_values_contract(+shape: Shape.Shape, +values: List<&2, F32>,
  counted: {List.length(&2, F32, values) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}) ->
  Laws.FromValuesContract<shape, values>:
  +counted = counted
  Laws.FromValuesContract{
    from_values_lift(shape, values, counted,
      result => {Result.is_done(&2, &1, Core.Error, Core.Tensor<shape>, result) == True{} : Bool}, fits => {==}),
    from_values_lift(shape, values, counted,
      result => {Observation.dimensions(shape, result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>}, fits => {==}),
    from_values_lift(shape, values, counted,
      result => {Allocation.tensor_length(shape, result) == Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>},
      fits => Equal.cong(Nat, Maybe<&2, Nat>, number => Some{number},
        Lifecycle.logical_length(StorageLaws.replay(writes(values, 0n), base(shape, Nil{}))), Shape.logical_numel(Shape.dimensions(shape)),
        filled_length(shape, values))),
    from_values_lift(shape, values, counted,
      result => {Observation.count(shape, result) == Some{Shape.count(shape)} : Maybe<&2, U32>}, fits => {==}),
    index => inside =>
      +inside = inside
      from_values_lift(shape, values, counted,
        result => {Observation.point(shape, index, result) == List.get(&2, F32, values, U32.to_nat(index)) : Maybe<&2, F32>},
        fits => filled_value(shape, values, counted, index, inside, fits))}
