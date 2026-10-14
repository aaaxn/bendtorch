import Base
import ../../shape/core.bend as Shape
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../word-conversion.bend as Conversion
import ../read-contract.bend as Reads
import ../from-values-contract.bend as FromValues
import ./replayed.bend as Replayed
import ./pointwise.bend as Pointwise
import ./pointwise-records.bend as Records

def append_nil(+ops: List<&2, StorageLaws.Op>) -> {List.append(&2, StorageLaws.Op, ops, Nil{}) == ops : List<&2, StorageLaws.Op>}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +rest:
      Equal.cong(List<&2, StorageLaws.Op>, List<&2, StorageLaws.Op>, tail => op <> tail,
        List.append(&2, StorageLaws.Op, rest, Nil{}), rest, append_nil(rest))

def append_snoc(+ops: List<&2, StorageLaws.Op>, +more: List<&2, StorageLaws.Op>, +last: StorageLaws.Op) ->
  {List.append(&2, StorageLaws.Op, ops, Replayed.snoc(more, last)) ==
    Replayed.snoc(List.append(&2, StorageLaws.Op, ops, more), last) : List<&2, StorageLaws.Op>}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +rest:
      Equal.cong(List<&2, StorageLaws.Op>, List<&2, StorageLaws.Op>, tail => op <> tail,
        List.append(&2, StorageLaws.Op, rest, Replayed.snoc(more, last)),
        Replayed.snoc(List.append(&2, StorageLaws.Op, rest, more), last), append_snoc(rest, more, last))

def expected_append(+ops: List<&2, StorageLaws.Op>, +more: List<&2, StorageLaws.Op>, +n: Nat, +index: U32, +word: F32) ->
  {StorageLaws.expected(List.append(&2, StorageLaws.Op, ops, more), n, index, word) ==
    StorageLaws.expected(more, n, index, StorageLaws.expected(ops, n, index, word)) : F32}:
  match ops:
    case Nil{}:
      {==}
    case +op <> +rest:
      expected_append(rest, more, n, index, StorageLaws.latest(n, op, index, word))

def expected_reads(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +n: Nat, +index: U32, +word: F32) ->
  {StorageLaws.expected(Pointwise.reads(~P, ~map, plan, count), n, index, word) == word : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      Equal.trans(F32,
        StorageLaws.expected(Pointwise.reads(~P, ~map, plan, 1n+previous), n, index, word),
        StorageLaws.expected(Pointwise.reads(~P, ~map, plan, previous), n, index, word), word,
        Replayed.expected_snoc(Pointwise.reads(~P, ~map, plan, previous), StorageLaws.Read{U32.from_nat(map(plan, previous))}, n, index, word),
        expected_reads(~P, ~map, plan, previous, n, index, word))

# The reads the loop makes leave every word of an input unchanged.
def value_reads(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +shape: Shape.Shape, +fill: F32,
  +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Replayed.value(shape, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, count)), index) ==
    Replayed.value(shape, fill, ops, index) : F32}:
  Equal.trans(F32,
    Replayed.value(shape, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, count)), index),
    StorageLaws.expected(Pointwise.reads(~P, ~map, plan, count), U32.to_nat(Shape.count(shape)), index, Replayed.value(shape, fill, ops, index)),
    Replayed.value(shape, fill, ops, index),
    expected_append(ops, Pointwise.reads(~P, ~map, plan, count), U32.to_nat(Shape.count(shape)), index, fill),
    expected_reads(~P, ~map, plan, count, U32.to_nat(Shape.count(shape)), index, Replayed.value(shape, fill, ops, index)))

def lt_add(+cursor: Nat, +rest: Nat) -> {Nat.is_lt(cursor, Nat.add(cursor, 1n+rest)) == True{} : Bool}:
  match cursor:
    case 0n:
      {==}
    case 1n+previous:
      lt_add(previous, rest)

def cursor_below(+cursor: Nat, +rest: Nat, +end: Nat, steps: {Nat.add(cursor, 1n+rest) == end : Nat}) ->
  {Nat.is_lt(cursor, end) == True{} : Bool}:
  %steps : {Nat.is_lt(cursor, _) == True{} : Bool}
  lt_add(cursor, rest)

def steps_next(+cursor: Nat, +rest: Nat, +end: Nat, steps: {Nat.add(cursor, 1n+rest) == end : Nat}) ->
  {Nat.add(1n+cursor, rest) == end : Nat}:
  Equal.trans(Nat, 1n+Nat.add(cursor, rest), Nat.add(cursor, 1n+rest), end,
    Equal.sym(Nat, Nat.add(cursor, 1n+rest), 1n+Nat.add(cursor, rest), Arith.add_succ(cursor, rest)), steps)

def steps_done(+cursor: Nat, +end: Nat, steps: {Nat.add(cursor, 0n) == end : Nat}) -> {cursor == end : Nat}:
  Equal.trans(Nat, cursor, Nat.add(cursor, 0n), end, Equal.sym(Nat, Nat.add(cursor, 0n), cursor, Conversion.add_zero(cursor)), steps)

def lt_succ_ne(+index: Nat, +previous: Nat, below: {Nat.is_lt(index, 1n+previous) == True{} : Bool},
  other: {False{} == Nat.is_eq(index, previous) : Bool}) -> {Nat.is_lt(index, previous) == True{} : Bool}:
  match index previous:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Arith.clash(other))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, Arith.lt_zero(p, below))
    case 1n+p 1n+q:
      lt_succ_ne(p, q, below, other)

def lt_succ_le(+index: Nat, +bound: Nat, below: {Nat.is_lt(index, bound) == True{} : Bool}) ->
  {Nat.is_le(1n+index, bound) == True{} : Bool}:
  match index bound:
    case 0n 0n:
      Empty.absurd({Nat.is_le(1n, 0n) == True{} : Bool}, Arith.clash(below))
    case 0n 1n+q:
      Arith.le_zero(q)
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(2n+p, 0n) == True{} : Bool}, Arith.clash(below))
    case 1n+p 1n+q:
      lt_succ_le(p, q, below)

# The bound `inside` promises at one index.
def inside_split(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +previous: Nat, +bound: Nat,
  all: {Pointwise.inside(~P, ~map, plan, 1n+previous, bound) == True{} : Bool}, +index: Nat,
  below: {Nat.is_lt(index, 1n+previous) == True{} : Bool},
  rec: {Pointwise.inside(~P, ~map, plan, previous, bound) == True{} : Bool} -> {Nat.is_lt(index, previous) == True{} : Bool} ->
    {Nat.is_lt(map(plan, index), bound) == True{} : Bool},
  decision: Shape.BoolCase<Nat.is_eq(index, previous)>) -> {Nat.is_lt(map(plan, index), bound) == True{} : Bool}:
  match decision:
    case Shape.BTrue{same}:
      %Equal.sym(Nat, index, previous, Lifecycle.natural_equal(index, previous, same)) : {Nat.is_lt(map(plan, _), bound) == True{} : Bool}
      Arith.and_right(Pointwise.inside(~P, ~map, plan, previous, bound), Nat.is_lt(map(plan, previous), bound), all)
    case Shape.BFalse{other}:
      rec(Arith.and_left(Pointwise.inside(~P, ~map, plan, previous, bound), Nat.is_lt(map(plan, previous), bound), all),
        lt_succ_ne(index, previous, below, other))

def inside_at(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +bound: Nat,
  all: {Pointwise.inside(~P, ~map, plan, count, bound) == True{} : Bool}, +index: Nat,
  below: {Nat.is_lt(index, count) == True{} : Bool}) -> {Nat.is_lt(map(plan, index), bound) == True{} : Bool}:
  match count:
    case 0n:
      Empty.absurd({Nat.is_lt(map(plan, index), bound) == True{} : Bool}, Arith.lt_zero(index, below))
    case 1n+previous:
      inside_split(~P, ~map, plan, previous, bound, all, index, below,
        earlier => before => inside_at(~P, ~map, plan, previous, bound, earlier, index, before),
        Shape.bool_case(Nat.is_eq(index, previous)))

def word_exact(+number: Nat, +shape: Shape.Shape,
  below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}:
  Conversion.natural_word_exact(number, Shape.count(shape),
    %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), Reads.count_exact(shape)) :
      {Nat.is_le(number, _) == True{} : Bool}
    Conversion.lt_le(number, Shape.logical_numel(Shape.dimensions(shape)), below))

def word_inside(+number: Nat, +shape: Shape.Shape,
  +below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(number)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(number)), number, word_exact(number, shape, below)) :
    {Nat.is_lt(_, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}
  below

def lt_count(+number: Nat, +shape: Shape.Shape,
  below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(number, U32.to_nat(Shape.count(shape))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(shape)), Shape.logical_numel(Shape.dimensions(shape)), Reads.count_exact(shape)) :
    {Nat.is_lt(number, _) == True{} : Bool}
  below

def write_hit(+shape: Shape.Shape, +previous: Nat, +index: U32, +word: F32, +old: F32,
  +room: {Nat.is_lt(previous, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  at: {U32.to_nat(index) == previous : Nat}) ->
  {StorageLaws.latest(U32.to_nat(Shape.count(shape)), StorageLaws.Write{U32.from_nat(previous), word}, index, old) == word : F32}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(previous)), previous, word_exact(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(_, U32.to_nat(Shape.count(shape))), Nat.is_eq(_, U32.to_nat(index))), word, old) == word : F32}
  %Equal.sym(Nat, U32.to_nat(index), previous, at) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(previous, U32.to_nat(Shape.count(shape))), Nat.is_eq(previous, _)), word, old) == word : F32}
  %Equal.sym(Bool, Nat.is_lt(previous, U32.to_nat(Shape.count(shape))), True{}, lt_count(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(previous, previous)), word, old) == word : F32}
  %Equal.sym(Bool, Nat.is_eq(previous, previous), True{}, Arith.eq_refl(previous)) :
    {Bool.pick(F32, _, word, old) == word : F32}
  {==}

def write_miss(+shape: Shape.Shape, +previous: Nat, +index: U32, +word: F32, +old: F32,
  +room: {Nat.is_lt(previous, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  earlier: {Nat.is_lt(U32.to_nat(index), previous) == True{} : Bool}) ->
  {StorageLaws.latest(U32.to_nat(Shape.count(shape)), StorageLaws.Write{U32.from_nat(previous), word}, index, old) == old : F32}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(previous)), previous, word_exact(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(_, U32.to_nat(Shape.count(shape))), Nat.is_eq(_, U32.to_nat(index))), word, old) == old : F32}
  %Equal.sym(Bool, Nat.is_lt(previous, U32.to_nat(Shape.count(shape))), True{}, lt_count(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(previous, U32.to_nat(index))), word, old) == old : F32}
  %Equal.sym(Bool, Nat.is_eq(previous, U32.to_nat(index)), False{}, FromValues.lt_different(U32.to_nat(index), previous, earlier)) :
    {Bool.pick(F32, _, word, old) == old : F32}
  {==}

def writes_split(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +previous: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +index: U32,
  below: {Nat.is_lt(U32.to_nat(index), 1n+previous) == True{} : Bool},
  +room: {Nat.is_le(1n+previous, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  rec: {Nat.is_lt(U32.to_nat(index), previous) == True{} : Bool} ->
    {Replayed.value(shape, 0.0, Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), index) ==
      Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32},
  decision: Shape.BoolCase<Nat.is_eq(U32.to_nat(index), previous)>) ->
  {Replayed.value(shape, 0.0, Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, 1n+previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), index) ==
    Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32}:
  match decision:
    case Shape.BTrue{same}:
      +written = Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      +history = Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
      +old = Replayed.value(shape, 0.0, history, index)
      +step = Replayed.expected_snoc(history, StorageLaws.Write{U32.from_nat(previous), written}, U32.to_nat(Shape.count(shape)), index, 0.0)
      +inside = Conversion.le_successor_lt(previous, Shape.logical_numel(Shape.dimensions(shape)), room)
      +at = Lifecycle.natural_equal(U32.to_nat(index), previous, same)
      Equal.trans(F32, Replayed.value(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(previous), written}), index),
        written, Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)),
        Equal.trans(F32, Replayed.value(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(previous), written}), index),
          StorageLaws.latest(U32.to_nat(Shape.count(shape)), StorageLaws.Write{U32.from_nat(previous), written}, index, old), written,
          step, write_hit(shape, previous, index, written, old, inside, at)),
        Equal.cong(Nat, F32, position => Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, position),
          previous, U32.to_nat(index), Equal.sym(Nat, U32.to_nat(index), previous, at)))
    case Shape.BFalse{other}:
      +written = Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      +history = Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
      +old = Replayed.value(shape, 0.0, history, index)
      +step = Replayed.expected_snoc(history, StorageLaws.Write{U32.from_nat(previous), written}, U32.to_nat(Shape.count(shape)), index, 0.0)
      +inside = Conversion.le_successor_lt(previous, Shape.logical_numel(Shape.dimensions(shape)), room)
      +earlier = lt_succ_ne(U32.to_nat(index), previous, below, other)
      Equal.trans(F32, Replayed.value(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(previous), written}), index),
        old, Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)),
        Equal.trans(F32, Replayed.value(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(previous), written}), index),
          StorageLaws.latest(U32.to_nat(Shape.count(shape)), StorageLaws.Write{U32.from_nat(previous), written}, index, old), old,
          step, write_miss(shape, previous, index, written, old, inside, earlier)),
        rec(earlier))

# After `count` cells, the output holds cell(i) at every i < count.
def writes_value(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +count: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +index: U32,
  below: {Nat.is_lt(U32.to_nat(index), count) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Replayed.value(shape, 0.0, Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops), index) ==
    Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32}:
  match count:
    case 0n:
      Empty.absurd({Replayed.value(shape, 0.0, Nil{}, index) ==
        Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32},
        Arith.lt_zero(U32.to_nat(index), below))
    case 1n+previous:
      writes_split(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, index,
        below, room,
        earlier => writes_value(~O, ~f, ~P, ~ia, ~ib, previous, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops,
          shape, index, earlier, Conversion.le_predecessor(previous, Shape.logical_numel(Shape.dimensions(shape)), room)),
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), previous)))

def input_storage(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +shape: Shape.Shape, +fill: F32,
  +ops: List<&2, StorageLaws.Op>, +count: Nat) -> Storage.Storage:
  Replayed.storage(shape, fill, List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, count)))

def output_storage(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +count: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape) -> Storage.Storage:
  Replayed.storage(shape, 0.0, Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops))

# Every successful erased length proof for the three storages gives Q of the
# loop state built from them.
def Shaped(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +right: Shape.Shape, +shape: Shape.Shape, +cursor: Nat,
  -left_storage: Storage.Storage, -right_storage: Storage.Storage, -output_storage: Storage.Storage,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(left_storage) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(right_storage) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_storage) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q(Pointwise.loop(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, shape, cursor,
      Pointwise.State{Core.Packed{left, Core.TTensor{left_storage, left_fits}}, Core.Packed{right, Core.TTensor{right_storage, right_fits}},
        Core.TTensor{output_storage, output_fits}}))

def Start(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  Shaped(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, right, shape, cursor,
    input_storage(~P, ~ia, plan, left, left_fill, left_ops, cursor), input_storage(~P, ~ib, plan, right, right_fill, right_ops, cursor),
    output_storage(~O, ~f, ~P, ~ia, ~ib, cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape), Q)

# Q holds of the result the loop returns after `count` cells, for every
# erased length proof.
def Final(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +count: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type) -> Type:
  @-left_fits: {Lifecycle.logical_length(input_storage(~P, ~ia, plan, left, left_fill, left_ops, count)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(input_storage(~P, ~ib, plan, right, right_fill, right_ops, count)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
  @-output_fits: {Lifecycle.logical_length(output_storage(~O, ~f, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape)) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q(((Core.Packed{left, Core.TTensor{input_storage(~P, ~ia, plan, left, left_fill, left_ops, count), left_fits}},
      Core.Packed{right, Core.TTensor{input_storage(~P, ~ib, plan, right, right_fill, right_ops, count), right_fits}}),
      Done{Core.Packed{shape, Core.TTensor{output_storage(~O, ~f, ~P, ~ia, ~ib, count, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape), output_fits}}}))

def final_at(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, -end: Nat, -count: Nat,
  same: {end == count : Nat},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Final(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, end, Q)) ->
  Final(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, count, Q):
  %same : Final(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, _, Q)
  h

# The storages one step leaves, as the next step expects them.
def shift(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +cursor: Nat,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  next: Start(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, 1n+cursor, Q)) ->
  Shaped(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, right, shape, 1n+cursor,
    Replayed.storage(left, left_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor)),
      StorageLaws.Read{U32.from_nat(ia(plan, cursor))})),
    Replayed.storage(right, right_fill, Replayed.snoc(List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor)),
      StorageLaws.Read{U32.from_nat(ib(plan, cursor))})),
    Replayed.storage(shape, 0.0, Replayed.snoc(
      Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops),
      StorageLaws.Write{U32.from_nat(cursor),
        f(operation, Replayed.value(left, left_fill, List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor)), U32.from_nat(ia(plan, cursor))),
          Replayed.value(right, right_fill, List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor)), U32.from_nat(ib(plan, cursor))))})),
    Q):
  +left_read = {StorageLaws.Read{U32.from_nat(ia(plan, cursor))} : StorageLaws.Op}
  +right_read = {StorageLaws.Read{U32.from_nat(ib(plan, cursor))} : StorageLaws.Op}
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor))
  +x = Replayed.value(left, left_fill, left_seen, U32.from_nat(ia(plan, cursor)))
  +y = Replayed.value(right, right_fill, right_seen, U32.from_nat(ib(plan, cursor)))
  +left_word = Replayed.value(left, left_fill, left_ops, U32.from_nat(ia(plan, cursor)))
  +right_word = Replayed.value(right, right_fill, right_ops, U32.from_nat(ib(plan, cursor)))
  +history = Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
  -left_after = Replayed.storage(left, left_fill, Replayed.snoc(left_seen, left_read))
  -right_after = Replayed.storage(right, right_fill, Replayed.snoc(right_seen, right_read))
  -output_after = Replayed.storage(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(cursor), f(operation, x, y)}))
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(left, left_fill, ops),
    List.append(&2, StorageLaws.Op, left_ops, Replayed.snoc(Pointwise.reads(~P, ~ia, plan, cursor), left_read)), Replayed.snoc(left_seen, left_read),
    append_snoc(left_ops, Pointwise.reads(~P, ~ia, plan, cursor), left_read)) :
    Shaped(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, right, shape, 1n+cursor, _, right_after, output_after, Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(right, right_fill, ops),
    List.append(&2, StorageLaws.Op, right_ops, Replayed.snoc(Pointwise.reads(~P, ~ib, plan, cursor), right_read)), Replayed.snoc(right_seen, right_read),
    append_snoc(right_ops, Pointwise.reads(~P, ~ib, plan, cursor), right_read)) :
    Shaped(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, right, shape, 1n+cursor,
      input_storage(~P, ~ia, plan, left, left_fill, left_ops, 1n+cursor), _, output_after, Q)
  %Equal.cong(F32, Storage.Storage, word => Replayed.storage(shape, 0.0, Replayed.snoc(history, StorageLaws.Write{U32.from_nat(cursor), word})),
    f(operation, left_word, right_word), f(operation, x, y),
    Equal.trans(F32, f(operation, left_word, right_word), f(operation, x, right_word), f(operation, x, y),
      Equal.cong(F32, F32, word => f(operation, word, right_word), left_word, x,
        Equal.sym(F32, x, left_word, value_reads(~P, ~ia, plan, cursor, left, left_fill, left_ops, U32.from_nat(ia(plan, cursor))))),
      Equal.cong(F32, F32, word => f(operation, x, word), right_word, y,
        Equal.sym(F32, y, right_word, value_reads(~P, ~ib, plan, cursor, right, right_fill, right_ops, U32.from_nat(ib(plan, cursor))))))) :
    Shaped(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, right, shape, 1n+cursor,
      input_storage(~P, ~ia, plan, left, left_fill, left_ops, 1n+cursor), input_storage(~P, ~ib, plan, right, right_fill, right_ops, 1n+cursor), _, Q)
  next

def step_lift(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +rest: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +cursor: Nat,
  left_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(ia(plan, cursor))), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  right_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(ib(plan, cursor))), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  output_inside: {Nat.is_lt(U32.to_nat(U32.from_nat(cursor)), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  next: Start(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, 1n+cursor, Q)) ->
  Start(~O, ~f, ~P, ~ia, ~ib, 1n+rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, cursor, Q):
  +left_seen = List.append(&2, StorageLaws.Op, left_ops, Pointwise.reads(~P, ~ia, plan, cursor))
  +right_seen = List.append(&2, StorageLaws.Op, right_ops, Pointwise.reads(~P, ~ib, plan, cursor))
  +history = Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, cursor, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
  +left_index = U32.from_nat(ia(plan, cursor))
  +right_index = U32.from_nat(ib(plan, cursor))
  +x = Replayed.value(left, left_fill, left_seen, left_index)
  +y = Replayed.value(right, right_fill, right_seen, right_index)
  left_fits => right_fits => output_fits =>
    Replayed.read_lift(left, left_fill, left_seen, left_fits, left_index, left_inside,
      pair => Q(Pointwise.left_step(~O, ~f, operation, shape, U32.from_nat(cursor), right_index,
        state => Pointwise.loop(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, shape, 1n+cursor, state),
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_seen), right_fits}},
        Core.TTensor{Replayed.storage(shape, 0.0, history), output_fits}, Core.read_dynamic_result(left, pair))),
      left_after => Replayed.read_lift(right, right_fill, right_seen, right_fits, right_index, right_inside,
        pair => Q(Pointwise.right_step(~O, ~f, operation, shape, U32.from_nat(cursor),
          state => Pointwise.loop(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, shape, 1n+cursor, state),
          Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}},
          Core.TTensor{Replayed.storage(shape, 0.0, history), output_fits}, x, Core.read_dynamic_result(right, pair))),
        right_after => Replayed.write_lift(shape, 0.0, history, output_fits, U32.from_nat(cursor), f(operation, x, y), output_inside,
          pair => Q(Pointwise.write_step(shape,
            state => Pointwise.loop(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, shape, 1n+cursor, state),
            Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, Replayed.snoc(left_seen, StorageLaws.Read{left_index})), left_after}},
            Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, Replayed.snoc(right_seen, StorageLaws.Read{right_index})), right_after}},
            pair)),
          output_after => shift(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, cursor,
            Q, next)(left_after, right_after, output_after))))

def loop_lift(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape, +cursor: Nat,
  +steps: {Nat.add(cursor, fuel) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +left_inside: {Pointwise.inside(~P, ~ia, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Final(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape,
    Shape.logical_numel(Shape.dimensions(shape)), Q)) ->
  Start(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, cursor, Q):
  match fuel:
    case 0n:
      final_at(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape,
        Shape.logical_numel(Shape.dimensions(shape)), cursor,
        Equal.sym(Nat, cursor, Shape.logical_numel(Shape.dimensions(shape)), steps_done(cursor, Shape.logical_numel(Shape.dimensions(shape)), steps)),
        Q, h)
    case 1n+rest:
      +below = cursor_below(cursor, rest, Shape.logical_numel(Shape.dimensions(shape)), steps)
      step_lift(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, cursor,
        word_inside(ia(plan, cursor), left, inside_at(~P, ~ia, plan, Shape.logical_numel(Shape.dimensions(shape)),
          Shape.logical_numel(Shape.dimensions(left)), left_inside, cursor, below)),
        word_inside(ib(plan, cursor), right, inside_at(~P, ~ib, plan, Shape.logical_numel(Shape.dimensions(shape)),
          Shape.logical_numel(Shape.dimensions(right)), right_inside, cursor, below)),
        word_inside(cursor, shape, below), Q,
        loop_lift(~O, ~f, ~P, ~ia, ~ib, rest, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, 1n+cursor,
          steps_next(cursor, rest, Shape.logical_numel(Shape.dimensions(shape)), steps), left_inside, right_inside, Q, h))

def start_cast(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +fuel: Nat, +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape,
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  start: Start(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, 0n, Q)) ->
  Shaped(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, right, shape, 0n,
    Replayed.storage(left, left_fill, left_ops), Replayed.storage(right, right_fill, right_ops), Replayed.storage(shape, 0.0, Nil{}), Q):
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(left, left_fill, ops),
    List.append(&2, StorageLaws.Op, left_ops, Nil{}), left_ops, append_nil(left_ops)) :
    Shaped(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, right, shape, 0n,
      _, Replayed.storage(right, right_fill, right_ops), Replayed.storage(shape, 0.0, Nil{}), Q)
  %Equal.cong(List<&2, StorageLaws.Op>, Storage.Storage, ops => Replayed.storage(right, right_fill, ops),
    List.append(&2, StorageLaws.Op, right_ops, Nil{}), right_ops, append_nil(right_ops)) :
    Shaped(~O, ~f, ~P, ~ia, ~ib, fuel, operation, plan, left, right, shape, 0n,
      input_storage(~P, ~ia, plan, left, left_fill, left_ops, 0n), _, Replayed.storage(shape, 0.0, Nil{}), Q)
  start

# The family result: for replayed inputs whose index maps stay in bounds, the
# loop returns both inputs with the reads it made appended to their histories
# and an output whose history writes cell(i) at i for every i, in order.
def run_lift(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape,
  +left_inside: {Pointwise.inside(~P, ~ia, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -Q: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic> -> Type,
  h: Final(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape,
    Shape.logical_numel(Shape.dimensions(shape)), Q)) ->
  @-left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat} ->
  @-right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat} ->
    Q(Pointwise.run(~O, ~f, ~P, ~ia, ~ib, operation, plan, shape,
      Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})):
  left_fits => right_fits =>
    Replayed.new_lift(shape, 0.0,
      result => Q(Pointwise.allocated(~O, ~f, ~P, ~ia, ~ib, operation, plan, shape,
        Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
        Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}}, result)),
      output_fits => start_cast(~O, ~f, ~P, ~ia, ~ib, Shape.elements(shape), operation, plan, left, left_fill, left_ops,
        right, right_fill, right_ops, shape, Q,
        loop_lift(~O, ~f, ~P, ~ia, ~ib, Shape.elements(shape), operation, plan, left, left_fill, left_ops, right, right_fill, right_ops,
          shape, 0n, Shape.elements_exact(shape), left_inside, right_inside, Q, h))(left_fits, right_fits, output_fits))

def dynamic_value(+shape: Shape.Shape, pair: Core.Tensor<shape> & Result<&2, &2, Core.Error, F32>) ->
  {Core.dynamic_read_value(Core.read_dynamic_result(shape, pair)) == Core.read_value(shape, pair) : Maybe<&2, F32>}:
  (tensor, result) = pair
  {==}

def replayed_point(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}, +index: U32,
  inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, index)),
    Core.read_value(shape, Core.read_flat(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, index)),
    Some{Replayed.value(shape, fill, ops, index)},
    dynamic_value(shape, Core.read_flat(shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}, index)),
    Replayed.read_found(shape, fill, ops, fits, index, inside))

def input_point(~P: Data, ~map: P -> Nat -> Nat, +plan: P, +count: Nat, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(input_storage(~P, ~map, plan, shape, fill, ops, count)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +index: U32, inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{input_storage(~P, ~map, plan, shape, fill, ops, count), fits}}, index)) ==
    Some{Replayed.value(shape, fill, ops, index)} : Maybe<&2, F32>}:
  +seen = List.append(&2, StorageLaws.Op, ops, Pointwise.reads(~P, ~map, plan, count))
  Equal.trans(Maybe<&2, F32>,
    Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, seen), fits}}, index)),
    Some{Replayed.value(shape, fill, seen, index)}, Some{Replayed.value(shape, fill, ops, index)},
    replayed_point(shape, fill, seen, fits, index, inside),
    Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, fill, seen, index), Replayed.value(shape, fill, ops, index),
      value_reads(~P, ~map, plan, count, shape, fill, ops, index)))

# The generic pointwise contract: one proof for every scalar function and
# every pair of in-bounds index maps.
def contract(~O: Data, ~f: O -> F32 -> F32 -> F32, ~P: Data, ~ia: P -> Nat -> Nat, ~ib: P -> Nat -> Nat,
  +operation: O, +plan: P, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +shape: Shape.Shape,
  +left_inside: {Pointwise.inside(~P, ~ia, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(left))) == True{} : Bool},
  +right_inside: {Pointwise.inside(~P, ~ib, plan, Shape.logical_numel(Shape.dimensions(shape)), Shape.logical_numel(Shape.dimensions(right))) == True{} : Bool},
  -left_fits: {Lifecycle.logical_length(Replayed.storage(left, left_fill, left_ops)) == Shape.logical_numel(Shape.dimensions(left)) : Nat},
  -right_fits: {Lifecycle.logical_length(Replayed.storage(right, right_fill, right_ops)) == Shape.logical_numel(Shape.dimensions(right)) : Nat}) ->
  Records.PointwiseContract<shape, left, right,
    index => Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, index),
    other => Replayed.value(left, left_fill, left_ops, other), other => Replayed.value(right, right_fill, right_ops, other),
    Pointwise.run(~O, ~f, ~P, ~ia, ~ib, operation, plan, shape,
      Core.Packed{left, Core.TTensor{Replayed.storage(left, left_fill, left_ops), left_fits}},
      Core.Packed{right, Core.TTensor{Replayed.storage(right, right_fill, right_ops), right_fits}})>:
  +total = Shape.logical_numel(Shape.dimensions(shape))
  Records.PointwiseContract{
    run_lift(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, left_inside, right_inside,
      result => {Pointwise.succeeded(result) == True{} : Bool},
      left_after => right_after => output_after => {==})(left_fits, right_fits),
    run_lift(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, left_inside, right_inside,
      result => {Pointwise.output_dimensions(result) == Some{Shape.dimensions(shape)} : Maybe<&2, List<&2, U32>>},
      left_after => right_after => output_after => {==})(left_fits, right_fits),
    index => inside =>
      +inside = inside
      run_lift(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, left_inside, right_inside,
        result => {Pointwise.output_point(result, index) ==
          Some{Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index))} : Maybe<&2, F32>},
        left_after => right_after => output_after =>
          +history = Pointwise.writes(~O, ~f, ~P, ~ia, ~ib, total, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops)
          Equal.trans(Maybe<&2, F32>,
            Core.dynamic_read_value(Core.read_dynamic(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, 0.0, history), output_after}}, index)),
            Some{Replayed.value(shape, 0.0, history, index)},
            Some{Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index))},
            replayed_point(shape, 0.0, history, output_after, index, inside),
            Equal.cong(F32, Maybe<&2, F32>, word => Some{word}, Replayed.value(shape, 0.0, history, index),
              Pointwise.cell(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)),
              writes_value(~O, ~f, ~P, ~ia, ~ib, total, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, index,
                inside, Conversion.le_reflexive(total)))))(left_fits, right_fits),
    other => inside =>
      run_lift(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, left_inside, right_inside,
        result => {Pointwise.left_point(result, other) == Some{Replayed.value(left, left_fill, left_ops, other)} : Maybe<&2, F32>},
        left_after => right_after => output_after =>
          input_point(~P, ~ia, plan, total, left, left_fill, left_ops, left_after, other, inside))(left_fits, right_fits),
    other => inside =>
      run_lift(~O, ~f, ~P, ~ia, ~ib, operation, plan, left, left_fill, left_ops, right, right_fill, right_ops, shape, left_inside, right_inside,
        result => {Pointwise.right_point(result, other) == Some{Replayed.value(right, right_fill, right_ops, other)} : Maybe<&2, F32>},
        left_after => right_after => output_after =>
          input_point(~P, ~ib, plan, total, right, right_fill, right_ops, right_after, other, inside))(left_fits, right_fits)}
