import Base
import ../shape/core.bend as Shape
import ../storage/model.bend as Model
import ../storage/main.bend as Storage
import ../storage/LAWS.bend as StorageLaws
import ../storage/refinement.bend as Refinement
import ../storage/arith.bend as Arith
import ../storage/array.bend as ArrayCore
import ../storage/api.bend as Api
import ../storage/certificates.bend as Certificates
import ../storage/route.bend as Route
import ../shape/index_proofs.bend as Index
import ../shape/nat_proofs.bend as Nats
import ../tensor/core.bend as Core
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/elementwise.bend as Elementwise
import ../tensor/word-conversion.bend as Conversion
import ../tensor/read-contract.bend as Reads
import ../tensor/from-values-contract.bend as FromValues
import ../tensor/matrix-index-proof.bend as MatrixIndex
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/pointwise-proof.bend as PointwiseProof
import ../tensor/contracts/LAWS.bend as Laws
import ../tensor/contracts/binary-storage-proof.bend as Owned
import ../tensor/contracts/binary-build-proof.bend as Build
import ../tensor/contracts/matmul-parallel-proof.bend as Par
import ../tensor/matmul.bend as Matrix
import ./reduce.bend as Reduce
import ./reduce-LAWS.bend as ReduceLaws
import ./reduce-fiber-proof.bend as Fiber
import ./reduce-parallel-proof.bend as ReduceParallel

# The parallel reduce over a replayed contribution. Its output is the
# replayed storage that writes, in index order, each target cell's fiber
# sum: the contribution's replayed words at the offsets the cell's walk
# reads, added in order from +0.0.

def PlanMap(+plan: Shape.BroadcastPlan) -> Data:
  Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>

# The replayed words at a list of offsets, added in order.
def rfold(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, positions: List<&2, Nat>, accumulator: F32) -> F32:
  match positions:
    case Nil{}:
      accumulator
    case position <> rest:
      rfold(shape, fill, ops, rest, F32.add(accumulator, Replayed.value(shape, fill, ops, U32.from_nat(position))))

def map_word(+plan: Shape.BroadcastPlan, map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +position: Nat) -> F32:
  match map:
    case Shape.Unindexed{zero}:
      0.0
    case Shape.Indexed{+rows, exact}:
      rfold(shape, fill, ops, ReduceParallel.visited_of(rows, Shape.extents(Shape.output_shape(plan)), position), 0.0)

def cell_word(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +position: Nat) -> F32:
  map_word(plan, map, shape, fill, ops, position)

def writes(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      Replayed.snoc(writes(plan, map, shape, fill, ops, previous),
        StorageLaws.Write{U32.from_nat(previous), cell_word(plan, map, shape, fill, ops, previous)})

def writes_value(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +count: Nat, +index: U32,
  below: {Nat.is_lt(U32.to_nat(index), count) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, writes(plan, map, shape, fill, ops, count), index) ==
    cell_word(plan, map, shape, fill, ops, U32.to_nat(index)) : F32}:
  match count:
    case 0n:
      Empty.absurd({Replayed.value(output, 0.0, Nil{}, index) ==
        cell_word(plan, map, shape, fill, ops, U32.to_nat(index)) : F32},
        Arith.lt_zero(U32.to_nat(index), below))
    case 1n+previous:
      PointwiseProof.write_split(output, writes(plan, map, shape, fill, ops, previous),
        previous, cell_word(plan, map, shape, fill, ops, previous), index,
        cell_word(plan, map, shape, fill, ops, U32.to_nat(index)),
        below, room,
        at => Equal.cong(Nat, F32, position => cell_word(plan, map, shape, fill, ops, position),
          previous, U32.to_nat(index), Equal.sym(Nat, U32.to_nat(index), previous, at)),
        earlier => writes_value(plan, map, shape, fill, ops, output, previous, index, earlier,
          Conversion.le_predecessor(previous, Shape.logical_numel(Shape.dimensions(output)), room)),
        Shape.bool_case(Nat.is_eq(U32.to_nat(index), previous)))

def lt_not_eq(+a: Nat, +b: Nat, below: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, Arith.clash(below))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      lt_not_eq(p, q, below)

# A write at an index below `index` leaves the word at `index`.
def write_skip(+shape: Shape.Shape, +previous: Nat, +index: U32, +word: F32, +old: F32,
  +room: {Nat.is_lt(previous, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  later: {Nat.is_lt(previous, U32.to_nat(index)) == True{} : Bool}) ->
  {StorageLaws.latest(U32.to_nat(Shape.count(shape)), StorageLaws.Write{U32.from_nat(previous), word}, index, old) == old : F32}:
  %Equal.sym(Nat, U32.to_nat(U32.from_nat(previous)), previous, MatrixIndex.word_exact(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(Nat.is_lt(_, U32.to_nat(Shape.count(shape))), Nat.is_eq(_, U32.to_nat(index))), word, old) == old : F32}
  %Equal.sym(Bool, Nat.is_lt(previous, U32.to_nat(Shape.count(shape))), True{}, PointwiseProof.lt_count(previous, shape, room)) :
    {Bool.pick(F32, Bool.and(_, Nat.is_eq(previous, U32.to_nat(index))), word, old) == old : F32}
  %Equal.sym(Bool, Nat.is_eq(previous, U32.to_nat(index)), False{}, lt_not_eq(previous, U32.to_nat(index), later)) :
    {Bool.pick(F32, _, word, old) == old : F32}
  {==}

# At or above the count, the writes leave the +0.0 fill.
def writes_zero(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +count: Nat, +index: U32,
  +above: {Nat.is_le(count, U32.to_nat(index)) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, writes(plan, map, shape, fill, ops, count), index) == 0.0 : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +history = writes(plan, map, shape, fill, ops, previous)
      +word = cell_word(plan, map, shape, fill, ops, previous)
      +op = {StorageLaws.Write{U32.from_nat(previous), word} : StorageLaws.Op}
      Equal.trans(F32, Replayed.value(output, 0.0, Replayed.snoc(history, op), index),
        StorageLaws.latest(U32.to_nat(Shape.count(output)), op, index, Replayed.value(output, 0.0, history, index)), 0.0,
        Replayed.expected_snoc(history, op, U32.to_nat(Shape.count(output)), index, 0.0),
        Equal.trans(F32, StorageLaws.latest(U32.to_nat(Shape.count(output)), op, index, Replayed.value(output, 0.0, history, index)),
          Replayed.value(output, 0.0, history, index), 0.0,
          write_skip(output, previous, index, word, Replayed.value(output, 0.0, history, index),
            Conversion.le_successor_lt(previous, Shape.logical_numel(Shape.dimensions(output)), room),
            Conversion.le_successor_lt(previous, U32.to_nat(index), above)),
          writes_zero(plan, map, shape, fill, ops, output, previous, index,
            Conversion.le_predecessor(previous, U32.to_nat(index), above),
            Conversion.le_predecessor(previous, Shape.logical_numel(Shape.dimensions(output)), room))))

def lt_zero_false(+x: Nat) -> {Nat.is_lt(x, 0n) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      {==}

def lt_self_succ(+k: Nat) -> {Nat.is_lt(k, 1n+k) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n+p:
      lt_self_succ(p)

def lt_succ_right(+b: Nat, +k: Nat, h: {Nat.is_lt(b, k) == True{} : Bool}) -> {Nat.is_lt(b, 1n+k) == True{} : Bool}:
  match b k:
    case 0n _:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 1n) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      lt_succ_right(p, q, h)

# k < b gives b < 1 + k false.
def above_false(+k: Nat, +b: Nat, h: {Nat.is_lt(k, b) == True{} : Bool}) -> {Nat.is_lt(b, 1n+k) == False{} : Bool}:
  match k b:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 1n) == False{} : Bool}, Arith.clash(h))
    case 0n 1n+q:
      lt_zero_false(q)
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(0n, 1n+(1n+p)) == False{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      above_false(p, q, h)

def lt_asym(+k: Nat, +b: Nat, h: {Nat.is_lt(k, b) == True{} : Bool}) -> {Nat.is_lt(b, k) == False{} : Bool}:
  match k b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(0n, 1n+p) == False{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      lt_asym(p, q, h)

# 1 + b <= k gives b < k.
def below_true(+b: Nat, +k: Nat, h: {Nat.is_le(1n+b, k) == True{} : Bool}) -> {Nat.is_lt(b, k) == True{} : Bool}:
  match b k:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Arith.clash(h))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      below_true(p, q, h)

def lt_add_right(+a: Nat, +c: Nat, +b: Nat, h: {Nat.is_lt(a, c) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(a, b), Nat.add(c, b)) == True{} : Bool}:
  match a c:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(b, b) == True{} : Bool}, Arith.clash(h))
    case 0n 1n+q:
      Arith.lt_succ(b, Nat.add(q, b), Api.le_add_left(q, b))
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+Nat.add(p, b), b) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      lt_add_right(p, q, b, h)

def le_add_cong(+c: Nat, +x: Nat, +y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(c, x), Nat.add(c, y)) == True{} : Bool}:
  match c:
    case 0n:
      h
    case 1n+r:
      le_add_cong(r, x, y, h)

def le_trans(+a: Nat, +b: Nat, +c: Nat, ab: {Nat.is_le(a, b) == True{} : Bool}, bc: {Nat.is_le(b, c) == True{} : Bool}) ->
  {Nat.is_le(a, c) == True{} : Bool}:
  match a b c:
    case 0n _ _:
      Arith.le_zero(c)
    case 1n+p 0n _:
      Empty.absurd({Nat.is_le(1n+p, c) == True{} : Bool}, Arith.clash(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, Arith.clash(bc))
    case 1n+p 1n+q 1n+r:
      le_trans(p, q, r, ab, bc)

def position_step(+capacity: Nat, +offset: Nat, +base: Nat) ->
  {Nat.add(offset, Nat.add(capacity, base)) == Nat.add(Nat.add(capacity, offset), base) : Nat}:
  Equal.trans(Nat, Nat.add(offset, Nat.add(capacity, base)), Nat.add(capacity, Nat.add(offset, base)), Nat.add(Nat.add(capacity, offset), base),
    Nats.add_swap(offset, capacity, base), Index.add_assoc(capacity, offset, base))

def add_zero_right(+number: Nat) -> {Nat.add(number, 0n) == number : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, x => 1n+x, Nat.add(p, 0n), p, add_zero_right(p))

# A word index below the storage length is the offset of its own route.
def output_position(+depth: Nat, +length: Nat, +index: U32, +valid: {Storage.bounds(depth, length) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index), length) == True{} : Bool}) ->
  {Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n) == U32.to_nat(index) : Nat}:
  +rest = Arith.and_right(Nat.is_le(depth, 20n), Bool.and(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth))), valid)
  Equal.trans(Nat, Nat.add(Model.offset(depth, Refinement.route(depth, index)), 0n), Model.offset(depth, Refinement.route(depth, index)), U32.to_nat(index),
    add_zero_right(Model.offset(depth, Refinement.route(depth, index))),
    Equal.sym(Nat, U32.to_nat(index), Model.offset(depth, Refinement.route(depth, index)),
      Route.route_offset(depth, index,
        Arith.and_left(Nat.is_le(depth, 20n), Bool.and(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth))), valid),
        Arith.lt_le_trans(U32.to_nat(index), length, Model.capacity(depth), inside,
          Arith.and_right(Nat.is_lt(0n, length), Nat.is_le(length, Model.capacity(depth)), rest)))))


# The output spec: cell `base` of the tree holds the cell's word below
# `count` and +0.0 above it.

def out_leaf(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +position: Nat, inside: Bool) -> F32:
  match inside:
    case False{}:
      0.0
    case True{}:
      cell_word(plan, map, shape, fill, ops, position)

def out_spec(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Array<F32>:
  match depth:
    case 0n:
      ALeaf{out_leaf(plan, map, shape, fill, ops, base, Nat.is_lt(base, count))}
    case 1n+inner:
      ANode{out_spec(plan, map, shape, fill, ops, count, inner, base), out_spec(plan, map, shape, fill, ops, count, inner, Nat.add(Model.capacity(inner), base))}

def out_balanced(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Model.Balanced(F32, depth, out_spec(plan, map, shape, fill, ops, count, depth, base)):
  match depth:
    case 0n:
      Unit{}
    case 1n+inner:
      (out_balanced(plan, map, shape, fill, ops, count, inner, base), out_balanced(plan, map, shape, fill, ops, count, inner, Nat.add(Model.capacity(inner), base)))

def new_spec(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +depth: Nat, +base: Nat) -> {Array.new(F32, depth, 0.0) == out_spec(plan, map, shape, fill, ops, 0n, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(base, 0n), False{}, lt_zero_false(base)) :
        {ALeaf{0.0} == ALeaf{out_leaf(plan, map, shape, fill, ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(plan, map, shape, fill, ops, 0n, inner, base), new_spec(plan, map, shape, fill, ops, inner, base)) :
        {ANode{_, Array.new(F32, inner, 0.0)} == out_spec(plan, map, shape, fill, ops, 0n, 1n+inner, base) : Array<F32>}
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(plan, map, shape, fill, ops, 0n, inner, Nat.add(Model.capacity(inner), base)),
        new_spec(plan, map, shape, fill, ops, inner, Nat.add(Model.capacity(inner), base))) :
        {ANode{out_spec(plan, map, shape, fill, ops, 0n, inner, base), _} == out_spec(plan, map, shape, fill, ops, 0n, 1n+inner, base) : Array<F32>}
      {==}

# A subtree whose positions all lie above k is the same for k and 1 + k.
def above(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_lt(k, b) == True{} : Bool}) ->
  {out_spec(plan, map, shape, fill, ops, k, depth, b) == out_spec(plan, map, shape, fill, ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(b, k), False{}, lt_asym(k, b, h)) :
        {ALeaf{out_leaf(plan, map, shape, fill, ops, b, _)} == ALeaf{out_leaf(plan, map, shape, fill, ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), False{}, above_false(k, b, h)) :
        {ALeaf{0.0} == ALeaf{out_leaf(plan, map, shape, fill, ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +high = Nat.add(Model.capacity(inner), b)
      %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, b), out_spec(plan, map, shape, fill, ops, 1n+k, inner, b), above(plan, map, shape, fill, ops, k, inner, b, h)) :
        {ANode{_, out_spec(plan, map, shape, fill, ops, k, inner, high)} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, high), out_spec(plan, map, shape, fill, ops, 1n+k, inner, high),
        above(plan, map, shape, fill, ops, k, inner, high, Arith.lt_le_trans(k, b, high, h, Api.le_add_left(Model.capacity(inner), b)))) :
        {ANode{out_spec(plan, map, shape, fill, ops, 1n+k, inner, b), _} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# A subtree whose positions all lie below k is the same for k and 1 + k.
def below(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_le(Nat.add(Model.capacity(depth), b), k) == True{} : Bool}) ->
  {out_spec(plan, map, shape, fill, ops, k, depth, b) == out_spec(plan, map, shape, fill, ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      +lower = below_true(b, k, h)
      %Equal.sym(Bool, Nat.is_lt(b, k), True{}, lower) :
        {ALeaf{out_leaf(plan, map, shape, fill, ops, b, _)} == ALeaf{out_leaf(plan, map, shape, fill, ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), True{}, lt_succ_right(b, k, lower)) :
        {ALeaf{cell_word(plan, map, shape, fill, ops, b)} == ALeaf{out_leaf(plan, map, shape, fill, ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +c = Model.capacity(inner)
      +high = Nat.add(c, b)
      +nested = Equal.trans(Bool, Nat.is_le(Nat.add(c, high), k), Nat.is_le(Nat.add(Nat.add(c, c), b), k), True{},
        Equal.cong(Nat, Bool, x => Nat.is_le(x, k), Nat.add(c, high), Nat.add(Nat.add(c, c), b), Index.add_assoc(c, c, b)), h)
      +low_h = le_trans(high, Nat.add(c, high), k, le_add_cong(c, b, high, Api.le_add_left(c, b)), nested)
      %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, b), out_spec(plan, map, shape, fill, ops, 1n+k, inner, b), below(plan, map, shape, fill, ops, k, inner, b, low_h)) :
        {ANode{_, out_spec(plan, map, shape, fill, ops, k, inner, high)} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, high), out_spec(plan, map, shape, fill, ops, 1n+k, inner, high), below(plan, map, shape, fill, ops, k, inner, high, nested)) :
        {ANode{out_spec(plan, map, shape, fill, ops, 1n+k, inner, b), _} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# Writing cell k at the path whose position is k turns the spec for k into
# the spec for 1 + k.
def write_spec(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +base: Nat, +path: Model.Path(depth),
  +h: {Nat.add(Model.offset(depth, path), base) == k : Nat}) ->
  {ArrayCore.write_path(F32, depth, out_spec(plan, map, shape, fill, ops, k, depth, base), path, cell_word(plan, map, shape, fill, ops, k)) == out_spec(plan, map, shape, fill, ops, 1n+k, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %h : {ALeaf{cell_word(plan, map, shape, fill, ops, _)} == ALeaf{out_leaf(plan, map, shape, fill, ops, base, Nat.is_lt(base, 1n+_))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(base, 1n+base), True{}, lt_self_succ(base)) :
        {ALeaf{cell_word(plan, map, shape, fill, ops, base)} == ALeaf{out_leaf(plan, map, shape, fill, ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      Model.Step{+bit, +rest} = path
      match bit:
        case False{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +inside = Equal.trans(Bool, Nat.is_lt(k, high), Nat.is_lt(Nat.add(Model.offset(inner, rest), base), high), True{},
            Equal.cong(Nat, Bool, x => Nat.is_lt(x, high), k, Nat.add(Model.offset(inner, rest), base),
              Equal.sym(Nat, Nat.add(Model.offset(inner, rest), base), k, h)),
            lt_add_right(Model.offset(inner, rest), c, base, Api.offset_lt(inner, rest)))
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(plan, map, shape, fill, ops, k, inner, base), rest, cell_word(plan, map, shape, fill, ops, k)),
            out_spec(plan, map, shape, fill, ops, 1n+k, inner, base), write_spec(plan, map, shape, fill, ops, k, inner, base, rest, h)) :
            {ANode{_, out_spec(plan, map, shape, fill, ops, k, inner, high)} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, high), out_spec(plan, map, shape, fill, ops, 1n+k, inner, high), above(plan, map, shape, fill, ops, k, inner, high, inside)) :
            {ANode{out_spec(plan, map, shape, fill, ops, 1n+k, inner, base), _} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}
        case True{}:
          +c = Model.capacity(inner)
          +high = Nat.add(c, base)
          +off = Model.offset(inner, rest)
          +lower = Equal.trans(Bool, Nat.is_le(high, k), Nat.is_le(high, Nat.add(c, Nat.add(off, base))), True{},
            Equal.cong(Nat, Bool, x => Nat.is_le(high, x), k, Nat.add(c, Nat.add(off, base)),
              Equal.trans(Nat, k, Nat.add(Nat.add(c, off), base), Nat.add(c, Nat.add(off, base)),
                Equal.sym(Nat, Nat.add(Nat.add(c, off), base), k, h),
                Equal.sym(Nat, Nat.add(c, Nat.add(off, base)), Nat.add(Nat.add(c, off), base), Index.add_assoc(c, off, base)))),
            le_add_cong(c, base, Nat.add(off, base), Api.le_add_left(off, base)))
          +shifted = Equal.trans(Nat, Nat.add(off, high), Nat.add(Nat.add(c, off), base), k, position_step(c, off, base), h)
          %Equal.sym(Array<F32>, out_spec(plan, map, shape, fill, ops, k, inner, base), out_spec(plan, map, shape, fill, ops, 1n+k, inner, base), below(plan, map, shape, fill, ops, k, inner, base, lower)) :
            {ANode{_, ArrayCore.write_path(F32, inner, out_spec(plan, map, shape, fill, ops, k, inner, high), rest, cell_word(plan, map, shape, fill, ops, k))} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(plan, map, shape, fill, ops, k, inner, high), rest, cell_word(plan, map, shape, fill, ops, k)),
            out_spec(plan, map, shape, fill, ops, 1n+k, inner, high), write_spec(plan, map, shape, fill, ops, k, inner, high, rest, shifted)) :
            {ANode{out_spec(plan, map, shape, fill, ops, 1n+k, inner, base), _} == out_spec(plan, map, shape, fill, ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}

# The family history, replayed on the owned storage that `created` makes,
# is the owned storage that holds the output spec.
def chain(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +d: Nat, +n: Nat, -f: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +limit: U32, +n_limit: {Nat.is_le(n, U32.to_nat(limit)) == True{} : Bool},
  +k: Nat, +room: {Nat.is_le(k, n) == True{} : Bool}) ->
  {StorageLaws.replay(writes(plan, map, shape, fill, ops, k), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, k, d, 0n), out_balanced(plan, map, shape, fill, ops, k, d, 0n), f} : Storage.Storage}:
  match k:
    case 0n:
      Owned.owned_eq(d, n, Array.new(F32, d, 0.0), out_spec(plan, map, shape, fill, ops, 0n, d, 0n), new_spec(plan, map, shape, fill, ops, d, 0n), ArrayCore.new_balanced(F32, d, 0.0), f)(
        out_balanced(plan, map, shape, fill, ops, 0n, d, 0n))
    case 1n+j:
      +index = U32.from_nat(j)
      +value = cell_word(plan, map, shape, fill, ops, j)
      +below_n = Conversion.le_successor_lt(j, n, room)
      +exact = Conversion.natural_word_exact(j, limit, le_trans(j, n, U32.to_nat(limit), Conversion.lt_le(j, n, below_n), n_limit))
      +inside = Equal.trans(Bool, Nat.is_lt(U32.to_nat(index), n), Nat.is_lt(j, n), True{},
        Equal.cong(Nat, Bool, x => Nat.is_lt(x, n), U32.to_nat(index), j, exact), below_n)
      +offset = Equal.trans(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), U32.to_nat(index), j,
        output_position(d, n, index, fits, inside), exact)
      +written = Equal.trans(Array<F32>, Array.set(F32, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value), ArrayCore.write_path(F32, d, out_spec(plan, map, shape, fill, ops, j, d, 0n), Refinement.route(d, index), value),
        out_spec(plan, map, shape, fill, ops, 1n+j, d, 0n), Refinement.law_set_refines(F32, d, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value, out_balanced(plan, map, shape, fill, ops, j, d, 0n)),
        write_spec(plan, map, shape, fill, ops, j, d, 0n, Refinement.route(d, index), offset))
      Equal.trans(Storage.Storage, StorageLaws.replay(writes(plan, map, shape, fill, ops, 1n+j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        StorageLaws.step(StorageLaws.replay(writes(plan, map, shape, fill, ops, j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), StorageLaws.Write{index, value}),
        Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, 1n+j, d, 0n), out_balanced(plan, map, shape, fill, ops, 1n+j, d, 0n), f},
        FromValues.replay_snoc(writes(plan, map, shape, fill, ops, j), StorageLaws.Write{index, value}, Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        %Equal.sym(Storage.Storage, StorageLaws.replay(writes(plan, map, shape, fill, ops, j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, j, d, 0n), out_balanced(plan, map, shape, fill, ops, j, d, 0n), f},
          chain(plan, map, shape, fill, ops, d, n, f, fits, limit, n_limit, j, Conversion.le_predecessor(j, n, room))) :
          {StorageLaws.step(_, StorageLaws.Write{index, value}) == Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, 1n+j, d, 0n), out_balanced(plan, map, shape, fill, ops, 1n+j, d, 0n), f} : Storage.Storage}
        %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
          Storage.set_checked(d, n, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(plan, map, shape, fill, ops, j, d, 0n), f),
          (Storage.Owned{d, n, Array.set(F32, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value),
            Certificates.set_result(F32, d, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value, Array.set(F32, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value), {==}, out_balanced(plan, map, shape, fill, ops, j, d, 0n)), f}, Done{Unit{}}),
          Api.set_checked_hit(d, n, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(plan, map, shape, fill, ops, j, d, 0n), f,
            Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), n), True{}, inside), fits, inside)) :
          {StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _) == Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, 1n+j, d, 0n), out_balanced(plan, map, shape, fill, ops, 1n+j, d, 0n), f} : Storage.Storage}
        Owned.owned_eq(d, n, Array.set(F32, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value), out_spec(plan, map, shape, fill, ops, 1n+j, d, 0n), written,
          Certificates.set_result(F32, d, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value, Array.set(F32, out_spec(plan, map, shape, fill, ops, j, d, 0n), index, value), {==}, out_balanced(plan, map, shape, fill, ops, j, d, 0n)), f)(
          out_balanced(plan, map, shape, fill, ops, 1n+j, d, 0n)))



# The walk over the copy of a replayed contribution adds its replayed words.

def below_front(+xs: List<&2, Nat>, +ys: List<&2, Nat>, +bound: Nat,
  +h: {ReduceParallel.all_below(List.append(&2, Nat, xs, ys), bound) == True{} : Bool}) ->
  {ReduceParallel.all_below(xs, bound) == True{} : Bool}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      +here = Arith.and_left(Nat.is_lt(x, bound), ReduceParallel.all_below(List.append(&2, Nat, rest, ys), bound), h)
      +later = Arith.and_right(Nat.is_lt(x, bound), ReduceParallel.all_below(List.append(&2, Nat, rest, ys), bound), h)
      %Equal.sym(Bool, Nat.is_lt(x, bound), True{}, here) : {Bool.and(_, ReduceParallel.all_below(rest, bound)) == True{} : Bool}
      below_front(rest, ys, bound, later)

def below_back(+xs: List<&2, Nat>, +ys: List<&2, Nat>, +bound: Nat,
  +h: {ReduceParallel.all_below(List.append(&2, Nat, xs, ys), bound) == True{} : Bool}) ->
  {ReduceParallel.all_below(ys, bound) == True{} : Bool}:
  match xs:
    case Nil{}:
      h
    case +x <> +rest:
      below_back(rest, ys, bound, Arith.and_right(Nat.is_lt(x, bound), ReduceParallel.all_below(List.append(&2, Nat, rest, ys), bound), h))

def rfold_append(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +xs: List<&2, Nat>, +ys: List<&2, Nat>, +accumulator: F32) ->
  {rfold(shape, fill, ops, List.append(&2, Nat, xs, ys), accumulator) == rfold(shape, fill, ops, ys, rfold(shape, fill, ops, xs, accumulator)) : F32}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      rfold_append(shape, fill, ops, rest, ys, F32.add(accumulator, Replayed.value(shape, fill, ops, U32.from_nat(x))))

def leaf_eq(+base: Nat, +accumulator: F32, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +source: Par.Source<shape, fill, ops, h, d, f>,
  +below: {Nat.is_lt(base, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool},
  pair: Array<F32> & F32, +exact: {pair == Array.get(F32, Api.arr(h, d, f), U32.from_nat(base)) : Array<F32> & F32}) ->
  {Reduce.leaf_add(accumulator, pair) == (rfold(shape, fill, ops, [base], accumulator), Api.arr(h, d, f)) : F32 & Array<F32>}:
  (kept, word) = pair
  Owned.pair_eq(F32, Array<F32>, F32.add(accumulator, word), F32.add(accumulator, Replayed.value(shape, fill, ops, U32.from_nat(base))),
    kept, Api.arr(h, d, f),
    Equal.cong(F32, F32, w => F32.add(accumulator, w), word, Replayed.value(shape, fill, ops, U32.from_nat(base)),
      Par.source_read(shape, fill, ops, h, d, f, source, U32.from_nat(base), Par.index_inside(shape, base, below), (kept, word), exact)),
    Build.got_kept(d, h, f, U32.from_nat(base), (kept, word), exact))

def next_eq(-rec: F32 -> Array<F32> -> F32 & Array<F32>, -low: F32 & Array<F32>, -value: F32, -array: Array<F32>,
  +first: {low == (value, array) : F32 & Array<F32>}) -> {Reduce.walk_next(rec, low) == rec(value, array) : F32 & Array<F32>}:
  %Equal.sym(F32 & Array<F32>, low, (value, array), first) : {Reduce.walk_next(rec, _) == rec(value, array) : F32 & Array<F32>}
  {==}

def fiber_eq(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +fuel: Nat, +target: Nat, +base: Nat, +digit: Nat, +accumulator: F32,
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32,
  +source: Par.Source<shape, fill, ops, h, d, f>,
  +below: {ReduceParallel.all_below(Fiber.visited(rows, extents, fuel, target, base, digit), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Reduce.fiber(rows, extents, fuel, target, base, digit, accumulator, Api.arr(h, d, f)) ==
    (rfold(shape, fill, ops, Fiber.visited(rows, extents, fuel, target, base, digit), accumulator), Api.arr(h, d, f)) : F32 & Array<F32>}:
  match rows extents fuel:
    case Nil{} _ _:
      leaf_eq(base, accumulator, shape, fill, ops, h, d, f, source,
        Arith.and_left(Nat.is_lt(base, Shape.logical_numel(Shape.dimensions(shape))), True{}, below),
        Array.get(F32, Api.arr(h, d, f), U32.from_nat(base)), {==})
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n:
      {==}
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +extents : List<&2, Nat> = extent <> others
      +inner = Nat.mod(target, left)
      +offset = Nat.add(base, Nat.mul(right, digit))
      +low = Fiber.visited(rest, others, Reduce.first_span(rest, others), inner, offset, Reduce.first_digit(rest, inner))
      +high = Fiber.visited(rows, extents, remaining, target, base, 1n+digit)
      +bound = Shape.logical_numel(Shape.dimensions(shape))
      +value = rfold(shape, fill, ops, low, accumulator)
      Equal.trans(F32 & Array<F32>,
        Reduce.walk_next(v => k => Reduce.fiber(rows, extents, remaining, target, base, 1n+digit, v, k),
          Reduce.fiber(rest, others, Reduce.first_span(rest, others), inner, offset, Reduce.first_digit(rest, inner), accumulator, Api.arr(h, d, f))),
        Reduce.fiber(rows, extents, remaining, target, base, 1n+digit, value, Api.arr(h, d, f)),
        (rfold(shape, fill, ops, List.append(&2, Nat, low, high), accumulator), Api.arr(h, d, f)),
        next_eq(v => k => Reduce.fiber(rows, extents, remaining, target, base, 1n+digit, v, k),
          Reduce.fiber(rest, others, Reduce.first_span(rest, others), inner, offset, Reduce.first_digit(rest, inner), accumulator, Api.arr(h, d, f)),
          value, Api.arr(h, d, f),
          fiber_eq(rest, others, Reduce.first_span(rest, others), inner, offset, Reduce.first_digit(rest, inner), accumulator,
            shape, fill, ops, h, d, f, source, below_front(low, high, bound, below))),
        %Equal.sym(F32, rfold(shape, fill, ops, List.append(&2, Nat, low, high), accumulator), rfold(shape, fill, ops, high, value),
          rfold_append(shape, fill, ops, low, high, accumulator)) :
          {Reduce.fiber(rows, extents, remaining, target, base, 1n+digit, value, Api.arr(h, d, f)) == (_, Api.arr(h, d, f)) : F32 & Array<F32>}
        fiber_eq(rows, extents, remaining, target, base, 1n+digit, value, shape, fill, ops, h, d, f, source, below_back(low, high, bound, below)))
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _:
      {==}

# What the cells of the build need from the plan: the target is the plan's
# left shape, with `count` cells, and the contribution is its right shape.
type Facts<-plan: Shape.BroadcastPlan, -shape: Shape.Shape, -count: Nat> is Data:
  Facts{count_left: {count == Shape.numel(Shape.extents(Shape.left_shape(plan))) : Nat},
    right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}}

# Every offset the walk of a cell below `count` reads lies inside the
# contribution.
def walk_below(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +count: Nat, +facts: Facts<plan, shape, count>, +base: Nat, +inside: {Nat.is_lt(base, count) == True{} : Bool}) ->
  {ReduceParallel.all_below(ReduceParallel.visited_of(rows, Shape.extents(Shape.output_shape(plan)), base), Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}:
  Facts{+count_left, +right} = facts
  +output = Shape.extents(Shape.output_shape(plan))
  +below_left : {Nat.is_lt(base, Shape.numel(Shape.extents(Shape.left_shape(plan)))) == True{} : Bool} =
    %count_left : {Nat.is_lt(base, _) == True{} : Bool}
    inside
  %right : {ReduceParallel.all_below(ReduceParallel.visited_of(rows, output, base), Shape.logical_numel(_)) == True{} : Bool}
  %ReduceParallel.plan_walk(plan, rows, rows_exact, base, below_left) :
    {ReduceParallel.all_below(_, Shape.logical_numel(Shape.dimensions(Shape.right_shape(plan)))) == True{} : Bool}
  ReduceParallel.masked_below(plan, rows, rows_exact, base, Shape.numel(output), 0n, Conversion.le_reflexive(Shape.numel(output)))

def TreeEq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat, +depth: Nat, +base: Nat,
  -result: Array<F32> & Array<F32>) -> Type:
  {Pair.fst(Array<F32>, Array<F32>, result) == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, depth, base) : Array<F32>} &
    {Pair.snd(Array<F32>, Array<F32>, result) == Api.arr(h, d, f) : Array<F32>}

def cell_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat,
  +source: Par.Source<shape, fill, ops, h, d, f>, +facts: Facts<plan, shape, count>, +base: Nat, inside: Bool,
  +decided: {inside == Nat.is_lt(base, count) : Bool}) ->
  TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 0n, base,
    Reduce.cell(rows, Shape.extents(Shape.output_shape(plan)), base, Api.arr(h, d, f), inside)):
  match inside:
    case False{}:
      (%decided : {ALeaf{0.0} == ALeaf{out_leaf(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, base, _)} : Array<F32>}
       {==}, {==})
    case True{}:
      +output = Shape.extents(Shape.output_shape(plan))
      +word = rfold(shape, fill, ops, ReduceParallel.visited_of(rows, output, base), 0.0)
      %Equal.sym(F32 & Array<F32>, Reduce.fiber_of(rows, output, base, Api.arr(h, d, f)), (word, Api.arr(h, d, f)),
        fiber_eq(rows, output, Reduce.first_span(rows, output), base, 0n, Reduce.first_digit(rows, base), 0.0, shape, fill, ops, h, d, f, source,
          walk_below(plan, rows, rows_exact, shape, count, facts, base, Equal.sym(Bool, True{}, Nat.is_lt(base, count), decided)))) :
        TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 0n, base, Reduce.cell_leaf(_))
      (%decided : {ALeaf{word} == ALeaf{out_leaf(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, base, _)} : Array<F32>}
       {==}, {==})

def tree_join_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat, +lower: Nat, +base: Nat,
  -a: Array<F32>, high: Array<F32> & Array<F32>,
  ea: {a == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base) : Array<F32>},
  facts: TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, Nat.add(Model.capacity(lower), base), high)) ->
  TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 1n+lower, base, Reduce.tree_join(a, high)):
  (b, array) = high
  (eb, kept) = facts
  (%Equal.sym(Array<F32>, a, out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), ea) : {ANode{_, b} == ANode{out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, Nat.add(Model.capacity(lower), base))} : Array<F32>}
   %Equal.sym(Array<F32>, b, out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, Nat.add(Model.capacity(lower), base)), eb) : {ANode{out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), _} == ANode{out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, Nat.add(Model.capacity(lower), base))} : Array<F32>}
   {==}, kept)

def tree_node_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat, +lower: Nat, +base: Nat,
  low: Array<F32> & Array<F32>,
  facts: TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, base, low),
  next: TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, Nat.add(Model.capacity(lower), base),
    Reduce.tree(lower, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), Api.arr(h, d, f)))) ->
  TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 1n+lower, base,
    Reduce.tree_node(x => Reduce.tree(lower, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), x), low)):
  (a, array) = low
  (ea, ek) = facts
  %Equal.sym(Array<F32>, array, Api.arr(h, d, f), ek) :
    TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 1n+lower, base,
      Reduce.tree_join(a, Reduce.tree(lower, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), _)))
  tree_join_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, base, a,
    Reduce.tree(lower, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), Api.arr(h, d, f)), ea, next)

def tree_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat,
  +source: Par.Source<shape, fill, ops, h, d, f>, +facts: Facts<plan, shape, count>, +depth: Nat, +base: Nat) ->
  TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, depth, base,
    Reduce.tree(depth, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f))):
  match depth:
    case 0n:
      cell_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, base, Nat.is_lt(base, count), {==})
    case 1n+lower:
      tree_node_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, base,
        Reduce.tree(lower, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f)),
        tree_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, lower, base),
        tree_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, lower, Nat.add(Model.capacity(lower), base)))

def tree_first(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat, +depth: Nat, +base: Nat,
  -result: Array<F32> & Array<F32>, facts: TreeEq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, depth, base, result)) ->
  {Pair.fst(Array<F32>, Array<F32>, result) == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, depth, base) : Array<F32>}:
  (first, kept) = facts
  first

def fork_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat,
  +lower: Nat, +rest: Nat, +base: Nat,
  low: {Reduce.build(lower, rest, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f)) ==
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base) : Array<F32>},
  high: {Reduce.build(lower, rest, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), Api.arr(h, d, f)) ==
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, Nat.add(Model.capacity(lower), base)) : Array<F32>},
  pair: Array<F32> & Array<F32>, +exact: {pair == Array.clone(F32, Api.arr(h, d, f)) : Array<F32> & Array<F32>}) ->
  {Reduce.fork(pair, x => Reduce.build(lower, rest, rows, Shape.extents(Shape.output_shape(plan)), count, base, x),
    x => Reduce.build(lower, rest, rows, Shape.extents(Shape.output_shape(plan)), count, Nat.add(Model.capacity(lower), base), x)) ==
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, 1n+lower, base) : Array<F32>}:
  (first, second) = pair
  +output = Shape.extents(Shape.output_shape(plan))
  +upper = Nat.add(Model.capacity(lower), base)
  %Equal.sym(Array<F32>, first, Api.arr(h, d, f), Build.clone_first_eq(d, h, f, (first, second), exact)) :
    {ANode{Reduce.build(lower, rest, rows, output, count, base, _), Reduce.build(lower, rest, rows, output, count, upper, second)} == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, second, Api.arr(h, d, f), Build.clone_second_eq(d, h, f, (first, second), exact)) :
    {ANode{Reduce.build(lower, rest, rows, output, count, base, Api.arr(h, d, f)), Reduce.build(lower, rest, rows, output, count, upper, _)} == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, Reduce.build(lower, rest, rows, output, count, base, Api.arr(h, d, f)),
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), low) :
    {ANode{_, Reduce.build(lower, rest, rows, output, count, upper, Api.arr(h, d, f))} == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, Reduce.build(lower, rest, rows, output, count, upper, Api.arr(h, d, f)),
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, upper), high) :
    {ANode{out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, lower, base), _} == out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, 1n+lower, base) : Array<F32>}
  {==}

# Whatever the split, the parallel build over the copy returns the spec.
def build_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32, +count: Nat,
  +source: Par.Source<shape, fill, ops, h, d, f>, +facts: Facts<plan, shape, count>, +depth: Nat, +splits: Nat, +base: Nat) ->
  {Reduce.build(depth, splits, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f)) ==
    out_spec(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, count, depth, base) : Array<F32>}:
  match depth splits:
    case 1n+lower 1n+rest:
      fork_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, lower, rest, base,
        build_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, lower, rest, base),
        build_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, lower, rest, Nat.add(Model.capacity(lower), base)),
        Array.clone(F32, Api.arr(h, d, f)), {==})
    case 0n _:
      tree_first(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 0n, base,
        Reduce.tree(0n, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f)),
        tree_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, 0n, base))
    case 1n+lower 0n:
      tree_first(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, 1n+lower, base,
        Reduce.tree(1n+lower, rows, Shape.extents(Shape.output_shape(plan)), count, base, Api.arr(h, d, f)),
        tree_eq(plan, rows, rows_exact, shape, fill, ops, h, d, f, count, source, facts, 1n+lower, base))

# As in reduce-fold-PROOF.bend: below the count, the listed words are the
# replayed words.
def length_snoc(+values: List<&2, F32>, +word: F32) ->
  {List.length(&2, F32, List.append(&2, F32, values, [word])) == 1n+List.length(&2, F32, values) : Nat}:
  match values:
    case Nil{}:
      {==}
    case +value <> +rest:
      Equal.cong(Nat, Nat, n => 1n+n, List.length(&2, F32, List.append(&2, F32, rest, [word])), 1n+List.length(&2, F32, rest),
        length_snoc(rest, word))

def words_length(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {List.length(&2, F32, Laws.words(shape, fill, ops, count)) == count : Nat}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +before = Laws.words(shape, fill, ops, previous)
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
    {ReduceLaws.word_at(Laws.words(shape, fill, ops, previous), position) ==
      Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}) ->
  {ReduceLaws.word_at(Laws.words(shape, fill, ops, 1n+previous), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}:
  match decision:
    case Shape.BFalse{equal}:
      +before = Laws.words(shape, fill, ops, previous)
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
      +before = Laws.words(shape, fill, ops, previous)
      +last = Replayed.value(shape, fill, ops, U32.from_nat(previous))
      %Equal.sym(Nat, position, previous, Arith.eq_true(position, previous, Equal.sym(Bool, True{}, Nat.is_eq(position, previous), equal))) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) == Some{Replayed.value(shape, fill, ops, U32.from_nat(_))} : Maybe<&2, F32>}
      %words_length(shape, fill, ops, previous) :
        {ReduceLaws.word_at(List.append(&2, F32, before, [last]), _) == Some{last} : Maybe<&2, F32>}
      word_at_last(before, last)

def words_at(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat, +position: Nat,
  below: {Nat.is_lt(position, count) == True{} : Bool}) ->
  {ReduceLaws.word_at(Laws.words(shape, fill, ops, count), position) ==
    Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>}:
  match count:
    case 0n:
      Empty.absurd({ReduceLaws.word_at(Nil{}, position) == Some{Replayed.value(shape, fill, ops, U32.from_nat(position))} : Maybe<&2, F32>},
        Arith.lt_zero(position, below))
    case 1n+previous:
      words_step(shape, fill, ops, previous, position, below, Shape.bool_case(Nat.is_eq(position, previous)),
        inside => words_at(shape, fill, ops, previous, position, inside))


def parallel_splits(+plan: Shape.BroadcastPlan, +depth: Nat, +length: Nat) -> Nat:
  Reduce.splits(plan, depth, length)

# The writes of every cell, replayed on the owned storage `created` makes for
# the target, give the owned storage the parallel build fills.
def output_eq(+plan: Shape.BroadcastPlan, +rows: List<&2, Shape.Row>,
  +rows_exact: {rows == Shape.index_table(Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))) : List<&2, Shape.Row>},
  +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +dc: Nat, +fc: F32,
  +source: Par.Source<shape, fill, ops, h, dc, fc>, +target: Shape.Shape, +facts: Facts<plan, shape, U32.to_nat(Shape.count(target))>,
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(target))) == True{} : Bool},
  -f: {Storage.bounds(d, U32.to_nat(Shape.count(target))) == True{} : Bool}) ->
  {StorageLaws.replay(writes(plan, Shape.Indexed{rows, rows_exact}, shape, fill, ops, U32.to_nat(Shape.count(target))),
    Storage.Owned{d, U32.to_nat(Shape.count(target)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) ==
    Storage.Owned{d, U32.to_nat(Shape.count(target)),
      Reduce.build(d, parallel_splits(plan, d, U32.to_nat(Shape.count(target))), rows, Shape.extents(Shape.output_shape(plan)),
        U32.to_nat(Shape.count(target)), 0n, Api.arr(h, dc, fc)),
      Reduce.build_balanced(d, parallel_splits(plan, d, U32.to_nat(Shape.count(target))), rows, Shape.extents(Shape.output_shape(plan)),
        U32.to_nat(Shape.count(target)), 0n, Api.arr(h, dc, fc)), f} : Storage.Storage}:
  +n = U32.to_nat(Shape.count(target))
  +map : PlanMap(plan) = Shape.Indexed{rows, rows_exact}
  +splits = parallel_splits(plan, d, n)
  +extents = Shape.extents(Shape.output_shape(plan))
  Equal.trans(Storage.Storage,
    StorageLaws.replay(writes(plan, map, shape, fill, ops, n), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
    Storage.Owned{d, n, out_spec(plan, map, shape, fill, ops, n, d, 0n), out_balanced(plan, map, shape, fill, ops, n, d, 0n), f},
    Storage.Owned{d, n, Reduce.build(d, splits, rows, extents, n, 0n, Api.arr(h, dc, fc)), Reduce.build_balanced(d, splits, rows, extents, n, 0n, Api.arr(h, dc, fc)), f},
    chain(plan, map, shape, fill, ops, d, n, f, fits, Shape.count(target), Conversion.le_reflexive(n), n, Conversion.le_reflexive(n)),
    Owned.owned_eq(d, n, out_spec(plan, map, shape, fill, ops, n, d, 0n), Reduce.build(d, splits, rows, extents, n, 0n, Api.arr(h, dc, fc)),
      Equal.sym(Array<F32>, Reduce.build(d, splits, rows, extents, n, 0n, Api.arr(h, dc, fc)), out_spec(plan, map, shape, fill, ops, n, d, 0n),
        build_eq(plan, rows, rows_exact, shape, fill, ops, h, dc, fc, n, source, facts, d, splits, 0n)),
      out_balanced(plan, map, shape, fill, ops, n, d, 0n), f)(Reduce.build_balanced(d, splits, rows, extents, n, 0n, Api.arr(h, dc, fc))))

# The parallel reduce over a replayed contribution, in transport form: P
# holds for the result if it holds for the contribution replayed with any
# history that keeps its words, next to the target replayed with any history
# whose cells hold the cell words.
def Final(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +map: PlanMap(plan),
  -s: Storage.Storage, -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @values: (@+other: U32 -> {Replayed.value(shape, fill, kept, other) == Replayed.value(shape, fill, ops, other) : F32}) ->
  @words: {Laws.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) ==
    Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) : List<&2, F32>} ->
  @-kept_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, kept)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @+out: List<&2, StorageLaws.Op> ->
  @cells: (@+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
    {Replayed.value(target, 0.0, out, index) == cell_word(plan, map, shape, fill, ops, U32.to_nat(index)) : F32}) ->
  @-out_fits: {Lifecycle.logical_length(StorageLaws.replay(out, s)) == Shape.logical_numel(Shape.dimensions(target)) : Nat} ->
    P((Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}},
      Done{Core.Packed{target, Core.TTensor{StorageLaws.replay(out, s), out_fits}}}))

def room_of(+target: Shape.Shape) -> {Nat.is_le(U32.to_nat(Shape.count(target)), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(target)), Shape.logical_numel(Shape.dimensions(target)), Reads.count_exact(target)) : {Nat.is_le(_, Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}
  Conversion.le_reflexive(Shape.logical_numel(Shape.dimensions(target)))

def inside_count(+target: Shape.Shape, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(target))) == True{} : Bool}:
  %Equal.sym(Nat, U32.to_nat(Shape.count(target)), Shape.logical_numel(Shape.dimensions(target)), Reads.count_exact(target)) :
    {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
  inside

def indexed_lift(+plan: Shape.BroadcastPlan, +target: Shape.Shape, +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  +facts: Facts<plan, shape, U32.to_nat(Shape.count(target))>,
  +d: Nat, +fits_d: {Storage.bounds(d, U32.to_nat(Shape.count(target))) == True{} : Bool},
  -f: {Storage.bounds(d, U32.to_nat(Shape.count(target))) == True{} : Bool},
  -fo: {Lifecycle.logical_length(Storage.Owned{d, U32.to_nat(Shape.count(target)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) ==
    Shape.logical_numel(Shape.dimensions(target)) : Nat},
  +map: PlanMap(plan), -P: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic> -> Type,
  final: Final(plan, target, shape, fill, ops, map,
    Storage.Owned{d, U32.to_nat(Shape.count(target)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}, P)) ->
  P(Reduce.parallel_indexed(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, target, plan,
    Core.TTensor{Storage.Owned{d, U32.to_nat(Shape.count(target)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}, fo},
    d, U32.to_nat(Shape.count(target)), f, fo, map)):
  match map:
    case Shape.Unindexed{zero}:
      final(ops)(other => {==})({==})(fits)(Nil{})(index => inside => {==})(fo)
    case Shape.Indexed{+rows, +rows_exact}:
      +n = U32.to_nat(Shape.count(target))
      +map : PlanMap(plan) = Shape.Indexed{rows, rows_exact}
      Par.input_take(shape, fill, ops,
        taken => P(Reduce.parallel_rows(plan, target, d, n, f, fo, taken, rows)),
        kept => values => words => h => dc => fc => source => kept_fits =>
          Par.storage_shift(target,
            StorageLaws.replay(writes(plan, map, shape, fill, ops, n), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
            Storage.Owned{d, n,
              Reduce.build(d, parallel_splits(plan, d, n), rows, Shape.extents(Shape.output_shape(plan)), n, 0n, Api.arr(h, dc, fc)),
              Reduce.build_balanced(d, parallel_splits(plan, d, n), rows, Shape.extents(Shape.output_shape(plan)), n, 0n, Api.arr(h, dc, fc)), f},
            output_eq(plan, rows, rows_exact, shape, fill, ops, h, dc, fc, source, target, facts, d, fits_d, f),
            st => k => P((Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, kept), kept_fits}}, Done{Core.Packed{target, Core.TTensor{st, k}}})),
            final(kept)(values)(words)(kept_fits)(writes(plan, map, shape, fill, ops, n))(
              index => inside => writes_value(plan, map, shape, fill, ops, target, n, index, inside_count(target, index, inside), room_of(target))))(fo))(fits)

# The cell word adds the replayed words the masked fold adds.

def rfold_gather(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +positions: List<&2, Nat>, +accumulator: F32,
  +below: {ReduceParallel.all_below(positions, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Some{rfold(shape, fill, ops, positions, accumulator)} ==
    ReduceParallel.gather(Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), positions, Some{accumulator}) : Maybe<&2, F32>}:
  match positions:
    case Nil{}:
      {==}
    case +p <> +ps:
      +count = Shape.logical_numel(Shape.dimensions(shape))
      +here = Arith.and_left(Nat.is_lt(p, count), ReduceParallel.all_below(ps, count), below)
      +later = Arith.and_right(Nat.is_lt(p, count), ReduceParallel.all_below(ps, count), below)
      +word = Replayed.value(shape, fill, ops, U32.from_nat(p))
      %Equal.sym(Maybe<&2, F32>, ReduceLaws.word_at(Laws.words(shape, fill, ops, count), p), Some{word}, words_at(shape, fill, ops, count, p, here)) :
        {Some{rfold(shape, fill, ops, ps, F32.add(accumulator, word))} ==
          ReduceParallel.gather(Laws.words(shape, fill, ops, count), ps, ReduceLaws.add_word(Some{accumulator}, _)) : Maybe<&2, F32>}
      rfold_gather(shape, fill, ops, ps, F32.add(accumulator, word), later)

def word_fiber(+plan: Shape.BroadcastPlan, +map: PlanMap(plan), +shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), Shape.numel(Shape.extents(Shape.left_shape(plan)))) == True{} : Bool},
  +right: {Shape.dimensions(Shape.right_shape(plan)) == Shape.dimensions(shape) : List<&2, U32>}) ->
  {Some{cell_word(plan, map, shape, fill, ops, U32.to_nat(index))} ==
    ReduceLaws.fiber(plan, Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index,
      Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : Maybe<&2, F32>}:
  match map:
    case Shape.Unindexed{zero}:
      %zero : {Some{0.0} == ReduceLaws.fiber(plan, Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))), index, _) : Maybe<&2, F32>}
      {==}
    case Shape.Indexed{+rows, +rows_exact}:
      +t = U32.to_nat(index)
      +output = Shape.extents(Shape.output_shape(plan))
      +count = Shape.numel(output)
      +values = Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape)))
      +walk = ReduceParallel.plan_walk(plan, rows, rows_exact, t, inside)
      %Equal.sym(Maybe<&2, F32>, ReduceLaws.fiber(plan, values, index, count), ReduceParallel.gather(values, Fiber.masked(count, rows, t, 0n, 0n), Some{0.0}),
        ReduceParallel.fiber_masked(plan, rows, rows_exact, values, index, count, Conversion.le_reflexive(count))) :
        {Some{rfold(shape, fill, ops, ReduceParallel.visited_of(rows, output, t), 0.0)} == _ : Maybe<&2, F32>}
      %Equal.sym(List<&2, Nat>, Fiber.masked(count, rows, t, 0n, 0n), ReduceParallel.visited_of(rows, output, t), walk) :
        {Some{rfold(shape, fill, ops, ReduceParallel.visited_of(rows, output, t), 0.0)} == ReduceParallel.gather(values, _, Some{0.0}) : Maybe<&2, F32>}
      rfold_gather(shape, fill, ops, ReduceParallel.visited_of(rows, output, t), 0.0,
        walk_below(plan, rows, rows_exact, shape, Shape.numel(Shape.extents(Shape.left_shape(plan))), Facts{{==}, right}, t, inside))
