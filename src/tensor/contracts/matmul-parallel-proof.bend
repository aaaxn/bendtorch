import Base
import ../../shape/core.bend as Shape
import ../../storage/model.bend as Model
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/refinement.bend as Refinement
import ../../storage/arith.bend as Arith
import ../core.bend as Core
import ../lifecycle.bend as Lifecycle
import ../matmul.bend as Matrix
import ../elementwise.bend as Elementwise
import ../dot-fold.bend as DotFold
import ../word-conversion.bend as Conversion
import ../read-contract.bend as Reads
import ../from-values-contract.bend as FromValues
import ../matrix-index-proof.bend as MatrixIndex
import ./replayed.bend as Replayed
import ./pointwise-proof.bend as PointwiseProof
import ./dot-proof.bend as DotProof
import ./LAWS.bend as Laws
import ./tree-storage-proof.bend as Tree
import ./binary-storage-proof.bend as Owned
import ./binary-build-proof.bend as Build
import ../../shape/index_proofs.bend as Index
import ../../shape/nat_proofs.bend as Nats
import ../../storage/array.bend as ArrayCore
import ../../storage/api.bend as Api
import ../../storage/certificates.bend as Certificates
import ../../storage/route.bend as Route

# The parallel matmul over replayed inputs. Its output is the replayed
# storage that writes, in index order, the word `matmul_ordered` names at each
# cell.

def cell_word(+inner: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +position: Nat) -> F32:
  DotFold.dot_fold(Laws.factors(Matrix.row_of(columns, position), Matrix.column_of(columns, position), inner, columns,
    left, left_fill, left_ops, right, right_fill, right_ops, inner), 0.0)

def writes(+inner: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat) -> List<&2, StorageLaws.Op>:
  match count:
    case 0n:
      Nil{}
    case 1n+previous:
      Replayed.snoc(writes(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        StorageLaws.Write{U32.from_nat(previous), cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous)})

def writes_value(+inner: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +count: Nat, +index: U32,
  below: {Nat.is_lt(U32.to_nat(index), count) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, writes(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), index) ==
    cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32}:
  match count:
    case 0n:
      Empty.absurd({Replayed.value(output, 0.0, Nil{}, index) ==
        cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)) : F32},
        Arith.lt_zero(U32.to_nat(index), below))
    case 1n+previous:
      PointwiseProof.write_split(output, writes(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous),
        previous, cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous), index,
        cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(index)),
        below, room,
        at => Equal.cong(Nat, F32, position => cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, position),
          previous, U32.to_nat(index), Equal.sym(Nat, U32.to_nat(index), previous, at)),
        earlier => writes_value(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, output, previous, index, earlier,
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
def writes_zero(+inner: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +output: Shape.Shape, +count: Nat, +index: U32,
  +above: {Nat.is_le(count, U32.to_nat(index)) == True{} : Bool},
  +room: {Nat.is_le(count, Shape.logical_numel(Shape.dimensions(output))) == True{} : Bool}) ->
  {Replayed.value(output, 0.0, writes(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, count), index) == 0.0 : F32}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +history = writes(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      +word = cell_word(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, previous)
      +op = {StorageLaws.Write{U32.from_nat(previous), word} : StorageLaws.Op}
      Equal.trans(F32, Replayed.value(output, 0.0, Replayed.snoc(history, op), index),
        StorageLaws.latest(U32.to_nat(Shape.count(output)), op, index, Replayed.value(output, 0.0, history, index)), 0.0,
        Replayed.expected_snoc(history, op, U32.to_nat(Shape.count(output)), index, 0.0),
        Equal.trans(F32, StorageLaws.latest(U32.to_nat(Shape.count(output)), op, index, Replayed.value(output, 0.0, history, index)),
          Replayed.value(output, 0.0, history, index), 0.0,
          write_skip(output, previous, index, word, Replayed.value(output, 0.0, history, index),
            Conversion.le_successor_lt(previous, Shape.logical_numel(Shape.dimensions(output)), room),
            Conversion.le_successor_lt(previous, U32.to_nat(index), above)),
          writes_zero(inner, columns, left, left_fill, left_ops, right, right_fill, right_ops, output, previous, index,
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

def out_leaf(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +position: Nat, inside: Bool) -> F32:
  match inside:
    case False{}:
      0.0
    case True{}:
      cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, position)

def out_spec(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Array<F32>:
  match depth:
    case 0n:
      ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, Nat.is_lt(base, count))}
    case 1n+inner:
      ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, base), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, Nat.add(Model.capacity(inner), base))}

def out_balanced(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +count: Nat, +depth: Nat, +base: Nat) -> Model.Balanced(F32, depth, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, depth, base)):
  match depth:
    case 0n:
      Unit{}
    case 1n+inner:
      (out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, base), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, inner, Nat.add(Model.capacity(inner), base)))

def new_spec(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +depth: Nat, +base: Nat) -> {Array.new(F32, depth, 0.0) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(base, 0n), False{}, lt_zero_false(base)) :
        {ALeaf{0.0} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
      {==}
    case 1n+inner:
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, base), new_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, inner, base)) :
        {ANode{_, Array.new(F32, inner, 0.0)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 1n+inner, base) : Array<F32>}
      %Equal.sym(Array<F32>, Array.new(F32, inner, 0.0), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, Nat.add(Model.capacity(inner), base)),
        new_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, inner, Nat.add(Model.capacity(inner), base))) :
        {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, inner, base), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 1n+inner, base) : Array<F32>}
      {==}

# A subtree whose positions all lie above k is the same for k and 1 + k.
def above(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_lt(k, b) == True{} : Bool}) ->
  {out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, b) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(b, k), False{}, lt_asym(k, b, h)) :
        {ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), False{}, above_false(k, b, h)) :
        {ALeaf{0.0} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +high = Nat.add(Model.capacity(inner), b)
      %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), above(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b, h)) :
        {ANode{_, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high),
        above(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, Arith.lt_le_trans(k, b, high, h, Api.le_add_left(Model.capacity(inner), b)))) :
        {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# A subtree whose positions all lie below k is the same for k and 1 + k.
def below(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +b: Nat, +h: {Nat.is_le(Nat.add(Model.capacity(depth), b), k) == True{} : Bool}) ->
  {out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, b) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, b) : Array<F32>}:
  match depth:
    case 0n:
      +lower = below_true(b, k, h)
      %Equal.sym(Bool, Nat.is_lt(b, k), True{}, lower) :
        {ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, Nat.is_lt(b, 1n+k))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(b, 1n+k), True{}, lt_succ_right(b, k, lower)) :
        {ALeaf{cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b)} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, b, _)} : Array<F32>}
      {==}
    case 1n+inner:
      +c = Model.capacity(inner)
      +high = Nat.add(c, b)
      +nested = Equal.trans(Bool, Nat.is_le(Nat.add(c, high), k), Nat.is_le(Nat.add(Nat.add(c, c), b), k), True{},
        Equal.cong(Nat, Bool, x => Nat.is_le(x, k), Nat.add(c, high), Nat.add(Nat.add(c, c), b), Index.add_assoc(c, c, b)), h)
      +low_h = le_trans(high, Nat.add(c, high), k, le_add_cong(c, b, high, Api.le_add_left(c, b)), nested)
      %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), below(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, b, low_h)) :
        {ANode{_, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), below(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, nested)) :
        {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, b), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, b) : Array<F32>}
      {==}

# Writing cell k at the path whose position is k turns the spec for k into
# the spec for 1 + k.
def write_spec(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +k: Nat, +depth: Nat, +base: Nat, +path: Model.Path(depth),
  +h: {Nat.add(Model.offset(depth, path), base) == k : Nat}) ->
  {ArrayCore.write_path(F32, depth, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, depth, base), path, cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k)) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, depth, base) : Array<F32>}:
  match depth:
    case 0n:
      %h : {ALeaf{cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, _)} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, Nat.is_lt(base, 1n+_))} : Array<F32>}
      %Equal.sym(Bool, Nat.is_lt(base, 1n+base), True{}, lt_self_succ(base)) :
        {ALeaf{cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base)} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
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
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base), rest, cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k)),
            out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), write_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base, rest, h)) :
            {ANode{_, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), above(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, inside)) :
            {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
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
          %Equal.sym(Array<F32>, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), below(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, base, lower)) :
            {ANode{_, ArrayCore.write_path(F32, inner, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), rest, cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k))} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          %Equal.sym(Array<F32>, ArrayCore.write_path(F32, inner, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high), rest, cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k)),
            out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, high), write_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, inner, high, rest, shifted)) :
            {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, inner, base), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+k, 1n+inner, base) : Array<F32>}
          {==}

# The family history, replayed on the owned storage that `created` makes,
# is the owned storage that holds the output spec.
def chain(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +d: Nat, +n: Nat, -f: {Storage.bounds(d, n) == True{} : Bool}, +fits: {Storage.bounds(d, n) == True{} : Bool},
  +limit: U32, +n_limit: {Nat.is_le(n, U32.to_nat(limit)) == True{} : Bool},
  +k: Nat, +room: {Nat.is_le(k, n) == True{} : Bool}) ->
  {StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k, d, 0n), f} : Storage.Storage}:
  match k:
    case 0n:
      Owned.owned_eq(d, n, Array.new(F32, d, 0.0), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, d, 0n), new_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, d, 0n), ArrayCore.new_balanced(F32, d, 0.0), f)(
        out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 0n, d, 0n))
    case 1n+j:
      +index = U32.from_nat(j)
      +value = cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j)
      +below_n = Conversion.le_successor_lt(j, n, room)
      +exact = Conversion.natural_word_exact(j, limit, le_trans(j, n, U32.to_nat(limit), Conversion.lt_le(j, n, below_n), n_limit))
      +inside = Equal.trans(Bool, Nat.is_lt(U32.to_nat(index), n), Nat.is_lt(j, n), True{},
        Equal.cong(Nat, Bool, x => Nat.is_lt(x, n), U32.to_nat(index), j, exact), below_n)
      +offset = Equal.trans(Nat, Nat.add(Model.offset(d, Refinement.route(d, index)), 0n), U32.to_nat(index), j,
        output_position(d, n, index, fits, inside), exact)
      +written = Equal.trans(Array<F32>, Array.set(F32, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), ArrayCore.write_path(F32, d, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), Refinement.route(d, index), value),
        out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), Refinement.law_set_refines(F32, d, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)),
        write_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n, Refinement.route(d, index), offset))
      Equal.trans(Storage.Storage, StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        StorageLaws.step(StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), StorageLaws.Write{index, value}),
        Storage.Owned{d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f},
        FromValues.replay_snoc(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j), StorageLaws.Write{index, value}, Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
        %Equal.sym(Storage.Storage, StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j), Storage.Owned{d, n, Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}), Storage.Owned{d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f},
          chain(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, d, n, f, fits, limit, n_limit, j, Conversion.le_predecessor(j, n, room))) :
          {StorageLaws.step(_, StorageLaws.Write{index, value}) == Storage.Owned{d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f} : Storage.Storage}
        %Equal.sym(Storage.Storage & Result<&2, &2, Storage.Error, Unit>,
          Storage.set_checked(d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f),
          (Storage.Owned{d, n, Array.set(F32, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value),
            Certificates.set_result(F32, d, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Array.set(F32, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), {==}, out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)), f}, Done{Unit{}}),
          Api.set_checked_hit(d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Nat.is_lt(U32.to_nat(index), n), {==}, out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), f,
            Equal.sym(Bool, Nat.is_lt(U32.to_nat(index), n), True{}, inside), fits, inside)) :
          {StorageLaws.owner(Result<&2, &2, Storage.Error, Unit>, _) == Storage.Owned{d, n, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), f} : Storage.Storage}
        Owned.owned_eq(d, n, Array.set(F32, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n), written,
          Certificates.set_result(F32, d, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value, Array.set(F32, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n), index, value), {==}, out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, j, d, 0n)), f)(
          out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, 1n+j, d, 0n)))


# The copy the build reads of a replayed input in normal form: the Second
# half of the clone of the array its normalized history makes.
def history(+shape: Shape.Shape, +ops: List<&2, StorageLaws.Op>) -> List<&2, Api.Effect>:
  Api.Right{} <> Owned.norm(U32.to_nat(Shape.count(shape)), ops, Nil{})

def copy(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +depth: Nat) -> Array<F32>:
  Api.arr(history(shape, ops), depth, fill)

def copy_valid(+shape: Shape.Shape, +ops: List<&2, StorageLaws.Op>) -> {Build.written(U32.to_nat(Shape.count(shape)), history(shape, ops)) == True{} : Bool}:
  Owned.norm_written(U32.to_nat(Shape.count(shape)), ops, Nil{}, Api.hit_of(U32.to_nat(Shape.count(shape)), ops), {==}, {==})

# A read of the copy below the input's count holds the replayed word.
def copy_word(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Api.current(history(shape, ops), index, fill) == Replayed.value(shape, fill, ops, index) : F32}:
  Owned.norm_current(U32.to_nat(Shape.count(shape)), ops, Nil{}, index, fill, Api.hit_of(U32.to_nat(Shape.count(shape)), ops), {==})

def index_inside(+shape: Shape.Shape, +number: Nat, +below: {Nat.is_lt(number, Shape.logical_numel(Shape.dimensions(shape))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(number)), U32.to_nat(Shape.count(shape))) == True{} : Bool}:
  PointwiseProof.lt_count(U32.to_nat(U32.from_nat(number)), shape, MatrixIndex.word_inside(number, shape, below))

# The terms `cursor .. cursor + fuel` of cell (row, column) over the replayed
# words, added to `accumulator` in increasing k.
def fold_from(fuel: Nat, +span: Nat, +columns: Nat, +row: Nat, +column: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat, accumulator: F32) -> F32:
  match fuel:
    case 0n:
      accumulator
    case 1n+rest:
      fold_from(rest, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor,
        F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, U32.from_nat(Nat.add(Nat.mul(row, span), cursor))),
          Replayed.value(right, right_fill, right_ops, U32.from_nat(Nat.add(Nat.mul(cursor, columns), column))))))

def fold_factors(+fuel: Nat, +span: Nat, +columns: Nat, +row: Nat, +column: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>,
  +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +cursor: Nat,
  +steps: {Nat.add(cursor, fuel) == span : Nat}) ->
  {fold_from(fuel, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, cursor,
    DotFold.dot_fold(Laws.factors(row, column, span, columns, left, left_fill, left_ops, right, right_fill, right_ops, cursor), 0.0)) ==
    DotFold.dot_fold(Laws.factors(row, column, span, columns, left, left_fill, left_ops, right, right_fill, right_ops, span), 0.0) : F32}:
  match fuel:
    case 0n:
      Equal.cong(Nat, F32,
        k => DotFold.dot_fold(Laws.factors(row, column, span, columns, left, left_fill, left_ops, right, right_fill, right_ops, k), 0.0),
        cursor, span, PointwiseProof.steps_done(cursor, span, steps))
    case 1n+rest:
      +x = Replayed.value(left, left_fill, left_ops, U32.from_nat(Nat.add(Nat.mul(row, span), cursor)))
      +y = Replayed.value(right, right_fill, right_ops, U32.from_nat(Nat.add(Nat.mul(cursor, columns), column)))
      +before = Laws.factors(row, column, span, columns, left, left_fill, left_ops, right, right_fill, right_ops, cursor)
      %DotProof.dot_snoc(before, x, y, 0.0) :
        {fold_from(rest, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor, _) ==
          DotFold.dot_fold(Laws.factors(row, column, span, columns, left, left_fill, left_ops, right, right_fill, right_ops, span), 0.0) : F32}
      fold_factors(rest, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 1n+cursor,
        MatrixIndex.steps_next(cursor, rest, span, steps))


# How a build copy answers the reads of its input: an empty input has no
# cell to read, and a normal one holds the input's normalized history.
type Source<-shape: Shape.Shape, -fill: F32, -ops: List<&2, StorageLaws.Op>, -h: List<&2, Api.Effect>, -d: Nat, -f: F32> is Data:
  NoCells{zero: {U32.to_nat(Shape.count(shape)) == 0n : Nat}}
  Normal{fits: {Storage.bounds(d, U32.to_nat(Shape.count(shape))) == True{} : Bool},
    same: {h == history(shape, ops) : List<&2, Api.Effect>}, filled: {f == fill : F32}}

def source_read(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +h: List<&2, Api.Effect>, +d: Nat, +f: F32,
  +source: Source<shape, fill, ops, h, d, f>, +index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index), U32.to_nat(Shape.count(shape))) == True{} : Bool}, -pair: Array<F32> & F32,
  +exact: {pair == Array.get(F32, Api.arr(h, d, f), index) : Array<F32> & F32}) ->
  {Refinement.second(F32, pair) == Replayed.value(shape, fill, ops, index) : F32}:
  match source:
    case NoCells{zero}:
      Empty.absurd({Refinement.second(F32, pair) == Replayed.value(shape, fill, ops, index) : F32},
        Arith.lt_zero(U32.to_nat(index),
          %zero : {Nat.is_lt(U32.to_nat(index), _) == True{} : Bool}
          inside))
    case Normal{fits, same, filled}:
      Equal.trans(F32, Refinement.second(F32, pair), Api.current(h, index, f), Replayed.value(shape, fill, ops, index),
        Build.got_value(d, h, f, U32.to_nat(Shape.count(shape)), index, pair, exact, fits, inside,
          %Equal.sym(List<&2, Api.Effect>, h, history(shape, ops), same) : {Build.written(U32.to_nat(Shape.count(shape)), _) == True{} : Bool}
          copy_valid(shape, ops)),
        %Equal.sym(List<&2, Api.Effect>, h, history(shape, ops), same) : {Api.current(_, index, f) == Replayed.value(shape, fill, ops, index) : F32}
        %Equal.sym(F32, f, fill, filled) : {Api.current(history(shape, ops), index, _) == Replayed.value(shape, fill, ops, index) : F32}
        copy_word(shape, fill, ops, index))

def DotIs(+fuel: Nat, +span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +row: Nat, +column: Nat, +cursor: Nat, +accumulator: F32,
  -result: F32 & (Array<F32> & Array<F32>)) -> Type:
  {result == (fold_from(fuel, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, cursor, accumulator), (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))) : F32 & (Array<F32> & Array<F32>)}

def dot_right_eq(+rest: Nat, +span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>, +rows: Nat, +row: Nat, +column: Nat, +cursor: Nat, +accumulator: F32,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>}, +below: {Nat.is_lt(cursor, span) == True{} : Bool},
  -x: F32, -kept: Array<F32>, pair: Array<F32> & F32,
  +exact: {pair == Array.get(F32, Api.arr(hb, dr, fb), Matrix.right_index(columns, column, cursor)) : Array<F32> & F32},
  ex: {x == Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)) : F32}, ek: {kept == Api.arr(ha, dl, fa) : Array<F32>},
  next: DotIs(rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)))) ->
  DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator,
    Matrix.term_right(value => l => r => Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, value, l, r), accumulator, x, kept, pair)):
  (other, y) = pair
  +ey = source_read(right, right_fill, right_ops, hb, dr, fb, rsrc, Matrix.right_index(columns, column, cursor),
    index_inside(right, Nat.add(Nat.mul(cursor, columns), column), MatrixIndex.right_below(span, columns, column, column_below, right, right_extents, cursor, below)),
    (other, y), exact)
  %Equal.sym(F32, x, Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), ex) :
    DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator, Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(_, y)), kept, other))
  %Equal.sym(F32, y, Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)), ey) :
    DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator, Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), _)), kept, other))
  %Equal.sym(Array<F32>, kept, Api.arr(ha, dl, fa), ek) :
    DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator, Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), _, other))
  %Equal.sym(Array<F32>, other, Api.arr(hb, dr, fb), Build.got_kept(dr, hb, fb, Matrix.right_index(columns, column, cursor), (other, y), exact)) :
    DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator, Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), Api.arr(ha, dl, fa), _))
  next

def dot_left_eq(+rest: Nat, +span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>, +rows: Nat, +row: Nat, +column: Nat, +cursor: Nat, +accumulator: F32,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>}, +below: {Nat.is_lt(cursor, span) == True{} : Bool},
  pair: Array<F32> & F32,
  +exact: {pair == Array.get(F32, Api.arr(ha, dl, fa), Matrix.left_index(span, row, cursor)) : Array<F32> & F32},
  next: DotIs(rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)))) ->
  DotIs(1n+rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator,
    Matrix.term_left(value => l => r => Matrix.cell_dot(rest, span, columns, row, column, 1n+cursor, value, l, r), accumulator, Api.arr(hb, dr, fb), Matrix.right_index(columns, column, cursor), pair)):
  (kept, x) = pair
  dot_right_eq(rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, lsrc, rsrc, rows, row, column, cursor, accumulator, row_below, left_extents, column_below, right_extents, below, x, kept,
    Array.get(F32, Api.arr(hb, dr, fb), Matrix.right_index(columns, column, cursor)), {==},
    source_read(left, left_fill, left_ops, ha, dl, fa, lsrc, Matrix.left_index(span, row, cursor),
      index_inside(left, Nat.add(Nat.mul(row, span), cursor), MatrixIndex.left_below(rows, span, row, row_below, left, left_extents, cursor, below)),
      (kept, x), exact),
    Build.got_kept(dl, ha, fa, Matrix.left_index(span, row, cursor), (kept, x), exact),
    next)

# Cell (row, column) of the build over the two copies folds the replayed
# words in increasing k and hands both copies back.
def dot_eq(+fuel: Nat, +span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>, +rows: Nat, +row: Nat, +column: Nat, +cursor: Nat, +accumulator: F32,
  +row_below: {Nat.is_lt(row, rows) == True{} : Bool}, +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +column_below: {Nat.is_lt(column, columns) == True{} : Bool}, +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>}, +steps: {Nat.add(cursor, fuel) == span : Nat}) ->
  DotIs(fuel, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, row, column, cursor, accumulator, Matrix.cell_dot(fuel, span, columns, row, column, cursor, accumulator, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))):
  match fuel:
    case 0n:
      {==}
    case 1n+rest:
      dot_left_eq(rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, lsrc, rsrc, rows, row, column, cursor, accumulator, row_below, left_extents, column_below, right_extents, MatrixIndex.cursor_below(cursor, rest, span, steps),
        Array.get(F32, Api.arr(ha, dl, fa), Matrix.left_index(span, row, cursor)), {==},
        dot_eq(rest, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, lsrc, rsrc, rows, row, column, 1n+cursor, F32.add(accumulator, F32.mul(Replayed.value(left, left_fill, left_ops, Matrix.left_index(span, row, cursor)), Replayed.value(right, right_fill, right_ops, Matrix.right_index(columns, column, cursor)))), row_below, left_extents, column_below, right_extents, MatrixIndex.steps_next(cursor, rest, span, steps)))

def TreeEq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>)) -> Type:
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, depth, base) : Array<F32>} &
    ({Pair.fst(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == Api.arr(ha, dl, fa) : Array<F32>} &
      {Pair.snd(Array<F32>, Array<F32>, Pair.snd(Array<F32>, Array<F32> & Array<F32>, result)) == Api.arr(hb, dr, fb) : Array<F32>})

def cell_inside_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>,
  +output: Shape.Shape, +rows: Nat,
  +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>},
  +output_extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>},
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(output)) : Nat}, +base: Nat,
  +decided: {True{} == Nat.is_lt(base, count) : Bool}) ->
  TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 0n, base, Matrix.cell(span, columns, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb), True{})):
  +row = Matrix.row_of(columns, base)
  +column = Matrix.column_of(columns, base)
  +below = MatrixIndex.lt_bound(base, count, Shape.logical_numel(Shape.dimensions(output)), count_exact,
    Equal.sym(Bool, True{}, Nat.is_lt(base, count), decided))
  +fit = MatrixIndex.position_fits(output, rows, columns, output_extents, base, below)
  +row_below = Arith.and_left(Nat.is_lt(row, rows), Bool.and(Nat.is_lt(column, columns), True{}), fit)
  +column_below = Arith.and_left(Nat.is_lt(column, columns), True{},
    Arith.and_right(Nat.is_lt(row, rows), Bool.and(Nat.is_lt(column, columns), True{}), fit))
  %Equal.sym(F32 & (Array<F32> & Array<F32>),
    Matrix.dot_of(span, columns, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
    (fold_from(span, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 0.0), (Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))),
    dot_eq(span, span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, lsrc, rsrc, rows, row, column, 0n, 0.0, row_below, left_extents, column_below, right_extents, {==})) :
    TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 0n, base, Matrix.cell_leaf(_))
  (%decided : {ALeaf{fold_from(span, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 0.0)} ==
      ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
    Equal.cong(F32, Array<F32>, w => ALeaf{w},
      fold_from(span, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 0n, 0.0),
      cell_word(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base),
      fold_factors(span, span, columns, row, column, left, left_fill, left_ops, right, right_fill, right_ops, 0n, {==})),
   ({==}, {==}))

def cell_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>,
  +output: Shape.Shape, +rows: Nat,
  +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>},
  +output_extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>},
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(output)) : Nat}, +base: Nat, inside: Bool,
  +decided: {inside == Nat.is_lt(base, count) : Bool}) ->
  TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 0n, base, Matrix.cell(span, columns, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb), inside)):
  match inside:
    case False{}:
      (%decided : {ALeaf{0.0} == ALeaf{out_leaf(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, base, _)} : Array<F32>}
       {==}, ({==}, {==}))
    case True{}:
      cell_inside_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, base, decided)

def tree_join_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lower: Nat, +base: Nat, -a: Array<F32>, high: Array<F32> & (Array<F32> & Array<F32>),
  ea: {a == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base) : Array<F32>},
  facts: TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, Nat.add(Model.capacity(lower), base), high)) ->
  TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 1n+lower, base, Elementwise.tree_join(a, high)):
  (b, inputs) = high
  (eb, kept) = facts
  (%Equal.sym(Array<F32>, a, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), ea) :
     {ANode{_, b} == ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, Nat.add(Model.capacity(lower), base))} : Array<F32>}
   %Equal.sym(Array<F32>, b, out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, Nat.add(Model.capacity(lower), base)), eb) :
     {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), _} == ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, Nat.add(Model.capacity(lower), base))} : Array<F32>}
   {==}, kept)

def tree_node_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lower: Nat, +base: Nat, low: Array<F32> & (Array<F32> & Array<F32>),
  facts: TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, base, low),
  next: TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, Nat.add(Model.capacity(lower), base), Matrix.tree(lower, span, columns, count, Nat.add(Model.capacity(lower), base), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)))) ->
  TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 1n+lower, base, Elementwise.tree_node(l => r => Matrix.tree(lower, span, columns, count, Nat.add(Model.capacity(lower), base), l, r), low)):
  (a, inputs) = low
  (la, ra) = inputs
  (ea, kept) = facts
  (el, er) = kept
  %Equal.sym(Array<F32>, la, Api.arr(ha, dl, fa), el) :
    TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 1n+lower, base, Elementwise.tree_join(a, Matrix.tree(lower, span, columns, count, Nat.add(Model.capacity(lower), base), _, ra)))
  %Equal.sym(Array<F32>, ra, Api.arr(hb, dr, fb), er) :
    TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 1n+lower, base, Elementwise.tree_join(a, Matrix.tree(lower, span, columns, count, Nat.add(Model.capacity(lower), base), Api.arr(ha, dl, fa), _)))
  tree_join_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, base, a, Matrix.tree(lower, span, columns, count, Nat.add(Model.capacity(lower), base), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), ea, next)

def tree_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>,
  +output: Shape.Shape, +rows: Nat,
  +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>},
  +output_extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>},
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(output)) : Nat}, +depth: Nat, +base: Nat) ->
  TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, depth, base, Matrix.tree(depth, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))):
  match depth:
    case 0n:
      cell_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, base, Nat.is_lt(base, count), {==})
    case 1n+lower:
      tree_node_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, base, Matrix.tree(lower, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
        tree_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, lower, base),
        tree_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, lower, Nat.add(Model.capacity(lower), base)))

def tree_first(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +depth: Nat, +base: Nat, -result: Array<F32> & (Array<F32> & Array<F32>), facts: TreeEq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, depth, base, result)) ->
  {Pair.fst(Array<F32>, Array<F32> & Array<F32>, result) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, depth, base) : Array<F32>}:
  (first, kept) = facts
  first

def fork_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lower: Nat, +rest: Nat, +base: Nat,
  low: {Matrix.build(lower, rest, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base) : Array<F32>},
  high: {Matrix.build(lower, rest, span, columns, count, Nat.add(Model.capacity(lower), base), Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, Nat.add(Model.capacity(lower), base)) : Array<F32>},
  lp: Array<F32> & Array<F32>, +le: {lp == Array.clone(F32, Api.arr(ha, dl, fa)) : Array<F32> & Array<F32>},
  rp: Array<F32> & Array<F32>, +re: {rp == Array.clone(F32, Api.arr(hb, dr, fb)) : Array<F32> & Array<F32>}) ->
  {Elementwise.fork(lp, rp, l => r => Matrix.build(lower, rest, span, columns, count, base, l, r),
    l => r => Matrix.build(lower, rest, span, columns, count, Nat.add(Model.capacity(lower), base), l, r)) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}:
  (l1, l2) = lp
  (r1, r2) = rp
  +upper = Nat.add(Model.capacity(lower), base)
  %Equal.sym(Array<F32>, l1, Api.arr(ha, dl, fa), Build.clone_first_eq(dl, ha, fa, (l1, l2), le)) :
    {ANode{Matrix.build(lower, rest, span, columns, count, base, _, r1), Matrix.build(lower, rest, span, columns, count, upper, l2, r2)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, r1, Api.arr(hb, dr, fb), Build.clone_first_eq(dr, hb, fb, (r1, r2), re)) :
    {ANode{Matrix.build(lower, rest, span, columns, count, base, Api.arr(ha, dl, fa), _), Matrix.build(lower, rest, span, columns, count, upper, l2, r2)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, l2, Api.arr(ha, dl, fa), Build.clone_second_eq(dl, ha, fa, (l1, l2), le)) :
    {ANode{Matrix.build(lower, rest, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Matrix.build(lower, rest, span, columns, count, upper, _, r2)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, r2, Api.arr(hb, dr, fb), Build.clone_second_eq(dr, hb, fb, (r1, r2), re)) :
    {ANode{Matrix.build(lower, rest, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Matrix.build(lower, rest, span, columns, count, upper, Api.arr(ha, dl, fa), _)} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, Matrix.build(lower, rest, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), low) :
    {ANode{_, Matrix.build(lower, rest, span, columns, count, upper, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  %Equal.sym(Array<F32>, Matrix.build(lower, rest, span, columns, count, upper, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, upper), high) :
    {ANode{out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, lower, base), _} == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, 1n+lower, base) : Array<F32>}
  {==}

# Whatever the split, the parallel build over the copies returns the spec.
def build_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +count: Nat, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>,
  +output: Shape.Shape, +rows: Nat,
  +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>},
  +output_extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>},
  +count_exact: {count == Shape.logical_numel(Shape.dimensions(output)) : Nat}, +depth: Nat, +splits: Nat, +base: Nat) ->
  {Matrix.build(depth, splits, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)) == out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, count, depth, base) : Array<F32>}:
  match depth splits:
    case 1n+lower 1n+rest:
      fork_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lower, rest, base, build_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, lower, rest, base),
        build_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, lower, rest, Nat.add(Model.capacity(lower), base)),
        Array.clone(F32, Api.arr(ha, dl, fa)), {==}, Array.clone(F32, Api.arr(hb, dr, fb)), {==})
    case 0n _:
      tree_first(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 0n, base, Matrix.tree(0n, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), tree_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, 0n, base))
    case 1n+lower 0n:
      tree_first(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, 1n+lower, base, Matrix.tree(1n+lower, span, columns, count, base, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), tree_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, count, lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, count_exact, 1n+lower, base))

# The writes of every cell, replayed on the owned storage `created` makes for
# the output, give the owned storage the parallel build fills.
def output_eq(+span: Nat, +columns: Nat, +left: Shape.Shape, +left_fill: F32, +left_ops: List<&2, StorageLaws.Op>, +ha: List<&2, Api.Effect>, +dl: Nat, +fa: F32, +right: Shape.Shape, +right_fill: F32, +right_ops: List<&2, StorageLaws.Op>, +hb: List<&2, Api.Effect>, +dr: Nat, +fb: F32, +lsrc: Source<left, left_fill, left_ops, ha, dl, fa>, +rsrc: Source<right, right_fill, right_ops, hb, dr, fb>,
  +output: Shape.Shape, +rows: Nat,
  +left_extents: {Shape.extents(left) == [rows, span] : List<&2, Nat>},
  +right_extents: {Shape.extents(right) == [span, columns] : List<&2, Nat>},
  +output_extents: {Shape.extents(output) == [rows, columns] : List<&2, Nat>},
  +d: Nat, +fits: {Storage.bounds(d, U32.to_nat(Shape.count(output))) == True{} : Bool}, -f: {Storage.bounds(d, U32.to_nat(Shape.count(output))) == True{} : Bool}) ->
  {StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Owned{d, U32.to_nat(Shape.count(output)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}) == Storage.Owned{d, U32.to_nat(Shape.count(output)), Matrix.build(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Matrix.build_balanced(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), f} : Storage.Storage}:
  Equal.trans(Storage.Storage, StorageLaws.replay(writes(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output))), Storage.Owned{d, U32.to_nat(Shape.count(output)), Array.new(F32, d, 0.0), ArrayCore.new_balanced(F32, d, 0.0), f}),
    Storage.Owned{d, U32.to_nat(Shape.count(output)), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)), d, 0n), out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)), d, 0n), f},
    Storage.Owned{d, U32.to_nat(Shape.count(output)), Matrix.build(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), Matrix.build_balanced(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), f},
    chain(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, d, U32.to_nat(Shape.count(output)), f, fits, Shape.count(output), Conversion.le_reflexive(U32.to_nat(Shape.count(output))), U32.to_nat(Shape.count(output)), Conversion.le_reflexive(U32.to_nat(Shape.count(output)))),
    Owned.owned_eq(d, U32.to_nat(Shape.count(output)), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)), d, 0n), Matrix.build(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)),
      Equal.sym(Array<F32>, Matrix.build(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb)), out_spec(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)), d, 0n),
        build_eq(span, columns, left, left_fill, left_ops, ha, dl, fa, right, right_fill, right_ops, hb, dr, fb, U32.to_nat(Shape.count(output)), lsrc, rsrc, output, rows, left_extents, right_extents, output_extents, Reads.count_exact(output), d, Matrix.splits(d, span), 0n)),
      out_balanced(span, columns, left, left_fill, left_ops, right, right_fill, right_ops, U32.to_nat(Shape.count(output)), d, 0n), f)(Matrix.build_balanced(d, Matrix.splits(d, span), span, columns, U32.to_nat(Shape.count(output)), 0n, Api.arr(ha, dl, fa), Api.arr(hb, dr, fb))))

# The inputs of the parallel product, as the elementwise export takes them.
def half_keeps(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +index: U32) ->
  {Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), index) == Replayed.value(shape, fill, ops, index) : F32}:
  Replayed.expected_snoc(ops, StorageLaws.First{}, U32.to_nat(Shape.count(shape)), index, fill)

def words_snoc(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +previous: Nat, +word: F32, +other: F32,
  +before: List<&2, F32>, rest: {Laws.words(shape, fill, ops, previous) == before : List<&2, F32>}, last: {word == other : F32}) ->
  {List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]) == List.append(&2, F32, before, [other]) : List<&2, F32>}:
  Equal.trans(List<&2, F32>, List.append(&2, F32, Laws.words(shape, fill, ops, previous), [word]),
    List.append(&2, F32, before, [word]), List.append(&2, F32, before, [other]),
    Equal.cong(List<&2, F32>, List<&2, F32>, values => List.append(&2, F32, values, [word]), Laws.words(shape, fill, ops, previous), before, rest),
    Equal.cong(F32, List<&2, F32>, value => List.append(&2, F32, before, [value]), word, other, last))

# The First half of a clone lists the same words.

def words_first(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, +count: Nat) ->
  {Laws.words(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), count) == Laws.words(shape, fill, ops, count) : List<&2, F32>}:
  match count:
    case 0n:
      {==}
    case 1n+previous:
      +position = U32.from_nat(previous)
      words_snoc(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), previous,
        Replayed.value(shape, fill, Replayed.snoc(ops, StorageLaws.First{}), position), Replayed.value(shape, fill, ops, position),
        Laws.words(shape, fill, ops, previous), words_first(shape, fill, ops, previous), half_keeps(shape, fill, ops, position))

# `take` of an owned tensor hands back the First half of its clone and the
# Second half's array.

def take_owned_lift(+shape: Shape.Shape, +d: Nat, +n: Nat, -array: Array<F32>, pair: Array<F32> & Array<F32>,
  -same: {pair == Array.clone(F32, array) : Array<F32> & Array<F32>}, -b: Model.Balanced(F32, d, array),
  -f: {Storage.bounds(d, n) == True{} : Bool}, -fits: {n == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  -Q: (Core.Dynamic & Array<F32>) -> Type,
  h: @-kept: {Lifecycle.logical_length(StorageLaws.owner(Storage.Storage, Storage.finish_clone(d, n, array, pair, same, b, f))) ==
    Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q((Core.Packed{shape, Core.TTensor{StorageLaws.owner(Storage.Storage, Storage.finish_clone(d, n, array, pair, same, b, f)), kept}},
      Pair.snd(Array<F32>, Array<F32>, pair)))) ->
  Q(Elementwise.take_owned(shape, d, n, array, pair, same, b, f, fits)):
  (kept, copy) = pair
  h(fits)

def storage_shift(-shape: Shape.Shape, -s: Storage.Storage, -t: Storage.Storage, e: {s == t : Storage.Storage},
  -R: @st: Storage.Storage -> @-k: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> Type,
  h: @-k: {Lifecycle.logical_length(s) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(s, k)) ->
  @-k: {Lifecycle.logical_length(t) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(t, k):
  %e : (@-k: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(_, k))
  h


# What the rest of the product asks of one input's `take`: the kept tensor,
# replayed with a history that keeps its words, and the copy the build reads,
# with its source.
def InputAt(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, -s: Storage.Storage,
  -Q: Core.Dynamic & Array<F32> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @values: (@+other: U32 -> {Replayed.value(shape, fill, kept, other) == Replayed.value(shape, fill, ops, other) : F32}) ->
  @words: {Laws.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) == Laws.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) : List<&2, F32>} ->
  @+h: List<&2, Api.Effect> -> @+d: Nat -> @+f: F32 -> @+source: Source<shape, fill, ops, h, d, f> ->
  @-kept_fits: {Lifecycle.logical_length(StorageLaws.replay(kept, s)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
    Q((Core.Packed{shape, Core.TTensor{StorageLaws.replay(kept, s), kept_fits}}, Api.arr(h, d, f)))

def input_take(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>, -Q: Core.Dynamic & Array<F32> -> Type,
  h: InputAt(shape, fill, ops, StorageLaws.created(Shape.count(shape), fill), Q)) ->
  @-fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  Q(Elementwise.take(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})):
  Owned.created_lift(Shape.count(shape), fill, Conversion.shape_count_within(shape),
    s => (InputAt(shape, fill, ops, s, Q) -> @-fits: {Lifecycle.logical_length(StorageLaws.replay(ops, s)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
      Q(Elementwise.take(Core.Packed{shape, Core.TTensor{StorageLaws.replay(ops, s), fits}}))),
    zero => k => fits =>
      storage_shift(shape, Storage.Empty{}, StorageLaws.replay(ops, Storage.Empty{}),
        Equal.sym(Storage.Storage, StorageLaws.replay(ops, Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(ops)),
        st => kk => Q(Elementwise.take(Core.Packed{shape, Core.TTensor{st, kk}})),
        kk => storage_shift(shape, StorageLaws.replay(ops, Storage.Empty{}), Storage.Empty{}, Owned.replay_empty(ops),
          st => k2 => Q((Core.Packed{shape, Core.TTensor{st, k2}}, Array.new(F32, 0n, 0.0))),
          k2 => k(ops)(other => {==})({==})(Nil{})(0n)(0.0)(NoCells{zero})(k2))(kk))(fits),
    d => fits_d => f => k => fits =>
      Owned.replay_lift(ops, d, U32.to_nat(Shape.count(shape)), fill, Nil{}, Api.hit_of(U32.to_nat(Shape.count(shape)), ops), {==},
        ArrayCore.new_balanced(F32, d, fill), f, fits_d,
        st => (@-fits: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
          (@-kk: {Lifecycle.logical_length(StorageLaws.step(st, StorageLaws.First{})) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
            Q((Core.Packed{shape, Core.TTensor{StorageLaws.step(st, StorageLaws.First{}), kk}}, Api.arr(history(shape, ops), d, fill)))) ->
          Q(Elementwise.take(Core.Packed{shape, Core.TTensor{st, fits}}))),
        c => fits2 => kc =>
          take_owned_lift(shape, d, U32.to_nat(Shape.count(shape)), Api.arr(Owned.norm(U32.to_nat(Shape.count(shape)), ops, Nil{}), d, fill),
            Array.clone(F32, Api.arr(Owned.norm(U32.to_nat(Shape.count(shape)), ops, Nil{}), d, fill)), {==}, c, f, fits2, Q, kc))(fits)(
        storage_shift(shape,
          StorageLaws.replay(Replayed.snoc(ops, StorageLaws.First{}), Storage.Owned{d, U32.to_nat(Shape.count(shape)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f}),
          StorageLaws.step(StorageLaws.replay(ops, Storage.Owned{d, U32.to_nat(Shape.count(shape)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f}), StorageLaws.First{}),
          FromValues.replay_snoc(ops, StorageLaws.First{}, Storage.Owned{d, U32.to_nat(Shape.count(shape)), Array.new(F32, d, fill), ArrayCore.new_balanced(F32, d, fill), f}),
          st => kk => Q((Core.Packed{shape, Core.TTensor{st, kk}}, Api.arr(history(shape, ops), d, fill))),
          k(Replayed.snoc(ops, StorageLaws.First{}))(other => half_keeps(shape, fill, ops, other))(words_first(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))))(
            history(shape, ops))(d)(fill)(Normal{fits_d, {==}, {==}}))))(h)
