import Base
import ../../shape/core.bend as Shape
import ../../shape/index_proofs.bend as Index
import ../../shape/nat_proofs.bend as Nats
import ../../storage/model.bend as Model
import ../../storage/main.bend as Storage
import ../../storage/LAWS.bend as StorageLaws
import ../../storage/array.bend as ArrayCore
import ../../storage/refinement.bend as Refinement
import ../../storage/api.bend as Api
import ../../storage/arith.bend as Arith
import ../../storage/route.bend as Route
import ../lifecycle.bend as Lifecycle
import ../word_conversion.bend as Conversion
import ./replayed.bend as Replayed
import ./binary_storage_proof.bend as Owned

# Facts the exports of the parallel builds share: natural-number facts about
# the positions of a balanced tree, and two lifts of replayed storages. The
# elementwise export (binary_proofs.bend) and the permute export
# (../permute_build_proof.bend) use them.

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

# The output spec: cell `base` of the tree holds the family cell below
# `count` and +0.0 above it.

def storage_shift(-shape: Shape.Shape, -s: Storage.Storage, -t: Storage.Storage, e: {s == t : Storage.Storage},
  -R: @st: Storage.Storage -> @-k: {Lifecycle.logical_length(st) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> Type,
  h: @-k: {Lifecycle.logical_length(s) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(s, k)) ->
  @-k: {Lifecycle.logical_length(t) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(t, k):
  %e : (@-k: {Lifecycle.logical_length(_) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} -> R(_, k))
  h

# A replayed tensor with a cell is an owned storage whose array is `arr` of
# its normalized history.
def input_lift(+shape: Shape.Shape, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +nonzero: {Nat.is_lt(0n, U32.to_nat(Shape.count(shape))) == True{} : Bool}, -Q: Storage.Storage -> Type,
  owned: @+d: Nat -> @+fits: {Storage.bounds(d, U32.to_nat(Shape.count(shape))) == True{} : Bool} ->
    @-b: Model.Balanced(F32, d, Api.arr(Owned.norm(U32.to_nat(Shape.count(shape)), ops, Nil{}), d, fill)) ->
    @-f: {Storage.bounds(d, U32.to_nat(Shape.count(shape))) == True{} : Bool} ->
    Q(Storage.Owned{d, U32.to_nat(Shape.count(shape)), Api.arr(Owned.norm(U32.to_nat(Shape.count(shape)), ops, Nil{}), d, fill), b, f})) ->
  Q(Replayed.storage(shape, fill, ops)):
  Owned.created_lift(Shape.count(shape), fill, Conversion.shape_count_within(shape), s => Q(StorageLaws.replay(ops, s)),
    zero => Empty.absurd(Q(StorageLaws.replay(ops, Storage.Empty{})),
      Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(0n, U32.to_nat(Shape.count(shape))), True{},
        Equal.cong(Nat, Bool, x => Nat.is_lt(0n, x), 0n, U32.to_nat(Shape.count(shape)), Equal.sym(Nat, U32.to_nat(Shape.count(shape)), 0n, zero)),
        nonzero))),
    d => fits => f => Owned.replay_lift(ops, d, U32.to_nat(Shape.count(shape)), fill, Nil{}, Api.hit_of(U32.to_nat(Shape.count(shape)), ops), {==},
      ArrayCore.new_balanced(F32, d, fill), f, fits, Q, c => owned(d, fits, c, f)))

# Both inputs are owned storages in normal form and the output storage is
# owned: the parallel build returns the replayed output of the family
# history, and the inputs come back as the First halves of their clones.
