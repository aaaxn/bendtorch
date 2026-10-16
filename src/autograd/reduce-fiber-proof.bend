import Base
import ../shape/core.bend as Shape
import ../shape/nat_proofs.bend as Nats
import ../shape/index_proofs.bend as Index
import ../shape/broadcast_proofs.bend as Broadcast
import ../storage/arith.bend as Arith
import ../storage/api.bend as Api
import ../tensor/matrix-index-proof.bend as MatrixIndex
import ./reduce.bend as Reduce

# The fiber walk visits the fiber. For index rows of a broadcast whose left
# (target) extents are each 1 or the output extent, and a target cell below
# the target's element count, the contribution offsets that `Reduce.fiber`
# reads are, in order, exactly the offsets `walk_right` names at the
# positions p = 0, 1, ..., count - 1 whose `walk_left` is the cell: the
# positions the masked fold of `ReduceLaws.fiber` adds, in the order it adds
# them. Everything here is about natural numbers and lists of them.

# Arithmetic.

def add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  %Broadcast.add_zero(b) : {Nat.add(a, _) == Nat.add(b, a) : Nat}
  %Broadcast.add_zero(a) : {Nat.add(a, Nat.add(b, 0n)) == Nat.add(b, _) : Nat}
  Nats.add_swap(a, b, 0n)

def lt_succ_eq(+r: Nat, +size: Nat, below: {Nat.is_lt(r, size) == True{} : Bool},
  not_below: {False{} == Nat.is_lt(1n+r, size) : Bool}) -> {size == 1n+r : Nat}:
  match r size:
    case 0n 0n:
      Empty.absurd({0n == 1n : Nat}, Arith.clash(below))
    case 0n 1n+s:
      match s:
        case 0n:
          {==}
        case 1n+t:
          Empty.absurd({2n+t == 1n : Nat}, Arith.clash(not_below))
    case 1n+p 0n:
      Empty.absurd({0n == 2n+p : Nat}, Arith.clash(below))
    case 1n+p 1n+s:
      Equal.cong(Nat, Nat, n => 1n+n, s, 1n+p, lt_succ_eq(p, s, below, not_below))

# t = quotient * size + remainder with remainder < size.
type Split<-t: Nat, -size: Nat> is Data:
  Split{quotient: Nat, remainder: Nat, below: {Nat.is_lt(remainder, size) == True{} : Bool},
    sum: {Nat.add(Nat.mul(quotient, size), remainder) == t : Nat}}

def split_step(+t: Nat, +size: Nat, +q: Nat, +r: Nat, below: {Nat.is_lt(r, size) == True{} : Bool},
  sum: {Nat.add(Nat.mul(q, size), r) == t : Nat}, last: Shape.BoolCase<Nat.is_lt(1n+r, size)>) -> Split<1n+t, size>:
  match last:
    case Shape.BTrue{equal}:
      Split{q, 1n+r, Equal.sym(Bool, True{}, Nat.is_lt(1n+r, size), equal),
        Equal.trans(Nat, Nat.add(Nat.mul(q, size), 1n+r), 1n+Nat.add(Nat.mul(q, size), r), 1n+t,
          Arith.add_succ(Nat.mul(q, size), r), Equal.cong(Nat, Nat, n => 1n+n, Nat.add(Nat.mul(q, size), r), t, sum))}
    case Shape.BFalse{equal}:
      +whole = lt_succ_eq(r, size, below, equal)
      Split{1n+q, 0n,
        %Equal.sym(Nat, size, 1n+r, whole) : {Nat.is_lt(0n, _) == True{} : Bool}
        {==},
        %Equal.sym(Nat, Nat.add(Nat.mul(1n+q, size), 0n), Nat.mul(1n+q, size), Broadcast.add_zero(Nat.mul(1n+q, size))) : {_ == 1n+t : Nat}
        %Equal.sym(Nat, size, 1n+r, whole) : {Nat.add(_, Nat.mul(q, size)) == 1n+t : Nat}
        %Equal.sym(Nat, Nat.add(r, Nat.mul(q, size)), Nat.add(Nat.mul(q, size), r), add_comm(r, Nat.mul(q, size))) : {1n+_ == 1n+t : Nat}
        Equal.cong(Nat, Nat, n => 1n+n, Nat.add(Nat.mul(q, size), r), t, sum)}

def split_next(+t: Nat, +size: Nat, previous: Split<t, size>) -> Split<1n+t, size>:
  Split{+q, +r, below, sum} = previous
  split_step(t, size, q, r, below, sum, Shape.bool_case(Nat.is_lt(1n+r, size)))

def split(+t: Nat, +size: Nat, positive: {Nat.is_lt(0n, size) == True{} : Bool}) -> Split<t, size>:
  match t:
    case 0n:
      Split{0n, 0n, positive, {==}}
    case 1n+p:
      split_next(p, size, split(p, size, positive))

# The quotient and remainder of a split are Nat.div and Nat.mod.
def split_divmod(-t: Nat, +size: Nat, +q: Nat, +r: Nat, below: {Nat.is_lt(r, size) == True{} : Bool},
  sum: {Nat.add(Nat.mul(q, size), r) == t : Nat}) -> {(q, r) == Nat.divmod(t, size) : Nat & Nat}:
  %sum : {(q, r) == Nat.divmod(_, size) : Nat & Nat}
  Nats.divmod_exact(q, size, r, MatrixIndex.lt_from(r, size, below))

def div_of(-t: Nat, +size: Nat, +q: Nat, +r: Nat, below: {Nat.is_lt(r, size) == True{} : Bool},
  sum: {Nat.add(Nat.mul(q, size), r) == t : Nat}) -> {Nat.div(t, size) == q : Nat}:
  Equal.sym(Nat, q, Nat.div(t, size),
    Equal.cong(Nat & Nat, Nat, pair => Pair.fst(Nat, Nat, pair), (q, r), Nat.divmod(t, size), split_divmod(t, size, q, r, below, sum)))

def mod_of(-t: Nat, +size: Nat, +q: Nat, +r: Nat, below: {Nat.is_lt(r, size) == True{} : Bool},
  sum: {Nat.add(Nat.mul(q, size), r) == t : Nat}) -> {Nat.mod(t, size) == r : Nat}:
  Equal.sym(Nat, r, Nat.mod(t, size),
    Equal.cong(Nat & Nat, Nat, pair => Pair.snd(Nat, Nat, pair), (q, r), Nat.divmod(t, size), split_divmod(t, size, q, r, below, sum)))

# t = (t / size) * size + t % size, and t % size < size.
type Decomposed<-t: Nat, -size: Nat> is Data:
  Decomposed{below: {Nat.is_lt(Nat.mod(t, size), size) == True{} : Bool},
    sum: {Nat.add(Nat.mul(Nat.div(t, size), size), Nat.mod(t, size)) == t : Nat}}

def decomposed(+t: Nat, +size: Nat, given: Split<t, size>) -> Decomposed<t, size>:
  Split{+q, +r, +below, +sum} = given
  +dq = div_of(t, size, q, r, below, sum)
  +dr = mod_of(t, size, q, r, below, sum)
  Decomposed{
    %Equal.sym(Nat, Nat.mod(t, size), r, dr) : {Nat.is_lt(_, size) == True{} : Bool}
    below,
    %Equal.sym(Nat, Nat.div(t, size), q, dq) : {Nat.add(Nat.mul(_, size), Nat.mod(t, size)) == t : Nat}
    %Equal.sym(Nat, Nat.mod(t, size), r, dr) : {Nat.add(Nat.mul(q, size), _) == t : Nat}
    sum}

def decompose(+t: Nat, +size: Nat, positive: {Nat.is_lt(0n, size) == True{} : Bool}) -> Decomposed<t, size>:
  decomposed(t, size, split(t, size, positive))

def lt_irrefl(+x: Nat) -> {Nat.is_lt(x, x) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      lt_irrefl(p)

def le_add_cong(+c: Nat, +x: Nat, +y: Nat, h: {Nat.is_le(x, y) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(c, x), Nat.add(c, y)) == True{} : Bool}:
  match c:
    case 0n:
      h
    case 1n+p:
      le_add_cong(p, x, y, h)

def le_trans(+a: Nat, +b: Nat, +c: Nat, ab: {Nat.is_le(a, b) == True{} : Bool}, bc: {Nat.is_le(b, c) == True{} : Bool}) ->
  {Nat.is_le(a, c) == True{} : Bool}:
  match a b c:
    case 0n b c:
      Arith.le_zero(c)
    case 1n+p 0n c:
      Empty.absurd({Nat.is_le(1n+p, c) == True{} : Bool}, Arith.clash(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, Arith.clash(bc))
    case 1n+p 1n+q 1n+r:
      le_trans(p, q, r, ab, bc)

def mul_le_mono(+a: Nat, +q: Nat, +size: Nat, h: {Nat.is_le(a, q) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(a, size), Nat.mul(q, size)) == True{} : Bool}:
  match a q:
    case 0n q:
      Arith.le_zero(Nat.mul(q, size))
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(Nat.mul(1n+p, size), 0n) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+r:
      le_add_cong(size, Nat.mul(p, size), Nat.mul(r, size), mul_le_mono(p, r, size, h))

# Below a * size, the quotient by size is below a.
def quotient_below(+t: Nat, +a: Nat, +size: Nat, below: {Nat.is_lt(t, Nat.mul(a, size)) == True{} : Bool},
  decision: Shape.BoolCase<Nat.is_lt(Nat.div(t, size), a)>, given: Decomposed<t, size>) ->
  {Nat.is_lt(Nat.div(t, size), a) == True{} : Bool}:
  match decision:
    case Shape.BTrue{equal}:
      Equal.sym(Bool, True{}, Nat.is_lt(Nat.div(t, size), a), equal)
    case Shape.BFalse{equal}:
      Decomposed{rest, sum} = given
      +q = Nat.div(t, size)
      +r = Nat.mod(t, size)
      +grown = le_trans(Nat.mul(a, size), Nat.mul(q, size), t,
        mul_le_mono(a, q, size, Arith.not_lt_le(q, a, Equal.sym(Bool, False{}, Nat.is_lt(q, a), equal))),
        %sum : {Nat.is_le(Nat.mul(q, size), _) == True{} : Bool}
        Api.le_add_right(Nat.mul(q, size), r))
      Empty.absurd({Nat.is_lt(q, a) == True{} : Bool},
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_lt(t, t), True{}, Equal.sym(Bool, Nat.is_lt(t, t), False{}, lt_irrefl(t)),
          Arith.lt_le_trans(t, Nat.mul(a, size), t, below, grown))))

def positive_of(+t: Nat, +a: Nat, +size: Nat, below: {Nat.is_lt(t, Nat.mul(a, size)) == True{} : Bool}) ->
  {Nat.is_lt(0n, size) == True{} : Bool}:
  match size:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool},
        Arith.lt_zero(t,
          %Nats.mul_zero_right(a) : {Nat.is_lt(t, _) == True{} : Bool}
          below))
    case 1n+p:
      {==}

def div_below(+t: Nat, +a: Nat, +size: Nat, +below: {Nat.is_lt(t, Nat.mul(a, size)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(t, size), a) == True{} : Bool}:
  quotient_below(t, a, size, below, Shape.bool_case(Nat.is_lt(Nat.div(t, size), a)), decompose(t, size, positive_of(t, a, size, below)))

# a + x = a + y exactly when x = y.
def eq_add_cancel(+a: Nat, +x: Nat, +y: Nat) -> {Nat.is_eq(Nat.add(a, x), Nat.add(a, y)) == Nat.is_eq(x, y) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      eq_add_cancel(p, x, y)

def ne_from(+d: Nat, +e: Nat, same: {e == d : Nat}, differ: {Nat.is_eq(d, e) == False{} : Bool}) -> Empty:
  Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(d, e), True{}, Equal.sym(Bool, Nat.is_eq(d, e), False{}, differ),
    %Equal.sym(Nat, e, d, same) : {Nat.is_eq(d, _) == True{} : Bool}
    Arith.eq_refl(d)))

# On a kept row of left stride `size`, the position with digit d and
# walk w below size lands on t exactly when d is t's quotient and w its
# remainder.
def mask_at(+size: Nat, +d: Nat, +w: Nat, +t: Nat, given: Decomposed<t, size>, same: {d == Nat.div(t, size) : Nat}) ->
  {Nat.is_eq(Nat.add(Nat.mul(size, d), w), t) == Nat.is_eq(w, Nat.mod(t, size)) : Bool}:
  Decomposed{below, sum} = given
  %sum :
    {Nat.is_eq(Nat.add(Nat.mul(size, d), w), _) == Nat.is_eq(w, Nat.mod(t, size)) : Bool}
  %same : {Nat.is_eq(Nat.add(Nat.mul(size, d), w), Nat.add(Nat.mul(_, size), Nat.mod(t, size))) == Nat.is_eq(w, Nat.mod(t, size)) : Bool}
  %Nats.mul_comm(size, d) : {Nat.is_eq(Nat.add(Nat.mul(size, d), w), Nat.add(_, Nat.mod(t, size))) == Nat.is_eq(w, Nat.mod(t, size)) : Bool}
  eq_add_cancel(Nat.mul(size, d), w, Nat.mod(t, size))

def mask_off_decided(+size: Nat, +d: Nat, +w: Nat, +t: Nat, below: {Nat.is_lt(w, size) == True{} : Bool},
  differ: {Nat.is_eq(d, Nat.div(t, size)) == False{} : Bool},
  decision: Shape.BoolCase<Nat.is_eq(Nat.add(Nat.mul(size, d), w), t)>) ->
  {Nat.is_eq(Nat.add(Nat.mul(size, d), w), t) == False{} : Bool}:
  match decision:
    case Shape.BFalse{equal}:
      Equal.sym(Bool, False{}, Nat.is_eq(Nat.add(Nat.mul(size, d), w), t), equal)
    case Shape.BTrue{equal}:
      +hit = Arith.eq_true(Nat.add(Nat.mul(size, d), w), t, Equal.sym(Bool, True{}, Nat.is_eq(Nat.add(Nat.mul(size, d), w), t), equal))
      Empty.absurd({Nat.is_eq(Nat.add(Nat.mul(size, d), w), t) == False{} : Bool},
        ne_from(d, Nat.div(t, size),
          div_of(t, size, d, w, below,
            %Nats.mul_comm(size, d) : {Nat.add(_, w) == t : Nat}
            hit),
          differ))

def mask_off(+size: Nat, +d: Nat, +w: Nat, +t: Nat, below: {Nat.is_lt(w, size) == True{} : Bool},
  differ: {Nat.is_eq(d, Nat.div(t, size)) == False{} : Bool}) ->
  {Nat.is_eq(Nat.add(Nat.mul(size, d), w), t) == False{} : Bool}:
  mask_off_decided(size, d, w, t, below, differ, Shape.bool_case(Nat.is_eq(Nat.add(Nat.mul(size, d), w), t)))

# The masked positions and the walk.

def kept(+keep: Bool, +position: Nat, rest: List<&2, Nat>) -> List<&2, Nat>:
  match keep:
    case True{}:
      position <> rest
    case False{}:
      rest

# The contribution offsets base + walk_right(p) of the positions
# p = cursor .. cursor + fuel - 1 whose walk_left is `target`, in order:
# the words the masked fold of `ReduceLaws.fiber` adds.
def masked(fuel: Nat, +rows: List<&2, Shape.Row>, +target: Nat, +base: Nat, +cursor: Nat) -> List<&2, Nat>:
  match fuel:
    case 0n:
      Nil{}
    case 1n+rest:
      kept(Nat.is_eq(Shape.walk_left(rows, cursor), target), Nat.add(base, Shape.walk_right(rows, cursor)),
        masked(rest, rows, target, base, 1n+cursor))

# The contribution offsets `Reduce.fiber` reads, in order.
def visited(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, fuel: Nat, +target: Nat, +base: Nat, +digit: Nat) -> List<&2, Nat>:
  match rows extents fuel:
    case Nil{} _ _:
      [base]
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n:
      Nil{}
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining:
      +inner = Nat.mod(target, left)
      List.append(&2, Nat,
        visited(rest, others, Reduce.first_span(rest, others), inner, Nat.add(base, Nat.mul(right, digit)), Reduce.first_digit(rest, inner)),
        visited(rows, extents, remaining, target, base, 1n+digit))
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _:
      Nil{}

def append_nil(+xs: List<&2, Nat>) -> {List.append(&2, Nat, xs, Nil{}) == xs : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, tail => x <> tail, List.append(&2, Nat, rest, Nil{}), rest, append_nil(rest))

def kept_append(+keep: Bool, +position: Nat, -xs: List<&2, Nat>, -ys: List<&2, Nat>) ->
  {kept(keep, position, List.append(&2, Nat, xs, ys)) == List.append(&2, Nat, kept(keep, position, xs), ys) : List<&2, Nat>}:
  match keep:
    case True{}:
      {==}
    case False{}:
      {==}

def masked_split(+a: Nat, +b: Nat, +rows: List<&2, Shape.Row>, +target: Nat, +base: Nat, +cursor: Nat) ->
  {masked(Nat.add(a, b), rows, target, base, cursor) ==
    List.append(&2, Nat, masked(a, rows, target, base, cursor), masked(b, rows, target, base, Nat.add(cursor, a))) : List<&2, Nat>}:
  match a:
    case 0n:
      %Equal.sym(Nat, Nat.add(cursor, 0n), cursor, Broadcast.add_zero(cursor)) :
        {masked(b, rows, target, base, cursor) == masked(b, rows, target, base, _) : List<&2, Nat>}
      {==}
    case 1n+p:
      +keep = Nat.is_eq(Shape.walk_left(rows, cursor), target)
      +position = Nat.add(base, Shape.walk_right(rows, cursor))
      +front = masked(p, rows, target, base, 1n+cursor)
      %Equal.sym(Nat, Nat.add(cursor, 1n+p), 1n+Nat.add(cursor, p), Arith.add_succ(cursor, p)) :
        {masked(1n+Nat.add(p, b), rows, target, base, cursor) ==
          List.append(&2, Nat, kept(keep, position, front), masked(b, rows, target, base, _)) : List<&2, Nat>}
      Equal.trans(List<&2, Nat>, kept(keep, position, masked(Nat.add(p, b), rows, target, base, 1n+cursor)),
        kept(keep, position, List.append(&2, Nat, front, masked(b, rows, target, base, Nat.add(1n+cursor, p)))),
        List.append(&2, Nat, kept(keep, position, front), masked(b, rows, target, base, Nat.add(1n+cursor, p))),
        Equal.cong(List<&2, Nat>, List<&2, Nat>, xs => kept(keep, position, xs),
          masked(Nat.add(p, b), rows, target, base, 1n+cursor),
          List.append(&2, Nat, front, masked(b, rows, target, base, Nat.add(1n+cursor, p))),
          masked_split(p, b, rows, target, base, 1n+cursor)),
        kept_append(keep, position, front, masked(b, rows, target, base, Nat.add(1n+cursor, p))))

def le_step(+q: Nat, +m: Nat, +s: Nat, room: {Nat.is_le(Nat.add(q, 1n+m), s) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+q, m), s) == True{} : Bool}:
  %Arith.add_succ(q, m) : {Nat.is_le(_, s) == True{} : Bool}
  room

def lt_of_room(+q: Nat, +m: Nat, +s: Nat, room: {Nat.is_le(Nat.add(q, 1n+m), s) == True{} : Bool}) ->
  {Nat.is_lt(q, s) == True{} : Bool}:
  Arith.lt_le_trans(q, Nat.add(q, 1n+m), s,
    %Equal.sym(Nat, Nat.add(q, 1n+m), 1n+Nat.add(q, m), Arith.add_succ(q, m)) : {Nat.is_lt(q, _) == True{} : Bool}
    Arith.lt_succ(q, Nat.add(q, m), Api.le_add_right(q, m)),
    room)

def lt_succ_right(+b: Nat, +k: Nat, h: {Nat.is_lt(b, k) == True{} : Bool}) -> {Nat.is_lt(b, 1n+k) == True{} : Bool}:
  match b k:
    case 0n k:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_lt(1n+p, 1n) == True{} : Bool}, Arith.clash(h))
    case 1n+p 1n+q:
      lt_succ_right(p, q, h)

def gt_not_eq(+a: Nat, +b: Nat, h: {Nat.is_lt(b, a) == True{} : Bool}) -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n b:
      Empty.absurd({Nat.is_eq(0n, b) == False{} : Bool}, Arith.lt_zero(b, h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      gt_not_eq(p, q, h)

def le_ne_succ(+a: Nat, +b: Nat, le: {Nat.is_le(a, b) == True{} : Bool}, ne: {Nat.is_eq(a, b) == False{} : Bool}) ->
  {Nat.is_le(1n+a, b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_le(1n, 0n) == True{} : Bool}, Arith.clash(Equal.sym(Bool, Nat.is_eq(0n, 0n), False{}, ne)))
    case 0n 1n+q:
      Arith.le_zero(q)
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(2n+p, 0n) == True{} : Bool}, Arith.clash(le))
    case 1n+p 1n+q:
      le_ne_succ(p, q, le, ne)

def le_lt_absurd(+a: Nat, +b: Nat, le: {Nat.is_le(a, b) == True{} : Bool}, lt: {Nat.is_lt(b, a) == True{} : Bool}) -> Empty:
  match a b:
    case 0n b:
      Arith.lt_zero(b, lt)
    case 1n+p 0n:
      Arith.clash(le)
    case 1n+p 1n+q:
      le_lt_absurd(p, q, le, lt)

def lt_self_succ(+k: Nat) -> {Nat.is_lt(k, 1n+k) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n+p:
      lt_self_succ(p)

def axis_zero(+rh: Nat, +oh: Nat, axis: Shape.Axis<0n, rh, oh>) -> {oh == 0n : Nat}:
  match axis:
    case Shape.AxisEqual{left_eq, right_eq}:
      left_eq
    case Shape.AxisLeftOne{one, right_eq}:
      Empty.absurd({oh == 0n : Nat}, Arith.clash(Equal.cong(Nat, Bool, n => Nat.is_eq(n, 0n), 1n, 0n, one)))
    case Shape.AxisRightOne{left_eq, one}:
      left_eq

def axis_kept(+j: Nat, +rh: Nat, +oh: Nat, axis: Shape.Axis<2n+j, rh, oh>) -> {oh == 2n+j : Nat}:
  match axis:
    case Shape.AxisEqual{left_eq, right_eq}:
      left_eq
    case Shape.AxisLeftOne{one, right_eq}:
      Empty.absurd({oh == 2n+j : Nat}, Arith.clash(Equal.cong(Nat, Bool, n => Nat.is_eq(n, 1n), 2n+j, 1n, Equal.sym(Nat, 1n, 2n+j, one))))
    case Shape.AxisRightOne{left_eq, one}:
      left_eq

def walk_left_row(+lh: Nat, +rh: Nat, +oh: Nat, axis: Shape.Axis<lh, rh, oh>, +size: Nat, +d: Nat, +w: Nat,
  d_below: {Nat.is_lt(d, oh) == True{} : Bool}, w_below: {Nat.is_lt(w, size) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Shape.broadcast_index(lh, size), d), w), Nat.mul(lh, size)) == True{} : Bool}:
  match lh:
    case 0n:
      Empty.absurd({Nat.is_lt(Nat.add(Nat.mul(size, d), w), 0n) == True{} : Bool},
        Arith.lt_zero(d,
          %axis_zero(rh, oh, axis) : {Nat.is_lt(d, _) == True{} : Bool}
          d_below))
    case 1n+k:
      match k:
        case 0n:
          %Equal.sym(Nat, Nat.add(size, 0n), size, Broadcast.add_zero(size)) : {Nat.is_lt(w, _) == True{} : Bool}
          w_below
        case 1n+j:
          %Nats.mul_comm(d, size) : {Nat.is_lt(Nat.add(_, w), Nat.mul(2n+j, size)) == True{} : Bool}
          MatrixIndex.lt_mul_add(d, 2n+j, w, size,
            %axis_kept(j, rh, oh, axis) : {Nat.is_lt(d, _) == True{} : Bool}
            d_below,
            w_below)

# The left walk of a position below the output count lies below the
# left count.
def walk_left_below(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +aligned: Shape.Aligned(left, right, output),
  +q: Nat, +below: {Nat.is_lt(q, Shape.numel(output)) == True{} : Bool}) ->
  {Nat.is_lt(Shape.walk_left(Shape.index_rows(left, right, output), q), Shape.numel(left)) == True{} : Bool}:
  match left right output:
    case Nil{} Nil{} Nil{}:
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{+axis, +tail} = aligned
      +s = Shape.numel(ot)
      walk_left_row(lh, rh, oh, axis, Shape.numel(lt), Nat.div(q, s), Shape.walk_left(Shape.index_rows(lt, rt, ot), Nat.mod(q, s)),
        div_below(q, oh, s, below),
        walk_left_below(lt, rt, ot, tail, Nat.mod(q, s),
          {\{Decomposed: m_below => sum => m_below} : Decomposed<q, s> -> {Nat.is_lt(Nat.mod(q, s), s) == True{} : Bool}}(
            decompose(q, s, positive_of(q, oh, s, below)))))
    case Nil{} Nil{} +oh <> +ot:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(Nil{}, Nil{}, oh <> ot), q), 1n) == True{} : Bool}, aligned)
    case Nil{} +rh <> +rt Nil{}:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(Nil{}, rh <> rt, Nil{}), q), 1n) == True{} : Bool}, aligned)
    case +lh <> +lt Nil{} Nil{}:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(lh <> lt, Nil{}, Nil{}), q), Shape.numel(lh <> lt)) == True{} : Bool}, aligned)
    case Nil{} +rh <> +rt +oh <> +ot:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(Nil{}, rh <> rt, oh <> ot), q), 1n) == True{} : Bool}, aligned)
    case +lh <> +lt Nil{} +oh <> +ot:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(lh <> lt, Nil{}, oh <> ot), q), Shape.numel(lh <> lt)) == True{} : Bool}, aligned)
    case +lh <> +lt +rh <> +rt Nil{}:
      Empty.absurd({Nat.is_lt(Shape.walk_left(Shape.index_rows(lh <> lt, rh <> rt, Nil{}), q), Shape.numel(lh <> lt)) == True{} : Bool}, aligned)

# Below a kept row of left stride `left`, every walk of the rows below
# stays under `left`: the rows are the index rows of an aligned broadcast
# whose left count is `left` and whose output count is `stride`.
type BoundBy<-rest: List<&2, Shape.Row>, -left: Nat, -stride: Nat> is Data:
  BoundBy{lt: List<&2, Nat>, rt: List<&2, Nat>, ot: List<&2, Nat>, aligned: Shape.Aligned(lt, rt, ot),
    rows: {Shape.index_rows(lt, rt, ot) == rest : List<&2, Shape.Row>},
    size: {Shape.numel(lt) == left : Nat}, count: {Shape.numel(ot) == stride : Nat}}

def bound_at(-rest: List<&2, Shape.Row>, -left: Nat, -stride: Nat, +by: BoundBy<rest, left, stride>, +q: Nat,
  +below: {Nat.is_lt(q, stride) == True{} : Bool}) -> {Nat.is_lt(Shape.walk_left(rest, q), left) == True{} : Bool}:
  BoundBy{+lt, +rt, +ot, +aligned, +rows, +size, +count} = by
  %rows : {Nat.is_lt(Shape.walk_left(_, q), left) == True{} : Bool}
  %size : {Nat.is_lt(Shape.walk_left(Shape.index_rows(lt, rt, ot), q), _) == True{} : Bool}
  walk_left_below(lt, rt, ot, aligned, q,
    %Equal.sym(Nat, Shape.numel(ot), stride, count) : {Nat.is_lt(q, _) == True{} : Bool}
    below)

# How the rows below a block decide which positions land on the target: on
# a reduced row (left stride 0) the rows below land on the target itself; on
# the kept row's target digit they land on the target's remainder; on any
# other digit of a kept row nothing lands.
type MaskMode<-left: Nat, -digit: Nat, -target: Nat, -inner: Nat> is Data:
  Free{zero: {0n == left : Nat}, same: {target == inner : Nat}}
  Hit{given: Decomposed<target, left>, same: {digit == Nat.div(target, left) : Nat}, remainder: {Nat.mod(target, left) == inner : Nat}}

def mode_mask(+left: Nat, +digit: Nat, +target: Nat, +inner: Nat, +mode: MaskMode<left, digit, target, inner>, +w: Nat) ->
  {Nat.is_eq(w, inner) == Nat.is_eq(Nat.add(Nat.mul(left, digit), w), target) : Bool}:
  match mode:
    case Free{zero, same}:
      %zero : {Nat.is_eq(w, inner) == Nat.is_eq(Nat.add(Nat.mul(_, digit), w), target) : Bool}
      %Equal.sym(Nat, target, inner, same) : {Nat.is_eq(w, inner) == Nat.is_eq(w, _) : Bool}
      {==}
    case Hit{given, same, remainder}:
      %remainder : {Nat.is_eq(w, _) == Nat.is_eq(Nat.add(Nat.mul(left, digit), w), target) : Bool}
      Equal.sym(Bool, Nat.is_eq(Nat.add(Nat.mul(left, digit), w), target), Nat.is_eq(w, Nat.mod(target, left)),
        mask_at(left, digit, w, target, given, same))

type OffMode<-rest: List<&2, Shape.Row>, -left: Nat, -stride: Nat, -digit: Nat, -target: Nat> is Data:
  Off{bound: BoundBy<rest, left, stride>, differ: {Nat.is_eq(digit, Nat.div(target, left)) == False{} : Bool}}

def off_mask(+rest: List<&2, Shape.Row>, +left: Nat, +stride: Nat, +digit: Nat, +target: Nat, +mode: OffMode<rest, left, stride, digit, target>,
  +q: Nat, +below: {Nat.is_lt(q, stride) == True{} : Bool}) ->
  {False{} == Nat.is_eq(Nat.add(Nat.mul(left, digit), Shape.walk_left(rest, q)), target) : Bool}:
  Off{+bound, +differ} = mode
  Equal.sym(Bool, Nat.is_eq(Nat.add(Nat.mul(left, digit), Shape.walk_left(rest, q)), target), False{},
    mask_off(left, digit, Shape.walk_left(rest, q), target, bound_at(rest, left, stride, bound, q, below), differ))

# On a row of stride s, left stride L and right stride R, the positions
# d * s + q .. d * s + q + n - 1 have digit d and walk the rows below at
# q .. q + n - 1.
def shifted(+n: Nat, +stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +inner: Nat, +base: Nat,
  +digit: Nat, +q: Nat, +room: {Nat.is_le(Nat.add(q, n), stride) == True{} : Bool}, +mode: MaskMode<left, digit, target, inner>) ->
  {masked(n, Shape.Row{stride, left, right} <> rest, target, base, Nat.add(Nat.mul(digit, stride), q)) ==
    masked(n, rest, inner, Nat.add(base, Nat.mul(right, digit)), q) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +c = Nat.add(Nat.mul(digit, stride), q)
      +below = lt_of_room(q, m, stride, room)
      +dq = div_of(c, stride, digit, q, below, {==})
      +mq = mod_of(c, stride, digit, q, below, {==})
      +wl = Shape.walk_left(rest, q)
      +wr = Shape.walk_right(rest, q)
      +tail = masked(m, rows, target, base, 1n+c)
      +goal_right = kept(Nat.is_eq(wl, inner), Nat.add(Nat.add(base, Nat.mul(right, digit)), wr),
        masked(m, rest, inner, Nat.add(base, Nat.mul(right, digit)), 1n+q))
      %Equal.sym(Nat, Nat.div(c, stride), digit, dq) :
        {kept(Nat.is_eq(Nat.add(Nat.mul(left, _), Shape.walk_left(rest, Nat.mod(c, stride))), target),
          Nat.add(base, Nat.add(Nat.mul(right, _), Shape.walk_right(rest, Nat.mod(c, stride)))), tail) == goal_right : List<&2, Nat>}
      %Equal.sym(Nat, Nat.mod(c, stride), q, mq) :
        {kept(Nat.is_eq(Nat.add(Nat.mul(left, digit), Shape.walk_left(rest, _)), target),
          Nat.add(base, Nat.add(Nat.mul(right, digit), Shape.walk_right(rest, _))), tail) == goal_right : List<&2, Nat>}
      %mode_mask(left, digit, target, inner, mode, wl) :
        {kept(_, Nat.add(base, Nat.add(Nat.mul(right, digit), wr)), tail) == goal_right : List<&2, Nat>}
      %Equal.sym(Nat, Nat.add(base, Nat.add(Nat.mul(right, digit), wr)), Nat.add(Nat.add(base, Nat.mul(right, digit)), wr),
        Index.add_assoc(base, Nat.mul(right, digit), wr)) :
        {kept(Nat.is_eq(wl, inner), _, tail) == goal_right : List<&2, Nat>}
      %Arith.add_succ(Nat.mul(digit, stride), q) :
        {kept(Nat.is_eq(wl, inner), Nat.add(Nat.add(base, Nat.mul(right, digit)), wr), masked(m, rows, target, base, _)) == goal_right : List<&2, Nat>}
      Equal.cong(List<&2, Nat>, List<&2, Nat>, xs => kept(Nat.is_eq(wl, inner), Nat.add(Nat.add(base, Nat.mul(right, digit)), wr), xs),
        masked(m, rows, target, base, Nat.add(Nat.mul(digit, stride), 1n+q)),
        masked(m, rest, inner, Nat.add(base, Nat.mul(right, digit)), 1n+q),
        shifted(m, stride, left, right, rest, target, inner, base, digit, 1n+q, le_step(q, m, stride, room), mode))

def shifted_off(+n: Nat, +stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +base: Nat,
  +digit: Nat, +q: Nat, +room: {Nat.is_le(Nat.add(q, n), stride) == True{} : Bool}, +mode: OffMode<rest, left, stride, digit, target>) ->
  {masked(n, Shape.Row{stride, left, right} <> rest, target, base, Nat.add(Nat.mul(digit, stride), q)) == Nil{} : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +c = Nat.add(Nat.mul(digit, stride), q)
      +below = lt_of_room(q, m, stride, room)
      +dq = div_of(c, stride, digit, q, below, {==})
      +mq = mod_of(c, stride, digit, q, below, {==})
      +tail = masked(m, rows, target, base, 1n+c)
      +position = Nat.add(base, Shape.walk_right(rows, c))
      %Equal.sym(Nat, Nat.div(c, stride), digit, dq) :
        {kept(Nat.is_eq(Nat.add(Nat.mul(left, _), Shape.walk_left(rest, Nat.mod(c, stride))), target), position, tail) == Nil{} : List<&2, Nat>}
      %Equal.sym(Nat, Nat.mod(c, stride), q, mq) :
        {kept(Nat.is_eq(Nat.add(Nat.mul(left, digit), Shape.walk_left(rest, _)), target), position, tail) == Nil{} : List<&2, Nat>}
      %off_mask(rest, left, stride, digit, target, mode, q, below) :
        {kept(_, position, tail) == Nil{} : List<&2, Nat>}
      %Arith.add_succ(Nat.mul(digit, stride), q) : {masked(m, rows, target, base, _) == Nil{} : List<&2, Nat>}
      shifted_off(m, stride, left, right, rest, target, base, digit, 1n+q, le_step(q, m, stride, room), mode)

# The first block of a row, at cursor digit * stride: the rows below at
# base + right * digit.
def block(+stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +inner: Nat, +base: Nat, +digit: Nat,
  +mode: MaskMode<left, digit, target, inner>) ->
  {masked(stride, Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) ==
    masked(stride, rest, inner, Nat.add(base, Nat.mul(right, digit)), 0n) : List<&2, Nat>}:
  %Broadcast.add_zero(Nat.mul(digit, stride)) :
    {masked(stride, Shape.Row{stride, left, right} <> rest, target, base, _) ==
      masked(stride, rest, inner, Nat.add(base, Nat.mul(right, digit)), 0n) : List<&2, Nat>}
  shifted(stride, stride, left, right, rest, target, inner, base, digit, 0n, Api.le_refl(stride), mode)

def block_off(+stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +base: Nat, +digit: Nat,
  +mode: OffMode<rest, left, stride, digit, target>) ->
  {masked(stride, Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) == Nil{} : List<&2, Nat>}:
  %Broadcast.add_zero(Nat.mul(digit, stride)) :
    {masked(stride, Shape.Row{stride, left, right} <> rest, target, base, _) == Nil{} : List<&2, Nat>}
  shifted_off(stride, stride, left, right, rest, target, base, digit, 0n, Api.le_refl(stride), mode)

def next_cursor(+digit: Nat, +stride: Nat) -> {Nat.mul(1n+digit, stride) == Nat.add(Nat.mul(digit, stride), stride) : Nat}:
  add_comm(stride, Nat.mul(digit, stride))

# Moving the base moves every offset: base + xs.
def shift(+base: Nat, xs: List<&2, Nat>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case x <> rest:
      Nat.add(base, x) <> shift(base, rest)

def shift_append(+base: Nat, +xs: List<&2, Nat>, +ys: List<&2, Nat>) ->
  {shift(base, List.append(&2, Nat, xs, ys)) == List.append(&2, Nat, shift(base, xs), shift(base, ys)) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, tail => Nat.add(base, x) <> tail,
        shift(base, List.append(&2, Nat, rest, ys)), List.append(&2, Nat, shift(base, rest), shift(base, ys)), shift_append(base, rest, ys))

def shift_shift(+a: Nat, +b: Nat, +xs: List<&2, Nat>) -> {shift(a, shift(b, xs)) == shift(Nat.add(a, b), xs) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +rest:
      %Equal.sym(Nat, Nat.add(a, Nat.add(b, x)), Nat.add(Nat.add(a, b), x), Index.add_assoc(a, b, x)) : {_ <> shift(a, shift(b, rest)) == Nat.add(Nat.add(a, b), x) <> shift(Nat.add(a, b), rest) : List<&2, Nat>}
      Equal.cong(List<&2, Nat>, List<&2, Nat>, tail => Nat.add(Nat.add(a, b), x) <> tail,
        shift(a, shift(b, rest)), shift(Nat.add(a, b), rest), shift_shift(a, b, rest))

def kept_shift(+keep: Bool, +base: Nat, +x: Nat, +rest: List<&2, Nat>) ->
  {kept(keep, Nat.add(base, x), shift(base, rest)) == shift(base, kept(keep, x, rest)) : List<&2, Nat>}:
  match keep:
    case True{}:
      {==}
    case False{}:
      {==}

def masked_base(+n: Nat, +rows: List<&2, Shape.Row>, +target: Nat, +base: Nat, +cursor: Nat) ->
  {masked(n, rows, target, base, cursor) == shift(base, masked(n, rows, target, 0n, cursor)) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +keep = Nat.is_eq(Shape.walk_left(rows, cursor), target)
      +x = Shape.walk_right(rows, cursor)
      Equal.trans(List<&2, Nat>, kept(keep, Nat.add(base, x), masked(m, rows, target, base, 1n+cursor)),
        kept(keep, Nat.add(base, x), shift(base, masked(m, rows, target, 0n, 1n+cursor))),
        shift(base, kept(keep, x, masked(m, rows, target, 0n, 1n+cursor))),
        Equal.cong(List<&2, Nat>, List<&2, Nat>, tail => kept(keep, Nat.add(base, x), tail),
          masked(m, rows, target, base, 1n+cursor), shift(base, masked(m, rows, target, 0n, 1n+cursor)), masked_base(m, rows, target, base, 1n+cursor)),
        kept_shift(keep, base, x, masked(m, rows, target, 0n, 1n+cursor)))

def visited_base(+rows: List<&2, Shape.Row>, +extents: List<&2, Nat>, +fuel: Nat, +target: Nat, +base: Nat, +digit: Nat) ->
  {visited(rows, extents, fuel, target, base, digit) == shift(base, visited(rows, extents, fuel, target, 0n, digit)) : List<&2, Nat>}:
  match rows extents fuel:
    case Nil{} _ _:
      %Equal.sym(Nat, Nat.add(base, 0n), base, Broadcast.add_zero(base)) : {[base] == [_] : List<&2, Nat>}
      {==}
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 0n:
      {==}
    case Shape.Row{+stride, +left, +right} <> +rest +extent <> +others 1n+remaining:
      +inner = Nat.mod(target, left)
      +offset = Nat.mul(right, digit)
      +span = Reduce.first_span(rest, others)
      +first = Reduce.first_digit(rest, inner)
      +low = visited(rest, others, span, inner, 0n, first)
      +high = visited(rows, extents, remaining, target, 0n, 1n+digit)
      %Equal.sym(List<&2, Nat>, visited(rest, others, span, inner, Nat.add(base, offset), first), shift(Nat.add(base, offset), low),
        visited_base(rest, others, span, inner, Nat.add(base, offset), first)) :
        {List.append(&2, Nat, _, visited(rows, extents, remaining, target, base, 1n+digit)) ==
          shift(base, List.append(&2, Nat, visited(rest, others, span, inner, offset, first), high)) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, visited(rows, extents, remaining, target, base, 1n+digit), shift(base, high),
        visited_base(rows, extents, remaining, target, base, 1n+digit)) :
        {List.append(&2, Nat, shift(Nat.add(base, offset), low), _) ==
          shift(base, List.append(&2, Nat, visited(rest, others, span, inner, offset, first), high)) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, visited(rest, others, span, inner, offset, first), shift(offset, low),
        visited_base(rest, others, span, inner, offset, first)) :
        {List.append(&2, Nat, shift(Nat.add(base, offset), low), shift(base, high)) ==
          shift(base, List.append(&2, Nat, _, high)) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, shift(base, List.append(&2, Nat, shift(offset, low), high)),
        List.append(&2, Nat, shift(base, shift(offset, low)), shift(base, high)), shift_append(base, shift(offset, low), high)) :
        {List.append(&2, Nat, shift(Nat.add(base, offset), low), shift(base, high)) == _ : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, shift(base, shift(offset, low)), shift(Nat.add(base, offset), low), shift_shift(base, offset, low)) :
        {List.append(&2, Nat, shift(Nat.add(base, offset), low), shift(base, high)) == List.append(&2, Nat, _, shift(base, high)) : List<&2, Nat>}
      {==}
    case Shape.Row{+stride, +left, +right} <> +rest Nil{} _:
      {==}

# The walk below a row, at base 0, visits the masked positions below it.
def RestWalk(+rest: List<&2, Shape.Row>, +others: List<&2, Nat>, +inner: Nat, +stride: Nat) -> Data:
  {masked(stride, rest, inner, 0n, 0n) ==
    visited(rest, others, Reduce.first_span(rest, others), inner, 0n, Reduce.first_digit(rest, inner)) : List<&2, Nat>}

# ... and so at every base.
def rest_at(+rest: List<&2, Shape.Row>, +others: List<&2, Nat>, +inner: Nat, +stride: Nat, +walk: RestWalk(rest, others, inner, stride),
  +base: Nat) ->
  {masked(stride, rest, inner, base, 0n) ==
    visited(rest, others, Reduce.first_span(rest, others), inner, base, Reduce.first_digit(rest, inner)) : List<&2, Nat>}:
  %Equal.sym(List<&2, Nat>, masked(stride, rest, inner, base, 0n), shift(base, masked(stride, rest, inner, 0n, 0n)),
    masked_base(stride, rest, inner, base, 0n)) :
    {_ == visited(rest, others, Reduce.first_span(rest, others), inner, base, Reduce.first_digit(rest, inner)) : List<&2, Nat>}
  %Equal.sym(List<&2, Nat>, visited(rest, others, Reduce.first_span(rest, others), inner, base, Reduce.first_digit(rest, inner)),
    shift(base, visited(rest, others, Reduce.first_span(rest, others), inner, 0n, Reduce.first_digit(rest, inner))),
    visited_base(rest, others, Reduce.first_span(rest, others), inner, base, Reduce.first_digit(rest, inner))) :
    {shift(base, masked(stride, rest, inner, 0n, 0n)) == _ : List<&2, Nat>}
  Equal.cong(List<&2, Nat>, List<&2, Nat>, xs => shift(base, xs), masked(stride, rest, inner, 0n, 0n),
    visited(rest, others, Reduce.first_span(rest, others), inner, 0n, Reduce.first_digit(rest, inner)), walk)

# A reduced row (left stride 0) walks every digit: the blocks digit ..
# digit + f - 1 of the masked positions are what the walk visits.
def reduced_blocks(+f: Nat, +stride: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +extent: Nat, +others: List<&2, Nat>,
  +target: Nat, +base: Nat, +digit: Nat, +walk: RestWalk(rest, others, target, stride)) ->
  {masked(Nat.mul(f, stride), Shape.Row{stride, 0n, right} <> rest, target, base, Nat.mul(digit, stride)) ==
    visited(Shape.Row{stride, 0n, right} <> rest, extent <> others, f, target, base, digit) : List<&2, Nat>}:
  match f:
    case 0n:
      {==}
    case 1n+g:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, 0n, right} <> rest
      +c = Nat.mul(digit, stride)
      +w = visited(rest, others, Reduce.first_span(rest, others), target, Nat.add(base, Nat.mul(right, digit)), Reduce.first_digit(rest, target))
      +v = visited(rows, extent <> others, g, target, base, 1n+digit)
      %Equal.sym(List<&2, Nat>, masked(Nat.add(stride, Nat.mul(g, stride)), rows, target, base, c),
        List.append(&2, Nat, masked(stride, rows, target, base, c), masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))),
        masked_split(stride, Nat.mul(g, stride), rows, target, base, c)) :
        {_ == List.append(&2, Nat, w, v) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rows, target, base, c), masked(stride, rest, target, Nat.add(base, Nat.mul(right, digit)), 0n),
        block(stride, 0n, right, rest, target, target, base, digit, Free{{==}, {==}})) :
        {List.append(&2, Nat, _, masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))) == List.append(&2, Nat, w, v) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rest, target, Nat.add(base, Nat.mul(right, digit)), 0n), w,
        rest_at(rest, others, target, stride, walk, Nat.add(base, Nat.mul(right, digit)))) :
        {List.append(&2, Nat, _, masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))) == List.append(&2, Nat, w, v) : List<&2, Nat>}
      %next_cursor(digit, stride) :
        {List.append(&2, Nat, w, masked(Nat.mul(g, stride), rows, target, base, _)) == List.append(&2, Nat, w, v) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(Nat.mul(g, stride), rows, target, base, Nat.mul(1n+digit, stride)), v,
        reduced_blocks(g, stride, right, rest, extent, others, target, base, 1n+digit, walk)) :
        {List.append(&2, Nat, w, _) == List.append(&2, Nat, w, v) : List<&2, Nat>}
      {==}

# On a kept row, the blocks after the target's digit hold no position.
def kept_after(+f: Nat, +stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +base: Nat, +digit: Nat,
  +after: {Nat.is_lt(Nat.div(target, left), digit) == True{} : Bool}, +bound: BoundBy<rest, left, stride>) ->
  {masked(Nat.mul(f, stride), Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) == Nil{} : List<&2, Nat>}:
  match f:
    case 0n:
      {==}
    case 1n+g:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +c = Nat.mul(digit, stride)
      %Equal.sym(List<&2, Nat>, masked(Nat.add(stride, Nat.mul(g, stride)), rows, target, base, c),
        List.append(&2, Nat, masked(stride, rows, target, base, c), masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))),
        masked_split(stride, Nat.mul(g, stride), rows, target, base, c)) :
        {_ == Nil{} : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rows, target, base, c), Nil{},
        block_off(stride, left, right, rest, target, base, digit, Off{bound, gt_not_eq(digit, Nat.div(target, left), after)})) :
        {List.append(&2, Nat, _, masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))) == Nil{} : List<&2, Nat>}
      %next_cursor(digit, stride) :
        {masked(Nat.mul(g, stride), rows, target, base, _) == Nil{} : List<&2, Nat>}
      kept_after(g, stride, left, right, rest, target, base, 1n+digit, lt_succ_right(Nat.div(target, left), digit, after), bound)

def kept_hit_step(+g: Nat, +stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +base: Nat, +digit: Nat,
  decision: Shape.BoolCase<Nat.is_eq(digit, Nat.div(target, left))>,
  +from: {Nat.is_le(digit, Nat.div(target, left)) == True{} : Bool},
  +until: {Nat.is_lt(Nat.div(target, left), Nat.add(digit, 1n+g)) == True{} : Bool},
  +given: Decomposed<target, left>, +bound: BoundBy<rest, left, stride>,
  rec: @+next: Nat -> @+from: {Nat.is_le(next, Nat.div(target, left)) == True{} : Bool} ->
    @+until: {Nat.is_lt(Nat.div(target, left), Nat.add(next, g)) == True{} : Bool} ->
    {masked(Nat.mul(g, stride), Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(next, stride)) ==
      masked(stride, rest, Nat.mod(target, left), Nat.add(base, Nat.mul(right, Nat.div(target, left))), 0n) : List<&2, Nat>}) ->
  {masked(Nat.mul(1n+g, stride), Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) ==
    masked(stride, rest, Nat.mod(target, left), Nat.add(base, Nat.mul(right, Nat.div(target, left))), 0n) : List<&2, Nat>}:
  match decision:
    case Shape.BTrue{equal}:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +c = Nat.mul(digit, stride)
      +dt = Nat.div(target, left)
      +tr = Nat.mod(target, left)
      +hit = masked(stride, rest, tr, Nat.add(base, Nat.mul(right, dt)), 0n)
      +same = Arith.eq_true(digit, dt, Equal.sym(Bool, True{}, Nat.is_eq(digit, dt), equal))
      +mine = masked(stride, rest, tr, Nat.add(base, Nat.mul(right, digit)), 0n)
      %Equal.sym(List<&2, Nat>, masked(Nat.add(stride, Nat.mul(g, stride)), rows, target, base, c),
        List.append(&2, Nat, masked(stride, rows, target, base, c), masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))),
        masked_split(stride, Nat.mul(g, stride), rows, target, base, c)) :
        {_ == hit : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rows, target, base, c), mine,
        block(stride, left, right, rest, target, tr, base, digit, Hit{given, same, {==}})) :
        {List.append(&2, Nat, _, masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))) == hit : List<&2, Nat>}
      %next_cursor(digit, stride) :
        {List.append(&2, Nat, mine, masked(Nat.mul(g, stride), rows, target, base, _)) == hit : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(Nat.mul(g, stride), rows, target, base, Nat.mul(1n+digit, stride)), Nil{},
        kept_after(g, stride, left, right, rest, target, base, 1n+digit,
          %same : {Nat.is_lt(_, 1n+digit) == True{} : Bool}
          lt_self_succ(digit), bound)) :
        {List.append(&2, Nat, mine, _) == hit : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, List.append(&2, Nat, mine, Nil{}), mine, append_nil(mine)) : {_ == hit : List<&2, Nat>}
      Equal.cong(Nat, List<&2, Nat>, d => masked(stride, rest, tr, Nat.add(base, Nat.mul(right, d)), 0n), digit, dt, same)
    case Shape.BFalse{equal}:
      +rows : List<&2, Shape.Row> = Shape.Row{stride, left, right} <> rest
      +c = Nat.mul(digit, stride)
      +dt = Nat.div(target, left)
      +tr = Nat.mod(target, left)
      +hit = masked(stride, rest, tr, Nat.add(base, Nat.mul(right, dt)), 0n)
      +differ = Equal.sym(Bool, False{}, Nat.is_eq(digit, dt), equal)
      %Equal.sym(List<&2, Nat>, masked(Nat.add(stride, Nat.mul(g, stride)), rows, target, base, c),
        List.append(&2, Nat, masked(stride, rows, target, base, c), masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))),
        masked_split(stride, Nat.mul(g, stride), rows, target, base, c)) :
        {_ == hit : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rows, target, base, c), Nil{},
        block_off(stride, left, right, rest, target, base, digit, Off{bound, differ})) :
        {List.append(&2, Nat, _, masked(Nat.mul(g, stride), rows, target, base, Nat.add(c, stride))) == hit : List<&2, Nat>}
      %next_cursor(digit, stride) :
        {masked(Nat.mul(g, stride), rows, target, base, _) == hit : List<&2, Nat>}
      rec(1n+digit, le_ne_succ(digit, dt, from, differ),
        %Arith.add_succ(digit, g) : {Nat.is_lt(dt, _) == True{} : Bool}
        until)

# On a kept row, the blocks from `digit` that cover the target's digit hold
# exactly the rows below at the target's remainder.
def kept_hit(+f: Nat, +stride: Nat, +left: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +target: Nat, +base: Nat, +digit: Nat,
  +from: {Nat.is_le(digit, Nat.div(target, left)) == True{} : Bool},
  +until: {Nat.is_lt(Nat.div(target, left), Nat.add(digit, f)) == True{} : Bool},
  +given: Decomposed<target, left>, +bound: BoundBy<rest, left, stride>) ->
  {masked(Nat.mul(f, stride), Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) ==
    masked(stride, rest, Nat.mod(target, left), Nat.add(base, Nat.mul(right, Nat.div(target, left))), 0n) : List<&2, Nat>}:
  match f:
    case 0n:
      Empty.absurd({masked(0n, Shape.Row{stride, left, right} <> rest, target, base, Nat.mul(digit, stride)) ==
        masked(stride, rest, Nat.mod(target, left), Nat.add(base, Nat.mul(right, Nat.div(target, left))), 0n) : List<&2, Nat>},
        le_lt_absurd(digit, Nat.div(target, left), from,
          %Broadcast.add_zero(digit) : {Nat.is_lt(Nat.div(target, left), _) == True{} : Bool}
          until))
    case 1n+g:
      kept_hit_step(g, stride, left, right, rest, target, base, digit, Shape.bool_case(Nat.is_eq(digit, Nat.div(target, left))),
        from, until, given, bound,
        next => from => until => kept_hit(g, stride, left, right, rest, target, base, next, from, until, given, bound))

# A kept row walks the one digit the target names there.
def kept_row(+size: Nat, +stride: Nat, +right: Nat, +rest: List<&2, Shape.Row>, +extent: Nat, +others: List<&2, Nat>,
  +target: Nat, +base: Nat, +inside: {Nat.is_lt(target, Nat.mul(extent, size)) == True{} : Bool},
  +walk: RestWalk(rest, others, Nat.mod(target, size), stride), +bound: BoundBy<rest, size, stride>) ->
  {masked(Nat.mul(extent, stride), Shape.Row{stride, size, right} <> rest, target, base, 0n) ==
    visited(Shape.Row{stride, size, right} <> rest, extent <> others, Reduce.span(size, extent), target, base, Nat.div(target, size)) :
    List<&2, Nat>}:
  match size:
    case 0n:
      Empty.absurd({masked(Nat.mul(extent, stride), Shape.Row{stride, 0n, right} <> rest, target, base, 0n) ==
        visited(Shape.Row{stride, 0n, right} <> rest, extent <> others, Reduce.span(0n, extent), target, base, Nat.div(target, 0n)) :
        List<&2, Nat>},
        Arith.lt_zero(target,
          %Nats.mul_zero_right(extent) : {Nat.is_lt(target, _) == True{} : Bool}
          inside))
    case 1n+l:
      +dt = Nat.div(target, 1n+l)
      +tr = Nat.mod(target, 1n+l)
      +w = visited(rest, others, Reduce.first_span(rest, others), tr, Nat.add(base, Nat.mul(right, dt)), Reduce.first_digit(rest, tr))
      %Equal.sym(List<&2, Nat>, masked(Nat.mul(extent, stride), Shape.Row{stride, 1n+l, right} <> rest, target, base, 0n),
        masked(stride, rest, tr, Nat.add(base, Nat.mul(right, dt)), 0n),
        kept_hit(extent, stride, 1n+l, right, rest, target, base, 0n, Arith.le_zero(dt),
          div_below(target, extent, 1n+l, inside), decompose(target, 1n+l, {==}), bound)) :
        {_ == List.append(&2, Nat, w, Nil{}) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, masked(stride, rest, tr, Nat.add(base, Nat.mul(right, dt)), 0n), w,
        rest_at(rest, others, tr, stride, walk, Nat.add(base, Nat.mul(right, dt)))) :
        {_ == List.append(&2, Nat, w, Nil{}) : List<&2, Nat>}
      Equal.sym(List<&2, Nat>, List.append(&2, Nat, w, Nil{}), w, append_nil(w))

def RowsWalk(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +target: Nat) -> Data:
  RestWalk(Shape.index_rows(left, right, output), output, target, Shape.numel(output))

def fiber_cons(+lh: Nat, +lt: List<&2, Nat>, +rh: Nat, +rt: List<&2, Nat>, +oh: Nat, +ot: List<&2, Nat>, +axis: Shape.Axis<lh, rh, oh>,
  +tail: Shape.Aligned(lt, rt, ot), +target: Nat, +below: {Nat.is_lt(target, Nat.mul(lh, Shape.numel(lt))) == True{} : Bool},
  rec: @+inner: Nat -> @+inside: {Nat.is_lt(inner, Shape.numel(lt)) == True{} : Bool} -> RowsWalk(lt, rt, ot, inner)) ->
  RowsWalk(lh <> lt, rh <> rt, oh <> ot, target):
  match lh:
    case 0n:
      Empty.absurd(RowsWalk(0n <> lt, rh <> rt, oh <> ot, target), Arith.lt_zero(target, below))
    case 1n+k:
      match k:
        case 0n:
          reduced_blocks(oh, Shape.numel(ot), Shape.broadcast_index(rh, Shape.numel(rt)), Shape.index_rows(lt, rt, ot), oh, ot, target, 0n, 0n,
            rec(target,
              %Broadcast.add_zero(Shape.numel(lt)) : {Nat.is_lt(target, _) == True{} : Bool}
              below))
        case 1n+j:
          +size = Shape.numel(lt)
          kept_row(size, Shape.numel(ot), Shape.broadcast_index(rh, Shape.numel(rt)), Shape.index_rows(lt, rt, ot), oh, ot, target, 0n,
            %Equal.sym(Nat, oh, 2n+j, axis_kept(j, rh, oh, axis)) : {Nat.is_lt(target, Nat.mul(_, size)) == True{} : Bool}
            below,
            rec(Nat.mod(target, size),
              {\{Decomposed: m_below => sum => m_below} : Decomposed<target, size> -> {Nat.is_lt(Nat.mod(target, size), size) == True{} : Bool}}(
                decompose(target, size, positive_of(target, 2n+j, size, below)))),
            BoundBy{lt, rt, ot, tail, {==}, {==}, {==}})

# The fiber-walk lemma: for the index rows of an aligned broadcast and a
# cell below the left count, the offsets the walk reads, in order, are the
# masked positions.
def fiber_walk(+left: List<&2, Nat>, +right: List<&2, Nat>, +output: List<&2, Nat>, +aligned: Shape.Aligned(left, right, output),
  +target: Nat, +below: {Nat.is_lt(target, Shape.numel(left)) == True{} : Bool}) -> RowsWalk(left, right, output, target):
  match left right output:
    case Nil{} Nil{} Nil{}:
      %Arith.lt_one(target, below) : {kept(Nat.is_eq(0n, _), 0n, Nil{}) == [0n] : List<&2, Nat>}
      {==}
    case +lh <> +lt +rh <> +rt +oh <> +ot:
      Shape.AStep{+axis, +tail} = aligned
      fiber_cons(lh, lt, rh, rt, oh, ot, axis, tail, target, below,
        inner => inside => fiber_walk(lt, rt, ot, tail, inner, inside))
    case Nil{} Nil{} +oh <> +ot:
      Empty.absurd(RowsWalk(Nil{}, Nil{}, oh <> ot, target), aligned)
    case Nil{} +rh <> +rt Nil{}:
      Empty.absurd(RowsWalk(Nil{}, rh <> rt, Nil{}, target), aligned)
    case +lh <> +lt Nil{} Nil{}:
      Empty.absurd(RowsWalk(lh <> lt, Nil{}, Nil{}, target), aligned)
    case Nil{} +rh <> +rt +oh <> +ot:
      Empty.absurd(RowsWalk(Nil{}, rh <> rt, oh <> ot, target), aligned)
    case +lh <> +lt Nil{} +oh <> +ot:
      Empty.absurd(RowsWalk(lh <> lt, Nil{}, oh <> ot, target), aligned)
    case +lh <> +lt +rh <> +rt Nil{}:
      Empty.absurd(RowsWalk(lh <> lt, rh <> rt, Nil{}, target), aligned)
