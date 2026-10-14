import Base
import ./core.bend as Shape
import ./index_proofs.bend as Index
import ./broadcast_proofs.bend as Broadcast

def contradiction(-value: Bool, yes: {True{} == value : Bool}, no: {False{} == value : Bool}) -> Empty:
  Shape.clash(Equal.trans(Bool, True{}, value, False{}, yes, Equal.sym(Bool, False{}, value, no)))

def le_true(+a: Nat, +b: Nat, bound: Shape.LE(a, b)) -> {True{} == Nat.is_le(a, b) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      match bound:
    case 1n+p 1n+q:
      le_true(p, q, bound)

def lt_true(+a: Nat, +b: Nat, bound: Shape.LT(a, b)) -> {True{} == Nat.is_lt(a, b) : Bool}:
  match a b:
    case 0n 0n:
      match bound:
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      match bound:
    case 1n+p 1n+q:
      lt_true(p, q, bound)

def le_add_left(+a: Nat, -x: Nat, +c: Nat, bound: Shape.LE(Nat.add(a, x), c)) -> Shape.LE(a, c):
  match a c:
    case 0n c:
      Unit{}
    case 1n+p 0n:
      match bound:
    case 1n+p 1n+q:
      le_add_left(p, x, q, bound)

def le_rewrite(-a: Nat, -b: Nat, -c: Nat, equal: {b == a : Nat}, bound: Shape.LE(b, c)) -> Shape.LE(a, c):
  %equal : Shape.LE(_, c)
  bound

def add_swap(+a: Nat, +b: Nat, -c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(b, Nat.add(a, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.add(p, Nat.add(b, c)), Nat.add(b, Nat.add(p, c)), add_swap(p, b, c)) : {1n+_ == Nat.add(b, 1n+Nat.add(p, c)) : Nat}
      Broadcast.add_succ(b, Nat.add(p, c))

def mul_succ_right(+a: Nat, +q: Nat) -> {Nat.mul(a, 1n+q) == Nat.add(a, Nat.mul(a, q)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(p, 1n+q), Nat.add(p, Nat.mul(p, q)), mul_succ_right(p, q)) : {1n+Nat.add(q, _) == 1n+Nat.add(p, Nat.add(q, Nat.mul(p, q))) : Nat}
      %add_swap(q, p, Nat.mul(p, q)) : {1n+Nat.add(q, Nat.add(p, Nat.mul(p, q))) == 1n+_ : Nat}
      {==}

def mul_zero_right(+a: Nat) -> {Nat.mul(a, 0n) == 0n : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      mul_zero_right(p)

def mul_one_right(+a: Nat) -> {Nat.mul(a, 1n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, value => 1n+value, Nat.mul(p, 1n), p, mul_one_right(p))

def mul_comm(+a: Nat, +b: Nat) -> {Nat.mul(a, b) == Nat.mul(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.mul(b, 0n), 0n, mul_zero_right(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(b, 1n+p), Nat.add(b, Nat.mul(b, p)), mul_succ_right(b, p)) : {Nat.add(b, Nat.mul(p, b)) == _ : Nat}
      Equal.cong(Nat, Nat, value => Nat.add(b, value), Nat.mul(p, b), Nat.mul(b, p), mul_comm(p, b))

def mul_add_right(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(Nat.add(a, b), c) == Nat.add(Nat.mul(a, c), Nat.mul(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(Nat.add(p, b), c), Nat.add(Nat.mul(p, c), Nat.mul(b, c)), mul_add_right(p, b, c)) : {Nat.add(c, _) == Nat.add(Nat.add(c, Nat.mul(p, c)), Nat.mul(b, c)) : Nat}
      Index.add_assoc(c, Nat.mul(p, c), Nat.mul(b, c))

def mul_assoc(+a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(Nat.mul(a, b), c) == Nat.mul(a, Nat.mul(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(Nat.add(b, Nat.mul(p, b)), c), Nat.add(Nat.mul(b, c), Nat.mul(Nat.mul(p, b), c)), mul_add_right(b, Nat.mul(p, b), c)) : {_ == Nat.mul(1n+p, Nat.mul(b, c)) : Nat}
      %Equal.sym(Nat, Nat.mul(Nat.mul(p, b), c), Nat.mul(p, Nat.mul(b, c)), mul_assoc(p, b, c)) : {Nat.add(Nat.mul(b, c), _) == Nat.mul(1n+p, Nat.mul(b, c)) : Nat}
      {==}

def lt_zero(+a: Nat, bound: Shape.LT(a, 0n)) -> Empty:
  match a:
    case 0n:
      match bound:
    case 1n+p:
      match bound:

def lt_le(+a: Nat, +b: Nat, bound: Shape.LT(a, 1n+b)) -> Shape.LE(a, b):
  match a b:
    case 0n b:
      Unit{}
    case 1n+p 0n:
      Empty.absurd(Empty, lt_zero(p, bound))
    case 1n+p 1n+q:
      lt_le(p, q, bound)

def drain(+k: Nat, -n: Nat, -m: Nat, -d: Nat, +r: Nat) -> {Nat.divmod.go(Nat.add(k, n), Nat.add(k, m), d, r) == Nat.divmod.go(n, m, d, Nat.add(k, r)) : Nat & Nat}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(j, n), Nat.add(j, m), d, 1n+r), Nat.divmod.go(n, m, d, Nat.add(j, 1n+r)), drain(j, n, m, d, 1n+r)) : {_ == Nat.divmod.go(n, m, d, 1n+Nat.add(j, r)) : Nat & Nat}
      %Broadcast.add_succ(j, r) : {Nat.divmod.go(n, m, d, _) == Nat.divmod.go(n, m, d, 1n+Nat.add(j, r)) : Nat & Nat}
      {==}

def remainder(+r: Nat, +m: Nat, -d: Nat, +a: Nat, bound: Shape.LE(r, m)) -> {Nat.divmod.go(r, m, d, a) == (d, Nat.add(r, a)) : Nat & Nat}:
  match r m:
    case 0n m:
      {==}
    case 1n+p 0n:
      match bound:
    case 1n+p 1n+q:
      %Equal.sym(Nat & Nat, Nat.divmod.go(p, q, d, 1n+a), (d, Nat.add(p, 1n+a)), remainder(p, q, d, 1n+a, bound)) : {_ == (d, 1n+Nat.add(p, a)) : Nat & Nat}
      %Broadcast.add_succ(p, a) : {(d, _) == (d, 1n+Nat.add(p, a)) : Nat & Nat}
      {==}

def periods(+q: Nat, +b: Nat, +r: Nat, +d: Nat, bound: Shape.LE(r, b)) -> {Nat.divmod.go(Nat.add(Nat.mul(q, 1n+b), r), b, d, 0n) == (Nat.add(q, d), r) : Nat & Nat}:
  match q:
    case 0n:
      %Equal.sym(Nat & Nat, Nat.divmod.go(r, b, d, 0n), (d, Nat.add(r, 0n)), remainder(r, b, d, 0n, bound)) : {_ == (d, r) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(r, 0n), r, Broadcast.add_zero(r)) : {(d, _) == (d, r) : Nat & Nat}
      {==}
    case 1n+p:
      +rest = Nat.add(Nat.mul(p, 1n+b), r)
      %Index.add_assoc(b, Nat.mul(p, 1n+b), r) : {Nat.divmod.go(1n+_, b, d, 0n) == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Equal.sym(Nat, 1n+Nat.add(b, rest), Nat.add(b, 1n+rest), Broadcast.add_succ(b, rest)) : {Nat.divmod.go(_, b, d, 0n) == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Broadcast.add_zero(b) : {Nat.divmod.go(Nat.add(b, 1n+rest), _, d, 0n) == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(b, 1n+rest), Nat.add(b, 0n), d, 0n), Nat.divmod.go(1n+rest, 0n, d, Nat.add(b, 0n)), drain(b, 1n+rest, 0n, d, 0n)) : {_ == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(b, 0n), b, Broadcast.add_zero(b)) : {Nat.divmod.go(1n+rest, 0n, d, _) == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(rest, b, 1n+d, 0n), (Nat.add(p, 1n+d), r), periods(p, b, r, 1n+d, bound)) : {_ == (1n+Nat.add(p, d), r) : Nat & Nat}
      %Broadcast.add_succ(p, d) : {(_, r) == (1n+Nat.add(p, d), r) : Nat & Nat}
      {==}

def divmod_exact(+q: Nat, +b: Nat, +r: Nat, bound: Shape.LT(r, b)) -> {(q, r) == Nat.divmod(Nat.add(Nat.mul(q, b), r), b) : Nat & Nat}:
  match b:
    case 0n:
      Empty.absurd({(q, r) == Nat.divmod(Nat.add(Nat.mul(q, 0n), r), 0n) : Nat & Nat}, lt_zero(r, bound))
    case 1n+p:
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(Nat.mul(q, 1n+p), r), p, 0n, 0n), (Nat.add(q, 0n), r), periods(q, p, r, 0n, lt_le(r, p, bound))) : {(q, r) == _ : Nat & Nat}
      %Equal.sym(Nat, Nat.add(q, 0n), q, Broadcast.add_zero(q)) : {(q, r) == (_, r) : Nat & Nat}
      {==}
