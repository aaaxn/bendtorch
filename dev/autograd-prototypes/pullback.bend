import Base

# A value used twice goes through `share`, which gives it a fresh slot and calls
# its pullback once with the summed cotangent.

def Back(-T: Data) -> Type:
  T -> List<&2, T> -> List<&2, T>

def V(-T: Data) -> Type:
  T & Back(T)

def Gen(-A: Type) -> Type:
  Nat -> Nat & A

def Gen.pure(-A: Type, x: A) -> Gen(A):
  n => (n, x)

def Gen.bind.go(-A: Type, -B: Type, k: A -> Gen(B), result: Nat & A) -> Nat & B:
  (n, x) = result
  k(x)(n)

def Gen.bind(-A: Type, -B: Type, m: Gen(A), k: A -> Gen(B)) -> Gen(B):
  n => Gen.bind.go(A, B, k, m(n))

def bump(~T: Data, ~plus: T -> T -> T, env: List<&2, T>, +slot: Nat, +delta: T) -> List<&2, T>:
  match env slot:
    case Nil{} _:
      Nil{}
    case Con{g, rest} 0n:
      plus(g, delta) <> rest
    case Con{g, rest} 1n+p:
      g <> bump(~T, ~plus, rest, p, delta)

def at(-T: Data, xs: List<&2, T>, +slot: Nat, +fallback: T) -> T:
  match xs slot:
    case Nil{} _:
      fallback
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      at(T, t, p, fallback)

def input(~T: Data, ~plus: T -> T -> T, value: T) -> Gen(V(T)):
  +slot => (1n+slot, (value, +g => env => bump(~T, ~plus, env, slot, g)))

def add_go(~T: Data, ~plus: T -> T -> T, av: T, ab: Back(T), b: V(T)) -> V(T):
  (bv, bb) = b
  (plus(av, bv), +g => env => bb(g)(ab(g)(env)))

def add(~T: Data, ~plus: T -> T -> T, a: V(T), b: V(T)) -> V(T):
  (av, ab) = a
  add_go(~T, ~plus, av, ab, b)

def mul_go(~T: Data, ~times: T -> T -> T, +av: T, ab: Back(T), b: V(T)) -> V(T):
  (+bv, bb) = b
  (times(av, bv), +g => env => bb(times(g, av))(ab(times(g, bv))(env)))

def mul(~T: Data, ~times: T -> T -> T, a: V(T), b: V(T)) -> V(T):
  (+av, ab) = a
  mul_go(~T, ~times, av, ab, b)

def finish_share(-T: Data, +slot: Nat, +zero: T, ab: Back(T), env: List<&2, T>) -> List<&2, T>:
  match env:
    case Nil{}:
      Nil{}
    case Con{+h, +t}:
      ab(at(T, Con{h, t}, slot, zero))(Con{h, t})

def shared(-T: Data, +slot: Nat, +zero: T, ab: Back(T), result: Nat & V(T)) -> Nat & V(T):
  (n, r) = result
  (rv, rb) = r
  (n, (rv, +g => env => finish_share(T, slot, zero, ab, rb(g)(env))))

def share(~T: Data, ~plus: T -> T -> T, a: V(T), +zero: T, k: V(T) -> V(T) -> Gen(V(T))) -> Gen(V(T)):
  (+av, ab) = a
  +slot => shared(T, slot, zero, ab,
    k((av, +g => env => bump(~T, ~plus, env, slot, g)), (av, +g => env => bump(~T, ~plus, env, slot, g)))(1n+slot))

def zeros(-T: Data, n: Nat, +zero: T) -> List<&2, T>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      zero <> zeros(T, p, zero)

def read(-T: Data, +env: List<&2, T>, slots: List<&2, Nat>, +zero: T) -> List<&2, T>:
  match slots:
    case Nil{}:
      Nil{}
    case Con{+slot, rest}:
      at(T, env, slot, zero) <> read(T, env, rest, zero)

def sweep(-T: Data, slots: List<&2, Nat>, +zero: T, +one: T, result: Nat & V(T)) -> T & List<&2, T>:
  (n, y) = result
  (yv, yb) = y
  (yv, read(T, yb(one)(zeros(T, n, zero)), slots, zero))

# Inputs are created first, so input k owns slot k.
def run(-T: Data, slots: List<&2, Nat>, +zero: T, +one: T, program: Gen(V(T))) -> T & List<&2, T>:
  sweep(T, slots, zero, one, program(0n))
