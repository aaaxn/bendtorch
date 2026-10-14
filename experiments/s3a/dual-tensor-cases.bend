import Base

import ../../src/tensor/core.bend as Core

import ./tensor-ops.bend as Ops

import ./dual-tensor.bend as Dual

# Case 4 with forward duals, y = sum(W*x + b) with W:[2,3], x:[3], b:[3].
# Parameters are numbered W[0..5], x[6..8], b[9..11]; pass k seeds parameter k.

def w_values() -> List<&2, F32>:
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

def x_values() -> List<&2, F32>:
  [0.5, F32.neg(1.0), 2.0]

def b_values() -> List<&2, F32>:
  [1.0, F32.neg(1.0), 0.5]

def pass(+k: Nat) -> Dual.R(Dual.D()):
  do Result<&2, &1, Ops.Error, Dual.D()>:
    w : Dual.D() <- Dual.dual([2, 3], w_values(), Dual.one_hot(6n, 0n, k))
    x : Dual.D() <- Dual.dual([3], x_values(), Dual.one_hot(3n, 6n, k))
    b : Dual.D() <- Dual.dual([3], b_values(), Dual.one_hot(3n, 9n, k))
    wx : Dual.D() <- Dual.mul(w, x)
    z : Dual.D() <- Dual.add(wx, b)
    Dual.sum(z)

def clone_parts(vs: Core.Dynamic & Core.Dynamic, ts: Core.Dynamic & Core.Dynamic) -> Dual.D() & Dual.D():
  (v1, v2) = vs
  (t1, t2) = ts
  ((v1, t1), (v2, t2))

def clone_dual(d: Dual.D()) -> Dual.D() & Dual.D():
  (v, t) = d
  clone_parts(Core.clone_dynamic(v), Core.clone_dynamic(t))

# Case 1 on a rank-0 dual: x is read three times, so the dual is cloned twice.
def case1_square(x1: Dual.D(), rest: Dual.D() & Dual.D()) -> Dual.R(Dual.D()):
  (x2, x3) = rest
  do Result<&2, &1, Ops.Error, Dual.D()>:
    sq : Dual.D() <- Dual.mul(x1, x2)
    Dual.add(sq, x3)

def case1_split(pair: Dual.D() & Dual.D()) -> Dual.R(Dual.D()):
  (x1, x23) = pair
  case1_square(x1, clone_dual(x23))

def case1_pass(x: Dual.D()) -> Dual.R(Dual.D()):
  case1_split(clone_dual(x))

def case1() -> Dual.R(Dual.D()):
  do Result<&2, &1, Ops.Error, Dual.D()>:
    x : Dual.D() <- Dual.dual([], [3.0], [1.0])
    case1_pass(x)

def show_parts(print_value: Bool, value: Core.Dynamic, tangent: Core.Dynamic) -> IO(Unit):
  match print_value:
    case True{}:
      do IO<Unit>:
        Ops.show_tensor("value", value)
        Ops.show_tensor("tangent", tangent)
    case False{}:
      Ops.show_tensor("tangent", tangent)

def show_pass(+print_value: Bool, result: Dual.R(Dual.D())) -> IO(Unit):
  match result:
    case Fail{e}:
      IO.print("dual-tensor error " ++ U32.show(Ops.error_code(e)))
    case Done{pair}:
      (value, tangent) = pair
      show_parts(print_value, value, tangent)

def passes(fuel: Nat, +k: Nat) -> IO(Unit):
  match fuel:
    case 0n:
      do IO<Unit>:
        return Unit{}
    case 1n+rest:
      do IO<Unit>:
        show_pass(Nat.is_eq(k, 0n), pass(k))
        passes(rest, 1n+k)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("dual-tensor case4")
    passes(12n, 0n)
    IO.print("dual-tensor case1")
    show_pass(True{}, case1())
    return Unit{}
