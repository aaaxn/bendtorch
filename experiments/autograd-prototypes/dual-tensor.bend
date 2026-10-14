import Base

import ../../src/tensor/core.bend as Core

import ./tensor-ops.bend as Ops

def R(-A: Type) -> Type:
  Result<&2, &1, Ops.Error, A>

def D() -> Type:
  Core.Dynamic & Core.Dynamic

# Keeps an op's output and frees its returned operands.
def out(result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(Core.Dynamic):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, output) = triple
      Done{output}

def add(a: D(), b: D()) -> R(D()):
  (av, ad) = a
  (bv, bd) = b
  do Result<&2, &1, Ops.Error, D()>:
    v : Core.Dynamic <- out(Ops.add(av, bv))
    t : Core.Dynamic <- out(Ops.add(ad, bd))
    return (v, t)

def mul_cloned(ad: Core.Dynamic, bd: Core.Dynamic, ac: Core.Dynamic & Core.Dynamic, bc: Core.Dynamic & Core.Dynamic) -> R(D()):
  (a1, a2) = ac
  (b1, b2) = bc
  do Result<&2, &1, Ops.Error, D()>:
    v : Core.Dynamic <- out(Ops.mul(a1, b1))
    left : Core.Dynamic <- out(Ops.mul(ad, b2))
    right : Core.Dynamic <- out(Ops.mul(a2, bd))
    t : Core.Dynamic <- out(Ops.add(left, right))
    return (v, t)

# d(a*b) = da*b + a*db.
def mul(a: D(), b: D()) -> R(D()):
  (av, ad) = a
  (bv, bd) = b
  mul_cloned(ad, bd, Core.clone_dynamic(av), Core.clone_dynamic(bv))

def sum_value(result: R(Core.Dynamic & Core.Dynamic)) -> R(Core.Dynamic):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (input, scalar) = pair
      Done{scalar}

def sum(a: D()) -> R(D()):
  (av, ad) = a
  do Result<&2, &1, Ops.Error, D()>:
    v : Core.Dynamic <- sum_value(Ops.sum(av))
    t : Core.Dynamic <- sum_value(Ops.sum(ad))
    return (v, t)

def dual(+dims: List<&2, U32>, values: List<&2, F32>, tangent: List<&2, F32>) -> R(D()):
  do Result<&2, &1, Ops.Error, D()>:
    v : Core.Dynamic <- Ops.construct(dims, values)
    t : Core.Dynamic <- Ops.construct(dims, tangent)
    return (v, t)

def one_hot_value(hit: Bool) -> F32:
  match hit:
    case True{}:
      1.0
    case False{}:
      0.0

# The tangent list with 1 at `hot` and 0 elsewhere, for `n` elements from `at`.
def one_hot(n: Nat, +at: Nat, +hot: Nat) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      one_hot_value(Nat.is_eq(at, hot)) <> one_hot(p, 1n+at, hot)
