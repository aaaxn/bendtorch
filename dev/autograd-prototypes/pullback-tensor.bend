import Base

import ../../src/shape/core.bend as Shape

import ../../src/tensor/core.bend as Core

import ./tensor-ops.bend as Ops

type Env is Type:
  Env{grads: List<&1, Ops.Slot()>, clones: Nat}

# The caller's copies of the inputs, newest first.
type St is Type:
  St{slots: Nat, clones: Nat, saved: Nat, kept: List<&1, Core.Dynamic>}

type Outcome is Type:
  Outcome{value: Core.Dynamic, inputs: List<&1, Core.Dynamic>, grads: List<&1, Ops.Slot()>,
    forward_clones: Nat, backward_clones: Nat, saved: Nat}

def R(-A: Type) -> Type:
  Result<&2, &1, Ops.Error, A>

def Back() -> Type:
  Core.Dynamic -> Env -> R(Env)

def V() -> Type:
  Core.Dynamic & Back()

def Gen(-A: Type) -> Type:
  St -> R(St & A)

def Gen.pure(-A: Type, x: A) -> Gen(A):
  st => Done{(st, x)}

def Gen.bind.go(-A: Type, -B: Type, k: A -> Gen(B), result: R(St & A)) -> R(St & B):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (st, x) = pair
      k(x)(st)

def Gen.bind(-A: Type, -B: Type, m: Gen(A), k: A -> Gen(B)) -> Gen(B):
  st => Gen.bind.go(A, B, k, m(st))

def env_done(+clones: Nat, result: R(List<&1, Ops.Slot()>)) -> R(Env):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{grads}:
      Done{Env{grads, clones}}

def deposit(+slot: Nat, g: Core.Dynamic, env: Env) -> R(Env):
  Env{grads, +clones} = env
  env_done(clones, Ops.accumulate(grads, slot, g))

def input_split(st: St, pair: Core.Dynamic & Core.Dynamic) -> R(St & V()):
  St{+slots, clones, saved, kept} = st
  (copy, value) = pair
  Done{(St{1n+slots, 1n+clones, saved, copy <> kept}, (value, g => env => deposit(slots, g, env)))}

def input(value: Core.Dynamic) -> Gen(V()):
  st => input_split(st, Core.clone_dynamic(value))

def then(result: R(Env), k: Env -> R(Env)) -> R(Env):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{env}:
      k(env)

def send(back: Back(), result: R(Core.Dynamic), env: Env) -> R(Env):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{g}:
      back(g)(env)

def count_clone(env: Env) -> Env:
  Env{grads, clones} = env
  Env{grads, 1n+clones}

def product_back(back: Back(), +shape: Shape.Shape, env: Env, result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(Env):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, product) = triple
      send(back, Ops.unbroadcast(product, shape), env)

def mul_pullback(+a_shape: Shape.Shape, +b_shape: Shape.Shape, av: Core.Dynamic, bv: Core.Dynamic,
  ab: Back(), bb: Back(), gs: Core.Dynamic & Core.Dynamic, env: Env) -> R(Env):
  (g1, g2) = gs
  then(product_back(ab, a_shape, count_clone(env), Ops.mul(g1, bv)),
    next => product_back(bb, b_shape, next, Ops.mul(g2, av)))

def mul_shapes(ab: Back(), bb: Back(), sa: Core.Dynamic & Shape.Shape, sb: Core.Dynamic & Shape.Shape, out: Core.Dynamic) -> V():
  (av, +a_shape) = sa
  (bv, +b_shape) = sb
  (out, g => env => mul_pullback(a_shape, b_shape, av, bv, ab, bb, Core.clone_dynamic(g), env))

def saved_two(st: St) -> St:
  St{slots, clones, saved, kept} = st
  St{slots, clones, 2n+saved, kept}

def mul_done(ab: Back(), bb: Back(), st: St, result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(St & V()):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, out) = triple
      (av, bv) = operands
      Done{(saved_two(st), mul_shapes(ab, bb, Core.shape(av), Core.shape(bv), out))}

# d(a*b) = (g*b, g*a).
def mul(a: V(), b: V()) -> Gen(V()):
  (av, ab) = a
  (bv, bb) = b
  st => mul_done(ab, bb, st, Ops.mul(av, bv))

def add_pullback(+a_shape: Shape.Shape, +b_shape: Shape.Shape, ab: Back(), bb: Back(), gs: Core.Dynamic & Core.Dynamic, env: Env) -> R(Env):
  (g1, g2) = gs
  then(send(ab, Ops.unbroadcast(g1, a_shape), count_clone(env)),
    next => send(bb, Ops.unbroadcast(g2, b_shape), next))

def add_shapes(ab: Back(), bb: Back(), sa: Core.Dynamic & Shape.Shape, sb: Core.Dynamic & Shape.Shape, out: Core.Dynamic) -> V():
  (av, +a_shape) = sa
  (bv, +b_shape) = sb
  (out, g => env => add_pullback(a_shape, b_shape, ab, bb, Core.clone_dynamic(g), env))

def add_done(ab: Back(), bb: Back(), st: St, result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(St & V()):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, out) = triple
      (av, bv) = operands
      Done{(st, add_shapes(ab, bb, Core.shape(av), Core.shape(bv), out))}

def add(a: V(), b: V()) -> Gen(V()):
  (av, ab) = a
  (bv, bb) = b
  st => add_done(ab, bb, st, Ops.add(av, bv))

def sum_fill(back: Back(), +shape: Shape.Shape, env: Env, result: R(Core.Dynamic & F32)) -> R(Env):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (g, value) = pair
      send(back, Ops.fill(shape, value), env)

def sum_shaped(back: Back(), sa: Core.Dynamic & Shape.Shape, out: Core.Dynamic) -> V():
  (av, +a_shape) = sa
  (out, g => env => sum_fill(back, a_shape, env, Ops.scalar(g)))

def sum_done(back: Back(), st: St, result: R(Core.Dynamic & Core.Dynamic)) -> R(St & V()):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (av, out) = pair
      Done{(st, sum_shaped(back, Core.shape(av), out))}

def sum(a: V()) -> Gen(V()):
  (av, ab) = a
  st => sum_done(ab, st, Ops.sum(av))

# Takes the summed cotangent of a shared value out of its slot and sends it on once.
def share_flush(back: Back(), +slot: Nat, taken: List<&1, Ops.Slot()> & Ops.Slot(), +clones: Nat) -> R(Env):
  (grads, got) = taken
  match got:
    case None{}:
      Done{Env{grads, clones}}
    case Some{g}:
      back(g)(Env{grads, clones})

def share_back(back: Back(), +slot: Nat, env: Env) -> R(Env):
  Env{grads, +clones} = env
  share_flush(back, slot, Ops.take(grads, slot), clones)

def share_after(back: Back(), +slot: Nat, result: R(St & V())) -> R(St & V()):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (st, y) = pair
      (yv, yb) = y
      Done{(st, (yv, g => env => then(yb(g)(env), next => share_back(back, slot, next))))}

def share_split(back: Back(), k: V() -> V() -> Gen(V()), st: St, pair: Core.Dynamic & Core.Dynamic) -> R(St & V()):
  St{+slots, clones, saved, kept} = st
  (first, second) = pair
  share_after(back, slots,
    k((first, g => env => deposit(slots, g, env)), (second, g => env => deposit(slots, g, env)))(St{1n+slots, 1n+clones, saved, kept}))

def share(a: V(), k: V() -> V() -> Gen(V())) -> Gen(V()):
  (av, ab) = a
  st => share_split(ab, k, st, Core.clone_dynamic(av))

def empties(n: Nat) -> List<&1, Ops.Slot()>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      None{} <> empties(p)

def outcome(value: Core.Dynamic, kept: List<&1, Core.Dynamic>, +clones: Nat, +saved: Nat, result: R(Env)) -> R(Outcome):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{env}:
      Env{grads, backward} = env
      Done{Outcome{value, List.reverse(&1, Core.Dynamic, kept), grads, clones, backward, saved}}

def seeded(seed: R(Core.Dynamic), value: Core.Dynamic, back: Back(), st: St) -> R(Outcome):
  match seed:
    case Fail{e}:
      Fail{e}
    case Done{one}:
      St{+slots, +clones, +saved, kept} = st
      outcome(value, kept, clones, saved, back(one)(Env{empties(slots), 0n}))

def shaped(back: Back(), st: St, pair: Core.Dynamic & Shape.Shape) -> R(Outcome):
  (value, +shape) = pair
  seeded(Ops.fill(shape, 1.0), value, back, st)

def run_built(result: R(St & V())) -> R(Outcome):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (st, y) = pair
      (yv, yb) = y
      shaped(yb, st, Core.shape(yv))

# Input k owns slot k.
def run(program: Gen(V())) -> R(Outcome):
  run_built(program(St{0n, 0n, 0n, Nil{}}))
