import Base

import ../../src/shape/core.bend as Shape

import ../../src/tensor/core.bend as Core

import ./tensor-ops.bend as Ops

# Taking an empty slot answers Missing, so an alias cannot be formed: x*x
# clones the operand once, and both contributions go to the original node id in
# the backward sweep.

type Node is Data:
  Input{}
  Add{a: Nat, b: Nat}
  Mul{a: Nat, b: Nat}
  Sum{a: Nat}

# Nodes and slots are newest first: the head is node `size - 1`.
type Tape is Type:
  Tape{size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat}

type Aux is Type:
  Aux{clones: Nat, leaves: List<&1, Leaf()>}

type Sweep is Type:
  Sweep{slots: List<&1, Ops.Slot()>, grads: List<&1, Ops.Slot()>, aux: Aux}

# `leaves` holds each input's (value, gradient), oldest first.
type Outcome is Type:
  Outcome{value: Core.Dynamic, leaves: List<&1, Leaf()>, forward_clones: Nat, backward_clones: Nat, saved: Nat}

def R(-A: Type) -> Type:
  Result<&2, &1, Ops.Error, A>

def Build(-A: Type) -> Type:
  Tape -> R(Tape & A)

def Build.pure(-A: Type, x: A) -> Build(A):
  tape => Done{(tape, x)}

def Build.bind.go(-A: Type, -B: Type, k: A -> Build(B), result: R(Tape & A)) -> R(Tape & B):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (tape, x) = pair
      k(x)(tape)

def Build.bind(-A: Type, -B: Type, m: Build(A), k: A -> Build(B)) -> Build(B):
  tape => Build.bind.go(A, B, k, m(tape))

def offset_of(+size: Nat, +id: Nat) -> Nat:
  Nat.sub(Nat.sub(size, 1n), id)

def push(node: Node, value: Core.Dynamic, tape: Tape) -> Tape & Nat:
  Tape{+size, nodes, slots, clones} = tape
  (Tape{1n+size, node <> nodes, Some{value} <> slots, clones}, size)

def input(value: Core.Dynamic) -> Build(Nat):
  tape => Done{push(Input{}, value, tape)}

# Puts both operands back and records the output.
def binary_done(node: Node, +size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat,
  +a: Nat, +b: Nat, result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(Tape & Nat):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, out) = triple
      (left, right) = operands
      Done{push(node, out, Tape{size, nodes, Ops.put(Ops.put(slots, offset_of(size, a), left), offset_of(size, b), right), clones})}

def binary_same_done(node: Node, +size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat,
  +a: Nat, result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(Tape & Nat):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, out) = triple
      (left, copy) = operands
      Done{push(node, out, Tape{size, nodes, Ops.put(slots, offset_of(size, a), left), clones})}

# The same id twice: one clone feeds the right operand and is dropped afterwards.
def binary_same(node: Node, +size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat,
  +a: Nat, op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic), pair: Core.Dynamic & Core.Dynamic) -> R(Tape & Nat):
  (left, right) = pair
  binary_same_done(node, size, nodes, slots, 1n+clones, a, op(left, right))

def binary_right(node: Node, +size: Nat, nodes: List<&2, Node>, clones: Nat, +a: Nat, +b: Nat,
  op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic), left: Core.Dynamic,
  taken: List<&1, Ops.Slot()> & Ops.Slot()) -> R(Tape & Nat):
  (slots, slot) = taken
  match slot:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{right}:
      binary_done(node, size, nodes, slots, clones, a, b, op(left, right))

def binary_choose(node: Node, +size: Nat, nodes: List<&2, Node>, clones: Nat, +a: Nat, +b: Nat, same: Bool,
  op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic),
  slots: List<&1, Ops.Slot()>, slot: Ops.Slot()) -> R(Tape & Nat):
  match same slot:
    case _ None{}:
      Fail{Ops.Missing{}}
    case True{} Some{left}:
      binary_same(node, size, nodes, slots, clones, a, op, Core.clone_dynamic(left))
    case False{} Some{left}:
      binary_right(node, size, nodes, clones, a, b, op, left, Ops.take(slots, offset_of(size, b)))

def binary_left(node: Node, +size: Nat, nodes: List<&2, Node>, clones: Nat, +a: Nat, +b: Nat, same: Bool,
  op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic),
  taken: List<&1, Ops.Slot()> & Ops.Slot()) -> R(Tape & Nat):
  (slots, slot) = taken
  binary_choose(node, size, nodes, clones, a, b, same, op, slots, slot)

def binary_tape(node: Node, +a: Nat, +b: Nat, op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic), tape: Tape) -> R(Tape & Nat):
  Tape{+size, nodes, slots, clones} = tape
  binary_left(node, size, nodes, clones, a, b, Nat.is_eq(a, b), op, Ops.take(slots, offset_of(size, a)))

def binary(node: Node, +a: Nat, +b: Nat, op: Core.Dynamic -> Core.Dynamic -> R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> Build(Nat):
  tape => binary_tape(node, a, b, op, tape)

def add(+a: Nat, +b: Nat) -> Build(Nat):
  binary(Add{a, b}, a, b, Ops.add)

def mul(+a: Nat, +b: Nat) -> Build(Nat):
  binary(Mul{a, b}, a, b, Ops.mul)

def sum_done(+size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat, +a: Nat,
  result: R(Core.Dynamic & Core.Dynamic)) -> R(Tape & Nat):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (input, scalar) = pair
      Done{push(Sum{a}, scalar, Tape{size, nodes, Ops.put(slots, offset_of(size, a), input), clones})}

def sum_taken(+size: Nat, nodes: List<&2, Node>, clones: Nat, +a: Nat, taken: List<&1, Ops.Slot()> & Ops.Slot()) -> R(Tape & Nat):
  (slots, slot) = taken
  match slot:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{value}:
      sum_done(size, nodes, slots, clones, a, Ops.sum(value))

def sum_tape(+a: Nat, tape: Tape) -> R(Tape & Nat):
  Tape{+size, nodes, slots, clones} = tape
  sum_taken(size, nodes, clones, a, Ops.take(slots, offset_of(size, a)))

def sum(+a: Nat) -> Build(Nat):
  tape => sum_tape(a, tape)

# The sweep pops one node per step, newest first. `grads` is aligned with the
# nodes still on the tape; an empty slot means no contribution yet.
# Input values and gradients move to `leaves`, oldest first.

def Leaf() -> Type:
  Ops.Slot() & Ops.Slot()

def shape_put(slots: List<&1, Ops.Slot()>, +offset: Nat, pair: Core.Dynamic & Shape.Shape) -> List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>:
  (value, shape) = pair
  (Ops.put(slots, offset, value), Some{shape})

def shape_taken(+offset: Nat, taken: List<&1, Ops.Slot()> & Ops.Slot()) -> List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>:
  (slots, slot) = taken
  match slot:
    case None{}:
      (slots, None{})
    case Some{value}:
      shape_put(slots, offset, Core.shape(value))

def shape_at(slots: List<&1, Ops.Slot()>, +offset: Nat) -> List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>:
  shape_taken(offset, Ops.take(slots, offset))

def rebuild(slots: List<&1, Ops.Slot()>, aux: Aux, result: R(List<&1, Ops.Slot()>)) -> R(Sweep):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{grads}:
      Done{Sweep{slots, grads, aux}}

def deliver(slots: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, grads: List<&1, Ops.Slot()>, result: R(Core.Dynamic)) -> R(Sweep):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{delta}:
      rebuild(slots, aux, Ops.accumulate(grads, target, delta))

def add_shaped(grads: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, g: Core.Dynamic,
  shaped: List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>) -> R(Sweep):
  (slots, shape) = shaped
  match shape:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{+target_shape}:
      deliver(slots, aux, target, grads, Ops.unbroadcast(g, target_shape))

# d(a+b)/da: g, reduced to a's shape.
def add_side(sweep: Sweep, +target: Nat, g: Core.Dynamic) -> R(Sweep):
  Sweep{slots, grads, aux} = sweep
  add_shaped(grads, aux, target, g, shape_at(slots, target))

def mul_product(slots: List<&1, Ops.Slot()>, grads: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, +target_shape: Shape.Shape, +other: Nat,
  result: R((Core.Dynamic & Core.Dynamic) & Core.Dynamic)) -> R(Sweep):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{triple}:
      (operands, product) = triple
      (g, value) = operands
      deliver(Ops.put(slots, other, value), aux, target, grads, Ops.unbroadcast(product, target_shape))

def mul_borrowed(grads: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, +target_shape: Shape.Shape, +other: Nat, g: Core.Dynamic,
  taken: List<&1, Ops.Slot()> & Ops.Slot()) -> R(Sweep):
  (slots, slot) = taken
  match slot:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{value}:
      mul_product(slots, grads, aux, target, target_shape, other, Ops.mul(g, value))

def mul_shaped(grads: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, +other: Nat, g: Core.Dynamic,
  shaped: List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>) -> R(Sweep):
  (slots, shape) = shaped
  match shape:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{+target_shape}:
      mul_borrowed(grads, aux, target, target_shape, other, g, Ops.take(slots, other))

# d(a*b)/da: g * b, reduced to a's shape.
def mul_side(sweep: Sweep, +target: Nat, +other: Nat, g: Core.Dynamic) -> R(Sweep):
  Sweep{slots, grads, aux} = sweep
  mul_shaped(grads, aux, target, other, g, shape_at(slots, target))

def sum_filled(grads: List<&1, Ops.Slot()>, aux: Aux, +target: Nat, value: F32, shaped: List<&1, Ops.Slot()> & Maybe<&2, Shape.Shape>) -> R(Sweep):
  (slots, shape) = shaped
  match shape:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{+target_shape}:
      deliver(slots, aux, target, grads, Ops.fill(target_shape, value))

def sum_shaped(sweep: Sweep, +target: Nat, value: F32) -> R(Sweep):
  Sweep{slots, grads, aux} = sweep
  sum_filled(grads, aux, target, value, shape_at(slots, target))

# d(sum a)/da: the scalar cotangent broadcast to a's shape.
def sum_side(sweep: Sweep, +target: Nat, result: R(Core.Dynamic & F32)) -> R(Sweep):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (g, value) = pair
      sum_shaped(sweep, target, value)

def then_side(result: R(Sweep), k: Sweep -> R(Sweep)) -> R(Sweep):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{sweep}:
      k(sweep)

def two_cotangents(sweep: Sweep, g: Core.Dynamic) -> Sweep & (Core.Dynamic & Core.Dynamic):
  Sweep{slots, grads, aux} = sweep
  Aux{clones, leaves} = aux
  (Sweep{slots, grads, Aux{1n+clones, leaves}}, Core.clone_dynamic(g))

def mul_back(+a: Nat, +b: Nat, pair: Sweep & (Core.Dynamic & Core.Dynamic)) -> R(Sweep):
  (sweep, gs) = pair
  (g1, g2) = gs
  then_side(mul_side(sweep, a, b, g1), next => mul_side(next, b, a, g2))

def add_back(+a: Nat, +b: Nat, pair: Sweep & (Core.Dynamic & Core.Dynamic)) -> R(Sweep):
  (sweep, gs) = pair
  (g1, g2) = gs
  then_side(add_side(sweep, a, g1), next => add_side(next, b, g2))

def contribute(node: Node, +below: Nat, sweep: Sweep, g: Core.Dynamic) -> R(Sweep):
  match node:
    case Input{}:
      Fail{Ops.Missing{}}
    case Add{+a, +b}:
      add_back(offset_of(below, a), offset_of(below, b), two_cotangents(sweep, g))
    case Mul{+a, +b}:
      mul_back(offset_of(below, a), offset_of(below, b), two_cotangents(sweep, g))
    case Sum{+a}:
      sum_side(sweep, offset_of(below, a), Ops.scalar(g))

def add_leaf(sweep: Sweep, value: Ops.Slot(), grad: Ops.Slot()) -> Sweep:
  Sweep{slots, grads, aux} = sweep
  Aux{clones, leaves} = aux
  Sweep{slots, grads, Aux{clones, (value, grad) <> leaves}}

def step_node(node: Node, +below: Nat, value: Ops.Slot(), grad: Ops.Slot(), sweep: Sweep) -> R(Sweep):
  match node grad:
    case Input{} _:
      Done{add_leaf(sweep, value, grad)}
    case _ None{}:
      Done{sweep}
    case _ Some{g}:
      contribute(node, below, sweep, g)

def step_pop(node: Node, +below: Nat, aux: Aux, slots: List<&1, Ops.Slot()>, grads: List<&1, Ops.Slot()>) -> R(Sweep):
  match slots grads:
    case Con{value, older_slots} Con{grad, older_grads}:
      step_node(node, below, value, grad, Sweep{older_slots, older_grads, aux})
    case _ _:
      Fail{Ops.Missing{}}

def step(node: Node, +below: Nat, sweep: Sweep) -> R(Sweep):
  Sweep{slots, grads, aux} = sweep
  step_pop(node, below, aux, slots, grads)

def backward(fuel: Nat, +size: Nat, nodes: List<&2, Node>, state: R(Sweep)) -> R(Sweep):
  match fuel nodes state:
    case _ _ Fail{e}:
      Fail{e}
    case 1n+rest Con{node, older} Done{sweep}:
      backward(rest, Nat.sub(size, 1n), older, step(node, Nat.sub(size, 1n), sweep))
    case _ _ Done{sweep}:
      Done{sweep}

def empties(n: Nat) -> List<&1, Ops.Slot()>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      None{} <> empties(p)

def outcome(value: Core.Dynamic, +saved: Nat, forward_clones: Nat, result: R(Sweep)) -> R(Outcome):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{sweep}:
      Sweep{slots, grads, aux} = sweep
      Aux{clones, leaves} = aux
      Done{Outcome{value, leaves, forward_clones, clones, saved}}

def seeded(value: Core.Dynamic, +size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, forward_clones: Nat,
  seed: R(Core.Dynamic)) -> R(Outcome):
  match seed:
    case Fail{e}:
      Fail{e}
    case Done{one}:
      outcome(value, size, forward_clones,
        backward(size, size, nodes, Done{Sweep{slots, Some{one} <> empties(Nat.sub(size, 1n)), Aux{0n, Nil{}}}}))

def output_shaped(+size: Nat, nodes: List<&2, Node>, slots: List<&1, Ops.Slot()>, clones: Nat, pair: Core.Dynamic & Shape.Shape) -> R(Outcome):
  (value, +shape) = pair
  seeded(value, size, nodes, slots, clones, Ops.fill(shape, 1.0))

def output_taken(+size: Nat, nodes: List<&2, Node>, clones: Nat, taken: List<&1, Ops.Slot()> & Ops.Slot()) -> R(Outcome):
  (slots, slot) = taken
  match slot:
    case None{}:
      Fail{Ops.Missing{}}
    case Some{value}:
      output_shaped(size, nodes, slots, clones, Core.shape(value))

def run_tape(tape: Tape) -> R(Outcome):
  Tape{+size, nodes, slots, clones} = tape
  output_taken(size, nodes, clones, Ops.take(slots, 0n))

def run_built(result: R(Tape & Nat)) -> R(Outcome):
  match result:
    case Fail{e}:
      Fail{e}
    case Done{pair}:
      (tape, output) = pair
      run_tape(tape)

def run(program: Build(Nat)) -> R(Outcome):
  run_built(program(Tape{0n, Nil{}, Nil{}, 0n}))
