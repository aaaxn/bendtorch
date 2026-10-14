import Base
import ../../src/tensor/core.bend as Core
import ./tensor-ops.bend as Ops
import ./pullback-tensor.bend as P

# Case 4 with pullback closures, y = sum(W*x + b) with W:[2,3], x:[3], b:[3].
# Case 1 again on rank-0 tensors: x is used three times, so it is shared twice.

def case4_program(w: Core.Dynamic, x: Core.Dynamic, b: Core.Dynamic) -> P.Gen(P.V()):
  do P.Gen<P.V()>:
    w : P.V() <- P.input(w)
    x : P.V() <- P.input(x)
    b : P.V() <- P.input(b)
    wx : P.V() <- P.mul(w, x)
    z : P.V() <- P.add(wx, b)
    P.sum(z)

def case1_program(x: Core.Dynamic) -> P.Gen(P.V()):
  do P.Gen<P.V()>:
    x : P.V() <- P.input(x)
    P.share(x, x1 => x23 =>
      P.share(x23, x2 => x3 =>
        do P.Gen<P.V()>:
          sq : P.V() <- P.mul(x1, x2)
          P.add(sq, x3)))

def case4() -> P.R(P.Outcome):
  do Result<&2, &1, Ops.Error, P.Outcome>:
    w : Core.Dynamic <- Ops.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x : Core.Dynamic <- Ops.construct([3], [0.5, F32.neg(1.0), 2.0])
    b : Core.Dynamic <- Ops.construct([3], [1.0, F32.neg(1.0), 0.5])
    P.run(case4_program(w, x, b))

def case1() -> P.R(P.Outcome):
  do Result<&2, &1, Ops.Error, P.Outcome>:
    x : Core.Dynamic <- Ops.construct([], [3.0])
    P.run(case1_program(x))

def show_slot(label: String, slot: Ops.Slot()) -> IO(Unit):
  match slot:
    case None{}:
      IO.print("no-gradient")
    case Some{t}:
      Ops.show_tensor(label, t)

def show_pairs(fuel: Nat, inputs: List<&1, Core.Dynamic>, grads: List<&1, Ops.Slot()>) -> IO(Unit):
  match fuel inputs grads:
    case 1n+rest Con{value, more} Con{grad, others}:
      do IO<Unit>:
        Ops.show_tensor("input", value)
        show_slot("grad", grad)
        show_pairs(rest, more, others)
    case _ _ _:
      do IO<Unit>:
        return Unit{}

def show(label: String, result: P.R(P.Outcome)) -> IO(Unit):
  match result:
    case Fail{e}:
      IO.print(label ++ " error " ++ U32.show(Ops.error_code(e)))
    case Done{outcome}:
      P.Outcome{value, inputs, grads, forward_clones, backward_clones, saved} = outcome
      do IO<Unit>:
        IO.print(label)
        Ops.show_tensor("value", value)
        show_pairs(3n, inputs, grads)
        IO.print("forward-clones " ++ U32.show(U32.from_nat(forward_clones)))
        IO.print("backward-clones " ++ U32.show(U32.from_nat(backward_clones)))
        IO.print("saved " ++ U32.show(U32.from_nat(saved)))

def main() -> IO(Unit):
  do IO<Unit>:
    show("pullback-tensor case4", case4())
    show("pullback-tensor case1", case1())
    return Unit{}
