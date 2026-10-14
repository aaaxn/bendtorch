import Base
import ../../src/tensor/core.bend as Core
import ./tensor-ops.bend as Ops
import ./tape-tensor.bend as Tape

# Case 4 on the tensor tape, y = sum(W*x + b) with W:[2,3], x:[3], b:[3].
# Case 1 again on rank-0 tensors, to exercise the x*x operand clone.

def case4_program(w: Core.Dynamic, x: Core.Dynamic, b: Core.Dynamic) -> Tape.Build(Nat):
  do Tape.Build<Nat>:
    +w : Nat <- Tape.input(w)
    +x : Nat <- Tape.input(x)
    +b : Nat <- Tape.input(b)
    wx : Nat <- Tape.mul(w, x)
    z : Nat <- Tape.add(wx, b)
    y : Nat <- Tape.sum(z)
    return y

def case1_program(x: Core.Dynamic) -> Tape.Build(Nat):
  do Tape.Build<Nat>:
    +x : Nat <- Tape.input(x)
    sq : Nat <- Tape.mul(x, x)
    y : Nat <- Tape.add(sq, x)
    return y

# Negative control: an operand id with no stored value answers Missing (code 400).
def dangling_program(x: Core.Dynamic) -> Tape.Build(Nat):
  do Tape.Build<Nat>:
    +x : Nat <- Tape.input(x)
    Tape.mul(x, 7n)

def dangling() -> Tape.R(Tape.Outcome):
  do Result<&2, &1, Ops.Error, Tape.Outcome>:
    x : Core.Dynamic <- Ops.construct([], [3.0])
    Tape.run(dangling_program(x))

def case4() -> Tape.R(Tape.Outcome):
  do Result<&2, &1, Ops.Error, Tape.Outcome>:
    w : Core.Dynamic <- Ops.construct([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x : Core.Dynamic <- Ops.construct([3], [0.5, F32.neg(1.0), 2.0])
    b : Core.Dynamic <- Ops.construct([3], [1.0, F32.neg(1.0), 0.5])
    Tape.run(case4_program(w, x, b))

def case1() -> Tape.R(Tape.Outcome):
  do Result<&2, &1, Ops.Error, Tape.Outcome>:
    x : Core.Dynamic <- Ops.construct([], [3.0])
    Tape.run(case1_program(x))

def show_slot(label: String, slot: Ops.Slot()) -> IO(Unit):
  match slot:
    case None{}:
      IO.print("no-gradient")
    case Some{t}:
      Ops.show_tensor(label, t)

def show_leaves(leaves: List<&1, Tape.Leaf()>) -> IO(Unit):
  match leaves:
    case Nil{}:
      do IO<Unit>:
        return Unit{}
    case Con{leaf, rest}:
      (value, grad) = leaf
      do IO<Unit>:
        show_slot("input", value)
        show_slot("grad", grad)
        show_leaves(rest)

def show(label: String, result: Tape.R(Tape.Outcome)) -> IO(Unit):
  match result:
    case Fail{e}:
      IO.print(label ++ " error " ++ U32.show(Ops.error_code(e)))
    case Done{outcome}:
      Tape.Outcome{value, leaves, forward_clones, backward_clones, saved} = outcome
      do IO<Unit>:
        IO.print(label)
        Ops.show_tensor("value", value)
        show_leaves(leaves)
        IO.print("forward-clones " ++ U32.show(U32.from_nat(forward_clones)))
        IO.print("backward-clones " ++ U32.show(U32.from_nat(backward_clones)))
        IO.print("saved " ++ U32.show(U32.from_nat(saved)))

def main() -> IO(Unit):
  do IO<Unit>:
    show("tape-tensor case4", case4())
    show("tape-tensor case1", case1())
    show("tape-tensor dangling-id", dangling())
    return Unit{}
