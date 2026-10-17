import Base

import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/harness.bend as Tensor
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward
import ./sgd.bend as Sgd

# Runs a script of tape operations through the public API of
# tensor-backward.bend and prints one line for each step, and every word of
# each value and gradient it reads.

type Step is Data:
  Leaf{dimensions: List<&2, U32>, words: List<&2, U32>}
  Constant{dimensions: List<&2, U32>, words: List<&2, U32>}
  Add{a: Nat, b: Nat}
  Sub{a: Nat, b: Nat}
  Mul{a: Nat, b: Nat}
  Matmul{a: Nat, b: Nat}
  MatmulNt{a: Nat, b: Nat}
  BatchedMatmul{a: Nat, b: Nat}
  Linear{x: Nat, w: Nat, bias: Nat}
  Sum{a: Nat}
  SumDims{a: Nat, axes: List<&2, U32>, keepdim: Bool}
  Relu{a: Nat}
  MseLoss{a: Nat, b: Nat}
  CrossEntropy{a: Nat, targets: List<&2, U32>}
  Run{output: Nat}
  ZeroGrad{}
  Step{lr: U32, params: List<&2, Nat>}
  Read{id: Nat}
  ForgeAdd{a: Nat, b: Nat, dimensions: List<&2, U32>, words: List<&2, U32>}
  ForgeSum{a: Nat, dimensions: List<&2, U32>, words: List<&2, U32>}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def values(words: List<&2, U32>) -> List<&2, F32>:
  match words:
    case Nil{}:
      Nil{}
    case Con{word, rest}:
      from_bits(word) <> values(rest)

def error(error: Backward.Error) -> IO(Unit):
  IO.print("autograd:error:" ++ U32.show(Backward.error_code(error)))

def recorded(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> IO(Unit):
  (tape, result) = pair
  match result:
    case Fail{failure}:
      do IO<Unit>:
        error(failure)
        rec(tape)
    case Done{id}:
      do IO<Unit>:
        IO.print("autograd:id:" ++ Nat.show(id))
        rec(tape)

def leafed(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Nat) -> IO(Unit):
  (tape, id) = pair
  do IO<Unit>:
    IO.print("autograd:id:" ++ Nat.show(id))
    rec(tape)

def constructed(tape: Backward.Tape, rec: Backward.Tape -> IO(Unit), result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{failure}:
      do IO<Unit>:
        IO.print("autograd:error:" ++ U32.show(Core.error_code(failure)))
        rec(tape)
    case Done{value}:
      leafed(rec, Backward.leaf(tape, value))

def constant_constructed(tape: Backward.Tape, rec: Backward.Tape -> IO(Unit), result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{failure}:
      do IO<Unit>:
        IO.print("autograd:error:" ++ U32.show(Core.error_code(failure)))
        rec(tape)
    case Done{value}:
      leafed(rec, Backward.constant(tape, value))

def ran(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>) -> IO(Unit):
  (tape, result) = pair
  match result:
    case Fail{failure}:
      do IO<Unit>:
        error(failure)
        rec(tape)
    case Done{unit}:
      do IO<Unit>:
        IO.print("autograd:backward")
        rec(tape)

def stepped(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>) -> IO(Unit):
  (tape, result) = pair
  match result:
    case Fail{failure}:
      do IO<Unit>:
        IO.print("autograd:error:" ++ U32.show(Sgd.error_code(failure)))
        rec(tape)
    case Done{unit}:
      do IO<Unit>:
        IO.print("autograd:step")
        rec(tape)

def print_value(result: Result<&2, &1, Backward.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{failure}:
      IO.print("autograd:value:error:" ++ U32.show(Backward.error_code(failure)))
    case Done{value}:
      do IO<Unit>:
        IO.print("autograd:value")
        Tensor.snapshot(value)

def print_grad(result: Result<&2, &1, Backward.Error, Maybe<&1, Core.Dynamic>>) -> IO(Unit):
  match result:
    case Fail{failure}:
      IO.print("autograd:grad:error:" ++ U32.show(Backward.error_code(failure)))
    case Done{None{}}:
      IO.print("autograd:grad:none")
    case Done{Some{gradient}}:
      do IO<Unit>:
        IO.print("autograd:grad")
        Tensor.snapshot(gradient)

def graded(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &1, Backward.Error, Maybe<&1, Core.Dynamic>>) -> IO(Unit):
  (tape, result) = pair
  do IO<Unit>:
    print_grad(result)
    rec(tape)

def read_grad(+id: Nat, rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>) -> IO(Unit):
  (tape, result) = pair
  do IO<Unit>:
    print_value(result)
    graded(rec, Backward.grad(tape, id))

def forged_value(tape: Backward.Tape, node: Backward.Node, value: Core.Dynamic) -> Backward.Tape:
  Backward.Tape{entries} = tape
  Core.Packed{+shape, tensor} = value
  Backward.Tape{Backward.Entry{node, shape, Core.Packed{shape, tensor}, Backward.Unsaved{}, Gradient.Unset{}} <> entries}

def forged_node(adds: Bool, +a: Nat, +b: Nat, +shape: Shape.Shape) -> Backward.Node:
  match adds:
    case True{}:
      Backward.Add{a, b, shape, shape}
    case False{}:
      Backward.Sum{a, shape}

def forge_shaped(tape: Backward.Tape, adds: Bool, +a: Nat, +b: Nat, pair: Core.Dynamic & Shape.Shape) -> Backward.Tape:
  (value, +shape) = pair
  forged_value(tape, forged_node(adds, a, b, shape), value)

# Pushes an entry without any check, so a script can build a tape whose node
# names itself, a newer node or a missing node.
def forge(tape: Backward.Tape, adds: Bool, +a: Nat, +b: Nat, rec: Backward.Tape -> IO(Unit),
  result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{failure}:
      do IO<Unit>:
        IO.print("autograd:error:" ++ U32.show(Core.error_code(failure)))
        rec(tape)
    case Done{value}:
      do IO<Unit>:
        IO.print("autograd:forged")
        rec(forge_shaped(tape, adds, a, b, Core.shape(value)))

def step(instruction: Step, tape: Backward.Tape, rec: Backward.Tape -> IO(Unit)) -> IO(Unit):
  match instruction:
    case Leaf{dimensions, words}:
      constructed(tape, rec, Core.construct(dimensions, values(words)))
    case Constant{dimensions, words}:
      constant_constructed(tape, rec, Core.construct(dimensions, values(words)))
    case Add{a, b}:
      recorded(rec, Backward.add(tape, a, b))
    case Sub{a, b}:
      recorded(rec, Backward.sub(tape, a, b))
    case Mul{a, b}:
      recorded(rec, Backward.mul(tape, a, b))
    case Matmul{a, b}:
      recorded(rec, Backward.matmul(tape, a, b))
    case MatmulNt{a, b}:
      recorded(rec, Backward.matmul_nt(tape, a, b))
    case BatchedMatmul{a, b}:
      recorded(rec, Backward.batched_matmul(tape, a, b))
    case Linear{x, w, bias}:
      recorded(rec, Backward.linear(tape, x, w, bias))
    case Sum{a}:
      recorded(rec, Backward.sum(tape, a))
    case SumDims{a, axes, keepdim}:
      recorded(rec, Backward.sum_dims(tape, a, axes, keepdim))
    case Relu{a}:
      recorded(rec, Backward.relu(tape, a))
    case MseLoss{a, b}:
      recorded(rec, Backward.mse_loss(tape, a, b))
    case CrossEntropy{a, targets}:
      recorded(rec, Backward.cross_entropy(tape, a, targets))
    case Run{output}:
      ran(rec, Backward.backward(tape, output))
    case ZeroGrad{}:
      do IO<Unit>:
        IO.print("autograd:zero_grad")
        rec(Backward.zero_grad(tape))
    case Step{lr, params}:
      stepped(rec, Sgd.step(tape, from_bits(lr), params))
    case Read{+id}:
      read_grad(id, rec, Backward.value(tape, id))
    case ForgeAdd{a, b, dimensions, words}:
      forge(tape, True{}, a, b, rec, Core.construct(dimensions, values(words)))
    case ForgeSum{+a, dimensions, words}:
      forge(tape, False{}, a, a, rec, Core.construct(dimensions, values(words)))

def script(steps: List<&2, Step>, tape: Backward.Tape) -> IO(Unit):
  match steps:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{head, rest}:
      step(head, tape, after => script(rest, after))
