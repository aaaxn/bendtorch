import Base

import ./backward.bend as Backward

# Runs a script of tape operations through the public API of backward.bend and
# prints one line for each step. Forge pushes an entry without any check, so
# the gate can build the malformed tapes that backward must refuse.

type Step is Data:
  Leaf{word: U32}
  Add{a: Nat, b: Nat}
  Sub{a: Nat, b: Nat}
  Mul{a: Nat, b: Nat}
  Neg{a: Nat}
  Run{output: Nat}
  ZeroGrad{}
  Read{id: Nat}
  Forge{node: Backward.Node, word: U32}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def error(error: Backward.Error) -> IO(Unit):
  IO.print("S3:error:" ++ U32.show(Backward.error_code(error)))

def recorded(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> IO(Unit):
  (tape, result) = pair
  match result:
    case Fail{failure}:
      do IO<Unit>:
        error(failure)
        rec(tape)
    case Done{id}:
      do IO<Unit>:
        IO.print("S3:id:" ++ Nat.show(id))
        rec(tape)

def leafed(rec: Backward.Tape -> IO(Unit), pair: Backward.Tape & Nat) -> IO(Unit):
  (tape, id) = pair
  do IO<Unit>:
    IO.print("S3:id:" ++ Nat.show(id))
    rec(tape)

def ran(+tape: Backward.Tape, rec: Backward.Tape -> IO(Unit), result: Result<&2, &2, Backward.Error, Backward.Tape>) -> IO(Unit):
  match result:
    case Fail{failure}:
      do IO<Unit>:
        error(failure)
        rec(tape)
    case Done{after}:
      do IO<Unit>:
        IO.print("S3:backward")
        rec(after)

def print_value(result: Result<&2, &2, Backward.Error, F32>) -> IO(Unit):
  match result:
    case Fail{failure}:
      IO.print("S3:value:error:" ++ U32.show(Backward.error_code(failure)))
    case Done{value}:
      IO.print("S3:value:" ++ U32.show(F32.bits(value)))

def print_grad(result: Result<&2, &2, Backward.Error, Maybe<&2, F32>>) -> IO(Unit):
  match result:
    case Fail{failure}:
      IO.print("S3:grad:error:" ++ U32.show(Backward.error_code(failure)))
    case Done{None{}}:
      IO.print("S3:grad:none")
    case Done{Some{value}}:
      IO.print("S3:grad:" ++ U32.show(F32.bits(value)))

def forged(tape: Backward.Tape, node: Backward.Node, word: U32) -> Backward.Tape:
  Backward.Tape{entries} = tape
  Backward.Tape{Backward.Entry{node, from_bits(word), None{}} <> entries}

def step(instruction: Step, +tape: Backward.Tape, rec: Backward.Tape -> IO(Unit)) -> IO(Unit):
  match instruction:
    case Leaf{word}:
      leafed(rec, Backward.leaf(tape, from_bits(word)))
    case Add{a, b}:
      recorded(rec, Backward.add(tape, a, b))
    case Sub{a, b}:
      recorded(rec, Backward.sub(tape, a, b))
    case Mul{a, b}:
      recorded(rec, Backward.mul(tape, a, b))
    case Neg{a}:
      recorded(rec, Backward.neg(tape, a))
    case Run{output}:
      ran(tape, rec, Backward.backward(tape, output))
    case ZeroGrad{}:
      do IO<Unit>:
        IO.print("S3:zero_grad")
        rec(Backward.zero_grad(tape))
    case Read{+id}:
      do IO<Unit>:
        print_value(Backward.value(tape, id))
        print_grad(Backward.grad(tape, id))
        rec(tape)
    case Forge{node, word}:
      do IO<Unit>:
        IO.print("S3:forged")
        rec(forged(tape, node, word))

def script(steps: List<&2, Step>, tape: Backward.Tape) -> IO(Unit):
  match steps:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{head, rest}:
      step(head, tape, after => script(rest, after))
