import Base

# Prototype A (scalar part): an explicit reverse tape.
# The tape is one owned store. Each entry holds a node and its forward value;
# operands are node ids (Nat), never values, so equal values cannot merge.
# Arithmetic is a template: laws instantiate it at U32 and run it at F32.

type Node is Data:
  Input{}
  Add{a: Nat, b: Nat}
  Mul{a: Nat, b: Nat}

type Entry<-T: Data> is Data:
  Entry{node: Node, value: T}

# Entries are newest first: the head is node `size - 1`.
type Tape<-T: Data> is Data:
  Tape{size: Nat, entries: List<&2, Entry<T>>}

def at(-T: Data, xs: List<&2, T>, +offset: Nat, +fallback: T) -> T:
  match xs offset:
    case Nil{} _:
      fallback
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      at(T, t, p, fallback)

# The value of node `id` in a newest-first list of `size` items.
def by_id(-T: Data, +size: Nat, +xs: List<&2, T>, +id: Nat, +fallback: T) -> T:
  at(T, xs, Nat.sub(Nat.sub(size, 1n), id), fallback)

def Build(-T: Data, -A: Type) -> Type:
  Tape<T> -> Tape<T> & A

def Build.pure(-T: Data, -A: Type, x: A) -> Build(T, A):
  tape => (tape, x)

def Build.bind.go(-T: Data, -A: Type, -B: Type, k: A -> Build(T, B), result: Tape<T> & A) -> Tape<T> & B:
  (tape, x) = result
  k(x)(tape)

def Build.bind(-T: Data, -A: Type, -B: Type, m: Build(T, A), k: A -> Build(T, B)) -> Build(T, B):
  tape => Build.bind.go(T, A, B, k, m(tape))

def value_at(-T: Data, entries: List<&2, Entry<T>>, +offset: Nat, +fallback: T) -> T:
  match entries offset:
    case Nil{} _:
      fallback
    case Con{Entry{node, value}, rest} 0n:
      value
    case Con{entry, rest} 1n+p:
      value_at(T, rest, p, fallback)

def push(-T: Data, node: Node, value: T, tape: Tape<T>) -> Tape<T> & Nat:
  Tape{+size, entries} = tape
  (Tape{1n+size, Entry{node, value} <> entries}, size)

def input(-T: Data, value: T) -> Build(T, Nat):
  tape => push(T, Input{}, value, tape)

# Operand `id` sits at offset `size - 1 - id` from the head.
def operand(-T: Data, +size: Nat, +entries: List<&2, Entry<T>>, +id: Nat, +zero: T) -> T:
  value_at(T, entries, Nat.sub(Nat.sub(size, 1n), id), zero)

def add_at(~T: Data, ~plus: T -> T -> T, +a: Nat, +b: Nat, +zero: T, tape: Tape<T>) -> Tape<T> & Nat:
  Tape{+size, +entries} = tape
  push(T, Add{a, b}, plus(operand(T, size, entries, a, zero), operand(T, size, entries, b, zero)), Tape{size, entries})

def mul_at(~T: Data, ~times: T -> T -> T, +a: Nat, +b: Nat, +zero: T, tape: Tape<T>) -> Tape<T> & Nat:
  Tape{+size, +entries} = tape
  push(T, Mul{a, b}, times(operand(T, size, entries, a, zero), operand(T, size, entries, b, zero)), Tape{size, entries})

def add(~T: Data, ~plus: T -> T -> T, +a: Nat, +b: Nat, +zero: T) -> Build(T, Nat):
  tape => add_at(~T, ~plus, a, b, zero, tape)

def mul(~T: Data, ~times: T -> T -> T, +a: Nat, +b: Nat, +zero: T) -> Build(T, Nat):
  tape => mul_at(~T, ~times, a, b, zero, tape)

# Gradients are a list aligned with the entries, newest first.
def bump(~T: Data, ~plus: T -> T -> T, grads: List<&2, T>, +offset: Nat, +delta: T) -> List<&2, T>:
  match grads offset:
    case Nil{} _:
      Nil{}
    case Con{g, rest} 0n:
      plus(g, delta) <> rest
    case Con{g, rest} 1n+p:
      g <> bump(~T, ~plus, rest, p, delta)

# `top` is the id at the head of `older` and `grads`.
def contribute(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, node: Node, +g: T, +top: Nat,
  +older: List<&2, Entry<T>>, +zero: T, grads: List<&2, T>) -> List<&2, T>:
  match node:
    case Input{}:
      grads
    case Add{+a, +b}:
      bump(~T, ~plus, bump(~T, ~plus, grads, Nat.sub(top, a), g), Nat.sub(top, b), g)
    case Mul{+a, +b}:
      bump(~T, ~plus,
        bump(~T, ~plus, grads, Nat.sub(top, a), times(g, value_at(T, older, Nat.sub(top, b), zero))),
        Nat.sub(top, b), times(g, value_at(T, older, Nat.sub(top, a), zero)))

def backward(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, entries: List<&2, Entry<T>>,
  grads: List<&2, T>, +top: Nat, +zero: T) -> List<&2, T>:
  match entries grads:
    case Nil{} _:
      Nil{}
    case Con{Entry{node, value}, +older} Nil{}:
      Nil{}
    case Con{Entry{node, value}, +older} Con{+g, rest}:
      g <> backward(~T, ~plus, ~times, older,
        contribute(~T, ~plus, ~times, node, g, Nat.sub(top, 1n), older, zero, rest), Nat.sub(top, 1n), zero)

def zeros(-T: Data, n: Nat, +zero: T) -> List<&2, T>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      zero <> zeros(T, p, zero)

# Seeds d(output)/d(output) = 1 at the head and runs the reverse sweep.
def gradients(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, tape: Tape<T>, +zero: T, +one: T) -> Tape<T> & List<&2, T>:
  Tape{+size, +entries} = tape
  (Tape{size, entries},
    backward(~T, ~plus, ~times, entries, one <> zeros(T, Nat.sub(size, 1n), zero), Nat.sub(size, 1n), zero))


def empty(-T: Data) -> Tape<T>:
  Tape{0n, Nil{}}

# Runs a program that answers (inputs, output id) and reads the output value and the
# gradient of every input id.
def read_grads(-T: Data, +size: Nat, +grads: List<&2, T>, ids: List<&2, Nat>, +zero: T) -> List<&2, T>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{+id, rest}:
      by_id(T, size, grads, id, zero) <> read_grads(T, size, grads, rest, zero)

def finish(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, ids: List<&2, Nat>, +output: Nat, +zero: T, +one: T,
  swept: Tape<T> & List<&2, T>) -> T & List<&2, T>:
  (tape, +grads) = swept
  Tape{+size, +entries} = tape
  (value_at(T, entries, Nat.sub(Nat.sub(size, 1n), output), zero), read_grads(T, size, grads, ids, zero))

def sweep(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, ids: List<&2, Nat>, +output: Nat, +zero: T, +one: T,
  tape: Tape<T>) -> T & List<&2, T>:
  finish(~T, ~plus, ~times, ids, output, zero, one, gradients(~T, ~plus, ~times, tape, zero, one))

def run_built(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +zero: T, +one: T,
  built: Tape<T> & (List<&2, Nat> & Nat)) -> T & List<&2, T>:
  (tape, result) = built
  (ids, output) = result
  sweep(~T, ~plus, ~times, ids, output, zero, one, tape)

def run(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +zero: T, +one: T,
  program: Build(T, List<&2, Nat> & Nat)) -> T & List<&2, T>:
  run_built(~T, ~plus, ~times, zero, one, program(empty(T)))
