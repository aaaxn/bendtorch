import Base
import ./tape.bend as Tape

# Cases 1-3 on the explicit tape. Each program answers (input ids, output id).
# x*x and u+u name one node twice; the sweep adds both contributions to it.

def Program(-T: Data) -> Type:
  Tape.Build(T, List<&2, Nat> & Nat)

def case1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T) -> Program(T):
  do Tape.Build<T, List<&2, Nat> & Nat>:
    +x : Nat <- Tape.input(T, x)
    sq : Nat <- Tape.mul(~T, ~times, x, x, zero)
    y : Nat <- Tape.add(~T, ~plus, sq, x, zero)
    return ([x], y)

def case2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T) -> Program(T):
  do Tape.Build<T, List<&2, Nat> & Nat>:
    +x : Nat <- Tape.input(T, x)
    +u : Nat <- Tape.mul(~T, ~times, x, x, zero)
    y : Nat <- Tape.add(~T, ~plus, u, u, zero)
    return ([x], y)

def case3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x1: T, x2: T, +zero: T) -> Program(T):
  do Tape.Build<T, List<&2, Nat> & Nat>:
    +a : Nat <- Tape.input(T, x1)
    +b : Nat <- Tape.input(T, x2)
    y : Nat <- Tape.mul(~T, ~times, a, b, zero)
    return ([a, b], y)

def run1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T, +one: T) -> T & List<&2, T>:
  Tape.run(~T, ~plus, ~times, zero, one, case1(~T, ~plus, ~times, x, zero))

def run2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T, +one: T) -> T & List<&2, T>:
  Tape.run(~T, ~plus, ~times, zero, one, case2(~T, ~plus, ~times, x, zero))

def run3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x1: T, x2: T, +zero: T, +one: T) -> T & List<&2, T>:
  Tape.run(~T, ~plus, ~times, zero, one, case3(~T, ~plus, ~times, x1, x2, zero))

def show_all(xs: List<&2, F32>) -> IO(Unit):
  match xs:
    case Nil{}:
      do IO<Unit>:
        return Unit{}
    case Con{h, t}:
      do IO<Unit>:
        IO.print(U32.show(F32.bits(h)))
        show_all(t)

def show(label: String, result: F32 & List<&2, F32>) -> IO(Unit):
  (value, grads) = result
  do IO<Unit>:
    IO.print(label)
    IO.print(U32.show(F32.bits(value)))
    show_all(grads)

def main() -> IO(Unit):
  do IO<Unit>:
    show("tape case1", run1(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("tape case2", run2(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("tape case3", run3(~F32, ~F32.add, ~F32.mul, 3.0, 3.0, 0.0, 1.0))
    return Unit{}
