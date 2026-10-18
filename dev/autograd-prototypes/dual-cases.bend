import Base
import ./dual.bend as Dual

def run1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +zero: T, +one: T) -> T & List<&2, T>:
  Dual.run1(~T, ~plus, ~times, x, zero, one)

def run2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +zero: T, +one: T) -> T & List<&2, T>:
  Dual.run2(~T, ~plus, ~times, x, zero, one)

def run3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x1: T, +x2: T, +zero: T, +one: T) -> T & List<&2, T>:
  Dual.run3(~T, ~plus, ~times, x1, x2, zero, one)

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
    show("dual case1", run1(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("dual case2", run2(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("dual case3", run3(~F32, ~F32.add, ~F32.mul, 3.0, 3.0, 0.0, 1.0))
    return Unit{}
