import Base
import ./pullback.bend as P

def case1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T) -> P.Gen(P.V(T)):
  do P.Gen<P.V(T)>:
    x : P.V(T) <- P.input(~T, ~plus, x)
    y : P.V(T) <- P.share(~T, ~plus, x, zero, x1 => x23 =>
      P.share(~T, ~plus, x23, zero, x2 => x3 =>
        P.Gen.pure(P.V(T), P.add(~T, ~plus, P.mul(~T, ~times, x1, x2), x3))))
    return y

def case2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T) -> P.Gen(P.V(T)):
  do P.Gen<P.V(T)>:
    x : P.V(T) <- P.input(~T, ~plus, x)
    u : P.V(T) <- P.share(~T, ~plus, x, zero, x1 => x2 => P.Gen.pure(P.V(T), P.mul(~T, ~times, x1, x2)))
    y : P.V(T) <- P.share(~T, ~plus, u, zero, u1 => u2 => P.Gen.pure(P.V(T), P.add(~T, ~plus, u1, u2)))
    return y

def case3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x1: T, x2: T) -> P.Gen(P.V(T)):
  do P.Gen<P.V(T)>:
    a : P.V(T) <- P.input(~T, ~plus, x1)
    b : P.V(T) <- P.input(~T, ~plus, x2)
    return P.mul(~T, ~times, a, b)

def run1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T, +one: T) -> T & List<&2, T>:
  P.run(T, [0n], zero, one, case1(~T, ~plus, ~times, x, zero))

def run2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x: T, +zero: T, +one: T) -> T & List<&2, T>:
  P.run(T, [0n], zero, one, case2(~T, ~plus, ~times, x, zero))

def run3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, x1: T, x2: T, +zero: T, +one: T) -> T & List<&2, T>:
  P.run(T, [0n, 1n], zero, one, case3(~T, ~plus, ~times, x1, x2))

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
    show("pullback case1", run1(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("pullback case2", run2(~F32, ~F32.add, ~F32.mul, 3.0, 0.0, 1.0))
    show("pullback case3", run3(~F32, ~F32.add, ~F32.mul, 3.0, 3.0, 0.0, 1.0))
    return Unit{}
