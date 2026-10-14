import Base

type Dual<-T: Data> is Data:
  Dual{value: T, tangent: T}

def add(~T: Data, ~plus: T -> T -> T, a: Dual<T>, b: Dual<T>) -> Dual<T>:
  Dual{av, ad} = a
  Dual{bv, bd} = b
  Dual{plus(av, bv), plus(ad, bd)}

def mul(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, a: Dual<T>, b: Dual<T>) -> Dual<T>:
  Dual{+av, ad} = a
  Dual{+bv, bd} = b
  Dual{times(av, bv), plus(times(ad, bv), times(av, bd))}

def value(-T: Data, d: Dual<T>) -> T:
  Dual{v, t} = d
  v

def tangent(-T: Data, d: Dual<T>) -> T:
  Dual{v, t} = d
  t

def case1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +seed: T) -> Dual<T>:
  +dx = {Dual{x, seed} : Dual<T>}
  add(~T, ~plus, mul(~T, ~plus, ~times, dx, dx), dx)

def case2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +seed: T) -> Dual<T>:
  +dx = {Dual{x, seed} : Dual<T>}
  +u = {mul(~T, ~plus, ~times, dx, dx) : Dual<T>}
  add(~T, ~plus, u, u)

def case3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x1: T, +x2: T, +s1: T, +s2: T) -> Dual<T>:
  mul(~T, ~plus, ~times, Dual{x1, s1}, Dual{x2, s2})

# Each runner answers (value, gradients), one forward pass per input.
def run1(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +zero: T, +one: T) -> T & List<&2, T>:
  (value(T, case1(~T, ~plus, ~times, x, zero)), [tangent(T, case1(~T, ~plus, ~times, x, one))])

def run2(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x: T, +zero: T, +one: T) -> T & List<&2, T>:
  (value(T, case2(~T, ~plus, ~times, x, zero)), [tangent(T, case2(~T, ~plus, ~times, x, one))])

def run3(~T: Data, ~plus: T -> T -> T, ~times: T -> T -> T, +x1: T, +x2: T, +zero: T, +one: T) -> T & List<&2, T>:
  (value(T, case3(~T, ~plus, ~times, x1, x2, zero, zero)),
    [tangent(T, case3(~T, ~plus, ~times, x1, x2, one, zero)), tangent(T, case3(~T, ~plus, ~times, x1, x2, zero, one))])
