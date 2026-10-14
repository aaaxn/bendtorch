import Base

def capacity(d: Nat) -> Nat:
  match d:
    case 0n:
      1n
    case 1n+p:
      +n = capacity(p)
      Nat.add(n, n)

def Path(d: Nat) -> Data:
  match d:
    case 0n:
      Unit
    case 1n+p:
      Step<p>

type Step<-d: Nat> is Data:
  Step{bit: Bool, tail: Path(d)}

def offset_step(b: Bool, half: Nat, tail: Nat) -> Nat:
  match b:
    case False{}:
      tail
    case True{}:
      Nat.add(half, tail)

def offset(d: Nat, path: Path(d)) -> Nat:
  match d:
    case 0n:
      0n
    case 1n++p:
      Step{b, rest} = path
      offset_step(b, capacity(p), offset(p, rest))

def same(d: Nat, a: Path(d), b: Path(d)) -> Bool:
  match d:
    case 0n:
      True{}
    case 1n+p:
      Step{ab, ar} = a
      Step{bb, br} = b
      Bool.and(Bool.not(Bool.xor(ab, bb)), same(p, ar, br))

def Balanced(-T: Type, d: Nat, array: Array<T>) -> Type:
  match d array:
    case 0n ALeaf{x}:
      Unit
    case 0n ANode{xs, ys}:
      Empty
    case 1n+p ALeaf{x}:
      Empty
    case 1n++p ANode{xs, ys}:
      Balanced(T, p, xs) & Balanced(T, p, ys)

def observe(-T: Data, d: Nat, array: Array<T>, path: Path(d)) -> Maybe<&2, T>:
  match d array:
    case 0n ALeaf{x}:
      Some{x}
    case 0n ANode{xs, ys}:
      None{}
    case 1n+p ALeaf{x}:
      None{}
    case 1n+p ANode{xs, ys}:
      Step{bit, rest} = path
      match bit:
        case False{}:
          observe(T, p, xs, rest)
        case True{}:
          observe(T, p, ys, rest)

def select(-T: Data, b: Bool, value: T, old: Maybe<&2, T>) -> Maybe<&2, T>:
  match b:
    case False{}:
      old
    case True{}:
      Some{value}
