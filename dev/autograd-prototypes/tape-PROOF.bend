import Base
import ./tape-LAWS.bend as Laws

def add_zero(x: Nat) -> {Nat.add(x, 0n) == x : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def Laws.case1_at_3():
  {==}

def Laws.case2_at_3():
  {==}

def Laws.case3_distinct_inputs():
  {==}

def Laws.case1_any_x(x):
  %add_zero(x) : {(Nat.add(Nat.mul(x, x), x), [1n+Nat.add(Nat.add(x, 0n), Nat.add(x, 0n))]) == (Nat.add(Nat.mul(x, x), x), [1n+Nat.add(_, _)]) : Nat & List<&2, Nat>}
  {==}
