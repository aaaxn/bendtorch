import Base
import ./pullback-LAWS.bend as Laws

def add_zero(x: Nat) -> {Nat.add(x, 0n) == x : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def add_succ(x: Nat, +y: Nat) -> {Nat.add(x, 1n+y) == 1n+Nat.add(x, y) : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, y) : {1n+Nat.add(p, 1n+y) == 1n+_ : Nat}
      {==}

def Laws.case1_at_3():
  {==}

def Laws.case2_at_3():
  {==}

def Laws.case3_distinct_inputs():
  {==}

def add_one(x: Nat) -> {Nat.add(x, 1n) == 1n+x : Nat}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      %add_one(p) : {1n+Nat.add(p, 1n) == 1n+_ : Nat}
      {==}

def double_succ(+x: Nat) -> {Nat.add(x, Nat.add(x, 1n)) == 1n+Nat.add(x, x) : Nat}:
  %add_succ(x, x) : {Nat.add(x, Nat.add(x, 1n)) == _ : Nat}
  %add_one(x) : {Nat.add(x, Nat.add(x, 1n)) == Nat.add(x, _) : Nat}
  {==}

def Laws.case1_any_x(x):
  %double_succ(x) : {(Nat.add(Nat.mul(x, x), x), [Nat.add(Nat.add(x, 0n), Nat.add(Nat.add(x, 0n), 1n))]) == (Nat.add(Nat.mul(x, x), x), [_]) : Nat & List<&2, Nat>}
  %add_zero(x) : {(Nat.add(Nat.mul(x, x), x), [Nat.add(Nat.add(x, 0n), Nat.add(Nat.add(x, 0n), 1n))]) == (Nat.add(Nat.mul(x, x), x), [Nat.add(_, Nat.add(_, 1n))]) : Nat & List<&2, Nat>}
  {==}
