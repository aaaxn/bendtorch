import Base
import ./tape-cases.bend as Cases

law case1_at_3:
  {Cases.run1(~U32, ~U32.add, ~U32.mul, 3, 0, 1) == (12, [7]) : U32 & List<&2, U32>}

law case2_at_3:
  {Cases.run2(~U32, ~U32.add, ~U32.mul, 3, 0, 1) == (18, [12]) : U32 & List<&2, U32>}

law case3_distinct_inputs:
  {Cases.run3(~U32, ~U32.add, ~U32.mul, 3, 3, 0, 1) == (9, [3, 3]) : U32 & List<&2, U32>}

def times(a: Nat, b: Nat) -> Nat:
  Nat.mul(a, b)

def Hand1(+x: Nat) -> Nat & List<&2, Nat>:
  (Nat.add(Nat.mul(x, x), x), [1n+Nat.add(x, x)])

law case1_any_x:
  for +x: Nat
  {Cases.run1(~Nat, ~Nat.add, ~times, x, 0n, 1n) == Hand1(x) : Nat & List<&2, Nat>}
