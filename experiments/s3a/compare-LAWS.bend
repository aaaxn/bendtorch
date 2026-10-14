import Base
import ./tape-cases.bend as TapeCases
import ./pullback-cases.bend as PullbackCases
import ./dual-cases.bend as DualCases
import ./tape-LAWS.bend as TapeLaws

# Cross-design laws: each reverse design agrees with the forward reference for every x.

law tape_matches_dual_case1:
  for +x: Nat
  {TapeCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n) == DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n) : Nat & List<&2, Nat>}

law pullback_matches_dual_case1:
  for +x: Nat
  {PullbackCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n) == DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n) : Nat & List<&2, Nat>}
