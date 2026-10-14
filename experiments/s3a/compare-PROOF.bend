import Base
import ./compare-LAWS.bend as Laws
import ./tape-cases.bend as TapeCases
import ./pullback-cases.bend as PullbackCases
import ./dual-cases.bend as DualCases
import ./tape-LAWS.bend as TapeLaws
import ./tape-PROOF.bend as TapeProof
import ./pullback-PROOF.bend as PullbackProof
import ./dual-PROOF.bend as DualProof
import ./pullback-LAWS.bend as PullbackLaws
import ./dual-LAWS.bend as DualLaws

# Composes the per-design proofs: both sides equal the hand derivative.

def Laws.tape_matches_dual_case1(x):
  Equal.trans(Nat & List<&2, Nat>, TapeCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n), TapeLaws.Hand1(x),
    DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n),
    TapeLaws.case1_any_x(x),
    Equal.sym(Nat & List<&2, Nat>, DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n), TapeLaws.Hand1(x),
      DualLaws.case1_any_x(x)))

def Laws.pullback_matches_dual_case1(x):
  Equal.trans(Nat & List<&2, Nat>, PullbackCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n), TapeLaws.Hand1(x),
    DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n),
    PullbackLaws.case1_any_x(x),
    Equal.sym(Nat & List<&2, Nat>, DualCases.run1(~Nat, ~Nat.add, ~TapeLaws.times, x, 0n, 1n), TapeLaws.Hand1(x),
      DualLaws.case1_any_x(x)))
