import Base

# Probe, not a law file: whole-run facts about case 4 and rank-0 case 1, stated
# by evaluation (input count, forward clones, backward clones, saved values).
# `--check-only` accepts all three. `--verdict` fails: the BendTT kernel reports
# "out of fuel" (its FUEL budget is 400000000 steps) on each one separately.
# Upstream: bendlang/bend #1322 and #1193; see kernel-fuel-adds.bend.
import ./tensor-ops.bend as Ops
import ./tape-tensor.bend as Tape
import ./tape-tensor-cases.bend as TapeCases
import ./pullback-tensor.bend as Pullback
import ./pullback-tensor-cases.bend as PullbackCases

def tape_shape(r: Tape.R(Tape.Outcome)) -> Maybe<&2, +List<Nat>>:
  match r:
    case Fail{e}:
      None{}
    case Done{o}:
      Tape.Outcome{value, leaves, fc, bc, saved} = o
      Some{[List.length(&1, Tape.Leaf(), leaves), fc, bc, saved]}

def pullback_shape(r: Pullback.R(Pullback.Outcome)) -> Maybe<&2, +List<Nat>>:
  match r:
    case Fail{e}:
      None{}
    case Done{o}:
      Pullback.Outcome{value, inputs, grads, fc, bc, saved} = o
      Some{[List.length(&1, Ops.Slot(), grads), fc, bc, saved]}

def tape_case4() -> {tape_shape(TapeCases.case4()) == Some{[3n, 0n, 2n, 6n]} : Maybe<&2, +List<Nat>>}:
  {==}

def tape_case1() -> {tape_shape(TapeCases.case1()) == Some{[1n, 1n, 2n, 3n]} : Maybe<&2, +List<Nat>>}:
  {==}

def pullback_case4() -> {pullback_shape(PullbackCases.case4()) == Some{[3n, 3n, 2n, 2n]} : Maybe<&2, +List<Nat>>}:
  {==}
