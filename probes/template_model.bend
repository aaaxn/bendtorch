# Known Bend2 bug on 32d23b43 and release 2.0.37: --check-only passes and
# --verdict fails with "no model for pick~make". The model search fills Pos
# with n = 0 and never tries n = 1, so it finds no Pos, although
# Pos{1n, {==}} exists. With
# Nat.is_le in place of Nat.is_lt, both pass.
import Base

type Pos is Data:
  Pos{n: Nat, positive: {Nat.is_lt(0n, n) == True{} : Bool}}

def pick(~make: Nat -> Pos, +p: Nat) -> Pos:
  make(p)
