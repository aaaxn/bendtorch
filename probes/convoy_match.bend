# Known Bend2 bug fixed upstream in 8828cc19: on 947db722, `convoy_bind` dropped
# a dependent parameter from a match's arms, so --check-only passed and --verdict failed.
import Base

type Owner is Type:
  Owner{n: Nat}

type Box<-original: Owner> is Type:
  Box{owner: Owner, -same: {owner == original : Owner}}

def convoy(-original: Owner, owner: Owner, -same: {owner == original : Owner}, flag: Bool) -> Box<original>:
  match flag:
    case False{}:
      Box{owner, same}
    case True{}:
      Box{owner, same}

def main() -> U32:
  0
