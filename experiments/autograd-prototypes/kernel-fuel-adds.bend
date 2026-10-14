import Base

# Minimal repro of the kernel fuel limit: three chained rank-0 S2 adds,
# evaluated by a law. `--check-only` accepts it; `--verdict` and `bendtt` on the
# exported file answer "In probe: out of fuel" (bendtt.lean FUEL = 400000000).
# Two chained adds pass both (~8 s in the kernel). Same result on 32d23b4, 63b2c0d, 0e14aa8 and 947db72.
# Known upstream kernel limitation, not a new bug: bendlang/bend #1322 and #1193
# (open); PR #1487 tried to address it and was withdrawn.
import ../../src/tensor/core.bend as Core
import ../../src/tensor/elementwise.bend as Elementwise

def built(r: Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{t}:
      True{}

def step0(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  (inputs, r) = pair
  built(r)

def done1(a: Core.Dynamic, r: Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{t}:
      step0(Elementwise.add(a, t))

def step1(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  (inputs, r) = pair
  (a, b) = inputs
  done1(a, r)

def done2(a: Core.Dynamic, r: Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{t}:
      step1(Elementwise.add(a, t))

def step2(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  (inputs, r) = pair
  (a, b) = inputs
  done2(a, r)



def right(a: Core.Dynamic, r: Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{b}:
      step2(Elementwise.add(a, b))

def left(r: Result<&2, &1, Core.Error, Core.Dynamic>) -> Bool:
  match r:
    case Fail{e}:
      False{}
    case Done{t}:
      right(t, Core.construct([], [2.0]))

def probe() -> {left(Core.construct([], [1.0])) == True{} : Bool}:
  {==}
