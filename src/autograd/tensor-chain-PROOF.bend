import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward
import ./tensor-chain-LAWS.bend as Laws

# What the pull side holds for a push-side list of pending slots: each pending
# slot x, the one with k older slots, is what take gathers for id k from the
# log into the starting slot w, and the rest of the log serves the older slots.
# Every witness is erased; only the equations are evidence.
type Cell<-x: Gradient.Slot, -rest: List<&1, Gradient.Slot>, -slots: List<&1, Gradient.Slot>, -below: Nat, -log: Laws.Log()> is Data:
  Cell{-w: Gradient.Slot, -older: List<&1, Gradient.Slot>, -left: Laws.Log(),
    at: {slots == w <> older : List<&1, Gradient.Slot>},
    took: {Laws.take(log, below, w) == (Done{x}, left) : Laws.Taken()},
    tail: Pulls(rest, older, below, left)}

def Pulls(pending: List<&1, Gradient.Slot>, -slots: List<&1, Gradient.Slot>, +count: Nat, -log: Laws.Log()) -> Data:
  match pending count:
    case Nil{} count:
      {slots == Nil{} : List<&1, Gradient.Slot>}
    case Con{x, rest} 0n:
      {slots == x <> rest : List<&1, Gradient.Slot>}
    case Con{x, rest} 1n+below:
      Cell<x, rest, slots, below, log>

# With no entry left, the pull side gathers exactly the pending slots.
def gathered(pending: List<&1, Gradient.Slot>, +count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log(),
  pulls: Pulls(pending, slots, count, log)) -> {Laws.gather(slots, count, log) == Done{pending} : Laws.Slots()}:
  match pending count:
    case Nil{} count:
      %Equal.sym(List<&1, Gradient.Slot>, slots, Nil{}, pulls) : {Laws.gather(_, count, log) == Done{Nil{}} : Laws.Slots()}
      {==}
    case Con{x, rest} 0n:
      %Equal.sym(List<&1, Gradient.Slot>, slots, x <> rest, pulls) : {Laws.gather(_, 0n, log) == Done{x <> rest} : Laws.Slots()}
      {==}
    case Con{x, rest} 1n+below:
      Cell{w, older, left, at, took, tail} = pulls
      %Equal.sym(List<&1, Gradient.Slot>, slots, w <> older, at) : {Laws.gather(_, 1n+below, log) == Done{x <> rest} : Laws.Slots()}
      %Equal.sym(Laws.Taken(), Laws.take(log, below, w), (Done{x}, left), took) :
        {Laws.gather_step(_, more => Laws.gather(older, below, more)) == Done{x <> rest} : Laws.Slots()}
      Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.prepend(Gradient.Slot, x, result),
        Laws.gather(older, below, left), Done{rest}, gathered(rest, below, older, left, tail))

# The starting slots, handed back, with the pull side of an empty log.
type Started<-pending: List<&1, Gradient.Slot>, -count: Nat> is Type:
  Started{copy: List<&1, Gradient.Slot>, same: {copy == pending : List<&1, Gradient.Slot>},
    pulls: Pulls(copy, pending, count, Nil{})}

def restarted(x: Gradient.Slot, -rest: List<&1, Gradient.Slot>, +below: Nat, out: Started<rest, below>) ->
  Started<x <> rest, 1n+below>:
  Started{copy, same, pulls} = out
  Started{x <> copy, Equal.cong(List<&1, Gradient.Slot>, List<&1, Gradient.Slot>, more => x <> more, copy, rest, same),
    Cell{x, rest, Nil{}, {==}, {==}, pulls}}

def started(pending: List<&1, Gradient.Slot>, +count: Nat) -> Started<pending, count>:
  match pending count:
    case Nil{} count:
      Started{Nil{}, {==}, {==}}
    case Con{x, rest} 0n:
      Started{x <> rest, {==}, {==}}
    case Con{x, rest} 1n+below:
      restarted(x, rest, below, started(rest, below))

def Pushed(state: Laws.Slots(), -slots: List<&1, Gradient.Slot>, +count: Nat, -log: Laws.Log()) -> Data:
  match state:
    case Fail{error}:
      Unit
    case Done{pending}:
      Pulls(pending, slots, count, log)

# One contribution bumped into the push side, the new state handed back.
type Bumped<-pending: List<&1, Gradient.Slot>, -count: Nat, -target: Nat, -shape: Shape.Shape, -gradient: Core.Dynamic, -same: Bool,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log()> is Type:
  Bumped{state: Laws.Slots(), bumped: {state == Backward.bump(pending, count, target, shape, gradient, same) : Laws.Slots()},
    pushed: Pushed(state, slots, count, Backward.Contribution{target, shape, gradient} <> log)}

type Hit<-outcome: Gradient.Outcome, -rest: List<&1, Gradient.Slot>, -below: Nat, -target: Nat, -shape: Shape.Shape, -gradient: Core.Dynamic,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log()> is Type:
  Hit{state: Laws.Slots(), placed: {state == Backward.placed(rest, outcome) : Laws.Slots()},
    pushed: Pushed(state, slots, 1n+below, Backward.Contribution{target, shape, gradient} <> log)}

def hit(outcome: Gradient.Outcome, rest: List<&1, Gradient.Slot>, +below: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(), -w: Gradient.Slot, -older: List<&1, Gradient.Slot>, -left: Laws.Log(),
  at: {slots == w <> older : List<&1, Gradient.Slot>}, tail: Pulls(rest, older, below, left),
  took: {Laws.take(Backward.Contribution{target, shape, gradient} <> log, below, w) == (Laws.settled(outcome), left) : Laws.Taken()}) ->
  Hit<outcome, rest, below, target, shape, gradient, slots, log>:
  match outcome:
    case Gradient.Accumulated{slot}:
      Hit{Done{slot <> rest}, {==}, Cell{w, older, left, at, took, tail}}
    case Gradient.Refused{slot, contribution, error}:
      Hit{Fail{Backward.GradientError{error}}, {==}, Unit{}}

def hit_bumped(-x: Gradient.Slot, -rest: List<&1, Gradient.Slot>, +below: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(), out: Hit<Gradient.accumulate(shape, x, gradient), rest, below, target, shape, gradient, slots, log>) ->
  Bumped<x <> rest, 1n+below, target, shape, gradient, True{}, slots, log>:
  Hit{state, placed, pushed} = out
  Bumped{state, placed, pushed}

def took_hit(+below: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic, -log: Laws.Log(), -w: Gradient.Slot,
  -x: Gradient.Slot, -left: Laws.Log(), took: {Laws.take(log, below, w) == (Done{x}, left) : Laws.Taken()},
  decided: {Nat.is_eq(below, target) == True{} : Bool}) ->
  {Laws.take(Backward.Contribution{target, shape, gradient} <> log, below, w) == (Laws.settled(Gradient.accumulate(shape, x, gradient)), left) : Laws.Taken()}:
  %Equal.sym(Bool, Nat.is_eq(below, target), True{}, decided) :
    {Laws.sorted(target, shape, gradient, _, Laws.take(log, below, w)) == (Laws.settled(Gradient.accumulate(shape, x, gradient)), left) : Laws.Taken()}
  %Equal.sym(Laws.Taken(), Laws.take(log, below, w), (Done{x}, left), took) :
    {Laws.sorted(target, shape, gradient, True{}, _) == (Laws.settled(Gradient.accumulate(shape, x, gradient)), left) : Laws.Taken()}
  {==}

def took_miss(+below: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic, -log: Laws.Log(), -w: Gradient.Slot,
  -x: Gradient.Slot, -left: Laws.Log(), took: {Laws.take(log, below, w) == (Done{x}, left) : Laws.Taken()},
  decided: {Nat.is_eq(below, target) == False{} : Bool}) ->
  {Laws.take(Backward.Contribution{target, shape, gradient} <> log, below, w) == (Done{x}, Backward.Contribution{target, shape, gradient} <> left) : Laws.Taken()}:
  %Equal.sym(Bool, Nat.is_eq(below, target), False{}, decided) :
    {Laws.sorted(target, shape, gradient, _, Laws.take(log, below, w)) == (Done{x}, Backward.Contribution{target, shape, gradient} <> left) : Laws.Taken()}
  %Equal.sym(Laws.Taken(), Laws.take(log, below, w), (Done{x}, left), took) :
    {Laws.sorted(target, shape, gradient, False{}, _) == (Done{x}, Backward.Contribution{target, shape, gradient} <> left) : Laws.Taken()}
  {==}

def missed(x: Gradient.Slot, -rest: List<&1, Gradient.Slot>, +below: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(), -w: Gradient.Slot, -older: List<&1, Gradient.Slot>, -left: Laws.Log(),
  at: {slots == w <> older : List<&1, Gradient.Slot>},
  took: {Laws.take(Backward.Contribution{target, shape, gradient} <> log, below, w) == (Done{x}, Backward.Contribution{target, shape, gradient} <> left) : Laws.Taken()},
  out: Bumped<rest, below, target, shape, gradient, Backward.at_head(below, target), older, left>) ->
  Bumped<x <> rest, 1n+below, target, shape, gradient, False{}, slots, log>:
  Bumped{state, bumped, pushed} = out
  match state:
    case Fail{error}:
      Bumped{Fail{error}, Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.prepend(Gradient.Slot, x, result), Fail{error},
        Backward.bump(rest, below, target, shape, gradient, Backward.at_head(below, target)), bumped), Unit{}}
    case Done{pending}:
      Bumped{Done{x <> pending}, Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.prepend(Gradient.Slot, x, result), Done{pending},
        Backward.bump(rest, below, target, shape, gradient, Backward.at_head(below, target)), bumped),
        Cell{w, older, Backward.Contribution{target, shape, gradient} <> left, at, took, pushed}}

# Bumping a contribution into the push side keeps the pull side in step, with
# the contribution at the head of the log.
def bump_pulls(pending: List<&1, Gradient.Slot>, +count: Nat, +target: Nat, +shape: Shape.Shape, gradient: Core.Dynamic, same: Bool,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(), pulls: Pulls(pending, slots, count, log),
  decided: {Backward.at_head(count, target) == same : Bool}) -> Bumped<pending, count, target, shape, gradient, same, slots, log>:
  match pending count same:
    case Nil{} count same:
      Bumped{Done{Nil{}}, {==}, pulls}
    case Con{x, rest} 0n same:
      Bumped{Done{x <> rest}, {==}, pulls}
    case Con{x, rest} 1n+below True{}:
      Cell{w, older, left, at, took, tail} = pulls
      hit_bumped(x, rest, below, target, shape, gradient, slots, log,
        hit(Gradient.accumulate(shape, x, gradient), rest, below, target, shape, gradient, slots, log, w, older, left, at, tail,
          took_hit(below, target, shape, gradient, log, w, x, left, took, decided)))
    case Con{x, rest} 1n+below False{}:
      Cell{w, older, left, at, took, tail} = pulls
      missed(x, rest, below, target, shape, gradient, slots, log, w, older, left, at,
        took_miss(below, target, shape, gradient, log, w, x, left, took, decided),
        bump_pulls(rest, below, target, shape, gradient, Backward.at_head(below, target), older, left, tail, {==}))

# A whole list of contributions delivered into any push-side state.
type Delivered<-items: Laws.Log(), -state: Laws.Slots(), -count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log()> is Type:
  Delivered{after: Laws.Slots(), delivered: {after == Backward.deliver(items, count, state) : Laws.Slots()},
    pushed: Pushed(after, slots, count, List.reverse.go(&1, Backward.Contribution, items, log))}

def relay(-rest: Laws.Log(), +count: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic, -pending: List<&1, Gradient.Slot>,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(), -state: Laws.Slots(),
  bumped: {state == Backward.bump(pending, count, target, shape, gradient, Backward.at_head(count, target)) : Laws.Slots()},
  out: Delivered<rest, state, count, slots, Backward.Contribution{target, shape, gradient} <> log>) ->
  Delivered<Backward.Contribution{target, shape, gradient} <> rest, Done{pending}, count, slots, log>:
  Delivered{after, delivered, pushed} = out
  Delivered{after, Equal.trans(Laws.Slots(), after, Backward.deliver(rest, count, state),
    Backward.deliver(rest, count, Backward.bump(pending, count, target, shape, gradient, Backward.at_head(count, target))), delivered,
    Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.deliver(rest, count, result), state,
      Backward.bump(pending, count, target, shape, gradient, Backward.at_head(count, target)), bumped)), pushed}

def rebump(-rest: Laws.Log(), +count: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic, -pending: List<&1, Gradient.Slot>,
  -slots: List<&1, Gradient.Slot>, -log: Laws.Log(),
  out: Bumped<pending, count, target, shape, gradient, Backward.at_head(count, target), slots, log>,
  ih: @state: Laws.Slots() -> @-slots2: List<&1, Gradient.Slot> -> @-log2: Laws.Log() -> @pushed2: Pushed(state, slots2, count, log2) ->
    Delivered<rest, state, count, slots2, log2>) ->
  Delivered<Backward.Contribution{target, shape, gradient} <> rest, Done{pending}, count, slots, log>:
  Bumped{state, bumped, pushed} = out
  relay(rest, count, target, shape, gradient, pending, slots, log, state, bumped,
    ih(state, slots, Backward.Contribution{target, shape, gradient} <> log, pushed))

def delivered(items: Laws.Log(), state: Laws.Slots(), +count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log(),
  pushed: Pushed(state, slots, count, log)) -> Delivered<items, state, count, slots, log>:
  match items state:
    case Nil{} state:
      Delivered{state, {==}, pushed}
    case Con{Backward.Contribution{+target, +shape, gradient}, rest} Fail{error}:
      Delivered{Fail{error}, {==}, Unit{}}
    case Con{Backward.Contribution{+target, +shape, gradient}, rest} Done{pending}:
      rebump(rest, count, target, shape, gradient, pending, slots, log,
        bump_pulls(pending, count, target, shape, gradient, Backward.at_head(count, target), slots, log, pushed, {==}),
        after => slots2 => log2 => pushed2 => delivered(rest, after, count, slots2, log2, pushed2))

# What a sweep from a push-side state does, with the pull side's answer when
# it succeeds: done and gradients are the sweep's own outputs.
type Stop<-entries: List<&1, Backward.Entry>, -error: Backward.Error, -count: Nat> is Data:
  Stop{-done: List<&1, Backward.Entry>, swept: {Backward.sweep(entries, Fail{error}, count) == (done, Fail{error}) : Backward.Swept()}}

type Run<-entries: List<&1, Backward.Entry>, -pending: List<&1, Gradient.Slot>, -count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log()> is Data:
  Kept{-done: List<&1, Backward.Entry>, -gradients: List<&1, Gradient.Slot>,
    swept: {Backward.sweep(entries, Done{pending}, count) == (done, Done{gradients}) : Backward.Swept()},
    pulled: {Laws.chained(entries, slots, count, log) == Done{gradients} : Laws.Slots()}}
  Stopped{-done: List<&1, Backward.Entry>, -error: Backward.Error,
    swept: {Backward.sweep(entries, Done{pending}, count) == (done, Fail{error}) : Backward.Swept()}}

def Out(-entries: List<&1, Backward.Entry>, state: Laws.Slots(), +count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log()) -> Data:
  match state:
    case Fail{error}:
      Stop<entries, error, count>
    case Done{pending}:
      Run<entries, pending, count, slots, log>

# The same for the entry whose VJP pass is known.
type Fired<-node: Backward.Node, -shape: Shape.Shape, -value: Core.Dynamic, -slot: Gradient.Slot, -rest: List<&1, Backward.Entry>,
  -more: List<&1, Gradient.Slot>, -below: Nat, -older: List<&1, Gradient.Slot>, -left: Laws.Log(), -pass: Backward.Pass> is Data:
  FiredKept{-done: List<&1, Backward.Entry>, -gradients: List<&1, Gradient.Slot>,
    swept: {Backward.passed(node, shape, value, slot, below, more, state => Backward.sweep(rest, state, below), pass) ==
      (done, Done{gradients}) : Backward.Swept()},
    pulled: {Laws.fired(pass, left, next => Laws.chained(rest, older, below, next)) == Done{gradients} : Laws.Slots()}}
  FiredStopped{-done: List<&1, Backward.Entry>, -error: Backward.Error,
    swept: {Backward.passed(node, shape, value, slot, below, more, state => Backward.sweep(rest, state, below), pass) ==
      (done, Fail{error}) : Backward.Swept()}}

def fired_stop(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -g: Core.Dynamic, -rest: List<&1, Backward.Entry>, -more: List<&1, Gradient.Slot>, +below: Nat, -older: List<&1, Gradient.Slot>,
  -left: Laws.Log(), -items: Laws.Log(), -error: Backward.Error,
  delivered: {Fail{error} == Backward.deliver(items, below, Done{more}) : Laws.Slots()}, out: Stop<rest, error, below>) ->
  Fired<node, shape, value, slot, rest, more, below, older, left, Backward.Pass{saved, g, Done{items}}>:
  Stop{done, swept} = out
  FiredStopped{Backward.Entry{node, shape, value, saved, slot} <> done, error,
    Equal.trans(Backward.Swept(),
      Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g},
        Backward.sweep(rest, Backward.deliver(items, below, Done{more}), below)),
      Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, Fail{error}, below)),
      Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, (done, Fail{error})),
      Equal.cong(Laws.Slots(), Backward.Swept(),
        state => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, state, below)),
        Backward.deliver(items, below, Done{more}), Fail{error}, Equal.sym(Laws.Slots(), Fail{error}, Backward.deliver(items, below, Done{more}), delivered)),
      Equal.cong(Backward.Swept(), Backward.Swept(),
        pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, pair),
        Backward.sweep(rest, Fail{error}, below), (done, Fail{error}), swept))}

def fired_go(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -g: Core.Dynamic, -rest: List<&1, Backward.Entry>, -more: List<&1, Gradient.Slot>, +below: Nat, -older: List<&1, Gradient.Slot>,
  -left: Laws.Log(), -items: Laws.Log(), -pending: List<&1, Gradient.Slot>,
  delivered: {Done{pending} == Backward.deliver(items, below, Done{more}) : Laws.Slots()},
  out: Run<rest, pending, below, older, List.reverse.go(&1, Backward.Contribution, items, left)>) ->
  Fired<node, shape, value, slot, rest, more, below, older, left, Backward.Pass{saved, g, Done{items}}>:
  match out:
    case Kept{done, gradients, swept, pulled}:
      FiredKept{Backward.Entry{node, shape, value, saved, slot} <> done, Gradient.Held{g} <> gradients,
        Equal.trans(Backward.Swept(),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g},
            Backward.sweep(rest, Backward.deliver(items, below, Done{more}), below)),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, Done{pending}, below)),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, (done, Done{gradients})),
          Equal.cong(Laws.Slots(), Backward.Swept(),
            state => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, state, below)),
            Backward.deliver(items, below, Done{more}), Done{pending},
            Equal.sym(Laws.Slots(), Done{pending}, Backward.deliver(items, below, Done{more}), delivered)),
          Equal.cong(Backward.Swept(), Backward.Swept(),
            pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, pair),
            Backward.sweep(rest, Done{pending}, below), (done, Done{gradients}), swept)),
        Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.prepend(Gradient.Slot, Gradient.Held{g}, result),
          Laws.chained(rest, older, below, List.reverse.go(&1, Backward.Contribution, items, left)), Done{gradients}, pulled)}
    case Stopped{done, error, swept}:
      FiredStopped{Backward.Entry{node, shape, value, saved, slot} <> done, error,
        Equal.trans(Backward.Swept(),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g},
            Backward.sweep(rest, Backward.deliver(items, below, Done{more}), below)),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, Done{pending}, below)),
          Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, (done, Fail{error})),
          Equal.cong(Laws.Slots(), Backward.Swept(),
            state => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, Backward.sweep(rest, state, below)),
            Backward.deliver(items, below, Done{more}), Done{pending},
            Equal.sym(Laws.Slots(), Done{pending}, Backward.deliver(items, below, Done{more}), delivered)),
          Equal.cong(Backward.Swept(), Backward.Swept(),
            pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, pair),
            Backward.sweep(rest, Done{pending}, below), (done, Fail{error}), swept))}

def fail_fired(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -g: Core.Dynamic, -rest: List<&1, Backward.Entry>, -more: List<&1, Gradient.Slot>, +below: Nat, -older: List<&1, Gradient.Slot>,
  -left: Laws.Log(), -error: Backward.Error, out: Stop<rest, error, below>) ->
  Fired<node, shape, value, slot, rest, more, below, older, left, Backward.Pass{saved, g, Fail{error}}>:
  Stop{done, swept} = out
  FiredStopped{Backward.Entry{node, shape, value, saved, slot} <> done, error,
    Equal.cong(Backward.Swept(), Backward.Swept(),
      pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Held{g}, pair),
      Backward.sweep(rest, Fail{error}, below), (done, Fail{error}), swept)}

def done_fired(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -g: Core.Dynamic, -rest: List<&1, Backward.Entry>, -more: List<&1, Gradient.Slot>, +below: Nat, -older: List<&1, Gradient.Slot>,
  -left: Laws.Log(), -items: Laws.Log(), sent: Delivered<items, Done{more}, below, older, left>,
  ih: @state: Laws.Slots() -> @-slots2: List<&1, Gradient.Slot> -> @+count2: Nat -> @-log2: Laws.Log() ->
    @pushed2: Pushed(state, slots2, count2, log2) -> Out(rest, state, count2, slots2, log2)) ->
  Fired<node, shape, value, slot, rest, more, below, older, left, Backward.Pass{saved, g, Done{items}}>:
  Delivered{after, delivered, pushed} = sent
  match after:
    case Fail{error}:
      fired_stop(node, shape, value, saved, slot, g, rest, more, below, older, left, items, error, delivered,
        ih(Fail{error}, older, below, List.reverse.go(&1, Backward.Contribution, items, left), pushed))
    case Done{pending}:
      fired_go(node, shape, value, saved, slot, g, rest, more, below, older, left, items, pending, delivered,
        ih(Done{pending}, older, below, List.reverse.go(&1, Backward.Contribution, items, left), pushed))

def held_step(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -slot: Gradient.Slot, -rest: List<&1, Backward.Entry>,
  more: List<&1, Gradient.Slot>, +below: Nat, -older: List<&1, Gradient.Slot>, -left: Laws.Log(), tail: Pulls(more, older, below, left),
  pass: Backward.Pass,
  ih: @state: Laws.Slots() -> @-slots2: List<&1, Gradient.Slot> -> @+count2: Nat -> @-log2: Laws.Log() ->
    @pushed2: Pushed(state, slots2, count2, log2) -> Out(rest, state, count2, slots2, log2)) ->
  Fired<node, shape, value, slot, rest, more, below, older, left, pass>:
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      fail_fired(node, shape, value, saved, slot, g, rest, more, below, older, left, error, ih(Fail{error}, older, below, left, Unit{}))
    case Done{items}:
      done_fired(node, shape, value, saved, slot, g, rest, more, below, older, left, items,
        delivered(items, Done{more}, below, older, left, tail), ih)

def unset_step(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -rest: List<&1, Backward.Entry>, -more: List<&1, Gradient.Slot>, +below: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log(),
  -w: Gradient.Slot, -older: List<&1, Gradient.Slot>, -left: Laws.Log(),
  at: {slots == w <> older : List<&1, Gradient.Slot>}, took: {Laws.take(log, below, w) == (Done{Gradient.Unset{}}, left) : Laws.Taken()},
  out: Run<rest, more, below, older, left>) ->
  Run<Backward.Entry{node, shape, value, saved, slot} <> rest, Gradient.Unset{} <> more, 1n+below, slots, log>:
  match out:
    case Kept{done, gradients, swept, pulled}:
      Kept{Backward.Entry{node, shape, value, saved, slot} <> done, Gradient.Unset{} <> gradients,
        Equal.cong(Backward.Swept(), Backward.Swept(),
          pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Unset{}, pair),
          Backward.sweep(rest, Done{more}, below), (done, Done{gradients}), swept),
        Equal.trans(Laws.Slots(), Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, slots, 1n+below, log),
          Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, w <> older, 1n+below, log),
          Done{Gradient.Unset{} <> gradients},
          Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(),
            list => Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, list, 1n+below, log), slots, w <> older, at),
          Equal.trans(Laws.Slots(), Laws.reached(node, saved, Laws.take(log, below, w), next => Laws.chained(rest, older, below, next)),
            Laws.reached(node, saved, (Done{Gradient.Unset{}}, left), next => Laws.chained(rest, older, below, next)),
            Done{Gradient.Unset{} <> gradients},
            Equal.cong(Laws.Taken(), Laws.Slots(), taken => Laws.reached(node, saved, taken, next => Laws.chained(rest, older, below, next)),
              Laws.take(log, below, w), (Done{Gradient.Unset{}}, left), took),
            Equal.cong(Laws.Slots(), Laws.Slots(), result => Backward.prepend(Gradient.Slot, Gradient.Unset{}, result),
              Laws.chained(rest, older, below, left), Done{gradients}, pulled)))}
    case Stopped{done, error, swept}:
      Stopped{Backward.Entry{node, shape, value, saved, slot} <> done, error,
        Equal.cong(Backward.Swept(), Backward.Swept(),
          pair => Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Unset{}, pair),
          Backward.sweep(rest, Done{more}, below), (done, Fail{error}), swept)}

def fired_run(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, -saved: Backward.Saved, -slot: Gradient.Slot,
  -rest: List<&1, Backward.Entry>, -g: Core.Dynamic, -more: List<&1, Gradient.Slot>, +below: Nat, -slots: List<&1, Gradient.Slot>,
  -log: Laws.Log(), -w: Gradient.Slot, -older: List<&1, Gradient.Slot>, -left: Laws.Log(),
  at: {slots == w <> older : List<&1, Gradient.Slot>}, took: {Laws.take(log, below, w) == (Done{Gradient.Held{g}}, left) : Laws.Taken()},
  out: Fired<node, shape, value, slot, rest, more, below, older, left, Backward.vjp(node, saved, g)>) ->
  Run<Backward.Entry{node, shape, value, saved, slot} <> rest, Gradient.Held{g} <> more, 1n+below, slots, log>:
  match out:
    case FiredKept{done, gradients, swept, pulled}:
      Kept{done, gradients, swept,
        Equal.trans(Laws.Slots(), Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, slots, 1n+below, log),
          Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, w <> older, 1n+below, log),
          Done{gradients},
          Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(),
            list => Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, list, 1n+below, log), slots, w <> older, at),
          Equal.trans(Laws.Slots(), Laws.reached(node, saved, Laws.take(log, below, w), next => Laws.chained(rest, older, below, next)),
            Laws.reached(node, saved, (Done{Gradient.Held{g}}, left), next => Laws.chained(rest, older, below, next)),
            Done{gradients},
            Equal.cong(Laws.Taken(), Laws.Slots(), taken => Laws.reached(node, saved, taken, next => Laws.chained(rest, older, below, next)),
              Laws.take(log, below, w), (Done{Gradient.Held{g}}, left), took),
            pulled))}
    case FiredStopped{done, error, swept}:
      Stopped{done, error, swept}

def slot_step(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, saved: Backward.Saved, -slot: Gradient.Slot,
  -rest: List<&1, Backward.Entry>, x: Gradient.Slot, more: List<&1, Gradient.Slot>, +below: Nat, -slots: List<&1, Gradient.Slot>,
  -log: Laws.Log(), pushed: Cell<x, more, slots, below, log>,
  ih: @state: Laws.Slots() -> @-slots2: List<&1, Gradient.Slot> -> @+count2: Nat -> @-log2: Laws.Log() ->
    @pushed2: Pushed(state, slots2, count2, log2) -> Out(rest, state, count2, slots2, log2)) ->
  Run<Backward.Entry{node, shape, value, saved, slot} <> rest, x <> more, 1n+below, slots, log>:
  match x:
    case Gradient.Unset{}:
      Cell{w, older, left, at, took, tail} = pushed
      unset_step(node, shape, value, saved, slot, rest, more, below, slots, log, w, older, left, at, took,
        ih(Done{more}, older, below, left, tail))
    case Gradient.Held{g}:
      Cell{w, older, left, at, took, tail} = pushed
      fired_run(node, shape, value, saved, slot, rest, g, more, below, slots, log, w, older, left, at, took,
        held_step(node, shape, value, slot, rest, more, below, older, left, tail, Backward.vjp(node, saved, g), ih))

def stepped(+node: Backward.Node, +shape: Shape.Shape, -value: Core.Dynamic, saved: Backward.Saved, -slot: Gradient.Slot,
  -rest: List<&1, Backward.Entry>, pending: List<&1, Gradient.Slot>, +count: Nat, -slots: List<&1, Gradient.Slot>, -log: Laws.Log(),
  pushed: Pulls(pending, slots, count, log),
  ih: @state: Laws.Slots() -> @-slots2: List<&1, Gradient.Slot> -> @+count2: Nat -> @-log2: Laws.Log() ->
    @pushed2: Pushed(state, slots2, count2, log2) -> Out(rest, state, count2, slots2, log2)) ->
  Run<Backward.Entry{node, shape, value, saved, slot} <> rest, pending, count, slots, log>:
  match pending count:
    case Nil{} count:
      Kept{Backward.Entry{node, shape, value, saved, slot} <> rest, Nil{}, {==},
        Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(),
          list => Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, list, count, log), slots, Nil{}, pushed)}
    case Con{x, more} 0n:
      Kept{Backward.Entry{node, shape, value, saved, slot} <> rest, Nil{}, {==},
        Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(),
          list => Laws.chained(Backward.Entry{node, shape, value, saved, slot} <> rest, list, 0n, log), slots, x <> more, pushed)}
    case Con{x, more} 1n+below:
      slot_step(node, shape, value, saved, slot, rest, x, more, below, slots, log, pushed, ih)

# The sweep from any push-side state, in step with the pull side.
def swept_pulls(entries: List<&1, Backward.Entry>, state: Laws.Slots(), -slots: List<&1, Gradient.Slot>, +count: Nat, -log: Laws.Log(),
  pushed: Pushed(state, slots, count, log)) -> Out(entries, state, count, slots, log):
  match entries state:
    case Nil{} Fail{error}:
      Stop{Nil{}, {==}}
    case Nil{} Done{pending}:
      Kept{Nil{}, pending, {==}, gathered(pending, count, slots, log, pushed)}
    case Con{Backward.Entry{+node, +shape, value, saved, slot}, rest} Fail{error}:
      Stop{Backward.Entry{node, shape, value, saved, slot} <> rest, {==}}
    case Con{Backward.Entry{+node, +shape, value, saved, slot}, rest} Done{pending}:
      stepped(node, shape, value, saved, slot, rest, pending, count, slots, log, pushed,
        after => slots2 => count2 => log2 => pushed2 => swept_pulls(rest, after, slots2, count2, log2, pushed2))

def is_done(result: Laws.Slots()) -> Type:
  match result:
    case Fail{error}:
      Empty
    case Done{slots}:
      Unit

def done_fail(-error: Backward.Error, -gs: List<&1, Gradient.Slot>, clash: {Fail{error} == Done{gs} : Laws.Slots()}) -> Empty:
  %Equal.sym(Laws.Slots(), Fail{error}, Done{gs}, clash) : is_done(_)
  Unit{}

def conclude(-entries: List<&1, Backward.Entry>, -copy: List<&1, Gradient.Slot>, -gradients: List<&1, Gradient.Slot>, +count: Nat,
  -gs: List<&1, Gradient.Slot>, same: {copy == gradients : List<&1, Gradient.Slot>},
  succeeded: {Laws.result(Backward.sweep(entries, Done{gradients}, count)) == Done{gs} : Laws.Slots()},
  out: Run<entries, copy, count, gradients, Nil{}>) -> {Laws.chained(entries, gradients, count, Nil{}) == Done{gs} : Laws.Slots()}:
  match out:
    case Kept{done, found, swept, pulled}:
      Equal.trans(Laws.Slots(), Laws.chained(entries, gradients, count, Nil{}), Done{found}, Done{gs}, pulled,
        Equal.trans(Laws.Slots(), Done{found}, Laws.result(Backward.sweep(entries, Done{copy}, count)), Done{gs},
          Equal.sym(Laws.Slots(), Laws.result(Backward.sweep(entries, Done{copy}, count)), Done{found},
            Equal.cong(Backward.Swept(), Laws.Slots(), pair => Laws.result(pair), Backward.sweep(entries, Done{copy}, count), (done, Done{found}), swept)),
          Equal.trans(Laws.Slots(), Laws.result(Backward.sweep(entries, Done{copy}, count)), Laws.result(Backward.sweep(entries, Done{gradients}, count)),
            Done{gs},
            Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(), list => Laws.result(Backward.sweep(entries, Done{list}, count)), copy, gradients, same),
            succeeded)))
    case Stopped{done, error, swept}:
      Empty.absurd({Laws.chained(entries, gradients, count, Nil{}) == Done{gs} : Laws.Slots()},
        done_fail(error, gs,
          Equal.trans(Laws.Slots(), Fail{error}, Laws.result(Backward.sweep(entries, Done{copy}, count)), Done{gs},
            Equal.sym(Laws.Slots(), Laws.result(Backward.sweep(entries, Done{copy}, count)), Fail{error},
              Equal.cong(Backward.Swept(), Laws.Slots(), pair => Laws.result(pair), Backward.sweep(entries, Done{copy}, count), (done, Fail{error}), swept)),
            Equal.trans(Laws.Slots(), Laws.result(Backward.sweep(entries, Done{copy}, count)), Laws.result(Backward.sweep(entries, Done{gradients}, count)),
              Done{gs},
              Equal.cong(List<&1, Gradient.Slot>, Laws.Slots(), list => Laws.result(Backward.sweep(entries, Done{list}, count)), copy, gradients, same),
              succeeded))))

def started_pulls(entries: List<&1, Backward.Entry>, -gradients: List<&1, Gradient.Slot>, +count: Nat, -gs: List<&1, Gradient.Slot>,
  succeeded: {Laws.result(Backward.sweep(entries, Done{gradients}, count)) == Done{gs} : Laws.Slots()}, init: Started<gradients, count>) ->
  {Laws.chained(entries, gradients, count, Nil{}) == Done{gs} : Laws.Slots()}:
  Started{copy, same, pulls} = init
  conclude(entries, copy, gradients, count, gs, same, succeeded, swept_pulls(entries, Done{copy}, gradients, count, Nil{}, pulls))

def Laws.tensor_sweep_chain_rule(entries, gradients, count, gs, succeeded):
  started_pulls(entries, gradients, count, gs, succeeded, started(gradients, count))
