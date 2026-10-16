import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward

# The gradient the whole sweep leaves, stated as the chain rule over the tape.
#
# The sweep pushes: it visits entries newest first, and an entry that holds a
# gradient sends its VJP's contributions into the working slots of its
# operands as it goes. The reference below pulls instead: it visits entries
# newest first, keeps every contribution sent so far in a log, newest first,
# and when it reaches the entry with id k it takes out of the log every
# contribution sent to k and accumulates them into k's starting slot, oldest
# first, with Gradient.accumulate. So the gradient of node k is its starting
# slot plus, in order, each contribution its consumers sent it: consumers
# newest first, and within one consumer in operand order. Each contribution is
# that consumer's Backward.vjp at the consumer's own gathered gradient, so the
# value of every contribution is the one its VJP value law states
# (tensor-vjp-LAWS.bend). The order of the float sum is part of the statement.
#
# The reference shares with the sweep only Backward.vjp and
# Gradient.accumulate, each stated by its own laws, and the list helpers
# Backward.prepend and Backward.unsets. It has no slot list that contributions
# are pushed into, no delivery and no at_head bookkeeping.

def Slots() -> Type:
  Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>

# Contributions sent so far, newest first.
def Log() -> Type:
  List<&1, Backward.Contribution>

# The gathered slot of one id, and the contributions left for the others.
def Taken() -> Type:
  Result<&2, &1, Backward.Error, Gradient.Slot> & List<&1, Backward.Contribution>

def settled(outcome: Gradient.Outcome) -> Result<&2, &1, Backward.Error, Gradient.Slot>:
  match outcome:
    case Gradient.Accumulated{slot}:
      Done{slot}
    case Gradient.Refused{slot, contribution, error}:
      Fail{Backward.GradientError{error}}

def poured(+shape: Shape.Shape, held: Result<&2, &1, Backward.Error, Gradient.Slot>, gradient: Core.Dynamic) ->
  Result<&2, &1, Backward.Error, Gradient.Slot>:
  match held:
    case Fail{error}:
      Fail{error}
    case Done{slot}:
      settled(Gradient.accumulate(shape, slot, gradient))

def sorted(+target: Nat, +shape: Shape.Shape, gradient: Core.Dynamic, mine: Bool, pair: Taken()) -> Taken():
  match mine:
    case True{}:
      (held, rest) = pair
      (poured(shape, held, gradient), rest)
    case False{}:
      (held, rest) = pair
      (held, Backward.Contribution{target, shape, gradient} <> rest)

# Takes out of a newest-first log every contribution to id and accumulates
# them into slot, oldest first; the others stay in the log, in order.
def take(log: Log(), +id: Nat, slot: Gradient.Slot) -> Taken():
  match log:
    case Nil{}:
      (Done{slot}, Nil{})
    case Con{Backward.Contribution{+target, +shape, gradient}, rest}:
      sorted(target, shape, gradient, Nat.is_eq(id, target), take(rest, id, slot))

def joined(held: Result<&2, &1, Backward.Error, Gradient.Slot>, rest: Slots()) -> Slots():
  match held:
    case Fail{error}:
      Fail{error}
    case Done{slot}:
      Backward.prepend(Gradient.Slot, slot, rest)

def gather_step(pair: Taken(), rec: Log() -> Slots()) -> Slots():
  (held, rest) = pair
  joined(held, rec(rest))

# Each slot of a newest-first list of count slots, the one with k older slots
# for id k, with what the log sends to it. A list longer than count keeps its
# slots past count, as the sweep does.
def gather(slots: List<&1, Gradient.Slot>, +count: Nat, log: Log()) -> Slots():
  match slots count:
    case Nil{} count:
      Done{Nil{}}
    case Con{slot, rest} 0n:
      Done{slot <> rest}
    case Con{slot, rest} 1n+below:
      gather_step(take(log, below, slot), more => gather(rest, below, more))

# An entry that holds g after gathering: its VJP's contributions join the log
# in operand order, so the first operand's arrives first.
def fired(pass: Backward.Pass, log: Log(), rec: Log() -> Slots()) -> Slots():
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      Fail{error}
    case Done{items}:
      Backward.prepend(Gradient.Slot, Gradient.Held{g}, rec(List.reverse.go(&1, Backward.Contribution, items, log)))

def reached(node: Backward.Node, saved: Backward.Saved, pair: Taken(), rec: Log() -> Slots()) -> Slots():
  (held, log) = pair
  match held:
    case Fail{error}:
      Fail{error}
    case Done{Gradient.Unset{}}:
      Backward.prepend(Gradient.Slot, Gradient.Unset{}, rec(log))
    case Done{Gradient.Held{g}}:
      fired(Backward.vjp(node, saved, g), log, rec)

# The working gradients by the chain rule, pulled, for entries newest first
# that start with slots and receive the contributions in log: the entry with
# id k gathers every contribution sent to k into its starting slot; if it then
# holds g, the contributions of vjp(node, saved, g) join the log.
def chained(entries: List<&1, Backward.Entry>, slots: List<&1, Gradient.Slot>, +count: Nat, log: Log()) -> Slots():
  match entries slots count:
    case Nil{} slots count:
      gather(slots, count, log)
    case Con{entry, rest} Nil{} count:
      Done{Nil{}}
    case Con{entry, rest} Con{slot, more} 0n:
      Done{Nil{}}
    case Con{Backward.Entry{node, shape, value, saved, grad}, rest} Con{slot, more} 1n+below:
      reached(node, saved, take(log, below, slot), next => chained(rest, more, below, next))

# The working gradients of a sweep, without its entries.
def result(pair: Backward.Swept()) -> Slots():
  (entries, result) = pair
  result

# The chain rule, for any recorded tape and any starting working gradients:
# when the sweep succeeds, the working gradient it leaves at every entry is
# the chained one, each slot the accumulation, in the order above, of the VJP
# values its consumers sent it at their own chained gradients. A refusal is
# stated by the structure laws in tensor-backward-LAWS.bend.
law tensor_sweep_chain_rule:
  for entries: List<&1, Backward.Entry>
  for gradients: List<&1, Gradient.Slot>
  for +count: Nat
  for -swept: List<&1, Gradient.Slot>
  for succeeded: {result(Backward.sweep(entries, Done{gradients}, count)) == Done{swept} : Slots()}
  {chained(entries, gradients, count, Nil{}) == Done{swept} : Slots()}

# The starting working gradients of a backward from the entry with older
# entries below it and newer entries above it: no gradient for the newer
# entries, ones of the output's shape at the output, as PyTorch's
# backward(torch.ones_like(output)), and no gradient below.
def seeds(+newer: Nat, +shape: Shape.Shape, ones: Core.Tensor<shape>, +older: Nat) -> List<&1, Gradient.Slot>:
  List.append(&1, Gradient.Slot, Backward.unsets(newer), Gradient.Held{Core.Packed{shape, ones}} <> Backward.unsets(older))

# The chain rule for the whole backward: when the pass of a backward from the
# output with older entries below it succeeds, the working gradients it hands
# to the merge (tensor_backward_success, tensor_merge_accumulates) are the
# chained ones from the seeds, for any tape, newer entries included.
law tensor_backward_chain_rule:
  for newer: List<&1, Backward.Entry>
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for +older: Nat
  for ones: Core.Tensor<shape>
  for made: {Core.new(shape, 1.0) == Done{ones} : Result<&2, &1, Core.Error, Core.Tensor<shape>>}
  for -gradients: List<&1, Gradient.Slot>
  for succeeded: {result(Backward.pass(List.append(&1, Backward.Entry, newer, Backward.Entry{node, shape, value, saved, slot} <> rest),
      1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), older,
      Backward.at_head(1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), older))) == Done{gradients} : Slots()}
  {chained(List.append(&1, Backward.Entry, newer, Backward.Entry{node, shape, value, saved, slot} <> rest),
      seeds(List.length(&1, Backward.Entry, newer), shape, ones, older),
      1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), Nil{}) == Done{gradients} : Slots()}
