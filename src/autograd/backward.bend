import Base

# A tape is a list of entries, newest first. The entry with k older entries
# has id k, so ids follow creation order. A lookup walks down from the newest
# entry and compares each entry's id with the requested one. It never computes
# an offset by subtraction, so an id the tape does not hold reads as
# UnknownNode and never as another entry.
#
# backward(tape, output) seeds d(output)/d(output) = 1.0 at the selected
# output, never at the newest entry. Entries newer than the output receive
# nothing. The sweep then visits the output and every older entry, newest
# first. An entry that received a gradient sends its vector-Jacobian product
# to its operands, the first operand before the second; an entry that received
# nothing sends nothing. A node's first contribution moves in unchanged, and
# every later one is added with one F32 add, the held total first. PyTorch's
# engine fills its input buffers in the same order: consumers in decreasing
# creation order, the operands of one consumer in order.
#
# The gradients of one backward are then accumulated into the slot of every
# entry with the same rule, so a repeated backward adds to the slots like
# PyTorch's backward(retain_graph=True). The gradients depend only on the
# nodes, never on the slots. A slot that no backward reached reads None, like
# PyTorch's .grad. zero_grad empties every slot.
#
# The sweep is sequential: the order in which contributions reach a slot is
# part of the contract.

type Error is Data:
  UnknownNode{}
  DanglingOperand{}

type Node is Data:
  Leaf{}
  Add{a: Nat, b: Nat}
  Sub{a: Nat, b: Nat}
  Mul{a: Nat, b: Nat, left: F32, right: F32}
  Neg{a: Nat}

type Entry is Data:
  Entry{node: Node, value: F32, grad: Maybe<&2, F32>}

type Tape is Data:
  Tape{entries: List<&2, Entry>}

type Contribution is Data:
  Contribution{target: Nat, value: F32}

type Binary is Data:
  Plus{}
  Minus{}
  Times{}

def empty() -> Tape:
  Tape{Nil{}}

def size(tape: Tape) -> Nat:
  Tape{entries} = tape
  List.length(&2, Entry, entries)

def picked(-A: Data, head: A, rec: Unit -> Maybe<&2, A>, same: Bool) -> Maybe<&2, A>:
  match same:
    case True{}:
      Some{head}
    case False{}:
      rec(Unit{})

# The element with id `id` in a newest-first list of count elements: the head
# has id count - 1, and the walk counts down from it.
def find(-A: Data, xs: List<&2, A>, +count: Nat, +id: Nat) -> Maybe<&2, A>:
  match xs count:
    case Nil{} count:
      None{}
    case Con{head, rest} 0n:
      None{}
    case Con{head, +rest} 1n+below:
      picked(A, head, unit => find(A, rest, below, id), Nat.is_eq(below, id))

def lookup(-A: Data, +xs: List<&2, A>, +id: Nat) -> Maybe<&2, A>:
  find(A, xs, List.length(&2, A, xs), id)

def accumulate(total: Maybe<&2, F32>, contribution: Maybe<&2, F32>) -> Maybe<&2, F32>:
  match total contribution:
    case None{} contribution:
      contribution
    case Some{held} None{}:
      Some{held}
    case Some{held} Some{value}:
      Some{F32.add(held, value)}

def bumped(slot: Maybe<&2, F32>, +rest: List<&2, Maybe<&2, F32>>, +contribution: F32,
  rec: Unit -> List<&2, Maybe<&2, F32>>, same: Bool) -> List<&2, Maybe<&2, F32>>:
  match same:
    case True{}:
      accumulate(slot, Some{contribution}) <> rest
    case False{}:
      slot <> rec(Unit{})

def bump(slots: List<&2, Maybe<&2, F32>>, +count: Nat, +target: Nat, +contribution: F32) -> List<&2, Maybe<&2, F32>>:
  match slots count:
    case Nil{} count:
      Nil{}
    case Con{slot, rest} 0n:
      slot <> rest
    case Con{slot, +rest} 1n+below:
      bumped(slot, rest, contribution, unit => bump(rest, below, target, contribution), Nat.is_eq(below, target))

# The vector-Jacobian product of one node: the contribution each operand
# receives when the node's gradient is g, in operand order.
def vjp(node: Node, +g: F32) -> List<&2, Contribution>:
  match node:
    case Leaf{}:
      []
    case Add{a, b}:
      [Contribution{a, g}, Contribution{b, g}]
    case Sub{a, b}:
      [Contribution{a, g}, Contribution{b, F32.neg(g)}]
    case Mul{a, b, left, right}:
      [Contribution{a, F32.mul(g, right)}, Contribution{b, F32.mul(g, left)}]
    case Neg{a}:
      [Contribution{a, F32.neg(g)}]

def deliver(contributions: List<&2, Contribution>, +count: Nat, slots: List<&2, Maybe<&2, F32>>) -> List<&2, Maybe<&2, F32>>:
  match contributions:
    case Nil{}:
      slots
    case Con{Contribution{+target, +value}, rest}:
      deliver(rest, count, bump(slots, count, target, value))

def contribute(node: Node, gradient: Maybe<&2, F32>, +count: Nat, slots: List<&2, Maybe<&2, F32>>) -> List<&2, Maybe<&2, F32>>:
  match gradient:
    case None{}:
      slots
    case Some{value}:
      deliver(vjp(node, value), count, slots)

def sweep(nodes: List<&2, Node>, gradients: List<&2, Maybe<&2, F32>>, +count: Nat) -> List<&2, Maybe<&2, F32>>:
  match nodes gradients count:
    case Nil{} gradients count:
      gradients
    case Con{node, rest} Nil{} count:
      Nil{}
    case Con{node, rest} Con{gradient, pending} 0n:
      Nil{}
    case Con{node, rest} Con{+gradient, pending} 1n+below:
      gradient <> sweep(rest, contribute(node, gradient, below, pending), below)

def nones(count: Nat) -> List<&2, Maybe<&2, F32>>:
  match count:
    case 0n:
      Nil{}
    case 1n+rest:
      None{} <> nones(rest)

def chosen(node: Node, +rest: List<&2, Node>, +below: Nat, rec: Unit -> List<&2, Maybe<&2, F32>>, selected: Bool) ->
  List<&2, Maybe<&2, F32>>:
  match selected:
    case True{}:
      sweep(node <> rest, Some{1.0} <> nones(below), 1n+below)
    case False{}:
      None{} <> rec(Unit{})

# The gradient of output with respect to each of count nodes, newest first:
# None for the nodes newer than output and for every node output does not
# reach.
def pass(nodes: List<&2, Node>, +count: Nat, +output: Nat) -> List<&2, Maybe<&2, F32>>:
  match nodes count:
    case Nil{} count:
      Nil{}
    case Con{node, rest} 0n:
      Nil{}
    case Con{node, +rest} 1n+below:
      chosen(node, rest, below, unit => pass(rest, below, output), Nat.is_eq(below, output))

def operands_below(node: Node, +bound: Nat) -> Bool:
  match node:
    case Leaf{}:
      True{}
    case Add{a, b}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Sub{a, b}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Mul{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Neg{a}:
      Nat.is_lt(a, bound)

def wired(nodes: List<&2, Node>, +count: Nat) -> Bool:
  match nodes count:
    case Nil{} count:
      True{}
    case Con{node, rest} 0n:
      False{}
    case Con{node, rest} 1n+below:
      Bool.and(operands_below(node, below), wired(rest, below))

def nodes(entries: List<&2, Entry>) -> List<&2, Node>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Entry{node, value, slot}, rest}:
      node <> nodes(rest)

def checked(+graph: List<&2, Node>, +count: Nat, +output: Nat, known: Bool, sound: Bool) ->
  Result<&2, &2, Error, List<&2, Maybe<&2, F32>>>:
  match known sound:
    case False{} sound:
      Fail{UnknownNode{}}
    case True{} False{}:
      Fail{DanglingOperand{}}
    case True{} True{}:
      Done{pass(graph, count, output)}

def planned(+graph: List<&2, Node>, +output: Nat) -> Result<&2, &2, Error, List<&2, Maybe<&2, F32>>>:
  +count = List.length(&2, Node, graph)
  checked(graph, count, output, Nat.is_lt(output, count), wired(graph, count))

# The gradients of one backward of output, newest first, read from the nodes
# alone. An output the tape does not hold is refused with UnknownNode; a tape
# with a node that refers to itself or to a newer or missing node is refused
# with DanglingOperand.
def gradients(tape: Tape, +output: Nat) -> Result<&2, &2, Error, List<&2, Maybe<&2, F32>>>:
  Tape{entries} = tape
  planned(nodes(entries), output)

def merge(entries: List<&2, Entry>, gradients: List<&2, Maybe<&2, F32>>) -> List<&2, Entry>:
  match entries gradients:
    case Nil{} gradients:
      Nil{}
    case Con{entry, rest} Nil{}:
      entry <> rest
    case Con{Entry{node, value, slot}, rest} Con{gradient, more}:
      Entry{node, value, accumulate(slot, gradient)} <> merge(rest, more)

def merged(entries: List<&2, Entry>, result: Result<&2, &2, Error, List<&2, Maybe<&2, F32>>>) -> Result<&2, &2, Error, Tape>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{found}:
      Done{Tape{merge(entries, found)}}

# Runs one backward of output and accumulates its gradients into the slots.
def backward(tape: Tape, +output: Nat) -> Result<&2, &2, Error, Tape>:
  Tape{+entries} = tape
  merged(entries, gradients(Tape{entries}, output))

def cleared(entries: List<&2, Entry>) -> List<&2, Entry>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Entry{node, value, slot}, rest}:
      Entry{node, value, None{}} <> cleared(rest)

# Empties every gradient slot, like PyTorch's zero_grad(set_to_none=True).
def zero_grad(tape: Tape) -> Tape:
  Tape{entries} = tape
  Tape{cleared(entries)}

def entry_value(found: Maybe<&2, Entry>) -> Result<&2, &2, Error, F32>:
  match found:
    case None{}:
      Fail{UnknownNode{}}
    case Some{Entry{node, value, slot}}:
      Done{value}

def entry_grad(found: Maybe<&2, Entry>) -> Result<&2, &2, Error, Maybe<&2, F32>>:
  match found:
    case None{}:
      Fail{UnknownNode{}}
    case Some{Entry{node, value, slot}}:
      Done{slot}

def value(tape: Tape, +id: Nat) -> Result<&2, &2, Error, F32>:
  Tape{entries} = tape
  entry_value(lookup(Entry, entries, id))

# The gradient slot of node id: None until a backward reaches it.
def grad(tape: Tape, +id: Nat) -> Result<&2, &2, Error, Maybe<&2, F32>>:
  Tape{entries} = tape
  entry_grad(lookup(Entry, entries, id))

# Records a leaf; its id is the number of entries before it.
def leaf(tape: Tape, value: F32) -> Tape & Nat:
  Tape{+entries} = tape
  (Tape{Entry{Leaf{}, value, None{}} <> entries}, List.length(&2, Entry, entries))

def forward(kind: Binary, +a: Nat, +b: Nat, +left: F32, +right: F32) -> Entry:
  match kind:
    case Plus{}:
      Entry{Add{a, b}, F32.add(left, right), None{}}
    case Minus{}:
      Entry{Sub{a, b}, F32.sub(left, right), None{}}
    case Times{}:
      Entry{Mul{a, b, left, right}, F32.mul(left, right), None{}}

def operands(+entries: List<&2, Entry>, kind: Binary, +a: Nat, +b: Nat,
  left: Result<&2, &2, Error, F32>, right: Result<&2, &2, Error, F32>) -> Tape & Result<&2, &2, Error, Nat>:
  match left right:
    case Fail{error} right:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{x} Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{x} Done{y}:
      (Tape{forward(kind, a, b, x, y) <> entries}, Done{List.length(&2, Entry, entries)})

# Records `a op b`. An operand id the tape does not hold is refused with
# UnknownNode, and the tape comes back unchanged.
def binary(tape: Tape, kind: Binary, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{+entries} = tape
  operands(entries, kind, a, b, value(Tape{entries}, a), value(Tape{entries}, b))

def add(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Plus{}, a, b)

def sub(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Minus{}, a, b)

def mul(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Times{}, a, b)

def negated(+entries: List<&2, Entry>, +a: Nat, operand: Result<&2, &2, Error, F32>) -> Tape & Result<&2, &2, Error, Nat>:
  match operand:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{x}:
      (Tape{Entry{Neg{a}, F32.neg(x), None{}} <> entries}, Done{List.length(&2, Entry, entries)})

# Records -a, with the same refusal as binary.
def neg(tape: Tape, +a: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{+entries} = tape
  negated(entries, a, value(Tape{entries}, a))

def error_code(error: Error) -> U32:
  match error:
    case UnknownNode{}:
      103
    case DanglingOperand{}:
      104
