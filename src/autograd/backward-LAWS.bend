import Base
import ./backward.bend as Backward

# Reachability, stated on flags: a node is reached when it is the selected
# output, or an operand of a reached node newer than it. Nodes newer than the
# output are never reached.

def marked(flag: Bool, +rest: List<&2, Bool>, rec: Unit -> List<&2, Bool>, same: Bool) -> List<&2, Bool>:
  match same:
    case True{}:
      True{} <> rest
    case False{}:
      flag <> rec(Unit{})

# Sets the flag of node target in a newest-first list of count flags.
def mark(flags: List<&2, Bool>, +count: Nat, +target: Nat) -> List<&2, Bool>:
  match flags count:
    case Nil{} count:
      Nil{}
    case Con{flag, rest} 0n:
      flag <> rest
    case Con{flag, +rest} 1n+below:
      marked(flag, rest, unit => mark(rest, below, target), Nat.is_eq(below, target))

# The operand ids of a node, in operand order.
def targets(node: Backward.Node) -> List<&2, Nat>:
  match node:
    case Backward.Leaf{}:
      []
    case Backward.Add{a, b}:
      [a, b]
    case Backward.Sub{a, b}:
      [a, b]
    case Backward.Mul{a, b, left, right}:
      [a, b]
    case Backward.Neg{a}:
      [a]

def mark_all(ids: List<&2, Nat>, +count: Nat, flags: List<&2, Bool>) -> List<&2, Bool>:
  match ids:
    case Nil{}:
      flags
    case Con{+id, rest}:
      mark_all(rest, count, mark(flags, count, id))

def spread(node: Backward.Node, flag: Bool, +count: Nat, flags: List<&2, Bool>) -> List<&2, Bool>:
  match flag:
    case False{}:
      flags
    case True{}:
      mark_all(targets(node), count, flags)

def reach(nodes: List<&2, Backward.Node>, flags: List<&2, Bool>, +count: Nat) -> List<&2, Bool>:
  match nodes flags count:
    case Nil{} flags count:
      flags
    case Con{node, rest} Nil{} count:
      Nil{}
    case Con{node, rest} Con{flag, pending} 0n:
      Nil{}
    case Con{node, rest} Con{+flag, pending} 1n+below:
      flag <> reach(rest, spread(node, flag, below, pending), below)

def falses(count: Nat) -> List<&2, Bool>:
  match count:
    case 0n:
      Nil{}
    case 1n+rest:
      False{} <> falses(rest)

def reached_chosen(node: Backward.Node, +rest: List<&2, Backward.Node>, +below: Nat, rec: Unit -> List<&2, Bool>, selected: Bool) ->
  List<&2, Bool>:
  match selected:
    case True{}:
      reach(node <> rest, True{} <> falses(below), 1n+below)
    case False{}:
      False{} <> rec(Unit{})

# The reached flag of each of count nodes, newest first.
def reached(nodes: List<&2, Backward.Node>, +count: Nat, +output: Nat) -> List<&2, Bool>:
  match nodes count:
    case Nil{} count:
      Nil{}
    case Con{node, rest} 0n:
      Nil{}
    case Con{node, +rest} 1n+below:
      reached_chosen(node, rest, below, unit => reached(rest, below, output), Nat.is_eq(below, output))

# Which gradients are present.
def somes(gradients: List<&2, Maybe<&2, F32>>) -> List<&2, Bool>:
  match gradients:
    case Nil{}:
      Nil{}
    case Con{gradient, rest}:
      Maybe.is_some(&2, F32, gradient) <> somes(rest)

# Views of a tape.

def nodes_of(tape: Backward.Tape) -> List<&2, Backward.Node>:
  Backward.Tape{entries} = tape
  Backward.nodes(entries)

def slots(entries: List<&2, Backward.Entry>) -> List<&2, Maybe<&2, F32>>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Backward.Entry{node, value, slot}, rest}:
      slot <> slots(rest)

def tape_slots(tape: Backward.Tape) -> List<&2, Maybe<&2, F32>>:
  Backward.Tape{entries} = tape
  slots(entries)

def slots_after(result: Result<&2, &2, Backward.Error, Backward.Tape>) -> Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tape}:
      Done{tape_slots(tape)}

def grad_after(result: Result<&2, &2, Backward.Error, Backward.Tape>, +id: Nat) -> Result<&2, &2, Backward.Error, Maybe<&2, F32>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tape}:
      Backward.grad(tape, id)

# A second backward of output on the result of the first.
def again(result: Result<&2, &2, Backward.Error, Backward.Tape>, +output: Nat) -> Result<&2, &2, Backward.Error, Backward.Tape>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{tape}:
      Backward.backward(tape, output)

def observe(+tape: Backward.Tape, +output: Nat) ->
  Result<&2, &2, Backward.Error, F32> & Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>:
  (Backward.value(tape, output), Backward.gradients(tape, output))

# Small programs recorded through the public API. Ids are creation order.

def tape_of(pair: Backward.Tape & Nat) -> Backward.Tape:
  (tape, id) = pair
  tape

def recorded(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape:
  (tape, result) = pair
  tape

# x = 0, y = 1, x op y = 2.
def program_binary(kind: Backward.Binary, x: F32, y: F32) -> Backward.Tape:
  recorded(Backward.binary(tape_of(Backward.leaf(tape_of(Backward.leaf(Backward.empty(), x)), y)), kind, 0n, 1n))

# x = 0, -x = 1.
def program_neg(x: F32) -> Backward.Tape:
  recorded(Backward.neg(tape_of(Backward.leaf(Backward.empty(), x)), 0n))

# x = 0, y = 1, z = 2, x * y = 3, z + z = 4. Node 4 is newer than the output
# 3 and uses z twice; z is not used by 3.
def program_unused(x: F32, y: F32, z: F32) -> Backward.Tape:
  recorded(Backward.add(recorded(Backward.mul(
    tape_of(Backward.leaf(tape_of(Backward.leaf(tape_of(Backward.leaf(Backward.empty(), x)), y)), z)), 0n, 1n)), 2n, 2n))

# x = 0, a = 1, b = 2, c = 3, p = x * a = 4, q = x * b = 5, r = x * c = 6,
# s = p + q = 7, t = s + r = 8. x has three consumers.
def program_fan_out(x: F32, a: F32, b: F32, c: F32) -> Backward.Tape:
  recorded(Backward.add(recorded(Backward.add(recorded(Backward.mul(recorded(Backward.mul(recorded(Backward.mul(
    tape_of(Backward.leaf(tape_of(Backward.leaf(tape_of(Backward.leaf(tape_of(Backward.leaf(Backward.empty(), x)), a)), b)), c)),
    0n, 1n)), 0n, 2n)), 0n, 3n)), 4n, 5n)), 7n, 6n))

# Vector-Jacobian products: the contribution each operand receives when the
# node's gradient is g, in operand order.

law vjp_leaf:
  for +g: F32
  {Backward.vjp(Backward.Leaf{}, g) == [] : List<&2, Backward.Contribution>}

law vjp_add:
  for +a: Nat
  for +b: Nat
  for +g: F32
  {Backward.vjp(Backward.Add{a, b}, g) == [Backward.Contribution{a, g}, Backward.Contribution{b, g}] : List<&2, Backward.Contribution>}

law vjp_sub:
  for +a: Nat
  for +b: Nat
  for +g: F32
  {Backward.vjp(Backward.Sub{a, b}, g) == [Backward.Contribution{a, g}, Backward.Contribution{b, F32.neg(g)}] : List<&2, Backward.Contribution>}

# d(a*b)/da = b and d(a*b)/db = a, with the operand values saved at record
# time: the first operand receives g * right, the second g * left.
law vjp_mul:
  for +a: Nat
  for +b: Nat
  for +left: F32
  for +right: F32
  for +g: F32
  {Backward.vjp(Backward.Mul{a, b, left, right}, g) ==
    [Backward.Contribution{a, F32.mul(g, right)}, Backward.Contribution{b, F32.mul(g, left)}] : List<&2, Backward.Contribution>}

law vjp_neg:
  for +a: Nat
  for +g: F32
  {Backward.vjp(Backward.Neg{a}, g) == [Backward.Contribution{a, F32.neg(g)}] : List<&2, Backward.Contribution>}

# Recording through the public API and running backward gives each op's
# forward value and its VJP at the leaves (gradients newest first).

law backward_add:
  for +x: F32
  for +y: F32
  {observe(program_binary(Backward.Plus{}, x, y), 2n) ==
    (Done{F32.add(x, y)}, Done{[Some{1.0}, Some{1.0}, Some{1.0}]}) :
    Result<&2, &2, Backward.Error, F32> & Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

law backward_sub:
  for +x: F32
  for +y: F32
  {observe(program_binary(Backward.Minus{}, x, y), 2n) ==
    (Done{F32.sub(x, y)}, Done{[Some{1.0}, Some{F32.neg(1.0)}, Some{1.0}]}) :
    Result<&2, &2, Backward.Error, F32> & Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

law backward_mul:
  for +x: F32
  for +y: F32
  {observe(program_binary(Backward.Times{}, x, y), 2n) ==
    (Done{F32.mul(x, y)}, Done{[Some{1.0}, Some{F32.mul(1.0, x)}, Some{F32.mul(1.0, y)}]}) :
    Result<&2, &2, Backward.Error, F32> & Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

law backward_neg:
  for +x: F32
  {observe(program_neg(x), 1n) ==
    (Done{F32.neg(x)}, Done{[Some{1.0}, Some{F32.neg(1.0)}]}) :
    Result<&2, &2, Backward.Error, F32> & Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

# The seed is the selected output (C1). For any graph, the nodes newer than
# the output get None, the output gets exactly 1.0, and the older nodes get
# what the sweep sends from the output's VJP, whatever the newer nodes are.
law seed_selected_output:
  for +newer: List<&2, Backward.Node>
  for +node: Backward.Node
  for +rest: List<&2, Backward.Node>
  {Backward.pass(List.append(&2, Backward.Node, newer, node <> rest),
      List.length(&2, Backward.Node, List.append(&2, Backward.Node, newer, node <> rest)), List.length(&2, Backward.Node, rest)) ==
    List.append(&2, Maybe<&2, F32>, Backward.nones(List.length(&2, Backward.Node, newer)),
      Some{1.0} <> Backward.sweep(rest,
        Backward.contribute(node, Some{1.0}, List.length(&2, Backward.Node, rest), Backward.nones(List.length(&2, Backward.Node, rest))),
        List.length(&2, Backward.Node, rest))) : List<&2, Maybe<&2, F32>>}

# Only the reachable graph: a node has a gradient exactly when the selected
# output reaches it. Every other node, newer or unused, reads None.
law reached_exactly:
  for +graph: List<&2, Backward.Node>
  for +output: Nat
  {somes(Backward.pass(graph, List.length(&2, Backward.Node, graph), output)) ==
    reached(graph, List.length(&2, Backward.Node, graph), output) : List<&2, Bool>}

# With x, y and z recorded and z only used by a node newer than the output
# x * y, backward of x * y seeds the output, leaves z and the newer node None,
# and gives x and y their VJP.
law unused_leaf_none:
  for +x: F32
  for +y: F32
  for +z: F32
  {slots_after(Backward.backward(program_unused(x, y, z), 3n)) ==
    Done{[None{}, Some{1.0}, None{}, Some{F32.mul(1.0, x)}, Some{F32.mul(1.0, y)}]} :
    Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

# Fan-out: a node's first contribution moves in, and each later one is added
# to the held total with one F32 add, total first.
law accumulate_moves_in:
  for contribution: Maybe<&2, F32>
  {Backward.accumulate(None{}, contribution) == contribution : Maybe<&2, F32>}

law accumulate_adds_total_first:
  for +held: F32
  for +value: F32
  {Backward.accumulate(Some{held}, Some{value}) == Some{F32.add(held, value)} : Maybe<&2, F32>}

law accumulate_keeps_without_contribution:
  for +held: F32
  {Backward.accumulate(Some{held}, None{}) == Some{held} : Maybe<&2, F32>}

# A contribution to node target reaches exactly the slot of that node and no
# other, for any list of slots.
law bump_target:
  for +newer: List<&2, Maybe<&2, F32>>
  for +slot: Maybe<&2, F32>
  for +rest: List<&2, Maybe<&2, F32>>
  for +value: F32
  {Backward.bump(List.append(&2, Maybe<&2, F32>, newer, slot <> rest),
      List.length(&2, Maybe<&2, F32>, List.append(&2, Maybe<&2, F32>, newer, slot <> rest)), List.length(&2, Maybe<&2, F32>, rest), value) ==
    List.append(&2, Maybe<&2, F32>, newer, Backward.accumulate(slot, Some{value}) <> rest) : List<&2, Maybe<&2, F32>>}

# The contributions of x's three consumers arrive newest consumer first, and
# each is added to the held total.
law fan_out_arrival_order:
  for +x: F32
  for +a: F32
  for +b: F32
  for +c: F32
  {grad_after(Backward.backward(program_fan_out(x, a, b, c), 8n), 0n) ==
    Done{Some{F32.add(F32.add(F32.mul(1.0, c), F32.mul(1.0, b)), F32.mul(1.0, a))}} :
    Result<&2, &2, Backward.Error, Maybe<&2, F32>>}

# Ids (C2): id k reads exactly the element with k older elements, and an id
# the list does not hold reads None, never another element.
law lookup_exact:
  for -A: Data
  for +newer: List<&2, A>
  for +item: A
  for +rest: List<&2, A>
  {Backward.lookup(A, List.append(&2, A, newer, item <> rest), List.length(&2, A, rest)) == Some{item} : Maybe<&2, A>}

law lookup_unknown:
  for -A: Data
  for +items: List<&2, A>
  for +id: Nat
  for outside: {Nat.is_lt(id, List.length(&2, A, items)) == False{} : Bool}
  {Backward.lookup(A, items, id) == None{} : Maybe<&2, A>}

law bump_unknown:
  for +gradients: List<&2, Maybe<&2, F32>>
  for +target: Nat
  for +value: F32
  for +outside: {Nat.is_lt(target, List.length(&2, Maybe<&2, F32>, gradients)) == False{} : Bool}
  {Backward.bump(gradients, List.length(&2, Maybe<&2, F32>, gradients), target, value) == gradients : List<&2, Maybe<&2, F32>>}

law value_unknown:
  for +entries: List<&2, Backward.Entry>
  for +id: Nat
  for outside: {Nat.is_lt(id, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.value(Backward.Tape{entries}, id) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, F32>}

law grad_unknown:
  for +entries: List<&2, Backward.Entry>
  for +id: Nat
  for outside: {Nat.is_lt(id, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.grad(Backward.Tape{entries}, id) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, Maybe<&2, F32>>}

# Recording with an operand id the tape does not hold is refused with
# UnknownNode, and the tape comes back unchanged.
law record_unknown_left:
  for +entries: List<&2, Backward.Entry>
  for kind: Backward.Binary
  for +a: Nat
  for +b: Nat
  for outside: {Nat.is_lt(a, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.binary(Backward.Tape{entries}, kind, a, b) == (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) :
    Backward.Tape & Result<&2, &2, Backward.Error, Nat>}

law record_unknown_right:
  for +entries: List<&2, Backward.Entry>
  for kind: Backward.Binary
  for +a: Nat
  for +b: Nat
  for outside: {Nat.is_lt(b, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.binary(Backward.Tape{entries}, kind, a, b) == (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) :
    Backward.Tape & Result<&2, &2, Backward.Error, Nat>}

law neg_unknown:
  for +entries: List<&2, Backward.Entry>
  for +a: Nat
  for outside: {Nat.is_lt(a, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.neg(Backward.Tape{entries}, a) == (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) :
    Backward.Tape & Result<&2, &2, Backward.Error, Nat>}

# backward of an output the tape does not hold is refused with UnknownNode.
law backward_unknown:
  for +entries: List<&2, Backward.Entry>
  for +output: Nat
  for outside: {Nat.is_lt(output, List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, Backward.Tape>}

# A tape with a node that refers to itself, to a newer node or to a missing
# node is refused with DanglingOperand.
law backward_dangling:
  for +entries: List<&2, Backward.Entry>
  for +output: Nat
  for inside: {Nat.is_lt(output, List.length(&2, Backward.Entry, entries)) == True{} : Bool}
  for unsound: {Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)) == False{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) == Fail{Backward.DanglingOperand{}} : Result<&2, &2, Backward.Error, Backward.Tape>}

# Otherwise backward accumulates the gradients of pass into the slots.
law backward_success:
  for +entries: List<&2, Backward.Entry>
  for +output: Nat
  for inside: {Nat.is_lt(output, List.length(&2, Backward.Entry, entries)) == True{} : Bool}
  for sound: {Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)) == True{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) ==
    Done{Backward.Tape{Backward.merge(entries, Backward.pass(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output))}} :
    Result<&2, &2, Backward.Error, Backward.Tape>}

# Repeated backward, like PyTorch's backward(retain_graph=True): the second
# backward computes the same gradients and accumulates them again.
law backward_twice_accumulates:
  for +entries: List<&2, Backward.Entry>
  for +output: Nat
  for +found: List<&2, Maybe<&2, F32>>
  for +computed: {Backward.gradients(Backward.Tape{entries}, output) == Done{found} : Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}
  {again(Backward.backward(Backward.Tape{entries}, output), output) ==
    Done{Backward.Tape{Backward.merge(Backward.merge(entries, found), found)}} : Result<&2, &2, Backward.Error, Backward.Tape>}

law backward_twice_mul:
  for +x: F32
  for +y: F32
  {slots_after(again(Backward.backward(program_binary(Backward.Times{}, x, y), 2n), 2n)) ==
    Done{[Some{F32.add(1.0, 1.0)}, Some{F32.add(F32.mul(1.0, x), F32.mul(1.0, x))}, Some{F32.add(F32.mul(1.0, y), F32.mul(1.0, y))}]} :
    Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>}

# zero_grad empties every slot and keeps every node.
law zero_grad_empties:
  for +entries: List<&2, Backward.Entry>
  {tape_slots(Backward.zero_grad(Backward.Tape{entries})) == Backward.nones(List.length(&2, Backward.Entry, entries)) : List<&2, Maybe<&2, F32>>}

law zero_grad_keeps_nodes:
  for +entries: List<&2, Backward.Entry>
  {nodes_of(Backward.zero_grad(Backward.Tape{entries})) == Backward.nodes(entries) : List<&2, Backward.Node>}
