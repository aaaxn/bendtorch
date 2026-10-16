import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward

# The nodes of a list of entries, newest first.
def nodes_of(entries: List<&1, Backward.Entry>) -> List<&2, Backward.Node>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Backward.Entry{node, shape, value, saved, slot}, rest}:
      node <> nodes_of(rest)

def slots_of(entries: List<&1, Backward.Entry>) -> List<&1, Gradient.Slot>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Backward.Entry{node, shape, value, saved, slot}, rest}:
      slot <> slots_of(rest)

def tape_nodes(tape: Backward.Tape) -> List<&2, Backward.Node>:
  Backward.Tape{entries} = tape
  nodes_of(entries)

def tape_slots(tape: Backward.Tape) -> List<&1, Gradient.Slot>:
  Backward.Tape{entries} = tape
  slots_of(entries)

# Operand bounds, stated without the engine's validator: a node may name only
# nodes older than itself, so each operand id must be below the number of
# older nodes.
def targets(node: Backward.Node) -> List<&2, Nat>:
  match node:
    case Backward.Leaf{}:
      []
    case Backward.Add{a, b, left, right}:
      [a, b]
    case Backward.Sub{a, b, left, right}:
      [a, b]
    case Backward.Mul{a, b, left, right}:
      [a, b]
    case Backward.Sum{a, input}:
      [a]
    case Backward.Matmul{a, b, left, right}:
      [a, b]
    case Backward.MatmulNT{a, b, left, right}:
      [a, b]
    case Backward.Relu{a, input}:
      [a]
    case Backward.MseLoss{a, b, left, right}:
      [a, b]
    case Backward.CrossEntropy{a, input}:
      [a]
    case Backward.Constant{}:
      []
    case Backward.Frozen{id, inner}:
      targets(inner)

def bounded(ids: List<&2, Nat>, +bound: Nat) -> Bool:
  match ids:
    case Nil{}:
      True{}
    case Con{+id, rest}:
      Bool.and(Nat.is_lt(id, bound), bounded(rest, bound))

def well_formed(nodes: List<&2, Backward.Node>) -> Bool:
  match nodes:
    case Nil{}:
      True{}
    case Con{node, +rest}:
      Bool.and(bounded(targets(node), List.length(&2, Backward.Node, rest)), well_formed(rest))

# The slots newer than a target, put back in front of the result.
def relisted(newer: List<&1, Gradient.Slot>, result: Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>) ->
  Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>:
  match newer:
    case Nil{}:
      result
    case Con{slot, rest}:
      Backward.prepend(Gradient.Slot, slot, relisted(rest, result))

# The entries newer than the output, each with no gradient, in front of the
# sweep's result.
def newer_unset(newer: List<&1, Backward.Entry>, swept: Backward.Swept()) -> Backward.Swept():
  match newer:
    case Nil{}:
      swept
    case Con{entry, rest}:
      Backward.finished(entry, Gradient.Unset{}, newer_unset(rest, swept))

# The only contribution of a VJP, when it sends exactly one.
def sole(pass: Backward.Pass) -> Maybe<&1, Backward.Contribution>:
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      None{}
    case Done{Nil{}}:
      None{}
    case Done{Con{contribution, Nil{}}}:
      Some{contribution}
    case Done{Con{contribution, Con{other, rest}}}:
      None{}

def sole_target(found: Maybe<&1, Backward.Contribution>) -> Maybe<&2, Nat>:
  match found:
    case None{}:
      None{}
    case Some{Backward.Contribution{target, shape, gradient}}:
      Some{target}

def sole_dimensions(found: Maybe<&1, Backward.Contribution>) -> Maybe<&2, List<&2, U32>>:
  match found:
    case None{}:
      None{}
    case Some{Backward.Contribution{target, shape, Core.Packed{dimensions, tensor}}}:
      Some{Shape.dimensions(dimensions)}

def sole_point(found: Maybe<&1, Backward.Contribution>, +index: U32) -> Maybe<&2, F32>:
  match found:
    case None{}:
      None{}
    case Some{Backward.Contribution{target, shape, gradient}}:
      Core.dynamic_read_value(Core.read_dynamic(gradient, index))

# What a VJP that sends one contribution promises: it goes to node a, it has
# the target's dimensions, and every in-bounds index i holds cell(i).
type SoleContract<-a: Nat, -target: Shape.Shape, -cell: Nat -> F32, -pass: Backward.Pass> is Type:
  SoleContract{
    target_id: {sole_target(sole(pass)) == Some{a} : Maybe<&2, Nat>},
    dimensions: {sole_dimensions(sole(pass)) == Some{Shape.dimensions(target)} : Maybe<&2, List<&2, U32>>},
    value: @+index: U32 -> @inside: {Nat.is_lt(U32.to_nat(index), Shape.logical_numel(Shape.dimensions(target))) == True{} : Bool} ->
      {sole_point(sole(pass), index) == Some{cell(U32.to_nat(index))} : Maybe<&2, F32>}}

# NegateLeft flips the sign bit of the left word, as PyTorch's neg does, and
# ignores the right word. The Sub VJP sends -g to its second operand with it.
law negate_left_scalar:
  for +left: F32
  for +right: F32
  {Elementwise.scalar(Elementwise.NegateLeft{}, left, right) == F32.neg(left) : F32}

# A leaf sends nothing, and hands back its saved values and its gradient.
law tensor_vjp_leaf:
  for saved: Backward.Saved
  for g: Core.Dynamic
  {Backward.vjp(Backward.Leaf{}, saved, g) == Backward.Pass{saved, g, Done{Nil{}}} : Backward.Pass}

# A constant sends nothing, and hands back its saved values and its gradient.
law tensor_vjp_constant:
  for saved: Backward.Saved
  for g: Core.Dynamic
  {Backward.vjp(Backward.Constant{}, saved, g) == Backward.Pass{saved, g, Done{Nil{}}} : Backward.Pass}

# Frozen{id, node}, the node of a recording whose operand id is a constant,
# sends node's contributions without the ones to id, in order, and hands back
# what node's VJP hands back. Where Backward.skips(id, node) holds (node is
# x @ W^T and id is x and not W), the VJP does not compute the product it
# would drop; tensor_matmul_nt_frozen_vjp_values states what it sends.
law tensor_vjp_frozen_unsent:
  for +id: Nat
  for +node: Backward.Node
  for saved: Backward.Saved
  for g: Core.Dynamic
  for computed: {Backward.skips(id, node) == False{} : Bool}
  {Backward.vjp(Backward.Frozen{id, node}, saved, g) == Backward.unsent(id, Backward.vjp(node, saved, g)) : Backward.Pass}

# Whether a list of contributions holds one to id.
def reaches(+id: Nat, items: List<&1, Backward.Contribution>) -> Bool:
  match items:
    case Nil{}:
      False{}
    case Con{Backward.Contribution{target, shape, gradient}, rest}:
      Bool.or(Nat.is_eq(id, target), reaches(id, rest))

# Whether a VJP's pass sends a contribution to id.
def sends_to(+id: Nat, pass: Backward.Pass) -> Bool:
  Backward.Pass{saved, g, contributions} = pass
  match contributions:
    case Fail{error}:
      False{}
    case Done{items}:
      reaches(id, items)

# A frozen operand gets nothing: for any node, saved values and g, no
# contribution of the VJP of Frozen{id, node} goes to id, skipped product
# included.
law tensor_vjp_frozen_spares:
  for +id: Nat
  for +node: Backward.Node
  for saved: Backward.Saved
  for g: Core.Dynamic
  {sends_to(id, Backward.vjp(Backward.Frozen{id, node}, saved, g)) == False{} : Bool}

# backward of an output the tape does not hold is refused with UnknownNode,
# and the tape comes back unchanged.
law tensor_backward_unknown:
  for entries: List<&1, Backward.Entry>
  for +output: Nat
  for outside: {Nat.is_lt(output, List.length(&2, Backward.Node, nodes_of(entries))) == False{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) == (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) :
    Backward.Tape & Result<&2, &2, Backward.Error, Unit>}

# A tape with a node that refers to itself, to a newer node or to a missing
# node is refused with DanglingOperand, and comes back unchanged.
law tensor_backward_dangling:
  for entries: List<&1, Backward.Entry>
  for +output: Nat
  for inside: {Nat.is_lt(output, List.length(&2, Backward.Node, nodes_of(entries))) == True{} : Bool}
  for unsound: {Backward.wired(nodes_of(entries), List.length(&2, Backward.Node, nodes_of(entries))) == False{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) == (Backward.Tape{entries}, Fail{Backward.DanglingOperand{}}) :
    Backward.Tape & Result<&2, &2, Backward.Error, Unit>}

# Otherwise backward runs pass from the selected output and accumulates its
# working gradients into the slots.
law tensor_backward_success:
  for entries: List<&1, Backward.Entry>
  for +output: Nat
  for inside: {Nat.is_lt(output, List.length(&2, Backward.Node, nodes_of(entries))) == True{} : Bool}
  for sound: {Backward.wired(nodes_of(entries), List.length(&2, Backward.Node, nodes_of(entries))) == True{} : Bool}
  {Backward.backward(Backward.Tape{entries}, output) ==
    Backward.ran(Backward.pass(entries, List.length(&2, Backward.Node, nodes_of(entries)), output,
      Backward.at_head(List.length(&2, Backward.Node, nodes_of(entries)), output))) :
    Backward.Tape & Result<&2, &2, Backward.Error, Unit>}

# The engine's wiring check is exactly the operand-bound specification above.
law tensor_wired_matches_operand_bounds:
  for +graph: List<&2, Backward.Node>
  {Backward.wired(graph, List.length(&2, Backward.Node, graph)) == well_formed(graph) : Bool}

# The seed is the selected output (C1): for any entries newer than it, those
# entries get no gradient, and the sweep starts at the output with ones of the
# output's shape, Core.new(shape, 1.0), and no gradient for the older entries.
law tensor_seed_selected_output:
  for newer: List<&1, Backward.Entry>
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for +older: Nat
  for fits: {List.length(&1, Backward.Entry, rest) == older : Nat}
  {Backward.pass(List.append(&1, Backward.Entry, newer, Backward.Entry{node, shape, value, saved, slot} <> rest),
      1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), older,
      Backward.at_head(1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), older)) ==
    newer_unset(newer, Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0))) : Backward.Swept()}

# The sweep starts from the seed: the output gets ones of its shape and every
# older entry starts with no gradient.
law tensor_seeded_sweep:
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for +below: Nat
  for ones: Core.Tensor<shape>
  {Backward.seeded(node, shape, value, saved, slot, rest, below, Done{ones}) ==
    Backward.sweep(Backward.Entry{node, shape, value, saved, slot} <> rest,
      Done{Gradient.Held{Core.Packed{shape, ones}} <> Backward.unsets(below)}, 1n+below) : Backward.Swept()}

# An entry that received no gradient sends nothing: the older entries are
# swept with their working gradients unchanged.
law tensor_unreached_sends_nothing:
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for pending: List<&1, Gradient.Slot>
  for +below: Nat
  {Backward.sweep(Backward.Entry{node, shape, value, saved, slot} <> rest, Done{Gradient.Unset{} <> pending}, 1n+below) ==
    Backward.finished(Backward.Entry{node, shape, value, saved, slot}, Gradient.Unset{},
      Backward.sweep(rest, Done{pending}, below)) : Backward.Swept()}

# The contributions of one VJP are delivered in list order, the first operand
# before the second.
law tensor_deliver_in_order:
  for +target: Nat
  for +shape: Shape.Shape
  for gradient: Core.Dynamic
  for rest: List<&1, Backward.Contribution>
  for +count: Nat
  for slots: List<&1, Gradient.Slot>
  {Backward.deliver(Backward.Contribution{target, shape, gradient} <> rest, count, Done{slots}) ==
    Backward.deliver(rest, count, Backward.bump(slots, count, target, shape, gradient, Backward.at_head(count, target))) :
    Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}

# A contribution to node target reaches exactly the slot of that node, whose
# Gradient.accumulate adds it to the held total (or moves it in); every other
# slot is kept, whatever the list holds.
law tensor_bump_target:
  for newer: List<&1, Gradient.Slot>
  for slot: Gradient.Slot
  for rest: List<&1, Gradient.Slot>
  for +shape: Shape.Shape
  for gradient: Core.Dynamic
  for +older: Nat
  for fits: {List.length(&1, Gradient.Slot, rest) == older : Nat}
  {Backward.bump(List.append(&1, Gradient.Slot, newer, slot <> rest), 1n+Nat.add(older, List.length(&1, Gradient.Slot, newer)),
      older, shape, gradient, Backward.at_head(1n+Nat.add(older, List.length(&1, Gradient.Slot, newer)), older)) ==
    relisted(newer, Backward.placed(rest, Gradient.accumulate(shape, slot, gradient))) :
    Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}

# After the sweep, an entry that received a gradient g accumulates it into its
# slot with Gradient.accumulate, so a repeated backward adds; an entry that
# received nothing keeps its slot.
law tensor_merge_accumulates:
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for g: Core.Dynamic
  for more: List<&1, Gradient.Slot>
  {Backward.merge(Backward.Entry{node, shape, value, saved, slot} <> rest, Gradient.Held{g} <> more) ==
    Backward.merged_entry(node, shape, value, saved, Gradient.accumulate(shape, slot, g), Backward.merge(rest, more)) :
    List<&1, Backward.Entry> & Result<&2, &2, Backward.Error, Unit>}

law tensor_merge_keeps_unreached:
  for +node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  for slot: Gradient.Slot
  for rest: List<&1, Backward.Entry>
  for more: List<&1, Gradient.Slot>
  {Backward.merge(Backward.Entry{node, shape, value, saved, slot} <> rest, Gradient.Unset{} <> more) ==
    Backward.merged_entry(node, shape, value, saved, Gradient.Accumulated{slot}, Backward.merge(rest, more)) :
    List<&1, Backward.Entry> & Result<&2, &2, Backward.Error, Unit>}

# zero_grad empties every slot, like PyTorch's zero_grad(set_to_none=True),
# and keeps every node.
law tensor_zero_grad_empties:
  for entries: List<&1, Backward.Entry>
  {tape_slots(Backward.zero_grad(Backward.Tape{entries})) == Backward.unsets(List.length(&1, Backward.Entry, entries)) :
    List<&1, Gradient.Slot>}

law tensor_zero_grad_keeps_nodes:
  for entries: List<&1, Backward.Entry>
  {tape_nodes(Backward.zero_grad(Backward.Tape{entries})) == nodes_of(entries) : List<&2, Backward.Node>}

# d(sum a)/da through the live broadcast_to_pointwise law: for a replayed
# gradient g whose shape broadcasts to exactly a's shape, the Sum VJP sends one
# contribution, to a, with a's dimensions, and every position i holds the
# gradient word at left_map(plan, i), its right-aligned broadcast projection;
# for the rank-0 g of a sum, that is g's only word at every position.
law tensor_sum_vjp_pointwise:
  for +a: Nat
  for +shape: Shape.Shape
  for +input: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(shape, input) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(input) : List<&2, U32>}
  for saved: Backward.Saved
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  SoleContract<a, input, index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    Backward.vjp(Backward.Sum{a, input}, saved, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}})>
