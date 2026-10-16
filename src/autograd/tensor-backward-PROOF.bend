import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ./broadcast.bend as Broadcast
import ./broadcast-pointwise-LAWS.bend as BroadcastPointwise
import ./broadcast-pointwise-PROOF.bend as BroadcastPointwiseProof
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward
import ./tensor-backward-LAWS.bend as Laws
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast

def eq_refl(+number: Nat) -> {True{} == Nat.is_eq(number, number) : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      eq_refl(rest)

def add_zero(+number: Nat) -> {number == Nat.add(number, 0n) : Nat}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      Equal.cong(Nat, Nat, value => 1n+value, rest, Nat.add(rest, 0n), add_zero(rest))

def add_succ(+left: Nat, -right: Nat) -> {1n+Nat.add(left, right) == Nat.add(left, 1n+right) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+rest:
      Equal.cong(Nat, Nat, value => 1n+value, 1n+Nat.add(rest, right), Nat.add(rest, 1n+right), add_succ(rest, right))

# A node with older nodes below it and extra + 1 newer ones above it never has
# the id of the target.
def older_ne(+older: Nat, -extra: Nat) -> {False{} == Nat.is_eq(Nat.add(older, 1n+extra), older) : Bool}:
  match older:
    case 0n:
      {==}
    case 1n+rest:
      older_ne(rest, extra)

def listed_back(entries: List<&1, Backward.Entry>) ->
  {(entries, Laws.nodes_of(entries)) == Backward.listed(entries) : List<&1, Backward.Entry> & List<&2, Backward.Node>}:
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, shape, value, saved, slot}, rest}:
      %listed_back(rest) :
        {(Backward.Entry{node, shape, value, saved, slot} <> rest, node <> Laws.nodes_of(rest)) ==
          Backward.entry_listed(Backward.Entry{node, shape, value, saved, slot}, _) : List<&1, Backward.Entry> & List<&2, Backward.Node>}
      {==}

def Laws.negate_left_scalar(left, right):
  {==}

def Laws.tensor_vjp_leaf(saved, g):
  {==}

def Laws.tensor_backward_unknown(entries, output, outside):
  %listed_back(entries) :
    {Backward.planned(output, _) == (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) :
      Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Node, Laws.nodes_of(entries))), False{}, outside) :
    {Backward.checked(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output, _,
      Backward.wired(Laws.nodes_of(entries), List.length(&2, Backward.Node, Laws.nodes_of(entries)))) ==
      (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  {==}

def Laws.tensor_backward_dangling(entries, output, inside, unsound):
  %listed_back(entries) :
    {Backward.planned(output, _) == (Backward.Tape{entries}, Fail{Backward.DanglingOperand{}}) :
      Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Node, Laws.nodes_of(entries))), True{}, inside) :
    {Backward.checked(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output, _,
      Backward.wired(Laws.nodes_of(entries), List.length(&2, Backward.Node, Laws.nodes_of(entries)))) ==
      (Backward.Tape{entries}, Fail{Backward.DanglingOperand{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  %Equal.sym(Bool, Backward.wired(Laws.nodes_of(entries), List.length(&2, Backward.Node, Laws.nodes_of(entries))), False{}, unsound) :
    {Backward.checked(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output, True{}, _) ==
      (Backward.Tape{entries}, Fail{Backward.DanglingOperand{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  {==}

def Laws.tensor_backward_success(entries, output, inside, sound):
  %listed_back(entries) :
    {Backward.planned(output, _) ==
      Backward.ran(Backward.pass(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output,
        Backward.at_head(List.length(&2, Backward.Node, Laws.nodes_of(entries)), output))) :
      Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Node, Laws.nodes_of(entries))), True{}, inside) :
    {Backward.checked(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output, _,
      Backward.wired(Laws.nodes_of(entries), List.length(&2, Backward.Node, Laws.nodes_of(entries)))) ==
      Backward.ran(Backward.pass(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output,
        Backward.at_head(List.length(&2, Backward.Node, Laws.nodes_of(entries)), output))) :
      Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  %Equal.sym(Bool, Backward.wired(Laws.nodes_of(entries), List.length(&2, Backward.Node, Laws.nodes_of(entries))), True{}, sound) :
    {Backward.checked(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output, True{}, _) ==
      Backward.ran(Backward.pass(entries, List.length(&2, Backward.Node, Laws.nodes_of(entries)), output,
        Backward.at_head(List.length(&2, Backward.Node, Laws.nodes_of(entries)), output))) :
      Backward.Tape & Result<&2, &2, Backward.Error, Unit>}
  {==}

def and_true(+value: Bool) -> {Bool.and(value, True{}) == value : Bool}:
  match value:
    case True{}:
      {==}
    case False{}:
      {==}

def pair_bounded(+a: Nat, +b: Nat, +bound: Nat) ->
  {Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound)) == Laws.bounded([a, b], bound) : Bool}:
  Equal.cong(Bool, Bool, flag => Bool.and(Nat.is_lt(a, bound), flag), Nat.is_lt(b, bound), Bool.and(Nat.is_lt(b, bound), True{}),
    Equal.sym(Bool, Bool.and(Nat.is_lt(b, bound), True{}), Nat.is_lt(b, bound), and_true(Nat.is_lt(b, bound))))

def operands_bounded(+node: Backward.Node, +bound: Nat) ->
  {Backward.operands_below(node, bound) == Laws.bounded(Laws.targets(node), bound) : Bool}:
  match node:
    case Backward.Leaf{}:
      {==}
    case Backward.Add{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.Sub{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.Mul{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.Sum{a, input}:
      Equal.sym(Bool, Bool.and(Nat.is_lt(a, bound), True{}), Nat.is_lt(a, bound), and_true(Nat.is_lt(a, bound)))
    case Backward.Matmul{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.MatmulNT{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.Relu{a, input}:
      Equal.sym(Bool, Bool.and(Nat.is_lt(a, bound), True{}), Nat.is_lt(a, bound), and_true(Nat.is_lt(a, bound)))
    case Backward.MseLoss{a, b, left, right}:
      pair_bounded(a, b, bound)
    case Backward.CrossEntropy{a, input}:
      Equal.sym(Bool, Bool.and(Nat.is_lt(a, bound), True{}), Nat.is_lt(a, bound), and_true(Nat.is_lt(a, bound)))

def Laws.tensor_wired_matches_operand_bounds(graph):
  match graph:
    case Nil{}:
      {==}
    case Con{node, rest}:
      %Equal.sym(Bool, Backward.operands_below(node, List.length(&2, Backward.Node, rest)),
        Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), operands_bounded(node, List.length(&2, Backward.Node, rest))) :
        {Bool.and(_, Backward.wired(rest, List.length(&2, Backward.Node, rest))) ==
          Bool.and(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), Laws.well_formed(rest)) : Bool}
      Equal.cong(Bool, Bool, flag => Bool.and(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), flag),
        Backward.wired(rest, List.length(&2, Backward.Node, rest)), Laws.well_formed(rest), Laws.tensor_wired_matches_operand_bounds(rest))

def Laws.tensor_seed_selected_output(newer, node, shape, value, saved, slot, rest, older, fits):
  match newer:
    case Nil{}:
      %add_zero(older) :
        {Backward.pass(Backward.Entry{node, shape, value, saved, slot} <> rest, 1n+_, older, Backward.at_head(1n+_, older)) ==
          Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0)) : Backward.Swept()}
      %eq_refl(older) :
        {Backward.pass(Backward.Entry{node, shape, value, saved, slot} <> rest, 1n+older, older, _) ==
          Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0)) : Backward.Swept()}
      {==}
    case Con{head, tail}:
      %older_ne(older, List.length(&1, Backward.Entry, tail)) :
        {Backward.pass(head <> List.append(&1, Backward.Entry, tail, Backward.Entry{node, shape, value, saved, slot} <> rest),
          1n+Nat.add(older, 1n+List.length(&1, Backward.Entry, tail)), older, _) ==
          Backward.finished(head, Gradient.Unset{},
            Laws.newer_unset(tail, Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0)))) : Backward.Swept()}
      %add_succ(older, List.length(&1, Backward.Entry, tail)) :
        {Backward.pass(head <> List.append(&1, Backward.Entry, tail, Backward.Entry{node, shape, value, saved, slot} <> rest),
          1n+_, older, False{}) ==
          Backward.finished(head, Gradient.Unset{},
            Laws.newer_unset(tail, Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0)))) : Backward.Swept()}
      Equal.cong(Backward.Swept(), Backward.Swept(), swept => Backward.finished(head, Gradient.Unset{}, swept),
        Backward.pass(List.append(&1, Backward.Entry, tail, Backward.Entry{node, shape, value, saved, slot} <> rest),
          1n+Nat.add(older, List.length(&1, Backward.Entry, tail)), older,
          Backward.at_head(1n+Nat.add(older, List.length(&1, Backward.Entry, tail)), older)),
        Laws.newer_unset(tail, Backward.seeded(node, shape, value, saved, slot, rest, older, Core.new(shape, 1.0))),
        Laws.tensor_seed_selected_output(tail, node, shape, value, saved, slot, rest, older, fits))

def Laws.tensor_seeded_sweep(node, shape, value, saved, slot, rest, below, ones):
  {==}

def Laws.tensor_unreached_sends_nothing(node, shape, value, saved, slot, rest, pending, below):
  {==}

def Laws.tensor_deliver_in_order(target, shape, gradient, rest, count, slots):
  {==}

def Laws.tensor_bump_target(newer, slot, rest, shape, gradient, older, fits):
  match newer:
    case Nil{}:
      %add_zero(older) :
        {Backward.bump(slot <> rest, 1n+_, older, shape, gradient, Backward.at_head(1n+_, older)) ==
          Backward.placed(rest, Gradient.accumulate(shape, slot, gradient)) : Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}
      %eq_refl(older) :
        {Backward.bump(slot <> rest, 1n+older, older, shape, gradient, _) ==
          Backward.placed(rest, Gradient.accumulate(shape, slot, gradient)) : Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}
      {==}
    case Con{head, tail}:
      %older_ne(older, List.length(&1, Gradient.Slot, tail)) :
        {Backward.bump(head <> List.append(&1, Gradient.Slot, tail, slot <> rest),
          1n+Nat.add(older, 1n+List.length(&1, Gradient.Slot, tail)), older, shape, gradient, _) ==
          Backward.prepend(Gradient.Slot, head,
            Laws.relisted(tail, Backward.placed(rest, Gradient.accumulate(shape, slot, gradient)))) :
          Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}
      %add_succ(older, List.length(&1, Gradient.Slot, tail)) :
        {Backward.bump(head <> List.append(&1, Gradient.Slot, tail, slot <> rest), 1n+_, older, shape, gradient, False{}) ==
          Backward.prepend(Gradient.Slot, head,
            Laws.relisted(tail, Backward.placed(rest, Gradient.accumulate(shape, slot, gradient)))) :
          Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>}
      Equal.cong(Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>, Result<&2, &1, Backward.Error, List<&1, Gradient.Slot>>,
        result => Backward.prepend(Gradient.Slot, head, result),
        Backward.bump(List.append(&1, Gradient.Slot, tail, slot <> rest), 1n+Nat.add(older, List.length(&1, Gradient.Slot, tail)),
          older, shape, gradient, Backward.at_head(1n+Nat.add(older, List.length(&1, Gradient.Slot, tail)), older)),
        Laws.relisted(tail, Backward.placed(rest, Gradient.accumulate(shape, slot, gradient))),
        Laws.tensor_bump_target(tail, slot, rest, shape, gradient, older, fits))

def Laws.tensor_merge_accumulates(node, shape, value, saved, slot, rest, g, more):
  {==}

def Laws.tensor_merge_keeps_unreached(node, shape, value, saved, slot, rest, more):
  {==}

def Laws.tensor_zero_grad_empties(entries):
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, shape, value, saved, slot}, rest}:
      Equal.cong(List<&1, Gradient.Slot>, List<&1, Gradient.Slot>, slots => Gradient.Unset{} <> slots,
        Laws.tape_slots(Backward.zero_grad(Backward.Tape{rest})), Backward.unsets(List.length(&1, Backward.Entry, rest)),
        Laws.tensor_zero_grad_empties(rest))

def Laws.tensor_zero_grad_keeps_nodes(entries):
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, shape, value, saved, slot}, rest}:
      Equal.cong(List<&2, Backward.Node>, List<&2, Backward.Node>, nodes => node <> nodes,
        Laws.tape_nodes(Backward.zero_grad(Backward.Tape{rest})), Laws.nodes_of(rest),
        Laws.tensor_zero_grad_keeps_nodes(rest))

def truth(value: Bool) -> Data:
  match value:
    case True{}:
      Unit
    case False{}:
      Empty

def false_true(equal: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool, False{}, True{}, equal) : truth(_)
  Unit{}

def sum_result(+a: Nat, +input: Shape.Shape, -shape: Shape.Shape, -cell: Nat -> F32, -word: U32 -> F32, -owner: Core.Dynamic,
  result: Result<&2, &1, Broadcast.Error, Core.Dynamic>,
  contract: BroadcastPointwise.BroadcastToContract<input, shape, cell, word, (owner, result)>) ->
  Laws.SoleContract<a, input, cell, Backward.expanded(a, input, (owner, result))>:
  match result:
    case Fail{error}:
      BroadcastPointwise.BroadcastToContract{successful, dimensions, value, kept} = contract
      Empty.absurd(Laws.SoleContract<a, input, cell, Backward.expanded(a, input, (owner, Fail{error}))>, false_true(successful))
    case Done{Core.Packed{output, tensor}}:
      BroadcastPointwise.BroadcastToContract{successful, dimensions, value, kept} = contract
      Laws.SoleContract{{==}, dimensions, value}

# A successful broadcast_to hands its output to the Sum VJP's one
# contribution unchanged.
def sum_contract(+a: Nat, +input: Shape.Shape, -shape: Shape.Shape, -cell: Nat -> F32, -word: U32 -> F32,
  pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>,
  contract: BroadcastPointwise.BroadcastToContract<input, shape, cell, word, pair>) ->
  Laws.SoleContract<a, input, cell, Backward.expanded(a, input, pair)>:
  (owner, result) = pair
  sum_result(a, input, shape, cell, word, owner, result, contract)

def Laws.tensor_sum_vjp_pointwise(a, shape, input, plan, planned, same, saved, fill, ops, fits):
  sum_contract(a, input, shape,
    index => Replayed.value(shape, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    other => Replayed.value(shape, fill, ops, other),
    Broadcast.broadcast_to(Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, input),
    BroadcastPointwise.broadcast_to_pointwise(shape, input, plan, planned, same, fill, ops, fits))
