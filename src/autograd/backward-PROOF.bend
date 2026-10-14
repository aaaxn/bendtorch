import Base
import ./backward.bend as Backward
import ./backward-LAWS.bend as Laws

# Bool and Nat facts.

def truth(value: Bool) -> Data:
  match value:
    case True{}:
      Unit
    case False{}:
      Empty

def true_false(equal: {True{} == False{} : Bool}) -> Empty:
  %equal : truth(_)
  Unit{}

def eq_refl(+number: Nat) -> {True{} == Nat.is_eq(number, number) : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      eq_refl(rest)

def add_succ(+left: Nat, +right: Nat) -> {Nat.add(left, 1n+right) == 1n+Nat.add(left, right) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+rest:
      Equal.cong(Nat, Nat, number => 1n+number, Nat.add(rest, 1n+right), 1n+Nat.add(rest, right), add_succ(rest, right))

def succ_add_ne(+left: Nat, +right: Nat) -> {False{} == Nat.is_eq(1n+Nat.add(left, right), right) : Bool}:
  match right:
    case 0n:
      {==}
    case 1n+rest:
      %Equal.sym(Nat, Nat.add(left, 1n+rest), 1n+Nat.add(left, rest), add_succ(left, rest)) :
        {False{} == Nat.is_eq(1n+_, 1n+rest) : Bool}
      succ_add_ne(left, rest)

def length_append(-A: Data, +xs: List<&2, A>, +ys: List<&2, A>) ->
  {List.length(&2, A, List.append(&2, A, xs, ys)) == Nat.add(List.length(&2, A, xs), List.length(&2, A, ys)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case Con{head, tail}:
      Equal.cong(Nat, Nat, number => 1n+number, List.length(&2, A, List.append(&2, A, tail, ys)),
        Nat.add(List.length(&2, A, tail), List.length(&2, A, ys)), length_append(A, tail, ys))

# An element with older elements after it never has the id of one of them.
def append_ne(-A: Data, +xs: List<&2, A>, +item: A, +rest: List<&2, A>) ->
  {False{} == Nat.is_eq(List.length(&2, A, List.append(&2, A, xs, item <> rest)), List.length(&2, A, rest)) : Bool}:
  %Equal.sym(Nat, List.length(&2, A, List.append(&2, A, xs, item <> rest)),
    Nat.add(List.length(&2, A, xs), 1n+List.length(&2, A, rest)), length_append(A, xs, item <> rest)) :
    {False{} == Nat.is_eq(_, List.length(&2, A, rest)) : Bool}
  %Equal.sym(Nat, Nat.add(List.length(&2, A, xs), 1n+List.length(&2, A, rest)),
    1n+Nat.add(List.length(&2, A, xs), List.length(&2, A, rest)), add_succ(List.length(&2, A, xs), List.length(&2, A, rest))) :
    {False{} == Nat.is_eq(_, List.length(&2, A, rest)) : Bool}
  succ_add_ne(List.length(&2, A, xs), List.length(&2, A, rest))

def not_lt_succ_ne(+id: Nat, +bound: Nat, outside: {Nat.is_lt(id, 1n+bound) == False{} : Bool}) ->
  {False{} == Nat.is_eq(bound, id) : Bool}:
  match id bound:
    case 0n bound:
      Empty.absurd({False{} == Nat.is_eq(bound, 0n) : Bool}, true_false(outside))
    case 1n+rest 0n:
      {==}
    case 1n+rest 1n+smaller:
      not_lt_succ_ne(rest, smaller, outside)

def not_lt_succ_lt(+id: Nat, +bound: Nat, outside: {Nat.is_lt(id, 1n+bound) == False{} : Bool}) ->
  {Nat.is_lt(id, bound) == False{} : Bool}:
  match id bound:
    case 0n bound:
      Empty.absurd({Nat.is_lt(0n, bound) == False{} : Bool}, true_false(outside))
    case 1n+rest 0n:
      {==}
    case 1n+rest 1n+smaller:
      not_lt_succ_lt(rest, smaller, outside)

# Vector-Jacobian products.

def Laws.vjp_leaf(g):
  {==}

def Laws.vjp_add(a, b, g):
  {==}

def Laws.vjp_sub(a, b, g):
  {==}

def Laws.vjp_mul(a, b, left, right, g):
  {==}

def Laws.vjp_neg(a, g):
  {==}

def Laws.backward_add(x, y):
  {==}

def Laws.backward_sub(x, y):
  {==}

def Laws.backward_mul(x, y):
  {==}

def Laws.backward_neg(x):
  {==}

# The seed.

def Laws.seed_selected_output(newer, node, rest):
  match newer:
    case Nil{}:
      %eq_refl(List.length(&2, Backward.Node, rest)) :
        {Backward.chosen(node, rest, List.length(&2, Backward.Node, rest),
          unit => Backward.pass(rest, List.length(&2, Backward.Node, rest), List.length(&2, Backward.Node, rest)), _) ==
          Some{1.0} <> Backward.sweep(rest,
            Backward.contribute(node, Some{1.0}, List.length(&2, Backward.Node, rest), Backward.nones(List.length(&2, Backward.Node, rest))),
            List.length(&2, Backward.Node, rest)) : List<&2, Maybe<&2, F32>>}
      {==}
    case Con{head, tail}:
      %append_ne(Backward.Node, tail, node, rest) :
        {Backward.chosen(head, List.append(&2, Backward.Node, tail, node <> rest),
          List.length(&2, Backward.Node, List.append(&2, Backward.Node, tail, node <> rest)),
          unit => Backward.pass(List.append(&2, Backward.Node, tail, node <> rest),
            List.length(&2, Backward.Node, List.append(&2, Backward.Node, tail, node <> rest)), List.length(&2, Backward.Node, rest)), _) ==
          None{} <> List.append(&2, Maybe<&2, F32>, Backward.nones(List.length(&2, Backward.Node, tail)),
            Some{1.0} <> Backward.sweep(rest,
              Backward.contribute(node, Some{1.0}, List.length(&2, Backward.Node, rest), Backward.nones(List.length(&2, Backward.Node, rest))),
              List.length(&2, Backward.Node, rest))) : List<&2, Maybe<&2, F32>>}
      Equal.cong(List<&2, Maybe<&2, F32>>, List<&2, Maybe<&2, F32>>, xs => None{} <> xs,
        Backward.pass(List.append(&2, Backward.Node, tail, node <> rest),
          List.length(&2, Backward.Node, List.append(&2, Backward.Node, tail, node <> rest)), List.length(&2, Backward.Node, rest)),
        List.append(&2, Maybe<&2, F32>, Backward.nones(List.length(&2, Backward.Node, tail)),
          Some{1.0} <> Backward.sweep(rest,
            Backward.contribute(node, Some{1.0}, List.length(&2, Backward.Node, rest), Backward.nones(List.length(&2, Backward.Node, rest))),
            List.length(&2, Backward.Node, rest))),
        Laws.seed_selected_output(tail, node, rest))

# Reachability: a gradient is present exactly where the flags say the output
# reaches.

def is_some_accumulate(slot: Maybe<&2, F32>, +value: F32) ->
  {Maybe.is_some(&2, F32, Backward.accumulate(slot, Some{value})) == True{} : Bool}:
  match slot:
    case None{}:
      {==}
    case Some{held}:
      {==}

def somes_bumped(slot: Maybe<&2, F32>, +rest: List<&2, Maybe<&2, F32>>, +below: Nat, +target: Nat, +value: F32, flag: Bool,
  ih: {Laws.somes(Backward.bump(rest, below, target, value)) == Laws.mark(Laws.somes(rest), below, target) : List<&2, Bool>}) ->
  {Laws.somes(Backward.bumped(slot, rest, value, unit => Backward.bump(rest, below, target, value), flag)) ==
    Laws.marked(Maybe.is_some(&2, F32, slot), Laws.somes(rest), unit => Laws.mark(Laws.somes(rest), below, target), flag) : List<&2, Bool>}:
  match flag:
    case True{}:
      Equal.cong(Bool, List<&2, Bool>, found => found <> Laws.somes(rest),
        Maybe.is_some(&2, F32, Backward.accumulate(slot, Some{value})), True{}, is_some_accumulate(slot, value))
    case False{}:
      Equal.cong(List<&2, Bool>, List<&2, Bool>, xs => Maybe.is_some(&2, F32, slot) <> xs,
        Laws.somes(Backward.bump(rest, below, target, value)), Laws.mark(Laws.somes(rest), below, target), ih)

def somes_bump(slots: List<&2, Maybe<&2, F32>>, +count: Nat, +target: Nat, +value: F32) ->
  {Laws.somes(Backward.bump(slots, count, target, value)) == Laws.mark(Laws.somes(slots), count, target) : List<&2, Bool>}:
  match slots count:
    case Nil{} count:
      {==}
    case Con{slot, rest} 0n:
      {==}
    case Con{slot, +rest} 1n+below:
      somes_bumped(slot, rest, below, target, value, Nat.is_eq(below, target), somes_bump(rest, below, target, value))

def contribution_targets(contributions: List<&2, Backward.Contribution>) -> List<&2, Nat>:
  match contributions:
    case Nil{}:
      Nil{}
    case Con{Backward.Contribution{target, value}, rest}:
      target <> contribution_targets(rest)

def somes_deliver(contributions: List<&2, Backward.Contribution>, +count: Nat, +slots: List<&2, Maybe<&2, F32>>) ->
  {Laws.somes(Backward.deliver(contributions, count, slots)) ==
    Laws.mark_all(contribution_targets(contributions), count, Laws.somes(slots)) : List<&2, Bool>}:
  match contributions:
    case Nil{}:
      {==}
    case Con{Backward.Contribution{+target, +value}, rest}:
      %somes_bump(slots, count, target, value) :
        {Laws.somes(Backward.deliver(rest, count, Backward.bump(slots, count, target, value))) ==
          Laws.mark_all(contribution_targets(rest), count, _) : List<&2, Bool>}
      somes_deliver(rest, count, Backward.bump(slots, count, target, value))

def vjp_targets(node: Backward.Node, +g: F32) -> {contribution_targets(Backward.vjp(node, g)) == Laws.targets(node) : List<&2, Nat>}:
  match node:
    case Backward.Leaf{}:
      {==}
    case Backward.Add{a, b}:
      {==}
    case Backward.Sub{a, b}:
      {==}
    case Backward.Mul{a, b, left, right}:
      {==}
    case Backward.Neg{a}:
      {==}

def somes_contribute(+node: Backward.Node, gradient: Maybe<&2, F32>, +count: Nat, +slots: List<&2, Maybe<&2, F32>>) ->
  {Laws.somes(Backward.contribute(node, gradient, count, slots)) ==
    Laws.spread(node, Maybe.is_some(&2, F32, gradient), count, Laws.somes(slots)) : List<&2, Bool>}:
  match gradient:
    case None{}:
      {==}
    case Some{+value}:
      %vjp_targets(node, value) :
        {Laws.somes(Backward.deliver(Backward.vjp(node, value), count, slots)) == Laws.mark_all(_, count, Laws.somes(slots)) : List<&2, Bool>}
      somes_deliver(Backward.vjp(node, value), count, slots)

def sweep_reach(nodes: List<&2, Backward.Node>, gradients: List<&2, Maybe<&2, F32>>, +count: Nat) ->
  {Laws.somes(Backward.sweep(nodes, gradients, count)) == Laws.reach(nodes, Laws.somes(gradients), count) : List<&2, Bool>}:
  match nodes gradients count:
    case Nil{} gradients count:
      {==}
    case Con{node, rest} Nil{} count:
      {==}
    case Con{node, rest} Con{gradient, pending} 0n:
      {==}
    case Con{+node, rest} Con{+gradient, +pending} 1n+below:
      %somes_contribute(node, gradient, below, pending) :
        {Maybe.is_some(&2, F32, gradient) <> Laws.somes(Backward.sweep(rest, Backward.contribute(node, gradient, below, pending), below)) ==
          Maybe.is_some(&2, F32, gradient) <> Laws.reach(rest, _, below) : List<&2, Bool>}
      Equal.cong(List<&2, Bool>, List<&2, Bool>, xs => Maybe.is_some(&2, F32, gradient) <> xs,
        Laws.somes(Backward.sweep(rest, Backward.contribute(node, gradient, below, pending), below)),
        Laws.reach(rest, Laws.somes(Backward.contribute(node, gradient, below, pending)), below),
        sweep_reach(rest, Backward.contribute(node, gradient, below, pending), below))

def somes_nones(+count: Nat) -> {Laws.somes(Backward.nones(count)) == Laws.falses(count) : List<&2, Bool>}:
  match count:
    case 0n:
      {==}
    case 1n+rest:
      Equal.cong(List<&2, Bool>, List<&2, Bool>, xs => False{} <> xs, Laws.somes(Backward.nones(rest)), Laws.falses(rest), somes_nones(rest))

def chosen_reached(+node: Backward.Node, +rest: List<&2, Backward.Node>, +below: Nat, +output: Nat, flag: Bool,
  ih: {Laws.somes(Backward.pass(rest, below, output)) == Laws.reached(rest, below, output) : List<&2, Bool>}) ->
  {Laws.somes(Backward.chosen(node, rest, below, unit => Backward.pass(rest, below, output), flag)) ==
    Laws.reached_chosen(node, rest, below, unit => Laws.reached(rest, below, output), flag) : List<&2, Bool>}:
  match flag:
    case True{}:
      %somes_nones(below) :
        {Laws.somes(Backward.sweep(node <> rest, Some{1.0} <> Backward.nones(below), 1n+below)) ==
          Laws.reach(node <> rest, True{} <> _, 1n+below) : List<&2, Bool>}
      sweep_reach(node <> rest, Some{1.0} <> Backward.nones(below), 1n+below)
    case False{}:
      Equal.cong(List<&2, Bool>, List<&2, Bool>, xs => False{} <> xs,
        Laws.somes(Backward.pass(rest, below, output)), Laws.reached(rest, below, output), ih)

def pass_reached(nodes: List<&2, Backward.Node>, +count: Nat, +output: Nat) ->
  {Laws.somes(Backward.pass(nodes, count, output)) == Laws.reached(nodes, count, output) : List<&2, Bool>}:
  match nodes count:
    case Nil{} count:
      {==}
    case Con{node, rest} 0n:
      {==}
    case Con{+node, +rest} 1n+below:
      chosen_reached(node, rest, below, output, Nat.is_eq(below, output), pass_reached(rest, below, output))

def Laws.reached_exactly(graph, output):
  pass_reached(graph, List.length(&2, Backward.Node, graph), output)

def Laws.unused_leaf_none(x, y, z):
  {==}

# Fan-out accumulation.

def Laws.accumulate_moves_in(contribution):
  {==}

def Laws.accumulate_adds_total_first(held, value):
  {==}

def Laws.accumulate_keeps_without_contribution(held):
  {==}

def Laws.bump_target(newer, slot, rest, value):
  match newer:
    case Nil{}:
      %eq_refl(List.length(&2, Maybe<&2, F32>, rest)) :
        {Backward.bumped(slot, rest, value,
          unit => Backward.bump(rest, List.length(&2, Maybe<&2, F32>, rest), List.length(&2, Maybe<&2, F32>, rest), value), _) ==
          Backward.accumulate(slot, Some{value}) <> rest : List<&2, Maybe<&2, F32>>}
      {==}
    case Con{head, tail}:
      %append_ne(Maybe<&2, F32>, tail, slot, rest) :
        {Backward.bumped(head, List.append(&2, Maybe<&2, F32>, tail, slot <> rest), value,
          unit => Backward.bump(List.append(&2, Maybe<&2, F32>, tail, slot <> rest),
            List.length(&2, Maybe<&2, F32>, List.append(&2, Maybe<&2, F32>, tail, slot <> rest)), List.length(&2, Maybe<&2, F32>, rest), value), _) ==
          head <> List.append(&2, Maybe<&2, F32>, tail, Backward.accumulate(slot, Some{value}) <> rest) : List<&2, Maybe<&2, F32>>}
      Equal.cong(List<&2, Maybe<&2, F32>>, List<&2, Maybe<&2, F32>>, xs => head <> xs,
        Backward.bump(List.append(&2, Maybe<&2, F32>, tail, slot <> rest),
          List.length(&2, Maybe<&2, F32>, List.append(&2, Maybe<&2, F32>, tail, slot <> rest)), List.length(&2, Maybe<&2, F32>, rest), value),
        List.append(&2, Maybe<&2, F32>, tail, Backward.accumulate(slot, Some{value}) <> rest),
        Laws.bump_target(tail, slot, rest, value))

def Laws.fan_out_arrival_order(x, a, b, c):
  {==}

# Ids.

def Laws.lookup_exact(A, newer, item, rest):
  match newer:
    case Nil{}:
      %eq_refl(List.length(&2, A, rest)) :
        {Backward.picked(A, item, unit => Backward.find(A, rest, List.length(&2, A, rest), List.length(&2, A, rest)), _) ==
          Some{item} : Maybe<&2, A>}
      {==}
    case Con{head, tail}:
      %append_ne(A, tail, item, rest) :
        {Backward.picked(A, head, unit => Backward.find(A, List.append(&2, A, tail, item <> rest),
          List.length(&2, A, List.append(&2, A, tail, item <> rest)), List.length(&2, A, rest)), _) == Some{item} : Maybe<&2, A>}
      Laws.lookup_exact(A, tail, item, rest)

def find_unknown(-A: Data, +xs: List<&2, A>, +id: Nat, +outside: {Nat.is_lt(id, List.length(&2, A, xs)) == False{} : Bool}) ->
  {Backward.find(A, xs, List.length(&2, A, xs), id) == None{} : Maybe<&2, A>}:
  match xs:
    case Nil{}:
      {==}
    case Con{head, tail}:
      %not_lt_succ_ne(id, List.length(&2, A, tail), outside) :
        {Backward.picked(A, head, unit => Backward.find(A, tail, List.length(&2, A, tail), id), _) == None{} : Maybe<&2, A>}
      find_unknown(A, tail, id, not_lt_succ_lt(id, List.length(&2, A, tail), outside))

def Laws.lookup_unknown(A, items, id, outside):
  find_unknown(A, items, id, outside)

def Laws.bump_unknown(gradients, target, value, outside):
  match gradients:
    case Nil{}:
      {==}
    case Con{slot, tail}:
      %not_lt_succ_ne(target, List.length(&2, Maybe<&2, F32>, tail), outside) :
        {Backward.bumped(slot, tail, value,
          unit => Backward.bump(tail, List.length(&2, Maybe<&2, F32>, tail), target, value), _) == slot <> tail : List<&2, Maybe<&2, F32>>}
      Equal.cong(List<&2, Maybe<&2, F32>>, List<&2, Maybe<&2, F32>>, xs => slot <> xs,
        Backward.bump(tail, List.length(&2, Maybe<&2, F32>, tail), target, value), tail,
        Laws.bump_unknown(tail, target, value, not_lt_succ_lt(target, List.length(&2, Maybe<&2, F32>, tail), outside)))

def Laws.value_unknown(entries, id, outside):
  %Equal.sym(Maybe<&2, Backward.Entry>, Backward.lookup(Backward.Entry, entries, id), None{},
    find_unknown(Backward.Entry, entries, id, outside)) :
    {Backward.entry_value(_) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, F32>}
  {==}

def Laws.grad_unknown(entries, id, outside):
  %Equal.sym(Maybe<&2, Backward.Entry>, Backward.lookup(Backward.Entry, entries, id), None{},
    find_unknown(Backward.Entry, entries, id, outside)) :
    {Backward.entry_grad(_) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, Maybe<&2, F32>>}
  {==}

def Laws.record_unknown_left(entries, kind, a, b, outside):
  %Equal.sym(Result<&2, &2, Backward.Error, F32>, Backward.value(Backward.Tape{entries}, a), Fail{Backward.UnknownNode{}},
    Laws.value_unknown(entries, a, outside)) :
    {Backward.operands(entries, kind, a, b, _, Backward.value(Backward.Tape{entries}, b)) ==
      (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Nat>}
  {==}

def operands_right(+entries: List<&2, Backward.Entry>, kind: Backward.Binary, +a: Nat, +b: Nat, left: Result<&2, &2, Backward.Error, F32>) ->
  {Backward.operands(entries, kind, a, b, left, Fail{Backward.UnknownNode{}}) ==
    (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Nat>}:
  match left:
    case Fail{error}:
      {==}
    case Done{x}:
      {==}

def Laws.record_unknown_right(entries, kind, a, b, outside):
  %Equal.sym(Result<&2, &2, Backward.Error, F32>, Backward.value(Backward.Tape{entries}, b), Fail{Backward.UnknownNode{}},
    Laws.value_unknown(entries, b, outside)) :
    {Backward.operands(entries, kind, a, b, Backward.value(Backward.Tape{entries}, a), _) ==
      (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Nat>}
  operands_right(entries, kind, a, b, Backward.value(Backward.Tape{entries}, a))

def Laws.neg_unknown(entries, a, outside):
  %Equal.sym(Result<&2, &2, Backward.Error, F32>, Backward.value(Backward.Tape{entries}, a), Fail{Backward.UnknownNode{}},
    Laws.value_unknown(entries, a, outside)) :
    {Backward.negated(entries, a, _) ==
      (Backward.Tape{entries}, Fail{Backward.UnknownNode{}}) : Backward.Tape & Result<&2, &2, Backward.Error, Nat>}
  {==}

# backward's checks.

def nodes_length(+entries: List<&2, Backward.Entry>) ->
  {List.length(&2, Backward.Entry, entries) == List.length(&2, Backward.Node, Backward.nodes(entries)) : Nat}:
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, value, slot}, rest}:
      Equal.cong(Nat, Nat, number => 1n+number, List.length(&2, Backward.Entry, rest),
        List.length(&2, Backward.Node, Backward.nodes(rest)), nodes_length(rest))

def Laws.backward_unknown(entries, output, outside):
  %nodes_length(entries) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), _, output, Nat.is_lt(output, _),
      Backward.wired(Backward.nodes(entries), _))) == Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Entry, entries)), False{}, outside) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output, _,
      Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)))) ==
      Fail{Backward.UnknownNode{}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  {==}

def Laws.backward_dangling(entries, output, inside, unsound):
  %nodes_length(entries) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), _, output, Nat.is_lt(output, _),
      Backward.wired(Backward.nodes(entries), _))) == Fail{Backward.DanglingOperand{}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Entry, entries)), True{}, inside) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output, _,
      Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)))) ==
      Fail{Backward.DanglingOperand{}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Bool, Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)), False{}, unsound) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output, True{}, _)) ==
      Fail{Backward.DanglingOperand{}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  {==}

def Laws.backward_success(entries, output, inside, sound):
  %nodes_length(entries) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), _, output, Nat.is_lt(output, _),
      Backward.wired(Backward.nodes(entries), _))) ==
      Done{Backward.Tape{Backward.merge(entries, Backward.pass(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output))}} :
      Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Bool, Nat.is_lt(output, List.length(&2, Backward.Entry, entries)), True{}, inside) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output, _,
      Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)))) ==
      Done{Backward.Tape{Backward.merge(entries, Backward.pass(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output))}} :
      Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Bool, Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)), True{}, sound) :
    {Backward.merged(entries, Backward.checked(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output, True{}, _)) ==
      Done{Backward.Tape{Backward.merge(entries, Backward.pass(Backward.nodes(entries), List.length(&2, Backward.Entry, entries), output))}} :
      Result<&2, &2, Backward.Error, Backward.Tape>}
  {==}

# Operand bounds.

def and_true(+value: Bool) -> {Bool.and(value, True{}) == value : Bool}:
  match value:
    case True{}:
      {==}
    case False{}:
      {==}

def and_false(+value: Bool) -> {Bool.and(value, False{}) == False{} : Bool}:
  match value:
    case True{}:
      {==}
    case False{}:
      {==}

def operands_bounded(+node: Backward.Node, +bound: Nat) ->
  {Backward.operands_below(node, bound) == Laws.bounded(Laws.targets(node), bound) : Bool}:
  match node:
    case Backward.Leaf{}:
      {==}
    case Backward.Add{a, b}:
      Equal.cong(Bool, Bool, flag => Bool.and(Nat.is_lt(a, bound), flag), Nat.is_lt(b, bound), Bool.and(Nat.is_lt(b, bound), True{}),
        Equal.sym(Bool, Bool.and(Nat.is_lt(b, bound), True{}), Nat.is_lt(b, bound), and_true(Nat.is_lt(b, bound))))
    case Backward.Sub{a, b}:
      Equal.cong(Bool, Bool, flag => Bool.and(Nat.is_lt(a, bound), flag), Nat.is_lt(b, bound), Bool.and(Nat.is_lt(b, bound), True{}),
        Equal.sym(Bool, Bool.and(Nat.is_lt(b, bound), True{}), Nat.is_lt(b, bound), and_true(Nat.is_lt(b, bound))))
    case Backward.Mul{a, b, left, right}:
      Equal.cong(Bool, Bool, flag => Bool.and(Nat.is_lt(a, bound), flag), Nat.is_lt(b, bound), Bool.and(Nat.is_lt(b, bound), True{}),
        Equal.sym(Bool, Bool.and(Nat.is_lt(b, bound), True{}), Nat.is_lt(b, bound), and_true(Nat.is_lt(b, bound))))
    case Backward.Neg{a}:
      Equal.sym(Bool, Bool.and(Nat.is_lt(a, bound), True{}), Nat.is_lt(a, bound), and_true(Nat.is_lt(a, bound)))

def Laws.wired_matches_operand_bounds(graph):
  match graph:
    case Nil{}:
      {==}
    case Con{node, rest}:
      %Equal.sym(Bool, Backward.operands_below(node, List.length(&2, Backward.Node, rest)),
        Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), operands_bounded(node, List.length(&2, Backward.Node, rest))) :
        {Bool.and(_, Backward.wired(rest, List.length(&2, Backward.Node, rest))) ==
          Bool.and(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), Laws.well_formed(rest)) : Bool}
      Equal.cong(Bool, Bool, flag => Bool.and(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, rest)), flag),
        Backward.wired(rest, List.length(&2, Backward.Node, rest)), Laws.well_formed(rest), Laws.wired_matches_operand_bounds(rest))

def wired_entries(+entries: List<&2, Backward.Entry>) ->
  {Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)) == Laws.well_formed(Backward.nodes(entries)) : Bool}:
  %Equal.sym(Nat, List.length(&2, Backward.Entry, entries), List.length(&2, Backward.Node, Backward.nodes(entries)), nodes_length(entries)) :
    {Backward.wired(Backward.nodes(entries), _) == Laws.well_formed(Backward.nodes(entries)) : Bool}
  Laws.wired_matches_operand_bounds(Backward.nodes(entries))

def head_ill_formed(+entry: Backward.Entry, +rest: List<&2, Backward.Entry>,
  unbound: {Laws.bounded(Laws.targets(Laws.entry_node(entry)), List.length(&2, Backward.Entry, rest)) == False{} : Bool}) ->
  {Laws.well_formed(Backward.nodes(entry <> rest)) == False{} : Bool}:
  match entry:
    case Backward.Entry{node, value, slot}:
      %nodes_length(rest) :
        {Bool.and(Laws.bounded(Laws.targets(node), _), Laws.well_formed(Backward.nodes(rest))) == False{} : Bool}
      %Equal.sym(Bool, Laws.bounded(Laws.targets(node), List.length(&2, Backward.Entry, rest)), False{}, unbound) :
        {Bool.and(_, Laws.well_formed(Backward.nodes(rest))) == False{} : Bool}
      {==}

def unbound_ill_formed(+newer: List<&2, Backward.Entry>, +entry: Backward.Entry, +rest: List<&2, Backward.Entry>,
  +unbound: {Laws.bounded(Laws.targets(Laws.entry_node(entry)), List.length(&2, Backward.Entry, rest)) == False{} : Bool}) ->
  {Laws.well_formed(Backward.nodes(List.append(&2, Backward.Entry, newer, entry <> rest))) == False{} : Bool}:
  match newer:
    case Nil{}:
      head_ill_formed(entry, rest, unbound)
    case Con{Backward.Entry{node, value, slot}, tail}:
      %Equal.sym(Bool, Laws.well_formed(Backward.nodes(List.append(&2, Backward.Entry, tail, entry <> rest))), False{},
        unbound_ill_formed(tail, entry, rest, unbound)) :
        {Bool.and(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, Backward.nodes(List.append(&2, Backward.Entry, tail, entry <> rest)))), _) ==
          False{} : Bool}
      and_false(Laws.bounded(Laws.targets(node), List.length(&2, Backward.Node, Backward.nodes(List.append(&2, Backward.Entry, tail, entry <> rest)))))

def Laws.backward_rejects_unbound_operand(newer, entry, rest, output, inside, unbound):
  Laws.backward_dangling(List.append(&2, Backward.Entry, newer, entry <> rest), output, inside,
    Equal.trans(Bool, Backward.wired(Backward.nodes(List.append(&2, Backward.Entry, newer, entry <> rest)),
        List.length(&2, Backward.Entry, List.append(&2, Backward.Entry, newer, entry <> rest))),
      Laws.well_formed(Backward.nodes(List.append(&2, Backward.Entry, newer, entry <> rest))), False{},
      wired_entries(List.append(&2, Backward.Entry, newer, entry <> rest)),
      unbound_ill_formed(newer, entry, rest, unbound)))

def Laws.backward_accepts_bound_operands(entries, output, inside, bound):
  Laws.backward_success(entries, output, inside,
    Equal.trans(Bool, Backward.wired(Backward.nodes(entries), List.length(&2, Backward.Entry, entries)),
      Laws.well_formed(Backward.nodes(entries)), True{}, wired_entries(entries), bound))

# Repeated backward.

def nodes_merge(+entries: List<&2, Backward.Entry>, +gradients: List<&2, Maybe<&2, F32>>) ->
  {Backward.nodes(entries) == Backward.nodes(Backward.merge(entries, gradients)) : List<&2, Backward.Node>}:
  match entries gradients:
    case Nil{} gradients:
      {==}
    case Con{Backward.Entry{node, value, slot}, rest} Nil{}:
      {==}
    case Con{Backward.Entry{node, value, slot}, rest} Con{gradient, more}:
      Equal.cong(List<&2, Backward.Node>, List<&2, Backward.Node>, xs => node <> xs,
        Backward.nodes(rest), Backward.nodes(Backward.merge(rest, more)), nodes_merge(rest, more))

def Laws.backward_twice_accumulates(entries, output, found, computed):
  %Equal.sym(Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>, Backward.gradients(Backward.Tape{entries}, output), Done{found}, computed) :
    {Laws.again(Backward.merged(entries, _), output) ==
      Done{Backward.Tape{Backward.merge(Backward.merge(entries, found), found)}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  %nodes_merge(entries, found) :
    {Backward.merged(Backward.merge(entries, found), Backward.planned(_, output)) ==
      Done{Backward.Tape{Backward.merge(Backward.merge(entries, found), found)}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  %Equal.sym(Result<&2, &2, Backward.Error, List<&2, Maybe<&2, F32>>>, Backward.gradients(Backward.Tape{entries}, output), Done{found}, computed) :
    {Backward.merged(Backward.merge(entries, found), _) ==
      Done{Backward.Tape{Backward.merge(Backward.merge(entries, found), found)}} : Result<&2, &2, Backward.Error, Backward.Tape>}
  {==}

def Laws.backward_twice_mul(x, y):
  {==}

# zero_grad.

def Laws.zero_grad_empties(entries):
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, value, slot}, rest}:
      Equal.cong(List<&2, Maybe<&2, F32>>, List<&2, Maybe<&2, F32>>, xs => None{} <> xs,
        Laws.slots(Backward.cleared(rest)), Backward.nones(List.length(&2, Backward.Entry, rest)), Laws.zero_grad_empties(rest))

def Laws.zero_grad_keeps_nodes(entries):
  match entries:
    case Nil{}:
      {==}
    case Con{Backward.Entry{node, value, slot}, rest}:
      Equal.cong(List<&2, Backward.Node>, List<&2, Backward.Node>, xs => node <> xs,
        Backward.nodes(Backward.cleared(rest)), Backward.nodes(rest), Laws.zero_grad_keeps_nodes(rest))
