import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/matmul.bend as Matrix
import ../storage/arith.bend as Arith
import ./broadcast.bend as Broadcast
import ./broadcast-pointwise-LAWS.bend as BroadcastPointwise
import ./broadcast-pointwise-PROOF.bend as BroadcastPointwiseProof
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward
import ./tensor-backward-LAWS.bend as Laws
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/broadcast.bend as PointwiseBroadcast
import ../tensor/transforms.bend as Transforms
import ../tensor/lifecycle.bend as Lifecycle
import ../storage/LAWS.bend as StorageLaws

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

def Laws.tensor_vjp_constant(saved, g):
  {==}

def frozen_nt_unsent(+id: Nat, +a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Backward.Saved, g: Core.Dynamic,
  computed: {Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b))) == False{} : Bool}) ->
  {Backward.nt_frozen(id, a, b, left, right, saved, g, Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b)))) ==
    Backward.unsent(id, Backward.vjp_matmul_nt(a, b, left, right, saved, g)) : Backward.Pass}:
  %Equal.sym(Bool, Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b))), False{}, computed) :
    {Backward.nt_frozen(id, a, b, left, right, saved, g, _) == Backward.unsent(id, Backward.vjp_matmul_nt(a, b, left, right, saved, g)) : Backward.Pass}
  {==}

def Laws.tensor_vjp_frozen_unsent(id, node, saved, g, computed):
  match node:
    case Backward.MatmulNT{+a, +b, +left, +right}:
      frozen_nt_unsent(id, a, b, left, right, saved, g, computed)
    case Backward.Leaf{}:
      {==}
    case Backward.Add{a, b, left, right}:
      {==}
    case Backward.Sub{a, b, left, right}:
      {==}
    case Backward.Mul{a, b, left, right}:
      {==}
    case Backward.Sum{a, input}:
      {==}
    case Backward.SumDims{a, input, kept}:
      {==}
    case Backward.Matmul{a, b, left, right}:
      {==}
    case Backward.Relu{a, input}:
      {==}
    case Backward.MseLoss{a, b, left, right}:
      {==}
    case Backward.CrossEntropy{a, input}:
      {==}
    case Backward.Constant{}:
      {==}
    case Backward.Frozen{inner_id, inner}:
      {==}

# kept_others leaves no contribution to id.
def spared_one(+id: Nat, +target: Nat, +shape: Shape.Shape, -gradient: Core.Dynamic, -rest: List<&1, Backward.Contribution>,
  ih: {Laws.reaches(id, Backward.kept_others(id, rest)) == False{} : Bool},
  decision: Shape.BoolCase<Nat.is_eq(id, target)>) ->
  {Laws.reaches(id, Backward.kept_one(target, shape, gradient, Nat.is_eq(id, target), Backward.kept_others(id, rest))) == False{} : Bool}:
  match decision:
    case Shape.BTrue{equal}:
      %equal : {Laws.reaches(id, Backward.kept_one(target, shape, gradient, _, Backward.kept_others(id, rest))) == False{} : Bool}
      ih
    case Shape.BFalse{+equal}:
      %equal : {Laws.reaches(id, Backward.kept_one(target, shape, gradient, _, Backward.kept_others(id, rest))) == False{} : Bool}
      %equal : {Bool.or(_, Laws.reaches(id, Backward.kept_others(id, rest))) == False{} : Bool}
      %Equal.sym(Bool, Laws.reaches(id, Backward.kept_others(id, rest)), False{}, ih) : {Bool.or(False{}, _) == False{} : Bool}
      {==}

def spared(+id: Nat, items: List<&1, Backward.Contribution>) -> {Laws.reaches(id, Backward.kept_others(id, items)) == False{} : Bool}:
  match items:
    case Nil{}:
      {==}
    case Con{Backward.Contribution{+target, +shape, gradient}, rest}:
      spared_one(id, target, shape, gradient, rest, spared(id, rest), Shape.bool_case(Nat.is_eq(id, target)))

def unsent_spared_result(+id: Nat, -saved: Backward.Saved, -g: Core.Dynamic,
  contributions: Result<&2, &1, Backward.Error, List<&1, Backward.Contribution>>) ->
  {Laws.sends_to(id, Backward.unsent(id, Backward.Pass{saved, g, contributions})) == False{} : Bool}:
  match contributions:
    case Fail{error}:
      {==}
    case Done{items}:
      spared(id, items)

def unsent_spared(+id: Nat, pass: Backward.Pass) -> {Laws.sends_to(id, Backward.unsent(id, pass)) == False{} : Bool}:
  match pass:
    case Backward.Pass{saved, g, contributions}:
      unsent_spared_result(id, saved, g, contributions)

def other_operand(x: Bool, y: Bool, both: {Bool.and(x, Bool.not(y)) == True{} : Bool}) -> {y == False{} : Bool}:
  match x:
    case False{}:
      Empty.absurd({y == False{} : Bool}, Arith.clash(both))
    case True{}:
      match y:
        case True{}:
          Empty.absurd({True{} == False{} : Bool}, Arith.clash(both))
        case False{}:
          {==}

def alone_spared(+id: Nat, +b: Nat, +right: Shape.Shape, -x: Core.Dynamic, -y: Core.Dynamic, -g: Core.Dynamic,
  other: {Nat.is_eq(id, b) == False{} : Bool}, result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) ->
  {Laws.sends_to(id, Backward.Pass{Backward.Operands{x, y}, g, Backward.alone(Backward.multiplied(b, right, result))}) == False{} : Bool}:
  match result:
    case Fail{error}:
      {==}
    case Done{triple}:
      match triple:
        case (kept_left, rest):
          match rest:
            case (kept_right, product):
              %Equal.sym(Bool, Nat.is_eq(id, b), False{}, other) : {Bool.or(_, False{}) == False{} : Bool}
              {==}

def right_spared(+id: Nat, +b: Nat, +right: Shape.Shape, -y: Core.Dynamic, -g: Core.Dynamic,
  other: {Nat.is_eq(id, b) == False{} : Bool}, copy: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) ->
  {Laws.sends_to(id, Backward.nt_alone_second(y, g, Backward.nt_right(b, right, copy, pair))) == False{} : Bool}:
  match pair:
    case (left, flipped):
      alone_spared(id, b, right, left, y, g, other, Backward.weight_product(copy, flipped))

def first_spared(+id: Nat, +b: Nat, +right: Shape.Shape, x: Core.Dynamic, -y: Core.Dynamic,
  other: {Nat.is_eq(id, b) == False{} : Bool}, pair: Core.Dynamic & Core.Dynamic) ->
  {Laws.sends_to(id, Backward.nt_alone_first(b, right, x, y, pair)) == False{} : Bool}:
  match pair:
    case (g, copy):
      right_spared(id, b, right, y, g, other, copy, Core.clone_dynamic(x))

def right_alone_spared(+id: Nat, +b: Nat, +right: Shape.Shape, saved: Backward.Saved, g: Core.Dynamic,
  other: {Nat.is_eq(id, b) == False{} : Bool}) ->
  {Laws.sends_to(id, Backward.vjp_matmul_nt_right(b, right, saved, g)) == False{} : Bool}:
  match saved:
    case Backward.Unsaved{}:
      {==}
    case Backward.Operands{x, y}:
      first_spared(id, b, right, x, y, other, Core.clone_dynamic(g))

def frozen_nt_spared(+id: Nat, +a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Backward.Saved, g: Core.Dynamic,
  decision: Shape.BoolCase<Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b)))>) ->
  {Laws.sends_to(id, Backward.nt_frozen(id, a, b, left, right, saved, g, Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b))))) == False{} : Bool}:
  match decision:
    case Shape.BTrue{+equal}:
      %equal : {Laws.sends_to(id, Backward.nt_frozen(id, a, b, left, right, saved, g, _)) == False{} : Bool}
      right_alone_spared(id, b, right, saved, g,
        other_operand(Nat.is_eq(id, a), Nat.is_eq(id, b), Equal.sym(Bool, True{}, Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b))), equal)))
    case Shape.BFalse{equal}:
      %equal : {Laws.sends_to(id, Backward.nt_frozen(id, a, b, left, right, saved, g, _)) == False{} : Bool}
      unsent_spared(id, Backward.vjp_matmul_nt(a, b, left, right, saved, g))

def Laws.tensor_vjp_frozen_spares(id, node, saved, g):
  match node:
    case Backward.MatmulNT{+a, +b, +left, +right}:
      frozen_nt_spared(id, a, b, left, right, saved, g, Shape.bool_case(Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b)))))
    case Backward.Leaf{}:
      unsent_spared(id, Backward.vjp(Backward.Leaf{}, saved, g))
    case Backward.Add{a, b, left, right}:
      unsent_spared(id, Backward.vjp(Backward.Add{a, b, left, right}, saved, g))
    case Backward.Sub{a, b, left, right}:
      unsent_spared(id, Backward.vjp(Backward.Sub{a, b, left, right}, saved, g))
    case Backward.Mul{a, b, left, right}:
      unsent_spared(id, Backward.vjp(Backward.Mul{a, b, left, right}, saved, g))
    case Backward.Sum{a, input}:
      unsent_spared(id, Backward.vjp(Backward.Sum{a, input}, saved, g))
    case Backward.SumDims{a, input, kept}:
      unsent_spared(id, Backward.vjp(Backward.SumDims{a, input, kept}, saved, g))
    case Backward.Matmul{a, b, left, right}:
      unsent_spared(id, Backward.vjp(Backward.Matmul{a, b, left, right}, saved, g))
    case Backward.Relu{a, input}:
      unsent_spared(id, Backward.vjp(Backward.Relu{a, input}, saved, g))
    case Backward.MseLoss{a, b, left, right}:
      unsent_spared(id, Backward.vjp(Backward.MseLoss{a, b, left, right}, saved, g))
    case Backward.CrossEntropy{a, input}:
      unsent_spared(id, Backward.vjp(Backward.CrossEntropy{a, input}, saved, g))
    case Backward.Constant{}:
      unsent_spared(id, Backward.vjp(Backward.Constant{}, saved, g))
    case Backward.Frozen{inner_id, inner}:
      unsent_spared(id, Backward.vjp(Backward.Frozen{inner_id, inner}, saved, g))

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
    case Backward.SumDims{a, input, kept}:
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
    case Backward.Constant{}:
      {==}
    case Backward.Frozen{id, inner}:
      operands_bounded(inner, bound)

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

# restored only puts g back in its own shape: the contributions are the pass's.
def restored_contract(-a: Nat, -target: Shape.Shape, -cell: Nat -> F32, +shape: Shape.Shape, pass: Backward.Pass,
  contract: Laws.SoleContract<a, target, cell, pass>) -> Laws.SoleContract<a, target, cell, Backward.restored(shape, pass)>:
  match pass:
    case Backward.Pass{saved, g, Fail{error}}:
      Laws.SoleContract{target_id, dimensions, value} = contract
      Laws.SoleContract{target_id, dimensions, value}
    case Backward.Pass{saved, g, Done{Nil{}}}:
      Laws.SoleContract{target_id, dimensions, value} = contract
      Laws.SoleContract{target_id, dimensions, value}
    case Backward.Pass{saved, g, Done{Con{contribution, Nil{}}}}:
      Laws.SoleContract{target_id, dimensions, value} = contract
      Laws.SoleContract{target_id, dimensions, value}
    case Backward.Pass{saved, g, Done{Con{contribution, Con{other, rest}}}}:
      Laws.SoleContract{target_id, dimensions, value} = contract
      Laws.SoleContract{target_id, dimensions, value}

def sum_dims_decided(+a: Nat, +shape: Shape.Shape, +input: Shape.Shape, +kept: Shape.Shape, +plan: Shape.BroadcastPlan,
  planned: {Shape.broadcast_checked(kept, input) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>},
  same: {Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(input) : List<&2, U32>},
  counted: {Nat.is_eq(Shape.elements(shape), Shape.elements(kept)) == True{} : Bool},
  saved: Backward.Saved, +fill: F32, +ops: List<&2, StorageLaws.Op>,
  -fits: {Lifecycle.logical_length(Replayed.storage(kept, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat},
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(shape), Shape.elements(kept))>) ->
  Laws.SoleContract<a, input, index => Replayed.value(kept, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
    Backward.unsqueezed(a, input, kept, shape, Core.TTensor{Replayed.storage(kept, fill, ops), fits}, decision)>:
  match decision:
    case Shape.BFalse{equal}:
      Empty.absurd(Laws.SoleContract<a, input, index => Replayed.value(kept, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))),
        Backward.unsqueezed(a, input, kept, shape, Core.TTensor{Replayed.storage(kept, fill, ops), fits}, Shape.BFalse{equal})>,
        Arith.clash(Equal.trans(Bool, False{}, Nat.is_eq(Shape.elements(shape), Shape.elements(kept)), True{}, equal, counted)))
    case Shape.BTrue{equal}:
      restored_contract(a, input, index => Replayed.value(kept, fill, ops, U32.from_nat(PointwiseBroadcast.left_map(plan, index))), shape,
        Backward.vjp_sum(a, input, Transforms.reshape_equal(shape, kept, Core.TTensor{Replayed.storage(kept, fill, ops), fits},
          Transforms.elements_equal(shape, kept, equal))),
        Laws.tensor_sum_vjp_pointwise(a, kept, input, plan, planned, same, saved, fill, ops,
          Equal.trans(Nat, Lifecycle.logical_length(Replayed.storage(kept, fill, ops)), Shape.logical_numel(Shape.dimensions(shape)),
            Shape.logical_numel(Shape.dimensions(kept)), fits, Transforms.elements_equal(shape, kept, equal))))

def Laws.tensor_sum_dims_vjp_pointwise(a, shape, input, kept, plan, planned, same, counted, saved, fill, ops, fits):
  sum_dims_decided(a, shape, input, kept, plan, planned, same, counted, saved, fill, ops, fits,
    Shape.bool_case(Nat.is_eq(Shape.elements(shape), Shape.elements(kept))))

def Laws.tensor_sum_dims_wraps(a, axes, keepdim, entries, input, shape):
  {==}
