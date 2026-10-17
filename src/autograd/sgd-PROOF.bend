import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/contracts/binary-proof.bend as BinaryProof
# Proves binary_replayed, which importing the proof above would otherwise leave open.
import ../tensor/contracts/binary-PROOF.bend as BinaryLaw
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward
import ./tensor-backward-LAWS.bend as TensorBackward
import ./tensor-backward-PROOF.bend as TensorBackwardProof
import ./sgd.bend as Sgd
import ./sgd-LAWS.bend as Laws

def Laws.sgd_descend_scalar(lr, p, g):
  {==}

def Laws.sgd_empty_slot_kept(lr, node, shape, value, saved):
  {==}

def Laws.sgd_step_values(lr, node, shape, saved, g_shape, plan, planned, fill, ops, fits, g_fill, g_ops, g_fits, P, h):
  +planned = planned
  BinaryProof.binary_replayed(Elementwise.Descend{lr}, plan, shape, fill, ops, g_shape, g_fill, g_ops, planned,
    pair => P(Sgd.descended(node, shape, saved, pair)), h)(fits, g_fits)

def Laws.sgd_update_reaches_target(lr, newer, entry, rest, older, fits):
  match newer:
    case Nil{}:
      %TensorBackwardProof.add_zero(older) :
        {Sgd.descend_at(lr, entry <> rest, 1n+_, older, Backward.at_head(1n+_, older)) ==
          Sgd.placed(rest, Sgd.descend(lr, entry)) : Sgd.Walked()}
      %TensorBackwardProof.eq_refl(older) :
        {Sgd.descend_at(lr, entry <> rest, 1n+older, older, _) == Sgd.placed(rest, Sgd.descend(lr, entry)) : Sgd.Walked()}
      {==}
    case Con{head, tail}:
      %TensorBackwardProof.older_ne(older, List.length(&1, Backward.Entry, tail)) :
        {Sgd.descend_at(lr, head <> List.append(&1, Backward.Entry, tail, entry <> rest),
          1n+Nat.add(older, 1n+List.length(&1, Backward.Entry, tail)), older, _) ==
          Sgd.kept(head, Laws.newer_kept(tail, Sgd.placed(rest, Sgd.descend(lr, entry)))) : Sgd.Walked()}
      %TensorBackwardProof.add_succ(older, List.length(&1, Backward.Entry, tail)) :
        {Sgd.descend_at(lr, head <> List.append(&1, Backward.Entry, tail, entry <> rest), 1n+_, older, False{}) ==
          Sgd.kept(head, Laws.newer_kept(tail, Sgd.placed(rest, Sgd.descend(lr, entry)))) : Sgd.Walked()}
      Equal.cong(Sgd.Walked(), Sgd.Walked(), walked => Sgd.kept(head, walked),
        Sgd.descend_at(lr, List.append(&1, Backward.Entry, tail, entry <> rest), 1n+Nat.add(older, List.length(&1, Backward.Entry, tail)),
          older, Backward.at_head(1n+Nat.add(older, List.length(&1, Backward.Entry, tail)), older)),
        Laws.newer_kept(tail, Sgd.placed(rest, Sgd.descend(lr, entry))),
        Laws.sgd_update_reaches_target(lr, tail, entry, rest, older, fits))

def Laws.sgd_only_leaves_accepted(node):
  match node:
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
    case Backward.MatmulNT{a, b, left, right}:
      {==}
    case Backward.Relu{a, input}:
      {==}
    case Backward.MseLoss{a, b, left, right}:
      {==}
    case Backward.CrossEntropy{a, input}:
      {==}
    case Backward.BatchedMatmul{a, b, left, right}:
      {==}
    case Backward.Slice{a, input, kept, plan}:
      {==}
    case Backward.Gathered{a, input, plan, entries}:
      {==}
    case Backward.Constant{}:
      {==}
    case Backward.Frozen{id, inner}:
      Laws.sgd_only_leaves_accepted(inner)

def Laws.sgd_no_params(lr, count, entries):
  {==}

def Laws.sgd_params_in_order(lr, count, id, rest, entries):
  {==}

def Laws.sgd_refusal_reads_target(newer, node, rest, older, fits):
  match newer:
    case Nil{}:
      %TensorBackwardProof.add_zero(older) :
        {Sgd.refusal_at(node <> rest, 1n+_, older, Backward.at_head(1n+_, older)) == Sgd.leaf_refusal(node) : Maybe<&2, Sgd.Error>}
      %TensorBackwardProof.eq_refl(older) :
        {Sgd.refusal_at(node <> rest, 1n+older, older, _) == Sgd.leaf_refusal(node) : Maybe<&2, Sgd.Error>}
      {==}
    case Con{head, tail}:
      %TensorBackwardProof.older_ne(older, List.length(&2, Backward.Node, tail)) :
        {Sgd.refusal_at(head <> List.append(&2, Backward.Node, tail, node <> rest),
          1n+Nat.add(older, 1n+List.length(&2, Backward.Node, tail)), older, _) == Sgd.leaf_refusal(node) : Maybe<&2, Sgd.Error>}
      %TensorBackwardProof.add_succ(older, List.length(&2, Backward.Node, tail)) :
        {Sgd.refusal_at(head <> List.append(&2, Backward.Node, tail, node <> rest), 1n+_, older, False{}) ==
          Sgd.leaf_refusal(node) : Maybe<&2, Sgd.Error>}
      Laws.sgd_refusal_reads_target(tail, node, rest, older, fits)

# A list of n nodes never has id n + 1 + extra at its head.
def ne_succ(+number: Nat, -extra: Nat) -> {False{} == Nat.is_eq(number, 1n+Nat.add(number, extra)) : Bool}:
  match number:
    case 0n:
      {==}
    case 1n+rest:
      ne_succ(rest, extra)

def Laws.sgd_refusal_unknown(nodes, extra):
  match nodes:
    case Nil{}:
      {==}
    case Con{node, +rest}:
      +below = List.length(&2, Backward.Node, rest)
      %ne_succ(below, extra) :
        {Sgd.refusal_at(node <> rest, 1n+below, 1n+Nat.add(below, extra), _) == Some{Sgd.UnknownNode{}} : Maybe<&2, Sgd.Error>}
      %Equal.sym(Nat, 1n+Nat.add(below, extra), Nat.add(below, 1n+extra), TensorBackwardProof.add_succ(below, extra)) :
        {Sgd.refusal_at(rest, below, _, Backward.at_head(below, _)) == Some{Sgd.UnknownNode{}} : Maybe<&2, Sgd.Error>}
      Laws.sgd_refusal_unknown(rest, 1n+extra)

def Laws.sgd_step_refused(entries, lr, params, error, refused):
  %TensorBackwardProof.listed_back(entries) :
    {Sgd.planned(lr, params, _) == (Backward.Tape{entries}, Fail{error}) : Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  %Equal.sym(Maybe<&2, Sgd.Error>, Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params), Some{error}, refused) :
    {Sgd.checked(lr, entries, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, _) ==
      (Backward.Tape{entries}, Fail{error}) : Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  {==}

def Laws.sgd_negative_rate_refused(lr, below):
  %Equal.sym(Bool, Elementwise.below_zero(lr), True{}, below) :
    {Sgd.refused_if(_) == Some{Sgd.InvalidLearningRate{}} : Maybe<&2, Sgd.Error>}
  {==}

def Laws.sgd_rate_accepted(lr, above):
  %Equal.sym(Bool, Elementwise.below_zero(lr), False{}, above) :
    {Sgd.refused_if(_) == None{} : Maybe<&2, Sgd.Error>}
  {==}

def Laws.sgd_rate_edges():
  {==}

def Laws.sgd_step_rate_refused(entries, lr, params, error, accepted, refused):
  %TensorBackwardProof.listed_back(entries) :
    {Sgd.planned(lr, params, _) == (Backward.Tape{entries}, Fail{error}) : Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  %Equal.sym(Maybe<&2, Sgd.Error>, Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params), None{}, accepted) :
    {Sgd.checked(lr, entries, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, _) ==
      (Backward.Tape{entries}, Fail{error}) : Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  %Equal.sym(Maybe<&2, Sgd.Error>, Sgd.rate_refusal(lr), Some{error}, refused) :
    {Sgd.rated(lr, entries, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, _) ==
      (Backward.Tape{entries}, Fail{error}) : Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  {==}

def Laws.sgd_step_accepted(entries, lr, params, accepted, rated):
  %TensorBackwardProof.listed_back(entries) :
    {Sgd.planned(lr, params, _) ==
      Sgd.walked_tape(Sgd.descend_all(lr, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, entries)) :
      Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  %Equal.sym(Maybe<&2, Sgd.Error>, Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params), None{}, accepted) :
    {Sgd.checked(lr, entries, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, _) ==
      Sgd.walked_tape(Sgd.descend_all(lr, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, entries)) :
      Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  %Equal.sym(Maybe<&2, Sgd.Error>, Sgd.rate_refusal(lr), None{}, rated) :
    {Sgd.rated(lr, entries, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, _) ==
      Sgd.walked_tape(Sgd.descend_all(lr, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, entries)) :
      Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
  {==}
