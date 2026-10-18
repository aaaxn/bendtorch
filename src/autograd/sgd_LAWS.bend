import Base
import ../shape/core.bend as Shape
import ../storage/LAWS.bend as StorageLaws
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/lifecycle.bend as Lifecycle
import ../tensor/contracts/replayed.bend as Replayed
import ../tensor/contracts/LAWS.bend as Contracts
import ../tensor/contracts/binary_LAWS.bend as Binary
import ./gradient.bend as Gradient
import ./tensor_backward.bend as Backward
import ./tensor_backward_LAWS.bend as TensorBackward
import ./sgd.bend as Sgd

# The cell of an SGD step: p - lr * g, with the product rounded before the
# difference, F32 operation by F32 operation.
law sgd_descend_scalar:
  for +lr: F32
  for +p: F32
  for +g: F32
  {Elementwise.scalar(Elementwise.Descend{lr}, p, g) == F32.sub(p, F32.mul(lr, g)) : F32}

# A parameter whose slot is empty, PyTorch's p.grad is None, is left as it
# was.
law sgd_empty_slot_kept:
  for +lr: F32
  for node: Backward.Node
  for +shape: Shape.Shape
  for value: Core.Dynamic
  for saved: Backward.Saved
  {Sgd.descend(lr, Backward.Entry{node, shape, value, saved, Gradient.Unset{}}) ==
    (Backward.Entry{node, shape, value, saved, Gradient.Unset{}}, Done{Unit{}}) : Backward.Entry & Result<&2, &2, Sgd.Error, Unit>}

# What the transport form of sgd_step_values asks of P: P holds for the entry
# with its node, shape and saved values, whose value is replayed over +0.0
# with any history whose words are the cells of Descend{lr} over the old
# value p and the gradient g, and whose slot still holds g with its words,
# next to Done.
def DescendWith(+lr: F32, node: Backward.Node, +shape: Shape.Shape, saved: Backward.Saved, +plan: Shape.BroadcastPlan,
  +fill: F32, +ops: List<&2, StorageLaws.Op>,
  +g_shape: Shape.Shape, +g_fill: F32, +g_ops: List<&2, StorageLaws.Op>,
  -P: Backward.Entry & Result<&2, &2, Sgd.Error, Unit> -> Type) -> Type:
  @+kept: List<&2, StorageLaws.Op> ->
  @kept_words: {Contracts.words(shape, fill, kept, Shape.logical_numel(Shape.dimensions(shape))) ==
    Contracts.words(shape, fill, ops, Shape.logical_numel(Shape.dimensions(shape))) : List<&2, F32>} ->
  @-kept_fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, kept)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat} ->
  @+g_kept: List<&2, StorageLaws.Op> ->
  @g_words: {Contracts.words(g_shape, g_fill, g_kept, Shape.logical_numel(Shape.dimensions(g_shape))) ==
    Contracts.words(g_shape, g_fill, g_ops, Shape.logical_numel(Shape.dimensions(g_shape))) : List<&2, F32>} ->
  @-g_fits: {Lifecycle.logical_length(Replayed.storage(g_shape, g_fill, g_kept)) == Shape.logical_numel(Shape.dimensions(g_shape)) : Nat} ->
  @+output_ops: List<&2, StorageLaws.Op> ->
  @values: {Contracts.words(Shape.output_shape(plan), 0.0, output_ops, Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) ==
    Binary.cells(Elementwise.Descend{lr}, plan, shape, fill, ops, g_shape, g_fill, g_ops,
      Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan)))) : List<&2, F32>} ->
  @-output_fits: {Lifecycle.logical_length(Replayed.storage(Shape.output_shape(plan), 0.0, output_ops)) ==
    Shape.logical_numel(Shape.dimensions(Shape.output_shape(plan))) : Nat} ->
    P((Backward.Entry{node, shape,
      Core.Packed{Shape.output_shape(plan), Core.TTensor{Replayed.storage(Shape.output_shape(plan), 0.0, output_ops), output_fits}},
      saved, Gradient.Held{Core.Packed{g_shape, Core.TTensor{Replayed.storage(g_shape, g_fill, g_kept), g_fits}}}}, Done{Unit{}}))

# A parameter whose slot holds g gets p - lr * g on the parallel elementwise
# build, in transport form: for a replayed value p and a replayed gradient g
# whose shapes broadcast, for every P, if P holds for every result that keeps
# the node, the shape, the saved values and g with its words, and holds a
# value of the plan's shape whose word k is scalar(Descend{lr},
# p[left_map(plan, k)], g[right_map(plan, k)]) (sgd_descend_scalar: p - lr *
# g), next to Done, then P holds for the step of that parameter.
law sgd_step_values:
  for +lr: F32
  for node: Backward.Node
  for +shape: Shape.Shape
  for saved: Backward.Saved
  for +g_shape: Shape.Shape
  for +plan: Shape.BroadcastPlan
  for planned: {Shape.broadcast_checked(shape, g_shape) == Done{plan} : Result<&2, &2, Shape.Error, Shape.BroadcastPlan>}
  for +fill: F32
  for +ops: List<&2, StorageLaws.Op>
  for -fits: {Lifecycle.logical_length(Replayed.storage(shape, fill, ops)) == Shape.logical_numel(Shape.dimensions(shape)) : Nat}
  for +g_fill: F32
  for +g_ops: List<&2, StorageLaws.Op>
  for -g_fits: {Lifecycle.logical_length(Replayed.storage(g_shape, g_fill, g_ops)) == Shape.logical_numel(Shape.dimensions(g_shape)) : Nat}
  for -P: Backward.Entry & Result<&2, &2, Sgd.Error, Unit> -> Type
  for h: DescendWith(lr, node, shape, saved, plan, fill, ops, g_shape, g_fill, g_ops, P)
  P(Sgd.descend(lr, Backward.Entry{node, shape, Core.Packed{shape, Core.TTensor{Replayed.storage(shape, fill, ops), fits}}, saved,
    Gradient.Held{Core.Packed{g_shape, Core.TTensor{Replayed.storage(g_shape, g_fill, g_ops), g_fits}}}}))

# The entries newer than the parameter, kept in front of the result.
def newer_kept(newer: List<&1, Backward.Entry>, walked: Sgd.Walked()) -> Sgd.Walked():
  match newer:
    case Nil{}:
      walked
    case Con{entry, rest}:
      Sgd.kept(entry, newer_kept(rest, walked))

# The update of parameter id reaches exactly the entry with id older entries
# below it, and keeps every other entry, whatever the list holds.
law sgd_update_reaches_target:
  for +lr: F32
  for newer: List<&1, Backward.Entry>
  for entry: Backward.Entry
  for rest: List<&1, Backward.Entry>
  for +older: Nat
  for fits: {List.length(&1, Backward.Entry, rest) == older : Nat}
  {Sgd.descend_at(lr, List.append(&1, Backward.Entry, newer, entry <> rest), 1n+Nat.add(older, List.length(&1, Backward.Entry, newer)),
      older, Backward.at_head(1n+Nat.add(older, List.length(&1, Backward.Entry, newer)), older)) ==
    newer_kept(newer, Sgd.placed(rest, Sgd.descend(lr, entry))) : Sgd.Walked()}

# The parameters are updated in list order: none leaves the entries as they
# are, and the first is updated before the rest, which see its result.
law sgd_no_params:
  for +lr: F32
  for +count: Nat
  for entries: List<&1, Backward.Entry>
  {Sgd.descend_all(lr, count, Nil{}, entries) == (entries, Done{Unit{}}) : Sgd.Walked()}

law sgd_params_in_order:
  for +lr: F32
  for +count: Nat
  for +id: Nat
  for rest: List<&2, Nat>
  for entries: List<&1, Backward.Entry>
  {Sgd.descend_all(lr, count, id <> rest, entries) ==
    Sgd.continued(after => Sgd.descend_all(lr, count, rest, after), Sgd.descend_at(lr, entries, count, id, Backward.at_head(count, id))) :
    Sgd.Walked()}

# Whether parameter id is refused is read from the node with id older nodes
# below it: a leaf is accepted, any other node refused with NotLeaf.
law sgd_refusal_reads_target:
  for newer: List<&2, Backward.Node>
  for node: Backward.Node
  for rest: List<&2, Backward.Node>
  for +older: Nat
  for fits: {List.length(&2, Backward.Node, rest) == older : Nat}
  {Sgd.refusal_at(List.append(&2, Backward.Node, newer, node <> rest), 1n+Nat.add(older, List.length(&2, Backward.Node, newer)),
      older, Backward.at_head(1n+Nat.add(older, List.length(&2, Backward.Node, newer)), older)) ==
    Sgd.leaf_refusal(node) : Maybe<&2, Sgd.Error>}

# What a parameter's node allows: a node with no operands, a leaf, is
# accepted; a node with operands is refused with NotLeaf.
def operand_refusal(targets: List<&2, Nat>) -> Maybe<&2, Sgd.Error>:
  match targets:
    case Nil{}:
      None{}
    case Con{target, rest}:
      Some{Sgd.NotLeaf{}}

law sgd_only_leaves_accepted:
  for node: Backward.Node
  {Sgd.leaf_refusal(node) == operand_refusal(TensorBackward.targets(node)) : Maybe<&2, Sgd.Error>}

# An id the tape does not hold, the number of its nodes or more, is refused
# with UnknownNode.
law sgd_refusal_unknown:
  for nodes: List<&2, Backward.Node>
  for +extra: Nat
  {Sgd.refusal_at(nodes, List.length(&2, Backward.Node, nodes), Nat.add(List.length(&2, Backward.Node, nodes), extra),
      Backward.at_head(List.length(&2, Backward.Node, nodes), Nat.add(List.length(&2, Backward.Node, nodes), extra))) ==
    Some{Sgd.UnknownNode{}} : Maybe<&2, Sgd.Error>}

# The learning rate is refused exactly when it is below zero, IEEE `lr < 0`
# on the bits (`below_zero`, stated by below_zero_words): PyTorch's
# `lr < 0.0`.
law sgd_negative_rate_refused:
  for +lr: F32
  for below: {Elementwise.below_zero(lr) == True{} : Bool}
  {Sgd.rate_refusal(lr) == Some{Sgd.InvalidLearningRate{}} : Maybe<&2, Sgd.Error>}

law sgd_rate_accepted:
  for +lr: F32
  for above: {Elementwise.below_zero(lr) == False{} : Bool}
  {Sgd.rate_refusal(lr) == None{} : Maybe<&2, Sgd.Error>}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def rate_refusals(words: List<&2, U32>) -> List<&2, Maybe<&2, Sgd.Error>>:
  match words:
    case Nil{}:
      Nil{}
    case Con{word, rest}:
      Sgd.rate_refusal(from_bits(word)) <> rate_refusals(rest)

# At the edge rates, as PyTorch 2.14.1 answers them: +0.0, -0.0, 0.01, the
# quiet NaN, a NaN with the sign bit and +inf are accepted; the smallest
# negative denormal, -0.25, the most negative finite word and -inf are
# refused.
law sgd_rate_edges:
  {rate_refusals([0, 2147483648, 1008981770, 2143289344, 4290772992, 2139095040,
      2147483649, 3196059648, 4286578687, 4286578688]) ==
    [None{}, None{}, None{}, None{}, None{}, None{},
      Some{Sgd.InvalidLearningRate{}}, Some{Sgd.InvalidLearningRate{}}, Some{Sgd.InvalidLearningRate{}},
      Some{Sgd.InvalidLearningRate{}}] : List<&2, Maybe<&2, Sgd.Error>>}

# A step with a refused parameter leaves the tape unchanged; with every id
# accepted, a refused learning rate leaves it unchanged too; otherwise the
# step updates the parameters in list order.
law sgd_step_refused:
  for entries: List<&1, Backward.Entry>
  for +lr: F32
  for +params: List<&2, Nat>
  for +error: Sgd.Error
  for refused: {Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params) ==
    Some{error} : Maybe<&2, Sgd.Error>}
  {Sgd.step(Backward.Tape{entries}, lr, params) == (Backward.Tape{entries}, Fail{error}) :
    Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}

law sgd_step_rate_refused:
  for entries: List<&1, Backward.Entry>
  for +lr: F32
  for +params: List<&2, Nat>
  for +error: Sgd.Error
  for accepted: {Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params) ==
    None{} : Maybe<&2, Sgd.Error>}
  for refused: {Sgd.rate_refusal(lr) == Some{error} : Maybe<&2, Sgd.Error>}
  {Sgd.step(Backward.Tape{entries}, lr, params) == (Backward.Tape{entries}, Fail{error}) :
    Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}

law sgd_step_accepted:
  for entries: List<&1, Backward.Entry>
  for +lr: F32
  for +params: List<&2, Nat>
  for accepted: {Sgd.refusal(TensorBackward.nodes_of(entries), List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params) ==
    None{} : Maybe<&2, Sgd.Error>}
  for rated: {Sgd.rate_refusal(lr) == None{} : Maybe<&2, Sgd.Error>}
  {Sgd.step(Backward.Tape{entries}, lr, params) ==
    Sgd.walked_tape(Sgd.descend_all(lr, List.length(&2, Backward.Node, TensorBackward.nodes_of(entries)), params, entries)) :
    Backward.Tape & Result<&2, &2, Sgd.Error, Unit>}
