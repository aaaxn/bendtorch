import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ./gradient.bend as Gradient
import ./tensor-backward.bend as Backward

# Plain stochastic gradient descent on the tape, PyTorch's
# torch.optim.SGD(params, lr) with no momentum, dampening, nesterov,
# weight_decay or maximize. The parameters are a list of tape ids, each a
# leaf. step visits them in list order; a parameter whose slot holds g gets
# the value p - lr * g, cell by cell on the parallel elementwise build
# (Elementwise.Descend), and keeps its slot, so a later read still sees g; a
# parameter whose slot is empty, PyTorch's p.grad is None, is left as it was.
# A listed id twice is updated twice, as PyTorch's step does. Recording an
# operation after a step reads the new values, as a forward after
# optimizer.step() reads the updated parameters. Backward.zero_grad empties
# the slots.
#
# step checks every id before it updates any: an id the tape does not hold is
# refused with UnknownNode, an id of a node that is not a leaf with NotLeaf,
# and the tape comes back unchanged.

type Error is Data:
  UnknownNode{}
  NotLeaf{}
  TensorError{error: Core.Error}

def leaf_refusal(node: Backward.Node) -> Maybe<&2, Error>:
  match node:
    case Backward.Leaf{}:
      None{}
    case Backward.Add{a, b, left, right}:
      Some{NotLeaf{}}
    case Backward.Sub{a, b, left, right}:
      Some{NotLeaf{}}
    case Backward.Mul{a, b, left, right}:
      Some{NotLeaf{}}
    case Backward.Sum{a, input}:
      Some{NotLeaf{}}
    case Backward.Matmul{a, b, left, right}:
      Some{NotLeaf{}}
    case Backward.MatmulNT{a, b, left, right}:
      Some{NotLeaf{}}
    case Backward.Relu{a, input}:
      Some{NotLeaf{}}

# Why id cannot be a parameter of a newest-first list of count nodes, or
# None when it is a leaf the list holds.
def refusal_at(nodes: List<&2, Backward.Node>, +count: Nat, +id: Nat, same: Bool) -> Maybe<&2, Error>:
  match nodes count same:
    case Nil{} count same:
      Some{UnknownNode{}}
    case Con{node, rest} 0n same:
      Some{UnknownNode{}}
    case Con{node, rest} 1n+below True{}:
      leaf_refusal(node)
    case Con{node, rest} 1n+below False{}:
      refusal_at(rest, below, id, Backward.at_head(below, id))

def first_refusal(found: Maybe<&2, Error>, later: Maybe<&2, Error>) -> Maybe<&2, Error>:
  match found:
    case Some{error}:
      Some{error}
    case None{}:
      later

# The refusal of the first parameter, in list order, that is not a leaf of
# the tape, or None when every one is.
def refusal(+nodes: List<&2, Backward.Node>, +count: Nat, params: List<&2, Nat>) -> Maybe<&2, Error>:
  match params:
    case Nil{}:
      None{}
    case Con{+id, rest}:
      first_refusal(refusal_at(nodes, count, id, Backward.at_head(count, id)), refusal(nodes, count, rest))

def Walked() -> Type:
  List<&1, Backward.Entry> & Result<&2, &2, Error, Unit>

def descended(node: Backward.Node, +shape: Shape.Shape, saved: Backward.Saved,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Backward.Entry & Result<&2, &2, Error, Unit>:
  (operands, result) = pair
  (value, g) = operands
  match result:
    case Fail{error}:
      (Backward.Entry{node, shape, value, saved, Gradient.Held{g}}, Fail{TensorError{error}})
    case Done{updated}:
      (Backward.Entry{node, shape, updated, saved, Gradient.Held{g}}, Done{Unit{}})

# One parameter: p - lr * g, cell by cell, when its slot holds g; unchanged
# when the slot is empty.
def descend(+lr: F32, entry: Backward.Entry) -> Backward.Entry & Result<&2, &2, Error, Unit>:
  Backward.Entry{node, +shape, value, saved, slot} = entry
  match slot:
    case Gradient.Unset{}:
      (Backward.Entry{node, shape, value, saved, Gradient.Unset{}}, Done{Unit{}})
    case Gradient.Held{g}:
      descended(node, shape, saved, Elementwise.binary(Elementwise.Descend{lr}, value, g))

def placed(rest: List<&1, Backward.Entry>, pair: Backward.Entry & Result<&2, &2, Error, Unit>) -> Walked():
  (entry, result) = pair
  (entry <> rest, result)

def kept(entry: Backward.Entry, pair: Walked()) -> Walked():
  (rest, result) = pair
  (entry <> rest, result)

# Updates entry id of a newest-first list of count entries and keeps every
# other entry.
def descend_at(+lr: F32, entries: List<&1, Backward.Entry>, +count: Nat, +id: Nat, same: Bool) -> Walked():
  match entries count same:
    case Nil{} count same:
      (Nil{}, Fail{UnknownNode{}})
    case Con{entry, rest} 0n same:
      (entry <> rest, Fail{UnknownNode{}})
    case Con{entry, rest} 1n+below True{}:
      placed(rest, descend(lr, entry))
    case Con{entry, rest} 1n+below False{}:
      kept(entry, descend_at(lr, rest, below, id, Backward.at_head(below, id)))

def continued(rec: List<&1, Backward.Entry> -> Walked(), pair: Walked()) -> Walked():
  (entries, result) = pair
  match result:
    case Fail{error}:
      (entries, Fail{error})
    case Done{unit}:
      rec(entries)

# Each parameter in list order, each on the entries the one before it left.
def descend_all(+lr: F32, +count: Nat, params: List<&2, Nat>, entries: List<&1, Backward.Entry>) -> Walked():
  match params:
    case Nil{}:
      (entries, Done{Unit{}})
    case Con{+id, rest}:
      continued(entries => descend_all(lr, count, rest, entries), descend_at(lr, entries, count, id, Backward.at_head(count, id)))

def walked_tape(pair: Walked()) -> Backward.Tape & Result<&2, &2, Error, Unit>:
  (entries, result) = pair
  (Backward.Tape{entries}, result)

def checked(+lr: F32, entries: List<&1, Backward.Entry>, +count: Nat, +params: List<&2, Nat>, refused: Maybe<&2, Error>) ->
  Backward.Tape & Result<&2, &2, Error, Unit>:
  match refused:
    case Some{error}:
      (Backward.Tape{entries}, Fail{error})
    case None{}:
      walked_tape(descend_all(lr, count, params, entries))

def planned(+lr: F32, +params: List<&2, Nat>, pair: List<&1, Backward.Entry> & List<&2, Backward.Node>) ->
  Backward.Tape & Result<&2, &2, Error, Unit>:
  (entries, +nodes) = pair
  +count = List.length(&2, Backward.Node, nodes)
  checked(lr, entries, count, params, refusal(nodes, count, params))

# One SGD step with learning rate lr over the parameters params.
def step(tape: Backward.Tape, +lr: F32, +params: List<&2, Nat>) -> Backward.Tape & Result<&2, &2, Error, Unit>:
  Backward.Tape{entries} = tape
  planned(lr, params, Backward.listed(entries))

def error_code(error: Error) -> U32:
  match error:
    case UnknownNode{}:
      Backward.error_code(Backward.UnknownNode{})
    case NotLeaf{}:
      107
    case TensorError{error}:
      Core.error_code(error)
