import Base
import ../shape/core.bend as Shape
import ../tensor/core.bend as Core
import ../tensor/elementwise.bend as Elementwise
import ../tensor/layout.bend as Layout
import ../tensor/matmul.bend as Matrix
import ../tensor/batched-matmul.bend as Batched
import ../tensor/reduction.bend as Reduction
import ../tensor/transforms.bend as Transforms
import ../tensor/signed.bend as Signed
import ./broadcast.bend as Broadcast
import ./reduce.bend as Reduce
import ./sum-to-size.bend as SumToSize
import ./sum-dims.bend as SumDims
import ./gradient.bend as Gradient
import ./loss.bend as Loss

# Reverse mode over tensors, on the scalar engine's tape discipline
# (backward.bend): entries newest first, the entry with k older entries has
# id k, a lookup compares ids and never subtracts, the seed goes to the
# selected output, and the sweep visits the output and every older entry,
# newest first. An entry that received a gradient sends its vector-Jacobian
# products to its operands, the first operand before the second; an entry that
# received nothing sends nothing.
#
# Each contribution is reduced to its operand's shape with sum_to_size, as
# PyTorch's engine does for a broadcast operand, and accumulated into the
# operand's working slot with Gradient.accumulate: the first contribution
# moves in unchanged, every later one is added position by position, the held
# total first. After the sweep, each working gradient is accumulated into its
# entry's slot with the same rule, so a repeated backward adds, like PyTorch's
# backward(retain_graph=True).
#
# The seed is a tensor of ones with the output's shape, as PyTorch's
# backward(torch.ones_like(output)). The products are
#   Add: g and g; Sub: g and -g; Mul: g * right and g * left, with the operand
#   values saved at record time; Sum: g broadcast to the operand's shape;
#   Matmul: g @ right^T and left^T @ g; MatmulNT (left @ right^T, the
#   product of Linear): g @ right and g^T @ left, read in place; Relu: g where
#   the saved input is above zero or NaN and +0.0 where it is at most zero,
#   PyTorch's threshold_backward; MseLoss: (2 / N * d) * g and (2 / N * e) * g
#   with the saved differences d = left - right and e = right - left;
#   CrossEntropy: two contributions to the logits, one_hot * -(g / N) and
#   -(exp(lp) * (-(g / N) + 0.0)) with the saved log_softmax lp, whose sum is
#   PyTorch's one_hot gradient minus exp(lp) times its row sum;
#   BatchedMatmul (torch.matmul of operands that are not both rank 2): g @
#   right^T and left^T @ g over the output's batch, each operand read in
#   place at its own batch, 1-D operands as a row or a column.
# Each is reduced to its operand's shape before it is delivered.
#
# A constant is a leaf that takes no gradient, PyTorch's tensor with
# requires_grad=False. A recording whose operands are all constants records a
# constant too, as PyTorch's result does not require grad either. Any other
# recording that reads a constant wraps its node in Frozen{id, node}, one wrap
# for each operand id that is a constant, and the VJP of a frozen node sends
# nothing to id: it drops every contribution to id from node's VJP, so a
# constant's slot stays empty, as PyTorch's .grad stays None. Where the
# dropped contribution is the product g @ W of `matmul_nt`, the input of
# Linear, the VJP does not compute it.
#
# Recording and reads hand back the tape. A failed call leaves every slot as
# it was.

type Error is Data:
  UnknownNode{}
  DanglingOperand{}
  TensorError{error: Core.Error}
  MatrixError{error: Matrix.Error}
  TransformError{error: Transforms.Error}
  BroadcastError{error: Broadcast.Error}
  ReduceError{error: Reduce.Error}
  GradientError{error: Gradient.Error}
  LossError{error: Loss.Error}

type Node is Data:
  Leaf{}
  Add{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  Sub{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  Mul{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  Sum{a: Nat, input: Shape.Shape}
  SumDims{a: Nat, input: Shape.Shape, kept: Shape.Shape}
  Matmul{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  MatmulNT{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  Relu{a: Nat, input: Shape.Shape}
  MseLoss{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  CrossEntropy{a: Nat, input: Shape.Shape}
  BatchedMatmul{a: Nat, b: Nat, left: Shape.Shape, right: Shape.Shape}
  Constant{}
  Frozen{id: Nat, node: Node}

# The operand values a product saves at record time.
type Saved is Type:
  Unsaved{}
  Operands{left: Core.Dynamic, right: Core.Dynamic}

type Entry is Type:
  Entry{node: Node, shape: Shape.Shape, value: Core.Dynamic, saved: Saved, grad: Gradient.Slot}

type Tape is Type:
  Tape{entries: List<&1, Entry>}

type Contribution is Type:
  Contribution{target: Nat, shape: Shape.Shape, gradient: Core.Dynamic}

# What one entry's VJP hands back: its saved values and its gradient, unchanged,
# and the contributions for its operands in operand order.
type Pass is Type:
  Pass{saved: Saved, gradient: Core.Dynamic, contributions: Result<&2, &1, Error, List<&1, Contribution>>}

type Binary is Data:
  Plus{}
  Minus{}
  Times{}
  Product{}
  ProductNT{}
  SquaredError{}
  BatchedProduct{}

def empty() -> Tape:
  Tape{Nil{}}

# Whether the head of a newest-first list of count elements has id `id`.
def at_head(+count: Nat, +id: Nat) -> Bool:
  match count:
    case 0n:
      False{}
    case 1n+below:
      Nat.is_eq(below, id)

def entry_listed(entry: Entry, pair: List<&1, Entry> & List<&2, Node>) -> List<&1, Entry> & List<&2, Node>:
  Entry{+node, shape, value, saved, slot} = entry
  (rest, nodes) = pair
  (Entry{node, shape, value, saved, slot} <> rest, node <> nodes)

# The nodes of a list of entries, newest first, next to the entries.
def listed(entries: List<&1, Entry>) -> List<&1, Entry> & List<&2, Node>:
  match entries:
    case Nil{}:
      (Nil{}, Nil{})
    case Con{entry, rest}:
      entry_listed(entry, listed(rest))

def prepend(-A: Type, head: A, result: Result<&2, &1, Error, List<&1, A>>) -> Result<&2, &1, Error, List<&1, A>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{rest}:
      Done{head <> rest}

def unsets(count: Nat) -> List<&1, Gradient.Slot>:
  match count:
    case 0n:
      Nil{}
    case 1n+rest:
      Gradient.Unset{} <> unsets(rest)

# Vector-Jacobian products.

def both(first: Result<&2, &1, Error, Contribution>, second: Result<&2, &1, Error, Contribution>) ->
  Result<&2, &1, Error, List<&1, Contribution>>:
  match first second:
    case Fail{error} second:
      Fail{error}
    case Done{left} Fail{error}:
      Fail{error}
    case Done{left} Done{right}:
      Done{[left, right]}

def delivered_as(+target: Nat, +shape: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (owner, result) = pair
  match result:
    case Fail{error}:
      (owner, Fail{ReduceError{error}})
    case Done{reduced}:
      (owner, Done{Contribution{target, shape, reduced}})

# A contribution for operand target: g reduced to the operand's shape. g comes
# back unchanged.
def reduced(+target: Nat, +shape: Shape.Shape, g: Core.Dynamic) -> Core.Dynamic & Result<&2, &1, Error, Contribution>:
  delivered_as(target, shape, SumToSize.sum_to_size(g, shape))

def contribution_of(pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Result<&2, &1, Error, Contribution>:
  (owner, result) = pair
  result

def product_reduced(+target: Nat, +shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&2, &1, Error, Contribution>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{product}:
      contribution_of(reduced(target, shape, product))

def added_second(first: Result<&2, &1, Error, Contribution>, pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (g, second) = pair
  Pass{Unsaved{}, g, both(first, second)}

def added_first(+b: Nat, +right: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (g, first) = pair
  added_second(first, reduced(b, right, g))

# d(a+b)/da and d(a+b)/db: g, each reduced to its operand's shape.
def vjp_add(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, g: Core.Dynamic) -> Pass:
  added_first(b, right, reduced(a, left, g))

def negated_pair(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  (operands, result) = pair
  (g, zero) = operands
  (g, result)

# -g, word by word: F32.neg flips the sign bit, as PyTorch's neg does.
def negated(g: Core.Dynamic, zero: Result<&2, &1, Core.Error, Core.Tensor<Reduction.scalar_shape()>>) ->
  Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  match zero:
    case Fail{error}:
      (g, Fail{error})
    case Done{tensor}:
      negated_pair(Elementwise.binary(Elementwise.NegateLeft{}, g, Core.pack(Reduction.scalar_shape(), tensor)))

def negate(g: Core.Dynamic) -> Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>:
  negated(g, Core.new(Reduction.scalar_shape(), 0.0))

def subtracted_second(first: Result<&2, &1, Error, Contribution>, +b: Nat, +right: Shape.Shape,
  pair: Core.Dynamic & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (g, minus) = pair
  Pass{Unsaved{}, g, both(first, product_reduced(b, right, minus))}

def subtracted_first(+b: Nat, +right: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (g, first) = pair
  subtracted_second(first, b, right, negate(g))

# d(a-b)/da: g; d(a-b)/db: -g, negated before it is reduced, as in PyTorch.
def vjp_sub(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, g: Core.Dynamic) -> Pass:
  subtracted_first(b, right, reduced(a, left, g))

def multiplied_second(+b: Nat, +right: Shape.Shape, first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, second) = pair
  (g, x) = operands
  Pass{Operands{x, y}, g, both(first, product_reduced(b, right, second))}

def multiplied_first(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, x: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, first) = pair
  (g, y) = operands
  multiplied_second(b, right, product_reduced(a, left, first), y, Elementwise.mul(g, x))

# d(a*b)/da: g * right; d(a*b)/db: g * left, with the operand values saved
# at record time, each product reduced to its operand's shape.
def vjp_mul(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, y}:
      multiplied_first(a, b, left, right, x, Elementwise.mul(g, y))

def expanded(+a: Nat, +input: Shape.Shape, pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Pass:
  (g, result) = pair
  match result:
    case Fail{error}:
      Pass{Unsaved{}, g, Fail{BroadcastError{error}}}
    case Done{expanded}:
      Pass{Unsaved{}, g, Done{[Contribution{a, input, expanded}]}}

# d(sum a)/da: g broadcast to a's shape, as PyTorch's g.expand(a.shape).
def vjp_sum(+a: Nat, +input: Shape.Shape, g: Core.Dynamic) -> Pass:
  expanded(a, input, Broadcast.broadcast_to(g, input))

def relabel_decided(+held: Shape.Shape, +shape: Shape.Shape, tensor: Core.Tensor<held>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(held), Shape.elements(shape))>) -> Core.Dynamic:
  match decision:
    case Shape.BFalse{equal}:
      Core.Packed{held, tensor}
    case Shape.BTrue{equal}:
      Transforms.reshape_equal(held, shape, tensor, Transforms.elements_equal(held, shape, equal))

# g with the dimensions of `shape`, which has as many elements: a reshape,
# which moves no word. A g with another element count keeps its own.
def relabeled(+shape: Shape.Shape, gradient: Core.Dynamic) -> Core.Dynamic:
  Core.Packed{+held, tensor} = gradient
  relabel_decided(held, shape, tensor, Shape.bool_case(Nat.is_eq(Shape.elements(held), Shape.elements(shape))))

# The pass with g back in its own shape.
def restored(+shape: Shape.Shape, pass: Pass) -> Pass:
  Pass{saved, gradient, contributions} = pass
  Pass{saved, relabeled(shape, gradient), contributions}

def unsqueezed(+a: Nat, +input: Shape.Shape, +kept: Shape.Shape, +shape: Shape.Shape, tensor: Core.Tensor<shape>,
  decision: Shape.BoolCase<Nat.is_eq(Shape.elements(shape), Shape.elements(kept))>) -> Pass:
  match decision:
    case Shape.BFalse{equal}:
      Pass{Unsaved{}, Core.Packed{shape, tensor}, Fail{TransformError{Transforms.LayoutError{Layout.ElementCountMismatch{}}}}}
    case Shape.BTrue{equal}:
      restored(shape, vjp_sum(a, input, Transforms.reshape_equal(shape, kept, tensor, Transforms.elements_equal(shape, kept, equal))))

# d(sum of a over some axes)/da: g read with the kept shape, the reduced axes
# put back as 1 (PyTorch's unsqueeze when keepdim is False), then broadcast
# to a's shape, as the Sum VJP does.
def vjp_sum_dims(+a: Nat, +input: Shape.Shape, +kept: Shape.Shape, g: Core.Dynamic) -> Pass:
  Core.Packed{+shape, tensor} = g
  unsqueezed(a, input, kept, shape, tensor, Shape.bool_case(Nat.is_eq(Shape.elements(shape), Shape.elements(kept))))

def multiplied(+target: Nat, +shape: Shape.Shape,
  result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Result<&2, &1, Error, Contribution>:
  match result:
    case Fail{error}:
      Fail{MatrixError{error}}
    case Done{triple}:
      (left, right, product) = triple
      Done{Contribution{target, shape, product}}

# g @ right^T for the left operand. g and right are copied first, so a failure
# loses neither.
def left_product(+target: Nat, +shape: Shape.Shape, copy: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (right, transposed) = pair
  match transposed:
    case Fail{error}:
      (right, Fail{TransformError{error}})
    case Done{flipped}:
      (right, multiplied(target, shape, Matrix.matmul(copy, flipped)))

# left^T @ g for the right operand.
def right_product(+target: Nat, +shape: Shape.Shape, copy: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (left, transposed) = pair
  match transposed:
    case Fail{error}:
      (left, Fail{TransformError{error}})
    case Done{flipped}:
      (left, multiplied(target, shape, Matrix.matmul(flipped, copy)))

def matmul_fourth(first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (x, second) = pair
  Pass{Operands{x, y}, g, both(first, second)}

def matmul_third(+b: Nat, +right: Shape.Shape, first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic, x: Core.Dynamic,
  pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  matmul_fourth(first, y, g, right_product(b, right, copy, Transforms.transpose_dynamic(x, 0, 1)))

def matmul_second(+b: Nat, +right: Shape.Shape, x: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (y, first) = pair
  matmul_third(b, right, first, y, x, Core.clone_dynamic(g))

def matmul_first(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, x: Core.Dynamic, y: Core.Dynamic,
  pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  matmul_second(b, right, x, g, left_product(a, left, copy, Transforms.transpose_dynamic(y, 0, 1)))

# d(a@b)/da: g @ b^T; d(a@b)/db: a^T @ g, with the operand values saved at
# record time.
def vjp_matmul(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, y}:
      matmul_first(a, b, left, right, x, y, Core.clone_dynamic(g))

# g @ W for the left operand of x @ W^T: g and W are copied first, so a
# failure loses neither.
def nt_left(+target: Nat, +shape: Shape.Shape, copy: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) ->
  Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (right, flipped) = pair
  (right, multiplied(target, shape, Matrix.matmul(copy, flipped)))

# g^T @ x, g read in place. It has its own name so that the proofs of where
# the contribution goes name it without restating the product; the value
# laws read what it computes.
def weight_product(copy: Core.Dynamic, flipped: Core.Dynamic) -> Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  Matrix.matmul_tn(copy, flipped)

# g^T @ x for the right operand.
def nt_right(+target: Nat, +shape: Shape.Shape, copy: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) ->
  Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (left, flipped) = pair
  (left, multiplied(target, shape, weight_product(copy, flipped)))

def nt_fourth(first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (x, second) = pair
  Pass{Operands{x, y}, g, both(first, second)}

def nt_third(+b: Nat, +right: Shape.Shape, first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic, x: Core.Dynamic,
  pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  nt_fourth(first, y, g, nt_right(b, right, copy, Core.clone_dynamic(x)))

def nt_second(+b: Nat, +right: Shape.Shape, x: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (y, first) = pair
  nt_third(b, right, first, y, x, Core.clone_dynamic(g))

def nt_first(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, x: Core.Dynamic, y: Core.Dynamic,
  pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  nt_second(b, right, x, g, nt_left(a, left, copy, Core.clone_dynamic(y)))

def alone(result: Result<&2, &1, Error, Contribution>) -> Result<&2, &1, Error, List<&1, Contribution>>:
  match result:
    case Fail{error}:
      Fail{error}
    case Done{contribution}:
      Done{[contribution]}

def nt_alone_second(y: Core.Dynamic, g: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (x, second) = pair
  Pass{Operands{x, y}, g, alone(second)}

def nt_alone_first(+b: Nat, +right: Shape.Shape, x: Core.Dynamic, y: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  nt_alone_second(y, g, nt_right(b, right, copy, Core.clone_dynamic(x)))

# d(x @ W^T)/dW alone, g^T @ x, for a product whose input x takes no
# gradient: the second contribution of vjp_matmul_nt, computed the same way,
# without g @ W.
def vjp_matmul_nt_right(+b: Nat, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, y}:
      nt_alone_first(b, right, x, y, Core.clone_dynamic(g))

# d(x @ W^T)/dx: g @ W; d(x @ W^T)/dW: g^T @ x, with the operand values
# saved at record time. Neither product builds a transpose: g @ W is
# `matmul`, and g^T @ x is `matmul_tn`, which reads g in place. Each product
# already has its operand's shape.
def vjp_matmul_nt(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, y}:
      nt_first(a, b, left, right, x, y, Core.clone_dynamic(g))

def relu_passed(+a: Nat, +input: Shape.Shape, zero: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, result) = pair
  (x, g) = operands
  Pass{Operands{x, zero}, g, alone(product_reduced(a, input, result))}

# d relu(a)/da: the pointwise ReluBackward of the saved input x and g, g's
# own word where x is above zero or NaN and +0.0 where x is at most zero,
# reduced to the input's shape.
def vjp_relu(+a: Nat, +input: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, zero}:
      relu_passed(a, input, zero, Elementwise.binary(Elementwise.ReluBackward{}, x, g))

def mse_second(+b: Nat, +right: Shape.Shape, first: Result<&2, &1, Error, Contribution>, d: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, second) = pair
  (g, e) = operands
  Pass{Operands{e, d}, g, both(first, product_reduced(b, right, second))}

def mse_first(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, +scale: F32, e: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, first) = pair
  (g, d) = operands
  mse_second(b, right, product_reduced(a, left, first), d, Elementwise.binary(Elementwise.MseGrad{scale}, g, e))

def mse_shaped(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, e: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Shape.Shape) -> Pass:
  (d, +shape) = pair
  mse_first(a, b, left, right, Loss.mse_scale(shape), e, Elementwise.binary(Elementwise.MseGrad{Loss.mse_scale(shape)}, g, d))

# d mse_loss(a, b)/da: (2 / N * d) * g, and d/db: (2 / N * e) * g, PyTorch's
# mse_loss_backward, with the differences e = right - left and d = left -
# right saved at record time over the broadcast shape of N cells; each is
# reduced to its operand's shape. The steps are the Mul VJP's, with
# MseGrad for Mul.
def vjp_mse(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{e, d}:
      mse_shaped(a, b, left, right, e, g, Core.shape(d))

def entropy_second(+a: Nat, +input: Shape.Shape, first: Result<&2, &1, Error, Contribution>, onehot: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, second) = pair
  (g, lp) = operands
  Pass{Operands{lp, onehot}, g, both(first, product_reduced(a, input, second))}

def entropy_first(+a: Nat, +input: Shape.Shape, +count: F32, +single: Bool, lp: Core.Dynamic,
  pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Pass:
  (operands, first) = pair
  (g, onehot) = operands
  entropy_second(a, input, product_reduced(a, input, first), onehot, Elementwise.binary(Elementwise.SoftmaxGrad{count, single}, g, lp))

# d cross_entropy(a)/da, as two contributions to a that its slot adds in
# order. The first is PyTorch's nll_loss backward, OneHotGrad of g and the
# saved one-hot rows: -(g / N) at each row's target and +0.0 elsewhere, for
# the N rows. The second is the rest of PyTorch's log_softmax backward,
# SoftmaxGrad of g and the saved lp: -(exp(lp) * r) with r the row sum of the
# first, -(g / N) + 0.0, or -(g / N) itself when the logits have one column
# (PyTorch's sum of one word keeps -0.0). Added into an empty slot, first + second is
# PyTorch's first - exp(lp) * r word for word. The steps are the Mul VJP's,
# with OneHotGrad and SoftmaxGrad for Mul and both contributions to a.
def vjp_cross_entropy(+a: Nat, +input: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{lp, onehot}:
      entropy_first(a, input, Loss.count_f32(Loss.rows(input)), Loss.one_class(input), lp,
        Elementwise.binary(Elementwise.OneHotGrad{Loss.count_f32(Loss.rows(input))}, g, onehot))

def kept_one(+target: Nat, +shape: Shape.Shape, gradient: Core.Dynamic, mine: Bool, rest: List<&1, Contribution>) ->
  List<&1, Contribution>:
  match mine:
    case True{}:
      rest
    case False{}:
      Contribution{target, shape, gradient} <> rest

# The contributions to every node but id, in order.
def kept_others(+id: Nat, items: List<&1, Contribution>) -> List<&1, Contribution>:
  match items:
    case Nil{}:
      Nil{}
    case Con{Contribution{+target, +shape, gradient}, rest}:
      kept_one(target, shape, gradient, Nat.is_eq(id, target), kept_others(id, rest))

def without(+id: Nat, contributions: Result<&2, &1, Error, List<&1, Contribution>>) -> Result<&2, &1, Error, List<&1, Contribution>>:
  match contributions:
    case Fail{error}:
      Fail{error}
    case Done{items}:
      Done{kept_others(id, items)}

# A VJP's pass without its contributions to id.
def unsent(+id: Nat, pass: Pass) -> Pass:
  Pass{saved, g, contributions} = pass
  Pass{saved, g, without(id, contributions)}

# Whether the VJP of Frozen{id, node} skips a product instead of dropping it:
# node is x @ W^T and id is x and not W.
def skips(+id: Nat, node: Node) -> Bool:
  match node:
    case MatmulNT{a, b, left, right}:
      Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b)))
    case other:
      False{}

def nt_frozen(+id: Nat, +a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic, skip: Bool) -> Pass:
  match skip:
    case True{}:
      vjp_matmul_nt_right(b, right, saved, g)
    case False{}:
      unsent(id, vjp_matmul_nt(a, b, left, right, saved, g))

# A contribution for operand target from a product over the output's batch:
# the product reduced to the operand's shape, as a broadcast operand of an
# elementwise operation is.
def batched_reduced(+target: Nat, +shape: Shape.Shape,
  result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Result<&2, &1, Error, Contribution>:
  match result:
    case Fail{error}:
      Fail{TensorError{error}}
    case Done{triple}:
      (first, second, product) = triple
      contribution_of(reduced(target, shape, product))

def batched_fourth(first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (x, second) = pair
  Pass{Operands{x, y}, g, both(first, second)}

# left^T @ g for the right operand, from copies of g and of the saved left
# operand.
def batched_right(+b: Nat, +right: Shape.Shape, +plan: Batched.Plan, copy: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) ->
  Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (x, left_copy) = pair
  (x, batched_reduced(b, right, Batched.right_gradient(plan, left_copy, copy)))

def batched_third(+b: Nat, +right: Shape.Shape, +plan: Batched.Plan, first: Result<&2, &1, Error, Contribution>, y: Core.Dynamic,
  x: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  batched_fourth(first, y, g, batched_right(b, right, plan, copy, Core.clone_dynamic(x)))

def batched_second(+b: Nat, +right: Shape.Shape, +plan: Batched.Plan, x: Core.Dynamic, g: Core.Dynamic,
  pair: Core.Dynamic & Result<&2, &1, Error, Contribution>) -> Pass:
  (y, first) = pair
  batched_third(b, right, plan, first, y, x, Core.clone_dynamic(g))

# g @ right^T for the left operand, from copies of g and of the saved right
# operand, so a failure loses neither.
def batched_left(+a: Nat, +left: Shape.Shape, +plan: Batched.Plan, copy: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) ->
  Core.Dynamic & Result<&2, &1, Error, Contribution>:
  (y, right_copy) = pair
  (y, batched_reduced(a, left, Batched.left_gradient(plan, copy, right_copy)))

def batched_first(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, +plan: Batched.Plan, x: Core.Dynamic, y: Core.Dynamic,
  pair: Core.Dynamic & Core.Dynamic) -> Pass:
  (g, copy) = pair
  batched_second(b, right, plan, x, g, batched_left(a, left, plan, copy, Core.clone_dynamic(y)))

def batched_planned(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, x: Core.Dynamic, y: Core.Dynamic, g: Core.Dynamic,
  result: Result<&2, &2, Matrix.Error, Batched.Plan>) -> Pass:
  match result:
    case Fail{error}:
      Pass{Operands{x, y}, g, Fail{MatrixError{error}}}
    case Done{+plan}:
      batched_first(a, b, left, right, plan, x, y, Core.clone_dynamic(g))

# d(a@b)/da and d(a@b)/db of torch.matmul: g @ b^T and a^T @ g over the
# output's batch (Batched.left_gradient and Batched.right_gradient), each
# then summed to its operand's shape with sum_to_size, which sums the batch
# dimensions the operand was broadcast along and keeps a product that
# already has the operand's shape. A 1-D operand's product already drops its
# row or column.
def vjp_batched(+a: Nat, +b: Nat, +left: Shape.Shape, +right: Shape.Shape, saved: Saved, g: Core.Dynamic) -> Pass:
  match saved:
    case Unsaved{}:
      Pass{Unsaved{}, g, Fail{TensorError{Core.StorageLengthMismatch{}}}}
    case Operands{x, y}:
      batched_planned(a, b, left, right, x, y, g, Batched.plan(left, right))

# The vector-Jacobian product of one node with gradient g: the contribution
# each operand receives, in operand order, already reduced to its shape. A
# constant sends nothing, and Frozen{id, node} sends node's contributions
# without the ones to id.
def vjp(node: Node, saved: Saved, g: Core.Dynamic) -> Pass:
  match node:
    case Leaf{}:
      Pass{saved, g, Done{Nil{}}}
    case Add{+a, +b, +left, +right}:
      vjp_add(a, b, left, right, g)
    case Sub{+a, +b, +left, +right}:
      vjp_sub(a, b, left, right, g)
    case Mul{+a, +b, +left, +right}:
      vjp_mul(a, b, left, right, saved, g)
    case Sum{+a, +input}:
      vjp_sum(a, input, g)
    case SumDims{+a, +input, +kept}:
      vjp_sum_dims(a, input, kept, g)
    case Matmul{+a, +b, +left, +right}:
      vjp_matmul(a, b, left, right, saved, g)
    case MatmulNT{+a, +b, +left, +right}:
      vjp_matmul_nt(a, b, left, right, saved, g)
    case Relu{+a, +input}:
      vjp_relu(a, input, saved, g)
    case MseLoss{+a, +b, +left, +right}:
      vjp_mse(a, b, left, right, saved, g)
    case CrossEntropy{+a, +input}:
      vjp_cross_entropy(a, input, saved, g)
    case BatchedMatmul{+a, +b, +left, +right}:
      vjp_batched(a, b, left, right, saved, g)
    case Constant{}:
      Pass{saved, g, Done{Nil{}}}
    case Frozen{+id, inner}:
      match inner:
        case MatmulNT{+a, +b, +left, +right}:
          nt_frozen(id, a, b, left, right, saved, g, Bool.and(Nat.is_eq(id, a), Bool.not(Nat.is_eq(id, b))))
        case other:
          unsent(id, vjp(other, saved, g))

# Delivery.

def placed(rest: List<&1, Gradient.Slot>, outcome: Gradient.Outcome) -> Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  match outcome:
    case Gradient.Accumulated{slot}:
      Done{slot <> rest}
    case Gradient.Refused{slot, contribution, error}:
      Fail{GradientError{error}}

# Accumulates a contribution into the slot of node target, in a newest-first
# list of count slots; same says whether the head is that node. A target the
# list does not hold changes nothing.
def bump(slots: List<&1, Gradient.Slot>, +count: Nat, +target: Nat, +shape: Shape.Shape, gradient: Core.Dynamic, same: Bool) ->
  Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  match slots count same:
    case Nil{} count same:
      Done{Nil{}}
    case Con{slot, rest} 0n same:
      Done{slot <> rest}
    case Con{slot, rest} 1n+below True{}:
      placed(rest, Gradient.accumulate(shape, slot, gradient))
    case Con{slot, rest} 1n+below False{}:
      prepend(Gradient.Slot, slot, bump(rest, below, target, shape, gradient, at_head(below, target)))

def deliver(contributions: List<&1, Contribution>, +count: Nat, state: Result<&2, &1, Error, List<&1, Gradient.Slot>>) ->
  Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  match contributions state:
    case Nil{} state:
      state
    case Con{contribution, rest} Fail{error}:
      Fail{error}
    case Con{Contribution{+target, +shape, gradient}, rest} Done{slots}:
      deliver(rest, count, bump(slots, count, target, shape, gradient, at_head(count, target)))

def sent(contributions: Result<&2, &1, Error, List<&1, Contribution>>, +count: Nat, slots: List<&1, Gradient.Slot>) ->
  Result<&2, &1, Error, List<&1, Gradient.Slot>>:
  match contributions:
    case Fail{error}:
      Fail{error}
    case Done{items}:
      deliver(items, count, Done{slots})

# The sweep.

def Swept() -> Type:
  List<&1, Entry> & Result<&2, &1, Error, List<&1, Gradient.Slot>>

def finished(entry: Entry, slot: Gradient.Slot, pair: Swept()) -> Swept():
  (rest, result) = pair
  (entry <> rest, prepend(Gradient.Slot, slot, result))

def passed(node: Node, +shape: Shape.Shape, value: Core.Dynamic, slot: Gradient.Slot, +below: Nat,
  pending: List<&1, Gradient.Slot>, rec: Result<&2, &1, Error, List<&1, Gradient.Slot>> -> Swept(), pass: Pass) -> Swept():
  Pass{saved, g, contributions} = pass
  finished(Entry{node, shape, value, saved, slot}, Gradient.Held{g}, rec(sent(contributions, below, pending)))

def visited(+node: Node, +shape: Shape.Shape, value: Core.Dynamic, saved: Saved, slot: Gradient.Slot, gradient: Gradient.Slot,
  +below: Nat, pending: List<&1, Gradient.Slot>, rec: Result<&2, &1, Error, List<&1, Gradient.Slot>> -> Swept()) -> Swept():
  match gradient:
    case Gradient.Unset{}:
      finished(Entry{node, shape, value, saved, slot}, Gradient.Unset{}, rec(Done{pending}))
    case Gradient.Held{g}:
      passed(node, shape, value, slot, below, pending, rec, vjp(node, saved, g))

# Visits entries newest first with their working gradients. Once a step
# fails, the remaining entries pass through unchanged.
def sweep(entries: List<&1, Entry>, state: Result<&2, &1, Error, List<&1, Gradient.Slot>>, +count: Nat) -> Swept():
  match entries state count:
    case Nil{} state count:
      (Nil{}, state)
    case Con{entry, rest} Fail{error} count:
      (entry <> rest, Fail{error})
    case Con{entry, rest} Done{Nil{}} count:
      (entry <> rest, Done{Nil{}})
    case Con{entry, rest} Done{Con{gradient, pending}} 0n:
      (entry <> rest, Done{Nil{}})
    case Con{Entry{+node, +shape, value, saved, slot}, rest} Done{Con{gradient, pending}} 1n+below:
      visited(node, shape, value, saved, slot, gradient, below, pending, state => sweep(rest, state, below))

def seeded(node: Node, +shape: Shape.Shape, value: Core.Dynamic, saved: Saved, slot: Gradient.Slot, rest: List<&1, Entry>,
  +below: Nat, ones: Result<&2, &1, Core.Error, Core.Tensor<shape>>) -> Swept():
  match ones:
    case Fail{error}:
      (Entry{node, shape, value, saved, slot} <> rest, Fail{TensorError{error}})
    case Done{tensor}:
      sweep(Entry{node, shape, value, saved, slot} <> rest,
        Done{Gradient.Held{Core.Packed{shape, tensor}} <> unsets(below)}, 1n+below)

# The seed: ones of the output's shape.
def seeded_entry(entry: Entry, rest: List<&1, Entry>, +below: Nat) -> Swept():
  Entry{node, +shape, value, saved, slot} = entry
  seeded(node, shape, value, saved, slot, rest, below, Core.new(shape, 1.0))

# The working gradient of every entry, newest first: Unset for the entries
# newer than output and for every entry output does not reach, ones of the
# output's shape at output, and the sweep's totals below it.
def pass(entries: List<&1, Entry>, +count: Nat, +output: Nat, selected: Bool) -> Swept():
  match entries count selected:
    case Nil{} count selected:
      (Nil{}, Done{Nil{}})
    case Con{entry, rest} 0n selected:
      (entry <> rest, Done{Nil{}})
    case Con{entry, rest} 1n+below False{}:
      finished(entry, Gradient.Unset{}, pass(rest, below, output, at_head(below, output)))
    case Con{entry, rest} 1n+below True{}:
      seeded_entry(entry, rest, below)

def operands_below(node: Node, +bound: Nat) -> Bool:
  match node:
    case Leaf{}:
      True{}
    case Add{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Sub{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Mul{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Sum{a, input}:
      Nat.is_lt(a, bound)
    case SumDims{a, input, kept}:
      Nat.is_lt(a, bound)
    case Matmul{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case MatmulNT{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Relu{a, input}:
      Nat.is_lt(a, bound)
    case MseLoss{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case CrossEntropy{a, input}:
      Nat.is_lt(a, bound)
    case BatchedMatmul{a, b, left, right}:
      Bool.and(Nat.is_lt(a, bound), Nat.is_lt(b, bound))
    case Constant{}:
      True{}
    case Frozen{id, inner}:
      operands_below(inner, bound)

def wired(nodes: List<&2, Node>, +count: Nat) -> Bool:
  match nodes count:
    case Nil{} count:
      True{}
    case Con{node, rest} 0n:
      False{}
    case Con{node, rest} 1n+below:
      Bool.and(operands_below(node, below), wired(rest, below))

# Accumulates each working gradient into its entry's slot. A refusal keeps
# the slot and is reported after the walk.
def merged_entry(node: Node, +shape: Shape.Shape, value: Core.Dynamic, saved: Saved, outcome: Gradient.Outcome,
  pair: List<&1, Entry> & Result<&2, &2, Error, Unit>) -> List<&1, Entry> & Result<&2, &2, Error, Unit>:
  match outcome:
    case Gradient.Accumulated{slot}:
      (rest, result) = pair
      (Entry{node, shape, value, saved, slot} <> rest, result)
    case Gradient.Refused{slot, contribution, error}:
      (rest, result) = pair
      (Entry{node, shape, value, saved, slot} <> rest, Fail{GradientError{error}})

def merge(entries: List<&1, Entry>, gradients: List<&1, Gradient.Slot>) -> List<&1, Entry> & Result<&2, &2, Error, Unit>:
  match entries gradients:
    case Nil{} gradients:
      (Nil{}, Done{Unit{}})
    case Con{entry, rest} Nil{}:
      (entry <> rest, Done{Unit{}})
    case Con{Entry{node, +shape, value, saved, slot}, rest} Con{Gradient.Unset{}, more}:
      merged_entry(node, shape, value, saved, Gradient.Accumulated{slot}, merge(rest, more))
    case Con{Entry{node, +shape, value, saved, slot}, rest} Con{Gradient.Held{g}, more}:
      merged_entry(node, shape, value, saved, Gradient.accumulate(shape, slot, g), merge(rest, more))

def merged_tape(pair: List<&1, Entry> & Result<&2, &2, Error, Unit>) -> Tape & Result<&2, &2, Error, Unit>:
  (entries, result) = pair
  (Tape{entries}, result)

def ran(pair: Swept()) -> Tape & Result<&2, &2, Error, Unit>:
  (entries, result) = pair
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{error})
    case Done{gradients}:
      merged_tape(merge(entries, gradients))

def checked(entries: List<&1, Entry>, +count: Nat, +output: Nat, known: Bool, sound: Bool) -> Tape & Result<&2, &2, Error, Unit>:
  match known sound:
    case False{} sound:
      (Tape{entries}, Fail{UnknownNode{}})
    case True{} False{}:
      (Tape{entries}, Fail{DanglingOperand{}})
    case True{} True{}:
      ran(pass(entries, count, output, at_head(count, output)))

def planned(+output: Nat, pair: List<&1, Entry> & List<&2, Node>) -> Tape & Result<&2, &2, Error, Unit>:
  (entries, +nodes) = pair
  +count = List.length(&2, Node, nodes)
  checked(entries, count, output, Nat.is_lt(output, count), wired(nodes, count))

# Runs one backward of output and accumulates its gradients into the slots.
# An output the tape does not hold is refused with UnknownNode; a tape with a
# node that refers to itself or to a newer or missing node is refused with
# DanglingOperand. Both refusals leave the tape unchanged.
def backward(tape: Tape, +output: Nat) -> Tape & Result<&2, &2, Error, Unit>:
  Tape{entries} = tape
  planned(output, listed(entries))

def cleared(entries: List<&1, Entry>) -> List<&1, Entry>:
  match entries:
    case Nil{}:
      Nil{}
    case Con{Entry{node, shape, value, saved, slot}, rest}:
      Entry{node, shape, value, saved, Gradient.reset(slot)} <> cleared(rest)

# Empties every gradient slot, like PyTorch's zero_grad(set_to_none=True).
def zero_grad(tape: Tape) -> Tape:
  Tape{entries} = tape
  Tape{cleared(entries)}

# Reads.

def Found() -> Type:
  List<&1, Entry> & Result<&2, &1, Error, Core.Dynamic>

def cloned_value(node: Node, +shape: Shape.Shape, saved: Saved, slot: Gradient.Slot, rest: List<&1, Entry>,
  pair: Core.Dynamic & Core.Dynamic) -> Found():
  (value, copy) = pair
  (Entry{node, shape, value, saved, slot} <> rest, Done{copy})

def found_value(entry: Entry, rest: List<&1, Entry>) -> Found():
  Entry{node, +shape, value, saved, slot} = entry
  cloned_value(node, shape, saved, slot, rest, Core.clone_dynamic(value))

def prepend_found(entry: Entry, pair: Found()) -> Found():
  (rest, result) = pair
  (entry <> rest, result)

# A copy of the value of node id, in a newest-first list of count entries.
def value_at(entries: List<&1, Entry>, +count: Nat, +id: Nat, same: Bool) -> Found():
  match entries count same:
    case Nil{} count same:
      (Nil{}, Fail{UnknownNode{}})
    case Con{entry, rest} 0n same:
      (entry <> rest, Fail{UnknownNode{}})
    case Con{entry, rest} 1n+below True{}:
      found_value(entry, rest)
    case Con{entry, rest} 1n+below False{}:
      prepend_found(entry, value_at(rest, below, id, at_head(below, id)))

def counted_entry(entry: Entry, pair: List<&1, Entry> & Nat) -> List<&1, Entry> & Nat:
  (rest, count) = pair
  (entry <> rest, 1n+count)

# The number of entries, next to the entries.
def counted(entries: List<&1, Entry>) -> List<&1, Entry> & Nat:
  match entries:
    case Nil{}:
      (Nil{}, 0n)
    case Con{entry, rest}:
      counted_entry(entry, counted(rest))

def value_counted(+id: Nat, pair: List<&1, Entry> & Nat) -> Found():
  (entries, +count) = pair
  value_at(entries, count, id, at_head(count, id))

def found_tape(pair: Found()) -> Tape & Result<&2, &1, Error, Core.Dynamic>:
  (entries, result) = pair
  (Tape{entries}, result)

# A copy of the forward value of node id.
def value(tape: Tape, +id: Nat) -> Tape & Result<&2, &1, Error, Core.Dynamic>:
  Tape{entries} = tape
  found_tape(value_counted(id, counted(entries)))

def Graded() -> Type:
  List<&1, Entry> & Result<&2, &1, Error, Maybe<&1, Core.Dynamic>>

def held_copy(pair: Core.Dynamic & Core.Dynamic) -> Gradient.Slot & Maybe<&1, Core.Dynamic>:
  (gradient, copy) = pair
  (Gradient.Held{gradient}, Some{copy})

def copied_grad(read: Maybe<&1, Core.Dynamic>) -> Gradient.Slot & Maybe<&1, Core.Dynamic>:
  match read:
    case None{}:
      (Gradient.Unset{}, None{})
    case Some{gradient}:
      held_copy(Core.clone_dynamic(gradient))

def graded_entry(node: Node, +shape: Shape.Shape, value: Core.Dynamic, saved: Saved, rest: List<&1, Entry>,
  pair: Gradient.Slot & Maybe<&1, Core.Dynamic>) -> Graded():
  (slot, copy) = pair
  (Entry{node, shape, value, saved, slot} <> rest, Done{copy})

def found_grad(entry: Entry, rest: List<&1, Entry>) -> Graded():
  Entry{node, +shape, value, saved, slot} = entry
  graded_entry(node, shape, value, saved, rest, copied_grad(Gradient.grad(slot)))

def prepend_graded(entry: Entry, pair: Graded()) -> Graded():
  (rest, result) = pair
  (entry <> rest, result)

def grad_at(entries: List<&1, Entry>, +count: Nat, +id: Nat, same: Bool) -> Graded():
  match entries count same:
    case Nil{} count same:
      (Nil{}, Fail{UnknownNode{}})
    case Con{entry, rest} 0n same:
      (entry <> rest, Fail{UnknownNode{}})
    case Con{entry, rest} 1n+below True{}:
      found_grad(entry, rest)
    case Con{entry, rest} 1n+below False{}:
      prepend_graded(entry, grad_at(rest, below, id, at_head(below, id)))

def grad_counted(+id: Nat, pair: List<&1, Entry> & Nat) -> Graded():
  (entries, +count) = pair
  grad_at(entries, count, id, at_head(count, id))

def graded_tape(pair: Graded()) -> Tape & Result<&2, &1, Error, Maybe<&1, Core.Dynamic>>:
  (entries, result) = pair
  (Tape{entries}, result)

# A copy of the gradient slot of node id: None until a backward reaches it.
def grad(tape: Tape, +id: Nat) -> Tape & Result<&2, &1, Error, Maybe<&1, Core.Dynamic>>:
  Tape{entries} = tape
  graded_tape(grad_counted(id, counted(entries)))

# Recording.

def pushed(node: Node, value: Core.Dynamic, saved: Saved, pair: List<&1, Entry> & Nat) -> Tape & Nat:
  Core.Packed{+shape, tensor} = value
  (entries, +count) = pair
  (Tape{Entry{node, shape, Core.Packed{shape, tensor}, saved, Gradient.Unset{}} <> entries}, count)

def done_id(pair: Tape & Nat) -> Tape & Result<&2, &2, Error, Nat>:
  (tape, id) = pair
  (tape, Done{id})

def is_constant(node: Node) -> Bool:
  match node:
    case Constant{}:
      True{}
    case other:
      False{}

# Whether node id of a newest-first list of count nodes is a constant.
def constant_at(nodes: List<&2, Node>, +count: Nat, +id: Nat, same: Bool) -> Bool:
  match nodes count same:
    case Nil{} count same:
      False{}
    case Con{node, rest} 0n same:
      False{}
    case Con{node, rest} 1n+below True{}:
      is_constant(node)
    case Con{node, rest} 1n+below False{}:
      constant_at(rest, below, id, at_head(below, id))

# The operand ids of a node, in operand order.
def operand_ids(node: Node) -> List<&2, Nat>:
  match node:
    case Leaf{}:
      []
    case Add{a, b, left, right}:
      [a, b]
    case Sub{a, b, left, right}:
      [a, b]
    case Mul{a, b, left, right}:
      [a, b]
    case Sum{a, input}:
      [a]
    case SumDims{a, input, kept}:
      [a]
    case Matmul{a, b, left, right}:
      [a, b]
    case MatmulNT{a, b, left, right}:
      [a, b]
    case Relu{a, input}:
      [a]
    case MseLoss{a, b, left, right}:
      [a, b]
    case CrossEntropy{a, input}:
      [a]
    case BatchedMatmul{a, b, left, right}:
      [a, b]
    case Constant{}:
      []
    case Frozen{id, inner}:
      operand_ids(inner)

def frozen_if(+id: Nat, node: Node, constant: Bool) -> Node:
  match constant:
    case True{}:
      Frozen{id, node}
    case False{}:
      node

# node wrapped in Frozen{id, .} for each operand id that is a constant of the
# newest-first list of count nodes, the first operand outermost.
def freeze(ids: List<&2, Nat>, +nodes: List<&2, Node>, +count: Nat, node: Node) -> Node:
  match ids:
    case Nil{}:
      node
    case Con{+id, rest}:
      frozen_if(id, freeze(rest, nodes, count, node), constant_at(nodes, count, id, at_head(count, id)))

# Whether every id of a list is a constant of the newest-first list of count
# nodes.
def all_constant(ids: List<&2, Nat>, +nodes: List<&2, Node>, +count: Nat) -> Bool:
  match ids:
    case Nil{}:
      True{}
    case Con{+id, rest}:
      Bool.and(constant_at(nodes, count, id, at_head(count, id)), all_constant(rest, nodes, count))

# A node whose operands are all constants is a constant itself, as in PyTorch
# an operation on tensors that do not require grad gives one that does not;
# any other node is frozen at each operand that is a constant.
def settled(+node: Node, +ids: List<&2, Nat>, +nodes: List<&2, Node>, +count: Nat, constant: Bool) -> Node:
  match constant:
    case True{}:
      Constant{}
    case False{}:
      freeze(ids, nodes, count, node)

def marked(+node: Node, value: Core.Dynamic, saved: Saved, pair: List<&1, Entry> & List<&2, Node>) -> Tape & Nat:
  (entries, +nodes) = pair
  +count = List.length(&2, Node, nodes)
  +ids = operand_ids(node)
  pushed(settled(node, ids, nodes, count, all_constant(ids, nodes, count)), value, saved, (entries, count))

# The new entry's id is the number of entries before it. A recording has at
# least one operand; when they are all constants it records a constant, and
# otherwise each operand that is a constant freezes the node.
def push(node: Node, value: Core.Dynamic, saved: Saved, entries: List<&1, Entry>) -> Tape & Result<&2, &2, Error, Nat>:
  done_id(marked(node, value, saved, listed(entries)))

# Records a leaf.
def leaf(tape: Tape, value: Core.Dynamic) -> Tape & Nat:
  Tape{entries} = tape
  pushed(Leaf{}, value, Unsaved{}, counted(entries))

# Records a constant: a leaf that takes no gradient, PyTorch's tensor with
# requires_grad=False. No VJP sends it a contribution, so its slot reads None
# after any backward that does not start at it; Sgd.step takes it as a leaf
# and keeps it.
def constant(tape: Tape, value: Core.Dynamic) -> Tape & Nat:
  Tape{entries} = tape
  pushed(Constant{}, value, Unsaved{}, counted(entries))

def recorded(node: Node, saved: Saved, entries: List<&1, Entry>, result: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Tape & Result<&2, &2, Error, Nat>:
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{TensorError{error}})
    case Done{output}:
      push(node, output, saved, entries)

def multiplied_record(node: Node, entries: List<&1, Entry>,
  result: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Tape & Result<&2, &2, Error, Nat>:
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{MatrixError{error}})
    case Done{triple}:
      (left, right, output) = triple
      push(node, output, Operands{left, right}, entries)

def dropped(node: Node, entries: List<&1, Entry>, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Tape & Result<&2, &2, Error, Nat>:
  (operands, result) = pair
  recorded(node, Unsaved{}, entries, result)

def kept(node: Node, entries: List<&1, Entry>, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Tape & Result<&2, &2, Error, Nat>:
  (operands, result) = pair
  (left, right) = operands
  recorded(node, Operands{left, right}, entries, result)

def loss_recorded(node: Node, entries: List<&1, Entry>, result: Result<&2, &1, Loss.Error, Loss.Recorded>) ->
  Tape & Result<&2, &2, Error, Nat>:
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{LossError{error}})
    case Done{Loss.Recorded{loss, left, right}}:
      push(node, loss, Operands{left, right}, entries)

def loss_dropped(node: Node, entries: List<&1, Entry>, pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Loss.Error, Loss.Recorded>) ->
  Tape & Result<&2, &2, Error, Nat>:
  (operands, result) = pair
  loss_recorded(node, entries, result)

def loss_kept(node: Node, entries: List<&1, Entry>, pair: Core.Dynamic & Result<&2, &1, Loss.Error, Loss.Recorded>) ->
  Tape & Result<&2, &2, Error, Nat>:
  (input, result) = pair
  loss_recorded(node, entries, result)

# torch.matmul on the tape: two rank-2 operands record `Matmul`, the
# product `matmul` records, and every other pair records `BatchedMatmul`.
def batched_record(+a: Nat, +b: Nat, +left_shape: Shape.Shape, +right_shape: Shape.Shape, entries: List<&1, Entry>,
  left: Core.Dynamic, right: Core.Dynamic, matrices: Bool) -> Tape & Result<&2, &2, Error, Nat>:
  match matrices:
    case True{}:
      multiplied_record(Matmul{a, b, left_shape, right_shape}, entries, Matrix.matmul(left, right))
    case False{}:
      multiplied_record(BatchedMatmul{a, b, left_shape, right_shape}, entries, Batched.matmul(left, right))

def operated(kind: Binary, +a: Nat, +b: Nat, +left_shape: Shape.Shape, +right_shape: Shape.Shape, entries: List<&1, Entry>,
  left: Core.Dynamic, right: Core.Dynamic) -> Tape & Result<&2, &2, Error, Nat>:
  match kind:
    case Plus{}:
      dropped(Add{a, b, left_shape, right_shape}, entries, Elementwise.add(left, right))
    case Minus{}:
      dropped(Sub{a, b, left_shape, right_shape}, entries, Elementwise.sub(left, right))
    case Times{}:
      kept(Mul{a, b, left_shape, right_shape}, entries, Elementwise.mul(left, right))
    case Product{}:
      multiplied_record(Matmul{a, b, left_shape, right_shape}, entries, Matrix.matmul(left, right))
    case ProductNT{}:
      multiplied_record(MatmulNT{a, b, left_shape, right_shape}, entries, Matrix.matmul_nt(left, right))
    case SquaredError{}:
      loss_dropped(MseLoss{a, b, left_shape, right_shape}, entries, Loss.mse_loss(left, right))
    case BatchedProduct{}:
      batched_record(a, b, left_shape, right_shape, entries, left, right,
        Batched.both_matrices(left_shape, right_shape))

def computed(kind: Binary, +a: Nat, +b: Nat, entries: List<&1, Entry>, left: Core.Dynamic & Shape.Shape,
  right: Core.Dynamic & Shape.Shape) -> Tape & Result<&2, &2, Error, Nat>:
  (left, +left_shape) = left
  (right, +right_shape) = right
  operated(kind, a, b, left_shape, right_shape, entries, left, right)

def right_found(kind: Binary, +a: Nat, +b: Nat, left: Core.Dynamic, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, right) = pair
  match right:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{right}:
      computed(kind, a, b, entries, Core.shape(left), Core.shape(right))

def left_found(kind: Binary, +a: Nat, +b: Nat, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, left) = pair
  match left:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{left}:
      right_found(kind, a, b, left, value_counted(b, counted(entries)))

# Records `a op b` from copies of the operands' values. An operand id the tape
# does not hold is refused with UnknownNode, and an operation the tensor layer
# refuses (shapes that do not broadcast, or a matmul of the wrong ranks or
# inner dimensions) with its error; the tape comes back unchanged.
def binary(tape: Tape, kind: Binary, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  left_found(kind, a, b, value_counted(a, counted(entries)))

def add(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Plus{}, a, b)

def sub(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Minus{}, a, b)

def mul(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Times{}, a, b)

def matmul(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, Product{}, a, b)

# Records torch.matmul(a, b): batched products and 1-D promotion
# (Batched.matmul). Two rank-2 operands record exactly what `matmul` records.
# A rank-0 operand is refused with RankMismatch (105), unequal inner extents
# with InnerDimensionMismatch (106), batch dimensions that do not broadcast
# with IncompatibleBroadcast (6); the tape comes back unchanged.
def batched_matmul(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, BatchedProduct{}, a, b)

# Records a @ b^T, with b read in place: PyTorch's linear(a, b) without a
# bias.
def matmul_nt(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, ProductNT{}, a, b)

# The tape without its newest entry.
def popped(tape: Tape) -> Tape:
  Tape{entries} = tape
  match entries:
    case Nil{}:
      Tape{Nil{}}
    case Con{entry, rest}:
      Tape{rest}

# A refused sum drops the product, the newest entry, so the tape comes back
# as it was before the product was recorded.
def unrecorded(pair: Tape & Result<&2, &2, Error, Nat>) -> Tape & Result<&2, &2, Error, Nat>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (popped(tape), Fail{error})
    case Done{id}:
      (tape, Done{id})

def biased(+bias: Nat, pair: Tape & Result<&2, &2, Error, Nat>) -> Tape & Result<&2, &2, Error, Nat>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (tape, Fail{error})
    case Done{+product}:
      unrecorded(add(tape, product, bias))

def linear_checked(+x: Nat, +w: Nat, +bias: Nat, entries: List<&1, Entry>, known: Bool) -> Tape & Result<&2, &2, Error, Nat>:
  match known:
    case False{}:
      (Tape{entries}, Fail{UnknownNode{}})
    case True{}:
      biased(bias, matmul_nt(Tape{entries}, x, w))

def linear_counted(+x: Nat, +w: Nat, +bias: Nat, pair: List<&1, Entry> & Nat) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, +count) = pair
  linear_checked(x, w, bias, entries, Nat.is_lt(bias, count))

# Linear: records x @ W^T with W of [out, in], PyTorch's layout, then adds
# the bias b of [out] to every row. The id is the sum's; the product is the
# entry before it. The bias id is checked against the tape the call receives,
# before the product is recorded, so the product's own id never stands in
# for a missing bias. A refusal of any step (an unknown id, a product or a
# sum the tensor layer refuses) comes back with its error and the tape as it
# was.
def linear(tape: Tape, +x: Nat, +w: Nat, +bias: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  linear_counted(x, w, bias, counted(entries))

def summed(+a: Nat, entries: List<&1, Entry>, +shape: Shape.Shape, result: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) ->
  Tape & Result<&2, &2, Error, Nat>:
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{TensorError{error}})
    case Done{pair}:
      (input, output) = pair
      push(Sum{a, shape}, output, Unsaved{}, entries)

def sum_shaped(+a: Nat, entries: List<&1, Entry>, pair: Core.Dynamic & Shape.Shape) -> Tape & Result<&2, &2, Error, Nat>:
  (input, +shape) = pair
  summed(a, entries, shape, Reduction.sum_dynamic(input))

def sum_found(+a: Nat, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, input) = pair
  match input:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{input}:
      sum_shaped(a, entries, Core.shape(input))

# Records the sum of every element of a, a rank-0 tensor.
def sum(tape: Tape, +a: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  sum_found(a, value_counted(a, counted(entries)))

def dims_error(error: SumDims.Error) -> Error:
  match error:
    case SumDims.ReduceError{error}:
      ReduceError{error}
    case SumDims.TransformError{error}:
      TransformError{error}

def dims_summed(node: Node, entries: List<&1, Entry>, pair: Core.Dynamic & Result<&2, &1, SumDims.Error, Core.Dynamic>) ->
  Tape & Result<&2, &2, Error, Nat>:
  (input, result) = pair
  match result:
    case Fail{error}:
      (Tape{entries}, Fail{dims_error(error)})
    case Done{output}:
      push(node, output, Unsaved{}, entries)

def dims_planned(+a: Nat, entries: List<&1, Entry>, input: Core.Dynamic, +shape: Shape.Shape, keepdim: Bool,
  planned: Result<&2, &2, SumDims.Error, SumDims.Plan>) -> Tape & Result<&2, &2, Error, Nat>:
  match planned:
    case Fail{error}:
      (Tape{entries}, Fail{dims_error(error)})
    case Done{SumDims.Plan{+kept, +dropped}}:
      dims_summed(SumDims{a, shape, kept}, entries, SumDims.sum_planned(input, SumDims.Plan{kept, dropped}, keepdim))

# The signed axes, each wrapped among the operand's axes (its rank, or 1 at
# rank 0), as PyTorch wraps them, before the plan checks them.
def dims_shaped(+a: Nat, +axes: List<&2, Signed.Position>, keepdim: Bool, entries: List<&1, Entry>, pair: Core.Dynamic & Shape.Shape) ->
  Tape & Result<&2, &2, Error, Nat>:
  (input, +shape) = pair
  dims_planned(a, entries, input, shape, keepdim, SumDims.plan(shape, Signed.wrap_all(SumDims.limit(Shape.dimensions(shape)), axes)))

def dims_found(+a: Nat, +axes: List<&2, Signed.Position>, keepdim: Bool, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, input) = pair
  match input:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{input}:
      dims_shaped(a, axes, keepdim, entries, Core.shape(input))

# Records PyTorch's a.sum(dim=axes, keepdim=keepdim), SumDims.sum_dims of a
# copy of a's value: the kept shape when keepdim is True, else the dropped
# one. The axes are PyTorch's signed ones: Last{0} is -1, a's last axis. An
# empty axis list sums every axis. An axis outside -rank..rank-1 is refused
# with AxisOutOfBounds (13), two axes that name one axis with DuplicateAxis
# (14), an unknown id with UnknownNode; the tape comes back unchanged.
def sum_dims(tape: Tape, +a: Nat, +axes: List<&2, Signed.Position>, keepdim: Bool) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  dims_found(a, axes, keepdim, value_counted(a, counted(entries)))

def relu_zeroed(+a: Nat, entries: List<&1, Entry>, +shape: Shape.Shape, input: Core.Dynamic,
  zero: Result<&2, &1, Core.Error, Core.Tensor<Reduction.scalar_shape()>>) -> Tape & Result<&2, &2, Error, Nat>:
  match zero:
    case Fail{error}:
      (Tape{entries}, Fail{TensorError{error}})
    case Done{tensor}:
      kept(Relu{a, shape}, entries,
        Elementwise.binary(Elementwise.ReluLeft{}, input, Core.pack(Reduction.scalar_shape(), tensor)))

def relu_shaped(+a: Nat, entries: List<&1, Entry>, pair: Core.Dynamic & Shape.Shape) -> Tape & Result<&2, &2, Error, Nat>:
  (input, +shape) = pair
  relu_zeroed(a, entries, shape, input, Core.new(Reduction.scalar_shape(), 0.0))

def relu_found(+a: Nat, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, input) = pair
  match input:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{input}:
      relu_shaped(a, entries, Core.shape(input))

# Records relu(a), PyTorch's relu, of a's shape: the pointwise product of a
# copy of a's value with ReluLeft against a rank-0 +0.0, on the parallel
# elementwise build. The copy is saved for the VJP.
def relu(tape: Tape, +a: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  relu_found(a, value_counted(a, counted(entries)))

# Records mse_loss(a, b), PyTorch's mse_loss(input, target) with mean
# reduction, a rank-0 tensor, from copies of the operands' values. The
# differences a - b and b - a, over the shapes' broadcast, are saved for the
# VJP. Shapes that do not broadcast are refused with IncompatibleBroadcast
# (6); an unknown id with UnknownNode; the tape comes back unchanged.
def mse_loss(tape: Tape, +a: Nat, +b: Nat) -> Tape & Result<&2, &2, Error, Nat>:
  binary(tape, SquaredError{}, a, b)

def entropy_shaped(+a: Nat, +targets: List<&2, U32>, entries: List<&1, Entry>, pair: Core.Dynamic & Shape.Shape) ->
  Tape & Result<&2, &2, Error, Nat>:
  (input, +shape) = pair
  loss_kept(CrossEntropy{a, shape}, entries, Loss.cross_entropy(input, targets))

def entropy_found(+a: Nat, +targets: List<&2, U32>, pair: Found()) -> Tape & Result<&2, &2, Error, Nat>:
  (entries, input) = pair
  match input:
    case Fail{error}:
      (Tape{entries}, Fail{UnknownNode{}})
    case Done{input}:
      entropy_shaped(a, targets, entries, Core.shape(input))

# Records cross_entropy(a, targets), PyTorch's cross_entropy with class
# indices and mean reduction, a rank-0 tensor, from a copy of a's value: a
# must be [rows, columns] logits and `targets` one class below `columns`
# per row. The log_softmax lp and the one-hot rows of the targets are saved
# for the VJP. Logits of another rank are refused with NotLogits (107),
# targets that do not fit with TargetOutOfRange (108), an unknown id with
# UnknownNode; the tape comes back unchanged.
def cross_entropy(tape: Tape, +a: Nat, +targets: List<&2, U32>) -> Tape & Result<&2, &2, Error, Nat>:
  Tape{entries} = tape
  entropy_found(a, targets, value_counted(a, counted(entries)))

def error_code(error: Error) -> U32:
  match error:
    case UnknownNode{}:
      103
    case DanglingOperand{}:
      104
    case TensorError{error}:
      Core.error_code(error)
    case MatrixError{Matrix.TensorError{error}}:
      Core.error_code(error)
    case MatrixError{Matrix.ShapeError{error}}:
      Shape.error_code(error)
    case MatrixError{Matrix.RankMismatch{}}:
      105
    case MatrixError{Matrix.InnerDimensionMismatch{}}:
      106
    case TransformError{error}:
      Transforms.error_code(error)
    case BroadcastError{error}:
      Broadcast.error_code(error)
    case ReduceError{error}:
      Reduce.error_code(error)
    case GradientError{error}:
      Gradient.error_code(error)
    case LossError{error}:
      Loss.error_code(error)
