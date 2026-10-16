# Tensor backward

`src/autograd/tensor-backward.bend` is the reverse-mode engine for tensors. It
keeps the tape discipline of the [scalar engine](backward.md): entries newest
first, an id is the number of entries recorded before the node, a lookup
compares ids and never subtracts, the seed goes to the output the caller
selects, and the sweep visits the output and every older entry, newest first.
What changes is the gradient: a tensor of the node's shape, held in a
[gradient slot](gradient-slots.md), and each contribution is reduced to its
operand's shape with [`sum_to_size`](sum-to-size.md) before it is added.

## Tape and recording

A `Tape` is a list of `Entry{node, shape, value, saved, grad}`, newest first.
`node` is `Leaf`, `Add{a, b, left, right}`, `Sub{...}`, `Mul{...}`,
`Matmul{...}`, `MatmulNT{...}`, `Sum{a, input}`, `Relu{a, input}`,
`MseLoss{a, b, left, right}` or `CrossEntropy{a, input}`: the operand
ids and the operand shapes. `MatmulNT` is `left @ right^T` with `right` read in place,
PyTorch's `linear(left, right)` without a bias. `value` is the forward value,
`saved` holds the operand values a product needs for its VJP
(`Operands{left, right}` for `Mul`, `Matmul` and `MatmulNT`, `Operands{x, zero}`
for `Relu`, its input and the rank-0 `+0.0` of its forward, `Operands{e, d}`
for `MseLoss`, the differences `b - a` and `a - b`, `Operands{lp, onehot}`
for `CrossEntropy`, its `log_softmax` and the one-hot rows of its targets,
`Unsaved` for the rest), and `grad` is the entry's gradient slot.

`TensorBackward.leaf(tape, value)` returns the new tape and the id.
`add`, `sub`, `mul`, `matmul`, `matmul_nt` and `mse_loss` take two ids, `sum`
and `relu` one, and `cross_entropy` one id and a `List<U32>` of class
indices; each
returns the new tape and `Done{id}`. A recording reads copies of its operands'
values through the same lookup as `TensorBackward.value`, runs the tensor
operation (`Elementwise.add`, `sub` or `mul` with broadcasting,
`Matrix.matmul`, `Matrix.matmul_nt`, `Reduction.sum_dynamic`, or for `relu`
`Elementwise.binary(ReluLeft, x, +0.0)` against a rank-0 `+0.0`, or
`Loss.mse_loss` and `Loss.cross_entropy`, see [losses](losses.md)) and pushes
the result. `linear(tape, x, w, b)` is a layer in PyTorch's layout, `W` of
`[out, in]`: it records `matmul_nt(x, w)` and then `add` of that product and
the bias `b` of `[out]`, and returns the sum's id; the product is the entry
before it. All three ids are checked against the tape the call receives, so a
bias id equal to the number of entries is unknown, not the product that would
take that id. If the bias does not broadcast, the product is dropped again.
An id the tape does not hold
gives `Fail{UnknownNode}` (103); an operation the tensor layer refuses gives
its error: `IncompatibleBroadcast` (6), a matmul rank mismatch (105) or inner
dimension mismatch (106), `cross_entropy` logits that are not of rank 2
(`NotLogits`, 107) or targets that do not fit them (`TargetOutOfRange`, 108). A refused recording, `linear` included, hands the
tape back unchanged.

Every call hands the tape back, because tensors have one owner.

## backward

`TensorBackward.backward(tape, output)` returns the tape and `Done{Unit}` or a
typed error:

1. If `output` is not an id on the tape: `UnknownNode` (103).
2. If an entry has an operand id that is not older than the entry itself:
   `DanglingOperand` (104). The recording functions cannot build such a tape.
3. Otherwise the engine computes one working gradient per entry and
   accumulates it into the entry's slot.

Both refusals hand the tape back unchanged.

The sweep:

- The seed is ones of the output's shape, `Core.new(shape, 1.0)`, as PyTorch's
  `output.backward(torch.ones_like(output))`. Entries newer than the output get
  nothing.
- Each visited entry with a working gradient `g` sends its vector-Jacobian
  products to its operands, the first operand before the second. An entry with
  no working gradient sends nothing.
- The products follow PyTorch's derivative formulas:

  | Node | First operand receives | Second operand receives |
  | --- | --- | --- |
  | `Add` | `sum_to_size(g, left)` | `sum_to_size(g, right)` |
  | `Sub` | `sum_to_size(g, left)` | `sum_to_size(-g, right)`: `-g` word by word first, then reduced |
  | `Mul` | `sum_to_size(g * right, left)` | `sum_to_size(g * left, right)` |
  | `Sum` | `broadcast_to(g, input)` | |
  | `Matmul` | `g @ right^T` | `left^T @ g` |
  | `MatmulNT` | `g @ right` | `g^T @ left`, `Matrix.matmul_tn`, `g` read in place |
  | `Relu` | `sum_to_size(relu_backward(x, g), x)`: `g`'s word where the saved input `x` is above zero or NaN, `+0.0` where it is at most zero | |
  | `MseLoss` | `sum_to_size((2 / N * d) * g, left)`, `d = a - b` saved | `sum_to_size((2 / N * e) * g, right)`, `e = b - a` saved |
  | `CrossEntropy` | two contributions to `a`: `-(g / N)` at each target and `+0.0` elsewhere, then `-(exp(lp) * r)` with `r = sum(gout)`, `-(g / N)` for one column and `-(g / N) + 0.0` for more; the slot adds them, PyTorch's `gout - exp(lp) * sum(gout)` | |

  `left` and `right` are the saved operand values (and shapes). `-g` uses the
  new `Elementwise.NegateLeft` operation, `scalar(NegateLeft, l, r) =
  F32.neg(l)`, which flips the sign bit like PyTorch's `neg`. The `Mul`
  products and `-g` come from `Elementwise.binary`, which builds its output in
  parallel halves above 2^15 cells. The live law `binary_replayed`
  ([contract families](contract-families.md#elementwisebinary-exports-its-replayed-result))
  states that this build writes the pointwise family's words whatever the
  split, so the [value laws](#vjp-values) chain through it. `relu` is
  `ReluLeft`, `scalar(ReluLeft, x, r) = relu(x)`, `+0.0` where `x < 0` and
  `x`'s own word otherwise, so `-0.0` stays `-0.0` and a NaN keeps its sign
  and payload, as PyTorch's `clamp_min(x, 0)`. Its gradient is
  `ReluBackward`, `scalar(ReluBackward, x, g) = relu_backward(x, g)`, `+0.0`
  where `x <= 0` and `g`'s own word otherwise, as PyTorch's
  `threshold_backward(g, relu(x), 0)`: `+0.0` at `x = +0.0` and `-0.0`, and
  `g` at a NaN `x`. PyTorch compares `relu(x)`, the tape compares `x`; both
  are at most zero for the same words. `x < 0` and `x <= 0` are
  `Elementwise.below_zero` and `at_most_zero`, ranges of the F32 bits read as
  a `U32` (the F32 comparisons of Base are opaque, so no law could state
  them); `src/tensor/relu-LAWS.bend` pins both ranges, the scalar cases and
  PyTorch's words at twenty edge words.
  `sum_to_size`
  copies `g` when no axis is reduced, so `-0.0` stays `-0.0`, and otherwise
  folds each fiber in ascending flat order from `+0.0`. `@` is the ordered
  matmul of the tensor layer.
- Each contribution is accumulated into its operand's working slot with
  `Gradient.accumulate`: the first moves in unchanged, each later one is added
  position by position, the held total first. So a node used several times
  receives its contributions from its consumers in decreasing id order, as in
  the scalar engine and in PyTorch.

After the sweep each working gradient is accumulated into its entry's slot
with the same rule, so a second `backward` adds, like PyTorch's
`retain_graph=True`, and a slot that received nothing keeps what it held.

**Reads.** `TensorBackward.value(tape, id)` returns a copy of the forward
value. `TensorBackward.grad(tape, id)` returns `None` for a node no backward
reached, like PyTorch's `.grad`, and a copy of the gradient otherwise. Both
give `UnknownNode` for an unknown id. `TensorBackward.zero_grad(tape)` resets
every slot, like `zero_grad(set_to_none=True)`.

**Cost.** Recording copies each operand once (the copies of `Mul` and
`Matmul` operands are kept as saved values). The sweep copies `g` once per
contribution that `sum_to_size` does not reduce, allocates `-g` for `Sub`, and
copies `g` twice and each saved matmul operand once (its transpose) for
`Matmul`, and `g` twice and each saved operand once (a plain clone) for
`MatmulNT`; `Relu` copies its input twice when it records (the value lookup
and the parallel build's own copy) and `g` once in its VJP (`sum_to_size`). The sweep is sequential: the order in which contributions reach a
slot is part of the contract. Within one VJP, the `Mul` and `Sub` products
fork into parallel halves above 2^15 cells, as every `Elementwise.binary`
does.

## Laws

`src/autograd/tensor-backward-LAWS.bend` states 16 laws, and
`tensor-backward-PROOF.bend` proves them. They quantify over every tape, list
of slots, shape and tensor.

- `negate_left_scalar`: `scalar(NegateLeft, left, right) == F32.neg(left)`.
- `tensor_vjp_leaf`: a leaf sends nothing.
- `tensor_sum_vjp_pointwise`: through the live `broadcast_to_pointwise` law of
  the pointwise family, for a replayed gradient `g` whose shape broadcasts to
  exactly the operand's shape, the `Sum` VJP sends one contribution, to the
  operand, with the operand's dimensions, and every position `i` holds the
  word of `g` at `left_map(plan, i)`. For the rank-0 gradient of a sum that is
  `g`'s only word at every position.
- `tensor_seed_selected_output` (C1): for any entries newer than the output,
  those entries get no gradient, and the sweep starts at the output with
  `Core.new(shape, 1.0)`. `tensor_seeded_sweep` states that start: ones at the
  output and no gradient for every older entry.
- `tensor_unreached_sends_nothing`: an entry with no working gradient sweeps
  the older entries with their working gradients unchanged.
- `tensor_deliver_in_order`: the contributions of one VJP are delivered in list
  order, the first operand before the second.
- `tensor_bump_target`: a contribution to node `k` reaches exactly slot `k`,
  through `Gradient.accumulate`, for every list of slots; every other slot is
  kept. With `accumulate_mapping` this fixes the held-first addition.
- `tensor_merge_accumulates` and `tensor_merge_keeps_unreached`: after the
  sweep, a reached entry accumulates its working gradient into its slot, and an
  unreached entry keeps its slot.
- `tensor_backward_unknown` and `tensor_backward_dangling`: the two refusals,
  with the tape unchanged. `tensor_backward_success`: otherwise backward runs
  the sweep from the selected output. `tensor_wired_matches_operand_bounds`:
  the wiring check equals a separate specification of operand bounds.
- `tensor_zero_grad_empties` and `tensor_zero_grad_keeps_nodes`.

These laws state the engine's structure and the `Sum` VJP's values. The
[VJP values](#vjp-values) below state the `Add`, `Sub`, `Mul`, `Matmul`,
`MatmulNT`, `Relu`, `MseLoss` and `CrossEntropy` VJPs' values.

The family controls in [`verification/families/tensor_vjp/`](../verification/families/README.md#tensor-vjp)
check the VJP value laws: five literal consumers, seven false twins and eighteen
mutants of `tensor-backward.bend`, each rejected by the source checker and by
BendTT. The autograd gate applies ten proof controls and two error controls to a copy
of the sources, and the proofs reject each one: `NegateLeft` keeping the sign
(at `negate_left_scalar`), seeding the newest node (`tensor_backward_success`),
seeding zeros and a head check that matches a larger id
(`tensor_seed_selected_output`), a slot update that overwrites
(`tensor_bump_target`), a merge that overwrites (`tensor_merge_accumulates`),
an unreached entry that drops the older gradients
(`tensor_unreached_sends_nothing`), a `Sum` VJP that sends to node 0
(`tensor_sum_vjp_pointwise`), a wiring check that skips the operand bounds and
a `zero_grad` that keeps the slot; and each refusal reported as the other.

## VJP values

`src/autograd/tensor-vjp-LAWS.bend` states the values of the `Add`, `Sub`,
`Mul`, `Matmul`, `MatmulNT`, `Relu`, `MseLoss` and `CrossEntropy` VJPs, and `tensor-vjp-PROOF.bend` proves them. The `Add`,
`Sub` and `Mul` proofs go through laws alone: the parallel product, whose
words are the pointwise family's, then `sum_to_size`, each handing the next a
replayed tensor. The `Matmul` proof composes the clone, transpose and
ordered-matmul transport laws with supporting value lemmas of the permute copy
loop (see `tensor_matmul_vjp_values` below). Each `Add`/`Sub`/`Mul` law takes a [replayed](contract-families.md#replayed-tensors-are-the-input-class)
gradient `g` (and, for `Mul`, replayed saved operands `x` and `y`) and the
broadcast plans the VJP uses, and concludes a `PairContract`: the VJP sends
exactly two contributions, the first to `a` with the left operand's
dimensions, the second to `b` with the right operand's, and every in-bounds
cell holds the stated word.

The word of a reduced contribution is `reduced(target, shape, plan, values,
index)`: the contribution's own word `values[index]` when the target has its
dimensions (nothing is reduced, so `-0.0` stays `-0.0`), and otherwise
`ReduceLaws.fiber`, the words that land on the cell added in index order from
`+0.0`. `cells(operation, plan, ...)` lists the words of a pointwise product:
word `k` is `scalar(operation, left[left_map(plan, k)], right[right_map(plan, k)])`.

- `tensor_add_vjp_values`: for operand shapes that each broadcast to exactly
  `g`'s shape, `a` receives `reduced` of `g`'s words to the left shape, and
  `b` the same to the right shape.
- `tensor_sub_vjp_values`: `a` receives what `Add` sends it; `b` receives
  `reduced` of `cells(NegateLeft, ...)` of `g` against a rank-0 `+0.0`, so each
  word of `g` has its sign flipped before any word is added.
- `tensor_mul_vjp_values`: `a` receives `reduced` of `cells(Mul, ...)` of `g`
  and `y`, and `b` of `g` and `x`.

Two transport laws make the chain possible. Each has the form of
`broadcast_to_replayed`: for every `P`, if `P` holds for every result of the
stated form, `P` holds for the call.

- `tensor_product_replayed` (`tensor-vjp-LAWS.bend`): `Elementwise.binary`
  of two replayed inputs whose shapes broadcast hands both inputs back with
  the same words and gives a replayed output of the plan's shape whose words
  are `cells`. It restates the tensor law `binary_replayed` for the backward,
  whose proof walks the parallel build with live code.
- `sum_to_size_replayed` (`sum-to-size-fold-LAWS.bend`): `sum_to_size` of a
  replayed contribution whose words are `values` hands it back with the same
  words and gives a replayed output with the target's dimensions whose cells
  hold `summed`, the value `reduced` names. The output is the clone's second
  half or `reduce_to`'s output; the proof covers both.

The hypotheses name the plans the VJP computes, so a law says nothing about a
pair of shapes that does not broadcast. A recording on the tape only exists
for shapes that broadcast, and the consumers in
`verification/families/tensor_vjp/` instantiate every law on literal shapes.

`tensor_matmul_vjp_values` states the `Matmul` VJP. With the saved
operands `x` (left) and `y` (right) and the gradient `g`, all replayed, it
names as hypotheses the plans the VJP computes: the transpose plans of `y`
and `x` from `Layout.permutation` by `swapped_axes(., 0, 1)`, with both axes
in range, and the matmul plans `first` of `g @ y^T` and `second` of
`x^T @ g`, whose outputs have the left and the right operand's dimensions.
Its conclusion is a `MatmulPairContract`, stated per `(row, column)` as
`matmul_ordered` is: the first contribution goes to `a` with the left
operand's dimensions, the second to `b` with the right operand's, and

- cell `(row, column)` of the first holds `DotFold.dot_fold` from `+0.0` of
  `right_transposed_factors`: for `k` from 0 to `inner - 1`, the pair
  `g[row * inner + k]` and the word of `y` at
  `permute_source(y, swapped, k * columns + column)`;
- cell `(row, column)` of the second holds `DotFold.dot_fold` from `+0.0` of
  `left_transposed_factors`: the word of `x` at
  `permute_source(x, swapped, row * inner + k)` and `g[k * columns + column]`.

`inner` and `columns` are the plan's. For a `[k, n]` operand
`permute_source` of a transposed index is the row-major index of the
swapped coordinate, so the pairs are `g[row, l] * y[column, l]` and
`x[l, row] * g[l, column]`, as PyTorch's `g @ y.T` and `x.T @ g`. For rank 2
the law `permute_source_transposed` states that index identity, and
`nt_reads_transpose` and `tn_reads_transpose` below restate these pairs as the
in-place reads of `matmul_nt` and `matmul_tn`.

`tensor_matmul_nt_vjp_values` states the `MatmulNT` VJP, the backward of
`linear`'s product, with `W[j, l]` read in place. With the saved operands `x`
(left, `[batch, in]`) and `W` (right, `[out, in]`) and the gradient `g`
(`[batch, out]`), all replayed, its hypotheses name the plan `first` of
`matmul(g, W)` and the plan `second` of `matmul_tn(g, x)`, whose outputs have
the left and the right operand's dimensions. Cell `(i, l)` of the first
contribution holds `DotFold.dot_fold` from `+0.0` of `Contracts.factors`:
`g[i * out + j]` and `W[j * in + l]` for `j` from 0 to `out - 1`. Cell
`(j, l)` of the second holds `DotFold.dot_fold` from `+0.0` of
`product_factors` for two `Swapped` operands: `g[i * out + j]` and
`x[i * in + l]` for `i` from 0 to `batch - 1`. No transpose is built and no
index map is involved. The bias gradient is the `Add` entry's:
`tensor_add_vjp_values` gives `sum_to_size(g, [out])`, each column of `g`
added over the batch in index order from `+0.0`. The proof clones `g`, `W`,
`g`'s first half and `x` (`clone_lift`), and composes `matmul_replayed` and
`product_replayed`.

`tensor_relu_vjp_values` states the `Relu` VJP. With the saved input `x`
(replayed, the input's shape), any rank-0 value saved beside it, and a
replayed gradient `g`, it concludes a `OneContract`: the VJP sends exactly one
contribution, to `a`, with the input's dimensions, and cell `i` holds
`reduced(input, shape, plan, values, i)`, where word `k` of `values` is
`scalar(ReluBackward, x[left_map(plan, k)], g[right_map(plan, k)])`. The
proof is the first half of the `Mul` proof: `product_replayed`, then
`sum_to_size_replayed`.

`tensor_mse_vjp_values` states the `MseLoss` VJP. With the saved differences
`e` and `d` (both of the broadcast shape `diff`) and the gradient `g`, all
replayed, and one plan for `g` against `diff`, it concludes a `PairContract`:
`a` receives `reduced` of `cells(MseGrad{2 / N}, ...)` of `g` and `d`, `b` the
same of `g` and `e`, with `N` the cells of `diff` (`Loss.mse_scale`). Word `k`
is `(2 / N * d[k]) * g`, PyTorch's `mse_loss_backward`.
`tensor_cross_entropy_vjp_values` states the `CrossEntropy` VJP: with the
saved `lp` and one-hot rows (both of the input's shape) and `g`, all replayed,
both contributions go to `a` with the input's dimensions, the first holding
`reduced` of `cells(OneHotGrad{N}, ...)` of `g` and the one-hot rows, the
second of `cells(SoftmaxGrad{N, Loss.one_class(input)}, ...)` of `g` and
`lp`, `N` the rows as an F32; `SoftmaxGrad` adds `+0.0` to the row sum only
when the input has more than one column. The `Elementwise` operations take `g` on the left, as `Mul` does, so both
proofs follow the `Mul` proof step for step. [Losses](losses.md) has the
operations and their order.

Two laws tie the in-place reads to the transpose copy, through
`permute_source_transposed` (for an input of dimensions `[rows, columns]`,
`permute_source` with axes `[1, 0]` sends `j * rows + l` to
`l * columns + j`). `nt_reads_transpose`: for a right operand of dimensions
`[columns, inner]`, the factor pairs of a `matmul_nt` cell are the pairs
`right_transposed_factors` reads through `permute_source` on `right^T`.
`tn_reads_transpose`: for a left operand of dimensions `[inner, rows]`, the
pairs of a `matmul_tn` cell are the pairs `left_transposed_factors` reads.
So `matmul_nt(g, y)` and `g @ y^T` of `tensor_matmul_vjp_values` fold the
same words in the same order, and so do `matmul_tn(x, g)` and `x^T @ g`.

The proof of `tensor_matmul_vjp_values` follows `vjp_matmul`. The clone halves of `g` are `ops ++ [First]`
and `ops ++ [Second]`; `transpose_replayed` gives the transposed copy as a
replayed tensor whose words are `permuted_words`; `matmul_replayed` gives the
product's cells as the ordered totals over the copies; and `first_factors`
and `second_factors` turn the factor pairs read on the copies into the pairs
stated on `g`, `x` and `y`. No sum is reduced: each product already has its
operand's dimensions.

## Chain rule

`src/autograd/tensor-chain-LAWS.bend` states what the whole sweep returns, and
`tensor-chain-PROOF.bend` proves it for every tape. The reference, `chained`,
is the chain rule computed by pulling, not by pushing. It visits the entries
newest first and keeps every contribution sent so far in a log, newest first.
At the entry with id `k` it takes out of the log every contribution whose
target is `k` and accumulates them, oldest first, into `k`'s starting slot with
`Gradient.accumulate`. If the entry then holds `g`, the contributions of
`Backward.vjp(node, saved, g)` join the log, the first operand's first. So the
gradient of a node is its starting slot plus each contribution its consumers
sent it, consumers newest first and each consumer's operands in order, and each
contribution is the VJP at the consumer's own chained gradient, whose words the
[VJP value laws](#vjp-values) state. The reference shares `Backward.vjp` and
`Gradient.accumulate` with the engine, and the list helpers `prepend` and
`unsets`, and nothing else: it has no slot list that contributions are pushed
into, no `deliver`, no `bump` and no `at_head`.

- `tensor_sweep_chain_rule`: for any entries, starting working gradients and
  count, when the sweep returns `Done{swept}`, `chained(entries, gradients,
  count, Nil{}) == Done{swept}`.
- `tensor_backward_chain_rule`: for any tape split into newer entries, the
  output and the entries below it, when the pass of a backward from the output
  succeeds, the working gradients it hands to the merge are `chained` from the
  seeds: no gradient above the output, `Core.new(shape, 1.0)` at the output and
  no gradient below. With `tensor_backward_success` and
  `tensor_merge_accumulates`, these are the gradients `backward` adds to the
  slots.

The order of the float sum is part of the statement: a slot adds its
contributions in arrival order, held total first, and each VJP computes its
words in its own order, which its value law states. Both laws hold for every
tape node, since they quantify over `Backward.vjp`. They say nothing about a
sweep that fails: in a failed sweep the reference goes on computing VJPs that
the sweep never ran, and the refusals have their own laws above. A sum over
paths of products of local VJPs is not what either side computes in F32, since
a node's VJP is applied once to its accumulated gradient.

The proof never inspects the reference's side of the tape. Values are affine
in Bend, so a proof that inspects a slot list or a log consumes it, and the
sweep and the reference would each need their own copy. Instead the proof
carries, for the sweep's pending slots, erased witnesses and equations of the
form `take(log, k, w) == (Done{x}, left)`, and inspects only the sweep's state,
each VJP's pass and each `Gradient.accumulate` outcome, once each.

The autograd gate checks `tensor-chain-PROOF.bend` on its own against four
broken sweeps, and the proof fails for each: a delivery that drops a
contribution (at `relay`), a slot update that adds the contribution twice (at
`hit_bumped`), a VJP's contributions delivered in reverse operand order (at
`fired_stop`) and a contribution sent to the next slot (at `relay`). The
tensor backward subgate rejects each of the four as a runtime control too.

## SGD

`src/autograd/sgd.bend` is PyTorch's `torch.optim.SGD(params, lr)` with the
learning rate only. Momentum, dampening, `nesterov`, `weight_decay` and
`maximize` are out of scope. `Sgd.step(tape, lr, params)` takes the
parameters as a list of tape ids, the ids the training loop already holds.
It checks every id first, in list order: an id the tape does not hold is
refused with `UnknownNode` (103), and the id of a node that is not a leaf with
`NotLeaf` (109). Then it checks the learning rate as PyTorch does, `lr < 0.0`:
a rate below zero, `-inf` and the negative denormals included, is refused
with `InvalidLearningRate` (110), and `-0.0`, `+0.0`, every NaN and `+inf`
are accepted. The comparison is `Elementwise.below_zero`, IEEE `x < 0` on the
bits. A refusal leaves the tape unchanged. Then it visits the ids in list
order. A parameter whose slot holds `g` gets the value `p - lr * g`,
one `Elementwise.binary(Descend{lr}, p, g)` on the parallel build, which forks
above 2^15 cells, and keeps `g` in its slot, as PyTorch keeps `p.grad`. A
parameter whose slot is empty, PyTorch's `p.grad is None`, keeps its value. An
id listed twice is updated twice, as in PyTorch. The values change on the
tape, so a forward recorded after the step reads them, as a forward after
`optimizer.step()` reads the updated parameters. `zero_grad` is
`TensorBackward.zero_grad`, which empties every slot (laws
`tensor_zero_grad_empties` and `tensor_zero_grad_keeps_nodes`).

Each cell is `F32.sub(p, F32.mul(lr, g))`: the product is rounded, then the
difference. For every non-NaN result that is the same arithmetic as
PyTorch's `p + (-lr) * g` with two roundings, since F32 rounding is
symmetric in sign. PyTorch's CPU kernel for `param.add_(grad, alpha=-lr)`
rounds once, a fused multiply-add, when it dispatches to AVX2 or AVX512, the
scalar tail included, and twice on its `DEFAULT` kernel. Bend2 has no fused
F32 operation, and its native lane is compiled without contraction. So every
non-NaN word bendtorch computes equals the `DEFAULT` kernel's, and differs
from the fused kernels' only by the rounding of the product. Which NaN a step
gives (its sign and payload) depends on the kernel and on the CPU's
instructions on both sides, and no law or gate states it. On the 11 SGD
scripts that both accept or both refuse (648 words), bendtorch's native words
against PyTorch's: on `DEFAULT`, 637 bit for bit, 11 NaN on both sides with
another sign or payload, no other word differs; on AVX512, 629 bit for bit,
13 such NaNs, and 6 words 1 to 6 ulp apart.

`src/autograd/sgd-LAWS.bend` states 15 laws and `sgd-PROOF.bend` proves them:

- `sgd_descend_scalar`: `scalar(Descend{lr}, p, g) == F32.sub(p, F32.mul(lr, g))`.
- `sgd_step_values`: in transport form, for a replayed value and a replayed
  gradient whose shapes broadcast, the updated parameter holds a value of the
  plan's shape whose word `k` is `scalar(Descend{lr}, p[left_map(plan, k)],
  g[right_map(plan, k)])`, keeps its node, shape and saved values and `g` with
  its words, next to `Done`. It is an instance of `binary_replayed`.
- `sgd_empty_slot_kept`: a parameter whose slot is empty is unchanged.
- `sgd_update_reaches_target`: the update of id `k` reaches exactly the entry
  with `k` older entries, and keeps every other entry.
- `sgd_no_params` and `sgd_params_in_order`: no parameter leaves the entries;
  the first is updated before the rest, which see its result.
- `sgd_refusal_reads_target`, `sgd_refusal_unknown` and
  `sgd_only_leaves_accepted`: id `k` is judged by the node with `k` older
  nodes; an id at or above the number of nodes is `UnknownNode`; a node with
  no operands is accepted and any other is `NotLeaf`.
- `sgd_negative_rate_refused` and `sgd_rate_accepted`: the rate is refused
  with `InvalidLearningRate` exactly when `below_zero(lr)`. `sgd_rate_edges`:
  at `+0.0`, `-0.0`, `0.01`, a NaN of each sign and `+inf` it is accepted; at
  the smallest negative denormal, `-0.25`, the most negative finite word and
  `-inf` it is refused, as PyTorch 2.14.1 answers each.
- `sgd_step_refused`, `sgd_step_rate_refused` and `sgd_step_accepted`: a
  refused id leaves the tape unchanged; with every id accepted, a refused rate
  leaves it unchanged; with both accepted, the step updates the parameters in
  list order.

The autograd gate applies nine SGD mutants to a copy of the sources, and the
check of `src/autograd/PROOF.bend` fails at the named proof for each one: a
step that adds (`sgd_descend_scalar`), twice the learning rate
(`sgd_step_values`), a step that skips every parameter after the first
(`sgd_params_in_order`), an empty slot read as zeros and applied
(`sgd_empty_slot_kept`), an update of a neighboring entry
(`sgd_update_reaches_target`), updates before the ids are checked
(`sgd_step_refused`), a `Mul` node accepted as a parameter
(`sgd_only_leaves_accepted`), a step that skips the rate check
(`sgd_step_rate_refused`) and a rate check that also refuses `+0.0` and
`-0.0` (`sgd_negative_rate_refused`, whose proof stops checking although the
law stays true; `sgd_rate_accepted` and `sgd_rate_edges` are the laws it
makes false). The tensor backward subgate rejects
each of the nine as a runtime control too.

## Runtime lanes

`tools/autograd_tensor_backward.py` runs 181 scripts on native Bend CPU
(`--threads N --gpu off`, N from `tools.check --threads`, default 8, or the CPU count if lower) and on the generated JavaScript. Each script records
leaves and operations, runs `backward` on chosen outputs, sometimes calls
`zero_grad` or an SGD `step`, and reads the value and gradient of every node, word by word. The
harness (`src/autograd/tensor-backward-harness.bend`) calls the public
functions; `ForgeAdd` and `ForgeSum` push an entry without checks so the gate
can build tapes with dangling operands.

- 93 handwritten scripts: broadcasting along rows, columns, a `[1, 3]` operand,
  a rank-0 operand and rank 4; `x*x`, `x+x` and `x-x`; matmul, a square
  `A @ A` and a linear layer `x @ W + b`; a non-scalar output seeded with
  ones; the unused-node graph with each output selected; repeated backward on
  one and on two outputs; `zero_grad` with a `-0.0` contribution; the
  three-consumer fan-out and a reduction whose fold order changes the result;
  an empty `[0, 3]` operand; NaN, infinity, overflow and subnormals; unknown
  operands and outputs, incompatible shapes, matmul rank and inner errors, an
  invalid leaf; and three forged tapes with an operand that is the node
  itself, a newer node or a missing node. Eleven of them run Linear's product
  in PyTorch's layout. Nine call the public `linear`: one layer with a scalar
  and a seeded output, words that round, two layers in a row, an inner
  mismatch, three refusals after which the next node takes the id the product
  would have taken (a bias that does not broadcast, a bias id equal to the
  number of entries, and a larger unknown bias id), and `linear`, `relu`,
  `linear`. Two call `matmul_nt` alone: `x @ x^T` and a rank-0 input. Eight
  run `relu`: at the twenty F32 edge words (both zeros, denormals,
  infinities, signaling, quiet and all-ones NaNs of both signs) times weights
  that reach its gradient; NaN gradients of both signs, with payloads and a
  signaling NaN, that it must pass word for word; a seeded output;
  `relu(relu(x)) * x`; a rank-0 `-0.0`; an empty `[0, 3]` input; `linear`,
  `relu`, `linear`; and an unknown operand. Seven run `mse_loss`: same
  shapes, a broadcast target, a scale after the loss with a count whose
  `2 / N` rounds, equal words and both zeros, infinities and NaN, an empty
  input, and refusals (shapes that do not broadcast, an unknown id).
  Fourteen run `cross_entropy`: one batch, a scale after the loss over three rows,
  logits whose unshifted `exp` overflows (with the large logit at a target or
  not), `+inf` in a row, `-inf` at a target and a row of `-inf` only, a NaN
  logit, ties (equal logits, and `+0.0` against `-0.0`), one class, an empty
  batch, an empty batch of 1048577 columns, one and two classes times
  `g = +0.0`, `-0.0`, `1.0` and NaN, logits that also feed a `sum`, and
  refusals (a target equal to the columns, too few and too many targets,
  logits of rank 3, an unknown id).
  One runs Linear, relu, Linear and `cross_entropy`, and one Linear and
  `mse_loss`.
  Twelve run an SGD `step`: two training steps of `linear`, `relu`, `linear`
  (forward, backward, step, `zero_grad`, a second forward, backward and
  step); the twenty edge words,
  each parameter word paired with a gradient word of another class, at the
  learning rate `1/3`; products that round, cancel, underflow to denormals and
  overflow, at `0.01` and `1/3`; a parameter that backward never reaches and a
  step after `zero_grad`; a step before any backward; an empty list and an id
  listed twice; an unknown id; a node that is not a leaf; learning rates of
  `-0.25`, `-inf` and the smallest negative denormal, each refused; `-0.0`
  and `+0.0` on parameters of both zero signs; a NaN rate of each sign; and
  `+inf`, with a zero gradient word.
- 32 generated scripts (fixed seed) over shapes `[]`, `[2]`, `[3]`, `[1, 3]`,
  `[2, 1]`, `[2, 3]`, `[3, 2]` and `[2, 2]`, with words whose sums and products
  stay exact, and 24 more with words that round, cancel and overflow. 16 more
  record `relu` in three of eight steps, over words on both sides of zero,
  both zeros and a denormal of each sign. 16 more record `cross_entropy` with
  random targets, or `mse_loss` against a target of the same, a broadcast or
  a rank-0 shape, on `[1..6, 1..12]` words that round, sometimes scaled after
  the loss.

Both lanes must print exactly the lines of a Python reference that applies the
same rules in the same order and rounds to F32 after each operation. A NaN
word only needs to be NaN on both sides: the tensor operations and gradient
slots the engine builds on do not specify NaN payloads. `relu` computes no
word, it moves one, so the native lane holds its words exact, NaNs included:
at each read of a `relu` node, its value must be `relu` of its input's value
word for word, and, when the input feeds only that `relu` and no backward
starts at it, the input's gradient must be `relu_backward` of the input's
value and the `relu` node's gradient, both as the lane printed them, so every
passed gradient word is `g`'s own word. The JavaScript lane prints every NaN
as `0x7FC00000`, without its sign or payload, so it keeps the NaN allowance,
and only the native lane and the laws check the NaN words `relu` keeps and
passes.

The subgate also builds forty runtime controls from a mutated copy of the
sources and requires the JavaScript lane to reject each one: a seed at the
newest node, a `Mul` VJP with swapped operands, a `Sub` VJP without the sign,
a `Matmul` VJP without the transpose, a merge that overwrites the slot, a
reduction that folds when the shapes agree (which turns `-0.0` into `+0.0`),
`matmul_nt` reading `W` untransposed, a `MatmulNT` VJP whose `matmul_tn`
swaps its operands, `matmul_tn` reading `g` untransposed, a `linear` that
drops the bias, a `reduce_to` that walks the wrong stride, so the bias
gradient sums rows instead of columns, a `linear` that checks the bias id
after it records the product, a `linear` that keeps the product when the
sum is refused, a `relu` that passes negatives, a `relu` that zeroes
`-0.0`, a `relu` gradient that passes `g` at zero, a `Relu` VJP that
reads `g` as the input, a `mean` that divides by 1, an inverted `2 / N`, a
`cross_entropy` without the max shift, one-hot rows shifted by one column, a
`cross_entropy` mean that divides by 1, a one-hot gradient of `+0.0`
everywhere, a target gradient without the division by `N`, a `SoftmaxGrad`
with its operands swapped, and a `CrossEntropy` VJP that adds `+0.0` to the
row sum of one-class logits, the nine SGD mutants of the [SGD](#sgd) section,
the four broken sweeps of the [chain rule](#chain-rule) and a `zero_grad`
that keeps the slot. Two more controls run on the native lane only, since
the JavaScript lane cannot tell NaNs apart: a `relu` and a `relu` gradient
that replace each NaN word, of `x` or of `g`, with `0x7FC00000`. The
reference accepts both, and the exact `relu` words reject each.

## PyTorch record

`oracle/autograd/tensor_backward.py` replays every script with PyTorch
`2.14.1+cpu` (`OMP_NUM_THREADS=1`, `output.backward(torch.ones_like(output),
retain_graph=True)`, `retain_grad()` on every non-leaf) and writes
`oracle/autograd/fixtures/tensor-backward-torch.json`. The root test
`tests/test_autograd_tensor_backward.py` compares it with the Python
reference. Bend never calls PyTorch.

131 of the 181 scripts agree with PyTorch line for line: every id, shape, value,
gradient word and `None`, including the arrival-order and fold-order scripts,
`-0.0`, infinities and subnormals, and every `relu` script but four, with
PyTorch's `-0.0`, NaN and zero-gradient words. Those four are listed below:
`relu_edges` and `relu_nan_gradient` (NaN payloads of `NaN * NaN` in the
forward product after `relu`), `relu_13` (a one-term matmul, no `relu`
involved) and `relu_unknown_operand` (an intentional refusal), 16 of the
39 loss scripts, and five SGD scripts: the empty slot, no gradients, the
empty list with the duplicate id, the signed-zero rates and `+inf`. The
other 50 differ as listed; outside the loss scripts every
gradient PyTorch can represent in them agrees:

| Scripts | bendtorch | PyTorch |
| --- | --- | --- |
| `unknown_operands`, `unknown_output`, `unknown_output_empty`, `linear_bias_next`, `linear_bias_unknown`, `relu_unknown_operand`, `sgd_unknown_param` | `UnknownNode` (103) for each unknown id | `IndexError` from the Python list that the replay uses as a tape |
| `incompatible_shapes`, `invalid_leaf`, `linear_inner_mismatch`, `linear_bias_mismatch`, `matmul_nt_rank` | the tensor layer's error code (6, 105, 106, 10) | `RuntimeError` |
| `mse_refused`, `cross_entropy_refused` | 6, 107, 108 and 103 | `RuntimeError`, `IndexError` (a target out of range, an unknown id) and `ValueError` (a target count other than the rows) |
| `dangling_self`, `dangling_newer`, `dangling_missing` | `DanglingOperand` (104) | not representable |
| `specials`, `relu_edges`, `relu_nan_gradient`, `sgd_edges`, `sgd_rate_nan` | the value of `NaN * NaN`, and of a step with a NaN parameter, gradient or rate, is NaN | NaN with another sign or payload |
| `mse_empty`, `cross_entropy_infinities`, `cross_entropy_nan`, `cross_entropy_empty`, `cross_entropy_empty_wide`, `cross_entropy_one_class_scaled`, `cross_entropy_two_class_scaled` | NaN losses and gradients | NaN with another payload |
| `cross_entropy`, `cross_entropy_scaled` and 12 generated `loss_` scripts | 9 loss words up to 3 ulp, 21 gradient words up to 2 ulp and 3 gradients of a scale (a loss word) up to 2 ulp from PyTorch's | PyTorch's cascade sums and vectorized `exp` ([losses](losses.md#against-pytorch)) |
| `sgd_rate_negative` | `InvalidLearningRate` (110) for each rate below zero | `ValueError` ("Invalid learning rate") |
| `sgd_two_layers`, `sgd_rounding` | an SGD step rounds `lr * g`, then `p - lr * g` | one rounding on AVX2 and AVX512: the lines equal a reference that rounds once, word for word; 6 of 337 words differ from bendtorch's, by 1 to 6 ulp |
| `sgd_not_leaf` | `NotLeaf` (109) | `torch.optim.SGD` accepts a non-leaf that retains its gradient, as every non-leaf of the replay does, and updates it |
| `rounding_09` | the forward `sum` of six words adds them in ascending order | PyTorch's vectorized sum gives another word |
| `rounding_14`, `relu_13` | a matmul with inner dimension 1 whose product is `-0.0` gives `+0.0`, because the ordered dot starts from `+0.0` | `-0.0` |

The `rounding_09`, `rounding_14` and `relu_13` rows are forward values of the tensor layer
([reduction order](torch-differences.md)); the gradients of these scripts agree.

## Reproduce

```sh
bend src/autograd/PROOF.bend --verdict
uv run --locked python -m tools.autograd_tensor_backward --output .evidence/<new directory>
uv run --locked python -m unittest tests.test_autograd_tensor_backward -v
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/tensor_backward.py --check
```
