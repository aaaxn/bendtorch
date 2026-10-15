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
`Matmul{...}` or `Sum{a, input}`: the operand ids and the operand shapes.
`value` is the forward value, `saved` holds the operand values a product needs
for its VJP (`Operands{left, right}` for `Mul` and `Matmul`, `Unsaved` for the
rest), and `grad` is the entry's gradient slot.

`TensorBackward.leaf(tape, value)` returns the new tape and the id.
`add`, `sub`, `mul` and `matmul` take two ids and `sum` one; each returns the
new tape and `Done{id}`. A recording reads copies of its operands' values
through the same lookup as `TensorBackward.value`, runs the tensor operation
(`Elementwise.add`, `sub` or `mul` with broadcasting, `Matrix.matmul`, or
`Reduction.sum_dynamic`) and pushes the result. An id the tape does not hold
gives `Fail{UnknownNode}` (103); an operation the tensor layer refuses gives
its error: `IncompatibleBroadcast` (6), a matmul rank mismatch (105) or inner
dimension mismatch (106). A refused recording hands the tape back unchanged.

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

  `left` and `right` are the saved operand values (and shapes). `-g` uses the
  new `Elementwise.NegateLeft` operation, `scalar(NegateLeft, l, r) =
  F32.neg(l)`, which flips the sign bit like PyTorch's `neg`. The `Mul`
  products and `-g` come from `TensorBackward.product`, which plans the
  broadcast and runs the tensor layer's sequential elementwise loop, the loop
  `broadcast_to` runs. `Elementwise.binary` builds the same words in parallel
  halves, but only erased laws describe that build, and the
  [value laws](#vjp-values) need the pointwise family's live statement.
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
`Matmul`. The sweep is sequential: the order in which contributions reach a
slot is part of the contract. The `Mul` and `Sub` products run on one thread,
also above the 2^15 cells where `Elementwise.binary` would fork.

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
[VJP values](#vjp-values) below state the `Add`, `Sub` and `Mul` VJPs' values.

The family controls in [`verification/families/tensor_vjp/`](../verification/families/README.md#tensor-vjp)
check the VJP value laws: literal consumers, five false twins and five
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

`src/autograd/tensor-vjp-LAWS.bend` states the values of the `Add`, `Sub` and
`Mul` VJPs, and `tensor-vjp-PROOF.bend` proves them through laws alone: the
pointwise family's product, then `sum_to_size`, each handing the next a
replayed tensor. Each law takes a [replayed](contract-families.md#replayed-tensors-are-the-input-class)
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

- `tensor_product_replayed` (`tensor-vjp-LAWS.bend`): `TensorBackward.product`
  of two replayed inputs whose shapes broadcast hands both inputs back with
  the same words and gives a replayed output of the plan's shape whose words
  are `cells`. Its proof is the broadcast instance of the pointwise family's
  walk of the elementwise loop in `broadcast-pointwise-PROOF.bend`.
- `sum_to_size_replayed` (`sum-to-size-fold-LAWS.bend`): `sum_to_size` of a
  replayed contribution whose words are `values` hands it back with the same
  words and gives a replayed output with the target's dimensions whose cells
  hold `summed`, the value `reduced` names. The output is the clone's second
  half or `reduce_to`'s output; the proof covers both.

The hypotheses name the plans the VJP computes, so a law says nothing about a
pair of shapes that does not broadcast. A recording on the tape only exists
for shapes that broadcast, and the consumers in
`verification/families/tensor_vjp/` instantiate every law on literal shapes.

The values of the `Matmul` VJP are not stated by a law. They are checked by
the runtime lanes and the PyTorch record. `g @ right^T` and `left^T @ g` read
the transposed saved operands, and `Transforms.transpose_dynamic` has only
erased laws: a live statement of its values needs the transpose copy loop
re-proved as live code, the follow-up that
[proof boundary](proof-boundary.md#live-transform-contracts-what-is-live-and-what-is-not)
describes. With that, `matmul_ordered` applies to the two products, since both
take replayed inputs.

## Runtime lanes

`tools/autograd_tensor_backward.py` runs 96 scripts on native Bend CPU
(`--threads 1 --gpu off`) and on the generated JavaScript. Each script records
leaves and operations, runs `backward` on chosen outputs, sometimes calls
`zero_grad`, and reads the value and gradient of every node, word by word. The
harness (`src/autograd/tensor-backward-harness.bend`) calls the public
functions; `ForgeAdd` and `ForgeSum` push an entry without checks so the gate
can build tapes with dangling operands.

- 40 handwritten scripts: broadcasting along rows, columns, a `[1, 3]` operand,
  a rank-0 operand and rank 4; `x*x`, `x+x` and `x-x`; matmul, a square
  `A @ A` and a linear layer `x @ W + b`; a non-scalar output seeded with
  ones; the unused-node graph with each output selected; repeated backward on
  one and on two outputs; `zero_grad` with a `-0.0` contribution; the
  three-consumer fan-out and a reduction whose fold order changes the result;
  an empty `[0, 3]` operand; NaN, infinity, overflow and subnormals; unknown
  operands and outputs, incompatible shapes, matmul rank and inner errors, an
  invalid leaf; and three forged tapes with an operand that is the node
  itself, a newer node or a missing node.
- 32 generated scripts (fixed seed) over shapes `[]`, `[2]`, `[3]`, `[1, 3]`,
  `[2, 1]`, `[2, 3]`, `[3, 2]` and `[2, 2]`, with words whose sums and products
  stay exact, and 24 more with words that round, cancel and overflow.

Both lanes must print exactly the lines of a Python reference that applies the
same rules in the same order and rounds to F32 after each operation. A NaN
word only needs to be NaN on both sides: the tensor operations and gradient
slots the engine builds on do not specify NaN payloads.

The subgate also builds six runtime controls from a mutated copy of
`tensor-backward.bend` and requires the JavaScript lane to reject each one: a
seed at the newest node, a `Mul` VJP with swapped operands, a `Sub` VJP
without the sign, a `Matmul` VJP without the transpose, a merge that
overwrites the slot, and a reduction that folds when the shapes agree (which
turns `-0.0` into `+0.0`).

## PyTorch record

`oracle/autograd/tensor_backward.py` replays every script with PyTorch
`2.14.1+cpu` (`OMP_NUM_THREADS=1`, `output.backward(torch.ones_like(output),
retain_graph=True)`, `retain_grad()` on every non-leaf) and writes
`oracle/autograd/fixtures/tensor-backward-torch.json`. The root test
`tests/test_autograd_tensor_backward.py` compares it with the Python
reference. Bend never calls PyTorch.

85 of the 96 scripts agree with PyTorch line for line: every id, shape, value,
gradient word and `None`, including the arrival-order and fold-order scripts,
`-0.0`, infinities and subnormals. The other 11 differ as listed, and every
gradient PyTorch can represent in them agrees:

| Scripts | bendtorch | PyTorch |
| --- | --- | --- |
| `unknown_operands`, `unknown_output`, `unknown_output_empty` | `UnknownNode` (103) for each unknown id | `IndexError` from the Python list that the replay uses as a tape |
| `incompatible_shapes`, `invalid_leaf` | the tensor layer's error code (6, 105, 106, 10) | `RuntimeError` |
| `dangling_self`, `dangling_newer`, `dangling_missing` | `DanglingOperand` (104) | not representable |
| `specials` | the value of `NaN * NaN` is NaN | NaN with another payload |
| `rounding_09` | the forward `sum` of six words adds them in ascending order | PyTorch's vectorized sum gives another word |
| `rounding_14` | a matmul with inner dimension 1 whose product is `-0.0` gives `+0.0`, because the ordered dot starts from `+0.0` | `-0.0` |

The last two are forward values of the tensor layer
([reduction order](torch-differences.md)); the gradients of both scripts agree.

## Reproduce

```sh
bend src/autograd/PROOF.bend --verdict
uv run --locked python -m tools.autograd_tensor_backward --output .evidence/<new directory>
uv run --locked python -m unittest tests.test_autograd_tensor_backward -v
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/tensor_backward.py --check
```
