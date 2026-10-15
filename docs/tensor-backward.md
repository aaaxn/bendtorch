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
  F32.neg(l)`, which flips the sign bit like PyTorch's `neg`. `sum_to_size`
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
slot is part of the contract.

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

The laws state the engine's structure and the `Sum` VJP's values. The values
of the `Add`, `Sub`, `Mul` and `Matmul` VJPs are checked by the runtime lanes
and the PyTorch record, not by a law: a law would need the pointwise family to
export the replayed output of `Elementwise.binary` and of `sum_to_size`, as
`broadcast_to_replayed` does for `broadcast_to`, and an ordered-fold export for
`matmul`.

The autograd gate applies ten proof controls and two error controls to a copy
of the sources, and the proofs reject each one: `NegateLeft` keeping the sign
(at `negate_left_scalar`), seeding the newest node (`tensor_backward_success`),
seeding zeros and a head check that matches a larger id
(`tensor_seed_selected_output`), a slot update that overwrites
(`tensor_bump_target`), a merge that overwrites (`tensor_merge_accumulates`),
an unreached entry that drops the older gradients
(`tensor_unreached_sends_nothing`), a `Sum` VJP that sends to node 0
(`tensor_sum_vjp_pointwise`), a wiring check that skips the operand bounds and
a `zero_grad` that keeps the slot; and each refusal reported as the other.

## Runtime lanes

`tools/autograd_tensor_backward.py` runs 96 scripts on native Bend CPU
(`--threads N --gpu off`, N from `tools.check --threads`, default 8) and on the generated JavaScript. Each script records
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
