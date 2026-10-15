# PyTorch-facing sum to size

`SumToSize.sum_to_size(input, target)` in `src/autograd/sum-to-size.bend`
reduces a broadcast contribution to an operand's shape the way PyTorch's
autograd engine does, so a PyTorch-facing backward pass should use it.
[`reduce_to`](reduce-to.md) stays unchanged as the ordered-fold
specification.

It matches PyTorch word for word only when nothing is reduced. When axes are
reduced, it uses `reduce_to`'s sequential fold, so it matches PyTorch only
within the rounding envelope, not word for word; see
[PyTorch parity](reduce-to.md#pytorch-parity). For example, with one thread,
PyTorch reduces `[1e8] + [1]*15 + [-1e8]` (shape `[17]` to `[]`) to `14.0`,
both directly and as the gradient of a broadcast addition, while the
sequential fold gives `+0.0`.

## Semantics

1. If the target's dimensions equal the contribution's dimensions, nothing is
   reduced. The output is a clone of the contribution, so every word is kept,
   `-0.0` and NaN payloads included. The call always succeeds.
2. Otherwise the result is exactly `Reduce.reduce_to(input, target)`: the same
   errors (`IncompatibleBroadcast` 6, `NotReducible` 101, shape rank and
   capacity), and on success the ordered fold from `+0.0`.

The contribution owner comes back next to the result in both cases.
`sum_to_size_dimensions(input, dims)` first builds `dims` with
`Shape.construct`.

`reduce_to` on equal dimensions computes `+0.0 + w` at every position, which
turns `-0.0` into `+0.0`. PyTorch does not reduce in that case. Observed with
PyTorch `2.14.1+cpu` (one thread), for a contribution of `-0.0` words:

| contribution | target | PyTorch | `reduce_to` | `sum_to_size` |
| --- | --- | --- | --- | --- |
| `[1]` | `[1]` | `80000000` | `00000000` | `80000000` |
| `[2,3]` | `[2,3]` | `80000000` ×6 | `00000000` ×6 | `80000000` ×6 |
| `[1,3]` | `[1,3]` | `80000000` ×3 | `00000000` ×3 | `80000000` ×3 |
| `[1,3]` | `[3]` | `00000000` ×3 | `00000000` ×3 | `00000000` ×3 |
| `[1]` | `[]` | `00000000` | `00000000` | `00000000` |

PyTorch's gradient of a same-shape operand also keeps `-0.0`: for `a + b`
with `a`, `b` of shape `[3]` and output gradient `-0.0`, both `a.grad` and
`b.grad` are `80000000`. For `a` of shape `[3]` broadcast against `b` of
shape `[2,3]`, `a.grad` is `00000000`. For a scalar `[]` contribution and
target `[]`, `sum_to_size` keeps `-0.0`, as PyTorch's autograd does for
`a + b` with scalar operands. A direct `torch.Tensor.sum_to_size(())` call on
a scalar returns `+0.0`, because that call reduces even when nothing needs to
be reduced; the autograd engine does not reduce there, and `sum_to_size`
follows the engine.

## Laws

`src/autograd/sum-to-size-LAWS.bend` states three laws, and
`sum-to-size-PROOF.bend` proves them. Each quantifies over every contribution
tensor and target shape.

- `sum_to_size_retains_input`: the returned owner has the same dimensions and
  checked and raw observations as the contribution, on success and on every
  error. It reuses `reduce_to`'s `Retained` statement.
- `sum_to_size_identity`: if the contribution's dimensions equal the target's,
  the call succeeds, the output has the target's dimensions, and every
  position's checked and raw observation equals the contribution's.
- `sum_to_size_reduces`: if the dimensions differ, the result equals
  `reduce_to(input, target)`, so every `reduce_to` law applies to it.

The autograd gate checks three proof controls, and the proofs must reject each:
folding the identity case through `reduce_to`, skipping the dimension check
(always cloning), and always folding.

### Live laws

The three laws above use erased evidence. `src/autograd/sum-to-size-fold-LAWS.bend`
states `sum_to_size` on a [replayed](contract-families.md#replayed-tensors-are-the-input-class)
contribution through `ReduceToContract`, the live record of
[`reduce_to_fiber`](contract-families.md#reduce_to-meets-the-same-specification),
so one record covers both cases:

- `sum_to_size_clone`: if the contribution's dimensions equal the target's,
  the call succeeds, the output has the target's dimensions, every in-bounds
  cell holds the contribution's own word, so `-0.0` stays `-0.0`, and the
  returned contribution keeps every word.
- `sum_to_size_fiber`: if the dimensions differ and the target broadcasts to
  exactly the contribution's dimensions, every target cell holds
  `ReduceLaws.fiber` over the contribution's words in index order from `+0.0`,
  the value that `reduce_to_fiber` states for `reduce_to`.
- `sum_to_size_replayed`: both cases in the transport form of
  `broadcast_to_replayed`. If the contribution's words are `values` and the
  target broadcasts to exactly its dimensions, `P` holds for the result when
  it holds for every replayed contribution with the same words next to a
  replayed output with the target's dimensions whose cells hold `summed`:
  `values[index]` when nothing is reduced, the fiber sum otherwise. The tensor
  backward's [VJP value laws](tensor-backward.md#vjp-values) chain it.

`sum-to-size-fold-PROOF.bend` proves the first by cloning the replayed
contribution. Cloning it gives two replayed tensors, the kept owner with the
history `ops ++ [First]` and the copy with `ops ++ [Second]`, and neither
operation changes a word. It proves the second from `reduce_to_fiber`, once the
dimension check is false. Their consumers, false twins and mutants are in
[`verification/families/reduce/`](../verification/families/README.md#reduce).

## Runtime lanes

`tools/autograd_sum_to_size.py` runs 8 handwritten cases plus every `reduce_to`
case on native Bend CPU (`--threads N --gpu off`, N from `tools.check --threads`, default 8) and on the generated-Bend
JavaScript executable. Identity cases must keep every word exactly on the
native lane, NaN payloads included; the JavaScript lane only needs NaN to stay
NaN. The other cases use `reduce_to`'s Python reference.
