# Gradient slots

`src/autograd/gradient.bend` holds the gradient of one tensor during reverse
differentiation. A `Slot` is `Unset{}` until it receives a contribution, then
`Held{gradient}`. The backward pass will keep one slot per tensor and feed it
the contributions of every use, after
[reduce to shape](reduce-to.md) has summed each contribution back to the
tensor's shape.

## Semantics

`Gradient.accumulate(shape, slot, contribution)` returns an `Outcome`:

1. If the contribution's dimensions differ from `shape`, the call returns
   `Refused{slot, contribution, ShapeMismatch}` (error code 102). The slot and
   the contribution come back unchanged. A broadcastable contribution is
   refused too: the caller reduces it first.
2. If the slot is `Unset`, the contribution moves in unchanged:
   `Accumulated{Held{contribution}}`. No copy is made, and a `-0.0` stays
   `-0.0`.
3. If the slot is `Held{gradient}` and the gradient's dimensions also agree
   with `shape`, the loop visits positions `0, 1, ..., n - 1` in ascending
   flat order. Each position reads the contribution word and the held total
   and writes `total + word` with one F32 add, the total first. The
   result is `Accumulated{Held{gradient}}`. If the held gradient has other
   dimensions, the call is refused with `ShapeMismatch` and nothing is
   written.

The contributions are summed in the order they arrive. For the scalar
contributions `1e8, 1, -1e8, 1` the gradient is `1.0`; in the reverse order it
is `0.0`. Contributions are added to the gradient one at a time, so a later
contribution never replaces an earlier one.

`Gradient.grad(slot)` reads a slot the way PyTorch's `.grad` does: an `Unset`
slot gives `None{}` and a `Held{g}` slot gives `Some{g}`. PyTorch's `.grad` is
`None` before the first backward and for a leaf that no backward reached, and
optimizers skip such parameters, so an absent gradient must not read as zeros.

`Gradient.reset(slot)` returns any slot to `Unset`, like PyTorch's
`zero_grad()` (whose default is `set_to_none=True`). The next contribution then
moves in unchanged, so a `-0.0` word stays `-0.0`. A reset that filled zeros
would compute `+0.0 + -0.0 = +0.0` on the next contribution instead.

`Gradient.gradient(shape, slot)` is the separately named zeros-materializing
read: an `Unset` slot gives zeros of `shape`, allocated at that moment, and a
`Held` slot gives its gradient. A shape that the shape limits reject fails
before any slot exists, with the shape rank (1) or capacity (2) error.

Cost: the first contribution costs nothing. Each later contribution costs, per
position, one checked read of the contribution, one checked read of the total
and one write. The gradient is updated in place, and no new tensor is
allocated.

## Laws

`src/autograd/gradient-LAWS.bend` states ten laws, and `gradient-PROOF.bend`
proves them. Each law quantifies over every shape, slot and contribution.

- `gradient_unset_zeros`: an `Unset` slot reads as a tensor with the slot's
  dimensions and `+0.0` at every in-bounds position.
- `gradient_held`: a `Held{g}` slot reads as `g`.
- `accumulate_mapping`: on success the slot holds a gradient with the slot's
  dimensions. At every in-bounds position it observes `combined(slot,
  contribution, i)`. For an `Unset` slot, that is the contribution's word.
  For a `Held` slot, it is `add_word(held[i], contribution[i])`, one F32 add
  with the held total first.
- `accumulate_refusal_keeps`: every refusal is `ShapeMismatch`. The returned
  contribution has the same dimensions and observations as the input, and the
  returned slot has the same state, dimensions and observations as the
  original slot.
- `accumulate_valid_success`: if the slot's gradient (when held) and the
  contribution both have the slot's dimensions, the call accumulates.
- `accumulate_wrong_shape_refused`: if the contribution's dimensions differ
  from the slot's, the outcome is exactly
  `Refused{slot, contribution, ShapeMismatch}`, even when the element counts
  agree (`[3, 2]` or `[6]` for a `[2, 3]` slot).
- `grad_unset_none`: `grad(Unset)` is `None{}`.
- `grad_held_some`: `grad(Held{g})` is `Some{g}`.
- `reset_unset`: `reset(slot)` is `Unset` for every slot.
- `reset_then_accumulate_moves_in`: after a reset, a contribution with the
  slot's dimensions gives `Accumulated{Held{contribution}}`: it moves in
  unchanged, whatever the slot held before.

The autograd gate also runs nine proof controls on a copy of the sources. The
proofs reject each of them:

- overwriting the total instead of adding
- adding in the reversed order
- dropping the first contribution
- skipping the contribution check
- skipping the held-gradient check
- reading an `Unset` slot as ones
- `grad` reading a `Held` slot as `None`
- `reset` keeping the held gradient
- refusing a wrong-shaped contribution with an error other than
  `ShapeMismatch`

## Runtime lanes

`tools/autograd_gradient.py` runs 203 requests on native Bend CPU
(`--threads N --gpu off`, N from `tools.check --threads`, default 8) and on the generated-Bend JavaScript executable.
Each request accumulates a list of contributions into a fresh slot. The harness
prints the outcome of every contribution, with the returned tensor for each
refusal, and then the slot's gradient. The requests are:

- 21 handwritten cases. They cover ordered cancellation in both orders,
  signed zero, NaN/Inf/subnormal words, refusals between accepted
  contributions, contributions with the same element count and other
  dimensions, rank 4, eight contributions in a row, invalid slot shapes and a
  `[32, 64]` slot.
- For every slot shape of rank 0 to 2 with extents 0, 1 and 2, the unset slot
  and the sequence (slot-shaped, other shape, slot-shaped) for every other
  shape in that set.

The lanes compare every outcome, every returned contribution and every
gradient word, bit for bit, with a Python reference that rounds to F32 after
each add. A NaN in a gradient only needs to be NaN on both sides, because
the payload after an add is not specified. A returned contribution must match
bit for bit in the native lane.

`tools/autograd_lifecycle.py` runs 9 more requests on both lanes. Each accumulates
a list of contributions into a fresh slot and prints `grad`, resets the slot
with `Gradient.reset` and prints `grad` again, accumulates a second list and
prints `grad`, and then prints the zeros-materializing `gradient`. The harness
prints the value that `Gradient.grad` returns and rebuilds the slot from it,
so the compiled lanes run the public accessor itself. Two runtime controls
compile the lifecycle program against a copy of `gradient.bend` where `grad`
reads a held slot as `None` or `reset` keeps the slot, and the subgate
requires the JavaScript lane to reject each one. A `grad` of an `Unset` slot must
print `S3:grad:none`, never a zero tensor. The cases cover a slot that never
received a contribution (rank 0, 1 and an empty slot), a reset of a held
gradient, a `-0.0` contribution after a reset, refusals that leave the slot
`Unset`, and invalid slot shapes.
