# S3.B reduce to shape

`Reduce.reduce_to(input, target)` in `src/autograd/reduce.bend` sums a
broadcast contribution back to a target shape. It is the adjoint of
[materialized broadcast](broadcast-to.md): if `x` of shape `target` was
broadcast to `input`'s shape, `reduce_to` returns a tensor of shape `target`
whose position `t` is the ordered sum of the contribution words whose
broadcast projection lands on `t`. The input owner always comes back unchanged
next to the result. `reduce_to_dimensions(input, dims)` first builds `dims`
with `Shape.construct`.

A PyTorch-facing backward pass should use [`sum_to_size`](sum-to-size.md). It keeps the
contribution's words when no dimension needs reduction, as PyTorch does, and
is `reduce_to` otherwise.

## Semantics

1. Plan the S1 broadcast of the target (left) against the contribution shape
   (right). An incompatible pair fails with `IncompatibleBroadcast` (code 6);
   the S1 rank and capacity errors keep their codes.
2. Require the planned output to equal the contribution shape. Otherwise the
   target is larger than the contribution in some axis and the request fails
   with `NotReducible` (code 101): for example `[3]` reduced to `[2, 3]`, or
   `[1]` reduced to `[3]`.
3. Allocate `+0.0` at every target position, then visit contribution positions
   `0, 1, ..., n - 1` in ascending flat order. Each position reads its word and
   the current total at its projected target position, then writes
   `total + word` with one F32 add. Later contributions never overwrite earlier
   ones.

The order is part of the contract: summing `[1e8, 1, -1e8, 1]` to a scalar gives
`1.0`, and the reversed list gives `0.0`. An empty contribution leaves every
target position at `+0.0`, so `[0, 3]` reduced to `[1, 3]` is three zeros, not
an empty tensor. Two `-0.0` words sum to `+0.0`, because the accumulation
starts from `+0.0`.

Cost: one allocation of the target shape, and per contribution position one
checked read of the contribution, one checked read of the total and one write.

| contribution | target | result |
| --- | --- | --- |
| `[2,3]` = 1..6 | `[3]` | 5, 7, 9 |
| `[2,3]` = 1..6 | `[2,1]` | 6, 15 |
| `[2,3]` = 1..6 | `[]` | 21 |
| `[0,3]` | `[1,3]` | 0, 0, 0 |
| `[3]` | `[2,3]` | error 101 |
| `[3]` | `[2]` | error 6 |

## Laws

`src/autograd/reduce-LAWS.bend` states six laws, and `reduce-PROOF.bend`
proves them. Each quantifies over every contribution tensor and target shape;
none evaluates a concrete tensor.

- `reduce_retains_input`: the returned owner has the same dimensions and the
  same checked and raw observations as the contribution, on success and on
  every error.
- `reduce_mapping`: on success the output dimensions equal the target. For
  every word list `values` that lists the contribution (`Listed`: the checked
  observation at every U32 position equals the word at that position), every
  in-bounds target position observes `fiber(plan, values, t, n)`. `fiber` is
  the specification: start at `Some{+0.0}`, and for positions `0 .. n - 1` in
  order, add the word of each position whose independent projection
  `Projection.offset_spec` (target as the left operand) lands on `t`.
- `reduce_valid_success`: if a `Shape.Broadcast` witness shows that the
  broadcast of the target with the contribution is the contribution shape
  itself, the call succeeds.
- `reduce_empty_zeros`: if the contribution has no elements, every in-bounds
  target position of a successful result observes `+0.0`.
- `reduce_shape_error`: if `Shape.broadcast_checked(target, contribution)`
  rejects the shapes with an error `e`, the result is exactly
  `(contribution, Fail{TensorError{ShapeError{e}}})`. For example `[2]` to
  `[0]` is `IncompatibleBroadcast`, never an empty tensor.
- `reduce_not_reducible`: if the planner succeeds but the planned output is not
  the contribution's shape (for example a `[2, 3]` target for a `[3]`
  contribution), the result is exactly `(contribution, Fail{NotReducible})`.

The S3 gate checks four more proof controls in its source copy, and the proofs
must reject each: overwriting the total instead of adding, adding in the
reversed order (`word + total`), starting from `1.0`, and skipping the target
check.

## Runtime lanes

`tools/autograd_reduce.py` runs 540 requests on native Bend CPU
(`--threads 1 --gpu off`) and on the generated-Bend JavaScript executable.
There are 20 handwritten cases, plus every pair of a contribution with rank 0
to 3 and a target with rank 0 to 2, using extents 0, 1 and 2. It compares the
retained contribution and every output word, bit for bit, with a Python
reference that accumulates in the same specified order with F32 rounding after
each add, including finite overflow to a signed infinity (the sum of two
largest finite words is `+Inf`). NaN outputs only need to be NaN on both
sides, because their payload after an add is not specified; the native and JS lanes agree. The
`wide_rows` case uses a `[32, 64]` contribution: a 4096-word constructor
literal overflows the JavaScript call stack, the same limit that
`docs/runtime-corpus.md` records.

## PyTorch parity

`reduce_to` matches `torch.Tensor.sum_to_size` and the gradient of `expand`
in shape, refusals and fiber mapping. Its values are word-identical to
PyTorch only while the sequential fold and PyTorch's vectorized order agree,
which holds for short fibers: 8-word probes agreed, and 16-word probes can
already differ. Beyond that, parity with PyTorch is envelope-only; see
[differences from PyTorch](torch-differences.md). The ordered fold above stays
the specification.
