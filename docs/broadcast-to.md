# Materialized broadcast

`Broadcast.broadcast_to(input, target)` in `src/autograd/broadcast.bend`
returns an owned tensor with the requested shape. Each output position holds
the input value at its right-aligned broadcast projection. It creates no view
and no zero stride. The input owner is always returned unchanged next to the
result. `broadcast_to_dimensions(input, dims)` first builds `dims` with
`Shape.construct`.

## Semantics

1. Plan the broadcast of the input shape against the target
   (`Shape.broadcast_checked`). An incompatible pair fails with the shape error
   (`IncompatibleBroadcast`, code 6). A planned output above capacity fails
   with `CapacityExceeded` (code 2).
2. Require the planned output dimensions to equal the target. Otherwise the
   request fails with `NotBroadcastable` (code 100): for example `[3]` to
   `[1]`, or `[2, 3]` to `[3]`. The planner accepts both pairs, but their
   broadcast is larger than the target.
3. Allocate a zero tensor of the target shape and run the tensor elementwise
   parallel build (`Elementwise.parallel_allocated`, the one `add` runs) with
   the `Elementwise.KeepLeft` operation (`scalar(KeepLeft, l, r) = l`)
   against it. Each output word is the source word itself, with no
   arithmetic, so NaN payloads, signed zeros and subnormals are copied bit for
   bit, at every thread count and whatever split the build takes.

Cost: two allocations of the target shape (the zero right operand and the
output), plus one read of each operand per output position. An output above
2^15 cells forks into halves of 2^15 cells built on their own threads, each
with its own clone of both operands. A later change can drop the zero operand
without changing the laws, because they state the result and not the build.

Before the parallel build, `broadcast_to` ran the checked one-cell-at-a-time
elementwise loop (`Elementwise.allocated`) with the same operands: about
600 ns per output cell on one thread at `1bb7271` (625 ms for `[1024]` to
`[1024,1024]`), against about 65 ns per cell for `add` with a broadcast
operand. The output words are the same.

| input | target | result |
| --- | --- | --- |
| `[3]` = 1,2,3 | `[2,3]` | 1,2,3,1,2,3 |
| scalar 7 | `[2,2]` | 7,7,7,7 |
| `[2,1]` = 1,2 | `[2,3]` | 1,1,1,2,2,2 |
| `[1,3]` | `[0,3]` | empty `[0,3]` tensor |
| `[3]` | `[1]` | error 100 |
| `[2,3]` | `[3]` | error 100 |
| `[2]` | `[3]` | error 6 |

## Laws

`src/autograd/broadcast-LAWS.bend` states six laws, and `broadcast-PROOF.bend`
proves them by reusing the tensor elementwise retention, dimension, mapping and
success certificates. Each law quantifies over every input tensor and target
shape; none of them evaluates a concrete tensor.

- `keep_left_scalar`: `scalar(KeepLeft, left, right) == left` for all F32s.
- `broadcast_retains_input`: the returned owner has the same dimensions and the
  same checked and raw observations as the input, on success and on every
  error.
- `broadcast_mapping`: on success, the output dimensions equal the target, and
  every in-bounds output index observes the input at the offset that the
  independent projection specification `Projection.offset_spec` assigns to it.
- `broadcast_valid_success`: if a `Shape.Broadcast` witness shows that the
  right-aligned broadcast of the input extents with the target extents is the
  target itself, the call succeeds. The target's rank and capacity bounds come
  from the `Shape` value itself.
- `broadcast_shape_error`: if `Shape.broadcast_checked` rejects the input and
  target shapes with an error `e`, the result is exactly
  `(input, Fail{TensorError{ShapeError{e}}})`. For example `[2]` to `[0]` is
  `IncompatibleBroadcast`, never an empty tensor.
- `broadcast_not_target`: if the planner succeeds but the planned output
  dimensions are not the target's (for example `[3]` to `[1]`), the result is
  exactly `(input, Fail{NotBroadcastable})`.

`broadcast_mapping` alone says nothing about a failing call, and an empty
output satisfies it vacuously. The two error laws fix the result of every
request the planner or the target check rejects, so together with
`broadcast_valid_success` every request has a specified outcome.

### Live laws

The six laws above use erased evidence, so a later proof cannot take their
values as inputs. `src/autograd/broadcast-pointwise-LAWS.bend` states two live
laws for a [replayed](contract-families.md#replayed-tensors-are-the-input-class)
input, given the plan with `Shape.broadcast_checked(shape, target) == Done{plan}`
and `Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target)`,
the case where `broadcast_to` succeeds:

- `broadcast_to_pointwise`: the call succeeds, the output has the target's
  dimensions, every in-bounds output index `i` holds the input word at
  `left_map(plan, i)`, its right-aligned broadcast projection, and the
  returned input keeps every word. The fields of its `BroadcastToContract`
  are live.
- `broadcast_to_replayed`: the result is itself replayed, stated by value.
  For every `P`, if `P` holds for every result that hands back the input
  replayed with any history that keeps its words, next to an output of the
  plan's shape replayed over `+0.0` with any history whose words are `cells`
  (word `k` is the input word at `left_map(plan, k)`), then `P` holds for
  `broadcast_to`. A law about a replayed tensor, such as `reduce_to_fiber`,
  then applies to the output. This is the transport form `binary_replayed`
  states for the elementwise build.

`broadcast-pointwise-PROOF.bend` proves `broadcast_to_replayed` from the
build's own replayed lemma, `allocated_replayed` in
`src/tensor/contracts/binary-proof.bend` (the part of `binary_replayed`
below the planner): on two replayed inputs the build's output words are the
pointwise family's cells, here `scalar(KeepLeft, input word, 0.0)`, the
input word itself (`cells_keep`). `broadcast_to_pointwise` follows from it:
an output word is the word `cells` lists there, and the returned input has
the input's words.

Until the build, `broadcast_to_replayed` stated the loop's exact storage
history: the input with `ops ++ reads` and the output with the family
history `writes`. The build hands back the input as the first half of a
clone and cannot keep that history, so the law now states the words. Every
result the old law described has those words, so a run the old law covered
still satisfies it, and its one consumer, `broadcast_then_reduce`, proves
the same value.

The autograd gate also checks three proof controls. Each one changes the source
in the gate's copy, and the proofs must then fail: `KeepLeft` returning
`right`, `finish` returning the zero tensor as the owner, and skipping the
target check. The live laws have their own consumers, false twins and mutants
in [`verification/families/broadcast_to/`](../verification/families/README.md#broadcast-to).

## Runtime lanes

`tools/autograd_broadcast.py` runs 536 requests on native Bend CPU (`--threads
N --gpu off`, N from `tools.check --threads`, default 8, or the CPU count if lower) and on the generated-Bend JavaScript executable. There are 16
handwritten cases and every pair of an input shape with rank 0 to 2 and a
target with rank 0 to 3, using extents 0, 1 and 2. It checks the retained input
and each output word against a Python projection reference, bit for bit. The
JavaScript lane may canonicalize NaN payloads, the same exception that storage
makes. Inputs use distinct counting words, so a wrong projection gives a wrong
word. The handwritten cases include signed zero, subnormals, infinities and NaN
payloads.

```sh
uv run --locked python -m tools.check autograd --output .evidence/<fresh-autograd>
```
