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
   loop with the new `Elementwise.KeepLeft` operation (`scalar(KeepLeft, l, r)
   = l`) against it. Each output word is the source word itself, with no
   arithmetic, so NaN payloads, signed zeros and subnormals are copied bit for
   bit.

Cost: two allocations of the target shape (the zero right operand and the
output), plus one checked read of each operand and one write per output
position. A later change can drop the zero operand without changing the laws,
because they state the result and not the loop.

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
- `broadcast_to_replayed`: the result is itself replayed. The returned input
  is the input with the history `ops ++ reads`, and the output is the replayed
  tensor over `+0.0` whose history `writes` is the pointwise family's history
  of `KeepLeft` against the zero tensor. A law about a replayed tensor, such as
  `reduce_to_fiber`, then applies to the output.

`broadcast-pointwise-PROOF.bend` proves both by walking the sequential
elementwise loop that `broadcast_to` runs, `Elementwise.allocated`, with live
code and without changing it. Step `k` reads the input at `left_map(plan, k)`
and the zero tensor at `right_map(plan, k)` and writes `scalar(KeepLeft, x, y)`
at `k`, which is the pointwise family's step with the broadcast maps. So the
output history is the family history `Pointwise.writes`, and each cell is
`scalar(KeepLeft, input word, 0.0)`, the input word itself. The six laws above
and `broadcast.bend` are unchanged.

The autograd gate also checks three proof controls. Each one changes the source
in the gate's copy, and the proofs must then fail: `KeepLeft` returning
`right`, `finish` returning the zero tensor as the owner, and skipping the
target check. The live laws have their own consumers, false twins and mutants
in [`verification/families/broadcast_to/`](../verification/families/README.md#broadcast-to).

## Runtime lanes

`tools/autograd_broadcast.py` runs 536 requests on native Bend CPU (`--threads
N --gpu off`, N from `tools.check --threads`, default 8) and on the generated-Bend JavaScript executable. There are 16
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
