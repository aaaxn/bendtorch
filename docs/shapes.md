# S1 shapes, indexing, and broadcasting

The implementation is `src/shape/core.bend`. A `Shape` is reusable `Data`: its
dimensions are a reusable list of `U32`, and its natural-number certificates
establish rank at most four and logical element count at most `2^20`.
`construct(dimensions)` returns `SBuilt{shape, preserved}` or `SError{error}`.
The success witness preserves the exact input dimensions. `dimensions(shape)`,
`rank(shape)`, and `count(shape)` expose dimensions, rank, and logical numel;
the latter two are U32 values suitable for the bounded storage layer.

## Construction and numerical boundaries

Scalar shape `[]` has one element. Any zero dimension makes the logical count
zero, including `[4294967295,4294967295,4294967295,0]`. `logical_numel` is
`numel` of the dimensions read as naturals, so a zero anywhere gives zero.

Rank is checked first. The capacity decision `capacity` then walks the
dimensions from the right on naturals. A size is formed only when the suffix
product is zero or both factors are at most `2^20`. The largest product ever
computed is therefore `2^40`, below the native `Nat` limit of `2^48`. A head
above `2^20` with a nonzero suffix, or a product above `2^20`, yields `Beyond`.
A zero head after `Beyond` restores size zero, so a zero at the end does not
reject a huge prefix. Each `Within` result carries `size == numel(dims)`, and
`construct` turns `size <= 2^20` into the `LE` certificate stored in the shape.
A wrong decision cannot build a `Shape`. The `LE` and `{a == b}` certificates
come from one `Nat.is_le` or `Nat.is_eq` comparison through an erased rewrite,
not from a unary walk.

`construction_complete` proves the converse: every dimension list of rank at
most four and `logical_numel` at most `2^20` is built, with its dimensions
unchanged. An implementation that rejects any such shape fails this law.

Safety-critical branch parameters are `BoolCase` witnesses of their actual
computed predicates, not freely supplied booleans. This covers the rank
decision, the capacity decisions, coordinate and flat-index bounds, and the
broadcast axis decisions. A checked Bend caller cannot force the constructor
past an oversized product by supplying `True`, since that would require proving
`True == False`.

The input type is U32, so negative integers and integers above `2^32-1` are not
Bend dimensions or coordinates. The gate's request encoder rejects them before
they reach Bend, rather than wrapping them. These transport errors are distinct
from the mathematical API's errors:

| Code | Constructor | Meaning |
| --- | --- | --- |
| 1 | `RankExceeded` | The shape rank exceeds four. |
| 2 | `CapacityExceeded` | Positive logical numel exceeds `2^20`. |
| 3 | `CoordinateRankMismatch` | Coordinate rank differs from shape rank. |
| 4 | `CoordinateOutOfBounds` | A coordinate is not below its dimension. |
| 5 | `FlatIndexOutOfBounds` | A flat index is not below logical numel. |
| 6 | `IncompatibleBroadcast` | Aligned dimensions are neither equal nor singleton. |

## Indexed core and checked boundary

`Fin(n)` is an indexed family: `Fin(0)` is empty; positive bounds permit zero or
a successor of a smaller finite index. `Coordinate(dimensions)` recursively
contains one such finite index per axis. It is reusable Data, including the
rank-zero coordinate `Unit`. Empty shapes have no coordinate or flat index.

`flatten` and `unflatten` convert between these coordinates and
`Fin(numel(dimensions))`. They use constructive mixed-radix packing and splitting,
not an assumed arithmetic inverse. `row_major` independently states the usual
natural arithmetic interpretation. `index_proofs.bend` proves packing/splitting
inverses and arithmetic interpretation by induction, and `broadcast_proofs.bend`
proves the projection laws. Six of the laws in `LAWS.bend` concern this core:
the row-major offset of every coordinate is below numel, flatten and unflatten
are mutually inverse, flatten computes the row-major offset, and each broadcast
projection maps an output coordinate to its right-aligned suffix with every
singleton input axis set to zero. That rule is `project_indices`, the same rule
the gate's Python oracle applies. Broadcast certificates left-pad with singleton
dimensions and require at least one padding count to be zero, enforcing right
alignment. Axis certificates admit equality or an actual singleton, including
`0/1 -> 0`.

The public boundary is `construct`, `flatten_checked`, `unflatten_checked`,
`broadcast_checked`, `project_left_checked`, `project_right_checked`, and the
flat projections `project_left_index` and `project_right_index` in
`src/shape/flat.bend`. Each reads its U32 inputs, if any, as naturals with
`U32.to_nat`, checks bounds on naturals, computes on naturals, and returns
results through `U32.from_nat`. The checks and arithmetic walk the shape's U32
dimensions directly, without building natural lists: flatten is one Horner pass
(`horner`), unflatten one quotient and remainder per axis (`digits_words`), and
a bounds check is one boolean per axis (`fit_words`). Each of these has a proved
equality with its natural-list specification (`offset`, `digits`, `fit`).
`CheckedCoordinate<shape>` stores the natural coordinate and a proof that
`fit(extents, values)` is `True`, which gives every axis `value < extent`.
`FlatIndex<shape>` stores the natural index and its bound below the shape's
element count.
`coordinate_values(shape, coordinate)` and `index_value(shape, index)` extract
U32 data; their shape arguments are erased in compiled JavaScript. Coordinate
rank errors take precedence over coordinate bounds errors. Unflatten rejects
flat indices before division, so an empty shape never divides by zero.

Laws that mention the executable functions directly relate them to the proved
core. `decode(extents, values)` is the specification of a valid coordinate. It
returns the `Coordinate` whose indices are `values`, or `None` when a rank or
bound fails:

- `flatten_checked_refines`: `flatten_checked` returns
  `U32.from_nat(row_major(extents, c))` exactly when the input decodes to `c`,
  and fails otherwise.
- `unflatten_checked_refines`: `unflatten_checked` returns
  `words(indices(extents, unflatten(extents, i)))` exactly when the index is a
  `Fin(numel(extents))` value `i`, and fails otherwise.
- `project_left_checked_refines` and `project_right_checked_refines`: a checked
  projection returns the `project_left` or `project_right` image of the decoded
  output coordinate, using the plan's own `Broadcast` certificate. It fails
  exactly when the coordinate does not decode.
- `project_left_index_refines` and `project_right_index_refines`: for every
  flat output index `i` of a plan and every index map of that plan, the flat
  projection returns `row_major` of the `project_left` or `project_right` image
  of `unflatten(output, i)`, again under the plan's own certificate.

With `flatten_unflatten` and `unflatten_flatten`, these laws give the round
trip on naturals. The proofs use the divmod identity
`divmod(q*b + r, b) == (q, r)` for `r < b`, proved in `nat_proofs.bend`. The
views treat every failure alike. Two more laws fix the error of a failed
flatten: `flatten_rank_error` says that a coordinate whose rank differs from
the shape's fails with exactly `CoordinateRankMismatch`, checked before any
bound (`[9]` on `[2, 3]`), and `flatten_bounds_error` says that a coordinate of
the right rank with a component outside its extent fails with exactly
`CoordinateOutOfBounds`. The other error codes are checked empirically, not
proved. The proofs end at the natural view. They assume that
Base's `U32.to_nat` and `U32.from_nat` are the standard conversions, which
native and JavaScript lowering implement as casts, and that `U32.from_nat` is
exact below `2^32`. Every value it receives is below `2^32`: the shape
certificates bound flat indices by `2^20`, and coordinates are below a U32
extent.

## Broadcasting API

`broadcast_checked(left, right)` returns a checked `BroadcastPlan` or an error.
`left_shape`, `right_shape`, and `output_shape` expose the plan's shapes. Output
construction enforces the same rank and capacity limits, even when both inputs
are individually within capacity. Dimensions are aligned from the right;
equal dimensions survive, a singleton yields to the other dimension, and any
other pair is incompatible. Using `max(0,1)` would be wrong: `[0,3]` with `[1,3]`
produces `[0,3]`, not `[1,3]`.

`broadcast_checked` decides each axis with natural comparisons and builds a Nat
`Broadcast` certificate for the extents of the left, right, and output shapes.
`BroadcastPlan` stores it, and `alignment(plan)` exposes it. Three laws pin the
plan down. `broadcast_checked_plan` says a plan's left and right shapes have
the input dimensions. `broadcast_output_unique` says two `Broadcast` certificates
for the same inputs have the same output, so the certificate is the
right-aligned rule. `broadcast_checked_complete` says that whenever such an
output exists within rank four and `2^20` elements, `broadcast_checked` returns
a plan.

`project_left_checked(plan, coordinates)` and `project_right_checked` validate
the output coordinate, compute `project_indices` on naturals (drop left padding,
map singleton axes to zero), and return checked coordinates of the
corresponding input. A zero-sized output has no valid projection input.

Elementwise kernels need one projection per output element, so the plan also
has a flat path that never materializes a coordinate. `index_map(plan)` builds
an `Index` once per plan: one row per output axis holding the output stride and
the left and right input strides, with stride zero on padded and singleton
input axes. Its `exact` field proves the rows equal `index_table` of the plan's
extents. `project_left_index(plan, map, i)` and `project_right_index` take a
checked output `FlatIndex` and return a checked input `FlatIndex` by one pass
over the rows, adding `input_stride * (i / output_stride)` and continuing with
`i % output_stride`. An empty output gets `Unindexed`, which carries
`numel(output) == 0`, so no flat index reaches it. The proof that the result is
below the input's element count is erased at runtime. A wrong stride makes
`project_left_index_refines` false: with zero replaced by the extent product
on a singleton axis, `[1]` broadcast against `[3]` sends output index 1 to
input index 1 instead of 0, and `--verdict` rejects the proofs.

## Reproduction and evidence

The shape stage runs inside the S1 gate and can also run alone:

```sh
uv run --locked python -m tools.shape --output .evidence/s1-shapes
uv run --locked python -m unittest tests.test_shapes -v
```

Each runner output directory must be new. It contains both checker transcripts,
the exported BendTT certificate, the request stream, the native binary and JS
program built from `src/shape/harness.bend`, each lane's build and run
transcripts, exact coverage counts, and the command/hash report. The harness
reads the request file named on its command line, calls only the public API
above, and prints one line per observation; the gate compares every line of
each lane with the Python oracle. The harness parses and prints; it does not
implement shape, flatten, unflatten, broadcast, or projection arithmetic. No
root Python code imports PyTorch or NumPy.

The exhaustive domain contains 237 shapes: all 156 of rank zero through three
with dimensions zero through four, and all 81 of rank four with dimensions zero
through two. Each shape is checked at every valid coordinate and flat index plus
boundary and wrong-rank coordinates, and all 56,169 ordered shape pairs are
broadcast. Every valid broadcast output coordinate, and every valid output flat
index, is projected to both inputs. Invalid ranks, coordinates, indices, and
incompatible pairs are also checked.
Separate cases cover capacity-edge indices, products that would overflow U32,
huge prefixes followed by zero, and output capacity overflow. Negative and
above-`2^32-1` values cannot be encoded as U32 words, so the gate checks that
the encoder rejects them instead of running them. Storage ownership and the root
S1 gate inventory are outside this module.
