# Reduce to shape

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

1. Plan the broadcast of the target (left) against the contribution shape
   (right). An incompatible pair fails with `IncompatibleBroadcast` (code 6);
   the shape rank and capacity errors keep their codes.
2. Require the planned output to equal the contribution shape. Otherwise the
   target is larger than the contribution in some axis and the request fails
   with `NotReducible` (code 101): for example `[3]` reduced to `[2, 3]`, or
   `[1]` reduced to `[3]`.
3. Target position `t` holds the contribution words whose projection lands on
   `t`, added in ascending flat contribution position from `+0.0`, one F32 add
   each: the ordered fold `fiber(plan, values, t, n)` below.

The order is part of the contract: summing `[1e8, 1, -1e8, 1]` to a scalar gives
`1.0`, and the reversed list gives `0.0`. An empty contribution leaves every
target position at `+0.0`, so `[0, 3]` reduced to `[1, 3]` is three zeros, not
an empty tensor. Two `-0.0` words sum to `+0.0`, because the accumulation
starts from `+0.0`.

Cost: one allocation of the target shape, one clone of the contribution that
`Elementwise.take` makes even when nothing forks, one more whole-contribution
clone per fork, and per contribution position one unchecked read of a clone
and one add. No position is added twice: every position lands on exactly one
target cell (the clones copy every word, but they do not add).

## Parallel build

The target cells are independent, so `reduce_to` builds them in parallel, as
`Matrix.matmul` builds its cells (see
[sum and matmul](sum-and-matmul.md#parallel-product)). It takes a clone of the
contribution's array with `Elementwise.take` (the kept half goes back to the
caller) and builds the target as a balanced tree with `Reduce.build`. The top
`Reduce.splits` levels fork, each half with its own clone; below the forks a
thread builds its cells in index order.

`Reduce.splits` forks only when every fiber is one contiguous run of the
contribution: no axis the target keeps comes after an axis it reduces,
counting only axes of extent above 1. Row sums (`[n, m]` to `[n, 1]`, or
`[a, b, c]` to `[a, 1, 1]`) are contiguous and split by work like a product,
`Matrix.splits(depth, n / cells)`, so a target with fewer than `2^13`
contribution words does not fork. Column and batch sums (`[n, m]` to `[1, m]`,
`[b, n, m]` to `[n, m]`), which include the bias gradient of a linear layer,
have strided fibers and never fork: there each fork's whole-contribution clone
cost more than the threads gave back. Milliseconds per `reduce_to`, input
construction subtracted, median of 3 runs on a 64-core machine (main is the
scatter loop this build replaced; "all forked" is the product rule applied to
every target, "never forked" is `splits = 0`):

| contribution to target | threads | main | all forked | never forked | `Reduce.splits` |
| --- | --- | --- | --- | --- | --- |
| `[1024, 1024]` to `[1, 1024]` | 1 / 8 / 24 | 1270 / 1314 / 1294 | 107 / 310 / 306 | 88 / 153 / 147 | 93 / 147 / 145 |
| `[1024, 1024]` to `[1024, 1]` | 1 / 8 / 24 | 1195 / 1221 / 1154 | 45 / 19 / 6 | 50 / 84 / 53 | 33 / 20 / 2 |
| `[16, 256, 256]` to `[256, 256]` | 1 / 8 / 24 | 1558 / 1528 / 1562 | 211 / 454 / 544 | 181 / 289 / 298 | 176 / 281 / 300 |
| `[128, 784]` to `[1, 784]` | 1 / 8 / 24 | 106 / 114 / 110 | 9.7 / 29 / 29 | 9.1 / 16 / 15 | 8.6 / 16 / 15 |
| `[256, 128]` to `[1, 128]` | 1 / 8 / 24 | 29 / 33 / 32 | 3.1 / 5.3 / 8.6 | 3.1 / 5.1 / 5.1 | 3.0 / 5.0 / 5.1 |
| `[64, 128]` to `[1, 128]` | 1 / 8 / 24 | 7.7 / 7.4 / 7.2 | 0.8 / 1.4 / 1.3 | 0.8 / 1.3 / 1.3 | 0.8 / 1.4 / 1.3 |

The unforked build also gets slower from 1 to 8 threads; that is runtime
overhead outside the build (see issue #24). The row-sum timings at 8 and 24
threads are within the noise of the subtracted construction time (about 15 ms).

Cell `t` walks only its own fiber. Over the plan's index rows (one per
contribution axis, with the output stride and the left and right strides of
that axis), `Reduce.fiber` visits, on a row whose left stride is 0 (an axis the
target reduces, or a padded one), every digit `0 .. extent - 1` in increasing
order, and on any other row the one digit `t / left` that `t` names there,
passing `t % left` to the rows below. Each digit `d` moves the contribution
offset by `right * d`. The cell adds the words at the offsets it reaches in the
order it reaches them, from `+0.0`.

`Fiber.fiber_walk` in `src/autograd/reduce-fiber-proof.bend` proves the walk
visits exactly the fiber: for the index rows of an aligned broadcast and a cell
`t` below the left count, the offsets `Reduce.fiber` reads are, in order, the
offsets `base + walk_right(p)` of the positions `p = 0 .. count - 1` with
`walk_left(p) == t`, the positions `fiber` adds, in the order it adds them.
So each cell is word-identical to the ordered fold, whatever the split.

`Reduce.fiber_of` runs `Reduce.fiber_fast`, which reads the same offsets in
the same order. On a row whose rows below all take one digit
(`Reduce.single_below`: column sums, row sums and sums over one batch axis),
it walks the row's digits in one strided loop, `Reduce.strided`: digit `d`
reads the one word at `base + right * d + single_offset(rest, others, t %
left, 0)`, with no closure per word; on any other row it walks as `fiber`
does. `fast_fiber` in `reduce-parallel-proof.bend` and `fast_arr` in
`reduce-replayed-proof.bend` prove that `fiber_fast` returns what `fiber`
returns, so the laws below hold for it as they did for `fiber`. On the bench,
five alternating runs, `[1024, 1024]` to `[1024]` took 123 ms on one thread
and 311 ms on eight with the closure walk, and takes 3.0 ms and 3.25 ms now;
`[1024, 1024]` to `[1024, 1]` went from 58 to 12.4 ms on one thread and from
22.5 to 10.5 ms on eight, with the same words. Strided fibers still do not
fork: with the loop, forking `[1024, 1024]` to `[1024]` took 10.5 ms on eight
threads, the clone per fork costing more than the threads give back.

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
proves them for the parallel build. Each quantifies over every contribution tensor and target shape;
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

`reduce_mapping` comes from `reduce-parallel-proof.bend`: the cells of the
build hold the words their walk saw (`build_cells`), the walk reads the
masked positions (`plan_walk`, from `fiber_walk`), every one of them lies
inside the contribution (`masked_below`), and adding the listed words there is
`fiber` (`fiber_masked`).

The autograd gate checks five more proof controls in its source copy, and the
proofs must reject each: overwriting the total instead of adding, adding in the
reversed order (`word + total`), starting from `1.0`, walking a digit with the
target's stride instead of the contribution's, and skipping the target check.

## Sum over axes

`SumDims.sum_dims(input, axes, keepdim)` in `src/autograd/sum-dims.bend` is
PyTorch's `input.sum(dim=axes, keepdim=keepdim)`. It plans, then reuses
`reduce_to` and `reshape`:

1. Check the axes as PyTorch does: any axis at or past the rank (past 0 for a
   rank-0 input, whose only axis is 0) fails with `AxisOutOfBounds` (13), then
   an axis listed twice with `DuplicateAxis` (14), the codes `permute` gives
   (`Layout.bounded` and `Layout.unique`, `permute`'s own checks). So `[0, 0, 5]`
   on a rank-2 input is out of bounds, not a repeat. Axes are U32, so a
   negative axis cannot be given; the word of `-1` is out of bounds.
2. The kept dimensions are the input's with each listed axis set to 1; an
   empty list lists every axis, as PyTorch's `dim=[]`. The dropped dimensions
   are the input's without the listed axes.
3. `reduce_to(input, kept)`: each output cell is the ordered fold of its
   fiber from `+0.0`, built in parallel as above. With keepdim that is the
   result.
4. Without keepdim, the result is reshaped to the dropped dimensions. A
   reshape moves the storage to the new shape and copies no word.

The input owner comes back next to the result, on success and on every
refusal. The whole-tensor `sum` (`Reduction.sum`, no axes) is unchanged. A
`-0.0` alone in its fiber, such as a reduced axis of extent 1, sums to `+0.0`,
as PyTorch's sum over that axis does.

| input | axes, keepdim | result |
| --- | --- | --- |
| `[2,3]` = 1..6 | `[1]`, False | `[2]`: 6, 15 |
| `[2,3]` = 1..6 | `[0]`, True | `[1,3]`: 5, 7, 9 |
| `[2,3]` = 1..6 | `[]` or `[1, 0]`, False | `[]`: 21 |
| `[2,3]` = 1..6 | `[]`, True | `[1,1]`: 21 |
| `[2,3,2]` = 1..12 | `[0, 2]`, False | `[3]`: 18, 26, 34 |
| `[]` = -0.0 | `[0]`, False | `[]`: +0.0 |
| `[0,3]` | `[0]`, False | `[3]`: 0, 0, 0 |
| `[2,3]` | `[2]` or `[0, 0, 5]` | error 13 |
| `[2,3]` | `[0, 0]` | error 14 |

`src/autograd/sum-dims-LAWS.bend` states 17 laws, and `sum-dims-PROOF.bend`
proves them. `sum_dims_keepdim` says the keepdim sum is exactly
`reduce_to(input, kept)`, so every `reduce_to` law above gives its words and
its parallel build; `sum_dims_drops` says the sum without keepdim is the
keepdim sum reshaped to the dropped dimensions, whose words the reshape laws
keep; `sum_dims_refused` says a refused plan hands the input back with its
error. `sum_dims_plan_dimensions` says the plan's shape has the kept
dimensions and its list the dropped ones, and `sum_dims_plan_refused` that a
refusal fails the plan. The bookkeeping laws fix the rest:
`sum_dims_every_axis` and `sum_dims_listed_axis` (which axes are reduced),
`sum_dims_kept_start`, `sum_dims_kept_step`, `sum_dims_dropped_start`,
`sum_dims_dropped_reduced` and `sum_dims_dropped_kept` (axes count from 0, a
reduced axis becomes 1 or is left out), `sum_dims_rank_zero_limit` and
`sum_dims_rank_limit` (a rank-0 input has axis 0), and
`sum_dims_axis_out_of_bounds`, `sum_dims_duplicate_axis` and
`sum_dims_axes_accepted` (the refusals and their order).

The autograd gate applies seven proof controls and three error controls to a
copy of the sources, and the proofs reject each: keepdim that drops the axes
(`sum_dims_keepdim`), no keepdim that keeps them (`sum_dims_drops`), axes
counted from 1 (`sum_dims_kept_start`), the axis after each listed one
marked (`sum_dims_kept_step`), reduced and kept axes swapped in the dropped
dimensions (`sum_dims_dropped_reduced`), an empty list that reduces nothing
(`sum_dims_every_axis`), a rank-0 input with no axis
(`sum_dims_rank_zero_limit`), each refusal reported as the other, and a
refusal reported as `NotReducible`. On the tape, `TensorBackward.sum_dims`
records it and its VJP reads `g` with the kept shape before broadcasting
([tensor backward](tensor-backward.md)); the tensor backward gate compares
29 scripts of sums over axes with PyTorch.

## Runtime lanes

`tools/autograd_reduce.py` runs 540 requests on native Bend CPU
(`--threads N --gpu off`, N from `tools.check --threads`, default 8, or the CPU count if lower) and on the generated-Bend JavaScript executable.
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
