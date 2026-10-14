# Contract families

Many tensor operations share one loop. `add`, `sub` and `mul`, with or
without broadcasting, all write `out[i] = f(a[ia(i)], b[ib(i)])` for every
output index `i`. A contract family proves that loop once, as a generic
theorem over `f`, `ia` and `ib`, and each operation gets its contract by
instantiating it. Reductions share a second loop: `sum`, the matmul dot
product and one cell of `reduce_to` fold words into an accumulator in a fixed
order. The code is in `src/tensor/contracts/`, and the `reduce_to` instance is
in `src/autograd/reduce-fold.bend`.

## Replayed tensors are the input class

The family theorems do not name a producer such as `Core.new`. They take any
replayed tensor: the storage that `Storage.new(count, fill)` creates, changed
by a list of storage operations applied in order.

- `Replayed.storage(shape, fill, ops)` is that storage,
  `StorageLaws.replay(ops, StorageLaws.created(Shape.count(shape), fill))`.
- `Replayed.value(shape, fill, ops, index)` is the word it holds at `index`,
  `StorageLaws.expected(ops, count, index, fill)`.

Three laws in `src/tensor/contracts/LAWS.bend` connect the public producers to
this class:

| Law | Statement |
| --- | --- |
| `new_replayed` | `Core.new(shape, fill)` is the replayed tensor with an empty history. |
| `from_values_replayed` | `Core.from_values(shape, values)` is the replayed tensor whose history writes each value once, in order, over a `0.0` fill. It needs `List.length(values) == numel`. |
| `replayed_read` | `read_dynamic` on a replayed tensor at an in-bounds index returns `Some(Replayed.value(...))`. |

`new_replayed` and `from_values_replayed` are stated as transports: for every
`P`, if `P` holds for the replayed tensor with any length proof `fits`, then
`P` holds for the producer's result. The length proof inside a tensor is an
erased field, and Bend2 does not identify two values that differ only in an
erased field. An equation between the two tensors would therefore need the
same proof term on both sides. The transport form avoids that.

## The pointwise family

`Pointwise.run` in `pointwise.bend` takes five templates: an operation type
`O`, the scalar function `f: O -> F32 -> F32 -> F32`, a plan type `P`, and two
index maps `ia, ib: P -> Nat -> Nat`. It allocates a `+0.0` output of `shape`
and, for `i` from 0 to `numel - 1` in increasing order, reads
`left[ia(plan, i)]` and `right[ib(plan, i)]` and writes
`f(operation, x, y)` at `i`. Both inputs come back to the caller, on success
and on failure.

`PointwiseProof.contract` in `pointwise-proof.bend` is the family theorem.
Bend2 checks a template body once with its template parameters abstract, so
the theorem holds for every `f`, `ia` and `ib`. Its hypotheses:

- Both inputs are replayed tensors of their shapes, with any `fill` and any
  history `ops`.
- `Pointwise.inside(ia, plan, numel(shape), numel(left)) == True`: every
  output index maps into the left input. The same holds for `ib` and the right
  input.

Its conclusion is a `PointwiseContract` (in `pointwise-records.bend`), whose
fields are all live:

- `successful`: the result is `Done`.
- `dimensions`: the output has `Shape.dimensions(shape)`.
- `value`: at every in-bounds index `i`, the output holds
  `f(operation, left_word(ia(plan, i)), right_word(ib(plan, i)))`, where
  `left_word` and `right_word` are the inputs' replayed values.
- `left_kept` and `right_kept`: each returned input still holds every word it
  held before.

The proof follows the loop. After `i` steps the left input is the replayed
tensor with history `left_ops ++ reads(ia, i)`, the right input likewise, and
the output is the replayed tensor over `+0.0` with history `writes(i)`, the
first `i` cell writes. Reads do not change a word, and a cell is written once,
at its own index, so the final history gives every value.

The index maps return `Nat` with a separate Bool bound, not a bounded index
type. A template parameter whose return type carries an equality field, such
as `Nat -> Shape.Shape`, has no BendTT model on the current pin, so
`--verdict` rejects it where `--check-only` accepts it. `Nat -> Nat` with a
checked bound states the same requirement and passes both checkers.

### Each instance is a law

A law cannot quantify over a template, so the family theorem is not a law by
itself. Each instance states its own law and proves it by instantiating the
theorem.

`Broadcast.binary(operation, plan, left, right)` in `broadcast.bend` is the
first instance. It runs the family with `Elementwise.scalar` and the
broadcast projections of the plan, `Shape.walk_left` and `Shape.walk_right`
over `Shape.index_map(plan)`. `broadcast-proof.bend` discharges the bound
hypotheses from `Flat.left_bound` and `Flat.right_bound`. The law
`broadcast_pointwise` states the contract for every operation, every
broadcast plan and every pair of replayed inputs of the plan's shapes:
output cell `i` holds
`Elementwise.scalar(operation, left[left_map(plan, i)], right[right_map(plan, i)])`.

## The ordered-fold family

`Fold.run` in `fold.bend` takes five templates: an operation type `O`, the
step function `g: O -> F32 -> F32 -> F32`, a plan type `P`, an index map
`ia: P -> Nat -> Nat`, and a mask `keep: P -> Nat -> Bool`. It starts at
`+0.0` and, for `k` from 0 to `count - 1` in increasing order, reads
`input[ia(plan, k)]` and, where `keep(plan, k)` holds, sets
`acc = g(operation, acc, word)`. Where the mask is false the word is still
read, and the accumulator stays as it is. `Fold.pair_run` reads
`left[ia(plan, k)]` and `right[ib(plan, k)]`, takes
`g: O -> F32 -> F32 -> F32 -> F32`, and folds every pair. The inputs come back
to the caller, on success and on failure.

The order is part of the contract. F32 addition is not associative, so the
same words summed in another order can round to another value. The checker and
the kernel keep F32 operations symbolic, so the stated total is the exact term
that the loop builds. For the words `[1, 2, 3]` it is
`F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)`, and a fold in another order,
or from another start, states a different term.

`FoldProof.contract` and `FoldProof.pair_contract` in `fold-proof.bend` are the
family theorems. Their hypotheses match the pointwise ones: replayed inputs,
and `Pointwise.inside(ia, plan, count, numel(input)) == True` for each index
map. The conclusion is a `FoldContract` (in `fold-records.bend`), whose fields
are live:

- `value`: the result is `Done` with `Fold.total(..., count)`. `Fold.total` is
  the specification of the loop: `+0.0` at `count = 0`, and
  `Fold.chosen(operation, keep(plan, k), total(k), input[ia(plan, k)])` at
  `count = k + 1`, which is `g(operation, total(k), input[ia(plan, k)])` where
  the mask holds and `total(k)` elsewhere.
- `kept`: the returned input still holds every word it held before.

`PairFoldContract` has `value`, `left_kept` and `right_kept`. The proof
follows the loop: after `k` steps the input is the replayed tensor with
history `ops ++ reads(ia, k)`, and the accumulator is `Fold.total(..., k)`.

### Each instance meets an existing specification

Each instance states its law against a specification that the repository
already uses, not against `Fold.total`:

| Law | Instance | Total |
| --- | --- | --- |
| `sum_ordered` | `Sum.sum(input)` in `sum.bend` folds all `numel` words with `F32.add(acc, word)`, reading word `k` at step `k`. | `Reduction.sequential_sum(words(..., numel), 0.0)` |
| `dot_ordered` | `Dot.dot(Cell{row, column, inner, columns}, left, right)` in `dot.bend` folds `inner` pairs with `F32.add(acc, F32.mul(x, y))`, reading `left[row * inner + k]` and `right[k * columns + column]`. | `MatmulLaws.dot_fold(factors(..., inner), 0.0)` |
| `reduce_fold_fiber` | `ReduceFold.cell(plan, index, contribution)` in `src/autograd/reduce-fold.bend` folds the `numel` words of the contribution with `F32.add(acc, word)`, reading `contribution[right_map(plan, k)]` at step `k` and keeping it where `left_map(plan, k) == index`. | `ReduceLaws.fiber(plan, words(..., numel), index, numel)`, the specification of one `reduce_to` cell |

`sum_ordered` passes the mask `Sum.every`, which is always true, so its
statement is the one it had before the mask existed.

`reduce_fold_fiber` takes the broadcast plan of the target (left) with the
contribution (right), and requires that the plan's output has the
contribution's extents, which is the case `reduce_to` uses. Its conclusion is a
`FiberContract` in `src/autograd/reduce-fold-LAWS.bend`, with live fields
`value` and `kept`. The proof walks the fold and the specification together:
at step `k`, `Projection.offset_spec(plan, k)` is
`Some{(left_map(plan, k), right_map(plan, k))}`, from the shape laws
`project_left_index_refines` and `project_right_index_refines`, and the word
the fold reads is the `k`-th entry of `words`.

`words` and `factors` are in `src/tensor/contracts/LAWS.bend`. `words` lists
the first `count` words of a replayed input in index order. `factors` lists
the pairs `left[row, k]` and `right[k, column]` of two row-major matrices in
increasing `k`. Both compute their indices themselves, so a wrong index map in
an instance fails the proof instead of changing what the law means.

`dot_ordered` states the matrix shapes as hypotheses:
`Shape.extents(left) == [rows, inner]`, `Shape.extents(right) == [inner, columns]`,
`row < rows` and `column < columns`. `dot-proof.bend` derives the index bounds
from `lt_mul_add`: `a * n + b < m * n` when `a < m` and `b < n`.

## What the families do not cover yet

- `Elementwise.binary`, the public operation in `src/tensor/elementwise.bend`,
  runs its own loop. `Broadcast.binary` is a separate function that runs the
  family loop, and no law yet relates the two. Routing `Elementwise.binary`
  through the family, or proving the two equal, comes after the open work on
  `elementwise.bend` lands.
- `Reduction.sum` in `src/tensor/reduction.bend` and `Matrix.dot` in
  `src/tensor/matmul.bend` run their own loops too. `Sum.sum` and `Dot.dot`
  are separate functions, and no law yet relates them to the public
  operations. That work also waits for the open work on `matmul.bend`.
- The family output is itself a replayed tensor, with history `writes(n)` over
  `+0.0`. The proof uses that fact, but no law exports it, so a consumer cannot
  yet chain two family operations through laws alone.
- The laws require every index in bounds, so they say nothing about a failed
  read. What `Fold.run` and `Fold.pair_run` return then, the error and the
  inputs, is checked only by the failure consumers in
  `verification/families/fold/`.
- F32 operations are opaque. A value is stated as the term
  `F32.sub(5.0, 20.0)`, not as `-15.0`.
- `Reduce.reduce_to` in `src/autograd/reduce.bend` runs its own loop.
  `ReduceFold.cell` meets the same specification, `ReduceLaws.fiber`, but no
  law yet says that `reduce_to` computes its cells with it.

## Controls

`verification/families/` holds literal consumers of `broadcast_pointwise`,
two of them on the actual `Core.from_values` and `Core.new` results, of
`sum_ordered` and `dot_ordered`, of failed folds, and of `reduce_fold_fiber`,
false twins of thirteen value statements, and twenty-five mutants of the
family loops and of their instances. Its [README](../verification/families/README.md) lists each
control and the rejection it expects.

The proofs and controls check what the laws state, not the compiled code.
`tools/tensor_families.py`, which the S2 gate runs, compiles `Broadcast.binary`
in the native and JavaScript lanes and runs the 67 S2 `add`, `sub` and `mul`
requests that reach Bend through it. They include the `ieee_edges` cases,
whose words are signed zeros, subnormals, the largest finite values, NaN and
infinities. Each lane must give the output words that the S2 numerical policy
accepts, the four broadcast and capacity errors, and both inputs unchanged.
Its results are in `pointwise-family/` of the S2 output, apart from the
`Elementwise.binary` corpus.
