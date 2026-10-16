# Contract families

Many tensor operations share one loop. `add`, `sub` and `mul`, with or
without broadcasting, all write `out[i] = f(a[ia(i)], b[ib(i)])` for every
output index `i`. A contract family proves that loop once, as a generic
theorem over `f`, `ia` and `ib`, and each operation gets its contract by
instantiating it. Reductions share a second loop: `sum`, the matmul dot
product and one cell of `reduce_to` fold words into an accumulator in a fixed
order. The code is in `src/tensor/contracts/`, and the `reduce_to` instance is
in `src/autograd/reduce-fold.bend`. The law `reduce_to_fiber` states that
`reduce_to` itself computes every cell with the same total, and
`matmul_ordered` states that `Matrix.matmul` computes every output cell with
the total of `dot_ordered`. `broadcast_to_pointwise` states that
`broadcast_to` writes every cell as the broadcast instance of the pointwise
family does with `KeepLeft`, and `sum_to_size_clone` and `sum_to_size_fiber`
state `sum_to_size` through the `reduce_to` contract.

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

### `broadcast_to` meets the family specification

`Broadcast.broadcast_to` in `src/autograd/broadcast.bend`, the materialized
broadcast that autograd calls, does not call `Pointwise.run`. It runs the
sequential elementwise loop `Elementwise.allocated` with `KeepLeft`, the input
on the left and a `+0.0` tensor of the target on the right. The law
`broadcast_to_pointwise` in `src/autograd/broadcast-pointwise-LAWS.bend` states
what it computes. It takes:

- an input and a target shape, and the plan with
  `Shape.broadcast_checked(shape, target) == Done{plan}` and
  `Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(target)`,
  the case where `broadcast_to` succeeds;
- any replayed input of that shape.

Its conclusion is a `BroadcastToContract`, with live fields `successful`,
`dimensions` (the target's), `value` and `kept`. `value` states that every
in-bounds output index `i` holds `input[left_map(plan, i)]`, the input's
replayed word at its broadcast projection.

`broadcast-pointwise-PROOF.bend` walks the loop with live code, without
changing it. The zero tensor and the output come from `Core.new`, so they are
replayed too. After `k` steps each input is the replayed tensor with its
history plus `reads(k)` of its map, and the output is the replayed tensor over
`+0.0` with the family history `Pointwise.writes(k)` of `Elementwise.scalar`
with `left_map` and `right_map`. The loop's index check, `input_indices`, gives
the two maps, through `Projection.input_indices_refines`. At the end
`writes_value` gives cell `i` as `scalar(KeepLeft, input word, 0.0)`, which is
the input word. A changed operation, swapped operands or another owner in
`broadcast.bend` fails the walk or the contract.

The walk also gives the replayed form of the result. The law
`broadcast_to_replayed` exports it as a transport, like `new_replayed`: for
every `P`, if `P` holds for the input with the history `ops ++ reads` and the
output over `+0.0` with the family history `writes`, for any length proofs,
then `P` holds for the `broadcast_to` result. So a law about a replayed
tensor applies to the output, and a consumer can chain `broadcast_to` into
`reduce_to` through laws alone.

### `Elementwise.binary` exports its replayed result

`Elementwise.binary` in `src/tensor/elementwise.bend`, the public `add`,
`sub` and `mul`, does not run a loop over the output. `build` allocates the
output as a balanced tree of depth `d`: the top `d - 15` levels fork into two
halves that run in parallel, each with its own clone of both input arrays,
and below the forks one thread builds its subtree in index order. Cell `k`
gets `scalar(operation, left[left_map(plan, k)], right[right_map(plan, k)])`
below the count and `+0.0` above it, so no word depends on the split.

The law `binary_replayed` in `src/tensor/contracts/binary-LAWS.bend` states
that for replayed inputs whose shapes broadcast, in the transport form of
`broadcast_to_replayed`: for every `P`, if `P` holds for every result that
hands both inputs back with their words, next to an output replayed over
`+0.0` whose words are `cells`, the words of the family's broadcast instance,
then `P` holds for `Elementwise.binary`. The returned inputs are the First
halves of their clones.

The proof walks the build with live code; the older proof in
`elementwise-parallel-proof.bend` follows the same build with erased evidence
only.

- `binary-storage-proof.bend` puts a replayed input in a normal form: the
  owned storage whose array is `Api.arr` of a history of storage effects, the
  form the storage proofs use. Every array the build reads or clones is then
  a term of reusable data, and the proof can name it as often as it needs.
- `binary-build-proof.bend` shows that `build` returns `spec`, the balanced
  tree that holds each family cell below the count and `+0.0` above it, for
  every number of forks: `build_eq` follows the forks, `tree_eq` the
  sequential subtrees, and each read of a cell is the input word the history
  names.
- `binary-proof.bend` replays the family history `Pointwise.writes` on the
  storage `Core.new` makes and shows it holds the same tree, by writing one
  cell at a time into `out_spec`. Two owned storages with equal arrays are
  equal, because a balanced certificate of an array is unique
  (`balanced_unique`).

The last step compares the output storages' erased certificates, so the
certificate the code stores has to be a live term. `build_balanced` in
`elementwise.bend` therefore follows the code with a `Certified` run that
pairs each subtree with its proof of balance. It sits in the erased field of
the output storage and does not run. The declarations of the existing laws on
`Elementwise.binary` and what they guarantee about values, dims and success
are unchanged, and `elementwise-parallel-proof.bend` and the erased laws that
use it check as before. The term those laws refer to does change in its
erased field, which is still a change to a definition a law calls, so a PR
that edits it names the affected laws as AGENTS.md requires.

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
| `dot_ordered` | `Dot.dot(Cell{row, column, inner, columns}, left, right)` in `dot.bend` folds `inner` pairs with `F32.add(acc, F32.mul(x, y))`, reading `left[row * inner + k]` and `right[k * columns + column]`. | `DotFold.dot_fold(factors(..., inner), 0.0)` |
| `reduce_fold_fiber` | `ReduceFold.cell(plan, index, contribution)` in `src/autograd/reduce-fold.bend` folds the `numel` words of the contribution with `F32.add(acc, word)`, reading `contribution[contribution_offset(plan, k)]` at step `k` and keeping it where `target_offset(plan, k) == index`; the two offsets are `right_map` and `left_map`. | `ReduceLaws.fiber(plan, words(..., numel), index, numel)`, the specification of one `reduce_to` cell |

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

### `reduce_to` meets the same specification

`reduce_fold_fiber` states one cell of `ReduceFold.cell`, a separate function.
`Reduce.reduce_to` in `src/autograd/reduce.bend`, the function that autograd
calls, builds its cells in parallel: each target cell walks only the
contribution positions that land on it, in increasing order (see
[reduce to shape](reduce-to.md#parallel-build)). The law `reduce_to_fiber` in
`src/autograd/reduce-fold-LAWS.bend` states what that build computes. It takes:

- a target and a contribution shape;
- the plan with `Shape.broadcast_checked(target, shape) == Done{plan}`, and
  `Shape.dimensions(Shape.output_shape(plan)) == Shape.dimensions(shape)`,
  the case where `reduce_to` succeeds;
- any replayed contribution of that shape.

Its conclusion is a `ReduceToContract`, with live fields:

- `successful`: `reduce_to` succeeds.
- `dimensions`: the output has the target's dimensions.
- `value`: every in-bounds target cell holds
  `ReduceLaws.fiber(plan, words(..., numel), index, numel)`, the value that
  `reduce_fold_fiber` states for `ReduceFold.cell`.
- `kept`: the returned contribution still holds every word.

The proof in `src/autograd/reduce-to-fold-PROOF.bend` (`run_lift`) follows
the parallel build with live code, as `contracts/matmul-parallel-proof.bend`
does for matmul. The contribution comes back replayed with the history
`ops ++ [First]` of its kept clone, and the build reads the copy
`Api.arr(history(ops))`. `reduce-replayed-proof.bend` shows that the build over
that copy is the tree whose cell `t` holds `cell_word(t)`, the replayed words
at the offsets the walk of `t` reads added from `+0.0` (`fiber_eq`,
`build_eq`), and that writing those words cell by cell over the `+0.0`
target gives the same owned storage (`output_eq`). So the output is the
replayed tensor with history `writes(count)`. `word_fiber` turns `cell_word(t)`
into `ReduceLaws.fiber`, through `Fiber.fiber_walk`. A walk over the wrong
offsets, a skipped digit, another addition order or a `-0.0` start in
`reduce.bend` fails the proofs of the build (`fiber_run`, `fiber_cons`,
`leaf_run`, `run_of`). A wrong landing map or an inverted mask in
`reduce-fold.bend` fails `fiber_total`, which `reduce_fold_fiber` uses.

`words` and `factors` are in `src/tensor/contracts/LAWS.bend`. `words` lists
the first `count` words of a replayed input in index order. `factors` lists
the pairs `left[row, k]` and `right[k, column]` of two row-major matrices in
increasing `k`. Both compute their indices themselves, so a wrong index map in
an instance fails the proof instead of changing what the law means.

`dot_ordered` states the matrix shapes as hypotheses:
`Shape.extents(left) == [rows, inner]`, `Shape.extents(right) == [inner, columns]`,
`row < rows` and `column < columns`. `dot-proof.bend` derives the index bounds
from `lt_mul_add`: `a * n + b < m * n` when `a < m` and `b < n`.

### `matmul` meets the dot specification

`Dot.dot` is a separate function. `Matrix.matmul` in `src/tensor/matmul.bend`,
the public operation, builds its output in parallel. Each output position `p`
sits at row `p / columns` and column `p % columns`; its cell folds
`left[row * inner + k] * right[k * columns + column]` for `k` from 0 to
`inner - 1` from `+0.0`, reading the words of a copy of each input with
`Array.get`. The cells form a balanced tree: the top `Matrix.splits` levels
fork, each half with its own clone of both input arrays, and a thread builds
the cells below the forks in index order. The law `matmul_ordered` in
`src/tensor/contracts/LAWS.bend` states what that build computes. It takes:

- two shapes and the plan with `Matrix.plan(left, right) == Done{plan}`, the
  case where `matmul` succeeds;
- any replayed left and right inputs of those shapes.

Its conclusion is a `MatmulContract` in `fold-records.bend`, with live fields:

- `successful`: `matmul` succeeds.
- `extents`: the output has the extents `[rows, columns]` of the plan.
- `value`: for every `row < rows` and `column < columns`, the output word at
  `row * columns + column` is
  `DotFold.dot_fold(factors(row, column, inner, columns, ..., inner), 0.0)`,
  the total that `dot_ordered` states for `Dot.dot`.
- `left_kept` and `right_kept`: both returned inputs still hold every word.

The proof in `matmul-parallel-proof.bend` walks the build with live code, as
the elementwise export does. A replayed input with a cell is an owned storage
whose array is `Api.arr` of its normalized history; `take` hands back the
First half of its clone, which lists the same words, and the build reads the
Second half. Every read of such an array keeps it and returns the word the
history names, and every clone is two copies of it, so whatever the split the
build returns `out_spec`, the tree whose cell `p` holds the total above below
the output count and `+0.0` above it (`build_eq`). The writes of every cell in
index order, replayed on the storage `Core.new` makes for the output, give the
owned storage that holds `out_spec` (`chain`), so the output is a replayed
tensor. `fold_factors` turns each cell's fold into `dot_fold` of `factors`,
and `Nat.divmod` maps `row * columns + column` back to its row and column. A
wrong start, another order of `k` or a transposed read in `matmul.bend` fails
a step of the walk. An input with no cell (`inner == 0`) has no word to read;
its copy is `Array.new` of depth 0 and every cell is the empty fold, `+0.0`.

`matmul_replayed` exports the same walk in the transport form of
`broadcast_to_replayed`: for every `P`, if `P` holds for every result that
hands both inputs back with the same words, next to an output of the plan's
shape, replayed over `+0.0`, whose cell `(row, column)` holds the total above,
then `P` holds for the product. Its input histories are the inputs' histories
with a `First` clone half appended (or unchanged for an input with no cell),
and its output history writes each cell's total in index order.

The sequential loops `Matrix.multiply`, `natural_product_loop` and
`natural_dot` remain the definition the laws in `src/tensor/matmul-LAWS.bend`
state for them; the gate's parallel probe checks, at 1, 8 and 24 threads, that
the public product returns the same words as `Matrix.multiply` on products
that fork.

### Products that read an operand in place

`product_ordered` and `product_replayed` state `Matrix.product(lo, ro, ...)`,
the product whose operands are each laid out `Kept` or `Swapped`
([sum and matmul](sum-and-matmul.md#products-that-read-an-operand-in-place)),
for every pair of orders: `matmul_nt` is `Kept`, `Kept` and `matmul_tn` is
`Swapped`, `Swapped`. They are `matmul_ordered` and `matmul_replayed` with
`factors` replaced by `product_factors`, which reads the left word at
`Matrix.read_offset(lo, inner, rows, row, k)` and the right word at
`Matrix.read_offset(ro, inner, columns, column, k)`. The proofs in
`product-parallel-proof.bend` and `product-proof.bend` are the matmul walk
with each read at `read_offset`; `Fits(order, inner, outers, shape)` names the
extents each order needs, and `read_below` bounds each read for either order.
`matmul_nt_ordered` and `matmul_tn_ordered` state `product_ordered` for the
public `matmul_nt` and `matmul_tn`, so the orders each passes are part of a
law.

### `permute_copy` and `transpose_copy` through the index-map copy

`Transforms.copy_loop` in `src/tensor/transforms.bend` writes output index
`i` from the source index `permute_source(input, axes, i)`, for `i` in
increasing order. `src/tensor/index-map.bend` treats that as the pointwise
family with a scalar function that keeps its first operand and the same map
on both operands, and `permute-values.bend` walks the copy loop with live
code, without changing it. The walk takes any replayed source, with any fill
and history.

Two laws in `src/tensor/permute-values-contract-LAWS.bend` export it in the
transport form:

- `permute_replayed`: once `Layout.permutation(input, axes)` gives a plan,
  `P` holds for `permute_dynamic` of a replayed tensor if it holds for every
  result that hands the source back with the same words, next to an output
  of the plan's shape, replayed over `+0.0`, whose words are
  `permuted_words`: word `i` is the source's word at
  `permute_source(input, axes, i)`.
- `transpose_replayed`: with both axes in range, `transpose_dynamic(a, b)` is
  that permute by `swapped_axes(input, a, b)`.

`permute_contract` and `transpose_contract` are the same walk on a
`Core.from_values` source, stated as observations; their statements did not
change.

### `sum_to_size` through the `reduce_to` contract

`SumToSize.sum_to_size` in `src/autograd/sum-to-size.bend` clones the
contribution when the target has its dimensions and is `reduce_to` otherwise.
Two laws in `src/autograd/sum-to-size-fold-LAWS.bend` state both cases as a
`ReduceToContract`, the record of `reduce_to_fiber`, on any replayed
contribution:

- `sum_to_size_clone`: when the dimensions are equal, every in-bounds cell
  holds the contribution's own word, not `F32.add(0.0, word)` as `reduce_to`
  gives, so `-0.0` stays `-0.0`.
- `sum_to_size_fiber`: when the dimensions differ and the target broadcasts to
  exactly the contribution's dimensions, every cell holds the `ReduceLaws.fiber`
  that `reduce_to_fiber` states.

`sum-to-size-fold-PROOF.bend` proves the first through `Core.clone` on a
replayed tensor: the storage laws model a clone's two halves as the histories
`ops ++ [First]` and `ops ++ [Second]`, and neither changes a word. It proves
the second by deciding the dimension check and calling `reduce_to_fiber`.

A third law, `sum_to_size_replayed`, exports the result in the transport form
of `broadcast_to_replayed`. For a replayed contribution whose words are a list
`values` and a target that broadcasts to exactly its dimensions, `P` holds for
the call if it holds for every result that hands the contribution back
replayed with the same words, next to a replayed output with the target's
dimensions whose cells hold `summed`: `values[index]` when the dimensions are
equal and `ReduceLaws.fiber(plan, values, index, numel)` otherwise. The proof
runs the two cases above with the histories exposed: the clone halves, and
`reduce_to`'s `ops ++ reads` and `history`. A consumer can then hand either
half to another family law, which the tensor backward's
[VJP value laws](tensor-backward.md#vjp-values) do.

## What the families do not cover yet

- `binary_replayed` covers replayed inputs whose shapes broadcast. A failed
  plan keeps the existing erased laws in `core-LAWS.bend`.
- `Reduction.sum` in `src/tensor/reduction.bend` runs its own loop too.
  `Sum.sum` is a separate function, and no law yet relates the two.
- `matmul_ordered` covers a successful product. A failed plan or read keeps
  the existing erased laws in `matmul-LAWS.bend`.
- The family output is itself a replayed tensor, with history `writes(n)` over
  `+0.0`. `broadcast_to_replayed` exports that fact for `broadcast_to`,
  `sum_to_size_replayed` for `sum_to_size` (and so for `reduce_to`'s output
  when axes are reduced), `binary_replayed` for the parallel
  `Elementwise.binary`, which `tensor_product_replayed` restates for the
  tensor backward's products, `permute_replayed` and `transpose_replayed` for
  the copy loop, and `matmul_replayed` for `Matrix.matmul`. No law exports it
  for `Broadcast.binary`. The tensor backward chains the exported ones: the
  `Sum`, `Add`, `Sub`, `Mul` and `Matmul` VJPs have value laws
  ([tensor backward](tensor-backward.md#vjp-values)).
- `broadcast_to_pointwise`, `sum_to_size_clone` and `sum_to_size_fiber` cover
  a successful call. A failed plan or target check keeps the existing erased
  laws in `broadcast-LAWS.bend`, `sum-to-size-LAWS.bend` and
  `reduce-LAWS.bend`.
- The laws require every index in bounds, so they say nothing about a failed
  read. What `Fold.run` and `Fold.pair_run` return then, the error and the
  inputs, is checked only by the failure consumers in
  `verification/families/fold/`.
- F32 operations are opaque. A value is stated as the term
  `F32.sub(5.0, 20.0)`, not as `-15.0`.

## Controls

`verification/families/` holds literal consumers of `broadcast_pointwise`,
two of them on the actual `Core.from_values` and `Core.new` results, of
`sum_ordered`, `dot_ordered` and `matmul_ordered`, of failed folds, of
`reduce_fold_fiber` and `reduce_to_fiber`, of `sum_to_size_clone` and
`sum_to_size_fiber`, of `broadcast_to_pointwise` and
`broadcast_to_replayed`, one of them a `broadcast_to` output reduced by
`reduce_to_fiber`, of the tensor backward's `Add`, `Sub`, `Mul`, `Matmul`
and `MatmulNT` VJP value laws, and of `binary_replayed`, one of them an
`Elementwise.binary` product reduced by `reduce_to_fiber`. It also holds false
twins of thirty-four value statements, and sixty mutants of the family
loops, of their instances and of the `reduce_to`, `matmul`, `sum_to_size`,
`broadcast_to`, tensor backward and parallel elementwise code. Its
[README](../verification/families/README.md) lists each control and the
rejection it expects.

The proofs and controls check what the laws state, not the compiled code.
`tools/tensor_families.py`, which the tensor gate runs, compiles
`Broadcast.binary` in the native and JavaScript lanes and runs the 67 tensor
`add`, `sub` and `mul` requests that reach Bend through it. They include the
`ieee_edges` cases, whose words are signed zeros, subnormals, the largest
finite values, NaN and infinities. Each lane must give the output words that
the tensor numerical policy accepts, the four broadcast and capacity errors,
and both inputs unchanged. Its results are in `pointwise-family/` of the tensor
gate output, apart from the `Elementwise.binary` corpus.
