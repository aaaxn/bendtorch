# Contract families

Many tensor operations share one loop. `add`, `sub` and `mul`, with or
without broadcasting, all write `out[i] = f(a[ia(i)], b[ib(i)])` for every
output index `i`. A contract family proves that loop once, as a generic
theorem over `f`, `ia` and `ib`, and each operation gets its contract by
instantiating it. The code is in `src/tensor/contracts/`.

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

## What the family does not cover yet

- `Elementwise.binary`, the public operation in `src/tensor/elementwise.bend`,
  runs its own loop. `Broadcast.binary` is a separate function that runs the
  family loop, and no law yet relates the two. Routing `Elementwise.binary`
  through the family, or proving the two equal, comes after the open work on
  `elementwise.bend` lands.
- The family output is itself a replayed tensor, with history `writes(n)` over
  `+0.0`. The proof uses that fact, but no law exports it, so a consumer cannot
  yet chain two family operations through laws alone.
- F32 operations are opaque. A value is stated as the term
  `F32.sub(5.0, 20.0)`, not as `-15.0`.
- The ordered-fold family (sum, the matmul dot product and `reduce_to`) comes
  next.

## Controls

`verification/families/` holds literal consumers of `broadcast_pointwise`,
two of them on the actual `Core.from_values` and `Core.new` results, false
twins of five value statements and seven mutants of the family loop and of
the broadcast instance. Its [README](../verification/families/README.md)
lists each control and the rejection it expects.

The proofs and controls check what the laws state, not the compiled code.
`tools/s2_families.py`, which the S2 gate runs, compiles `Broadcast.binary`
in the native and JavaScript lanes and runs the 67 S2 `add`, `sub` and `mul`
requests that reach Bend through it. They include the `ieee_edges` cases,
whose words are signed zeros, subnormals, the largest finite values, NaN and
infinities. Each lane must give the output words that the S2 numerical policy
accepts, the four broadcast and capacity errors, and both inputs unchanged. Its results are in `pointwise-family/` of the S2
output, apart from the `Elementwise.binary` corpus.
