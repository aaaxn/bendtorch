# Contract family controls

The family laws in `src/tensor/contracts/LAWS.bend` are proved in
`src/tensor/contracts/PROOF.bend`, aggregated by `src/tensor/LAWS.bend`, and
required by `verification/tensor.json`, so the root `PROOF.bend --verdict` and
the tensor gate protect them. The `reduce_to` laws are stated in
`src/autograd/reduce-fold-LAWS.bend` and required by
`verification/autograd.json`. [Contract
families](../../docs/contract-families.md) explains what they state. This
directory holds literal consumers, false twins and implementation mutants,
beside the runner that checks them.

## Pointwise

`pointwise/` holds:

- `consumers.bend`: ten consumers of `broadcast_pointwise`. Each one takes
  its plan from `Shape.broadcast_checked`. The first eight take their inputs
  as the replayed tensors that `Core.from_values` returns, and restate
  literal values:
  - `[2, 3] - [3]` at output indices 0, 4 and 5.
  - `[2, 1] * [1, 3]`, an outer product, at indices 1 and 3.
  - `[3] + [3]` at index 2.
  - A rank-0 `KeepLeft` against `[3]` at index 2.
  - The empty `[2, 0] + [0]`, which states success and dimensions only.

  Each value consumer also states one word of each returned input. F32
  operations are opaque, so `[2, 3] - [3]` at index 4 is stated as
  `F32.sub(5.0, 20.0)`.

  The last two name the producers themselves. `from_values_sub` states
  `[3] - [3]` at index 2 for the actual `Core.from_values` results, and gets
  the contract through `from_values_replayed`. `new_sub` does the same for
  two `Core.new` fills through `new_replayed`, and states
  `F32.sub(2.0, 5.0)`. `from_values_pointwise` and `new_pointwise` are the
  generic compositions they instantiate. BendTT evaluates the produced inputs
  while it checks these two, and on the `[2, 3]` grid it runs out of fuel, so
  they use `[3]`.
- `controls.bend`: five value witnesses, each paired with one wrong value.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `sub_witness` | `F32.sub(20.0, 5.0)`, operands swapped | the witness | the witness |
| False twin `outer_witness` | `F32.mul(2.0, 30.0)`, left row not broadcast | the witness | the witness |
| False twin `keep_witness` | `8.0` for a `7.0` fill | the witness | the witness |
| False twin `kept_witness` | left word 4 reads the output value | the witness | the witness |
| False twin `produced_witness` | `F32.sub(30.0, 3.0)` for the `Core.from_values` inputs, operands swapped | the witness | the witness |
| Swapped operands | `right_step` writes `f(operation, y, x)` | `step_lift` | `step_lift.qvvvvv` |
| Wrong write index | `right_step` writes at `index + 1` | `step_lift` | `step_lift.qvvvvv` |
| Missing write | `right_step` skips the write | `step_lift` | `step_lift.qvvvvv` |
| Wrong input map | `step` reads the left input at `ib` | `step_lift` | `step_lift.qvvvvv` |
| Cursor off by one | `allocated` starts the loop at 1 | `run_lift` | `run_lift.qvvvvv` |
| Swapped maps | `Broadcast.binary` passes `right_map, left_map` | `broadcast_pointwise` | `step_lift.qvvvvv_1` |
| Wrong output shape | `Broadcast.binary` allocates the output with `Shape.left_shape(plan)` | `broadcast_pointwise` | `broadcast_pointwise` |

The first five mutants change the generic loop in `pointwise.bend`, and the
generic proof rejects them. The last two change only the broadcast instance.
The source checker rejects them where `broadcast-proof.bend` instantiates the
theorem. BendTT rechecks each template instance, so it rejects the swapped
maps in the broadcast copy of the generic step proof, `step_lift.qvvvvv_1`.
It rejects the wrong output shape in `broadcast_pointwise` itself, whose
statement names `Shape.output_shape(plan)`.

Each mutated operation still checks on its own. The runner checks
`pointwise/entry.bend`, which imports only `broadcast.bend`, in each mutant
copy. For BendTT, the runner exports that entry and transplants its changed
declarations into the positive export, so the kernel rechecks the unchanged
proofs against the mutated operation. The runner checks that each transplant
replaces exactly the mutated declarations: the generic copy and the broadcast
instance copy, only the instance copy for the swapped maps, or only
`Broadcast.binary` for the wrong output shape.

## Fold

`fold/` holds:

- `consumers.bend`: nine consumers of `sum_ordered`, `dot_ordered` and
  `matmul_ordered`, and three of failed folds. Each one takes its inputs as the replayed tensors
  that `Core.from_values` returns. The first nine restate literal values:
  - The sums of `[1, 2, 3]`, of a `[2, 3]` grid and of a rank-0 `[7]`.
  - The sum of the empty `[2, 0]`, which is `0.0`.
  - Cells `(1, 0)` and `(0, 1)` of the product of a `[2, 3]` and a `[3, 2]`
    matrix.
  - Cell `(1, 2)` of the product of a `[2, 0]` and a `[0, 3]` matrix, which is
    `0.0`.
  - Cells `(1, 0)` and `(0, 1)` of `Matrix.matmul` on the same `[2, 3]` and
    `[3, 2]` inputs, with the output extents `[2, 2]`. `grid_tall` is the plan
    that `Matrix.plan` returns for them.

  Each non-empty sum and dot consumer also states one word of each returned
  input, and each `matmul` consumer one word of the returned right input. A total
  is stated as the term the fold builds: the sum of `[1, 2, 3]` is
  `F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)`.

  The family laws require every index in bounds, so no law says what a failed
  fold returns. The last three consumers run `Fold.run`, with the mask
  `Sum.every` that keeps every word, and `Fold.pair_run` directly with the
  index map `reads_past_input(k) = 2k`, which reads index 4 of a three-word
  input at step 2, after two successful steps. `sum_fails` folds `[1, 2, 3]`
  that way, `pair_fails_left` reads the left input with `reads_past_input`
  and `pair_fails_right` reads the right input with it, after the left read
  of the same step succeeds. Each states the error,
  `StorageError{OutOfBounds}`, and the first and last word of each returned
  input.
- `controls.bend`: five value witnesses, each paired with one wrong value.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `sum_witness` | the words added in reverse order | the witness | the witness |
| False twin `start_witness` | `7.0` for `F32.add(0.0, 7.0)`, the fold starts at the first word | the witness | the witness |
| False twin `dot_witness` | `F32.mul(5.0, 20.0)`, the right matrix read along a row | the witness | the witness |
| False twin `kept_witness` | left word 5 reads the right input | the witness | the witness |
| False twin `matmul_witness` | `F32.mul(5.0, 20.0)` in `matmul` cell `(1, 0)`, the right matrix read along a row | the witness | the witness |
| Negative zero start | `run` starts at `F32.neg(0.0)` | `contract` | `contract.qvvvvv` |
| Swapped accumulator | `step` folds `g(operation, x, accumulator)` | `step_lift` | `step_lift.qvvvvv` |
| Dropped word | `step` keeps the accumulator | `step_lift` | `step_lift.qvvvvv` |
| Ignored mask | `loop` keeps every word instead of asking `keep` | `step_lift` | `step_lift.qvvvvv` |
| Cursor off by one | `run` starts the loop at 1 | `contract` | `contract.qvvvvv` |
| Pair swapped operands | `pair_right` folds `g(operation, accumulator, y, x)` | `pair_step_lift` | `pair_step_lift.qvvvvv` |
| Pair wrong map | `pair_loop` reads the left input at `ib` | `pair_step_lift` | `pair_step_lift.qvvvvv` |
| Pair negative zero start | `pair_run` starts at `F32.neg(0.0)` | `pair_contract` | `pair_contract.qvvvvv` |
| Pair cursor off by one | `pair_run` starts the loop at 1 | `pair_contract` | `pair_contract.qvvvvv` |
| Failed step done | `step` returns `Done{accumulator}` when a read fails | `sum_fails` | `consumers.sum_fails` |
| Pair left failed done | `pair_left` returns `Done{accumulator}` when the left read fails | `pair_fails_left` | `consumers.pair_fails_left` |
| Pair right failed swapped | `pair_right` returns the inputs as `(right, left)` when the right read fails | `pair_fails_right` | `consumers.pair_fails_right` |
| Sum swapped add | `Sum.add` returns `F32.add(word, accumulator)` | `total_words` | `total_words` |
| Dot untransposed right | `Dot.right_index` reads `column * inner + k` | `total_factors` | `total_factors` |
| `matmul` reversed order | `natural_dot` reads `k` from `inner - 1` down to 0 | `dot_step_lift` | `successful_dot_interpretation` |
| `matmul` negative zero start | `product_coordinates` starts the dot at `F32.neg(0.0)` | `product_step_lift` | `product_coordinates_proof` |
| `matmul` transposed right | `dot_decided` reads the right matrix at `[column, k]` | `dot_decided_lift` | `dot_decision` |

The first nine mutants change the generic loops in `fold.bend`, and the
generic proofs reject them. The next three change what the loops return when
a read fails. The proofs accept them, since no law covers a failed read, and
the runner checks them against `fold/consumers.bend` instead, where the
failure consumers reject them. The next two change one instance each. The
proof that relates the instance's `Fold.total` to its specification rejects
them. The last three change the `matmul` loops in `src/tensor/matmul.bend`.
The erased proofs in `src/tensor/matmul-PROOF.bend` reject them too, so the
source check runs `src/tensor/contracts/matmul-proof.bend`, the live proof of
`matmul_ordered`, instead of `src/tensor/contracts/PROOF.bend`, and rejects
them in a step of its walk. The kernel artifact holds the erased proofs too:
`src/tensor/contracts/PROOF.bend` imports `matmul-PROOF.bend` to discharge the
laws of `matmul-LAWS.bend`, which defines `dot_fold`. So BendTT reports each of
these mutants first in an erased `matmul-LAWS.bend` or `matmul-PROOF.bend`
declaration, and the table names that declaration. The fold family's BendTT
prefix is `src/tensor/`, so the transplant also carries the mutated
`matmul.bend` declarations.

`fold/entry.bend` imports `broadcast.bend` before `sum.bend`, `dot.bend` and
`matmul.bend`.
The BendTT export names each opaque template constant after the first module
that declares one of its type, and the positive export reaches `pointwise.bend`
before `fold.bend`. With the same import order, the mutant export names the
constants the same way, and each transplant replaces only the mutated
declarations.

## Reduce

`reduce/` holds:

- `consumers.bend`: five consumers of `reduce_fold_fiber`, the law that states
  `ReduceFold.cell` as the `reduce_to` specification of one output cell: an
  ordered fold over the contribution words that land on it. Each consumer
  reduces the `[2, 3]` contribution `[1, 2, 3, 4, 5, 6]`, replayed as the
  writes of `Core.from_values`, to a target:
  - `[3]`: cell 1 adds column 1, words 1 and 4.
  - `[2, 1]`: cell 1 adds row 1, words 3, 4 and 5.
  - `[]`: the one cell adds every word.
  - `[2, 3]`: no axis is reduced, and cell 4 adds word 4 to `+0.0`.
  - `[0]` of a `[2, 0]` contribution: no word lands, and the cell is `+0.0`.

  Each non-empty consumer also states one word of the returned contribution.

  Two more consumers, `reduce_to_columns` and `reduce_to_rows`, take the same
  contribution through `reduce_to_fiber`, the law about `Reduce.reduce_to`
  itself. They reduce it to `[3]` and to `[2, 1]` and state the output
  dimensions, the values of cell 1 that `reduce_columns` and `reduce_rows`
  state, and one word of the returned contribution.
- `controls.bend`: five value witnesses, each paired with one wrong value.
  The kernel reports a mismatch against the type of the witness body, where
  the law's total is still an unevaluated fiber, so `stated` restates each
  total as the literal it evaluates to.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `column_witness` | every word of the grid, the mask ignored | the witness | the witness |
| False twin `row_witness` | row 0 instead of row 1 | the witness | the witness |
| False twin `same_witness` | `5.0` for `F32.add(0.0, 5.0)`, the word copied without the fold | the witness | the witness |
| False twin `word_witness` | contribution word 4 read at word 5 | the witness | the witness |
| False twin `reduce_to_witness` | every word of the grid in `reduce_to` cell 1, the landing ignored | the witness | the witness |
| Lands on source | `lands` compares the contribution offset with the cell | `fiber_total` | `fiber_total` |
| Inverted mask | `lands` keeps the words that do not land on the cell | `fiber_total` | `fiber_total` |
| Reads target offset | `source` reads the word at the target offset | `fiber_total` | `fiber_total` |
| Target count | `cell` folds over the target's element count | `reduce_fold_fiber` | `reduce_fold_fiber` |
| `reduce_to` lands on source | `step_indices` adds each word into the cell at its contribution offset | `step_lift` | `step_lift` |
| `reduce_to` reversed order | `step` visits positions from the last to the first | `step_lift` | `step_lift` |
| `reduce_to` swapped add | `step_output` writes `F32.add(value, accumulated)` | `step_lift` | `step_lift` |

The first four mutants change `src/autograd/reduce-fold.bend`, and the proofs
in `src/autograd/reduce-fold-PROOF.bend` reject them. The last three change
the `reduce_to` loop in `src/autograd/reduce.bend`. The erased proofs in
`src/autograd/reduce-PROOF.bend` reject them too, and each checker reports
the first failure it meets. So the source check runs
`src/autograd/reduce-to-fold-PROOF.bend`, the live proof of `reduce_to_fiber`,
instead of `src/autograd/PROOF.bend`, and `controls.bend` imports only the two
family proofs, so the kernel artifact holds no erased `reduce_to` proof. Both
checkers then reject these mutants in `step_lift`. `reduce/entry.bend` imports
`broadcast.bend` and `sum.bend` before `reduce-fold.bend` and `reduce.bend`,
for the same export names as `fold/entry.bend`.

## Run

```sh
uv run --locked python -m verification.families.run --output <fresh directory> --workers 24
uv run --locked python -m unittest verification/families/test_run.py -v
```

The runner snapshots `src/`, `pointwise/`, `fold/` and `reduce/`. For each family it
builds the false twins and the mutant copies, and runs the source checks,
`--verdict` on the consumers and the controls, the exports, and BendTT on the
positive and mutated exports. Each family writes its outputs to its own
subdirectory. `summary.json` records the repository and Bend commits, every
command, its exit code and time, and the rejection reason of every control.
