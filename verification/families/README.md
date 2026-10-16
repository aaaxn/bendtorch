# Contract family controls

The family laws in `src/tensor/contracts/LAWS.bend` are proved in
`src/tensor/contracts/PROOF.bend`, aggregated by `src/tensor/LAWS.bend`, and
required by `verification/tensor.json`, so the root `PROOF.bend --verdict` and
the tensor gate protect them. The `reduce_to` laws are stated in
`src/autograd/reduce-fold-LAWS.bend`, the `sum_to_size` laws in
`src/autograd/sum-to-size-fold-LAWS.bend`, the `broadcast_to` laws in
`src/autograd/broadcast-pointwise-LAWS.bend` and the tensor VJP value laws in
`src/autograd/tensor-vjp-LAWS.bend`, and `verification/autograd.json`
requires them. [Contract
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
| `matmul` reversed order | `cell_dot` reads `k` from `inner - 1` down to 0 | `dot_eq` | `dot_eq` |
| `matmul` negative zero start | `dot_of` starts the cell at `F32.neg(0.0)` | `cell_inside_eq` | `cell_inside_eq` |
| `matmul` transposed right | `right_index` reads the right matrix at `column * columns + k` | `dot_right_eq` | `dot_right_eq` |
| `matmul_nt` reads the weight untransposed | `matmul_nt` is `product(Kept, Swapped)`, which reads `W` as `[inner, columns]` | `Laws.matmul_nt_ordered` | `matmul_nt_ordered` |
| `matmul_tn` swaps operands | `matmul_tn(g, x)` multiplies `x^T @ g` | `Laws.matmul_tn_ordered` | `matmul_tn_ordered` |
| Product reads a `Kept` operand swapped | `read_offset` reads a `Kept` operand at `k * inner + outer` | `read_below` | `read_below` |

The first nine mutants change the generic loops in `fold.bend`, and the
generic proofs reject them. The next three change what the loops return when
a read fails. The proofs accept them, since no law covers a failed read, and
the runner checks them against `fold/consumers.bend` instead, where the
failure consumers reject them. The next two change one instance each. The
proof that relates the instance's `Fold.total` to its specification rejects
them. The last three change the parallel `matmul` build in
`src/tensor/matmul.bend`. `src/tensor/contracts/matmul-parallel-proof.bend`,
the live walk of the build that proves `matmul_ordered`, rejects them in a
step of the walk: `dot_eq`, `cell_inside_eq` and `dot_right_eq`. The contracts
take `dot_fold` from `src/tensor/dot-fold.bend`, which holds no law, so the
kernel artifact holds no erased matmul proof and BendTT rejects them at the
same declarations. The products that read an operand in place add three:
`matmul_nt_ordered` and `matmul_tn_ordered` state which orders the public
`matmul_nt` and `matmul_tn` pass to `Matrix.product`, so they reject a
`matmul_nt` that reads `W` untransposed and a `matmul_tn` with swapped
operands; `read_below`, which bounds every read of `product-parallel-proof.bend`,
rejects a `Kept` read at the swapped index. The fold family's BendTT prefix is `src/tensor/`, so the
transplant also carries the mutated `matmul.bend` declarations.

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

  The last two take it through `sum_to_size`. `sum_to_size_same` uses
  `sum_to_size_clone` with the target `[2, 3]` and states that cell 4 holds
  `5.0` itself, where `reduce_same` states `F32.add(0.0, 5.0)` for the fold.
  `sum_to_size_columns` uses `sum_to_size_fiber` with the target `[3]` and
  states the cell 1 value of `reduce_to_columns`.
- `controls.bend`: seven value witnesses, each paired with one wrong value.
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
| False twin `sum_to_size_witness` | `F32.add(0.0, 5.0)` for the cloned word `5.0`, the identity case folded | the witness | the witness |
| False twin `sum_to_size_fiber_witness` | every word of the grid in `sum_to_size` cell 1, the landing ignored | the witness | the witness |
| Lands on source | `lands` compares the contribution offset with the cell | `fiber_total` | `fiber_total` |
| Inverted mask | `lands` keeps the words that do not land on the cell | `fiber_total` | `fiber_total` |
| Reads target offset | `source` reads the word at the target offset | `fiber_total` | `fiber_total` |
| Target count | `cell` folds over the target's element count | `reduce_fold_fiber` | `reduce_fold_fiber` |
| `reduce_to` walks target stride | `fiber` moves the offset by the left stride, not the right one | `fiber_run` | `fiber_run` |
| `reduce_to` skips last digit | `span` walks `extent - 1` digits of a reduced axis | `fiber_cons` | `fiber_cons` |
| `reduce_to` starts at -0.0 | `fiber_of` starts each cell at `F32.neg(0.0)` | `run_of` | `run_of` |
| `reduce_to` swapped add | `leaf_add` adds `F32.add(word, accumulator)` | `leaf_run` | `leaf_run` |
| `sum_to_size` identity folds | `decided` runs `reduce_to` when the dimensions are equal | `sum_to_size_clone` | `sum_to_size_clone` |
| `sum_to_size` reduced clones | `decided` clones when the dimensions differ | `sum_to_size_fiber` | `sum_to_size_fiber` |

The first four mutants change `src/autograd/reduce-fold.bend`, and the proofs
in `src/autograd/reduce-fold-PROOF.bend` reject them. The four `reduce_to`
mutants change the parallel build in `src/autograd/reduce.bend`. The source
check runs `src/autograd/reduce-to-fold-PROOF.bend`, the live proof of
`reduce_to_fiber`, instead of `src/autograd/PROOF.bend`. It imports the build
proofs, so both checkers reject each mutant in the first build proof it
breaks: `fiber_run`, `run_of` and `leaf_run` in
`src/autograd/reduce-parallel-proof.bend`, which the live proof imports
before its own, and `fiber_cons`, the fiber-walk step of
`src/autograd/reduce-fiber-proof.bend`, for a skipped digit.
The two `sum_to_size` mutants change `src/autograd/sum-to-size.bend`. The
erased `sum-to-size-PROOF.bend` would reject them too, so the source check runs
`src/autograd/sum-to-size-fold-PROOF.bend`, and the kernel artifact holds no
erased `sum_to_size` proof. Each mutant breaks one case and leaves the other
case's law true, and both checkers reject it in the law of the case it breaks.
`reduce/entry.bend` imports
`broadcast.bend` and `sum.bend` before `reduce-fold.bend`, `reduce.bend` and
`sum-to-size.bend`, for the same export names as `fold/entry.bend`.

## Broadcast to

`broadcast_to/` holds:

- `consumers.bend`: six consumers of `broadcast_to_pointwise` and
  `broadcast_to_replayed`. Each takes its input as the replayed tensor that
  `Core.from_values` writes and its plan from `Shape.broadcast_checked`:
  - `[3] = [1, 2, 3]` to `[2, 3]`: index 4, row 1 and column 1, holds input
    word 1.
  - `[2, 1] = [1, 2]` to `[2, 3]`: index 4, row 1, holds input word 1.
  - A scalar `7` to `[2, 2]` at index 3.
  - `[1] = [-0.0]` to `[2]` at index 1: the word is `F32.neg(0.0)` itself,
    not `F32.add(F32.neg(0.0), 0.0)`, because `KeepLeft` copies it.
  - The empty `[1, 3]` to `[0, 3]`, which states success and dimensions only.

  Each value consumer also states one word of the returned input. The last,
  `broadcast_then_reduce`, chains two laws: `broadcast_to_replayed` makes the
  output of `[7]` to `[2]` a replayed tensor, `reduce_to_fiber` reduces it
  back to `[1]`, and the cell is `F32.add(F32.add(0.0, 7.0), 7.0)`. BendTT
  evaluates the broadcast history while it checks this consumer, and on a
  `[2, 3]` grid it runs out of fuel, so the chain uses `[2]`.
- `controls.bend`: five value witnesses, each paired with one wrong value.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `columns_witness` | `1.0` at index 4 of `[2, 1]` to `[2, 3]`, row 0 read | the witness | the witness |
| False twin `rows_witness` | `2.0` at index 5 of `[3]` to `[2, 3]`, the row read as the column | the witness | the witness |
| False twin `keep_witness` | `F32.add(7.0, 0.0)` for `7.0`, an add against the zero tensor | the witness | the witness |
| False twin `kept_witness` | `0.0` for input word 2, the zero tensor returned as the owner | the witness | the witness |
| False twin `chain_witness` | `F32.add(0.0, 7.0)`, one copy reduced | the witness | the witness |
| Adds zero | `zeroed` runs `Add` instead of `KeepLeft` | `run_lift` | `run_lift` |
| Swapped operands | `zeroed` passes the zero tensor on the left | `run_lift` | `run_lift` |
| Returns zero owner | `finish` returns the zero tensor as the owner | `broadcast_to_pointwise` | `broadcast_to_pointwise` |

The mutants change `src/autograd/broadcast.bend`. The erased
`broadcast-PROOF.bend` would reject them too, so the source check runs
`src/autograd/broadcast-pointwise-PROOF.bend`, the live proof, and the kernel
artifact holds no erased `broadcast_to` proof. `run_lift` follows
`broadcast_to` through the planner, the target check and both allocations into
the loop, so a changed operation or operand order fails it. Another owner
passes the walk and fails the `kept` field of the contract.
`broadcast_to/entry.bend` imports only `broadcast.bend`, which declares no
template constant.

## Tensor VJP

`tensor_vjp/` holds:

- `consumers.bend`: five consumers, one for each of `tensor_add_vjp_values`,
  `tensor_sub_vjp_values`, `tensor_mul_vjp_values`,
  `tensor_matmul_vjp_values` and `tensor_matmul_nt_vjp_values`. The first three run the VJP of
  node 2 on a tape whose operands are node 0, of shape `[2]`, and node 1, of
  shape `[1]`, for the gradient `[3, 5]` of shape `[2]`, replayed as the
  writes of `Core.from_values`. It takes every plan from
  `Shape.broadcast_checked`, the plans that reduce a product from that
  product's own output shape, and states both targets, both dimensions and
  one cell of each contribution:
  - `add_values`: node 0 has the gradient's dimensions, so its cell 1 is
    `5.0` itself; node 1's cell is `F32.add(F32.add(0.0, 3.0), 5.0)`.
  - `sub_values`: node 0's cell 0 is `3.0`; node 1's cell is
    `F32.add(F32.add(0.0, F32.neg(3.0)), F32.neg(5.0))`, each word negated
    before it is added.
  - `mul_values`, with the saved operands `x = [2, 7]` and `y = [10]`: node
    0's cell 1 is `F32.mul(5.0, 10.0)`, from `g * y`; node 1's cell is
    `F32.add(F32.add(0.0, F32.mul(3.0, 2.0)), F32.mul(5.0, 7.0))`, from
    `g * x` summed.
  - `matmul_values` runs the VJP of node 2 = `a @ b` with `a` = node 0 of
    shape `[1, 2]` holding `[2, 7]`, `b` = node 1 of shape `[2, 2]` holding
    `[10, 20, 30, 50]`, and the gradient `[3, 5]` of shape `[1, 2]`. It takes
    the transpose plans from `Layout.permutation` and the two matmul plans
    from `Matrix.plan` on the transposed shapes. `b` is square, so reading it
    without the transpose gives other words. Node 0's cell `(0, 1)` is
    `F32.add(F32.add(0.0, F32.mul(3.0, 30.0)), F32.mul(5.0, 50.0))`, from
    `g @ b^T`; node 1's cell `(1, 0)` is `F32.add(0.0, F32.mul(7.0, 3.0))`,
    from `a^T @ g`. Both dimensions are stated as well.
  - `linear_values` runs the `MatmulNT` VJP of node 2 = `a @ b^T`, Linear's
    product with `b` in PyTorch's `[out, in]` layout, on the same tape. It
    takes the plan of `g @ b` from `Matrix.plan` and the plan of `g^T @ a`
    from `Matrix.product_plan` with two `Swapped` operands. Node 0's cell
    `(0, 1)` is `F32.add(F32.add(0.0, F32.mul(3.0, 20.0)), F32.mul(5.0, 50.0))`,
    from `g @ b` read in place, a column of `b` where the `Matmul` VJP reads a
    row; node 1's cell `(1, 0)` is `F32.add(0.0, F32.mul(5.0, 2.0))`, from
    `g^T @ a`.
- `controls.bend`: seven value witnesses, each paired with one wrong value.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `add_kept_witness` | `F32.add(0.0, 5.0)` for `5.0`, the gradient folded where nothing is reduced | the witness | the witness |
| False twin `add_sum_witness` | the two words added in reverse order | the witness | the witness |
| False twin `sub_witness` | the words added without their sign flipped | the witness | the witness |
| False twin `mul_witness` | `F32.mul(5.0, 7.0)`, `g * x` sent to node 0 | the witness | the witness |
| False twin `mul_sum_witness` | `g * y` summed for node 1 | the witness | the witness |
| False twin `matmul_front_witness` | `g @ b` instead of `g @ b^T`: `3.0 * 20.0` for `3.0 * 30.0` | the witness | the witness |
| False twin `matmul_back_witness` | `F32.mul(3.0, 7.0)`, the product's operands swapped | the witness | the witness |
| Add reduces to the other operand | `vjp_add` reduces the first contribution to the right operand's shape | `Laws.tensor_add_vjp_values` | `tensor_add_vjp_values` |
| Add sends the second to the first | `vjp_add` sends the second contribution to node `a` | `Laws.tensor_add_vjp_values` | `tensor_add_vjp_values` |
| Sub keeps the sign | `negated` runs `KeepLeft` instead of `NegateLeft` | `Laws.tensor_sub_vjp_values` | `tensor_sub_vjp_values` |
| Mul swaps operands | `vjp_mul` multiplies `g` by `x` for node `a` and by `y` for node `b` | `Laws.tensor_mul_vjp_values` | `tensor_mul_vjp_values` |
| Mul product swaps inputs | `multiplied_first` computes `x * g` for node `b` instead of `g * x` | `Laws.tensor_mul_vjp_values` | `tensor_mul_vjp_values` |
| Matmul left untransposed | `matmul_first` transposes `b` by `(0, 0)`, which keeps it as it is | `matmul_values` | `matmul_values` |
| Matmul right operands swapped | `right_product` multiplies `g @ a^T` instead of `a^T @ g` | `matmul_values` | `matmul_values` |
| Matmul sends the second to the first | `vjp_matmul` passes `a` for `b` | `matmul_values` | `matmul_values` |
| MatmulNT `matmul_tn` swaps operands | `nt_right` multiplies `x^T @ g` instead of `g^T @ x` | `nt_values` | `nt_values` |
| MatmulNT left reads the weight transposed | `nt_left` runs `matmul_nt(g, W)` instead of `matmul(g, W)` | `nt_values` | `nt_values` |
| MatmulNT sends the second to the first | `vjp_matmul_nt` passes `a` for `b` | `nt_values` | `nt_values` |

The mutants change `src/autograd/tensor-backward.bend`, and the source check
runs `src/autograd/PROOF.bend`. The structural laws in
`tensor-backward-LAWS.bend` say nothing about the values the VJPs send, so
only the value laws reject these mutants. The `Matmul` mutants fail in
`matmul_values`, the proof of `tensor_matmul_vjp_values` in
`tensor-vjp-PROOF.bend`: the first hands `transpose_replayed` a call that
law does not state, the other two produce a pass other than the stated one.
The `MatmulNT` mutants fail in `nt_values`, the proof of
`tensor_matmul_nt_vjp_values`. The runtime controls of the autograd gate
reject the `Mul` and `Sub` ones, a `Matmul` VJP without the transpose, and
the `MatmulNT` `matmul_tn` with swapped operands, as well, in the JavaScript
lane.

`permute_replayed`, `transpose_replayed` and `matmul_replayed` reuse the
walks of the copy loop and of the matmul loops. The permute value contract's
mutants in `verification/contracts/permute-values/` and the `matmul` mutants
of the fold family fail those walks, so they cover the transport laws too.
`tensor_vjp/entry.bend` imports only `tensor-backward.bend`.

## Binary

`binary/` holds the controls of `binary_replayed`, the export of the parallel
`Elementwise.binary` build:

- `consumers.bend`: four consumers on inputs replayed as the writes of
  `Core.from_values`, each taking its plan from `Shape.broadcast_checked`:
  - `mul_broadcast`: `[3, 5] * [10]`, shapes `[2]` and `[1]`: output cell 1
    is `F32.mul(5.0, 10.0)`, and the inputs come back with words `5.0` and
    `10.0`.
  - `sub_broadcast`: `[7] - [2, 3]`: output cell 1 is `F32.sub(7.0, 3.0)`.
  - `negate_scalar`: `NegateLeft` of `[3, 5]` against a rank-0 `+0.0`, as the
    `Sub` VJP runs it: output cell 0 is `F32.neg(3.0)`.
  - `binary_then_reduce`: the product of `mul_broadcast` reduced back to `[1]`
    through `reduce_to_fiber`, which applies because `binary_replayed` makes
    the product a replayed tensor:
    `F32.add(F32.add(0.0, F32.mul(3.0, 10.0)), F32.mul(5.0, 10.0))`.
- `controls.bend`: five value witnesses, each paired with one wrong value.
- `mutants.toml`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `mul_witness` | `F32.mul(10.0, 5.0)`, the operands swapped | the witness | the witness |
| False twin `sub_witness` | `F32.sub(3.0, 7.0)` | the witness | the witness |
| False twin `negate_witness` | `3.0`, the sign kept | the witness | the witness |
| False twin `kept_witness` | `3.0`, another word of the left input | the witness | the witness |
| False twin `chain_witness` | only the second product, from `+0.0` | the witness | the witness |
| High fork reuses the base | `Parallel.upper` returns `base`, so both halves build the low cells | `tree_eq` | `tree_eq` |
| Cell swaps operands | `cell_right` writes `scalar(operation, y, x)` | `cell_right_eq` | `cell_right_eq` |
| Cell swaps returned inputs | `cell_right` hands the input arrays back swapped | `cell_right_eq` | `cell_right_eq` |
| Padding not zero | a cell past the count holds `1.0` | `cell_eq` | `cell_eq` |
| Leaf writes zero | `Parallel.leaf` stores `0.0` instead of the cell's F32 | `cell_right_eq` | `cell_right_eq` |
| Pair writes zero | `Parallel.pair_join` stores `0.0` in the second leaf of a depth-1 node | `pair_join_eq` | `pair_join_eq` |

Three mutants change `src/tensor/elementwise.bend`, and three change
`src/tensor/parallel.bend`, the build that elementwise, matmul and reduce_to
share: there `upper`, `leaf` and `pair_join` keep the build balanced, so its
certificate still checks, and only the laws reject them. Every subtree of
depth 1 or more below the forks builds its bottom level with `pair_join`;
`leaf` builds a subtree of depth 0, a one-cell output or a branch of a build
that forks down to single cells. The generic law `parallel_build_cells`
(`src/tensor/parallel-LAWS.bend`) rejects all three too, in `node_cell`,
`leaf_facts` and `pair_join_facts` of `src/tensor/parallel-proof.bend`, for
every operation. The erased `elementwise-parallel-proof.bend` would reject most of
the mutants too, so the source check runs
`src/tensor/contracts/binary-PROOF.bend`, which imports only the live proof. The erased `BinaryModel` states only the cells below the count;
`binary_replayed` states the whole output storage, so its statement, and not
only its proof, excludes `padding-not-zero`. A change to `splits`, the number
of forks, keeps every word and is not a mutant: `build_eq` holds for any
number.
`binary/entry.bend` imports only `elementwise.bend`.

## Run

```sh
uv run --locked python -m verification.families.run --output <fresh directory>
uv run --locked python -m unittest verification/families/test_run.py -v
```

The runner snapshots `src/`, `pointwise/`, `fold/`, `reduce/`,
`broadcast_to/`, `tensor_vjp/` and `binary/`. For each family it
builds the false twins and the mutant copies, and runs the source checks,
`--verdict` on the consumers and the controls, the exports, and BendTT on the
positive and mutated exports. The families run at once and at most
`--workers` commands run at once; the default is 9/16 of the CPUs the process
may use (`os.sched_getaffinity`, which `taskset` narrows), 36 on 64 CPUs.
Each family writes its outputs to its own
subdirectory. `summary.json` records the repository and Bend commits, every
command, its exit code and time, and the rejection reason of every control.
