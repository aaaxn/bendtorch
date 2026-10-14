# Contract family controls

The family laws in `src/tensor/contracts/LAWS.bend` are proved in
`src/tensor/contracts/PROOF.bend`, aggregated by `src/tensor/LAWS.bend`, and
required by `verification/s2.json`, so the root `PROOF.bend --verdict` and the
S2 gate protect them. [Contract families](../../docs/contract-families.md)
explains what they state. This directory holds literal consumers, false
twins and implementation mutants, beside the runner that checks them.

## Pointwise

`pointwise/` holds:

- `consumers.bend`: eight consumers of `broadcast_pointwise`. Each one takes
  its plan from `Shape.broadcast_checked` and its inputs as the replayed
  tensors that `Core.from_values` returns, and restates literal values:
  - `[2, 3] - [3]` at output indices 0, 4 and 5.
  - `[2, 1] * [1, 3]`, an outer product, at indices 1 and 3.
  - `[3] + [3]` at index 2.
  - A rank-0 `KeepLeft` against `[3]` at index 2.
  - The empty `[2, 0] + [0]`, which states success and dimensions only.

  Each value consumer also states one word of each returned input. F32
  operations are opaque, so `[2, 3] - [3]` at index 4 is stated as
  `F32.sub(5.0, 20.0)`.
- `controls.bend`: four value witnesses, each paired with one wrong value.
- `mutants.txt`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `sub_witness` | `F32.sub(20.0, 5.0)`, operands swapped | the witness | the witness |
| False twin `outer_witness` | `F32.mul(2.0, 30.0)`, left row not broadcast | the witness | the witness |
| False twin `keep_witness` | `8.0` for a `7.0` fill | the witness | the witness |
| False twin `kept_witness` | left word 4 reads the output value | the witness | the witness |
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

- `consumers.bend`: seven consumers of `sum_ordered` and `dot_ordered`. Each
  one takes its inputs as the replayed tensors that `Core.from_values`
  returns, and restates literal values:
  - The sums of `[1, 2, 3]`, of a `[2, 3]` grid and of a rank-0 `[7]`.
  - The sum of the empty `[2, 0]`, which is `0.0`.
  - Cells `(1, 0)` and `(0, 1)` of the product of a `[2, 3]` and a `[3, 2]`
    matrix.
  - Cell `(1, 2)` of the product of a `[2, 0]` and a `[0, 3]` matrix, which is
    `0.0`.

  Each non-empty consumer also states one word of each returned input. A total
  is stated as the term the fold builds: the sum of `[1, 2, 3]` is
  `F32.add(F32.add(F32.add(0.0, 1.0), 2.0), 3.0)`.
- `controls.bend`: four value witnesses, each paired with one wrong value.
- `mutants.txt`: the before and after text of each mutation.

| Control | Change | Source rejection | BendTT rejection |
| --- | --- | --- | --- |
| False twin `sum_witness` | the words added in reverse order | the witness | the witness |
| False twin `start_witness` | `7.0` for `F32.add(0.0, 7.0)`, the fold starts at the first word | the witness | the witness |
| False twin `dot_witness` | `F32.mul(5.0, 20.0)`, the right matrix read along a row | the witness | the witness |
| False twin `kept_witness` | left word 5 reads the right input | the witness | the witness |
| Negative zero start | `run` starts at `F32.neg(0.0)` | `contract` | `contract.qvvvv` |
| Swapped accumulator | `step` folds `g(operation, x, accumulator)` | `step_lift` | `step_lift.qvvvv` |
| Dropped word | `step` keeps the accumulator | `step_lift` | `step_lift.qvvvv` |
| Cursor off by one | `run` starts the loop at 1 | `contract` | `contract.qvvvv` |
| Pair swapped operands | `pair_right` folds `g(operation, accumulator, y, x)` | `pair_step_lift` | `pair_step_lift.qvvvvv` |
| Pair wrong map | `pair_loop` reads the left input at `ib` | `pair_step_lift` | `pair_step_lift.qvvvvv` |
| Sum swapped add | `Sum.add` returns `F32.add(word, accumulator)` | `total_words` | `total_words` |
| Dot untransposed right | `Dot.right_index` reads `column * inner + k` | `total_factors` | `total_factors` |

The first six mutants change the generic loops in `fold.bend`, and the generic
proofs reject them. The last two change one instance each. The proof that
relates the instance's `Fold.total` to its specification rejects them.

`fold/entry.bend` imports `broadcast.bend` before `sum.bend` and `dot.bend`.
The BendTT export names each opaque template constant after the first module
that declares one of its type, and the positive export reaches `pointwise.bend`
before `fold.bend`. With the same import order, the mutant export names the
constants the same way, and each transplant replaces only the mutated
declarations.

## Run

```sh
uv run --locked python -m verification.families.run --output <fresh directory> --workers 24
uv run --locked python -m unittest verification/families/test_run.py -v
```

The runner snapshots `src/`, `pointwise/` and `fold/`. For each family it
builds the false twins and the mutant copies, and runs the source checks,
`--verdict` on the consumers and the controls, the exports, and BendTT on the
positive and mutated exports. Each family writes its outputs to its own
subdirectory. `summary.json` records the repository and Bend commits, every
command, its exit code and time, and the rejection reason of every control.
