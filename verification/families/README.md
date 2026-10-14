# Contract family controls

The family laws in `src/tensor/contracts/LAWS.bend` are proved in
`src/tensor/contracts/PROOF.bend`, aggregated by `src/tensor/LAWS.bend`, and
required by `verification/s2.json`, so the root `PROOF.bend --verdict` and the
S2 gate protect them. [Contract families](../../docs/contract-families.md)
explains what they state. This directory holds literal consumers, false
twins and implementation mutants, beside the runner that checks them.

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
- `mutants.txt`: the before and after text of each mutation.

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

## Run

```sh
uv run --locked python -m verification.families.run --output <fresh directory> --workers 24
uv run --locked python -m unittest verification/families/test_run.py -v
```

The runner snapshots `src/` and `pointwise/`, builds the false twins and the
mutant copies, and runs the source checks, `--verdict` on the consumers and
the controls, the exports, and BendTT on the positive and mutated exports. It
writes `summary.json` with the repository and Bend commits, every command,
its exit code and time, and the rejection reason of every control. It takes
about four minutes with 24 workers and peaks near 4 GB per job.
