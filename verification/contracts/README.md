# Public proof contracts

These contracts turn the pattern of `experiments/proof-boundary/` into laws,
one capability per PR. Each capability states its public contract as a law in
`src/tensor/<capability>-contract-LAWS.bend` and proves it in the paired
`-PROOF.bend`, so the root `PROOF.bend --verdict` and the tensor gate protect
it. The literal consumers, false twins and implementation mutants live here,
beside the runner that checks them.

## Construction

Laws in `src/tensor/construction-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`:

- `new_metadata(shape, fill) : NewMetadata<shape, fill>`: actual
  `Core.new(shape, fill)` succeeds, and the packed tensor has the shape's
  dimensions, natural length and word count.
- `new_reads(shape, fill, index, inside) : NewReads<shape, fill, index>`: the
  same four fields, plus `read_flat` at `index` returns `Some(fill)`, the owner
  it returns keeps the dimensions, length and word count, and a second
  `read_flat` on that owner at any in-bounds `other` returns `Some(fill)`.

Every field of both records is LIVE (no `-` field), so a consumer that calls
the law gets ordinary evidence. The observation functions the fields use are in
the public module `src/tensor/observation.bend`.

`construction/` holds:

- `consumers.bend`: scalar, `[3]` at indices 0, 1, 2, empty, and both
  maximum-U32 huge-zero shapes, each restating literal metadata and values.
- `controls.bend`: four value witnesses, `expected_value` (7.0),
  `wrong_value` (8.0) and `wrong_fill_value` (0.0).
- `metadata-control.bend`: metadata-only consumer used on the wrong-fill copy.

The runner adds three kinds of control:

| Control | Change | Expected rejection |
| --- | --- | --- |
| False twins (4) | one witness statement says `wrong_value` | `Some(7.0) == Some(8.0)` at that witness, source and kernel |
| Wrong fill | `Core.new` allocates `0.0`; only the body of the true count proof follows | `point_value`: `0.0` vs `fill` against `Some(fill)`, source and kernel; metadata control still passes |
| Unjustified failure | `Core.new_result` returns `Fail{StorageLengthMismatch}` on a successful allocation | `new_result_length`: `None == Some(numel)`, source and kernel |

## Reads and writes

Laws in `src/tensor/read-write-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`. The record types
are in `src/tensor/read-write-records.bend`; the proofs are split into
`read-contract.bend` and `write-contract.bend` so a read-only control does not
import the write proofs.

Each law is generic over a `Core.new(shape, fill)` producer and talks about the
actual public call on its result:

- `read_inside(shape, fill, index, inside) : ReadInside`: `read_flat` succeeds
  (error code 0) and returns `Some(fill)`; the owner it returns keeps the
  dimensions, length and word count, and a second `read_flat` on that owner
  at any in-bounds `other` returns `Some(fill)`.
- `read_outside(shape, fill, index, outside) : ReadContract<.., 8>`: `read_flat`
  fails with `OutOfBounds` (code 8) and the returned owner is unchanged:
  same metadata, and every in-bounds read still returns `fill`.
- `write_inside(shape, fill, index, value, inside) : WriteInside`: `write_flat`
  succeeds, the returned owner keeps its metadata, reads `value` at `index`,
  and reads `fill` at every other in-bounds index.
- `write_outside(shape, fill, index, value, outside) : WriteOutside`:
  `write_flat` fails with code 8 and the returned owner still reads `fill`
  everywhere.
- `coordinate_inside(shape, fill, coordinates, index, value, located, inside)`:
  when `Shape.flatten_checked(shape, coordinates)` locates `index`,
  `read_coordinate` after a flat write at `index` reads `value`, and
  `write_coordinate` writes `index` and leaves every other index at `fill`.
- `dynamic_read_inside(shape, fill, index, inside)`: `read_dynamic` on the
  packed tensor returns `Some(fill)`.

Both failure laws state the retained-owner contract: read and write give their
operand back on failure.

`reads/` holds:

- `consumers.bend`: scalar, `[3]` at every index, `[2,3]` at its last flat
  index, writes at every `[3]` index with a neighbour read, the four `[2,3]`
  corners through coordinates, out-of-bounds reads and writes on `[3]` and at
  `4294967295` on a scalar, the empty `[0]` tensor, and one dynamic read.
- `controls.bend`: four value witnesses on `[3]` (read at 1, written value at
  1, neighbours 0 and 2 after a write at 1), `expected_fill` (7.0),
  `expected_written` (9.0) and `wrong_value` (8.0).
- `read-control.bend`: read-only consumer used on the write mutants.
- `mutants.toml`: the before/after text of each `core.bend` mutation.

| Control | Change in `core.bend` | Expected rejection |
| --- | --- | --- |
| False twins (4) | one witness statement says `wrong_value` | `Some(v) == Some(8.0)` at that witness, source and kernel |
| Wrong index | `write_flat` writes `i + 1` | `write_value_done`, source and kernel; read control passes |
| Missing write | `write_flat` returns the tensor unwritten with `Done` | `write_value_done`, source and kernel; read control passes |
| Corrupted neighbour | `write_flat` also writes `0.0` at `i + 1` | `write_value_done`, source and kernel; read control passes |
| Write unjustified failure | a successful storage write reports `Fail` | `write_inside`: `WriteContract<.., 0>` vs `<.., 8>`, source and kernel; read control passes |
| Read unjustified failure | a successful storage read reports `Fail` | `read_result_value`: `None` vs `Some(value)`, source and kernel |

The kernel mutants are built by exporting the mutant `core.bend` alone and
transplanting its changed and new declarations into the positive export, so the
kernel rechecks the unchanged proofs against the mutated operation.

## Clone and metadata

Laws in `src/tensor/clone-metadata-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`. The record types
are in `src/tensor/clone-metadata-records.bend`; the proofs are in
`clone-contract.bend` and `metadata-contract.bend`, each a separate import so a
clone mutant leaves the metadata proof checking and the reverse.

Each law is generic over a `Core.new(shape, fill)` producer:

- `clone_contract(shape, fill) : CloneContract`: both owners `Core.clone`
  returns have the producer's dimensions, length and word count, and a
  `read_flat` on either owner at any in-bounds index returns `Some(fill)`.
- `clone_write(shape, fill, index, value, inside) : CloneWrite`: after a
  `write_flat` of `value` at `index` on the first owner, that owner reads
  `value` at `index` and `fill` at every other in-bounds index, and the second
  owner reads `fill` at every in-bounds index, including `index`. This is the
  aliasing control AD needs when it fans a value out.
- `metadata_contract(shape, fill) : MetadataContract`: `Core.length` returns
  `numel`; the owner it returns keeps dimensions, length, word count and every
  value; `Core.shape` and `Core.count` on the packed tensor return the
  dimensions and word count, and a `read_dynamic` on the owner each of them
  returns reads `fill`.

`clone/` holds:

- `consumers.bend`: clone of a scalar, of `[3]` (last index) and of `[2,3]`
  (first and last index), clone of the empty `[0]`; clone then write on `[3]`
  at 1, on `[2,3]` at 5 and on a scalar; `Core.length`, `Core.shape` and
  `Core.count` on a scalar, `[2,3]` and `[0]`.
- `controls.bend`: four witnesses on `[3]` (second owner at 2, written value
  at 1 on the first owner, second owner at 1 after that write, length owner at
  0), with `expected_fill` (7.0), `expected_written` (9.0) and `wrong_value`
  (8.0).
- `mutants.toml`: the before/after text of each `core.bend` mutation.

| Control | Change in `core.bend` | Expected rejection |
| --- | --- | --- |
| False twins (3) | one witness statement says `wrong_value` | `Some(v) == Some(8.0)` at that witness, source and kernel |
| Aliasing twin | the second owner after a write on the first is claimed to read 9.0 | `Some(7.0) == Some(9.0)` at `alias_one_witness`, source and kernel |
| Bad clone (first owner drops last element) | `clone` writes `0.0` at the last index of the first owner | `first_value_done`, source and kernel; metadata proof passes |
| Bad clone (second owner zeroed) | `clone_result` writes `0.0` at index 0 of the second owner | `second_value_projection`, source and kernel; metadata proof passes |
| Wrong length | `length_result` returns `numel + 1` | `number_projection`, source and kernel; clone proof passes |
| Changed owner | `length_result` writes `0.0` at index 0 of the owner it returns | `owner_value_projection`, source and kernel; clone proof passes |
| Wrong word count | `Core.count` returns `count + 1` | `read_count_pair` (the first proof that states the count), source and kernel |

## Transforms

Laws in `src/tensor/transform-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`. Record types
are in `src/tensor/transform-records.bend` and observations in
`src/tensor/transform-observation.bend`. The proofs are in
`reshape-contract.bend` and `permute-contract.bend`, separate imports so a
reshape mutant leaves the permute proof checking and the reverse.

Each law is generic over a `Core.new(input, fill)` producer:

- `reshape_contract(input, fill, target, same) : ReshapeContract`: when the
  element counts agree, `reshape` to `target`'s dimensions succeeds (code 0),
  the output has those dimensions, length and word count, and every in-bounds
  `read_dynamic` on it returns `Some(fill)`.
- `reshape_write(input, fill, target, index, value, inside, same) : ReshapeWrite`:
  after `write_flat(value)` at `index`, the reshaped output reads `value` at
  the same linear `index` and `fill` at every other in-bounds index.
- `reshape_count_mismatch(input, fill, target, different)`: code 15. `reshape`
  consumes its input on failure, so nothing else is returned.
- `permute_rank_mismatch` (12), `permute_axis_out_of_bounds` (13),
  `permute_duplicate_axis` (14) and `transpose_out_of_bounds` (13) :
  `RetainedFailure`: the error code, and the source owner returned with it
  keeps its dimensions, length, word count and every value.

The values a successful permute or transpose writes are not covered here; see
`experiments/proof-boundary/BOUNDARY.md`.

`transforms/` holds:

- `consumers.bend`: `[2,3] -> [3,2]`, `[2,3] -> [6]`, scalar `-> [1]`,
  `[0] -> [2,0]`; writes at 1 and 5 before a reshape; count mismatches
  `[2,3] -> [2,2]` and `[0] -> [1]`; permute `[2,3]` by `[0]`, `[2,3,4]` by
  `[2,0,3]` and `[2,0,2]`; transpose `[2,3]` by `(0,2)` and a scalar by `(1,0)`.
- `controls.bend`: four witnesses (reshaped value, written value after a
  reshape, retained value after a duplicate-axis permute and after an
  out-of-bounds transpose).
- `mutants.toml`: before/after text of each mutation, with the file it edits.

| Control | Change | Expected rejection |
| --- | --- | --- |
| False twins (4) | one witness statement says `wrong_value` | `Some(v) == Some(8.0)` at that witness, source and kernel |
| Reshape changes values | `reshape_equal` writes `0.0` at index 0 of the output | `decided_lift`, source and kernel; permute proof passes |
| Reshape unjustified failure | `reshape_decided` returns `ElementCountMismatch` when the counts agree | `decided_lift`, source and kernel; permute proof passes |
| Reshape keeps input dimensions | `reshape_built` reshapes to the input shape | `built_lift`, source and kernel; permute proof passes |
| Permute failure changes source | `copy_planned` writes `0.0` into the source it returns on failure | `planned_failure`, source and kernel; reshape proof passes |
| Transpose failure changes source | `transpose_decided` writes `0.0` into the source it returns on failure | `transpose_failure`, source and kernel; reshape proof passes |
| Wrong permute error | `plan_bounded` reports `DuplicateAxis` for an out-of-bounds axis | `bounded_failure`, source and kernel; reshape proof passes |

The kernel lane exports the mutated module alone (`transforms.bend` or
`layout.bend`) and transplants its changed declarations into the positive
export.

## From values

Law in `src/tensor/from-values-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`. The record type
is in `src/tensor/from-values-records.bend` and the proof in
`src/tensor/from-values-contract.bend`.

- `from_values_contract(shape, values, counted) : FromValuesContract`: when
  `length(values) == numel(shape)`, `Core.from_values(shape, values)` succeeds,
  has the shape's dimensions, length and word count, and `read_flat` at every
  in-bounds `index` returns `List.get(values, index)`.

Every field is LIVE. The proof follows the fill loop (`fill_values`,
`fill_values_decided`, `fill_values_step`) with live code: after `k` steps the
storage is the zero-filled allocation replayed through the writes
`Write{i, values[i]}` for `i < k`, so the storage replay laws give the value at
each index. The existing `public_from_values_elements` and
`public_from_values_complete` laws are unchanged.

`from-values/` holds:

- `consumers.bend`: a scalar `[5.0]`, `[3]` from `[1, 2, 3]` at every index,
  `[2,3]` from `[1..6]` at flat indices 0, 3 and 5, and the empty `[0]`.
- `controls.bend`: three value witnesses (`[3]` at 0 and 1, `[2,3]` at 5),
  with `expected_first` (1.0), `expected_second` (2.0), `expected_last` (6.0)
  and `wrong_value` (8.0).
- `mutants.toml`: before/after text of each `core.bend` mutation.

| Control | Change in `core.bend` | Expected rejection |
| --- | --- | --- |
| False twins (3) | one witness statement says `wrong_value` | `Some(v) == Some(8.0)` at that witness, source and kernel |
| Wrong fill (WF) | `fill_values_decided` writes `0.0` instead of the value | `decided_step`, source and kernel; construction proof passes |
| Wrong index (WI) | `from_values_new` starts the cursor at 1 | `counted_lift`, source and kernel; construction proof passes |
| Missing write (MW) | `fill_values` skips the write of the last value | `fill_lift`, source and kernel; construction proof passes |

The "construction proof passes" check runs `construction-contract-PROOF.bend`
on each mutated copy: the `Core.new` laws do not depend on the fill loop. The
kernel lane exports the mutated `core.bend` alone and transplants its changed
declarations into the positive export.

## Permute and transpose values

Laws in `src/tensor/permute-values-contract-LAWS.bend`, aggregated by
`src/tensor/LAWS.bend` and required by `verification/tensor.json`. The record
types are in `src/tensor/permute-values-records.bend` and the proof in
`src/tensor/permute-values.bend`.

- `permute_contract(input, values, counted, axes, valid) : PermuteContract`:
  for `Core.from_values(input, values)` with `length(values) == numel(input)`
  and a valid permutation `axes`, `permute_copy` succeeds (code 0), the output
  has dimensions `Axis.order(dims, axes)`, the input's length and word count,
  output index `i` holds `values[permute_source(input, axes, i)]` for every
  in-bounds `i`, and the returned source still holds `values[j]` at every
  in-bounds `j`.
- `transpose_contract(input, values, counted, a, b, inside) : TransposeContract`:
  the same fields for `transpose_copy(a, b)` with both axes in range, with the
  axes `Transpose.requested_axes(rank, a, b)` that swap `a` and `b`.

`Values.permute_source(input, axes, i)` in `src/tensor/observation.bend` is
the reference map: decode `i` against the output dimensions, scatter the
output coordinate back through `axes`, and flatten it against the input
dimensions. Every field is LIVE.

The proof is generic over the index map. `src/tensor/index-map.bend` states
the copy `out[i] = input[map(plan, i)]` as the pointwise family of
`src/tensor/contracts/` with a scalar function that keeps its first operand,
so the write history and the final-value theorem
(`PointwiseProof.writes_value`) are the family's. Permute supplies the map
`permute_source` and discharges its bound from the `Layout.Mapping` of each
output index; transpose reuses the permute proof with the swapped axes. The
copy itself is the parallel build of `src/tensor/permute-build.bend`, and
`permute-values.bend` walks it with live code: the source comes back as the
First half of its clone, and the build's output, cell by cell, is the
family's write history replayed on the storage `Core.new` allocates
(`made`, from `build_eq` in `permute-build-proof.bend`).

`permute-values/` holds:

- `consumers.bend`: a scalar permute by `[]`; `[2,3]` from `[1..6]` permuted
  by `[1,0]` at output indices 0, 1 and 4; `[2,2,2]` from `[1..8]` rotated by
  `[1,2,0]` at output indices 1, 2 and 6; transposes of `[2,3]` by `(0,1)` and
  `(1,0)`, of `[2,2,2]` by `(0,2)` and of a scalar by `(0,0)`. Each restates
  the code, dimensions, one output value and one retained source value.
- `controls.bend`: three witnesses (rotated value 5.0, transposed value 4.0,
  retained source value 3.0) and `wrong_value` (8.0).
- `mutants.toml`: before/after text of each mutation, with the file it edits.

| Control | Change | Expected rejection |
| --- | --- | --- |
| False twins (3) | one witness statement says `wrong_value` | `Some(v) == Some(8.0)` at that witness, source and kernel |
| Forward map (WI) | `recipe` in `permute-build.bend` builds the inverse map, so output index `map(i)` gets source word `i` | `source_index` in `permute-build-proof.bend`, source and kernel; the released transform contract passes |
| Identity map (WI) | `Build.at` in `permute-build.bend` reads the source at the output index itself | `source_word` in `permute-build-proof.bend`, source and kernel; the released transform contract passes |
| Swap ignored (WI) | `transpose_decided` permutes by `transpose_axes(dims, a, a)` | `transpose_route`, source and kernel; the released transform contract passes |

The three WI mutants keep every existing law: the erased value laws do not
reach the kernel, and `transform-contract-PROOF.bend`, which states failures
and reshape values only, still checks on each mutated copy. The rotation
consumer uses a 3-cycle because a swap of two axes is its own inverse, so the
forward map reads the same words as the permute on a swap. On `[2,2,2]` by
`(2,0,1)` the permute reads source indices `0 2 4 6 1 3 5 7`, the forward map
`0 4 1 5 2 6 3 7` and the identity map `0 1 2 3 4 5 6 7`. The kernel lane exports the mutated module
alone (`permute-build.bend` or `transforms.bend`) and transplants its changed
declarations into the positive export.

## Run

```sh
uv run --locked python -m verification.contracts.run --output <fresh directory>
uv run --locked python -m unittest verification/contracts/test_run.py -v
```

The runner snapshots `src/`, `construction/`, `reads/`, `clone/`, `transforms/`, `from-values/` and `permute-values/`, builds the controls and
mutant copies, runs source checks, `--verdict`, exports and checks the exported
statements, then runs BendTT on the positive and mutated exports. The six
sections (construction, reads, clone, transforms, from-values, permute-values)
run at once; a section makes its false twins and mutant copies only once the
section before it has made its own (it signals just before its first checks), so every copy holds
the same files as in a one-section-at-a-time run. At most `--workers` commands run at once; the
default is 5/16 of the CPUs the process may use (`os.sched_getaffinity`, which
`taskset` narrows), 20 on 64 CPUs. It writes
`summary.json` with the repository and Bend commits, every command, exit code
and duration, and each rejection reason. It accepts a rejection only when the
diagnostic names the intended proposition.
