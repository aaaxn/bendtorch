# Public proof interface and trust boundary

The proof-boundary experiment is qualified on a local Linux x86_64 host with
the current pinned Bend2 (`0e14aa8b`, the newest upstream main) and with the
previous pin (`947db722`). On the bytes in this branch, the source checker,
`--verdict` and a direct BendTT check of a fresh export agree on every positive
consumer. Four false expected-value twins and an aligned wrong-fill
implementation mutant are rejected for the intended value reason, and the
metadata-only control passes on the wrong-fill copy. The work stays in
`experiments/proof-boundary/`; no law, runtime file, pin, corpus or numerical
policy changed.

The proof-boundary experiment covers construction through
`Core.new(shape, fill)` and its first and retained reads. It does not cover the
rest of the tensor surface. The live contracts (`verification/contracts/`)
extend the same pattern operation by operation.

## Public producer and separate LIVE consumer

The capability is `generic-public.produce(shape, fill, index, inside)`. Its
inputs are a lawful `Shape.Shape`, an arbitrary F32 fill, a U32 index and the
ordinary in-bounds proof `Nat.is_lt(index, logical_numel(dimensions)) == True`.
The caller supplies no allocation, attachment or read success, no count
conversion, no retained-frame claim and no value conclusion.

`generic-public.expose` takes an ordinary `PublicRead` argument and returns an
ordinary `Claims` record. `generic-public.consume` calls `expose` on the
producer's result through a normal argument. `Claims` holds nine fields:

| Field | Statement about actual `Core.new(shape, fill)` |
| --- | --- |
| `successful` | `Result.is_done` is `True` |
| `dimensions` | the packed tensor's dimensions are `Shape.dimensions(shape)` |
| `count` | the natural storage length is `logical_numel(dimensions)` |
| `word_count` | `Core.count` returns `Shape.count(shape)` |
| `value` | `Core.read_value(Core.read_flat(tensor, index))` is `Some(fill)` |
| `retained_dimensions` | the owner returned by that read keeps the dimensions |
| `retained_count` | the returned owner keeps the natural length |
| `retained_word_count` | the returned owner keeps the word count |
| `retained_value` | for every in-bounds `other`, a second `read_flat` on the returned owner is `Some(fill)` |

The retention proof reuses the storage layer's replay-read and replay-size
evidence with one read, plus ordinary `Core` read-result projections. It does
not extract the erased `SourceRead`, `SourceReadKeeps` or construction witness.
No proof code runs inside a numerical tensor kernel. The record does not
establish one physical allocation, pointer identity, buffer separation or
native runtime behavior.

## Literal consumers

`consumers.bend` specializes the same producer and `expose` path. Each `Case`
restates the literal metadata, the requested `7.0` value at the primary index
and a retained read at a second index on the returned owner.

| Consumer | Expected metadata and values |
| --- | --- |
| `scalar` | dimensions `[]`, count 1, index 0 and retained index 0 read `7.0` |
| `three_zero` | dimensions `[3]`, count 3, index 0 and retained index 2 read `7.0` |
| `three_one` | dimensions `[3]`, count 3, index 1 and retained index 0 read `7.0` |
| `three_two` | dimensions `[3]`, count 3, index 2 and retained index 1 read `7.0` |
| `empty` | `Done`, dimensions `[0]`, natural and word count 0 |
| `huge_leading` | `Done`, dimensions `[0, 4294967295, 4294967295, 4294967295]`, count 0 |
| `huge_trailing` | `Done`, dimensions `[4294967295, 4294967295, 4294967295, 0]`, count 0 |

The empty and huge-zero consumers use the metadata producer without an index
premise. Trailing-zero algebra moves `logical_numel` to 0 without bounding any
extent or intermediate product. The older `public.bend`,
`huge-zero-last.bend` and `minimal-*.bend` probes stay as smoke history and are
not part of acceptance.

## Controls

`controls.bend` derives four public value witnesses
(`scalar_witness`, `three_zero_witness`, `three_one_witness`,
`three_two_witness`) from the producer. Each states
`Observation.point(shape, index, Core.new(shape, 7.0)) == expected_value()`,
with `expected_value() = Some(7.0)` and `wrong_value() = Some(8.0)`.

- **False twins (source).** Each twin changes only one witness's statement
  from `expected_value()` to `wrong_value()`. The checker reports
  `expected {Some{7.0} == Some{8.0}}`, `observed {Some{7.0} == Some{7.0}}` at
  that witness.
- **False twins (kernel).** The same change is made to the witness header in
  the fresh export. BendTT reports `expected … == wrong_value`,
  `observed … == expected_value` at that witness.
- **Wrong fill (source).** A copy of `src/` makes `Core.new` allocate `0.0`
  instead of `fill`, and changes only the body of
  `AllocationProof.core_new` to follow the new allocation (its statement is
  unchanged and still true, because fill does not change size). The checker
  rejects `generic-read.produce` at the public value field: the expected
  term contains `Storage.new_checked(…, 0.0, …)` where the observed term has
  `fill`, both against `Some{fill}`.
- **Wrong fill (kernel).** The export's `Core.new` and
  `AllocationProof.core_new` bodies replace the fill variable with the
  `F32.lit0` term. BendTT rejects `generic_2d_read.produce` with
  `expected … core.new +x0 x1 … == Some x1`, `observed … core.new_result … ==
  Some x1`.
- **Metadata-only control.** `metadata-control.bend` on the wrong-fill copy
  passes source, `--verdict`, export and direct kernel, so the mutant breaks
  only the value claim.

The runner accepts a rejection only when its diagnostic names the intended
value proposition. A quantity, parse, termination or resource failure is
refused. This matters in practice: replacing the fill with a reference to the
named constant `wrong_fill_value` instead of its body made the kernel reject
`core.new` with "affine live code, calls that descend", a termination
failure that is not value evidence. The runner substitutes the constant's body.

## Export fidelity

The runner checks the fresh export for the exact declarations it relies on:

- All seven consumers, the four witnesses, `expected_value`, `wrong_value`,
  `generic_2d_public.produce` and `consume` are present.
- No field in the seven evidence records (`Construction`, `Read`, `Retained`,
  `PublicRead`, `Claims`, `Case`, `EmptyCase`) is erased (`Σ-`).
- `produce` and `consume` take `+shape`, `+fill: F32`, `+index` and the
  in-bounds proof as an ordinary (`∀x3`) argument.
- Each witness states `observations.point` of `core.new` at
  `F32.lit1088421888` (7.0) against `expected_value`, and the export's
  `expected_value` and `wrong_value` are `Some(F32.lit1088421888)` (7.0) and
  `Some(F32.lit1090519040)` (8.0).

These declarations are written to `export-statements.txt` in the evidence
directory for review.

## Verification record

Run from the repository root with up to four CPUs:

```sh
uv run --locked python -m experiments.proof-boundary.run --output <fresh directory>
uv run --locked python -m experiments.proof-boundary.run --output <fresh directory> \
  --bend-root <bend main checkout> --kernel <bendtt built from its bendtt.lean>
```

The runner snapshots `src/` and `experiments/proof-boundary/`, writes the twins and
the wrong-fill copy, runs the source checks, `--verdict` and exports, checks
the export, writes the export-level negatives and runs BendTT on all of them.
It writes `summary.json` with the repository commit, Bend commit, each command,
exit code and duration, and each rejection reason.

Results on a local Linux x86_64 host (Lean 4.34.0, `LEAN_STACK_SIZE_KB=4194304`,
`LEAN_NUM_THREADS=1`). The current-pin run is at commit `8f8a48f` with
`--workers 11` on 28 CPUs; the only later commit edits this table. The
previous-pin run used the same Bend sources against the runner at `fa890c5`
(before the rebase onto the pin bump) with 4 workers on 4 CPUs.

| Step | Bend 0e14aa8b (current pin) | Bend 947db722 (previous pin) |
| --- | --- | --- |
| source: controls, metadata, wrong-fill metadata | exit 0 | exit 0 |
| source: four false twins | exit 1, value reason | exit 1, value reason |
| source: wrong fill | exit 1, value reason | exit 1, value reason |
| `--verdict` controls | exit 0, 92 s | exit 0, 98 s |
| `--verdict` wrong-fill metadata | exit 0, 90 s | exit 0, 98 s |
| export controls, export wrong-fill metadata | exit 0, 7 s each | exit 0, 11 to 12 s each |
| BendTT controls, wrong-fill metadata | exit 0, 87 s each | exit 0, 89 s each |
| BendTT four false twins | exit 1, value reason, 87 s each | exit 1, value reason, 88 to 89 s each |
| BendTT wrong fill | exit 1, value reason, 87 s | exit 1, value reason, 89 s |

The two Bend versions agree on every step. The newer export spells
`Maybe<&2, F32>` as `(Maybe -.Q2 -F32)` instead of `(Maybe.q2 -F32)`; the
runner accepts both spellings and nothing else. The whole current-pin run took
3 minutes of wall time. Peak RSS per process was 2.0 to 3.6 GB. The earlier
cloud verdict that was SIGKILLed did not reproduce on that host.

## Trust boundary

| Boundary | Status |
| --- | --- |
| Generic `Core.new` producer and LIVE consumers | Qualified: source, verdict, fresh export and direct kernel agree; value and wrong-fill controls rejected for the intended reason, on the pin and on Bend main. |
| Raw `Tensor`/`Storage` erased invariants | Not changed. The proof-boundary experiment follows actual construction. Arbitrary raw packages and LIVE versions of erased evidence are questions for the live contracts. |
| F32 and arrays | Opaque F32 semantics and primitive array lowering stay trusted. No IEEE algebra, native bits or heap separation claim. |
| Compiler, export, kernel and Lean | Trusted at the recorded commits. The controls show sensitivity for these statements; they do not prove verifier soundness. |
| Error ownership | Unchanged: elementwise failures retain operands; sum and matmul failures consume them. |
| Runtime, tensor layer, data | No tensor runtime file changed. No gate, native/JS corpus, oracle, GPU, training or benchmark ran for this PR. |

## Live transform contracts: what is LIVE and what is not

The live transform contract (`src/tensor/transform-contract-LAWS.bend`) states
LIVE laws for the parts of `reshape`, `permute_copy` and `transpose_copy` that
can be proved live today:

- `reshape` success on a `Core.new` producer: status 0, the requested
  dimensions, length and word count, every in-bounds value of the output read
  through `read_dynamic`, and a `write_flat` before the reshape showing up at
  the same linear index afterwards (linear order).
- `reshape` failure when the element counts differ: code 15
  (`ElementCountMismatch`). `reshape` consumes its input on failure, so the law
  states only the code.
- `permute_copy` failures for a wrong axis count (12), an axis out of bounds
  (13) and a duplicate axis (14), and `transpose_copy` with an axis out of
  bounds (13). Each states the code and that the source owner it returns keeps
  its dimensions, length, word count and every value.

The values a successful `permute_copy` or `transpose_copy` writes are **not**
LIVE. They stay under the existing laws (`permute_result_interpretation`,
`transpose_requested_result`, `permute_typed_success` and the copy-loop laws).
Those laws use erased evidence (`Transforms.Prefix`, `PermuteOutput`,
`Totality.*Success` hold `-` fields), and live code cannot consume erased
evidence. Making the value mapping LIVE needs the copy loop re-proved as live
code plus a LIVE distinct-values producer (`from_values`). That is tracked as a
separate follow-up; until it lands, a permute or transpose value claim is
checked by the existing erased laws and by the tensor native/JS corpus, not by
a LIVE contract.

The LIVE distinct-values producer for `from_values` is `from_values_contract`
(`src/tensor/from-values-contract-LAWS.bend`), which states, with LIVE fields,
that `Core.from_values(shape, values)` with `length(values) == numel(shape)`
succeeds with the shape's metadata and reads `values[i]` at every in-bounds
`i`.

The LIVE permute and transpose value contracts over this producer are
`permute_contract` and `transpose_contract`
(`src/tensor/permute-values-contract-LAWS.bend`). They state that output index
`i` holds `values[permute_source(input, axes, i)]` and that the returned source
keeps every value. The proof re-walks the copy loop with live code, as an
instance of a generic index-map copy built on the pointwise contract family.
The nine existing permute/transpose statements are unchanged and still hold
erased evidence.
