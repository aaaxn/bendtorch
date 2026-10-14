# S2 elementwise projection refinement

The supplemental contracts in `src/tensor/projection-LAWS.bend` connect the
actual elementwise offset method to S1's independent coordinate
specification. They do not replace or weaken the existing Core or
Elementwise contracts.

## Independent specification

`offset_spec(plan, position)` first decodes the natural output position with
`ShapeLaws.fin_of(Shape.numel(output), position)`. It then uses
`ShapeLaws.project_left_coordinate` and `project_right_coordinate` to
unflatten that `Fin`, right-align the input ranks, and send singleton axes
to coordinate zero. `ShapeLaws.flatten_spec` flattens each projected
coordinate with `Shape.row_major` and converts its natural offset to `U32`.
`pair_offsets` returns both offsets only if both specifications succeed.

This specification does not call `Elementwise.input_indices`,
`Elementwise.flat_input_indices`, `Shape.index_map`, the flat projection
methods, or the executable stride-table walks. Its coordinate projections
are the stable S1 mathematical implementation, whose S1 laws relate them
to independent right-aligned natural coordinate lists.

## Added laws

- `actual_flat_projection` quantifies over every broadcast plan and every
  valid `Shape.FlatIndex` of its output. It states that the successful
  branch of the actual `Elementwise.flat_input_indices(plan, Done{index})`
  returns exactly the pair specified at that index's natural position.
  The proof obtains both S1 flat-projection refinements for the actual
  `Shape.index_map(plan)` and combines them into one pair refinement.
- `input_indices_refines` quantifies over every broadcast plan and every
  `U32` output index. Its actual-method result observer is `None` on
  failure and `Some{(left, right)}` on success. It equals `offset_spec` at
  `U32.to_nat(index)` in both cases, so it covers invalid and empty-output
  indices as well as successful offsets.
- `input_indices_acceptance` has the same unrestricted domain. It states
  that the actual offset method succeeds if and only if the natural
  output index is less than `Shape.numel(Shape.extents(output_shape))`.
  The cached executable `Shape.elements` decision is connected to that
  mathematical size by S1's checked exact-size bridge.
- `expected_projection_refines` quantifies over every operation, broadcast
  plan, pair of source tensors, and `U32` index. It states that the original
  `Elementwise.expected` equals `specified_expected` applied to the
  independently projected offsets. The latter reads the two checked
  source observations at those offsets and applies the operation's scalar
  function. It shares the downstream `Elementwise.expected_left` read and
  scalar interpretation, which takes already-chosen offsets and does not
  project them. It does not use the executable offset method.

There are no caller-provided frame, mapping, successful-conversion, rank,
capacity, or nonempty hypotheses in these public laws. The first law's
`FlatIndex` is a checked index type, not a restriction on the unrestricted
public `input_indices` laws. All existing validated shapes and broadcast
plans remain covered, including rank zero and output shapes containing a
zero extent.

## What closes the seam

The pre-existing elementwise mapping theorem uses `Elementwise.expected`,
which itself calls the actual projection method. On its own, that mapping
statement can remain true after an incorrect executable projection changes
both the computation and its expected value. The new total offset
refinement and `expected_projection_refines` connect that expected value
to a specification that does not share the executable projection. A
wrong constant, swapped offset, or altered stride walk therefore requires
a proof of equality to the unchanged coordinate specification; the current
proof cannot establish that equality by reflexivity.

## Trust boundaries

These contracts certify the source-level U32 offset pair and checked
source-read interpretation. They use S1's checked Nat/Fin coordinate
projection and row-major flatten proofs. The specification exposes the
same final `U32.from_nat` conversion as the stable S1 offset refinement;
the new public laws do not assume a conversion succeeds or hide an
unchecked output-index conversion. Output positions enter through
`U32.to_nat`, and `Shape.flat_checked` validates the natural bound before
the flat projections run. The new statements certify the outward `U32`
offsets; they do not separately assert a round-trip theorem for
`U32.to_nat(U32.from_nat(n))`. The natural input offsets are bounded by
the S1 flat projection's `FlatIndex` type, while the last Nat-to-U32
conversion is the same primitive used by the independent S1 flatten
specification.

All new definitions are specification observers or proof producers,
reached only by proof modules. No executable storage or tensor is cloned,
and no runtime implementation, S1 source, source pin, compiler, or kernel
is changed by this supplement. The existing elementwise loop proof and
constructor proof remain separate obligations; this module does not
claim allocator, C/JS compilation, IEEE-754 arithmetic, or backend runtime
correctness. The stage's native/JS corpus is still required.

The expected-value proof's source tensors are erased parameters of
`expected_indices_view`; its remaining observer applications occur inside
equality statements or the erased function argument of `Equal.cong`.
No live observation of source storage is introduced by that proof.

## Verification

The finalized actual runtime seam and the supplement passed the pinned
checker, `--verdict`, export, and a separate direct invocation of the
pinned BendTT kernel. The fresh evidence directory on the verification host is:

```text
<scratch>/s2-projection-certified-20261003T171833Z/
```

`report.json` records the four commands and their successful results.
`sources.json` records all imported Bend source hashes, verified unchanged
across the certification, and the required exported declarations. The
exported artifact is `projection.bendtt`. With
`BEND_NO_TELEMETRY=1` and `BENDTT=$PWD/.tools/kernel/bendtt`, the checks are:

```sh
bun .tools/bend/bend2/main.ts src/tensor/projection-PROOF.bend --check-only
bun .tools/bend/bend2/main.ts src/tensor/projection-PROOF.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/projection-PROOF.bend -o <fresh scratch>/projection.bendtt
.tools/kernel/bendtt <fresh scratch>/projection.bendtt
```

The exact law exports in that direct module artifact are:

```text
projection_2d_LAWS.actual_flat_projection
projection_2d_LAWS.input_indices_refines
projection_2d_LAWS.input_indices_acceptance
projection_2d_LAWS.expected_projection_refines
```

Importing the module through an aggregate changes its exported namespace;
the aggregate owner must record those resulting names in the stage
inventory and recertify that aggregate.

A copied-source negative test lives in
`<scratch>/s2-projection-mutation-20261003T171804Z/`. Its unmodified
baseline passes the pinned checker. Replacing only the actual
`flat_input_indices` successful branch with `Done{(0, 0)}`, while leaving
the independent specification and proofs unchanged, fails checking at
`Laws.actual_flat_projection` with exit code 1. The expected type contains
`Some{(0, 0)}`; the supplied S1 refinement contains the actual projected
offsets instead. `wrong-constant-check-only.stderr` records that mismatch.
The live worktree was not mutated.

Earlier native/JS screening is not proof evidence for these final source
files. No final aggregate or backend gate result is claimed by this scoped
module certification.
