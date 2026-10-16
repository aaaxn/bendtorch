# Owned tensor transforms

The scalar `(0,0)` specification correction is described in
[scalar-transpose.md](scalar-transpose.md). Historical verification results
below belong to the original tensor contract; they do not certify the
correction.

The transform contract is stated before the implementation. Reshape consumes a
contiguous tensor and transfers its sole storage owner to a checked target
shape with equal natural logical element count. It neither aliases nor
reorders that storage. Shape construction and natural equality checks precede
the transfer; a rank, capacity, or product mismatch is an explicit error.

Permutation axes name input dimensions in output order. A checked plan requires
the axes to have the input rank, each axis to lie below that rank, and all axes
to be distinct. Thus rank-zero uses the empty permutation and rank-four is the
maximum. The output dimensions are selected in the specified order, constructed
through the frozen shape API, and checked to retain the natural element count.
Every Boolean dispatch carries equality to the predicate it decides.

`permute_copy` and `transpose_copy` retain the input owner and allocate separate
output storage. They do not use shared views or silently default invalid
coordinates. Empty outputs skip the copy entirely, including huge-zero
dimensions.

The copy runs every output cell at once, as `Parallel.build` with the cell
function of `src/tensor/permute-build.bend`: output cell `i` reads the input
at `Values.permute_source_index(input, axes, i)`, the index the checked copy
loop below reads, computed from a recipe of strides worked out once per
build instead of three coordinate lists per cell
(`permute-build-proof.bend` proves the two indexes equal). The input comes
back as the First half of its clone and the build reads the Second half.
On a `[1024,1024]` transpose this takes 98 ms on one thread and 36 ms on
eight, against 1.7 s and 2.4 s for the loop; the words are the same at every
thread count.

The laws below about the loop (`copy_loop`, `natural_step`, `copy_step`,
`copy_read`) state what the checked sequential copy does; `copy_allocated`
no longer runs it. `permute_result_interpretation`,
`permute_typed_success` and the other public laws are proved for the parallel
copy: `permute-build-proof.bend` shows that it meets the loop's own meaning
(`LoopMeaning`, the successful prefix of the whole output) on every source.

For each output flat index the loop obtains a checked output coordinate,
scatters it back to the original axes, obtains a checked input flat index,
reads the input through the owned storage adapter, and writes the output.

The copy loop uses natural logical-numel fuel and a natural cursor. At each
step it converts the cursor to U32 and checks that converting that candidate
back to a natural number equals the cursor. A mismatch is `MappingError`, not
an assumed bounded-word identity or a silently wrapped destination. The exact
word/cursor equality is erased. The shape domain remains ranks zero through
four and at most `2^20` logical elements.

On the reviewed shape layer the executable count comes from `Shape.elements`,
whose erased `elements_exact` certificate connects it to natural logical numel.
Runtime reshape and permutation count decisions also use that stored size, so a
huge-zero shape does not eagerly recompute a large intermediate natural
product. Layout keeps local checked word-coordinate/index witnesses; these are
metadata witnesses, not tensor storage or array traversals. With the user's
explicit approval, the old U32 Horner/divmod codec has been replaced by the
frozen shape natural codec. `flatten_words` now computes `Shape.horner` and
converts its natural result to U32; `unflatten_words` uses `Shape.digits_words`
and `Shape.words`. No old/new universal equivalence is claimed. The
word-coordinate law statements are unchanged, but `mapped_flat` and
`mapped_output` now refer to these new definitions. Definitions that
transitively call these codecs also have this dependency-meaning change.

Proofs concern validated metadata, the inverse element map, ownership, and the
source storage observation/write refinement. F32 values are copied as opaque
values, with no floating-point ring laws or native IEEE correctness claim.
Certificate builders and mathematical array traversals are proof-only and
erased at executable storage call sites. Live array access uses only the frozen
Base-backed storage adapter. Compiler lowering, allocation and hardware remain
trusted boundaries.

## Public signatures

`reshape(shape, tensor, dimensions)` consumes the input and returns a checked
dynamic tensor or an error. `permute_copy(shape, tensor, axes)` and
`transpose_copy(shape, tensor, a, b)` return the retained input tensor alongside
an output/error result. Dynamic wrappers are `reshape_dynamic`,
`permute_dynamic`, and `transpose_dynamic`. Axes are zero-based U32 values.
Transpose accepts existing axes at positive rank and `(0,0)` at rank zero.
Scalar permutation still uses `[]`. A scalar transpose materializes the same
shape and value while returning the retained input owner. Any positive scalar
axis is rejected.

Errors delegate shape and core errors. Transform-specific codes are wrong axis
list length 12, out-of-range axis 13, duplicate axis 14, unequal logical numel
15, and invalid mapping/conversion 16. The input is retained on copy errors as
well as success; a partially filled output is dropped on error.

## Checked individual, success and inverse contracts

`transform-LAWS.bend` and `transform-PROOF.bend` prove accepted permutation
predicates, selected output dimensions, equal natural counts, exact output
decoding and inverse coordinates, source-index bounds, checked coordinate
roundtrip, and reshape's unchanged linear observations. The actual copy-step
producer consumes the normal mapping, read and write results together with
their exact computational equalities. It obtains the independent source
meaning from `Core.source_read_model` and the frame from the frozen Core
checked-update model, rather than accepting either as a caller premise.

The successful-prefix invariant retains the original source observations,
checked encodings and output routes for prior natural positions, and each
prior output's independent inverse-mapped source observation. The concrete
step extends that prefix, and natural-fuel induction certifies the actual
whole `copy_loop`. `whole_copy_interpretation` constructs the empty prefix
internally and uses the erased cached-count bridge to quantify the final
prefix over the mathematical logical count. It needs no mapping, frame,
initial-prefix or cursor-conversion premise from its caller.

`permute_result_interpretation` follows the actual plan, allocation, parallel
copy and packing computations. A successful result carries the selected output
dimensions, retained original source observations, exact natural-to-word
cursors, checked output routes and every in-range final observation. Those
observations are proved both for `Core.checked_observe_dynamic` and for the
independent `Core.observe` projection through `dynamic_observe`.
`transpose_result_interpretation` applies the same result guarantee to the
actual public axis-validity dispatch.

`mapping_reference` connects every returned mapping's actual source index to
the independent axis specification's `Axis.scatter`, frozen natural
`Shape.digits` and natural row-major `Shape.offset`. The reference does not
call the implementation's `inverse_coordinates`, `coordinate_for`,
`flatten_words`, `unflatten_words` or `map_index`. The proof uses the separately
certified axis-algebra bridge, not an assumed caller mapping equality.

`codec_flatten_decode` proves that decoding and then flattening the actual new
codec restores every U32 index below the shape's mathematical element count.
It derives digit fitting from the shape layer's finite coordinate, uses the
fitting-list word conversion theorem, and restores the original U32 through the
certified word/natural inverse. `natural_word_exact` proves the general
natural-to-word identity when a supplied U32 limit bounds the natural value.
Its bounded-limit hypothesis is not a closed-capacity bridge for all tensor
shapes. The separately certified `WordConversion.shape_count_exact` now
supplies that exact bridge over every lawful shape on the unchanged domain.

The proof module also derives actual `Core.read_flat` success for every lawful
tensor and mathematically in-range U32 index. It derives route success through
`Route.index_complete`, present observation through primitive read refinement,
and rejects a failed read through the frozen source-read model. These producers
do not accept route equality, read success or primitive refinement from callers.

`reshape_result_interpretation` follows actual shape construction and cached
count dispatch, proving the requested dimensions, equal mathematical count
and unchanged linear observations for every U32 index. The actual second
`reshape_dynamic` back to the original dimensions is covered by
`reshape_inverse_observations`: the final dimensions and every `Core.observe`
value are the original ones on success. `reshape_valid_success` now proves that
the actual first call must succeed for any lawful target shape with the same
mathematical count. `reshape_inverse_valid_success` proves that both the first
call and the actual second call back to the original dimensions must succeed
under that same valid-input condition. Neither accepts construction, cached
count conversion or intermediate success from callers.

The original 21 law statements are retained verbatim. The eleven added laws
are `whole_copy_interpretation`, `permute_result_interpretation`,
`transpose_result_interpretation`, `reshape_result_interpretation` and
`reshape_inverse_observations`, plus `mapping_reference`, `natural_word_exact`,
`codec_flatten_decode`, `reshape_valid_success` and
`reshape_inverse_valid_success` and `mapped_mathematical_bounded`. The last
law refines returned FIndex source bounds to mathematical logical-numel bounds
using the general truncation inequality; it does not assert count exactness.
The codec definitions called by the preserved
mapping laws changed as disclosed above; `reference_source` uses the new
independent axis specification. All result models are Data with erased semantic fields; the
computational owner/pair certificates remain Type.

The proof-only totality extension adds ten further laws without changing any
of the preceding 32 statements or any executable definition:

- `permutation_valid_success` proves that actual `Layout.permutation` must
  return `Done` under the existing `Layout.bijective` predicate. Independent
  axis completeness, rank and product preservation derive the shape
  construction bounds internally, including zero-product and huge-zero
  dimensions.
- `tensor_read_valid_success` and `tensor_write_valid_success` prove actual
  `Core.read_flat` and `Core.write_flat` must return `Done` for a lawful owner
  and a mathematically in-range U32 index. Storage bounds and route completeness
  are derived internally; callers supply neither route nor intermediate
  success certificates.
- `decoded_words_valid_success` proves the actual `Layout.words_check` of the
  actual decoded coordinate list must return `Done` for an in-range index. The
  shape coordinate construction and the certified per-extent conversion theorem
  supply coordinate validity without a closed-capacity conversion premise.
- `permute_axes_length_rejection`, `permute_axis_bounds_rejection` and
  `permute_duplicate_axes_rejection` prove the exact errors of the actual
  public permutation call and exact retained-source identity. The premises
  follow dispatcher priority: axis bounds are checked only after rank, and
  duplicates only after rank and bounds.
- `transpose_axis_bounds_rejection` proves the actual public transpose returns
  the exact axis-bounds error and the unchanged source owner.
- `reshape_element_count_rejection` proves the actual public reshape rejects
  unequal mathematical counts of lawful shapes with the exact count-mismatch
  error, deriving successful target construction internally.
- `mapping_index_rejection` proves the actual mapping call returns the exact
  flat-index-bounds error when its existing U32 index-validity predicate fails.

The 42-law checkpoint remains immutable. The current proof-only extension
adds fifteen more laws; all 32 preceding statement blocks and every executable
definition remain unchanged. The erased models and projections live in
`transform-totality.bend`, which no executable module imports.

`codec_decode_flatten` supplies the other actual codec direction for every
fitting word coordinate. `mapping_valid_success` then proves mandatory actual
mapping success, deriving inverse fitting, exact source-offset conversion and
coordinate restoration internally. `natural_cursor_exact_conversion` includes
the logical count endpoint. `copy_step_valid_success`,
`natural_step_valid_success` and `copy_loop_valid_success` compose actual
mapping, read, write and conversion results rather than assuming successful
intermediates. The loop's only progress premise is its remaining logical range;
the public producer supplies the full count and zero cursor itself.

`permute_valid_success` consumes the independently certified
`AllocationProof.core_new_done` producer, so lawful permutation callers provide
only a tensor and the existing bijective-axis predicate. `permute_typed_success`
strengthens that result to an erased actual returned-plan, retained-owner and
typed-target witness, exact equality to the public pair, and the complete
source-preserving observation prefix. No allocation, initial-prefix, route,
mapping or intermediate-success premise is exposed to public callers.

`inverse_axes_valid` derives the native inverse request from independent
`Axis.inverse_axes`, with exact conversion of each bounded natural axis.
`inverse_permutation_dimensions` restores the original dimension list,
including zero-product shapes. `inverse_mapping_restores_index` composes two
actual returned mappings: exact decode-after-flatten joins their coordinates,
the inverse request cancels the axis ordering, and exact flatten-after-decode
restores the original U32 index. `permute_inverse_restoration` invokes the actual
first public call and actual second `permute_dynamic` call with that independent
inverse request. Its erased witness contains both successful result equalities,
the actual retained/intermediate/restored owners, restored dimensions and
`Core.observe` equality at every mathematical logical index. It neither
accepts a caller-supplied intermediate nor concludes restoration only
conditionally on the inverse succeeding. Raw physical padding observations are
outside this logical-index contract.

`transpose_valid_success` states mandatory actual public transpose success from
the transpose-specific validity predicate, including scalar `(0,0)`.
`transpose_requested_result` supplies
the actual returned owner and typed target, selected dimensions, full prefix,
independent requested source map and every logical-index `Core.observe` value.
The map uses the separately certified natural-position swap specification;
neither the dimension nor element contract calls `Layout.transpose_axes` as its
reference. The erased output plan is linked to the actual public pair, and
mapping/read/write success is derived internally, not requested from callers.

`transpose_inverse_restoration` proves that the actual second public
`transpose_dynamic` call with the same requested `a` and `b` must succeed and
restore the original dimensions and every mathematical logical observation.
Preserved rank derives validity for that second call. Independent requested-axis
equality and self-inversion identify its actual generated request with the
certified permutation inverse. Callers supply no generated-axis, intermediate
success or inverse premise. The three added laws consume the frozen
`transpose-PROOF.bend` producers without changing that specification or runtime.

The structural closed-capacity bridge and allocation producers were certified
without altering `2^20`, the rank-zero-through-four domain, executable behavior
or any tool budget. Their earlier literal-expansion diagnostics are historical,
not substituted bounds or live blockers. The 54-law public permutation/inverse
checkpoint is certified in both source modes and a fresh direct-kernel export.
The 57-law requested-transpose extension also passes both source modes, fresh
export and direct-kernel replay in its frozen import graph. Separate public
outcomes for `[0]` and the actual rank-four
`[4294967295,4294967295,0,4294967295]` shape pass all four modes: actual first
success and requested dimensions, actual mandatory inverse success and original
dimensions, for both permutation and transpose. Their witnesses include the
non-vacuous public results and shapes rather than relying on empty-index
observation laws. Huge-zero construction and consumer witnesses are organized
generically before concrete specialization, without changing the dimensions,
domain or checker budgets.

Fresh native and JS demo outputs match their respective immutable 32-law and
42-law outputs byte-for-byte, including raw nonfinite output. The freshly
instrumented native demo reports `STORAGE_COPY_AUDIT 0 0` and unchanged output.
Pertinent semantic controls and the final aggregate gate remain separate
closure obligations. The tensor layer stays unpublished until those obligations
are complete.

## Native and negative evidence

`transform-demo.bend` exercises rank-zero through rank-four permutations,
transpose, inverse transforms, unchanged retained inputs, reshape linear order,
wrong axes and products, empty shapes, and huge-zero dimensions. Raw-word
cases include signed zero, a subnormal, infinities, and a NaN payload. Native
words after inverse copies and inverse reshape match their original observations. The generated-C
split/repack audit reports `STORAGE_COPY_AUDIT 0 0`.

The approved-codec proof and native artifacts are outside the repository in
`<scratch>/s2-transform-codec-s1-01/checkpoint32-final/`; the proof-only
42-law extension is in `<scratch>/s2-transform-totality-01/checkpoint42-final/`.
The preceding checked checkpoint is in
`<scratch>/s2-transform-replacement-01/`, and the earlier negative controls
remain in `<scratch>/s2-transforms/`. False duplicate axes, wrong selected dimensions,
unequal-numel reshape and wrong inverse coordinates are rejected by both
checking entrypoints. The naive outer reader-projection failure is retained
as `opaque-core-reader-fail.log`; the exact reader certificate seam resolves
that projection. The old prefix mismatch and incomplete Core producer logs
are historical diagnostics, not current blockers: the frozen Core import,
actual step, natural certificate, concrete prefix extension and loop induction
pass the pinned formal verdict with the independent axis reference integrated.
The frozen checkpoint covers check-only, formal verdict, export and direct
BendTT kernel checking, six accepted/rejected control pairs with both checking
entrypoints and accepted direct-kernel checks, and the fresh native copy audit.
The native demo's 95 observation lines match the pre-codec demo exactly; that
is observed corpus coverage, not an old/new equivalence theorem.

The generated JavaScript demo differs from native only at six NaN-payload
observations: native retains `0x7fc12345`, while JS reports `0x7fc00000`. The
Base-only reduction `../f32-word-lane-disagreement.bend` relative to this
checkpoint reproduces that difference on pinned Bend2 2.0.34 and installed
2.0.35 without importing any tensor or transform module. Both versions pass
both checking entrypoints. This is the existing boundary documented in
`docs/storage.md` and `docs/runtime-corpus.md`, not a new codec regression.
The frozen `oracle/tensor/numerical-policy.json` requires matching NaN classification
and records payload/sign/quiet-bit metadata; it does not require portable NaN
payload equality. All finite words, infinity words, shapes and error outputs
match exactly between these demos. Raw lane outputs and the reduced reproduction
are preserved; no normalization, policy change or toolchain change is used.

```bash
export BENDTT="$PWD/.tools/kernel/bendtt" BEND_NO_TELEMETRY=1
export LEAN_STACK_SIZE_KB=4194304
bun .tools/bend/bend2/main.ts src/tensor/transform-PROOF.bend --check-only
bun .tools/bend/bend2/main.ts src/tensor/transform-PROOF.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/transform-PROOF.bend \
  -o ".evidence/tensor-transforms/proof.bendtt"
.tools/kernel/bendtt ".evidence/tensor-transforms/proof.bendtt"
CC=.tools/llvm/usr/bin/clang-19 \
LD_LIBRARY_PATH=.tools/llvm/usr/lib/x86_64-linux-gnu \
bun .tools/bend/bend2/main.ts src/tensor/transform-demo.bend \
  -o ".evidence/tensor-transforms/demo.native" \
  -o ".evidence/tensor-transforms/demo.c"
LD_LIBRARY_PATH=.tools/llvm/usr/lib/x86_64-linux-gnu \
".evidence/tensor-transforms/demo.native" --threads 1 --gpu off
```
