# Public construction and clone source contracts

`src/tensor/construction-LAWS.bend` states five public source-contract laws.
`construction-PROOF.bend` supplies checked producers for all five. No caller
supplies a result, a mapping, a retained-frame hypothesis, or a conversion
certificate to these laws.

## Added laws

### `public_from_values_elements`

For every validated `Shape.Shape` and every supplied `List<&2, F32>`, the law
returns `FromValuesModel<shape, values>`. Its erased field states
`FromValuesMeaning(shape, values, Core.from_values(shape, values))`, tied to
the actual public call rather than an assumed output.

The meaning is `Unit` when that call fails. For a successful returned tensor,
it quantifies over every `U32` index whose natural value is in the shape's
mathematical logical domain. For each such index it produces:

1. `Core.ObservationChecked(Core.Packed{shape, tensor}, index)`.
2. An equality between `Core.checked_observe(shape, tensor, index)` and
   `list_element(values, U32.to_nat(index))`.
3. An equality between `Core.observe(shape, tensor, index)` and that same
   independent list observation.

`list_element` recursively indexes the supplied list by a natural position;
it does not inspect storage, call a constructor, or reproduce the write loop.
For owned storage, `Core.observe` is the structural `Model.observe` at the
actual `Refinement.route(depth, index)`. The law therefore establishes the
supplied logical elements under both checked source observations and the
independent storage model.

The quantifier uses the full shape logical size, not a smaller prefix or a
route-matching subset. Validated shapes retain the existing rank-zero-through-
four, size-at-most-`2^20` domain. The successful source value-count decision
and the erased `Shape.elements_exact` bridge connect the cached live size to
this mathematical domain; no new rank, size, or numerical premise is added.

### `public_construction_elements`

For every dimensions list and every supplied values list, the law returns
`ConstructionModel<dimensions, values>`. Its erased field states
`ConstructionMeaning(values, Core.construct(dimensions, values))` for the
actual public dynamic construction call.

Failure has meaning `Unit`. Every successful `Core.Packed{shape, tensor}`
has exactly the element meaning above over all logical indices of its
returned shape. `construction_shape_elements` follows the actual
`Shape.construct` result and `Core.construct_shape` branch;
`construction_values_elements` follows the actual `Core.construct_values`
packing branch. They invoke the checked `public_from_values_elements`
producer for the same values passed to the public constructor.

The existing Core successful-dimensions and successful-value-count laws
remain unchanged. These new laws add the missing supplied-element meaning;
they do not replace those contracts.

### `public_clone_elements`

For every validated shape and every `Core.Tensor<shape>`, the law returns
`CloneMeaning<shape, original>`. Each of its four erased fields quantifies
over every `U32` index, without an in-bounds or route-mapping premise:

- `first_checked` says the first owner returned by the actual
  `Core.clone(shape, original)` preserves `Core.checked_observe`.
- `second_checked` says the second returned owner preserves that observer.
- `first_model` says the first returned owner preserves `Core.observe`.
- `second_model` says the second returned owner preserves that observer.

The proof covers empty storage and arbitrary balanced owned storage,
including padded arrays and tensors produced by earlier operations. It is
not limited to fresh allocations or finite examples. Both actual returned
owners have type `Core.Tensor<shape>`, retaining their indexed shape/storage-
length invariant.

### `public_from_values_complete`

For every lawful `Shape.Shape` and supplied `List<&2, F32>` whose length equals
`Shape.logical_numel(Shape.dimensions(shape))`, the law returns
`construction-totality.TensorDone<shape, Core.from_values(shape, values)>`.
The count premise uses the mathematical shape count, not a cached word count,
a caller conversion, or a successful count decision.

The certificate has three erased fields. `witness` is an
`Allocation.TensorDone` containing the actual returned tensor and an equality
`Done{tensor} == Core.from_values(shape, values)`. `successful` independently
states that `Result.is_done` of the actual call equals `True`. `readable`
supplies the existing source-read success certificate at every word index
inside the mathematical logical domain of that returned tensor. The witness
and status exclude the vacuous failure branch of the element-meaning law.
Composing with unchanged `public_from_values_elements` retains the supplied
element semantics for this successful tensor.

No premise requires `Core.new`, a checked read/write, a cursor conversion, or
the fill loop to succeed. A lawful shape already carries the existing rank
and inclusive capacity contracts. Scalars, empty shapes, shapes with huge
dimensions and a zero mathematical product, and nonempty shapes up to the
`2^20` capacity endpoint remain admitted. No factor bound is inferred from
a zero product, and no restriction is placed on the supplied F32 values.

### `public_construction_complete`

For every dimensions list and values list, the admission premises are exactly:

1. `Shape.LE(Shape.rank_dimensions(dimensions), 4n)`.
2. `Shape.LE(Shape.logical_numel(dimensions), 1048576n)`.
3. `List.length(&2, F32, values) == Shape.logical_numel(dimensions)`.

These are input contracts; none assumes that `Shape.construct` or another
intermediate operation succeeds. The law returns
`construction-totality.ConstructionDone<dimensions, Core.construct(dimensions, values)>`.
Its erased `ConstructionWitness` supplies the actual returned shape and
tensor, the equality `Done{Core.Packed{shape, tensor}} == Core.construct(dimensions, values)`,
and preservation of the requested dimensions. Its other erased fields state
actual public-call success and source-read success throughout the requested
mathematical logical domain. The unchanged public construction element law
and Core successful-dimensions/value-count laws retain their old statements
and producers.

The two completeness laws are additive. None of the original three
construction statements or existing definition bodies changed.

## Producer bodies and induction

`fill_model` interprets the actual natural-cursor `Core.fill_values` call.
It follows the empty-list branch, actual conversion decision, actual
`Core.write_flat` result, and recursive call. A failed conversion or write
has the failed-result meaning; a successful conversion supplies the natural
cursor equality from the actual `Shape.BTrue` decision.

`fill_write_pair` obtains write meaning from
`CoreProof.checked_update_model`, transporting it through an equality to
the actual write pair. `written_model` converts that checked meaning into
the written position's observation and earlier-position frame.
`CoreProof.u32_natural_injective` connects equal natural indices to the
actual word index; the frame's separation proof follows from a genuine
strict earlier-position bound. No free mapping/frame assumption is used.

The list induction proves both the successfully written suffix and the
preservation of earlier checked positions. `fill_elements_advance` uses the
recursive frame to retain the current head write, and the recursive mapping
to interpret each tail position. `fill_frame_advance` composes the actual
write frame with the recursive frame. The empty branch derives the
impossibility of observing an element of an empty supplied suffix and
retains every earlier observation reflexively.

`complete_fill` specializes the interpretation to the actual `0n` entry
cursor. `supplied_count` uses the actual successful cached-size decision
plus `Shape.elements_exact` to cover the entire mathematical logical domain.
`element_point` supplies the model-backed equality using the checked
observation bridge. `from_values_new_elements` and
`from_values_counted_elements` follow the actual public constructor branches;
`Laws.public_from_values_elements` packages the final source-linked producer.

For clone, `clone_meaning` handles the storage constructors.
`clone_owned_meaning` composes explicit projection equalities through the
actual `Core.clone_result` and `Storage.finish_clone` wrappers with the
existing shape and storage layer `Clone.certify`, `Clone.first_equal`, and
`Clone.second_equal` producers for the actual `Base.Array.clone` operation.

For completeness, `fill_success` inducts on the supplied suffix with the
invariant `cursor + List.length(values) <= Shape.elements(shape)`. In a
nonempty branch, `cursor_below` derives a strict logical bound without
examining or bounding individual dimensions. `cursor_exact` obtains the
actual word-to-natural round trip internally from
`WordConversion.shape_count_exact` and `natural_word_exact`, using the
validated shape's count as the word limit. `add_successor` transports the
remaining-suffix bound to the next natural cursor. The empty branch succeeds
without converting or writing the endpoint cursor.

`fill_decided_success` excludes the failed conversion branch from that
derived equality. `fill_step_success` excludes actual write failure using
the certified `TransformLaws.tensor_write_valid_success` producer supplied
by `TransformProof`; its recursive premise is produced by the list
induction, not admitted at the public boundary. `allocated_fill_success`
uses `AllocationProof.core_new_done`, which internally reuses
`WordConversion.shape_count_within` and `shape_count_exact`. The exact
mathematical value-count premise is bridged to the cached natural size with
unchanged `Shape.elements_exact`.

Dynamic completeness invokes the certified shape-construction completeness
producer under the original rank/capacity bounds. It follows the actual
`Shape.construct`, `Core.construct_shape`, and `Core.construct_values`
branches, transports the supplied length through preserved dimensions, and
uses the completed `Core.from_values` producer. The final witnesses are tied
to actual returned results. Their read-success fields use
`TransformProof.read_success` and `read_success_status`, so read success is
also derived rather than assumed.

## Erasure and trusted boundaries

The public certificates, construction outcome certificates, write meaning,
and clone meaning keep their evidence in erased fields. Actual returned
tensor/shape witnesses, result equalities, success statuses, and read evidence
in `construction-totality.bend` are erased too. The executable constructor
modules do not import these proof modules. No executable definition changed,
and no copy or observer traversal was added to the runtime implementation.
Base storage stays in-place at runtime.

The first three laws remain structural source/model partial-correctness
claims. The two completeness laws additionally prove actual source-model
success on admitted inputs; Bend's termination discipline checks the supplied
producers and source recursion. This does not add native allocation or
hardware guarantees beyond the existing source-model trusted base. None of
these laws proves exact rejection codes, native/JavaScript heap implementation
correctness, IEEE behavior, operating-system allocation behavior, or pointer
separation.
Affine sole ownership is checked by Bend's source quantity discipline;
native/JavaScript buffer independence remains a runtime conformance boundary.
The tensor corpus and existing shape/storage laws still cover their separate
responsibilities. The supplied-element and both-owner public clone meaning
gaps are closed by the original three laws; the two additive completeness
laws close the successful-public-construction gap under the input contracts.

The checked statements rely on the pinned Bend2 source translation, the
BendTT kernel, and the existing shape and storage layer's structural Base array
model and producers. No shape or storage definition, law, checker, kernel,
source pin, bound, or tolerance was changed by this proof work. Aggregator and
gate integration belong to the tensor gate owner.

## Fresh verification

The command host and task-owned worktree (`<worktree>`) were verified
on `bendtorch/s2-cpu-tensors` at HEAD
`0bbfb1906e2b809cc9c78b6a1a706e9b40d45535` before work began.

The fresh output directory is
`<scratch>/s2-review-construction-totality-01/`. With
`BEND_NO_TELEMETRY=1` and `BENDTT=$PWD/.tools/kernel/bendtt`, these commands
completed successfully:

```sh
bun .tools/bend/bend2/main.ts src/tensor/construction-PROOF.bend --check-only
bun .tools/bend/bend2/main.ts src/tensor/construction-PROOF.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/construction-PROOF.bend -o <output>/PROOF.bendtt
.tools/kernel/bendtt <output>/PROOF.bendtt
```

`check-only.log`, `verdict.log`, and `direct-kernel.log` each report
`ALL PROOFS CHECK`. The direct export contains:

- `construction_2d_LAWS.public_clone_elements`.
- `construction_2d_LAWS.public_from_values_elements`.
- `construction_2d_LAWS.public_construction_elements`.
- `construction_2d_LAWS.public_from_values_complete`.
- `construction_2d_LAWS.public_construction_complete`.
- `construction_2d_LAWS.list_element` and the certificate/meaning definitions.
- `clone_meaning`, `fill_model`, `supplied_count`,
  `from_values_counted_elements`, and `construction_shape_elements`.
- `fill_success`, `from_values_complete`, `construction_shape_complete`,
  and the new completion/witness types.

The six closed positive controls in `controls.bend` check actual public
source-call success for scalar (`[]`, one value), empty (`[0]`, no values),
and nonempty (`[2]`, two values) inputs. Source checking, verdict, export,
and direct kernel checking pass. The original `api-controls.bend` applies
both actual new law APIs to those inputs and to huge-zero dimensions
`[0, 4294967295, 4294967295]`. Its source check passes, but its verdict fails
with the source/kernel mismatch diagnostic and its direct kernel run aborts
with a stack overflow. That closed attempt is resource-blocked, not
kernel-certified or a semantic refutation. Its mathematical count remains
zero without requiring the nonzero factors to fit the allocation limit.

Mutation controls remain entirely outside `src/`. For each public wrapper,
an isolated source copy replaces only that wrapper's body with
`Fail{ValueCountMismatch{}}`. Each of the three closed success statements
then fails source checking with `expected : False{}` and
`observed : True{}`. Independent mutations of the exported kernel wrapper
bodies produce the corresponding six semantic kernel rejections, each
naming its scalar, empty, or nonempty success declaration. Each negative
control exits with status 1 and an ordinary proof mismatch; none relies on
a timeout, stack overflow, missing import, or another resource failure.
Full-production source/export mutation checks also reject the changed
wrappers, first at existing Core partial-correctness producer bodies.

`frozen-source.sha256` records the proof-source hashes established before
controls and documentation. `s1-preservation.log` and
`dependency-preservation.log` confirm unchanged protected shape/storage,
tensor source/proof dependencies, and the pin. The original three statements
and existing construction definition bodies were compared byte-for-byte
with the preserved baseline. The inclusive capacity endpoint is covered by
the universal laws' unchanged `<= 1048576n` contract; these controls do not
claim a native million-element allocation measurement.

Importing the module into an aggregator changes its qualification prefix.
The proof module imports the stabilized Core proof producer, so its verdict
and direct export also recheck that source dependency. This verification is
not a substitute for the root-owned tensor runtime gate.

## Authorized generic-first control certification

The user authorized proving generic facts before specializing into the exact
same closed cases, without changing dimensions, supplied values, expected
success, data guarantees, law meanings, executable definitions, pin, tools,
or resource limits. The original `-01` directory and every reproduction/log
in it remain immutable. The fresh `-02` phase reverified the host,
the same task worktree/branch/HEAD, and the three frozen production-source
hashes. It snapshots the root's updated Core proof dependency closure,
including its additive dispatch laws, without changing that closure.

Fresh evidence is in
`<scratch>/s2-review-construction-totality-02/`. The generic
`actual_from_values` and `actual_construction` lemmas in
`generic-controls.bend` use the unchanged public laws under their exact
existing admission premises. Generic zero-pair and zero-triple producers
derive the same lawful shape, zero count, and actual public construction
certificate for arbitrary U32 dimensions. No new caller hypothesis assumes
an intermediate success or a conversion, and no factor bound is introduced.
The generic file passed source checking, verdict, export, and direct kernel
checking before its closed specialization was introduced.

`closed-direct-specialization.bend` then specializes those certified generic
producers to all ten unchanged cases:

| Dimensions | Supplied values | Public APIs certified |
| --- | --- | --- |
| `[]` | `[3.0]` | `Core.from_values` and `Core.construct` |
| `[0]` | `[]` | `Core.from_values` and `Core.construct` |
| `[2]` | `[3.0, 4.0]` | `Core.from_values` and `Core.construct` |
| `[0, 4294967295]` | `[]` | `Core.from_values` and `Core.construct` |
| `[0, 4294967295, 4294967295]` | `[]` | `Core.from_values` and `Core.construct` |

For `from_values`, dimensions denote the lawful indexed shape with that
exact dimensions list. Every returned certificate is still the frozen
`Totality.TensorDone<shape, Core.from_values(shape, values)>` or
`Totality.ConstructionDone<dimensions, Core.construct(dimensions, values)>`.
It therefore retains the actual returned tensor/Done witness, actual success
status, requested dimensions for dynamic construction, and source-read
guarantees over the full mathematical logical domain. The unchanged
element-meaning laws apply to the same actual calls and supplied values.
The closed direct file passes `--check-only`, `--verdict`, export, and
direct kernel checking with exit status 0 in each checking mode. No old
resource-failed attempt was counted as that new pass or removed from evidence.

The supplementary `generic-witness.bend` route also certifies a generic
actual-result package before closed specialization. It stores an erased
result, an exact equality to the unchanged public call, the full frozen
completion certificate indexed by that result, and the unchanged
supplied-element meaning for the same result. Thus it does not replace the
actual constructor with a specification or discard success, dimensions,
readability, or data guarantees. `closed-witness-specialization.bend`
specializes it to the same ten cases in the table. Both generic and closed
witness files pass source checking, verdict, export, and direct kernel
checking. These supplementary packages remain proof/control artifacts
outside `src/`; the production certificate API is unchanged.

The original six direct positive status controls and all six source plus six
kernel fail-only mutations were also rerun freshly with unchanged cases and
expected outcomes. The positives pass both checking modes and direct kernel
checking. Each negative exits with status 1 and its ordinary semantic proof
mismatch. The fresh case metadata separates these certified positives,
genuine semantic negatives, and historical resource-blocked routes.
`control-metadata.json` records the exact dimensions/F32 literals, expected
success, generic-before-closed certification, route statuses, source hashes,
updated Core proof snapshot, and all twelve semantic mutation results.

This is a proof/control organization change outside `src/`, not a change to
`Core.construct`, any old law block, or the frozen production API. The
historical bare `Shape.construct([0, 4294967295])` reduction still records
source acceptance and a kernel resource failure; the generic-first
certificate proves the same admitted public-construction requirement through
the checked generic producers. None of these results is a native allocation
measurement or a hardware guarantee beyond the source-model trusted base.

## Authorized cursor refactor

The original constructor used `U32.add(index, 1)`. Developing a no-wrap
bridge exposed a diagnostic-normalization crash with a closed `2^32` natural
bound. The standalone reduction and both-version outputs remain outside
`src/` in `<scratch>/s2-bend-bug-report/large-capacity-context/`.
The deliberately invalid equality should produce an ordinary rejected-proof
diagnostic, but pinned commit `947db722640c86247849343657bf2f7ef01cb7f1` and
freshly updated release `2.0.35` instead overflow the checker stack.

The user authorized the root's judgment after disclosure. The Core owner
implemented the selected natural checked-cursor loop, entering at `0n` and
checking each word-to-natural round trip before the actual write. This proof
follows that authorized source directly, without a closed `2^32` context,
weakened law, or altered pin/tool. The stopped arithmetic experiment remains
outside `src/` beside the reduction.
