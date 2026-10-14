# Public elementwise guarantees

The elementwise review additions certify the actual public
`Elementwise.binary` result. They do not change tensor arithmetic, storage,
shape planning, cursor execution, or the existing 35 Core laws. The new
statements live in `src/tensor/elementwise-LAWS.bend`, their models in
`elementwise-totality.bend`, and their proofs in `elementwise-PROOF.bend`.

## Laws

`binary_retains_inputs` is unconditional for every `Operation` and pair of
lawful dynamic tensors. It observes the actual retained pair, including
error results, and relates the first retained tensor to the original left
tensor and the second to the original right tensor. `SameSource` quantifies
over `Dimensions`, `Checked{index}`, and `Raw{index}` selectors. Thus it
preserves both dimension lists and both observation functions for every
word index, rather than preserving only the planned output prefix or an
unordered pair of sources. Raw observations include the underlying storage
projection outside the logical input length; checked observations retain
the logical bounds check.

`binary_successful_output_dimensions` takes an independent natural-number
`Shape.Broadcast` witness from the actual input extents to a proposed output
dimension list. If the public result is `Done`, its actual packed output
dimensions equal that list. There is no output rank or capacity hypothesis
on this partial-correctness statement. The proof combines the broadcast
planner's input-dimension preservation with natural broadcast-output uniqueness
and the word/natural round trip, then follows the actual allocation and loop
result. A zero-length output with dimensions `[0, 2]` cannot satisfy the
statement for the independently specified `[0]` output of `[0]` with `[0]`.

`binary_valid_success` covers every `Operation`, including `Add`, `Sub`,
and `Mul`, for lawful inputs with a `Compatible` witness. That witness
contains only an independent natural `Shape.Broadcast` alignment, the
tensor layer's output rank bound of four, and its inclusive output cardinality
bound of 1,048,576. It contains no successful actual planner result, allocator
result, read or write result, conversion equality, or loop certificate.
The conclusion states that the actual public result is `Done`, so returning
`Fail` on every valid input cannot satisfy it.

These additions strengthen the public specification separately. No existing
law statement or definition called by an existing law was altered. The
root-owned tensor gate aggregators and export inventory must protect all three
new law declarations and their proof implementations.

## Proof boundary and domain

The valid-input proof derives actual planner success using shape-layer
broadcast completeness. It derives allocation success from `AllocationProof`,
bounded left and right offsets from shape-layer flat projection, source reads
and target writes from `TransformProof`, and exact cursor conversion from the
certified word-conversion and cursor producers. The induction follows the
actual natural step and loop. Unconditional retention carries the original
input dimension equalities to each next state, so every subsequent read retains
its valid logical bound. The left and right projections remain distinct;
asymmetric broadcast axes are never identified with one another.

Dynamic shape extraction occurs in erased model indices and admission types.
The public proof consumes each affine owner once. Intermediate erased proof
models do not require copying a live dynamic tensor or array. The runtime
modules do not import these additional models.

Cardinality bounds apply to the mathematical total, not to individual
dimensions. No factor bound is inferred from a zero product. Scalar shapes,
empty shapes, huge nonzero dimensions in a zero-cardinality shape, and
broadcasts of zero with one remain within the stated domain. Empty outputs
skip the actual loop and impose no input-index or nonzero-output condition.
The proofs contain no `@unsafe`, open proof holes, axioms, new fuel, or
numerical tolerance changes.

## Verification and countermodels

The API checkpoint is recorded in
`<scratch>/s2-review-elementwise-totality-01/frozen-api.sha256`. Before
integration, the pinned Bend 2.0.34 at commit
`947db722640c86247849343657bf2f7ef01cb7f1` accepted the complete new proof
module with `--check-only` and `--verdict`. Its exported
`elementwise-PROOF.bendtt` also passed a separate direct invocation of
`.tools/kernel/bendtt`. The protected shape/storage sources, tensor runtime,
proof dependencies, and tool pin were compared with the checkout baseline;
`immutable-dependencies.diff` is empty. The three new Bend files were frozen
before integration and subsequent control work.

`run-controls.py` keeps sixteen accepted/rejected countermodel pairs entirely
outside `src/`. They cover the wrong `[0, 2]` empty output shape, changed left
and right retained dimensions, replacement and swapped retained tensors on
both success and error, and changed raw padding while the checked logical
value remains unchanged. Three further pairs refute a `Fail` result under
the success model for each operation. All sixteen pairs passed their expected
outcome in both source modes and in direct exported-kernel checks: 112 checks
in total, recorded in `controls-results.json`. The rejected kernel controls
change only the accepted export's concrete candidate-result producer. Every
negative result is an ordinary expected/observed proof mismatch at `witness`,
not a timeout, source budget, stack overflow, or missing-import rejection.

This proof verification is not a replacement for the root-owned tensor
native/JS runtime corpus, retained-input harness checks, or Python gate checks.

## Stopped closed huge-zero control

An additional actual-public-call control prelude exposed a suspected Bend
kernel bug in an unused closed shape constructor. The standalone reduction
`huge-zero-kernel-overflow.bend` contains only an import and:

```bend
def shape() -> Shape.Shape:
  Shape.SShape{[0, 4294967295, 4294967295], 0n, {==}, Unit{}, Unit{}}
```

Pinned Bend 2.0.34 and the installed latest release 2.0.35 accept the file
with `--check-only`, but both report a checker/kernel mismatch with
`--verdict`. Exporting with the pin and directly checking the artifact exits
134 with `Stack overflow detected. Aborting.` A lawful zero-element shape
should certify independently of its nonzero factors. The reduced scalar
public call itself passes both checkers; the original failure therefore does
not establish a scalar-operation failure or invalidate the certified general
laws.

The reduction and both-version outputs remain in the same evidence directory.
The dependent closed huge-zero control route was stopped and reported before
any workaround. No runtime, law, tool, pin, source-domain, or fixture change
was used to claim that route passed. Its resource failure is not included in
the semantic countermodel count.

## Authorized generic-first replay

After the user authorized deriving generic certificates before instantiating
the original closed cases, the replay used
`<scratch>/s2-review-elementwise-totality-02/`. It independently reconfirmed
the host, original worktree, branch,
checkout, and all three frozen Bend-source hashes. The production laws,
models, and proofs were not rewritten.

`generic-huge-zero-shape.bend` first certifies a zero-headed shape for an
arbitrary tail and its original rank bound. It then instantiates that producer
at `[4294967295, 4294967295]`. The resulting closed shape still has exactly
`[0, 4294967295, 4294967295]` as its dimensions and zero as its count; those
two equalities are checked explicitly. Source checking, verdict, export, and
direct kernel checking pass for this shape certificate.

`generic-contracts.bend` separately certifies an actual-public-success
producer for every operation, lawful shape, and pair of tensors of that shape.
It constructs the independent self-broadcast witness and takes rank and
capacity evidence from the lawful shape, then invokes `binary_valid_success`.
Its source, verdict, export, and direct-kernel checks pass against the updated
Core proof closure containing the root-owned three dispatch laws. No erased
information is evaluated as live evidence, and no two-step erased eliminator
was needed or introduced.

The exact closed cases in this first replay remained a separate result. The scalar, empty, and
original huge-zero success bundle in `closed-success-controls.bend` passes
source checking, but not verdict; its separate direct-kernel diagnostic exits
134 with a stack overflow. The huge-zero output-dimension certificate in
`generic-public-controls.bend` also hit a source stack overflow when called
directly on the closed witness;
pin 2.0.34 and latest 2.0.35 agree on that resource failure. A generic-first
output-dimension certificate instantiated at the same closed dimensions then
exceeded the existing 120-second command limit, without producing a source
verdict. Neither failure is a semantic countermodel rejection or an accepted
closed proof.

The resumed public-call mutation prelude derives both shape and empty-storage
length certificates generically, while retaining the exact original huge-zero
dimensions. Pin and latest accept its source check but reject verdict, and the
pinned exported direct kernel exits 134 with a stack overflow. Consequently,
the dependent public-call mutation route was stopped before counting any of
its controls. The separate closed-retention certification that imported the
failing closed bundle was cancelled, not counted as a pass. The generic and
frozen universal contracts remain certified; those particular closed bundles remain
resource-blocked, with their original sources and diagnostics preserved for
the root's independent triage. `generic-dependency-closure.sha256` records the
56-file source closure used by the successful generic producer, and its
recorded hashes were checked again before handoff.

## Final exact public-call retry

The next authorized retry uses
`<scratch>/s2-review-elementwise-totality-03/`. All nine original public
cases now pass source checking, verdict, fresh export, and separate direct
kernel checking: Add, Sub, and Mul on scalar `[]`, empty `[0]`, and exactly
`[0, 4294967295, 4294967295]`. Each case requires actual success, exact output
dimensions, and ordered input dimensions plus checked/raw observation
preservation together. There are 36 successful final case checks; none is a
resource failure counted as a pass.

`generic-producers.bend` constructs the same independent Nat alignment
structurally from the original live U32 dimension list, instead of supplying
a reusable closed Nat list. Nat conversion remains in erased type indices.
`two-step-producers.bend` executes the unchanged binary call once, analyzes
its actual returned result, and packages the actual pair with erased
success, retention, and dimension certificates. It never computes or
branches live on erased evidence.

A first public projection with the concrete public result as an erased type
argument still hit the original source limit. That failed route is preserved
in `public-result-index-attempt/`; the root's independent replay reproduced
the 120-second huge-zero Add timeout. It is neither a pass nor a semantic
refutation.

The final `public-call-certificates.bend` instead uses `CheckedAdd`,
`CheckedSub`, and `CheckedMul`. Each stores its exact equality to the named
original public call inside the actual-pair binder, composing the protected
Core dispatch law. Its caller supplies only the original lawful shape and
tensors, not a replacement result or a success/retention hypothesis. The
three unchanged production laws, models, and proofs remain frozen.

`final-replay-manifest.json` lists all nine cases and the precise thirteen
scratch source files; `final-scratch-source.sha256` freezes those files,
`final-dependency-closure.sha256` freezes their 67-file closure, and
`final-verifier.sha256` identifies the pinned checker and kernel.
`public-call-case-results.json` records all four checks per case. Earlier
direct-signature, bundle, and public-result-index resource diagnostics remain
separate and unchanged. No dimension, premise, required outcome, observer,
reference model, bound, runtime call, tool, pin, or resource limit changed.
