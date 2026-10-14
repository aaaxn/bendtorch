# Base word conversion bridge

`src/tensor/word-conversion.bend` proves conversion identities for the pinned
Base definitions. It does not introduce another codec or assume that two codecs
agree. The public proofs are independent of tensor construction and permutation
execution.

## Consumer API

```bend
def natural_word_exact(+number: Nat, +limit: U32,
  +within: {Nat.is_le(number, U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}

def natural_word_le(+number: Nat) ->
  {Nat.is_le(U32.to_nat(U32.from_nat(number)), number) == True{} : Bool}

def capacity_word_nat() -> {U32.to_nat(1048576) == 1048576n : Nat}

def capacity_natural_word_exact(+number: Nat,
  within: Shape.LE(number, 1048576n)) ->
  {U32.to_nat(U32.from_nat(number)) == number : Nat}

def shape_count_exact(+shape: Shape.Shape) ->
  {U32.to_nat(Shape.count(shape)) == Shape.elements(shape) : Nat}

def shape_count_within(+shape: Shape.Shape) ->
  {U32.is_le(Shape.count(shape), 1048576) == True{} : Bool}

def word_natural_exact(+word: U32) ->
  {U32.from_nat(U32.to_nat(word)) == word : U32}

def naturals_words_fitting(+dims: List<&2, U32>, +values: List<&2, Nat>,
  +fit: {True{} == Shape.fit(Shape.naturals(dims), values) : Bool}) ->
  {Shape.naturals(Shape.words(values)) == values : List<&2, Nat>}
```

Import the implementation as `Conversion` and the paired
`word-conversion-PROOF.bend` module to include its discharged laws in an
aggregate proof. The implementation imports Base, `storage/arith.bend`,
`shape/core.bend` and the unchanged `tensor/core-PROOF.bend`. Its checked
dependency closure includes the existing Core, projection and storage proofs;
the exact capacity extension additionally imports unchanged
`storage/model.bend` and `shape/nat_proofs.bend`. It does not import tensor
layout or transforms.

The scalar theorem applies to every natural bounded by an actual U32 value.
Its inclusive bound is a representability requirement, not a conversion-equality
assumption. Specializing it from the tensor layer's literal capacity bound
requires a separate bridge to `U32.to_nat(1048576)`: source literal semantics
and executable compiler lowering are not proof of definitional equality. That
previously unproved bridge is now supplied by the exact `capacity_word_nat`
theorem. The new `capacity_natural_word_exact` API accepts only the existing
`Shape.LE(number, 1048576n)` contract and derives the representation transport
internally; callers supply no new representation or conversion equality.

`natural_word_le` applies to every Nat without a capacity or exactness premise.
It permits wrapping: conversion may decrease the number, but it never increases
it. Together with `Arith.u32_lt` and `Arith.lt_le_trans`, it transports a checked
U32 index bound against `U32.from_nat(number)` to the mathematical bound
`U32.to_nat(index) < number`. This inequality still does not prove conversion
exactness at the tensor layer's closed literal capacity; that is the separate
identity proved by `capacity_word_nat` and `capacity_natural_word_exact`.

The list theorem uses each actual dimension as the scalar theorem's limit.
`Shape.fit` supplies the strict coordinate bound and the matching list lengths.
There is no rank restriction or global coordinate cap in this theorem, so it
covers tensor ranks 0 through 4 without excluding arbitrary U32 dimensions on
zero-count shapes. Such shapes have no fitting coordinates when any extent is
zero. The rank-zero case proves the identity on empty lists.

## Proof mechanism and laws

The increment proof recurses on symbolic word width. A word strictly below
another word of the same width must contain a zero bit. Incrementing such a
word either changes its first zero bit or carries through leading one bits;
`Word.to_nat` then increases by exactly one. The scalar conversion proof
recurses on `number`, following Base's actual `U32.from_nat` definition: zero
becomes zero, and a successor calls `U32.inc`. The bound gives a strictly larger
U32 limit at each predecessor, so the increment theorem applies. The U32 inverse
uses that theorem with `number = U32.to_nat(word)` and `limit = word`, a proved
reflexive natural bound, and Core's already certified
`u32_natural_injective(left, right)(equal)`. It has no hypothesis on `word` and
never instantiates a closed max-U32 natural. The list proof
recurses on the matching dimension and coordinate lists and applies the scalar
identity at each coordinate.

The one-sided increment theorem also recurses on symbolic word width, but it
has no non-overflow premise. At width zero, increment wraps to zero, so zero is
at most one. At a zero low bit, increment equals the natural successor. At a
one low bit, the induction bound on the tail is preserved by monotonic
doubling; this includes a carry that wraps the entire word to zero.
`natural_word_le` then follows Base's Nat recursion and combines the increment
bound with the predecessor's induction bound by natural <= transitivity.

The new law module states, and its paired `def Laws.*` entries prove:

| Law | Statement and reason |
| --- | --- |
| `word_increment_exact` | `Word.to_nat(width, Word.inc(width, word))` equals the successor of `Word.to_nat(width, word)` when the word contains a zero bit. This records the non-overflow step for symbolic widths. |
| `increment_below_exact` | `U32.to_nat(U32.inc(word))` equals the successor of `U32.to_nat(word)` when `word < limit`. This provides a public comparison-bound form of the increment result. |
| `natural_word_exact` | `U32.to_nat(U32.from_nat(number)) == number` under the inclusive actual-U32 bound. This is the general Base conversion bridge required by Nat-Horner and digit algorithms. |
| `word_natural_exact` | `U32.from_nat(U32.to_nat(word)) == word` for every U32 word, without any additional hypothesis. This restores the actual word after a natural-coordinate proof. |
| `naturals_words_fitting` | `Shape.naturals(Shape.words(values)) == values` when the coordinates fit the actual dimension list. This transports the scalar bridge to the live shape consumer. |
| `word_increment_le` | Incrementing a word never decodes above the successor of its previous natural value, for every symbolic word width including wraparound. This records the unconditional increment bound. |
| `increment_le` | The corresponding unconditional U32 increment bound. This lifts the symbolic-width result to Base's actual U32 increment. |
| `natural_word_le` | `U32.to_nat(U32.from_nat(number)) <= number` for every Nat with no premises. This transports U32 bounds to mathematical natural bounds without asserting conversion exactness. |
| `capacity_word_nat` | The actual U32 word 1048576 decodes to the source Nat literal 1048576n, as an exact equality rather than an inequality or compiler-lowering assumption. |
| `capacity_natural_word_exact` | The natural-to-word-to-natural identity for every natural satisfying the existing `Shape.LE(number, 1048576n)` contract, including zero and the capacity endpoint. |
| `shape_count_exact` | The actual `Shape.count(shape)` word decodes to `Shape.elements(shape)` for every existing lawful shape, using its stored size equality and capacity contract internally. |
| `shape_count_within` | The actual shape-count word is U32-bounded by 1048576, derived from the exact count theorem and existing shape capacity. This is the bound consumed by the native-word allocator proof. |

The original eight laws, APIs and implementation definitions remain unchanged;
four exact-capacity laws and proof-only lemmas are added. No existing law or
definition outside this new module is changed. There are
no axioms, unsafe definitions, open proof holes, or caller-supplied codec
equivalence hypotheses.

## Exact capacity proof organization

The U32 literal's 32-bit constructor representation equals
`U32{Arith.unit(32n, 20n)}`. The existing symbolic `Arith.unit_nat` theorem
connects that single-bit word to `Model.capacity(20n)`. A symbolic U32 wrapper
establishes the representation coercion before specializing the bit index,
so the closed endpoints retain identical `U32.to_nat` heads instead of forcing
the checker to compare a million unary constructors.

A general proof establishes
`Model.capacity(Nat.add(a, b)) == Nat.mul(Model.capacity(a), Model.capacity(b))`.
Its closed index is transported through the proved small equality
`Nat.add(8n, 12n) == 20n` while still symbolic. The independently certified
small factors are `Model.capacity(8n) == 256n` and
`Model.capacity(12n) == 4096n`.

The remaining exact literal identity is
`Nat.mul(256n, 4096n) == 1048576n`. A symbolic lemma proves
`Nat.add(4096n, tail) == 4096n+tail`. The literal-product step transports a
predecessor multiplication equality through that fixed-size addition. Sixteen
named checkpoint lemmas each contain sixteen such steps, proving the same
capacity literal by 256 bounded constructor chunks. The deterministic
`generate_blocks.py` in the evidence directory records the numeric proof-term
generation; every generated step is checked independently by the source
checker and formal kernel. No generated equality is accepted as an axiom.

Composing these equalities proves `capacity_word_nat`. Transporting the existing
`Shape.LE` witness across that equality then supplies the already certified
symbolic `natural_word_exact` theorem's U32 bound. The count APIs derive their
size bound from `Shape.SShape`'s existing stored equality and capacity witness,
not from a new caller premise or bounds on individual factors of zero products.
They therefore cover empty shapes with arbitrary U32 extents, as well as
nonempty and rank-zero shapes. Public copy success and actual inverse-operation
restoration remain separate obligations for the transform consumers; these
APIs do not claim those integration obligations are already complete.

## Certification and controls

Certification used a local Linux x86_64 worktree at reviewed shape-layer HEAD
`1e43717834311191fe4d3a977d622da3cf44ca66`, with the pinned Bend2 commit
`947db722640c86247849343657bf2f7ef01cb7f1` (Bend 2.0.34). Source fingerprints
match `verification/references.json`.

The original 23-check and 30-check checkpoints remain immutable in
`<scratch>/s2-word-conversion-01/` and
`<scratch>/s2-word-conversion-02/`. Fresh evidence for the exact capacity
extension is preserved outside the repository in
`<scratch>/s2-capacity-conversion-01/`. It includes the source and kernel logs,
`PROOF.bendtt`, `inventory.json`, `hashes.json`, `checks.json`, accepted and
rejected controls, the previous source snapshots, the previous checkpoint's
file hashes, the generated proof-term inputs, and the reproducible `verify.py`
script. This run verifies that all eight previous laws and all previous
implementation definitions are unchanged. The distinction between source
literal semantics and compiler lowering remains explicit above.

```sh
export BEND_NO_TELEMETRY=1 BENDTT="$PWD/.tools/kernel/bendtt"
bun .tools/bend/bend2/main.ts src/tensor/word-conversion-PROOF.bend --check-only
bun .tools/bend/bend2/main.ts src/tensor/word-conversion-PROOF.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/word-conversion-PROOF.bend \
  -o ".evidence/tensor-capacity-conversion-01/PROOF.bendtt"
.tools/kernel/bendtt ".evidence/tensor-capacity-conversion-01/PROOF.bendtt"
uv run --locked python "<scratch>/s2-capacity-conversion-01/verify.py"
uv run --locked python "<scratch>/s2-capacity-conversion-controls-01/verify.py"
```

The proof and accepted controls pass source checking, verdict checking, export,
and direct kernel checking. Accepted controls cover zero, an inclusive scalar
limit, symbolic bounds, zero/small/symbolic U32 inverses, ranks zero and four, a carry chain, symbolic fitting
coordinates, and the impossibility of fitting through a zero extent with an
arbitrary following extent. Rejected source controls check a wrong scalar
result, a wrong inverse result, an invalid coordinate at its extent, a non-strict increment limit, and
two-bit overflow. The direct kernel also rejects separate exported artifacts
whose scalar, inverse and list law conclusions are deliberately falsified.

The preceding upper-bound run records 30 checks, including the original control coverage.
Additional accepted controls instantiate the all-Nat bound at zero, a small
natural, and a symbolic natural; instantiate the unconditional increment bounds
at symbolic widths and U32 words; and check zero-width and two-bit wraparound
plus a nonwrapping carry. Additional source controls reject a false all-Nat
bound and an invalid strict increment bound. Three further exported artifacts
with falsified upper-bound conclusions are rejected by the direct kernel.

The exact-capacity run extends that coverage with the literal representation
identity, a symbolic conversion under the existing capacity contract, explicit
zero and endpoint conversions, symbolic count APIs, rank-zero scalar counts,
and empty shapes with an arbitrary following U32 extent. Negative source
controls reject a missing existing capacity witness and a wrong conversion
result with typed mismatches. The original wrong-literal-capacity source
control produces only checker-stack exhaustion in both source modes. Those
two outcomes are explicitly inconclusive, preserved as diagnostics, and
excluded from meaningful negative-capacity evidence. The four added law
conclusions also have genuine typed-mismatch falsified-export direct-kernel
controls, including the exact `capacity_word_nat` statement.

Fresh triage and false-value controls are preserved in
`<scratch>/s2-capacity-conversion-controls-01/`. The well-typed false equation
`U32.is_eq(1048576, 1048577) == True{}` is rejected at `wrong_capacity` by both
source modes. A separately exported, valid False-valued comparison twin is
checked with the direct kernel, then its conclusion is falsified to True; the
direct kernel rejects that same false equation without stack or fuel exhaustion.
This native-word comparison does not replace the exact Nat bridge law or its
separate genuine kernel negative. The supplemental `control-triage.json`
classifies the original run as 38 meaningful checks and two inconclusive
resource diagnostics; seven fresh checks bring the meaningful total to 45.
The historical 40-entry record is retained rather than relabeling resource
exhaustion as a semantic rejection.

An initial redundant specialization with a closed 1048576n bound reproduced
the previously observed unary-Nat checker stack exhaustion. Its source and
diagnostic are retained as `closed-capacity-stack-*.bend` and
`check-only-initial.log`. The direct reflexivity/specialization approach still
exhausts the source-checker stack; the authorized exact structural proof above
reorganizes the proof terms without changing any domain, tool or verifier
limit. An optional Base-only division normalization probe accepted by the
source checker exhausted the kernel's fuel on both pinned 2.0.34 and installed
2.0.35. That division route remains suspended, is not part of the certified
source or controls, and is not evidence of a semantic counterexample. Its
separate diagnostics are preserved in the capacity evidence directory.

The standalone export contains these required law declarations:

```text
word_2d_conversion_2d_LAWS.word_increment_exact
word_2d_conversion_2d_LAWS.increment_below_exact
word_2d_conversion_2d_LAWS.natural_word_exact
word_2d_conversion_2d_LAWS.word_natural_exact
word_2d_conversion_2d_LAWS.naturals_words_fitting
word_2d_conversion_2d_LAWS.word_increment_le
word_2d_conversion_2d_LAWS.increment_le
word_2d_conversion_2d_LAWS.natural_word_le
word_2d_conversion_2d_LAWS.capacity_word_nat
word_2d_conversion_2d_LAWS.capacity_natural_word_exact
word_2d_conversion_2d_LAWS.shape_count_exact
word_2d_conversion_2d_LAWS.shape_count_within
```

The implementation declarations have the corresponding
`word_2d_conversion.*` prefix. Aggregate import paths may encode different
module prefixes; the names above are the standalone export inventory.

## Trusted boundary

These proofs establish identities of the formal pinned Base definitions.
They do not automatically certify the compiler's native or JavaScript
arithmetic replacements or their execution. The native/compiler arithmetic
boundary remains explicitly trusted and still needs the tensor gate's
independent native and JavaScript corpus checks. This standalone proof task
does not claim to have run or certified the aggregate tensor gate.
