# Exact capacity-conversion obligation

This document fixed the obligation before the authorized structural/symbolic
attempt. The bridge, shape-count applications, public copy-success and actual
inverse applications are now certified on the frozen proof graph. The
executable definitions, numerical policy, mandatory corpus,
independent axis specification and admitted shape domain stay unchanged.

## Exact statements

Let `C = 1048576n : Nat` and `W = 1048576 : U32`. The representation bridge is

```bend
{U32.to_nat(W) == C : Nat}
```

The conversion theorem needed by the consumers is stronger than a range bound:

```bend
for +number: Nat
for within: Shape.LE(number, 1048576n)
{U32.to_nat(U32.from_nat(number)) == number : Nat}
```

The `within` condition is the existing capacity contract, not a new caller
premise. For a lawful shape it is derived from `Shape.SShape.capacity_bound`
and the stored size equality. For a cursor or a row-major offset it is derived
from its strict logical-count bound and that same shape capacity contract.
No consumer may be asked to supply the representation bridge, a conversion
equality, success of an intermediate operation, or a smaller capacity bound.

The already certified symbolic `WordConversion.natural_word_exact` proves the
conversion identity when the natural value is bounded by `U32.to_nat(limit)`
for a U32 limit. Instantiating its limit with `W` therefore requires connecting
the existing `C` bound to that exact representation. The separate theorem
`natural_word_le` only says that conversion cannot increase the natural value;
it does not preserve information and cannot replace this exact bridge.

The source checker represents a natural literal as a compact literal whose
constructor meaning is unary. Its `U32.to_nat` definition instead computes
from the word bits through `Word.to_nat` and `Nat.double`. The executable
compiler lowers natural literals above 256 through `U32.to_nat`; that lowering
is not itself a formal source proof of the bridge. The previous closed
reflexivity attempt overflowed the checker stack before kernel export.

## Required applications

1. **Shape count and allocation.** Prove
   `U32.to_nat(Shape.count(shape)) == Shape.elements(shape)`, then use the
   existing size equality to connect the actual storage allocation length to
   mathematical logical numel. Prove the real allocation succeeds, rather
   than interpreting only an already successful allocation.
2. **Copy cursors and checked mappings.** For every natural cursor below the
   output logical count, prove exact conversion, actual mapping success,
   actual source-read success and actual destination-write success. Combine
   those producers with the existing whole-loop interpretation to prove
   success of the public permutation and transpose operations for every
   admitted input.
3. **Flatten followed by decode.** A fitting coordinate's natural row-major
   offset is below the nonempty shape's logical count. The exact conversion
   identity must preserve that offset through the actual `flatten_words`
   U32 result before the actual decoder is applied. The existing opposite
   theorem, decode followed by flatten, does not establish this direction.
4. **Actual inverse operations.** Apply the independently specified inverse
   axes of the requested permutation to the actual first public result;
   transpose uses the actual inverse transpose. Prove both calls succeed,
   restore the original dimensions, and restore the original logical-index
   `Core.observe` values. Keep each individual transformation's independent
   element-mapping law alongside the roundtrip law, so an identity substitute
   cannot satisfy the contract.

`Core.observe` is an unchecked mathematical storage projection, whereas
`Core.read_flat` and the existing copy result contract concern logical
indices. The inverse-copy observation obligation uses that existing logical
index domain; it does not silently claim equality of arbitrary padding or
out-of-range physical projections. Reshape transfers the same owner and its
already certified stronger all-U32 observation law remains unchanged.

## Empty and nonempty shapes

For an empty shape there are no logical indices or fitting element
coordinates, but plan construction, allocation, public success, selected
dimensions and inverse shape restoration remain obligations. Empty shape
extents may be arbitrary U32 words, including huge-zero cases. No proof may
infer an extent or intermediate-product capacity bound merely from a zero
total product.

For a nonempty shape, any factor or intermediate offset bound actually used
must be derived from positivity, coordinate fitting and the existing total
capacity bound. It is not an additional input restriction. The bridge itself
includes both `number = 0` and the closed capacity endpoint.

## Preservation and acceptance

New auxiliary lemmas and proof-term reorganizations are permitted. They must
be checked with the pinned source checker, formal verdict, fresh exported
declarations and direct BendTT kernel. Altering executable definitions that
change existing law meanings, adding external assumptions, axioms or unsafe
paths, or changing policy/domain is outside this authorization.

The frozen checkpoint remains the control: 32 transform laws, 19 axis laws,
eight word-conversion laws, 130 required exports, both executable corpus
lanes, the inherited proofs, negative identity/axis/inverse controls, tests,
lint, format and type checks. Zero differences in the preserved corpus mean
observed equivalence in that corpus, separately per backend; they are not
universal old/new codec equivalence or a substitute for this bridge. The
existing JavaScript NaN-payload classification stays unchanged.

The required public-success and actual-inverse applications are now covered by
the certified laws described below. The tensor layer remains unpublished and
autograd work remains unstarted; verification success does not authorize
publication.

## Certified exact bridge checkpoint

`WordConversion.capacity_word_nat` proves the representation identity exactly.
It transports the word's bit-20 unit to `Model.capacity(20n)`, proves the
symbolic capacity-product identity at `8n + 12n`, and connects that product to
the unchanged literal through 256 checked 4096-constructor additions arranged
in 16 named proof checkpoints. It does not use division or executable literal
lowering as an assumption.

`capacity_natural_word_exact(number, within)` converts the existing
`Shape.LE(number, 1048576n)` bound to the certified word-natural limit and
applies the preserved symbolic conversion theorem. `shape_count_exact(shape)`
and `shape_count_within(shape)` derive their bounds internally from the stored
shape contracts; they require no extra caller premise.

Root independently rechecked the frozen 12-law conversion module in both
checking modes, exported it again and checked the export directly with BendTT.
The export SHA256 is
`3802a2e261b9afeea6b31adde7c712ac08c6058b9896bba0177af8e6eaf0c734`;
the immutable root evidence is in `<scratch>/s2-root-capacity-certified-01/`.
The prior eight laws and implementation definitions are retained verbatim.

Two optional direct-reflexivity normalizations remain excluded from certified
evidence: closed division of the capacity by 4096 and closed multiplication of
zero by `U32.to_nat(4294967295)`, both of which exhausted the kernel's fuel.
The same zero-product formula was independently accepted in both checking
modes and by the direct kernel when its proof applies a generic erased
`zero_mul(value)` lemma. No value, domain, verifier budget or executable
definition changed; the closed resource-exhausted attempts are retained as
diagnostics rather than passed proof controls.

## Applied obligations

`AllocationProof.core_new` and `core_new_done` use the count and range producers
internally to prove actual allocation and attachment success for every lawful
shape. Their callers supply no conversion or allocator-success premise.
`codec_decode_flatten` uses exact natural row-major offsets to certify the
previously missing actual decoder direction. `mapping_valid_success` and the
copy step, natural step and complete-loop producers then derive intermediate
success rather than accepting it from callers.

`permute_valid_success` and `permute_typed_success` prove actual public success
from the existing bijective-axis contract. `permute_inverse_restoration`
contains exact equalities for both the actual first public call and the actual
second `permute_dynamic` call using the independent inverse of the requested
axes; it restores original dimensions and logical-index `Core.observe` values.

The independent natural-position swap is connected to the unchanged actual
transpose generator by `transpose_axes_requested`. `transpose_requested_result`
protects its individual requested dimensions and element map, alongside
mandatory `transpose_valid_success` and `transpose_inverse_restoration` for
the actual second `transpose_dynamic` call with the same requested axes.
The roundtrip does not substitute for the individual transformation law.

Separate controls for `[0]` and
`[4294967295, 4294967295, 0, 4294967295]` certify first public success and
requested dimensions, then mandatory second-call success and original
dimensions, for permutation and transpose. They do not use empty index
quantification as a substitute for public outcomes. Existing shape contracts
are the only shape premises; no large-extent or intermediate-product bound is
inferred from a zero total product.

The complete transform graph has 57 laws with all previous 32 statement blocks
preserved. The final aggregate requires 168 exported declarations, including
the six allocation laws and three requested-transpose laws. Root's frozen gate
evidence is `<scratch>/s2-root-final-after-reset-01/`; it separately checks
proof/runtime source agreement and the unchanged numerical corpus in both
executable lanes. Publication and autograd work remain prohibited by the
current user instruction, regardless of verification success.
