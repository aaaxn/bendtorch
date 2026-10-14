# S2 requested transpose axes

`src/tensor/transpose-spec.bend` specifies the requested `a`/`b` exchange
without importing Layout. `swap(A, items, a, b)` independently looks up both
natural-number positions through `Axis.lookup`, then replaces position `a`
with the original value at `b` and position `b` with the original value at
`a`. The lookups are independent parallel calls. If either lookup has no
value, the specification returns the original list; that fallback does not
certify an invalid transpose request.

`natural_axes(rank, a, b)` applies this generic position exchange to the
natural identity list `Axis.indices(rank, 0n)`. `requested_axes` converts that
list to words through `Shape.words`. Neither definition calls
`Layout.transpose_axes` or duplicates its word-equality decision tree.

## Domain and proof method

The three laws quantify over every existing `Shape.Shape` and every U32
`a` and `b` satisfying the existing conjunction of axis bounds:

```text
inside : {True == Bool.and(U32.is_lt(a, Shape.rank(shape)),
                          U32.is_lt(b, Shape.rank(shape))) : Bool}
```

There is no additional extent, capacity, nonzero, axes-validity, inverse, or
requested-output premise. The shape's existing rank certificate supplies the
internal rank-at-most-four bound. The existing frozen word conversion and
axis rank proofs translate the word bounds to natural bounds.

`axes_facts` exhausts all 30 ordered in-range pairs at ranks 1 through 4.
Each case retains symbolic arbitrary U32 dimension values. It certifies the
actual word helper's equality to the independently requested exchange, the
requested order's natural permutation validity, and its independently
computed inverse. The public producers transport these certificates to the
actual helper. The validity bridge uses the frozen axis boundedness,
uniqueness, rank, and word/natural list-length proofs.

The ten same-axis pairs are included: exchanging one position with itself is
the identity order. Distinct-axis exchanges are not replaced by an identity
order. Rank zero and every out-of-range axis contradict `inside`; no proof
producer certifies such a request. This is a bounds rejection, not a claim
that the empty helper output is an invalid empty permutation.

The proof examines list length and axis positions, not dimension products.
It covers zero-element shapes with huge U32 extents without evaluating a
large product or imposing a smaller extent bound. The intrinsic shape
capacity remains unchanged.

## Frozen consumer API

Import `src/tensor/transpose-PROOF.bend` as `TransposeProof`. Every public
producer takes `(+shape: Shape.Shape, +a: U32, +b: U32, +inside: ...)`, with
exactly the bounds proof above. Using the following abbreviations:

```text
rank = Shape.rank_dimensions(Shape.dimensions(shape))
axes = Layout.transpose_axes(Shape.dimensions(shape), a, b, 0)
requested = TransposeSpec.requested_axes(rank, U32.to_nat(a), U32.to_nat(b))
```

| Producer | Result |
| --- | --- |
| `TransposeProof.transpose_axes_requested(shape, a, b, inside)` | `{axes == requested : List<&2, U32>}` |
| `TransposeProof.valid_axes(shape, a, b, inside)` | `{True{} == Layout.bijective(shape, axes) : Bool}` |
| `TransposeProof.inverse_axes(shape, a, b, inside)` | `{Axis.inverse_axes(rank, Shape.naturals(axes)) == Shape.naturals(axes) : List<&2, Nat>}` |

The first producer protects the individual requested transform, rather than
only a composition of two transforms. Identity substitution and a different
valid, self-inverse exchange cannot satisfy this equality. The other two
producers supply the actual validity and inverse witnesses needed by the
existing axis algebra. Tensor interpretation, output selection, storage
preservation, and tensor restoration remain obligations of the transform
integration; these metadata laws do not claim them.

## Added law inventory

All three statements are new. No existing definition, law, proof, runtime,
pin, proof budget, or S1 storage/Core definition is changed.

| Law | Statement protected |
| --- | --- |
| `transpose_axes_requested` | The actual `Layout.transpose_axes(..., a, b, 0)` equals the independent generic exchange of the requested positions. |
| `transpose_axes_valid` | Every bounded request produces actual Layout-bijective axes. |
| `transpose_axes_self_inverse` | The independent computed inverse of those actual axes equals those actual axes. |

## Certification and controls

The pinned Bend2 commit is
`947db722640c86247849343657bf2f7ef01cb7f1` (2.0.34). The law module passed
separate check-only, verdict, fresh standalone export, and direct BendTT
kernel checks. Its three new Bend files were then frozen; subsequent
controls and this documentation did not modify them.

The early API checkpoint is
`<scratch>/s2-transpose-algebra-01/api-checkpoint/`. It contains the
immutable 43-file import closure, exact exported signatures, hashes, and
the standalone law artifact with its successful certification logs. The
artifact SHA256 is
`9987413ae9d8cd9a67e2655179906a56fa6f59d74a3a164dde78ae3846d3bfac`.

The accepted corpus instantiates all 30 ordered pairs twice: once on
ordinary nonempty shapes and once on empty shapes. Rank-one empty input is
`[0]`; ranks two through four include `4294967295` extents after a leading
zero. All 60 pairs instantiate each of the three public producers. Together
with rank-zero and outside-bound rejection witnesses, these 182 proof
instances pass check-only, verdict, fresh export, and direct kernel checks.
The huge-zero fixtures reuse the previously approved erased generic
zero-multiplication proof, without an eager giant-product reflexivity proof.

Six source negatives reject an identity substitution, a different requested
pair, an identity inverse of a nonidentity exchange, an incorrect same-axis
exchange, a rank-zero request, and an outside axis. All six fail both
check-only and verdict. Six corresponding mutations of accepted standalone
declarations fail direct kernel checking at their intended declarations;
none is a resource-failure substitute for rejection. The wrong-pair order
is itself covered by successful validity and self-inverse controls, so its
rejection distinguishes the individual request from mere permutation or
inverse correctness.

The JavaScript and native CPU lanes each return `1` for a 265-assertion
metadata corpus: four independent checks for each of the 60 pairs plus 25
bounds-rejection checks. Each pair checks its literal expected order,
independent requested order, actual Layout bijectivity, and independently
computed inverse. Evidence and the final manifest are retained under
`<scratch>/s2-transpose-algebra-01/`. The protected allocation and axis
checkpoint files, frozen count conversion, S1 HEAD, and pin were rechecked
unchanged at the API and final freezes.
