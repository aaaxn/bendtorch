# Independent axis algebra

`src/tensor/axis-spec.bend` specifies axis order without importing Layout.
Its axes are natural-number indices. `lookup` drops a natural number of list
elements and reads the first remaining element; `gather` selects those indices
in axis order. `position` finds an input axis's position in the axis list.
`scatter` independently looks up that position in the output coordinate list,
then enumerates the input axes in natural-number order. `inverse` scatters the
natural identity list. None of these definitions calls `Layout.lookup`,
`Layout.coordinate_for`, or `Layout.inverse_coordinates`.

The live reference seam is `Axis.scatter(U32, rank, Shape.naturals(axes),
coordinates, 0n)`. The transform reference uses the shape layer's natural digit
decoder, converts its result to words, and passes those words through this
independent scatter. The algebra proves that Layout's word scatter implements
this seam. It does not introduce a separate natural-coordinate source-index
API.

## Domain and proof method

The shape theorems quantify over every existing `Shape.Shape`: ranks 0 through
4, arbitrary U32 extents, and mathematical element counts at most 2^20.
They add no extent bound, nonzero hypothesis, smaller capacity, or numerical
tolerance. In particular, dimensions containing both a huge extent and a zero
remain covered.

`Algebra.facts` classifies the whole rank-0-through-4 axis domain. There are 34
valid orders, including the empty order. Wrong lengths, out-of-range natural
indices, and repeated indices contradict the supplied validity proof; they
are not excluded by a second caller premise. For each valid order, its
`Lists` certificate proves selection, scatter selection, rank, both inverse
list compositions, and scatter after gather for arbitrary Data elements.
Element values and dimensions are not enumerated.

The product proof permutes symbolic natural factors using multiplication
commutativity and associativity. It does not execute a word product or eagerly
normalize a giant product. The coordinate-fit proof permutes the same
component inequalities using Boolean conjunction algebra. The only
natural-to-word conversion bound introduced by the axis bridge is the tiny
axis cursor bound of 4, derived from the existing shape rank bound.

## Consumer API

The principal specification operations are:

```text
Axis.lookup(A, items, index) -> Maybe A
Axis.gather(A, items, axes: List Nat) -> Maybe (List A)
Axis.position(axes: List Nat, axis: Nat) -> Maybe Nat
Axis.scatter(A, rank, axes: List Nat, coordinates, cursor) -> Maybe (List A)
Axis.inverse(rank, axes: List Nat) -> Maybe (List Nat)
Axis.order(A, items, axes: List Nat) -> List A
Axis.inverse_axes(rank, axes: List Nat) -> List Nat
Axis.valid(rank, axes: List Nat) -> Bool
```

`order` and `inverse_axes` extract successful lists and use an empty fallback.
The totality laws explicitly prove that the corresponding Maybe results are
`Some` for valid inputs. A consumer must not treat the fallback as evidence
that invalid axes succeeded.

The principal proof producers are:

| Producer | Guarantee |
| --- | --- |
| `Algebra.lookup(items, index)` | Layout lookup equals independent natural-index lookup, without validity assumptions. |
| `Algebra.reorder(axes, items)` | Layout reorder equals independent gather, including unsuccessful inputs. |
| `Algebra.inverse_coordinates(shape, axes, coordinates)` | Layout scatter equals independent scatter, without an axes-validity or coordinate-fit assumption. |
| `Algebra.valid(shape, axes, accepted)` | Layout's accepted bijection is a valid natural axis order. |
| `Algebra.lookup_complete(A, items, index, bound)` | A bounded lookup has a value and an equality witnessing `Some`. |
| `Algebra.shape_facts(shape, axes, accepted)` | An affine `Facts` certificate for the shape's full rank and natural axis list. |
| `Algebra.facts(rank, axes, bound, accepted)` | The same certificate for any natural axis list with rank at most 4. |
| `Algebra.facts_lists(A, rank, axes, items, length, certificate)` | A `ListFacts` certificate for an arbitrary matching-rank element list. |
| `Algebra.permutation_complete(shape, axes, accepted)` | Actual Layout reorder necessarily returns the independent ordered dimensions. |
| `Algebra.permutation_rank(shape, axes, accepted)` | The ordered dimensions have exactly the input rank. |
| `Algebra.permutation_product(shape, axes, accepted)` | The ordered dimensions have exactly the input mathematical product. |
| `Algebra.scatter_complete(shape, axes, coordinates, accepted, length)` | Actual Layout scatter necessarily returns the independent inverse-order coordinates. |
| `Algebra.scatter_gather(shape, axes, coordinates, accepted, length)` | Actual Layout scatter after forward gather restores the original coordinate list. |
| `Algebra.inverse_valid(rank, axes, bound, accepted)` | The computed inverse axis order is valid. |
| `Algebra.inverse_composition(A, rank, axes, items, bound, accepted, length)` | Forward gather followed by computed inverse gather restores every element. |
| `Algebra.inverse_scatter_composition(A, rank, axes, items, bound, accepted, length)` | Applying inverse scatter to the coordinates produced by original scatter restores the list. |

`accepted` means `True == Layout.bijective(shape, axes)` for shape APIs and
`True == Axis.valid(rank, axes)` for natural APIs. The coordinate-list length
premise is exact rank equality. None of these producers asks the caller to
assume that an inverse is equal, that a composition is an identity, or that a
transform was successful.

`Facts` is affine because its list, product, and fit producers are function
fields. Consume it through the focused accessors, or destructure a parameter
with `Algebra.Certified{...}`. Do not destructure a computed function call.
`axis-PROOF.bend` also exposes `coordinate_fit` and the proof of
`axis_scatter_fit` for the coordinate-validity consumer.

## Added law inventory

All of these statements are new. This work changes no pre-existing law,
definition, proof, pin, storage module, or Core module.

| Law | Statement protected |
| --- | --- |
| `axis_lookup_refines` | Layout lookup implements the independent list/index specification. |
| `axis_gather_refines` | Layout reorder implements natural-index gather. |
| `axis_scatter_refines` | Layout inverse coordinates implement independent position/lookup scatter. |
| `axis_validity_refines` | Accepted Layout axes satisfy independent natural validity. |
| `axis_lookup_complete` | A bounded generic lookup has a selected value. |
| `axis_permutation_complete` | Valid axes require actual Layout reorder success. |
| `axis_permutation_rejects` | Invalid axes cannot produce a successful actual Layout permutation plan. |
| `axis_dimension_rank` | Ordered dimensions preserve the input rank. |
| `axis_dimension_product` | Ordered dimensions preserve the exact mathematical product. |
| `axis_scatter_complete` | Valid axes and a matching-rank list require actual Layout scatter success. |
| `axis_scatter_gather` | Actual scatter of forward-gathered coordinates restores the input list. |
| `axis_inverse_valid` | Computed inverse axes are valid. |
| `axis_inverse_selected` | Valid axes require computed inverse-list selection success. |
| `axis_inverse_involution` | Inverting the computed inverse restores the original axis list. |
| `axis_inverse_list_composition` | Forward then inverse gather restores every arbitrary Data element. |
| `axis_inverse_reverse_composition` | Inverse then forward gather restores every arbitrary Data element. |
| `axis_inverse_scatter_composition` | Inverse scatter restores original-scatter coordinates. |
| `axis_coordinate_fit` | Gathering dimensions and coordinates together preserves component bounds. |
| `axis_scatter_fit` | A fitting output coordinate scatters to a fitting input coordinate. |

## Verification and controls

Verification artifacts are kept outside the repository in
`<scratch>/s2-axis-algebra-01`. The pinned Bend2 commit is
`947db722640c86247849343657bf2f7ef01cb7f1`. The focused law proof is checked
with `--check-only`, `--verdict`, a fresh `.bendtt` export, and a separate direct
BendTT invocation. Immutable source captures and SHA256 inventories accompany
the final exports.

The accepted controls instantiate all 34 valid permutations with arbitrary
matching-rank natural element lists. They include a symbolic huge-zero product
and an equal-extent three-cycle whose correct inverse restores distinct
coordinates. A standalone functional probe has 178 checks covering all valid
orders, inverse selection, gather/scatter restoration, invalid axes, and
reordering U32-max/zero dimensions. Its JavaScript and native executables both
return `1`.

Rejected controls deliberately substitute identity for a nonidentity gather,
shift the scatter's target axis, and replace computed inverse axes with
identity. Separate valid-domain probes reject `[11, 23, 37]` in place of the
required `[23, 37, 11]`, and reject a missing scatter in place of
`Some([0, 1, 2])`. The equal-extent wrong-inverse control keeps dimensions
`[5, 5, 5]` unchanged but fails to restore `[1, 2, 3]`, so dimension equality
cannot conceal a wrong element map. Source-checker logs and deliberately
mutated exported books record the corresponding direct kernel rejections.

These are axis/list algebra proofs and focused functional controls. They do
not, alone, prove final tensor-owner restoration or universal success of the
entire copy pipeline. That requires the transform consumer's storage/copy and
codec exactness proofs. The separate closed-capacity exactness tool limitation
is not bypassed or resolved by this work.
