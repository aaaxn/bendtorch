# Actual allocation totality

The allocation modules prove that actual `Storage.new` succeeds throughout its
unchanged U32 capacity domain and that actual `Core.new` succeeds for every
lawful shape. Both producers certify exact mathematical logical length.
Neither Core producer asks its caller for count equality, capacity conversion,
allocator success, or successful attachment.

## Public outcome certificates

`Allocation.storage_length` maps an actual allocator failure to `None` and an
actual success to `Some` of the returned owner's logical length. The certified
producer has only the unchanged allocator range contract:

```bend
def storage_new(+word: U32, +fill: F32,
  +within: {U32.is_le(word, 1048576) == True{} : Bool}) ->
  {Allocation.storage_length(Storage.new(word, fill)) ==
    Some{U32.to_nat(word)} : Maybe<&2, Nat>}
```

`Allocation.tensor_length` similarly distinguishes actual Core failure from
actual success. Its public producer takes only the lawful shape and fill:

```bend
def core_new(+shape: Shape.Shape, +fill: F32) ->
  {Allocation.tensor_length(shape, Core.new(shape, fill)) ==
    Some{Shape.logical_numel(Shape.dimensions(shape))} : Maybe<&2, Nat>}
```

A `Some` conclusion rules out the actual `Fail` branch, including at logical
length zero. The projection does not assume that allocation already succeeded.
The fill is arbitrary; the length proof does not change floating-point policy.

Consumers that need the actual successful tensor can use the erased witness:

```bend
type TensorDone<-shape: Shape.Shape,
  -result: Result<&2, &1, Core.Error, Core.Tensor<shape>>> is Data:
  TensorDone{-tensor: Core.Tensor<shape>,
    -same: {Done{tensor} == result :
      Result<&2, &1, Core.Error, Core.Tensor<shape>>}}

def core_new_done(+shape: Shape.Shape, +fill: F32) ->
  Allocation.TensorDone<shape, Core.new(shape, fill)>
```

The witness contains the original tensor returned by the actual result, with
its existing owner and logical-length certificate intact. It does not create
a replacement owner or erase a mismatch by rebuilding a tensor. Its fields
are proof-only; it is not an additional runtime allocation API.

## Composition through the unchanged implementations

Frozen `Api.cap_from` derives the allocator's natural capacity bound from its
U32 range contract. For a positive request, frozen `Api.seek` proves that the
actual `Storage.depth_seek` result satisfies every existing `Storage.bounds`
condition: depth at most 20, positive logical length, and enough physical
capacity. The new proof follows actual `new_checked`, `allocate_at`, and
`allocate` branches to produce the exact requested length. Zero is separately
certified as exactly `Done{Storage.Empty{}}` for any fill.

For Core allocation, the certified `WordConversion.shape_count_within` and
`WordConversion.shape_count_exact` derive the required range and count
conversion from the lawful shape itself. The proof transports the allocator's
exact length to `Shape.elements(shape)`, follows the actual `Core.new_result`
and `Core.attach`, and uses the stored shape equality to obtain mathematical
logical numel. The false attachment decision is contradictory; the true
decision retains the original storage in `Core.TTensor` with its existing
exact length certificate.

Internal composition helpers take the equalities they combine, but public
Core callers do not supply them. The proof covers the full intrinsic shape
domain, including rank-zero scalars, non-power-of-two lengths, the capacity
endpoint, and lawful huge-zero shapes. It does not infer bounds on extents or
intermediate products merely because the final product is zero.

## Added laws

- `storage_new_complete` states actual allocator success and exact natural
  logical length for every request in the existing U32 capacity domain.
- `storage_new_zero` states the exact empty-owner outcome for zero.
- `storage_new_depth_bounds` states valid metadata for the actual positive
  request's search result.
- `storage_seek_bounds` exposes the frozen symbolic search invariant for a
  lawful fuel, depth, capacity, decision, positive length, and final bound.
  Those premises describe an internal search state, not public callers.
- `core_new_complete` states actual Core success and exact logical numel for
  every lawful shape and every fill, with no additional caller premise.
- `core_new_done` certifies the erased witness of that actual successful
  tensor and result equality, with the same premise-free public inputs.

All six laws are new. No existing law statement, executable definition,
numerical policy, shape domain, or capacity was changed. The original four
storage statements and their producers remain unchanged when the two Core
laws are appended.

## Controls and evidence

The allocation proof module is checked with the pinned source checker,
formal verdict, fresh BendTT export, and direct kernel. Positive controls
exercise actual scalar and huge-zero Core allocation, the actual capacity
endpoint, symbolic producers over the full domains, and the erased tensor
witness. Source and exported-kernel negative controls reject incorrect length,
zero failure, invalid search bounds, and false attachment/Core failure
contracts. Source controls also reject over-capacity success and a fabricated
witness for a failed Core result.

The JS and native functional control checks zero, small, non-power-of-two, and
endpoint storage allocations, verifies metadata, and observes the last logical
element of every nonempty accepted allocation. It checks precise `TooLarge`
rejection for 1048577 and U32 maximum, and actual scalar and rank-4 huge-zero
Core allocations. These checks are not timings or a replacement for the
universal laws and complete tensor gate.

The direct-reflexivity path for the closed huge-zero constructor exhausted
the normal kernel budget on pinned Bend 2.0.34 and installed latest 2.0.35.
Its reduction and diagnostics are retained and excluded from certified
passing evidence. The exact same dimensions
`[0, 4294967295, 4294967295, 4294967295]` are instead instantiated after a
generic erased zero-product proof and generic lawful empty-shape construction.
The statement and domain are unchanged; no tool, budget, or executable
definition is changed by this proof-term organization.

The independent 19-law axis checkpoint remains byte-for-byte frozen. These
allocation certificates supply the real new-owner prerequisite for tensor
copy proofs; whole-copy success and actual inverse restoration are separate
transform obligations, not consequences of axis algebra or allocation alone.
