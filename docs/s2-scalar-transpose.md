# S2 scalar transpose specification correction

This correction is based on R3 parent
`a295fa66daadb76daa7673d09aa692c13cc14e47`. The widened S2 producers pass the
pinned source checker, verdict, export and direct BendTT check. The scalar
probe also passes those checks and actual native/generated-JS execution.
R1 proof liveness remains a separate gap; no R1 workaround or S3 promotion is
implied by this scalar result.

## Counterexample and corrected domain

The old predicate required `a < rank(input) && b < rank(input)`. It rejected
scalar shape `[]` with axes `(0,0)` even though a scalar has one logical element.
The old law and `transpose_scalar` frozen case explicitly required that error.

Pinned PyTorch `5c4886908584029761b579af026dcfb627c84070` normalizes both axes
with `maybe_wrap_dim` in
[TensorShape.cpp:3752-3755](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/TensorShape.cpp#L3752-L3755).
The default scalar-wrapping behavior is enabled in
[WrapDimMinimal.h:19-38](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/c10/core/WrapDimMinimal.h#L19-L38).
At rank zero,
[WrapDimMinimal.cpp:11-18](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/c10/core/WrapDimMinimal.cpp#L11-L18)
checks the axis as if the rank were one. Dense equal-axis transpose returns a
value identity at
[TensorShape.cpp:3773-3776](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/TensorShape.cpp#L3773-L3776).
This source entails shape `[]` and value `7.0` for scalar transpose `(0,0)`.
The isolated pinned CPU wheel also observes that result when generating the
corrected corpus. Its exact version/source/CUDA/HIP identity is checked before
generation.

The corrected nonnegative domain is `0 <= a,b < max(1,rank)`. Bend axes remain
U32, so negative-axis normalization is still excluded. `Transpose.axis_limit`
and `Transpose.valid` implement this transpose-specific rule. They do not change
the input's true rank, dimensions, logical count or generic permutation bounds.

The accepted branch still calls `permute_copy` with generated transpose axes.
At rank zero both runtime and independent requested axes are `[]`. The existing
copy loop copies the single scalar element into a separately allocated output,
retaining the original owner. There is no alias return or special owner
duplication. Axis one and U32 maximum are rejected with the same axis error and
retained input as other invalid requests.

## Law changes

All names remain. No unrelated law is removed or weakened.

| Law | Old statement | Corrected statement and reason |
| --- | --- | --- |
| `transpose_axis_bounds_rejection` | The negation of strict existing-axis bounds requires exact axis failure and retention. | The negation of `Transpose.valid` requires the same failure/retention. Scalar `(0,0)` leaves this rejection domain because pinned source refutes its previous classification. |
| `transpose_valid_success` | Strict existing-axis bounds require public success. | `Transpose.valid` requires public success, adding scalar `(0,0)` and removing no successful case. |
| `transpose_requested_result` | Strict bounds require requested dimensions, mapping, values and retained owner. | The same conclusions apply to the enlarged valid domain, including the scalar's sole element at flat index zero. |
| `transpose_inverse_restoration` | Strict bounds require both actual requested calls to succeed and restore dimensions/values. | The same conclusions apply under `Transpose.valid`, including both actual scalar transpose calls. |
| `transpose_axes_requested` | Strict bounds require generated axes to equal independent requested axes. | The equality applies under `Transpose.valid`, including the empty scalar permutation. |
| `transpose_axes_valid` | Strict bounds require a bijective generated permutation. | The same conclusion applies under `Transpose.valid`; `[]` is bijective at rank zero. |
| `transpose_axes_self_inverse` | Strict bounds require the generated transpose permutation to be self-inverse. | The same conclusion applies under `Transpose.valid`, including `[]`. |
| `transpose_result_interpretation` | Successful results of the old public operation carry its permutation interpretation. | The written return type is unchanged, but the called operation now succeeds at scalar `(0,0)`, so that branch must carry the same interpretation. |

The proof producers add the rank-zero `(0,0)` `AxesFacts` case and reject its
positive-axis branches. The positive-rank cases remain. Output-rank preservation
now carries the same corrected validity predicate through the second requested
transpose. Existing evidence quantities are unchanged; erased certificates do
not become live guarantees merely because this domain correction was made.

## Corrected active corpus and preserved history

`oracle/s2/fixtures/s2-v1.json`, its CPU profile, the numerical policy, R3 ordered
records and all historical evidence remain byte-for-byte unchanged.
`scalar-transpose-correction.json` retains the source-backed proposal and old
request/ID; it is not relabeled as a PyTorch observation. The separately
generated `fixtures/s2-v2.json` is the corrected active reference.

The observed successful case keeps ID `transpose_scalar`, input shape `[]`, axes
`[0,0]` and supplies word `40e00000` for `7.0`. Output shape is `[]` with that same
word. Every other old case record is identical. Five new scalar invalid-axis
cases cover `(1,0)`, `(0,1)`, `(1,1)`, `(4294967295,0)` and `(0,4294967295)`.
The current inventory is 161 successes and 45 expected errors, including 34
executable API errors, four decoder controls and seven dtype controls. The seven
shape cases, 67 binary-retention checks and 51 ordered reductions remain.
Tolerances and numerical policy hashes are unchanged. The current corpus SHA256
is `35d996e7147b79444a78862fd1ffbfec44ed8728e5ea242ab96b7719bb251ed2`;
the preserved historical hash is
`e8999f9170f836b86dcfb8462b5f65fc14619a9d3305fa952302ad1dc25fd356`.

The isolated generator defaults to the current `s2-v2` contract. Explicit
`--historical-scalar-transpose` replays the original `s2-v1` records for historical
verification only; it is not a legacy Bend execution backend. Generation requires
a fresh output directory and cannot overwrite existing records. `--check` is
read-only. Actual CPU provenance is separate in `cpu-config-v2.json`; the former CI compared
the portable numbers before replacing only its local current CPU profile with
its runner observation. The historical CPU profile is never rewritten.

The independent map uses the stdlib `transpose_reference.requested_axes`.
It validates the corrected domain and explicitly returns `[]` at rank zero,
rather than indexing an empty axes list or catching exceptions. The old
positive-rank coordinate mapping is unchanged. Historical replay uses the
original error classification only to reproduce archived evidence. The single
active runtime loader consumes `s2-v2` with the corrected domain and protected
inventory/hash checks.

Activation follows an explicit by-ID comparison: all 208 old IDs survive, only
`transpose_scalar` changes, and exactly the five named invalid controls are
added. The independent raw-word result equals the actual pinned PyTorch result.
Policy, mathematical goldens and R3 records must remain unchanged; any unrelated
generator/corpus mismatch stops integration rather than triggering regeneration.

## Behavioral and mutation obligations

The lightweight source tests call the actual stdlib reference-axis helper and
existing request decoder/comparator. They exercise rank-zero success, positive
scalar axes, U32 boundaries, unchanged positive-rank mappings, negative-axis
decoder rejection, exact scalar value/shape rejection and the one-ID proposal
change against unchanged frozen hashes. They do not execute Bend or PyTorch.

`src/tensor/scalar-transpose-demo.bend` is a native/JS behavioral probe
using the existing transform demo's owner-threaded reporters. It exercises:

- Scalar `7.0`, negative zero and the smallest positive subnormal, including
  actual repeated public transpose and retained/intermediate/restored values.
- Invalid axes `(1,0)`, `(0,1)`, `(1,1)`, `(4294967295,0)` and `(0,4294967295)`,
  each requiring error code 13 and unchanged retained shape/value.
- A separate scalar copy whose index zero is written to `99.0`; the retained
  source must still be `7.0`. The existing dynamic-write API is used unchanged.

For every scalar success the output and retained shape is `[]`. Original
transport words are `1088421888`, `2147483648` and `1`; writing `99.0` must change
only the output to `1120272384`. The native and generated-JS probe outputs match
these literal words, all retained/intermediate/inverse shapes and all five error
codes. Empty/non-scalar transform cases remain unchanged in both corpus versions.

Native integration should use the existing pinned-tool recipe to check root
laws, export/recheck the kernel and compile/run this probe with `--threads 1
--gpu off`, then emit its generated JS with `-o <fresh-path>.js` and run that file
with Bun. Preserve complete outputs and assert each labeled shape, word and
error, rather than checking exit zero. These jobs require an allocated native
slot and the exact integrated source patch.

Mutate only the transpose predicate back to strict rank bounds and require the
scalar-success law/probe to reject it. Accept all scalar axes and require the
axis-one/U32 failure obligations to reject it. Preserve existing requested-map,
wrong-pair and wrong-inverse controls for positive ranks. Proof syntax failures,
timeouts or unsupported compiler lowering are not successful semantic
negative controls. Public fill/clone/write live-contract work stays deferred to
R1's interfaces.
