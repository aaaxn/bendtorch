# Fixed-F32 tensor core

## Contracts stated before implementation

`Tensor<shape>` has one affine `Storage.Storage` owner and an erased equality
between that storage's natural logical length and
`Shape.logical_numel(Shape.dimensions(shape))`. The shape is validated by the
frozen shape API: ranks zero through four and at most `2^20` logical elements.
A dynamic tensor packages this validated shape with its indexed owner. Scalars
have one element; empty shapes have zero elements and no valid coordinate.

Construction takes F32 values in row-major order and rejects an incorrect value
count explicitly. Reads and writes retain the owner on both success and error;
clone returns independent owners. Coordinate errors come from the shape layer,
storage errors are preserved, and internal length inconsistencies are errors
rather than default values. No numerical fallback is permitted.

Binary add, subtract, and multiply retain both inputs and allocate a separate
output. They use the shape layer's right-aligned broadcast plan, including
`0/1 -> 0`. Each checked output flat index is projected directly through the
shape layer’s zero-stride flat index map and read through the owned Storage
adapter. No coordinate roundtrip is materialized. The operation applies exactly
one corresponding Base F32 primitive and writes the result through Storage.
An incompatible or oversized output shape is rejected before allocation.

All executable numerical work is Bend2. Array access, allocation, cloning and
mutation use only the frozen Storage adapter, whose executable array operations
are Base primitives. Mathematical array observations and certificate builders
are proof-only and erased at their executable call sites. Metadata destructuring
does not inspect or repack arrays.

Structural and element-mapping laws describe ownership, lengths, checked
coordinates and the selected opaque scalar operation. They do not assert real
arithmetic, associativity, distributivity, NaN equality, or any other IEEE F32
ring law. Scalar primitive semantics, compiler lowering, allocation and hardware
remain trusted boundaries. Allocation failure may fail-stop the runtime.

## Integration interface

Import `src/tensor/core.bend` as `Core` and the frozen
`src/shape/core.bend` as `Shape`. Indexed owner parameters use the exact shape
value, not merely an equal rank or count. `Core.Packed{shape, tensor}` can be
matched to obtain a validated shape and its corresponding owner.

| Function | Signature / result |
| --- | --- |
| `construct(dimensions, values)` | `List<&2,U32>`, `List<&2,F32>` → `Result<&2,&1,Error,Dynamic>` |
| `from_values(+shape, +values)` | `Shape.Shape`, `List<&2,F32>` → `Result<&2,&1,Error,Tensor<shape>>` |
| `new(+shape, fill)` | `Shape.Shape`, `F32` → `Result<&2,&1,Error,Tensor<shape>>` |
| `attach(+shape, storage)` | `Shape.Shape`, `Storage.Storage` → checked indexed construction; consumes the supplied storage on either outcome |
| `pack(shape, tensor)` | `Shape.Shape`, `Tensor<shape>` → `Dynamic` |
| `shape(dynamic)` | `Dynamic` → `Dynamic & Shape.Shape` |
| `count(dynamic)` | `Dynamic` → `Dynamic & U32` |
| `length(-shape, tensor)` | `Tensor<shape>` → `Tensor<shape> & Nat` |
| `read_flat(-shape, tensor, index)` | `Tensor<shape>`, `U32` → `Tensor<shape> & Result<&2,&2,Error,F32>` |
| `write_flat(-shape, tensor, index, value)` | `Tensor<shape>`, `U32`, `F32` → `Tensor<shape> & Result<&2,&2,Error,Unit>` |
| `read_coordinate(+shape, tensor, coordinates)` | `Tensor<shape>`, `List<&2,U32>` → retained owner and F32/error |
| `write_coordinate(+shape, tensor, coordinates, value)` | `Tensor<shape>`, `List<&2,U32>`, `F32` → retained owner and Unit/error |
| `clone(-shape, tensor)` | `Tensor<shape>` → `Tensor<shape> & Tensor<shape>` |
| `read_dynamic(dynamic, index)` | `Dynamic`, `U32` → `Dynamic & Result<&2,&2,Error,F32>` |
| `write_dynamic(dynamic, index, value)` | `Dynamic`, `U32`, `F32` → `Dynamic & Result<&2,&2,Error,Unit>` |
| `clone_dynamic(dynamic)` | `Dynamic` → `Dynamic & Dynamic` |
| `error_code(error)` | `Error` → `U32` |

`Shape.dimensions(shape)`, `Shape.rank(shape)` and `Shape.count(shape)` expose
copyable shape metadata. The leading `-shape` parameters above are erased;
coordinate operations need a live shape for their checked shape conversion.
`logical_length`, `observe`, `write_observe`, `ConstructionDimensions`,
`ValuesCount`, `LengthValid` and `ObservationChecked` are proof-model interfaces, not retained-owner
runtime observations. In particular, do not substitute a mathematical array
observer for `read_flat` in executable kernels.

The proof-only `read_certificate(-shape, original, +index)` returns
`Read<shape,original,index>`. Matching `Read{tensor,result,exact}` exposes the
computational read pair and a **normal** equality tying that pair to the actual
`read_flat` call. This supports affine proof composition without asking the
checker to eliminate an unknown Base array-read pair. `read_value(-shape,pair)`
and `result_value(result)` project `Done` to `Some` and `Fail` to `None` for
proof interpretation only; the retained-owner runtime API still returns the
original explicit error. `certificate_value` projects the certificate, and its
checked law equates it with the actual read-value projection. This certificate
does not, by itself, assert equality with an independent mathematical source
observation. Certificate builders must not become live executable call sites.

Import `src/tensor/elementwise.bend` as `Elementwise`.
`add(left, right)`, `sub(left, right)` and `mul(left, right)` each return
`(Core.Dynamic & Core.Dynamic) & Result<&2,&1,Core.Error,Core.Dynamic>`.
The first pair is the retained left/right input pair on every outcome.
`binary(operation, left, right)` exposes the same contract with
`Add{}`, `Sub{}` or `Mul{}`. `input_indices(plan, index)` returns an explicit
error or the two checked projected flat indices; it performs no numerical
array work. The output is independent storage, never an input view.

`loop(fuel, operation, plan, cursor, state)` now uses a natural cursor and fuel
taken from cached `Shape.elements(output_shape(plan))`. Its equality with
natural logical numel is used only in erased proofs. It starts at
`0n` and advances by `1n+cursor`. Before a step, `natural_step` converts the
cursor to U32 and checks the actual predicate
`Nat.is_eq(U32.to_nat(candidate), cursor)` through `Shape.BoolCase`. A false
decision returns code 9 (`StorageError{InvalidIndex{}}`) and retains both
inputs. A successful `NaturalStep<shape,cursor>` stores erased candidate and
exact natural-conversion evidence. Construction fill uses the same checked
natural-cursor scheme: `fill_values(values,-shape,tensor,+cursor:Nat)` starts at
`0n` from `from_values_new`; `fill_values_word` preserves the old U32 helper
boundary. Live attachment/value-count decisions also use cached
`Shape.elements`, so huge prefixes followed by zero do not eagerly evaluate a
mathematical product. Successful execution does not rely on an
assumed U32 roundtrip or increment law. Universal conversion totality is a
separate, unclaimed property; this defensive check removes no public shape or
index cases and changes no numerical tolerance or scalar policy.

## Errors and precedence

`ShapeError{error}` preserves the shape layer's codes 1–6: rank, capacity,
coordinate rank, coordinate bounds, flat-index bounds and incompatible
broadcast. Shape construction fails before inspecting the input value count.
Coordinate rank validation precedes coordinate bounds validation. Binary plan
validation precedes output allocation or scalar computation.

`StorageError{TooLarge{}}`, `StorageError{OutOfBounds{}}` and
`StorageError{InvalidIndex{}}` use codes 7, 8 and 9. Flat tensor reads/writes
are checked by Storage and return code 8 for a logical out-of-range index,
including all indices of an empty tensor. `ValueCountMismatch{}` is code 10;
`StorageLengthMismatch{}` is code 11. No failed scalar read substitutes zero.
Initial zero-filled output storage is only an allocation mechanism: a failure
drops the partial output and returns the original inputs with the error.

## Checked laws and proof boundary

`core-LAWS.bend` and `core-PROOF.bend` discharge **38 core laws** on the
reviewed shape base. No earlier law was removed, narrowed or weakened. The
scalar dispatch laws universally pin Add/Sub/Mul to the respective opaque Base
F32 primitive; they assert no IEEE ring identities.

The three added `public_add_dispatch`, `public_sub_dispatch`, and
`public_mul_dispatch` laws equate each actual public wrapper with
`Elementwise.binary` using its corresponding operation. The original scalar
dispatch laws alone did not protect that wrapper selection. No original
statement or executable definition changes.

Actual checked read-value projections refine independent mathematical array
observations through `SourceRead` / `DynamicSourceRead`. Their retained-source
companions universally preserve those observations on reads. Computational
`Read`, `DynamicRead` and `Write` certificates expose the actual retained owner,
result and normal equality to the public method's pair. `UpdateModel` and
`CheckedUpdateModel` certify successful writes and a universal frame on another
previously checked index. All shape-layer array traversal/refinement builders
used for these claims occur inside erased proof fields, never numerical
execution.

### Actual step, prefix, loop and public wrappers

`step_certificate` reifies the real projection/read/scalar/write sequence. It
provides the written route, exact scalar observation, both retained input
interpretations and the output frame. `natural_certificate` adds the actual
checked cursor equality. `invariant_advance` extends a natural prefix: earlier
positions are preserved by the shape layer's primitive frame law and checked
route separation, while the new position gets the selected scalar value. The
logical index equal to the current cursor is identified by structurally proved
U32 natural injectivity; no global conversion-totality premise is supplied.

`loop_certificate` inducts on the real loop's fuel and is tied to the actual
returned pair by a normal equality. Its initial invariant has an empty prefix
and reflexive input observations, not a free frame premise. The loop, cached
fuel, allocation and public binary producers establish `SuccessfulMapping`
for their actual result: on success, every U32 index whose natural value is
below the output's logical numel observes the specified value. Error results
have no numerical observation claim; explicit runtime errors and retained
inputs remain the runtime contract and are exercised by the corpus.

The additive `elementwise-LAWS.bend` contracts separately preserve both
returned owners' dimensions, checked observations, and raw observations on
every exit. They constrain successful output dimensions by the independent
Nat broadcast relation, including empty outputs, and force actual success
when that output meets the unchanged tensor rank and capacity bounds. The
caller supplies no plan, allocation, conversion, projection, or access-success
premise. These contracts strengthen, rather than replace, the original
conditional element-value interpretation.

`expected` shares the executable offset dispatcher, so it alone is not an
independent projection specification. The imported, checked
`projection-LAWS.bend` / `projection-PROOF.bend` close that seam with four
universal laws: `actual_flat_projection`, `input_indices_refines`,
`input_indices_acceptance` and `expected_projection_refines`. They relate the
actual offsets to the independent Nat/Fin unflatten, right-aligned
singleton-to-zero projection and row-major flatten specification under the
plan's own shape-layer Broadcast certificate. `independent_elementwise_loop`
and `independent_elementwise_binary` compose that refinement with the actual
whole-result theorem. A wrong executable projection cannot change both sides
of these independent claims.

The flat runtime helper builds `Shape.index_map(plan)` on a successful output
index and runs `Flat.project_left_index` / `project_right_index`; the removed
coordinate-helper pipeline is unreachable and deleted. This version does not
cache one map across the whole loop. Plan-wide map caching is an optimization,
not a missing observation proof, and has not been substituted for verification.

### Exact core law inventory

- `scalar_add`
- `scalar_sub`
- `scalar_mul`
- `indexed_length`
- `successful_construction_dimensions`
- `successful_value_count`
- `checked_reader_pair`
- `reader_certificate_value`
- `checked_reader_source`
- `checked_dynamic_reader_source`
- `checked_reader_retains_source`
- `checked_dynamic_reader_retains_source`
- `primitive_read_retains_elements`
- `storage_read_retains_length`
- `storage_write_retains_length`
- `storage_clone_first_length`
- `storage_clone_second_length`
- `primitive_clone_first_elements`
- `primitive_clone_second_elements`
- `tensor_write_mapping`
- `broadcast_step_mapping`
- `checked_route_separation`
- `tensor_write_frame`
- `checked_conversion_value`
- `natural_cursor_step`
- `u32_natural_injective`
- `checked_writer_pair`
- `checked_writer_source`
- `actual_elementwise_step`
- `successful_elementwise_loop`
- `successful_elementwise_cached_loop`
- `successful_elementwise_allocated`
- `successful_elementwise_binary`
- `independent_elementwise_loop`
- `independent_elementwise_binary`

The independent public constructor/list-element and clone-outcome laws live in
`construction-LAWS.bend` / `construction-PROOF.bend`, owned and aggregated
separately. The root tensor gate aggregates these modules with transforms,
reductions and matmul; this core handoff is not an assertion that every tensor
gate or pull request has already passed.

### Trusted boundaries and deliberately unclaimed properties

Opaque Base F32 primitive semantics, Bend's compiler/native and JS lowering,
allocation, Clang, operating system and hardware remain trusted. Allocation
or resource exhaustion may fail-stop execution. There is no IEEE associativity,
distributivity, real-arithmetic equality or unconditional natural-to-U32
conversion-totality claim. Successful conversion comes from the actual checked
decision; mismatch returns the existing InvalidIndex error without shrinking
the accepted shape/count domain.

### Preserved diagnostics and authorized source-only contours

The closed unary-Nat word bridge and the owner-convoy mismatch are preserved
under `<scratch>/s2-core/loop-proof`; neither is an assumed theorem. The
recorded binder-order contour puts matched computational certificates before
owner-dependent erased hypotheses and passes both pinned checkers.

A separate 20-line erased-owner double-use reproducer passes source checking
but is rejected by the formal kernel on pinned 2.0.34 and installed 2.0.35 with
`expected *2 / observed *1`. Its source, both certificates, complete logs and
version/commit hashes remain in
`<scratch>/s2-core-reviewed-2cd8d04-final`. The explicitly authorized
source-only contour moves supplementary reasoning to
`independent_binary_reasoning(+operation,-left,-right)`, called once in the
public producer's erased field. The promise, domain, all laws, scalar dispatch
and runtime semantics are unchanged; no pin, compiler or kernel was patched.
The contour was first checked in a new scratch reproducer and then checked in
the full source and exported formal certificate.

## Reproduction and fresh evidence

From the repository root, use the pinned tools:

```sh
export BEND_NO_TELEMETRY=1
export BENDTT="$PWD/.tools/kernel/bendtt"
export LEAN_STACK_SIZE_KB=4194304
export CC="$PWD/.tools/llvm/usr/bin/clang-19"
export LD_LIBRARY_PATH="$PWD/.tools/llvm/usr/lib/x86_64-linux-gnu"
mkdir -p ".evidence/tensor-core-new-evidence"
bun .tools/bend/bend2/main.ts src/tensor/core-PROOF.bend --check-only
bun .tools/bend/bend2/main.ts src/tensor/core-PROOF.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/core-PROOF.bend \
  -o ".evidence/tensor-core-new-evidence/core-PROOF.bendtt"
.tools/kernel/bendtt ".evidence/tensor-core-new-evidence/core-PROOF.bendtt"
bun .tools/bend/bend2/main.ts src/tensor/core-demo.bend --verdict
bun .tools/bend/bend2/main.ts src/tensor/core-demo.bend \
  -o ".evidence/tensor-core-new-evidence/core-demo"
".evidence/tensor-core-new-evidence/core-demo" --threads 1 --gpu off
```

Fresh final core logs/artifacts and the source/law inventory on the reviewed
shape layer `1e43717834311191fe4d3a977d622da3cf44ca66` are in
`<scratch>/s2-core-s1-1e437178-final`. The authorized minimal-contour checks
remain separately in `<scratch>/s2-core-erased-owner-contour-final`.
The demo emits 105 exact
expected lines covering scalar, singleton expansion, both orders of 0/1
broadcasting, non-power-of-two lengths, incompatible shapes with retained
inputs, rank four, invalid coordinates/count/rank/capacity, maximum-U32 index
rejection, retained reads/writes, independent clones and empty access. No
PyTorch runtime or widened numerical tolerance is involved. The emitted-C
packed-array audit must report no split and no join other than those of the
parallel elementwise build, `2^depth - 1` per output of `n > 0` cells at depth
`ceil(log2 n)` (see `docs/runtime-corpus.md`).

Negative controls require their intended type/equality diagnostic as well as
exit status. Old invalid-import parser failures remain preserved separately and
are excluded. The final fixtures reject forged Tensor lengths, count/attach
comparisons, read/write successes, reader certificates, route distinction and
cursor/conversion witnesses. The native false-conversion probe reports error
9 and both retained owners' zero counts. Failed historical artifacts are never
overwritten or counted as passes.
