# Ordered reduction and rank-two matrix multiplication

## Parallel product

`Matrix.matmul` builds its output in parallel. It plans the product and
allocates the output as before, takes a clone of each input array with
`Elementwise.take` (the kept halves go back to the caller), and builds the
output cells as a balanced tree with `Matrix.build`. Cell `p` sits at row
`p / columns` and column `p % columns` and folds
`left[row * inner + k] * right[k * columns + column]` for `k = 0, 1, ...,
inner - 1` from `+0.0`, the terms and order of `natural_dot`, so no output word
depends on the split. The top `Matrix.splits(depth, inner)` levels of the tree
fork, each half with its own clone of both inputs; a subtree below the forks
is built in index order on one thread. The build splits by work: an output
stored at depth `depth` (at most `2^depth` cells) forks

    min(max(depth + floor_log2(inner) - 13, 0), 6, depth)

levels, with `floor_log2(0) = 0` (`Matrix.splits`; `Nat.sub` gives the
`max(..., 0)`). While no bound binds, a subtree spans
`2^(13 - floor_log2(inner))` cells, between `2^13` and `2^14` terms when all of
them lie inside the output. A product of fewer than `2^13` terms does not
fork; the six-level cap leaves bigger subtrees, and the `depth` cap leaves one
cell per subtree.
An output of `[64, 784] @ [784, 128]` forks six levels although it has only
8192 cells; the cap binds there, and each of its 64 subtrees holds 128 cells
of 784 terms, about `2^16.6` terms.

`Matrix.multiply`, `natural_product_loop` and `natural_dot` stay as the
sequential definition: their laws below still state them. The public laws
about `Matrix.matmul` (`successful_matmul_interpretation`,
`matmul_valid_success` and the three error laws) are proved for the parallel
build. `matmul-parallel-proof.bend` proves, for any two balanced input arrays,
that every cell of the build below the count folds the words the arrays hold
at its row and column, and every cell above it is `+0.0`, whatever the split;
`matmul-PROOF.bend` lifts that to `ProductPrefix` through the checked
observations of the inputs that `take` copies. The replayed laws
`matmul_ordered` and `matmul_replayed` are proved over the same build in
`contracts/matmul-parallel-proof.bend` (see
[contract families](contract-families.md)). The tensor gate's parallel probe
(`probes/parallel_builds.bend`) runs products that fork next to
`Matrix.multiply` at 1, 8 and 24 threads and requires the same words. The
inputs of `matmul-64x48x64-rounding`, which forks four levels, round in F32:
adding its cells' terms in decreasing `k` changes 3245 of its 4096 words,
where the other products, of multiples of `1/64`, add exactly in any order.
The same probe runs the parallel `reduce_to` (see
[reduce to shape](reduce-to.md#parallel-build)) on four rounding reductions
next to `ReduceFold.cell`, the masked fold of `reduce_fold_fiber`, cell by
cell. `[256, 64]` to `[256, 1]` and `[64, 16, 32]` to `[64, 1, 1]` have
contiguous fibers and fork one and two levels; `[256, 128]` to `[1, 128]` and
`[64, 16, 32]` to `[16, 1]` have strided fibers, so `Reduce.splits` builds
them on one thread and they check the fiber walk alone. Walking each reduced
axis's digits in decreasing order changes 251, 62, 121 and 15 of their 256,
64, 128 and 16 words.
## Products that read an operand in place

`Matrix.product(lo, ro, left, right)` is a rank-2 product whose operands are
each laid out in an order: `Kept`, as `[outer, inner]`, where term `k` of
output row (or column) `outer` sits at `outer * inner + k`, or `Swapped`, as
`[inner, outer]`, where it sits at `k * outers + outer`
(`Matrix.read_offset`). `matmul` is the product of a `Kept` left and a
`Swapped` right operand. Two instances are public:

- `matmul_nt(x, W)`, two `Kept` operands: `x @ W^T` for `x` of `[rows, inner]`
  and `W` of `[columns, inner]`, cell `(row, column)` adding
  `x[row, k] * W[column, k]`;
- `matmul_tn(g, x)`, two `Swapped` operands: `g^T @ x` for `g` of
  `[inner, rows]` and `x` of `[inner, columns]`, cell `(row, column)` adding
  `g[k, row] * x[k, column]`.

Neither builds a transpose. Each cell adds its terms in increasing `k` from
`+0.0`, and the cells are built with `Parallel.build` exactly as `matmul`
builds its own (`Matrix.product_build`, split by `Matrix.splits`), so
`matmul_nt(x, W)` holds the words of `matmul(x, transpose_copy(W))` and
`matmul_tn(g, x)` those of `matmul(transpose_copy(g), x)`. `Matrix.product_plan`
checks both ranks (`RankMismatch`, the left operand first) and the inner
extents (`InnerDimensionMismatch`). The replayed laws `product_ordered` and
`product_replayed` state the cells for any two orders (see
[contract families](contract-families.md#products-that-read-an-operand-in-place)).
`Linear` uses `matmul_nt` for its forward product and `matmul_tn` for its
weight gradient ([tensor backward](tensor-backward.md)).

A cell runs as one flat loop, `Matrix.product_run`: it carries the reads of
the current term and both read offsets, and steps each offset by its
operand's stride (`Matrix.stride`: 1 for `Kept`, `outers` for `Swapped`)
instead of computing `read_offset` again for every term; `stepped` in
`product-parallel-proof.bend` proves the step lands on `read_offset` of the
next term. The compiler turns the loop into a C loop with no allocation per
term. On this repository's 64-core x86-64 machine, `matmul_nt` of a
`[64, 784]` and a `[128, 784]` operand took 276 ms on one thread and 59 ms on
eight when each term went through closures and recomputed its offsets, and
takes 5.9 ms and 2.6 ms now, with the same words (a program that times 5 to
20 products, three runs each). One cell's terms form one chain of dependent
F32 additions, about 3 cycles each, so the loop alone needs about 3.7 ms on
one thread; the rest is the parallel build's copies of both operands.
`matmul` keeps its own `cell_dot`.

## Public math completeness supplement

The original eight reduction laws and seven matrix laws, their definitions,
and every runtime definition are preserved. The additional public success
laws close the gap between interpreting a successful result and proving that
every admitted input actually produces one in the checked source model.

`sum_valid_success` proves `Done` for the actual typed `Reduction.sum` on
every lawful tensor of every supported shape. `sum_dynamic_valid_success`
derives the actual public `Reduction.sum_dynamic` result by unpacking its
input, proving typed success, and analysing the actual typed result before
packing it. Neither law asks callers for a `LogicalValues` list, observed
values, a conversion certificate, successful reads, or successful allocation.
`sum_valid_result` and `sum_dynamic_valid_result` combine these successful
statuses and erased exact `Done` witnesses with the unchanged ordered-fold,
scalar-shape, and retained-input interpretation laws.

Three laws fix the error of a rejected product. `matmul_left_rank_error` and
`matmul_right_rank_error` say that an operand that is not rank 2 fails with
exactly `RankMismatch` (no 1-D promotion, no batching), and
`matmul_inner_error` says that two rank-2 operands with different inner
extents fail with exactly `InnerDimensionMismatch`, so `[2, 3] @ [4, 2]` is
never a rank or capacity error.

`matmul_valid_success` proves the actual public `Matrix.matmul` result is
`Done` and supplies the unchanged matrix interpretation. Its independent
`MatrixInputs` premise describes the operand dimensions `[M,K]` and `[K',N]`,
equality of the natural inner dimensions, and an output count at most `2^20`.
It does not mention `Matrix.plan`, `Shape.construct`, allocation, conversions,
read or write results, or either production loop. The input tensor and shape
types already establish lawful storage and supported input shapes. The
successful interpretation supplies the actual output plan and `[M,N]` shape,
every ordered dot-product value, and preservation of both operands.

All successful results carry an actual `Result.is_done == True` proof and an
erased returned-value witness with `Done{witness} == actual_result`. These
are not conditional specifications with `Unit` in an admitted failure case.
The combined interpretation certificates use the original interpretations;
success makes their failure branches unreachable rather than changing their
meaning.

The proofs reuse the twelve-law word-conversion bridge, the independent
shape-layer checked flatten/unflatten interpretation and coordinate bounds,
`AllocationProof.core_new_done`, and the transform read/write success
producers. Natural output/reduction cursors are bounded by the shape's exact
logical count. Dot cursors are bounded by the actual U32 inner dimension;
row and column bounds are derived from the valid rank-two output coordinate,
not added as public hypotheses. Coordinate word conversion is justified by
the coordinate's own extent, so a zero product never implies an extent cap.

Empty and scalar reductions still require successful scalar allocation.
Empty outputs still require successful output allocation. For `K=0`, a
nonempty output is successfully allocated and every cell is written with the
unchanged positive-zero ordered dot identity. Arbitrarily large dimensions
paired with zero remain admitted without narrowing the shape domain.

This is a source-model completeness theorem under the existing trusted Base
array/F32 boundary. It does not assert that a physical allocator cannot run
out of memory, that arbitrary backends never fail, or that the native F32
implementation is formally equivalent to the source model. It adds no real
field or associativity laws and changes no numerical tolerance or policy.

The user-authorized live-scrutinee reformulation analyses each actual API
result first and uses its success certificate only in an erased proof field.
It neither computes live data from an erased witness nor substitutes an
assumed successful result. Original failed formulations and checker/kernel
diagnostics remain immutable in `<scratch>/s2-review-math-totality-01`;
the fresh source snapshot and checks use the separate `-02` directory.

Tensor gate aggregate, Python, native/JS corpus, and oracle revalidation remain
the root's responsibility. These supplements do not promote the tensor layer or
authorize the autograd work.

## Verification status: restored Core dependency

After the authorized erased-owner proof packaging resolved the Core quantity
disagreement, the root independently rechecked all fifteen math laws against
the restored dependency. Both source-checking paths and the exported direct
BendTT checks passed from a preserved source tree. No math code or law
statement changed after the earlier `s1-2cd8d04-stable-01` checkpoint.

These module checks do not establish whole-layer readiness. The final aggregate
proof, native/JS corpus, and source-snapshot consistency remain requirements of
the authoritative tensor gate. They do not authorize starting the autograd
work.

The mathematical contract is an **ordered-machine model** using the pinned
Base `F32.add` and `F32.mul` operations. It does not assume associativity,
commutativity, distributivity, an F32 ring, or a proof that the native backend
implements IEEE arithmetic. Native numerical acceptance is a separate check
against the frozen tensor corpus and policy.

## Independent interpretations

Reduction's specification is `sequential_sum(List<F32>, accumulator)`. The
`LogicalValues` relation describes the list independently: its successive
values are the original tensor's checked mathematical observations at natural
positions `cursor`, `cursor + 1`, and so on. It includes an exact checked
natural-to-word conversion at each position. It does not invoke `sum_loop`,
`natural_sum_loop`, or the production reader as its specification. The list's
structure fixes the number and order of values, so the public sum uses exactly
the logical element count and excludes storage padding.

Matrix multiplication's specification is `dot_fold(List<Factors>, accumulator)`,
defined in `dot-fold.bend`.
`LogicalDot` relates each factor pair to independently checked shape/coordinate
observations of the original operands at `[row, k]` and `[k, column]`, in
increasing natural `k` order. Coordinate observations use the shape's checked
flattening interpretation and the storage's mathematical observer, not the
production coordinate reader. Each successful output cell carries a checked
output-coordinate decoding, its factor list, the ordered fold equality, and
the final output observation.

These interpretations preserve the original operands as the reference models.
Reads do not silently change the tensors used by the specification.

## Universal source laws

`reduction-PROOF.bend` fills the following exported laws:

- `successful_loop_interpretation` constructs an independent logical list and
  its sequential-fold equality for every successful loop result, for arbitrary
  fuel, cursor, accumulator, shape, and tensor. It has no supplied-list or
  loop-correctness premise.
- `ordered_loop_preservation` proves that every successful loop result retains
  every checked input observation.
- `ordered_loop_interpretation` and `ordered_sum_interpretation` additionally
  prove the result for any supplied list satisfying `LogicalValues`.
- `successful_sum_interpretation` and
  `successful_dynamic_sum_interpretation` lift the unconditional source
  interpretation and preservation result through scalar allocation and dynamic
  packing. The scalar has shape `[]` and contains the specified sequential
  fold from positive zero.
- The existing `empty_accumulation` and `empty_sum_identity` laws remain proved.

`matmul-PROOF.bend` fills the following exported laws:

- `successful_dot_interpretation` constructs the independently observed factor
  list, its ordered dot-fold equality, and preservation of every checked
  observation of both operands, for arbitrary fuel, cursor, row, column, and
  accumulator.
- `successful_product_interpretation` proves the whole output loop by natural
  fuel induction. Its `ProductPrefix` invariant combines input preservation
  with a `ProductCell` interpretation for **every** earlier logical position.
  The step uses Core's checked write mapping and distinct-index frame model:
  the current write establishes its cell and preserves every earlier cell.
  This is a constructed invariant theorem, not just the definition of a
  `SuccessfulMapping` proposition.
- `successful_multiply_interpretation` supplies the empty prefix and lifts the
  whole-loop theorem through allocation. Its final prefix covers the complete
  logical output count. `successful_matmul_interpretation` lifts the same
  interpretation through planning and dynamic packing, carrying the actual
  rank-two plan and its output shape.
- `zero_inner_dot` proves the zero-length dot result without applying any F32
  operation. `zero_inner_multiply` proves that every logical observation of a
  successful zero-K output is positive F32 zero. `output_dimensions` retains
  the plan's `[rows, columns]` shape contract.

The source certificates are proof-only data with erased mathematical contents.
The mathematical list walks, coordinate interpretations, prefix builders,
and storage mapping/frame proofs are in the law/proof modules; native programs
do not import those modules. Production computation uses Base in-place reads
and writes and scalar F32 operations.

## Reviewed shape cached-size adaptation

The reviewed shape representation carries a cached natural size and an exact
relation to the mathematical element count. `scalar_shape` now supplies the
cached size `1` and its exactness witness. The live sum and matrix output loops
take their fuel from `Shape.elements(shape)`, rather than evaluating
`Shape.logical_numel(Shape.dimensions(shape))`. This matters for valid empty
shapes whose other extents are huge: runtime work follows the cached zero
without walking an unnecessary mathematical product.

All reduction and matrix law statements remain unchanged. Their element-count
interpretations continue to use the independent mathematical count. Public
source certificates connect the cached live fuel to that count through
`Shape.elements_exact`, only inside erased proof projections. The empty-sum
proof uses the shape constructor's exact cached-size witness directly. Neither
adaptation introduces a loop-result premise or a smaller numerical/domain
bound.

## Exports and proof boundaries

Every exported law has an actual `def Laws.<name>` producer in its matching
proof module. The standalone artifacts export these eight names under
`reduction_2d_LAWS`:

```text
successful_loop_interpretation
ordered_loop_interpretation
ordered_loop_preservation
ordered_sum_interpretation
successful_sum_interpretation
successful_dynamic_sum_interpretation
empty_accumulation
empty_sum_identity
```

The standalone matrix artifact exports these seven names under
`matmul_2d_LAWS`:

```text
successful_product_interpretation
successful_multiply_interpretation
successful_matmul_interpretation
zero_inner_multiply
successful_dot_interpretation
zero_inner_dot
output_dimensions
```

The directly checked export manifest also records all 29 local reduction proof
definitions and 44 local matrix proof definitions, including these producers.
These names describe the standalone module artifacts; an aggregate import may
qualify their namespaces differently.

The dependencies are checked source proofs, not extra loop-result hypotheses:

- The reviewed shape layer supplies the cached-size equality and the natural
  checked flatten/unflatten coordinate interpretation. The live cached size is
  linked to the unchanged mathematical statements only in erased certificate
  fields.
- Core's computational reader certificates connect the actual reader pair to
  `Core.source_read_model` and `Core.source_read_keeps`. They provide the
  original checked observations and their preservation after each read.
- `CoreProof.checked_update_model` provides checked write mapping and the
  distinct-index frame relation. The whole matrix prefix producer uses these
  to establish the current cell and preserve every previous cell.
- `Shape.BoolCase` and `Lifecycle.natural_equal` turn a checked cursor decision
  into its exact conversion witness. No closed 20-bit conversion theorem or
  assumed U32 arithmetic bridge is substituted.

The arithmetic boundary is the pinned Base F32 operation model. The storage
boundary is the checked source refinement of Base array accesses; proving the
native implementation of those primitives and F32 operations equivalent to the
source model is outside these math laws. Original input contents are defined by
their checked observers. Public construction/supplied-value and clone laws are
separate dependencies of the full tensor interface, not hidden premises of the
sum or matrix loop theorems.

## Checked cursors, domain, and ownership

The live reduction, dot, and output loops use natural cursor/fuel pairs. Before
each access they convert the cursor with `U32.from_nat` and branch on
`Nat.is_eq(U32.to_nat(index), cursor)` through `Shape.BoolCase`. A successful
branch carries an exact conversion equality; an unsuccessful conversion returns
an explicit storage-index error. There is no default scalar. Matrix output
coordinates are obtained with `Shape.unflatten_checked` and must have exactly
two coordinates, rather than relying on an unproved flat div/mod bridge.

No rank, logical-capacity, dtype, arithmetic order, or numerical-policy limit
was narrowed. The public shape domain remains rank at most four and at most
`2^20` logical elements; matrix operands remain rank two. The previous U32
entrypoints for the private loops remain wrappers around the natural loops,
and the public dynamic result signatures are unchanged.

Successful sum returns the original input owner and a scalar owner.
Successful matmul returns the kept clone of each input owner and an output
owner.
The universal laws prove observation preservation of these retained owners.
Errors in these APIs still consume their input owners because their existing
`Result` signatures contain only the error in the failure branch. This differs
from elementwise's retained-input failure signature and is not a claim of
transactional error retention.

The original interpretation laws establish correctness **on successful
results**. The public completeness supplements above additionally prove source
success for every admitted valid input. Neither set proves native IEEE
implementation correctness or that every backend conforms to the model.
The interpretation prefix supplies checked cursor equalities, while the new
success producers justify each checked cursor independently through the
certified exact-count/capacity word bridge.

## Verification and native evidence

### Restored-dependency recheck

The root's independent recheck is preserved in
`<scratch>/s2-root-supplements-recertified-01`. Its source inventory records
the exact Bend files used by the checks, `closure.json` lists eight reduction
and seven matrix laws, and `report.json` records fresh `--check-only`,
`--verdict`, export, and direct-kernel commands for each module. The same run
also certifies the three public construction/clone supplement laws.

### Reviewed-base recheck

After the reviewed shape layer `2cd8d04` integration and the Core reader/update
stability signal, all 15 unchanged laws passed fresh pinned source checks,
BendTT verdicts, and direct checks of the independently exported `.bendtt`
artifacts. This is the verified snapshot that predates the later Core
supplement and its quantity disagreement, not a check of the subsequently
modified integrated source. Its evidence is in
`<scratch>/s2-math-proofs/s1-2cd8d04-stable-01`; `verified-exports.json`
records the checked law names and every local proof definition, and
`verification-summary.json` records the native assertions.

The expanded native demo passed 23 exact output lines. Its original five F32
words decode to sum `415` and `AB + b = [[59,63],[140,153]]`. The zero-K case
returns a six-element output of positive-zero words and both empty input owners.
The empty sum returns positive zero and an empty retained owner. The
empty-output matrix case retains the six-element right input and its first
value `1.0`. The instrumented executable has identical observations and reports
`STORAGE_COPY_AUDIT 0 0`. Math proof/model builder names are absent from the
emitted native C.

Six counterfeit equalities/witnesses were also rejected at their intended
obligation by both `--check-only` and `--verdict` in fresh reviewed-base scratch.
Those are entrypoint rejection controls; the positive artifacts above are the
ones checked directly by BendTT.

This math task did not rerun the root aggregate proof or frozen corpus and
tensor gate for the reviewed base. The root reruns those after all writers
stop; the pre-integration corpus results below remain historical evidence only.

### Preserved original evidence and reproduction

The original verification was run in a local worktree before the
reviewed shape-layer reintegration. Both proof modules passed
the pinned Bend2 source checker and the pinned BendTT verifier; their directly
exported `.bendtt` artifacts were also checked independently. That source
closure and evidence are preserved under
`<scratch>/s2-math-proofs/pre-base-integration`; those passes are not a
substitute for rechecking the reviewed base and stable Core. Reproduce from
the repository root:

```sh
export BEND_NO_TELEMETRY=1
export BENDTT="$PWD/.tools/kernel/bendtt"
export LEAN_STACK_SIZE_KB=4194304
export CC="$PWD/.tools/llvm/usr/bin/clang-19"
export LD_LIBRARY_PATH="$PWD/.tools/llvm/usr/lib/x86_64-linux-gnu"
EVIDENCE=".evidence/tensor-math-proofs/recheck-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EVIDENCE"
for module in reduction matmul; do
  bun .tools/bend/bend2/main.ts "src/tensor/$module-PROOF.bend" --check-only
  bun .tools/bend/bend2/main.ts "src/tensor/$module-PROOF.bend" --verdict
  bun .tools/bend/bend2/main.ts "src/tensor/$module-PROOF.bend" \
    -o "$EVIDENCE/$module-PROOF.bendtt"
  .tools/kernel/bendtt "$EVIDENCE/$module-PROOF.bendtt"
done
bun .tools/bend/bend2/main.ts src/tensor/matmul-demo.bend \
  -o "$EVIDENCE/matmul-demo"
"$EVIDENCE/matmul-demo" --threads 1 --gpu off
uv run --locked python -m tools.tensor_runtime \
  --output "$EVIDENCE/native-corpus"
```

The original native demo yielded `AB + b = [[59, 63], [140, 153]]` and sum `415`.
The demo now also includes a `[2,0] @ [0,3]` zero-K product, an empty sum, and
an empty-output product with observations of retained input owners.
The pre-integration native corpus executed all 160 successful cases, including 34 sums,
16 matrix products, and the matrix-plus-bias case, plus 30 native API errors,
4 decoder errors, 7 compile-negative cases, and 7 shape cases. All 160 outputs
passed the frozen independent-reference comparison through
`tools.tensor_numerics.verify_envelope`; no tolerance or reference was changed.
The pre-integration packed-array audit reported zero live splits and zero live reconstructions.
These are executed conformance facts, not universal native/backend theorems
or performance measurements.

Source-check/kernel/export transcripts, reduced failed formulations, explicit
negative controls, the native source snapshot and results, and the numerical
acceptance output are retained in `<scratch>/s2-math-proofs` on the
verification host.
They are not repository source files. The rejected formulations are historical
evidence: in particular a reusable live `Tensor` is not `Data`, and erased
reader equalities cannot be projected into live proof computation. The final
certificates keep those eliminations in erased regions instead of weakening
the laws or introducing axioms.

Reviewed-base adaptation and fresh verification work uses the separate
`<scratch>/s2-math-proofs/s1-2cd8d04-01` directory. The root owns the final
new-base corpus and tensor gate evidence.

The preserved checkpoint contains complete reduction/matmul source
interpretation producers. Current integrated-source readiness remains paused
pending the Core quantity-disagreement decision and root revalidation.
Promoting the tensor layer remains the root's review and gate decision; this
document does not promote it or substitute the snapshot checks for the frozen
numerical gates.
