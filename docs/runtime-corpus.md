# Tensor native CPU process adapter

`src/tensor/harness.bend` dispatches operations to the tensor modules and
serializes their actual results. `tools/tensor_runtime.py` generates a corpus-specific
Bend entry point, checks it with both verifiers, exports C, and builds and runs
that C with the pinned Clang. This is a native CPU execution lane, not an oracle
implementation or a Python/JavaScript numerical backend.

Frozen `add`, `sub`, and `mul` requests select `Harness.Add`, `Harness.Sub`, and
`Harness.Mul`; those branches call the public `Elementwise.add`,
`Elementwise.sub`, and `Elementwise.mul` functions. They do not bypass the public
wrappers with `Elementwise.binary`. `Harness.Generic` keeps that generic entry
executable in separate native and generated-Bend JS tests using the
discriminating inputs `[2, 5]` and `[3, 4]` for all three operations.

The initial adapter embeds inputs in generated source rather than accepting
JSON inside the native process. Each F32 input is constructed from its raw U32
word with `Harness.from_bits`, which destructures `U32` and constructs `F32`
with the same bits. Python never decodes input words into host floats, performs
tensor arithmetic, supplies expected outputs, or imports PyTorch. Shape-only
requests call `Shape.construct` without tensor allocation. Error requests without
data use `Core.new` to initialize valid input storage to positive zero; requests
with explicit data preserve those words and their invalid lengths.

## Execution and stable interfaces

Run these from the repository root on a Linux x86_64 machine with the pinned
tools bootstrapped:

```sh
hostname && pwd && git rev-parse --show-toplevel
uv run --locked python -m tools.check tensor --output .evidence/tensor-new-run
uv run --locked python -m tools.tensor_runtime \
  --output ".evidence/tensor-runtime-new-run" --javascript
uv run --frozen --no-sync python -m unittest tests.test_tensor_runtime -v
uv run --frozen --no-sync ruff check tools/tensor_runtime.py tests/test_tensor_runtime.py
uv run --frozen --no-sync ruff format --check tools/tensor_runtime.py tests/test_tensor_runtime.py
uv run --frozen --no-sync ty check tools/tensor_runtime.py tests/test_tensor_runtime.py
```

Use a new evidence directory for every run; existing evidence is never
overwritten. No Git/index/branch/commit operation is part of this adapter.
Native invocations include `--gpu off` and `--threads N`, where N comes from
`tools.check --threads` (default 8) and is recorded in `identity.json`; the
`--benchmark` screens run on one thread. The tool environment
uses `.tools/kernel/bendtt` and `.tools/llvm/usr/bin/clang-19`, and the pinned Bend
source is checked against `verification/references.json` before building.

The Python integration interfaces are:

```python
load_corpus(path: Path = CORPUS) -> dict
decode_case(case: dict) -> dict
parse_native_output(text: str, case_ids: list[str] | None = None) -> list[dict]
run_native_corpus(
    tool_root: Path,
    output: Path,
    corpus_path: Path = CORPUS,
    *,
    benchmark: bool = False,
    javascript: bool = False,
    screening: bool = False,
) -> dict
```

`run_native_corpus` returns `results`, `javascript_results`, `summary`, and
`evidence_directory`. `results` contains the mandatory native lane and shared
pretransport/static boundaries; `javascript_results` contains only the separately
executed generated-Bend JS lane when requested.
Results retain the frozen corpus order: finite, IEEE edge, nonfinite, errors,
then shape-only cases. A tensor observation is
`{"id": "...", "result": {"shape": [...], "f32_bits": ["........", ...]}}`.
A native error preserves both its API name and diagnostic provenance:
`{"id": "...", "error": "...", "raw_error": "core:8"}`. Shape-only observations
use `shape_result` with `shape`, `rank`, and `numel`.

An actual public or generic binary return also includes
`"retained": {"left": {"shape": [...], "f32_bits": [...]}, "right": {"shape": [...], "f32_bits": [...]}}`.
These are snapshots of the owners in the actual returned pair, on both success
and failure, not copies of the requested inputs. The existing output `result`
or `error`/`raw_error` fields are unchanged. A failed constructor before the
binary call instead has `"boundary": "construction"`, a core error, and no
`retained` field. Both left and right rank, capacity, and data-length failures
are exercised separately. `matmul_add` retains its existing output-only
observation: its final add operands are the matrix product and bias, not the
fixture's matrix inputs.

`tools.tensor.verify_lane` checks each retained owner against its corresponding
decoded original input on both successful and failing actual calls. Dimensions,
logical element counts, and finite, signed-zero, subnormal, and infinity words
must match exactly. A NaN word must equal the matching input word's actual
construction observation in the same backend lane; the existing cross-lane
NaN-class numerical policy is unchanged. Missing or swapped owners, changed
shapes or words, and a construction-boundary label on an actual call fail
acceptance. Pre-call rank, capacity, or data-length failures must instead have
the construction boundary and no invented retained pair.

The raw-code mapping follows the source enums: Shape codes 1–6, Core codes 7–11,
and Transforms codes 12–16. Core storage bounds code 8 maps to
`FlatIndexOutOfBounds`; Transforms axes-length code 12 and duplicate-axis code 14
both map to `InvalidPermutation`. Matrix rank and inner-dimension errors have
separate `matrix:rank` and `matrix:inner` observations; delegated Shape/Core
errors retain their own domains.

## Strict output framing and boundaries

Every generated case is serialized independently:

```text
tensor:begin:case_id
tensor:shape:[2, 1]
tensor:count:2
tensor:word:1065353216
tensor:word:2147483648
tensor:end
```

The words above describe F32 `1.0` and negative zero. Decimal U32 transport is
converted to eight-digit lowercase hexadecimal words by the parser without
floating-point conversion. It requires one logical word per element, excludes
storage padding, checks shape/count consistency, and rejects truncated frames,
duplicate/missing/reordered case IDs, extra stdout, unknown errors, and invalid
U32 values. Scalars have shape `[]` and one word; any zero dimension has no words.
Shape-only frames replace `tensor:count` with `tensor:numel` and contain no word records.

Binary frames identify the call and serialize each returned owner before its
output:

```text
tensor:begin:binary_case
tensor:binary:called
tensor:retained:left
tensor:shape:[1]
tensor:count:1
tensor:word:1073741824
tensor:retained:right
tensor:shape:[1]
tensor:count:1
tensor:word:1077936128
tensor:output
tensor:shape:[1]
tensor:count:1
tensor:word:1084227584
tensor:end
```

The parser requires exactly one ordered left, right, and output section. Each
retained section must be a complete tensor snapshot with matching dimensions,
count, and logical words; errors and shape-only results are not retained owners.
An actual binary failure uses the same retained sections followed by one output
error. A constructor failure uses `tensor:boundary:construction` followed by its core
error without a call marker or retained sections. The parser preserves raw NaN
payloads as well as finite words, signed zeros, and subnormals. It validates
transport, not semantic identity: the gate must compare the retained left and
right against the corresponding decoded fixture inputs to reject validly framed
swapped, replaced, or corrupted owners.

Generated JS represents F32s as JavaScript numbers and reconstructs words with
`Float32Array`; its existing construction lane canonicalizes NaNs to
`7fc00000`. Native construction preserves the frozen NaN payloads. Retention
checks require exact fixture words for finite values, signed zero, subnormals,
and infinities. NaNs must retain their raw same-lane constructed word and match
the fixture's NaN classification, rather than requiring equal payload bits
between native and JS. The parser itself never canonicalizes a word.

The active `s2-v2` inventory contains 161 tensor successes, 45 expected-error
requests, and 7 shape-only successes. Thirty-four errors execute through the
native API. The [scalar transpose correction](scalar-transpose.md) preserves
all old IDs, moves only scalar `(0,0)` to success, and adds five invalid-axis
controls. The original `s2-v1` corpus and historical evidence remain unchanged.
Four invalid U32 requests are rejected by `decode_case` before any source
transport; their `results.json` entries use `boundary: "decoder"` and include
the rejection diagnostic. Negative dimensions/coordinates/axes are never
wrapped into unsigned words.

The remaining seven dtype requests cannot be expressed as a dynamic dtype
argument: `Core.construct` statically accepts only `List<&2, F32>`. Each gets a
separate generated compile-negative source using a distinct nominal dtype value
type and attempting to pass that list to `Core.construct`. Both source checking
and BendTT must exit 1 and emit the expected-type diagnostic. This establishes
the F32-only interface restriction; it does not implement these dtypes or invent
a runtime `UnsupportedDtype` result. Entries use `compile_negative` and
`boundary: "static_f32_type"`, not `error`.

## Frozen references and independent acceptance

The runtime pins the active corpus and unchanged policy before observing Bend
results. The historical corpus remains available for explicit oracle replay:

| Artifact | SHA-256 |
| --- | --- |
| Active `oracle/tensor/fixtures/s2-v2.json` | `35d996e7147b79444a78862fd1ffbfec44ed8728e5ea242ab96b7719bb251ed2` |
| `oracle/tensor/fixtures/s2-v1.json` | `e8999f9170f836b86dcfb8462b5f65fc14619a9d3305fa952302ad1dc25fd356` |
| `oracle/tensor/numerical-policy.json` | `0a0e28b2e4666aa9a859af9a091cf0a2cad93ac10f7bb4c68413f1250853d9a4` |
| `oracle/tensor/numerical-policy-v3.json` | `e3b1bd0807b1e6f686a57574c64f961fc9de6fedec97f0a74a3bc2142add001f` |
| `oracle/tensor/fixtures/s2-supplement-v1.json` | `64069566511462006eb67044099170ab8db24f73880e13cd362b9e2994fd9460` |

The gate applies policy version 3 (`tools.tensor_numerics.verify_case`), which
runs every version 2 check (`tools.tensor_numerics.verify_envelope`) and adds the exact-word lane for bounded
add/sub/mul and the zero-sign check for sum and matmul; see
[oracle/tensor/README.md](../oracle/tensor/README.md). Version 2 and its retained
evidence are unchanged. Each lane also runs the 29 supplement cases and 2
supplement errors, and requires the sequential-fold word on the 12 cases that
carry `ordered_reference`.

The runner rejects mismatched hashes and rechecks them after execution. It
doesn't adjust tolerances, drop cases, normalize arithmetic outputs, or change
the oracle. The test suite verifies the 2,921 active input/output-word inventory,
all native result shapes/counts, all error semantics, static certificates, and
the copy audit. It also checks exact retained input words for public binary
successes and errors in each lane, with raw NaNs compared to the same lane's
matching frozen construction observation, and
applies the unchanged version 2 policy, `tools.tensor_numerics.verify_envelope`,
to every public binary success. Complete numerical acceptance remains the root gate's job:

```python
import json
from pathlib import Path

from tools.tensor_numerics import verify_case
from tools.tensor_runtime import SUCCESS_GROUPS, load_corpus

corpus = load_corpus()
observed = {
    row["id"]: row
    for row in json.loads(Path("/path/to/evidence/results.json").read_text())
}
for group in SUCCESS_GROUPS:
    for case in corpus[group]:
        verify_case(case, observed[case["id"]]["result"])
```

The authoritative `tools.tensor` gate requires the complete ordered inventory and
independently applies that acceptance policy to each executable lane. It also
checks every required proof export and accepted/rejected certificate twins.
The transform twins observe the actual `Layout.map_index` result: a square
axis swap must not become identity, a rank-three request must not select the
wrong axis, and a cycle on equal extents must be composed with its actual
inverse rather than with itself. The equal-extent cases prevent an unchanged
output shape from hiding a wrong element map. Both checking entrypoints must
reject the false witnesses, while their valid twins must also export and pass
direct BendTT checking.
Its `source/` directory preserves all Bend source and proof roots, the gate and
adapter code, the protected tool references, and the frozen numerical inputs;
`source-inventory.json` records their hashes. A source change during the gate
invalidates the run, so preliminary screening cannot be published as final
gate evidence. The proof checks and BendTT export use this preserved source
tree, not mutable live modules. Every executable source hash must also match
that proof snapshot, so the two lanes cannot certify different implementations.

## Evidence, copy audit, and screen

Evidence preserves the generated `.bend`, emitted C, native executables,
instrumented C/executable, all command arguments/cwds/exits/stdout/stderr and
hashes, snapshots of transitive runtime sources, frozen reference copies,
tool binary identities, and parsed observations. `report.json` hash-addresses
every saved artifact; `identity.json` records hostname, worktree, thread/GPU
settings, and source hashes. Both executable lanes compile the preserved source
snapshot rather than importing mutable live files. Runtime sources are rehashed
at the end so a concurrent source edit cannot silently change the evidence's
provenance. Default runs reject live-source drift. `--screening` explicitly labels
the run as preliminary and records drift in `live_source_drift`, allowing a defect
screen while proof helpers are still being edited; it is not a final-source claim.
The integration tests also label their independent snapshot runs as screening.
Failures retain their actual diagnostics and are never recorded as passing
checks. Runtime tests leave fresh run directories named `tensor-runtime-tests-*`
under `.evidence/` for inspection.

`tools.storage.instrument_copy_audit` instruments the emitted C without
changing the shape and storage code. It counts `blk_half` splits and
`blk_node` joins. The audited executable must reproduce the same parsed
observations and report exactly `STORAGE_COPY_AUDIT 0 <joins>`, where `<joins>`
is what the parallel elementwise and matmul builds must make for the requests
the corpus runs. Each build assembles its output as a balanced tree and joins
every inner node once with `ANode`, so an output of `n > 0` cells, stored at
depth `ceil(log2 n)`, costs `2^depth - 1` joins; a `matmul_add` request runs
both builds. Empty outputs and calls that fail at construction, rank, inner
extent, broadcast or capacity cost none. `tools.tensor_runtime` computes the
count from the request shapes alone (771 for the frozen corpus: 723 from the
elementwise builds and 48 from the matmul builds), so any
split, and any join outside these builds, including in construction, clone,
transforms, numerical compute, and serialization, still fails the gate.
Sanctioned storage initialization/clone intrinsics do not increment these
counters. The counters are incremented atomically, so the count is exact at
any thread count.

`--benchmark` runs a minimal separate native executable three times under
`/usr/bin/time`, recording elapsed/user/system time and peak RSS. The screen
contains `add_broadcast_3` (`[2,1,3,1]` plus `[1,3]`),
`matmul_rectangular_0` (`[1,1]` by `[1,1]`), `golden_ab` (`[2,3]` by `[3,2]`),
and `golden_ab_plus_bias` (the same product plus `[2]`). `screen.json` identifies
these inputs and explicitly includes process startup, construction, computation,
and serialization. `driver_wall_seconds` also preserves the evidence recorder's
monotonic wall measurement around the time-wrapped subprocess. Time's coarse
resolution and these small cases do not support throughput, PyTorch parity,
or speedup claims.

## Optional generated-Bend JavaScript executable

`--javascript` compiles the same snapshot-backed `corpus.bend` entry point to
`corpus.js` and invokes it with Bun. Numerical operations remain the tensor
functions written in Bend; the adapter does not implement arithmetic in Python
or handwritten JavaScript. This executable lane avoids `.mjs` library-export
rooting of unrelated generic proof helpers. It is independent runtime evidence,
not a claim that the library ABI exports successfully.

Generated IO sequencing uses per-case functions and batches of 16 cases inside
one executable, preserving the complete case order. A first actual attempt with
all 197 frames directly in one `main` exported successfully but Bun failed with
`Maximum call stack size exceeded` before any framed stdout. That failed source,
JS, and command evidence remain in `s2-runtime-js-screen-v1`. Batching avoids the
deeply nested generated JS main expression without changing numerical code,
skipping cases, or patching the pinned compiler.

`javascript-executable-results.json` contains the 202 actual JS executable
observations: all 161 tensor cases, 34 API errors, and 7 shape-only cases. The
four U32 decoder rejections and seven static dtype certificates remain shared
adapter/type boundaries rather than being falsely labeled JS runtime errors.
`coverage.json` identifies the lane as `generated_bend_javascript_executable`;
the root must apply the unchanged numerical comparator separately to its
successful observations. No exact native/JS NaN payload equality is asserted.

## Tensor gate and supported interface

`tools.tensor` requires both executable lanes, applies the frozen numerical
comparator to each independently, checks every error and shape-only result,
checks the exported law inventory directly with BendTT, and runs accepted and
rejected certificate twins. It also rejects source drift and a native binary
that links CUDA or PyTorch. A JavaScript failure is fatal to the tensor gate;
native success does not substitute for it. The standalone runner keeps the
native evidence when JS fails, but its CLI exits nonzero.

Both reference lanes also require the unchanged 51 ordered reduction records
and two separate cancellation/fusion controls under the
[ordered-reference contract](../oracle/tensor/ORDERED_REFERENCE.md). Their verdicts
are separate from numerical compatibility; the scalar corpus correction does
not change those records or relax either contract.

The actual mapping controls include nonidentity permutation, wrong-axis and
wrong-inverse twins, plus requested-transpose, wrong-pair and wrong-inverse
transpose twins. The transpose pairs use equal extents so shape equality cannot
hide a wrong element order. Every accepted twin passes both checking modes and
a fresh exported direct-kernel check; every rejected twin must fail at its
semantic witness, not merely exhaust a verifier resource budget.

The public tensor interface consists of the Bend modules imported by a Bend
program. Neither executable conformance nor these source APIs establish a
JavaScript module ABI. The unsupported experimental `.mjs` bridge is not part
of the tensor layer; no handwritten JavaScript performs tensor computations.
