# S2 review-fix verification

These fixes follow the review of PR #4 at
`0bbfb1906e2b809cc9c78b6a1a706e9b40d45535`. They add thirteen laws without
changing any original statement or its tensor-runtime meaning dependencies:
three public dispatch laws, two constructor completeness laws, three public
elementwise contracts, four reduction completeness/result laws, and one
matrix completeness/result law. Their statements and mechanisms are described
in `tensor-core.md`, `construction.md`, `elementwise.md`, and `sum-and-matmul.md`.
The inventory grows from 168 to 181 declarations: 167 laws and the fourteen
previous tensor entrypoints, with no removal.

## Snapshot identity

The single integrated S2 gate run froze 88 inputs.
The SHA256 of `source-inventory.json` is
`723b8b7ec013b3a67aea5a7603d4a31eeebce9c243409268fbe7f1298b203568`.
Its freshly exported, directly checked proof has SHA256
`5812a44f5b0a8b12318aa5c000a53e40e35b7e48ff8229fad56073725a58475e`;
the compressed package decompresses to that exact artifact. Every frozen
input matched again before publication.

The corpus remains
`e8999f9170f836b86dcfb8462b5f65fc14619a9d3305fa952302ad1dc25fd356`, and the
numerical policy remains
`0a0e28b2e4666aa9a859af9a091cf0a2cad93ac10f7bb4c68413f1250853d9a4`.
S1 source, tensor arithmetic/transform runtime bodies, both lockfiles, the
tool pin, bounds, and tolerances are unchanged. The harness changes which
public arithmetic wrappers it calls and serializes their actual retained
owners; Python still performs no tensor computation in the gate.

## Independent local checks

| Scope | Source | Verdict | Fresh export | Direct kernel |
| --- | --- | --- | --- | --- |
| Integrated S2 proof and all 181 required exports | Pass | Pass | Pass | Pass |
| Released root proof | Pass | Pass | Pass | Pass |
| Constructor producers and all ten original closed cases, both certificate routes | Pass | Pass | Pass | Pass |
| Public elementwise producers and all nine original named public-call cases | Pass | Pass | Pass | Pass |
| Reduction/matrix proofs and scalar, empty, nonempty, zero-inner and arbitrary-zero-extent positives | Pass | Pass | Pass | Pass |

The final elementwise replay checked one operation-specific public producer
and nine consumers: forty checks, each exiting zero. Its 13 frozen scratch
sources and 67-file closure match the manifest in
`<scratch>/s2-review-elementwise-totality-03/`. The independent root record
is `<scratch>/s2-root-final-elementwise-7aO4x2/report.json`. Each actual pair
has an erased equality to its original add/sub/mul call, actual success,
exact original dimensions, and ordered dimension/checked/raw retention.

Six constructor fail-only mutations, four math fail-only mutations, and
wrong-operation public subtraction/multiplication mutations reject in source,
verdict and independently mutated direct-kernel artifacts. Sixteen elementwise
semantic countermodel pairs pass all 112 expected checks, including wrong
empty output dimensions, swapped/replaced owners on success and error, and
raw-only padding changes. These rejections are semantic, not resource failures.

All 66 root tests and twenty-one isolated S2 oracle tests pass, together with
the relevant Ruff, formatting, ty and diff checks. Oracle regeneration checks
preserve every frozen numerical document. The native and generated-Bend JS
lanes each pass 160 numerical cases, seven shape cases, thirty actual API
errors, and 67 ordered public binary-retention observations; native also
records the four decoder and seven static F32 boundaries. Copy auditing remains
`STORAGE_COPY_AUDIT 0 0`.

Earlier direct/bundled and public-result-index huge-zero routes remain
resource-blocked in their preserved reproductions and transcripts. They were
not rerated as passes or counterexamples. The final two-step, actual-pair-binder
route certifies the same nine cases without changing requirements, computing
live from erased information, or increasing any limit.

## CI evidence boundary

Historical: GitHub CI has since been removed and verification is local, so the
section below describes the workflow that existed when it was written.

The post-push Astra high review found one remaining mixed-invalid error-oracle
ordering defect. Transport validation now precedes shape admission for
coordinates, axes, transpose dimensions, reshape targets, and right-input
dimensions as well as scalar indices. Left-shape transport also precedes static
dtype rejection, while that static rejection still precedes validation of
unused operation parameters. Regression tests preserve rank/capacity and
operation-level precedence after transport succeeds; the frozen corpus,
numerical policy, and all 88 proof/gate inputs remain unchanged.

CI uploads only `.evidence/`, includes hidden files and errors if evidence is
missing. A separate deliberately failing proof-verification control writes a
real failed report and checker transcripts. Only that expected-failure step
uses `continue-on-error`; a mandatory follow-up verifies the semantic rejection,
retained files and upload artifact ID. The original positive gates remain
mandatory. This is a controlled failing verification run, not a claim that a
new unexpected failing workflow has already been observed and inspected.

The downloadable artifact and CI result must be checked at the newly published
head. Local verification alone does not certify the remote upload. Base F32,
primitive arrays, compiler lowering, physical allocation, hardware, and
proof-export translation remain the unchanged trust boundaries.
