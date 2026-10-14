# S2 evidence

The JSON files in this directory come from one completed run of
`uv run --locked python -m tools.s2 --output <fresh directory>` on a local
Linux x86_64 host. Before publication, host names and local paths in
`report.json` and `runtime/identity.json` were replaced by placeholders such as
`<host>`, `<worktree>` and `<scratch>`, and the digest of `runtime/identity.json`
recorded in `report.json` was updated to match. Nothing else changed.
`report.json` records its 74 expected-outcome proof, boundary and
certificate-control commands, exit codes and artifact hashes.
`source-inventory.json` identifies its 88 frozen inputs. `acceptance.json`
records the independently evaluated numerical, error, shape and corresponding
retained-operand results.

`runtime/identity.json`, `coverage.json`, `requests.json`, `decoder-results.json`,
`native-results.json` and `javascript-executable-results.json` are the original
runner records from that same run. Compiler outputs, binaries, full source
copies and detailed command transcripts stay in the original run directory;
they are not additional repository inputs.

## Export and proof scope

`PROOF.bendtt.zst` losslessly compresses that run's unchanged `PROOF.bendtt`.
Its decompressed SHA256 is recorded in `proof.sha256`; compression is only
storage packaging, not a change to the theorem artifact or verifier. The
export passed source checking, formal verdict and direct BendTT checking.

`verification/s2.json` requires 181 declarations: 167 named S2 laws and 14
tensor entrypoints. The laws comprise 38 Core, five construction, three
elementwise, four
projection, 12 word conversion, six allocation, 19 axis, three requested
transpose, 57 transform, twelve reduction and eight matrix laws. All S1
definitions remain unchanged. The exact capacity bridge, premise-free
allocation, mandatory public transform success, individual requested mapping
and actual inverse restoration are documented in
`docs/s2-capacity-bridge.md` and `docs/s2-transforms.md`.

To check the retained package independently with the pinned kernel:

```sh
mkdir -p .evidence/s2-retained
zstd -dc evidence/s2/PROOF.bendtt.zst > .evidence/s2-retained/PROOF.bendtt
(cd .evidence/s2-retained && sha256sum -c ../../evidence/s2/proof.sha256)
LEAN_STACK_SIZE_KB=4194304 .tools/kernel/bendtt .evidence/s2-retained/PROOF.bendtt
```

## Executable conformance

Both native CPU and generated-Bend JavaScript independently satisfy the
unchanged policy for all 160 numerical cases and seven shape-only cases.
The native acceptance inventory includes 30 actual API errors, four shared
decoder rejections and seven static F32 type certificates; JavaScript has the
same 30 actual API errors and shares the adapter/type boundaries. Unsupported
dtypes are not invented runtime errors in the F32-only API.

The native split/reconstruction instrumentation reports
`STORAGE_COPY_AUDIT 0 0`. Raw floating-point words remain recorded. Finite
exact cases include signed zero and subnormals; NaN classification is required
but portable native/JavaScript payload equality is not. No numerical tolerance
or mandatory case was altered.

All 67 actual public add/sub/mul returns per lane preserve their corresponding
input dimensions and logical raw words on success and error. NaN retention
compares the actual returned word with the matching same-lane construction
observation; it does not silently canonicalize a returned owner or strengthen
the unchanged cross-lane arithmetic policy. Wrong-operation wrapper mutations
are rejected by the new dispatch laws and executable controls.

This completed aggregate gate is distinct from the supplementary closed
elementwise huge-zero controls. Their historical resource failures are not
counted as successful checks or semantic counterexamples; see
`docs/s2-elementwise.md` for the certified universal contracts and the separate
closed-control status.

Base scalar semantics, compiler lowering, allocation, hardware and proof-export
translation remain the disclosed trust boundaries. These local records do not
claim CI success, performance results, a JavaScript module ABI, or universal
equivalence with an earlier codec.
