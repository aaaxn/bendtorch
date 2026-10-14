# S1 evidence

These files are copied verbatim from one run of
`uv run --locked python -m tools.s1 --output <dir>`. `root.json`, `shapes.json`,
and `storage.json` are its `report.json`, `shapes/report.json`, and
`storage/report.json`: commands, exit codes, timings, and output and artifact
hashes. `shapes-coverage.json` and `storage-coverage.json` are the two
`coverage.json` files, `storage-copy-audit.txt` is the copy-audit stderr, and
`PROOF.bendtt` is the exported theorem package. Request and output transcripts
stay in the run directory. Before publication, host names and local paths in
`root.json`, `shapes.json` and `storage.json` were replaced by placeholders
such as `<host>` and `<worktrees>`. Nothing else changed.

`verification/s1/PROOF.bend` passed source checking, `--verdict`, and direct
kernel checking of `PROOF.bendtt`. The gate requires the 54 exports listed in
`verification/s1.json`: 31 named laws (16 shape, 15 storage), 19 general
Base representation theorems, and four executable storage entrypoints.

Six shape laws concern the Nat/Fin core and the Nat `Broadcast` certificate.
The other ten state the executable API directly: construction and
broadcasting are complete, flatten, unflatten, both checked projections and
both flat-index projections refine the core functions, broadcast plans keep
their input shapes, and a `Broadcast` certificate determines its output. Error codes and the U32/Nat
conversions at the boundary are cross-checked against an independent Python
oracle, not proved. One Bend harness, compiled to a native CPU binary and to
JavaScript, runs the same shape corpus on both lanes: 237 exhaustive shapes
(rank zero to three with dimensions zero to four, and rank four with dimensions
zero to two), all 56,169 ordered shape pairs, 21 adversarial shapes, nine
additional pairs, 2,854 flatten observations, 1,722 unflatten observations,
333,846 projection observations, and 196,492 flat projection observations. Four non-U32 values are rejected by the
request encoder before they reach Bend.

The owned F32 storage corpus has 55 cases and 2,322 observations on each of the
native CPU and JavaScript lanes, including sparse writes and read-backs at every
depth from 6 to 20 and at the 2^20 capacity. Malformed balance, length, and
depth certificates are rejected through both checking entrypoints with their
own mismatch, and a valid twin of each is accepted. Native generated-C
instrumentation prints `STORAGE_COPY_AUDIT 0 0`: no packed-array splits or
reconstructions execute, including the maximum-capacity case and cloning.

Allocation failure, native lowering, scalar primitives, and hardware remain
trusted boundaries. No automatic GitHub CI or repository protection settings
are asserted by these local gate results.
