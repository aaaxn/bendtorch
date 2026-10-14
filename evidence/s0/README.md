# S0 evidence, 2026-10-03

These four files come from one gate run on a local Ubuntu 24.04 x86_64 host.
Before publication, host names and local paths in `report.json` and
`fingerprints.json` were replaced by placeholders such as `<host>` and
`<worktrees>`, and the digest of `fingerprints.json` recorded in `report.json`
was updated to match. Nothing else changed:

```sh
uv run --locked python -m tools.s0 --output .evidence/s0
```

- `report.json` lists every command with its exit code, expected exit code,
  elapsed seconds and the SHA-256 of its stdout and stderr, plus the SHA-256
  of every file the run wrote. Those files (logs, native binaries, generated
  JS, the mutated proof copies) stay in the local run directory.
- `PROOF.bendtt` is the exported theorem package. Its required declarations
  are listed in `verification/s0.json`.
- `numbers.json` holds the F32 words each lane printed for `probes/numbers.bend`.
- `fingerprints.json` identifies the tools bootstrap installed.

This is one run on one host. Clean-directory reconstruction (root README) is
not recorded here.

## What the run checks

The unchanged upstream parallel-sum proof passes `--check-only` and
`--verdict`. BendTT then checks the exported `PROOF.bendtt` directly and prints
exactly `ALL PROOFS CHECK`. A copy of the export whose `LAWS.tree_is_seq`
statement claims `seq(2^d, 2i)` fails in BendTT with `SOME PROOFS FAIL` in that
law. Four source mutations fail `--verdict` with `SOME PROOFS FAIL` and their
own reason: a false law (`Location: Laws.tree_is_seq`), an unproven law
(`1 TODO found`), a missing `LAWS.bend` import, and the law proved by an
`@unsafe` def (`LAWS.tree_is_seq` listed as relying on unsafe code). Bend's
TypeScript checker rejects those four before BendTT runs; only the falsified
export exercises the kernel's rejection. Renaming a required declaration in
the export fails the inventory check.

Native binaries are built without the CUDA lane (no libcuda or libnvrtc, no
`.gpu` program) and run with one worker thread and the GPU off. Native
parallel sum prints `2147450880`. Durations are not benchmarks.

The array probes show an owner retained after reading, a raw index 5 on a
capacity-4 array reading slot 1 (the index is masked; clamping would read
slot 3), and independent cloned storage. A malformed unbalanced array passes
source checking and native compilation, then exits 1 with a runtime fail-stop.
Safe tensor construction must prevent that.

Every F32 probe operand is offset by a zero computed from the argument count,
so clang cannot fold the arithmetic: the `-O3` native code contains `addss`,
`subss`, `mulss`, `divss`, `cvtsi2ss` and `cvttss2si`. The gate compares every
word against the table in `tools/s0.py`, per lane and exactly, infinity signs
and NaN words included. The lanes agree on every row but one: native passes
the signaling NaN word `0x7f800001` through, JavaScript returns `0x7fc00000`.
`0 / 0` gives the x86 default NaN `0xffc00000` on both lanes.

The isolated `oracle/` project verifies the PyTorch 2.14.1 CPU wheel and source
commit; its 15 fixtures and 13 tests are described in `oracle/README.md`. None
of this establishes tensor implementation, differentiation, native numerical
refinement, or performance parity. Those are later stage gates.
