# Tensor CPU F32 reference oracle

The active `s2-v2` corpus corrects scalar transpose `(0,0)` against the exact
pinned source and observed CPU wheel. See
[`SCALAR_TRANSPOSE.md`](SCALAR_TRANSPOSE.md) for old/new laws,
hashes and the one-ID correction. The source-backed proposal remains separately
labeled in `fixtures/scalar-transpose-correction.json`; the original `s2-v1`
records, CPU profile, policy and ordered reference records remain byte-identical.
Generation defaults to the current contract. `--historical-scalar-transpose`
replays the old records only; it does not select a legacy runtime backend.

This directory is reference evidence, not an implementation backend. The root's
native Bend runner must consume frozen JSON and compute its own results; it must
not import this generator, invoke PyTorch, or delegate a missing operation here.
There are **no Bend implementation results** in these artifacts.

The existing isolated `oracle/.venv` is used without adding dependencies or
changing either lockfile. Actual identity is verified before generation: PyTorch
`2.14.1+cpu`, source `5c4886908584029761b579af026dcfb627c84070`, no CUDA or HIP.
Generation configures CPU F32, one intra-op and inter-op thread, deterministic
algorithms, highest F32 matmul precision, and gradual underflow. The exact build,
CPU capability, configuration, interpreter, uv version, and both lockfile hashes
are recorded in `fixtures/cpu-config-v2.json`. The absent-NumPy import warning is
preserved in evidence; NumPy isn't needed or installed.

## Artifacts and specification

`numerical-policy.json` is policy **version 2**, frozen before any Bend comparison
and hash-pinned in the generator and corpus. Version 1 and its freeze evidence
are preserved under `evidence/`. Independent arithmetic review amended the
underflow allowance from `n*eta` to `n*eta/(1-n*u)` to cover later amplification;
this correction preceded all Bend results. Corpus `s2-v2` changes scalar
transpose admission only, not the numerical policy. A policy hash mismatch stops
generation; do not loosen a policy in response to an implementation failure.

`fixtures/s2-v2.json` contains 161 independent successful operation cases:
141 finite cases, 3 wholly finite IEEE edge cases, and 17 cases with nonfinite
inputs or outputs. It also contains 45 expected-error requests and 7 independent
shape-only boundary cases. There are 2,921 raw input/output words; nothing is
removed because it is NaN, infinite, empty, subnormal, or numerically awkward.

The tensor encoding is compatible with the toolchain fixture's
`shape`/`f32_bits` encoding, but these cases are independent of it. Words are
flat logical row-major, eight-digit lowercase hexadecimal IEEE binary32 words
in most-significant-digit order, independent of host byte order. Rank zero `[]`
has one word; any zero dimension has zero words. `nonfinite` uses the toolchain
fixture's index/classification entries. The added `classifications` array
covers **every word**, distinguishing normal values, subnormals, both zero
signs, both infinity signs, and NaNs. NaN payload/sign/quiet bits remain
observable in `f32_bits`; their equality is not a portable requirement.

Every successful case has named `inputs`, `outputs.result` (observed PyTorch),
`comparison_policy`, and `independent_reference.result`. Parameters carry reshape
targets, transpose dimensions, permutation axes, or coordinates. Independent
references use raw input words, integer row-major coordinate maps, IEEE special
cases, and exact rational arithmetic with an independently implemented
round-to-nearest-ties-to-even binary32 conversion. Rational numerators and
denominators are decimal **strings**, so consumers must not round them through
JSON binary64 numbers. `independent_reference.exact_values` preserves the exact
mathematical arithmetic separately from rounded words. Reduction `envelopes`
record the exact result, sum of absolute exact terms, rounding count, absolute
error bound, and cancellation condition number, including explicit `infinite`
conditioning for nontrivial exact-zero cancellation.

`fixtures/mathematical-goldens-v1.json` preserves the integer specification,
separately from PyTorch: `AB = [[58,64],[139,154]]`,
`AB+b = [[59,63],[140,153]]`, and its total is `415`. The tests recompute these
from integer inputs without PyTorch and require exact words in the corresponding
oracle cases.

## Numerical acceptance for the root runner

Construction, clone, coordinate reads, reshape, materializing transpose, and
materializing permutation require exact finite words, including signed zero and
subnormals. Infinity signs must match; NaNs must remain NaNs, but payloads are
observations. Reshape preserves linear order. `transpose_copy` and `permute_copy`
materialize their logical results; there is **no storage-alias or stride demand**.
Axes are nonnegative U32, not PyTorch-style negative-axis normalization.

Bounded finite elementwise add/sub/mul start at `atol=1e-6`, `rtol=1e-5` against
the exact mathematical result, and the root must also compare against the frozen
PyTorch result with that tolerance. Inputs satisfy `abs(value)<=16` and are normal
or zero; exceptional arithmetic belongs to separately named `ieee_edges` cases,
where finite results require exact IEEE words and nonfinite results require their
classification. A case's placement in `nonfinite_cases` does not waive checks on
its finite elements.

Sums and rectangular rank-two matmul have no universal tolerance. For each
output, with `u=2^-24`, `eta=2^-150`, and `S=sum(abs(exact terms))`, the disclosed
absolute envelope is:

```text
gamma(n) = n*u / (1-n*u)
bound    = gamma(n)*S + n*eta/(1-n*u)
n        = element count for sum; 2*K for matmul; 2*K+1 for matmul+bias
```

The domain requires `n*u<1`, finite exact outputs, and `S<2^120`, avoiding
intermediate overflow for every admitted accumulation tree. It covers separately
rounded products/additions and fused multiply-add. Each operation can accumulate
in a different order, so both implementations must independently satisfy the
serialized rational bound against the exact result; their mutual difference
must also be at most `2*bound`. This is a conservative forward-error envelope,
not a proof of correct implementation. Cancellation can legitimately make it
large: `[2^24,1,-2^24]` has exact result `1`, while this PyTorch sum observes `0`.
The discrepancy is preserved rather than filtered or called exact parity.

Empty sums and `K=0` matmul require **positive-zero identity words**, even with a
zero error bound. Nonfinite/overflow reductions are separate, unambiguous named
cases, not candidates for the finite envelope. No order-dependent overflow
classification is accepted as a universal reduction rule.

Expected errors specify bounded API semantics, not PyTorch exception text:
rank/capacity, U32 transport range, data length, coordinates/flat indices, axes,
permutation validity, reshape counts, broadcasting, matmul rank/inner dimensions,
output capacity, and unsupported dtypes. `request` is an invalid boundary request,
not a tensor snapshot; shape-only requests can be rejected before allocating
storage. Shape-only successes include the exact capacity boundary and enormous
U32 prefixes with a trailing zero, using independent integer arithmetic rather
than forcing PyTorch allocation or stride behavior. Unsupported dtype cases use
a dynamic tagged interface where available; if the Bend API is F32-only by type,
the root must supply compile-negative evidence instead. These artifacts do not
claim that the Bend error lane has been tested.

## Numerical policy version 3 and the supplement

`numerical-policy.json` (version 2) and `fixtures/s2-v2.json` stay frozen.
`numerical-policy-v3.json` keeps every version 2 check and adds two:

- **Exact-word lane for bounded add, sub and mul.** The observed word must
  equal the round-to-nearest-even binary32 word of the exact result and the
  frozen PyTorch word. PyTorch CPU add/sub/mul is correctly rounded on all 60
  bounded cases in `s2-v2`, so the tolerance no longer admits a word PyTorch
  would not produce.
- **Zero sign in the reduction envelope.** When a sum or matmul output is a
  zero, its sign must equal the PyTorch word's sign if that word is a zero, and
  be positive otherwise. With a positive-zero start, exact cancellation of
  nonzero terms rounds to `+0` in every order, so `-0` there is a wrong result
  that the version 2 envelope accepted.

`fixtures/s2-supplement-v1.json` adds 29 cases and 2 expected errors beside
the frozen corpus; it changes or removes no `s2-v2` case. It covers
`matmul_add` with scalar, row, column, full and rounded biases, cancellation,
`K=0`, a `-0` bias, `-0` and underflowed negative products and invalid bias
shapes; `sum` of `-0` words; `get` on `-0`, subnormal, infinite and NaN words
and on a scalar `-0`; and elementwise operations with a lower-rank left
operand. Twelve reduction cases also carry the `s2_ordered_f32_rne` word of the
sequential fold. The file records the `s2-v2` hash it extends.
`supplement.py` builds it with the same identity check and CPU configuration
as `generate.py`; `--check` regenerates it in memory and compares bytes.

## Reproduce locally

From the repository root, verify hostname/worktree first, then use the existing
oracle environment without resolution, synchronization, or new packages.
`--check` compares `cpu-config-v2.json` byte for byte, and that file records
`torch.__config__.parallel_info()`, which prints the `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` environment variables. The frozen profile was generated with
both set to `1`, so export them before every command below. Without them,
`--check` exits 1 on the CPU profile and two oracle tests fail, although the
numerical corpus itself still verifies.

```sh
hostname && pwd
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/tensor/generate.py --identity-only
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/tensor/generate.py --check
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/tensor/supplement.py --check
(cd oracle/tensor && PYTHONDONTWRITEBYTECODE=1 uv run --project .. --frozen --no-sync python -m unittest discover -s tests -v)
uv run --locked --no-sync ruff check --config pyproject.toml oracle/tensor/generate.py oracle/tensor/transpose_reference.py oracle/tensor/ordered_reference.py oracle/tensor/supplement.py oracle/tensor/tests/test_oracle.py oracle/tensor/tests/test_ordered_reference.py oracle/tensor/tests/test_supplement.py
uv run --locked --no-sync ruff format --check --config pyproject.toml oracle/tensor/generate.py oracle/tensor/transpose_reference.py oracle/tensor/ordered_reference.py oracle/tensor/supplement.py oracle/tensor/tests/test_oracle.py oracle/tensor/tests/test_ordered_reference.py oracle/tensor/tests/test_supplement.py
uv run --frozen --no-sync ty check --python oracle/.venv/bin/python --python-version 3.12 --extra-search-path oracle/tensor --config 'src.include=["oracle/tensor"]' --error-on-warning oracle/tensor
```

Ruff `0.16.10` and ty `0.0.84` come from the root's pinned uv environment; no tool
dependency is added to the oracle. The explicit ty include override is necessary
because the root configuration otherwise includes only `tools` and `tests`.
The initial zero-file invocation is preserved, but is not verification; the final
scoped invocation must actually select the generator, helpers and tests.

To deliberately regenerate, omit `--check` and supply a fresh `--output-dir`.
Generation refuses an existing directory. Outputs are restricted to
`oracle/tensor/`, even when supplying `--output-dir`. The tensor suite
validates all words/classifications, independent references, explicit errors,
goldens, rounding ties/overflow, policy rejection, and mutation controls; it
also runs two actual uv subprocess regenerations in temporary tensor
directories, compares both byte-for-byte with frozen artifacts, and verifies
stale `--check` is read-only. `evidence/` retains actual command outcomes,
including initial diagnostics, precomparison freezes, final verification, and
preservation checks for the toolchain fixtures and both lockfiles. Before
publication, host names and local paths in those transcripts were replaced by
`<host>` and `<worktree>`. No scientific timings or Bend performance results
are reported.

## CI host provenance

Historical: GitHub CI has since been removed and verification is local. The
text below describes the former CI procedure; run the generator and `--check`
recipes above locally.

The committed `cpu-config-v2.json` records the reference host, not a claim that
every runner has that CPU model or SIMD capability. CI preserves that record as
`reference-cpu-config.json`, generates into a fresh directory on its actual
host, and requires zero Git differences in the frozen tensor numerical corpus,
mathematical goldens, numerical policy, toolchain fixtures and both lockfiles.
Only the local, uncommitted CPU provenance record is replaced by the runner's
actual observation and retained as `runner-cpu-config.json` in the uploaded
artifact. No runner identity is fabricated and no numerical field is ignored.

The existing strict `--check` then checks the unchanged portable files and the
copied actual runner profile. Historical `s2-v1` and its CPU profile are never
rewritten. Oracle tests cover the corrected active inventory and explicit
historical replay, including scalar signed-zero/subnormal and invalid-axis calls.
The committed reference profile remains unchanged in the PR. This separates
host-specific provenance from portable frozen numbers without changing the
numerical policy or tolerances. The scalar admission and inventory correction
is explicitly versioned above, with every old case ID preserved.
