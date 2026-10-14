# Proof-interface experiment

This directory holds the qualified public proof interface for `Core.new(shape,
fill)`. See [the proof boundary](../../docs/proof-boundary.md) for what it
claims, the controls and the recorded results.

## Files

- `generic-public.bend`: the public producer, `expose` and `consume`. Inputs are
  a lawful shape, a fill, an index and the ordinary in-bounds proof.
- `generic-read.bend`: the value and construction proof over actual
  `Core.new` and `Core.read_flat`.
- `generic-metadata.bend`, `status.bend`: creation status, dimensions and
  counts, without an index premise.
- `retention.bend`: observations on the owner returned by the first read,
  including a second public read.
- `observations.bend`: functions that interpret actual public calls. They do
  no alternate tensor computation.
- `consumers.bend`: literal scalar, `[3]`, empty and both huge-zero consumers.
- `controls.bend`: four value witnesses, `expected_value` (7.0),
  `wrong_value` (8.0) and `wrong_fill_value` (0.0).
- `metadata-control.bend`: the metadata-only control used on the wrong-fill
  copy.
- `run.py`, `test_run.py`: the verification runner and its unit tests.
- `public.bend`, `huge-zero-last.bend`, `minimal-*.bend`,
  `erased-transfer.bend`: earlier smoke probes, kept as history. They are not
  part of acceptance.

## Run

Bootstrap the pinned tools first (`uv sync --locked` and
`uv run --locked python -m tools.bootstrap`). Then, from the repository root:

```sh
uv run --locked python -m experiments.proof-boundary.run --output <fresh directory>
```

To check against another Bend checkout, pass its root and a BendTT kernel built
from its `bend2/bendtt.lean` with the pinned Lean:

```sh
uv run --locked python -m experiments.proof-boundary.run --output <fresh directory> \
  --bend-root <bend checkout> --kernel <bendtt binary>
```

The runner refuses an existing output directory. It runs `--workers`
subprocesses at once (default 8) with `LEAN_NUM_THREADS=1`; bound the CPUs
with `taskset` to your allocation. With `--workers 11` on 28 CPUs a full run
took 3 minutes and peaked near 3.6 GB per process. It exits non-zero on any unexpected exit code or on a rejection whose
diagnostic is not the intended value proposition, and writes `summary.json`
either way.

Unit tests and lint for the runner:

```sh
uv run --locked python -m unittest experiments/proof-boundary/test_run.py -v
uv run --locked ruff check experiments/proof-boundary
uv run --locked ruff format --check experiments/proof-boundary
uv run --locked ty check experiments/proof-boundary/run.py experiments/proof-boundary/test_run.py
```
