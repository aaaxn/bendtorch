# S0 reference oracle

This is an isolated, reference-only PyTorch project for the progressively native
Bend2 implementation. The root harness must read the frozen JSON; it must not
import this code or delegate computation to PyTorch. PyTorch is a dependency only
of `oracle/`.

The required source is PyTorch tag `v2.14.1`, commit
`5c4886908584029761b579af026dcfb627c84070`. The CPU wheel's actual runtime version
must be exactly `2.14.1+cpu`, its `torch.version.git_version` must equal that commit,
and its CUDA and HIP versions must both be absent. Generation stops before writing
fixtures if any of those checks fail. The wheel name alone is not verification.

## Frozen environment and regeneration

Run on Linux x86-64 with uv, from this directory:

```sh
cd oracle
uv sync --frozen --python 3.12.12
uv run --frozen python generate.py --identity-only
uv run --frozen python generate.py --check
uv run --frozen python -m unittest discover -s tests -v
```

The development checks use pinned standalone uv tools without adding them to the
reference environment:

```sh
uvx ruff@0.14.0 check --select E4,E7,E9,F,I,UP generate.py tests
uvx ruff@0.14.0 format --check generate.py tests
uvx ty@0.0.84 check --python .venv/bin/python generate.py tests
```

`uv.lock` pins every dependency and artifact hash; `.python-version` selects
CPython 3.12.12. The explicitly selected PyTorch CPU index prevents an accidental
CUDA dependency tree. NumPy is not needed: snapshots use integer views of F32
storage, not NumPy conversion. PyTorch can report an absent-NumPy warning at import.

To deliberately regenerate the versioned references, use:

```sh
uv run --frozen python generate.py
```

Generation uses fixed inputs, CPU F32 operations, one intra-op thread, one inter-op
thread, deterministic algorithms, and the highest F32 matmul precision. No random
inputs or timestamps enter the artifacts. `--check` is read-only and compares
`fixtures/toolchain.json` byte for byte. The unittest suite regenerates twice in
isolated temporary directories under `oracle/` and checks both `s0.json` results
against the frozen bytes.

## Fixture format

`fixtures/toolchain.json` declares the verified `oracle` identity,
`dtype: "float32"`, `device: "cpu"`, and independent `finite_cases` and
`nonfinite_cases`. Each case has an `id`, `op`, named `inputs`, and named `outputs`.
The `matmul_add` operation means `(a @ b) + bias`. All reduction cases reduce the
entire input. No case depends on another case's outputs.

Each tensor has `shape` and `f32_bits`. The latter is a flat logical row-major
array of eight-digit lowercase hexadecimal IEEE 754 binary32 words, written most
significant digit first, independent of host byte order. It preserves negative
zero, subnormals, rounding, infinities, and NaN payloads without JSON `NaN` or
`Infinity` tokens. The number of words is the product of the shape dimensions:
shape `[]` is a scalar with one word; any zero dimension has zero words.

Every nonfinite value additionally has an entry in the tensor's `nonfinite` array
with its flat `index` and `classification`: `nan`, `positive_infinity`, or
`negative_infinity`. The field is absent for wholly finite tensors. Consumers
must cover the separate nonfinite cases and compare classification rather than
applying a numerical tolerance or dropping values. NaN payload words are retained
as observed evidence, not imposed as portable NaN-equality semantics.

The transpose case includes `stride` (in elements) and `storage_offset` for each
snapshot. `parameters.dims` gives the transposed dimensions, and
`parameters.index` identifies the view element overwritten with input `value`.
Its `outputs` contain the view before mutation, then the base and view after
mutation. `alias.shares_storage` is observed from their shared storage pointers,
not inferred from equal values; no process-specific address is serialized.

The finite references cover scalar addition and reduction, empty sum, zero-sized
broadcasting with singleton dimensions (including `[0,3] + [1,3]`), ordinary
broadcasting, F32 edge bit patterns, and a transpose alias. For
`A = [[1,2,3],[4,5,6]]` and `B = [[7,8],[9,10],[11,12]]`, matmul gives
`[[58,64],[139,154]]`; adding `[1,-1]` gives `[[59,63],[140,153]]`, whose independent
sum fixture yields `415`. Nonfinite references cover empty mean, division by zero,
and propagation of NaN and signed infinities.

`fixtures/cpu_config.json` records the host that last regenerated the fixtures:
the verified identity, interpreter and uv versions, lockfile SHA-256, CPU
model/capability, deterministic execution settings, and the actual
`torch.__config__.show()` and `parallel_info()` outputs. It is rewritten by
regeneration and is not checked: another CPU or `OMP_NUM_THREADS` changes it
without changing `s0.json`.
