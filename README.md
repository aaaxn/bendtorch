# bendtorch

bendtorch is a PyTorch-like tensor library written in
[Bend2](https://github.com/bendlang/bend). All tensor code is Bend. The rules
the code must keep are written as laws in `LAWS.bend` files. The Bend checker
proves each law, and the BendTT kernel, a separate checker built with Lean,
checks the exported proofs again. A separate PyTorch project in `oracle/`
produces frozen reference outputs. Python gates in `tools/` compile the Bend
code, run it natively and as JavaScript, and compare the results with those
references. PyTorch never computes a bendtorch result.

bendtorch is early research code. It is used from Bend programs; it has no
Python API and is not a drop-in replacement for PyTorch. It runs on the CPU,
with F32 tensors only, and it cannot train a model yet.

> **Disclaimer.** bendtorch is an experiment. Most of its code was vibe-coded

## What works today

| Capability | Code | What it does |
| --- | --- | --- |
| Shapes and indexing | `src/shape/` | Validated shapes of rank 0 to 4 with at most 2^20 elements, row-major flatten and unflatten, checked coordinates, and right-aligned broadcast plans. |
| Owned storage | `src/storage/` | F32 storage with one owner: `new`, `get`, `set`, `clone` and `length`, with bounds checks before every access. |
| Tensor core | `src/tensor/core.bend` | `Core.new`, `from_values`, `construct`, flat and coordinate reads and writes, `clone`, and the metadata queries `length`, `shape` and `count`. |
| Elementwise ops | `src/tensor/elementwise.bend` | `add`, `sub` and `mul` with PyTorch-style broadcasting. Each output element is one F32 operation. Both inputs come back to the caller. |
| Sum | `src/tensor/reduction.bend` | Sum of a whole tensor to a scalar, in row-major order, starting from `+0.0`. |
| Matmul | `src/tensor/matmul.bend` | `[m,k] @ [k,n]`. Each dot product adds in increasing `k`, starting from `+0.0`. |
| Reshape, permute, transpose | `src/tensor/transforms.bend` | `reshape` keeps the linear order. `permute_copy` and `transpose_copy` write a new contiguous tensor. |
| Proved contracts | `src/tensor/*-contract-LAWS.bend`, `src/tensor/contracts/` | Laws that later proofs can use as inputs, for construction, reads and writes, clone and metadata, and reshape, plus the error cases of permute and transpose. A generic theorem for pointwise loops proves the values that a broadcast binary operation writes. A second one, for ordered folds, proves the total of a sum and of one matmul dot product, with its addition order. |
| Autograd building blocks | `src/autograd/` | `broadcast_to` (copies a tensor to a larger shape), `reduce_to` (sums a gradient back to the shape of its input), `sum_to_size` (the PyTorch-facing form of `reduce_to`, which keeps the input unchanged when no axis is reduced) and gradient slots (`Gradient.accumulate`, `Gradient.grad`, `Gradient.reset`, `Gradient.gradient`) that add the contributions to a gradient in the order they arrive, read as `None` until they receive one, and reset to `None`. |
| Scalar backward | `src/autograd/backward.bend` | Reverse-mode `backward` for F32 scalars on a tape of `add`, `sub`, `mul` and `neg`. It seeds the output the caller selects, gives `None` to every node that output does not reach, adds the contributions to a node used several times in a fixed order, accumulates repeated calls like PyTorch's `retain_graph=True`, and reports unknown node ids as errors. |
| Tensor backward | `src/autograd/tensor-backward.bend` | Reverse-mode `backward` for tensors on a tape of `add`, `sub`, `mul` (with broadcasting), whole-tensor `sum` and 2-D `matmul`. It seeds ones at the output the caller selects, reduces each broadcast contribution to its operand's shape with `sum_to_size`, adds the contributions to a node in the order they arrive, and agrees with PyTorch's gradients word for word on its corpus. |
| Experimental tape | `experiments/autograd-prototypes/` | Three autograd prototypes: a reverse tape, pullback closures and forward dual numbers. The reverse tape was chosen as the base for the real engine. |

Every operation either consumes its inputs or gives them back to the caller
explicitly. No two tensors share mutable storage.

The [tensor core](docs/tensor-core.md), [construction and clone](docs/construction.md),
[elementwise ops](docs/elementwise.md), [sum and matmul](docs/sum-and-matmul.md),
[transforms](docs/transforms.md), [broadcast](docs/broadcast-to.md),
[reduce to shape](docs/reduce-to.md), [sum to size](docs/sum-to-size.md),
[gradient slots](docs/gradient-slots.md), [scalar backward](docs/backward.md),
[tensor backward](docs/tensor-backward.md) and
[contract families](docs/contract-families.md)
pages document each capability. The [shape](docs/shapes.md) and
[storage](docs/storage.md) pages document the layers below them. The
[autograd design comparison](docs/autograd-design.md) explains why the reverse
tape was chosen.

## What is proved and what is tested

**Proved.** The laws in `src/*/LAWS.bend` state what each public operation
returns: the output shape, which input element each output element comes
from, the order of each sum, the error code of many failures, and that a
returned input is unchanged. Each law quantifies over every valid input, not
a list of examples. The gates also change the code on purpose (for example,
they add in the wrong order or write the wrong value) and check that the
proofs then fail for the expected reason.

The laws treat F32 arithmetic as opaque. A law can say "this element is
`F32.add` of these two words in this order", but it does not say that the
result is correctly rounded. No law proves an IEEE 754 fact. The trusted base
also includes the Bend compiler, the native runtime, Clang, the BendTT
translation, Lean and the hardware. The second check by the kernel does not
verify how Bend compiles the code. The [guarantee ledger](docs/guarantees.md)
lists the full boundary.

**Live and erased proofs.** Bend has two kinds of proof evidence. Erased
evidence is checked and then removed, so other code cannot take it as an
input. Live evidence is an ordinary value that another proof or a later
operation can receive as an argument. The contract laws make these facts live:
construction, `from_values`, reads and writes, clone and metadata, reshape,
the values and failures of permute and transpose, the values of the pointwise
and ordered-fold [contract families](docs/contract-families.md) with their
broadcast, sum, dot and `matmul` instances, the values of the autograd
`broadcast_to`, `reduce_to` and `sum_to_size`, and the values the tensor
backward's `add`, `sub`, `mul`, `sum` and `matmul` VJPs send. The public
`Elementwise.binary`, built in parallel halves, exports the pointwise family's
words for replayed inputs; `Reduction.sum` runs its own loop, which no live
law relates to the families yet, and the other autograd laws are erased. The
[proof boundary](docs/proof-boundary.md) page explains the difference.

**Tested.** The numerical results are tested, not proved. The frozen corpus in
`oracle/tensor/fixtures/s2-v2.json` has 161 successful operation cases, 45
expected errors and 7 shape-only cases, including empty and scalar tensors,
signed zeros, subnormals, infinities and NaNs. The tensor gate runs every case
in two lanes, native CPU and generated JavaScript. Data-moving operations must
match bit for bit, except for NaN payloads. Finite `add`, `sub` and `mul`
results must be the correctly rounded word, which is also PyTorch's word.
IEEE edge cases must give the exact word, or the same class (NaN or signed
infinity) when the result is not finite. `sum` and `matmul` must be within a
published rounding-error bound, and a zero result must have PyTorch's sign.
The [numerical policy](oracle/tensor/README.md) defines these rules. Version 2 of
the policy was frozen before any Bend result was compared with it; version 3
only adds checks to it. A supplement of 29 cases and 2 errors extends the
corpus with `matmul_add` bias, cancellation and signed-zero cases and `get` on
special words. The cases were then used
during development, so they are regression evidence, not held-out evidence.
The autograd building blocks have their own native and JavaScript lanes, which
compare each output word, bit for bit, with a Python reference that follows
the same order. A frozen [PyTorch record](oracle/autograd/README.md) of every one of
those requests ties the references to PyTorch's `expand`, `sum_to_size` and
`backward`: they agree word for word, apart from the listed rank, capacity,
NaN-payload and `None`-gradient differences.

## Limits

- F32 only. There is no other dtype and no dtype promotion.
- At most rank 4 and 2^20 elements per tensor.
- No views. `reshape` moves the storage to the new shape, and `permute_copy`
  and `transpose_copy` always copy. There are no strides, no in-place methods
  and no `out=` variants.
- Axes, dimensions and indices are unsigned 32-bit integers. There are no
  negative axes or indices, no `-1` in `reshape`, no slicing and no advanced
  indexing.
- `sum` reduces the whole tensor. It has no `dim` or `keepdim` argument.
- `matmul` takes two rank-2 tensors. It does not promote 1-D inputs and does
  not batch.
- `sum`, `matmul`, `reduce_to` and `sum_to_size` (when it reduces axes) add in
  a fixed sequential order. PyTorch uses a different order, so the two agree
  within the rounding bound, not word for word, on longer inputs.
- `reduce_to` turns `-0.0` into `+0.0` when the input and target shapes are
  equal, as its ordered-fold specification says. PyTorch's autograd returns the
  input unchanged there; `sum_to_size` matches PyTorch word for word in that
  case only, and within the rounding bound when it reduces axes.
  `Gradient.grad` reads a slot that received nothing as `None`, like
  PyTorch's `.grad`; `Gradient.gradient` is the separate read that gives zeros.
- Tensor `backward` covers `add`, `sub`, `mul`, whole-tensor `sum` and 2-D
  `matmul`. Its laws state the engine's structure and the values each VJP
  sends and how a slot accumulates them, but no law composes them into the
  gradients a whole `backward` leaves in the slots; those are checked by the
  runtime lanes and the PyTorch record. The prototype tape in `experiments/` is not checked by any
  gate and is not used.
- CPU only. The gates run native code on one thread, and no performance work
  has been done yet.

The [differences from PyTorch](docs/torch-differences.md) page lists each
request that PyTorch accepts and bendtorch rejects.

## Roadmap

The road from Bend to PyTorch has nine phases. bendtorch has finished the
tensor layer, the pieces that autograd needs and a backward pass over
tensors. It cannot train a model yet: it has no modules, losses or
optimizers.

| # | Phase | Status | What it delivers |
| --- | --- | --- | --- |
| 1 | Shapes and storage | Done | Validated shapes up to rank 4, row-major indexing, broadcast plans and bounds-checked F32 storage with one owner. |
| 2 | Tensor core and operations | Done | Tensors with `add`, `sub`, `mul`, whole-tensor `sum`, 2-D `matmul`, `reshape`, `permute_copy` and `transpose_copy`, checked against frozen PyTorch outputs. |
| 3 | Live proof contracts | Done | Laws that later proofs can take as inputs. Construction, `from_values`, reads and writes, clone, metadata, reshape, the permute and transpose values and failures, the pointwise and ordered-fold families with their broadcast, sum, dot and `matmul` instances, the values of `broadcast_to`, `reduce_to` and `sum_to_size`, and the tensor backward's `add`, `sub`, `mul`, `sum` and `matmul` VJP values are live. |
| 4 | Autograd building blocks | Done | `broadcast_to`, `reduce_to`, `sum_to_size` and gradient slots that accumulate in arrival order and read as `None` until they receive a value. |
| 5 | Reverse-mode engine | Done | `backward` on a tape, built on the gradient slots: for scalars (`add`, `sub`, `mul`, `neg`) and for tensors (`add`, `sub`, `mul` with broadcasting, `sum`, `matmul`). |
| 6 | Neural-network modules | Planned | `Linear`, activations and losses, each with its forward and backward pass. |
| 7 | Optimizers and training | Planned | SGD-style optimizers and a training loop that trains a small model end to end. |
| 8 | An API closer to PyTorch | Planned | More dtypes, views and strides, `dim` and `keepdim` on reductions, negative axes and indices, slicing and batched `matmul`. |
| 9 | Performance | Planned | Parallel CPU code and GPU kernels. Today the gates run native code on one thread. |

Each phase lands as a series of capabilities, each with its laws, proofs and a
gate. No step weakens an existing law, removes a test case or widens a
numerical tolerance.

## Quickstart

bendtorch runs on Ubuntu 24.04 on x86_64. Install
[Bun](https://bun.sh/) 1.3.14, which runs Bend2, and
[uv](https://docs.astral.sh/uv/) 0.10.2. Bootstrap checks both versions.

Native builds link with the system linker and C library. A minimal Ubuntu
image, such as the `ubuntu:24.04` container, also needs these packages:

```sh
sudo apt-get install git ca-certificates zstd binutils libc6-dev libgcc-13-dev libedit2 libxml2
```

bendtorch uses one exact Bend2 commit, pinned in
`verification/references.json`. Bootstrap fetches that commit, builds the
BendTT kernel, and downloads checksum-pinned Lean 4.34.0 and Clang 19.1.1
into `.tools/`. It changes nothing outside the repository. A `bend` already
on your `PATH` can be a different version, and it can reject this code.

```sh
uv sync --locked
uv run --locked python -m tools.bootstrap
```

To run the pinned Bend2 by hand:

```sh
export BEND_NO_TELEMETRY=1 BENDTT="$PWD/.tools/kernel/bendtt" LEAN_STACK_SIZE_KB=4194304
alias bend="bun $PWD/.tools/bend/bend2/main.ts"
bend guide
```

[Bumping the Bend pin](docs/bend-pin.md) describes how the pin follows
upstream main.

### Fast checks

These finish in seconds:

```sh
uv run --locked python -m unittest discover -s tests -p test_tensor_numerics.py -v
uv run --locked ruff check tools tests verification experiments
uv run --locked ruff format --check tools tests verification experiments
uv run --locked ty check
```

To compile and run a small native program that builds two matrices, multiplies
them, adds a bias and sums the result:

```sh
export CC="$PWD/.tools/llvm/usr/bin/clang-19"
export LD_LIBRARY_PATH="$PWD/.tools/llvm/usr/lib/x86_64-linux-gnu"
mkdir -p .evidence/demo
bend src/tensor/matmul-demo.bend -o .evidence/demo/matmul-demo
.evidence/demo/matmul-demo --threads 1 --gpu off
```

It prints 23 lines of F32 bit patterns as decimal integers. The first five,
`1137672192`, `1114374144`, `1115422720`, `1124859904` and `1125711872`, are
the sum 415 and the matrix `[[59, 63], [140, 153]]`.

### Full verification

These checks are slower and use several GB of memory. Each gate writes its
evidence to a new directory and refuses to overwrite an existing one.

```sh
bend PROOF.bend --verdict
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m tools.check all --output .evidence/<new directory>
uv run --locked python -m verification.contracts.run --output <new directory>
uv run --locked python -m verification.families.run --output <new directory>
```

`PROOF.bend` checks the laws of every module. `tools.check all` runs one gate
per suite: `toolchain` checks the pinned Bend2 and BendTT themselves, `shape`
and `storage` their modules, `tensor` the tensor laws and the frozen PyTorch
corpus, and `autograd` the autograd laws and the backward pass. Name suites to
run only those, as in `tools.check tensor autograd`. The last two commands
check the proved contracts and their negative controls.

The oracle has its own uv project with a CPU-only PyTorch, so PyTorch is
never installed in the root environment. The [oracle](oracle/README.md) and
[tensor oracle](oracle/tensor/README.md) pages explain how to check and regenerate
the references.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | The Bend library: `shape`, `storage`, `tensor` and `autograd`, each with its `LAWS.bend` and `PROOF.bend`. |
| `LAWS.bend`, `PROOF.bend` | The root law and proof files. |
| `verification/` | The proof exports each gate requires, the Bend pin, and the contract consumers and controls. |
| `tools/` | The Python gates, one per suite and run with `tools.check`, and the bootstrap. |
| `tests/` | Python tests for the gates. |
| `oracle/` | The isolated PyTorch reference project and its frozen fixtures. |
| `experiments/` | Experimental code, outside `src/` until it is promoted. |
| `probes/` | Small Bend programs used by the gates. |
| `docs/` | Design notes and contracts for each capability. |

## Contributing

[AGENTS.md](AGENTS.md) has the working rules: use the pinned Bend2, add a law
for each rule a later change must not break, never weaken an existing law, and
run the checks above before a pull request. There is no CI. Each pull request
lists the commands it ran, their exit codes and the Bend commit.
