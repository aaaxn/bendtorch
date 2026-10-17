<p align="center">
  <img src="docs/art/banner.gif" alt="bendtorch: a PyTorch-like tensor library, written in Bend. A girl in a koi hat rides a whale shark through Bend code." width="100%">
</p>

bendtorch is a PyTorch-like tensor library written in
[Bend2](https://github.com/bendlang/bend). Its tensors, its autograd and its
optimizer are Bend code. The rules that code must keep are laws in
`LAWS.bend` files. The Bend checker proves each law, and the BendTT kernel, a
separate checker built with Lean, checks the proofs again. Python gates in
`tools/` run the same code natively and as JavaScript, and compare every
result with references, including frozen records from a separate PyTorch
project in `oracle/`. PyTorch never computes a bendtorch result.

> **Disclaimer.** bendtorch is an experiment, and im learning while building this... so expects things to be rough, broken and stupid....

<p align="center">
  <img src="docs/art/hero.gif" alt="A short tour: the MNIST step in PyTorch and in Bend, matmul cells built in parallel halves, a law with its proof verdict, and the MNIST loss of bendtorch on top of PyTorch's." width="100%">
</p>

## It reads like PyTorch

<p align="center">
  <img src="docs/art/ops.png" alt="One MNIST step side by side: PyTorch calls F.linear, F.relu, F.cross_entropy, backward and step; bendtorch calls Backward.linear, Backward.relu, Backward.cross_entropy, Backward.backward and Sgd.step." width="100%">
</p>

`experiments/mnist/train.bend` trains a small MLP with the same steps as its
PyTorch copy in `oracle/mnist/train.py`: `linear`, `relu`, `linear`,
`cross_entropy`, `backward`, and an SGD step. The calls look alike, but
bendtorch is a Bend library, used from Bend programs. It has no Python API
and is not a drop-in replacement for PyTorch.

Every operation either consumes its inputs or gives them back to the caller.
No two tensors share mutable storage.

## Autograd records a tape and walks it back

<p align="center">
  <img src="docs/art/autograd.gif" alt="The tape of one MNIST step: forward arrows record linear, relu, linear and cross_entropy; backward arrows return gradients to the slots of W1, b1, W2 and b2; the batch, a constant, keeps None; then Sgd.step updates each parameter to p - lr * g." width="100%">
</p>

Each operation of a forward pass adds an entry to a tape.
`Backward.backward` seeds ones at the output you select and walks the tape in
reverse. Each entry sends its contributions to the gradient slots of its
inputs. A slot reads `None` until a contribution reaches it, as PyTorch's
`.grad` does. A constant, which is a leaf that takes no gradient, never gets
one. A broadcast contribution sums back to the shape of its operand with
`sum_to_size`. Two laws state the chain rule for every tape. `Sgd.step` then
moves each parameter to `p - lr * g`.

The [tensor backward](docs/tensor-backward.md) page lists the operations the
tape records. The [gradient slots](docs/gradient-slots.md) and
[losses](docs/losses.md) pages cover the rest.

## Large outputs build in parallel halves

<p align="center">
  <img src="docs/art/parallel.gif" alt="A matmul output of 8 rows: the work splits in two, then in four, then in eight; eight small whales fill one row each, then join into one tensor." width="100%">
</p>

Bend runs `a b = low(l1, r1) high(l2, r2)` as two parallel calls.
`src/tensor/parallel.bend` uses that line to split the cells of a large
output, such as a `matmul` product, an elementwise result, or a `reduce_to`
gradient, into balanced halves. Every cell gets the word that the one-thread build computes there, so
no output word depends on the number of threads. The
[parallel build](docs/parallel-build.md) page explains the split and its
proofs.

## Laws state the rules, and Bend proves them

<p align="center">
  <img src="docs/art/laws.gif" alt="The law sgd_descend_scalar in src/autograd/sgd-LAWS.bend, read line by line: for every learning rate, parameter word and gradient word, one SGD cell is exactly p - lr * g. Then bend PROOF.bend --verdict prints ALL PROOFS CHECK." width="100%">
</p>

Each module keeps its rules in `src/<module>/LAWS.bend` and their proofs in
`src/<module>/PROOF.bend`. `bend PROOF.bend --verdict` checks every law with
the Bend checker, then rechecks the proofs with the BendTT kernel. The
general laws quantify over every input that meets their hypotheses, not over
a list of examples. A few more, such as the `relu` and SGD learning-rate edge
words, check fixed lists of edge cases.

### A one-word mistake fails the proof

<p align="center">
  <img src="docs/art/blocked.gif" alt="A diff in src/tensor/elementwise.bend turns F32.sub into F32.add in the SGD cell. bend PROOF.bend --verdict prints SOME PROOFS FAIL, the expected and observed terms, and Location: Laws.sgd_descend_scalar." width="100%">
</p>

Change `F32.sub` to `F32.add` in the SGD cell of
`src/tensor/elementwise.bend`, and each step climbs the loss instead of
descending it. `bend PROOF.bend --verdict` then exits with code 1 at the law
`sgd_descend_scalar`. The terminal in the picture shows that run. The gates
make changes like this one on purpose and check that the proofs fail for the
expected reason.

## What is proved and what is tested

<p align="center">
  <img src="docs/art/checks.png" alt="Two columns. Proved: LAWS.bend, then bend PROOF.bend --verdict, then the Bend checker, then the BendTT kernel built with Lean. Tested: the Bend code runs natively and as JavaScript, and its words are compared bit for bit with a Python reference, and with PyTorch's frozen records: its word, within a rounding bound, or listed." width="100%">
</p>

The laws state what each public operation returns: the output shape, the
input element behind each output element, the order of each sum, the error
code of many failures, and that a returned input is unchanged. They treat F32
arithmetic as opaque. A law can say "this element is `F32.add` of these two
words in this order", but not that the result is correctly rounded. The
trusted base also includes the Bend compiler, the native runtime, Clang, the
BendTT translation, Lean, and the hardware. The
[guarantee ledger](docs/guarantees.md) lists the whole boundary, and the
[proof boundary](docs/proof-boundary.md) page says which proofs later code can
use as inputs.

The numbers are tested at three levels, which the
[numeric policy](AGENTS.md#numeric-policy) defines:

1. Every gate lane, native and JavaScript, must match its Python reference
   bit for bit.
2. Against PyTorch, an operation that rounds once per output word must give
   PyTorch's word. A sum, a matmul, and every word built on one must lie
   within a published rounding-error bound. A word built on `exp` or `log`
   is listed with both values.
3. A training run must stay within a tolerance of its PyTorch copy, fixed
   from PyTorch against itself before any Bend result is compared.

The [oracle](oracle/README.md), [tensor oracle](oracle/tensor/README.md) and
[autograd oracle](oracle/autograd/README.md) pages describe the PyTorch
records and how to regenerate them. The frozen tensor corpus was used during
development, so it is regression evidence, not held-out evidence.

## It trains MNIST next to PyTorch

<p align="center">
  <img src="docs/art/mnist.png" alt="The loss at each of 937 steps of one MNIST epoch: bendtorch's violet curve lies on PyTorch's gray one. Test rows right: bendtorch 9242, PyTorch 9242, of 10000." width="100%">
</p>

The experiment trains a 784-128-10 MLP for one epoch, from the same initial
words and the same batches as PyTorch, with a new tape for every mini-batch.
`oracle/mnist/compare.py` checks each step's loss against PyTorch's: within
`1e-4` relative for the first 100 steps and `1e-2` after. It also checks that
the test count lies within `ceil(sqrt(p (1 - p) n))` rows of PyTorch's, where
`p` is PyTorch's accuracy on the `n` test rows. The figure is seed 0, with
PyTorch on its AVX512 kernels: the two runs agree on the loss word in 165 of
937 steps, differ by at most 1.4e-4 relative, and each classify 9242 test
rows right. [MNIST](docs/mnist.md) has every
seed, both PyTorch kernels, and the timings.

## How the code is laid out

<p align="center">
  <img src="docs/art/modules.png" alt="Four modules stacked bottom-up: shape, storage, tensor, autograd, each with its own LAWS.bend and PROOF.bend, and the MNIST experiment on top." width="100%">
</p>

Each module builds on the one below it.

| Path | What it holds | Pages |
| --- | --- | --- |
| `src/shape/` | Validated shapes, row-major flatten and unflatten, checked coordinates, broadcast plans | [shapes](docs/shapes.md) |
| `src/storage/` | F32 storage with one owner and a bounds check before every access | [storage](docs/storage.md) |
| `src/tensor/` | The tensor core, elementwise operations, `sum`, `matmul`, `reshape`, `permute_copy`, `transpose_copy`, and the proved contracts | [tensor core](docs/tensor-core.md), [construction and clone](docs/construction.md), [elementwise](docs/elementwise.md), [sum and matmul](docs/sum-and-matmul.md), [transforms](docs/transforms.md), [contract families](docs/contract-families.md) |
| `src/autograd/` | `broadcast_to`, `reduce_to`, `sum_to_size`, `sum_dims`, gradient slots, scalar and tensor `backward`, the losses, SGD | [broadcast](docs/broadcast-to.md), [reduce to shape](docs/reduce-to.md), [sum over axes](docs/reduce-to.md#sum-over-axes), [sum to size](docs/sum-to-size.md), [scalar backward](docs/backward.md), [tensor backward](docs/tensor-backward.md) |
| `experiments/` | The MNIST training run, the autograd prototypes, the benchmarks, and other experiments outside `src/` | [MNIST](docs/mnist.md), [autograd design](docs/autograd-design.md) |
| `oracle/` | The PyTorch references, in their own uv project | [oracle](oracle/README.md) |
| `tools/` | One gate per suite, run by `python -m tools.check` | [AGENTS.md](AGENTS.md#gates) |

## Run it

bendtorch runs on Ubuntu 24.04 on x86_64. It needs [Bun](https://bun.sh/)
1.3.14, which runs Bend2, and [uv](https://docs.astral.sh/uv/) 0.10.2.
Bootstrap checks both versions. Native builds link with the system linker and
C library, so a minimal image, such as the `ubuntu:24.04` container, also
needs these packages:

```sh
sudo apt-get install git ca-certificates zstd binutils libc6-dev libgcc-13-dev libedit2 libxml2
```

Bootstrap fetches the one Bend2 commit pinned in
`verification/references.json`, builds the BendTT kernel, and downloads
checksum-pinned Lean 4.34.0 and Clang 19.1.1 into `.tools/`. It changes
nothing outside the repository. A `bend` already on your `PATH` can be a
different version, and it can reject this code.
[Bumping the Bend pin](docs/bend-pin.md) describes how the pin follows
upstream main. Bootstrap, then run the pinned Bend2:

```sh
uv sync --locked
uv run --locked python -m tools.bootstrap
export BEND_NO_TELEMETRY=1 BENDTT="$PWD/.tools/kernel/bendtt" LEAN_STACK_SIZE_KB=4194304
alias bend="bun $PWD/.tools/bend/bend2/main.ts"
bend guide
```

### Multiply two matrices

This program builds two matrices, multiplies them, adds a bias, and sums the
result:

```sh
export CC="$PWD/.tools/llvm/usr/bin/clang-19"
export LD_LIBRARY_PATH="$PWD/.tools/llvm/usr/lib/x86_64-linux-gnu"
mkdir -p .evidence/demo
bend src/tensor/matmul-demo.bend -o .evidence/demo/matmul-demo
.evidence/demo/matmul-demo --threads 1 --gpu off
```

It prints F32 bit patterns as decimal integers. The first five,
`1137672192`, `1114374144`, `1115422720`, `1124859904` and `1125711872`, are
the sum 415 and the matrix `[[59, 63], [140, 153]]`.

### Check the code

These finish in seconds:

```sh
uv run --locked python -m unittest discover -s tests -p test_tensor_numerics.py -v
uv run --locked ruff check tools tests verification experiments
uv run --locked ruff format --check tools tests verification experiments
uv run --locked ty check
```

The full verification is slower and uses several GB of memory. Each gate
writes its evidence to a new directory and refuses to overwrite one, so pick
a fresh name for each run:

```sh
bend PROOF.bend --verdict
uv run --locked python -m tools.test
uv run --locked python -m tools.check all --output .evidence/check-run-1
uv run --locked python -m verification.contracts.run --output .evidence/contracts-run-1
uv run --locked python -m verification.families.run --output .evidence/families-run-1
```

`PROOF.bend` checks the laws of every module. `tools.check all` runs one gate
per suite: `toolchain` checks the pinned Bend2 and BendTT, `shape` and
`storage` check their modules, `tensor` checks the tensor laws and the frozen
PyTorch corpus, and `autograd` checks the autograd laws and the backward
pass. Name suites to run only those, as in `tools.check tensor autograd`. The
last two commands check the proved contracts and their negative controls. The
oracle has its own uv project with a CPU-only PyTorch, so PyTorch never enters
the root environment.

## Limits

<p align="center">
  <img src="docs/art/limits.png" alt="The rider asleep on her whale. Early research code: a Bend library, used from Bend programs, not a drop-in replacement for PyTorch." width="100%">
</p>

- F32 only, with no dtype promotion. At most rank 4 and 2^20 elements per
  tensor.
- No views. `reshape` moves the storage to the new shape, and `permute_copy`
  and `transpose_copy` always copy. There are no strides, no in-place methods,
  and no `out=` variants.
- Axes, dimensions and indices are unsigned 32-bit integers. There are no
  negative axes or indices, no `-1` in `reshape`, no slicing, and no advanced
  indexing.
- `sum` reduces the whole tensor. On the tape, `sum_dims` sums the listed
  axes, with or without `keepdim`. `matmul` takes two rank-2 tensors. It does
  not promote 1-D inputs and does not batch.
- `sum`, `sum_dims`, `matmul`, `reduce_to` and a reducing `sum_to_size` add
  in a fixed order. PyTorch adds in another order, so on longer inputs the
  two agree within the rounding bound, not word for word.
- Tensor `backward` records only the operations its page lists.
- CPU only.

The [differences from PyTorch](docs/torch-differences.md) page lists each
request that PyTorch accepts and bendtorch rejects.

## Contributing

[AGENTS.md](AGENTS.md) has the working rules: use the pinned Bend2, add a law
for each rule a later change must not break, never weaken an existing law, and
run the checks above before a pull request. There is no CI. Each pull request
lists the commands it ran, their exit codes, and the Bend commit.

## License

bendtorch is released under the [MIT License](LICENSE).
