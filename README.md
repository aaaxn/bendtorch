# bendtorch

bendtorch is a PyTorch-like tensor library written in
[Bend2](https://github.com/bendlang/bend). All tensor code is Bend. The rules
the code must keep are laws in `LAWS.bend` files: the Bend checker proves each
law, and the BendTT kernel, a separate checker built with Lean, checks the
exported proofs again. Python gates in `tools/` compile the Bend code, run it
natively and as JavaScript, and compare the results with frozen references
from a separate PyTorch project in `oracle/`. PyTorch never computes a
bendtorch result.

bendtorch is early research code, used from Bend programs. It has no Python
API, is not a drop-in replacement for PyTorch, and runs on the CPU with F32
tensors only. An experiment trains a 784-128-10 MLP on MNIST with it end to
end ([MNIST](docs/mnist.md)).

> **Disclaimer.** bendtorch is an experiment.

## What works today

| Capability | Code | What it does |
| --- | --- | --- |
| Shapes and indexing | `src/shape/` | Validated shapes of rank 0 to 4 with at most 2^20 elements, row-major flatten and unflatten, checked coordinates and right-aligned broadcast plans ([shapes](docs/shapes.md)). |
| Owned storage | `src/storage/` | F32 storage with one owner: `new`, `get`, `set`, `clone` and `length`, with bounds checks before every access ([storage](docs/storage.md)). |
| Tensor core | `src/tensor/core.bend` | `Core.new`, `from_values`, `construct`, flat and coordinate reads and writes, `clone`, and the metadata queries `length`, `shape` and `count` ([tensor core](docs/tensor-core.md), [construction and clone](docs/construction.md)). |
| Elementwise ops | `src/tensor/elementwise.bend` | `add`, `sub` and `mul` with PyTorch-style broadcasting, one F32 operation per output element, both inputs given back to the caller. `ReluLeft` and `ReluBackward` are PyTorch's `relu` and its gradient, built on IEEE comparisons with zero that read the F32 bits. `Divide`, `ExpLeft`, `LogLeft`, `Matches`, `PickNegated`, `MseGrad`, `OneHotGrad` and `SoftmaxGrad` are the steps of the mean and the losses. `Descend{lr}` is one cell of an SGD step, `p - lr * g` ([elementwise ops](docs/elementwise.md)). |
| Sum | `src/tensor/reduction.bend` | Sum of a whole tensor to a scalar, in row-major order, starting from `+0.0`. |
| Matmul | `src/tensor/matmul.bend` | `[m,k] @ [k,n]`. Each dot product adds in increasing `k`, starting from `+0.0`. Large products build their output cells in parallel ([the shared parallel build](docs/parallel-build.md)). `matmul_nt` (`x @ W^T`) and `matmul_tn` (`g^T @ x`) read the transposed operand in place, with the same words as a transpose copy and `matmul` ([sum and matmul](docs/sum-and-matmul.md)). |
| Reshape, permute, transpose | `src/tensor/transforms.bend` | `reshape` keeps the linear order. `permute_copy` and `transpose_copy` write a new contiguous tensor ([transforms](docs/transforms.md)). |
| Proved contracts | `src/tensor/*-contract-LAWS.bend`, `src/tensor/contracts/` | Laws that later proofs can use as inputs, for construction, reads and writes, clone and metadata, and reshape, plus the error cases of permute and transpose. A generic theorem for pointwise loops proves the values that a broadcast binary operation writes; one for ordered folds proves the total of a sum and of one matmul dot product, with its addition order ([contract families](docs/contract-families.md)). |
| Autograd building blocks | `src/autograd/` | `broadcast_to` (copies a tensor to a larger shape), `reduce_to` (sums a gradient back to the shape of its input), `sum_to_size` (the PyTorch-facing form of `reduce_to`, which keeps the input unchanged when no axis is reduced) and gradient slots (`Gradient.accumulate`, `Gradient.grad`, `Gradient.reset`, `Gradient.gradient`) that add contributions in arrival order, read as `None` until they receive one, and reset to `None` ([broadcast](docs/broadcast-to.md), [reduce to shape](docs/reduce-to.md), [sum to size](docs/sum-to-size.md), [gradient slots](docs/gradient-slots.md)). |
| Scalar backward | `src/autograd/backward.bend` | Reverse-mode `backward` for F32 scalars on a tape of `add`, `sub`, `mul` and `neg`. It seeds the output the caller selects, gives `None` to every node that output does not reach, adds the contributions to a node used several times in a fixed order, accumulates repeated calls like PyTorch's `retain_graph=True`, and reports unknown node ids as errors ([scalar backward](docs/backward.md)). |
| Tensor backward | `src/autograd/tensor-backward.bend` | Reverse-mode `backward` for tensors on a tape of `add`, `sub`, `mul` (with broadcasting), whole-tensor `sum`, 2-D `matmul`, `linear` (`x @ W^T + b` with `W` of `[out, in]`, PyTorch's layout), `relu`, and the losses `mse_loss` and `cross_entropy` with mean reduction ([losses](docs/losses.md)). It seeds ones at the output the caller selects, reduces each broadcast contribution to its operand's shape with `sum_to_size`, adds the contributions to a node in the order they arrive, which two laws state for every tape as the chain rule ([chain rule](docs/tensor-backward.md#chain-rule)), and agrees with PyTorch's gradients word for word on its corpus, except the last bits of some `mse_loss` and `cross_entropy` gradient words, where PyTorch sums in another order or its backward uses a vectorized `exp` ([losses](docs/losses.md#against-pytorch), [tensor backward](docs/tensor-backward.md)). |
| SGD | `src/autograd/sgd.bend` | `torch.optim.SGD(params, lr)` without momentum or weight decay: `step` updates each listed leaf whose gradient slot holds `g` to `p - lr * g`, in list order, and keeps a leaf whose slot is empty. It checks every id and the learning rate (refused below zero, as PyTorch's `lr < 0.0`) first, so a refusal leaves the tape unchanged. The product and the difference each round once, as PyTorch's `DEFAULT` kernel does; its AVX2 and AVX512 kernels fuse them. |
| MNIST training | `experiments/mnist/`, `oracle/mnist/` | An experiment that trains a 784-128-10 MLP on MNIST with the tape, `linear`, `relu`, `cross_entropy` and SGD, one new tape per mini-batch, next to PyTorch's copy from the same initial words and batches, and a command that compares the two over seeds ([MNIST](docs/mnist.md)). |
| Experimental tape | `experiments/autograd-prototypes/` | Three autograd prototypes: a reverse tape, pullback closures and forward dual numbers. The reverse tape was chosen as the base for the real engine ([autograd design comparison](docs/autograd-design.md)). |

Every operation either consumes its inputs or gives them back to the caller
explicitly. No two tensors share mutable storage.

## What is proved and what is tested

**Proved.** The laws in `src/*/LAWS.bend` state what each public operation
returns: the output shape, which input element each output element comes
from, the order of each sum, the error code of many failures, and that a
returned input is unchanged. The general laws quantify over every input that
meets their hypotheses, not over a list of examples; a few more laws, such as
the `relu` and SGD learning-rate edge words, check fixed lists of edge cases.
The gates also change the code on purpose (for example, they add in the wrong
order or write the wrong value) and check that the proofs then fail for the
expected reason.

The laws treat F32 arithmetic as opaque: a law can say "this element is
`F32.add` of these two words in this order", not that the result is correctly
rounded, and no law proves an IEEE 754 fact. The trusted base also includes the
Bend compiler, the native runtime, Clang, the BendTT translation, Lean and the
hardware; the kernel's second check does not verify how Bend compiles the
code. The [guarantee ledger](docs/guarantees.md) lists the full boundary.

**Live and erased proofs.** Erased evidence is checked and then removed, so
other code cannot take it as an input; live evidence is an ordinary value that
later proofs and operations can receive
([proof boundary](docs/proof-boundary.md)). Live laws include: construction,
`from_values`, reads and writes, clone and metadata, reshape, the values and
failures of permute and transpose, the values of the pointwise and
ordered-fold families with their broadcast, sum, dot and `matmul` instances,
the values of `broadcast_to`, `reduce_to` and `sum_to_size`, the values the
tensor backward's `add`, `sub`, `mul`, `sum`, `matmul`, `linear`, `relu`,
`mse_loss` and `cross_entropy` VJPs send, and the values of an SGD step. The
public `Elementwise.binary`, built in parallel halves, exports the pointwise
family's words for replayed inputs; `Reduction.sum` runs its own loop, which
no live law relates to the families yet.

**Tested.** The numerical results are tested, not proved. The frozen corpus in
`oracle/tensor/fixtures/s2-v2.json` has 161 successful operation cases, 45
expected errors and 7 shape-only cases, including empty and scalar tensors,
signed zeros, subnormals, infinities and NaNs. The tensor gate runs 202 of
these 213 cases natively and as generated JavaScript (the 161 successes, 34
errors and the 7 shape-only cases); the Python decoder refuses 4 errors whose
values fall outside the unsigned 32-bit range, and the 7 unsupported-dtype
errors become compile-negative programs that must fail Bend source checking
under both `--check-only` and `--verdict`. Data-moving operations must match
bit for bit, except for NaN payloads. Finite `add`, `sub` and `mul` results
must be the correctly rounded word, which is also PyTorch's word. IEEE edge
cases must give the exact word, or the same class (NaN or signed infinity) when
the result is not finite. `sum` and `matmul` must be within a published
rounding-error bound, and a zero result must have PyTorch's sign. The
[numerical policy](oracle/tensor/README.md) defines these rules. Version 2 was
frozen before any Bend result was compared with it; version 3 only adds checks,
and a supplement adds 29 cases and 2 errors. The cases were then used during
development, so they are regression evidence, not held-out evidence. The
autograd building blocks have their own native and JavaScript lanes, which
compare each output word, bit for bit, with a Python reference that follows the
same order. A frozen [PyTorch record](oracle/autograd/README.md) of those
requests ties the references to PyTorch's `expand`, `sum_to_size` and
`backward`: they agree word for word, apart from the listed rank, capacity,
NaN-payload and `None`-gradient differences.

## Limits

- F32 only, with no dtype promotion. At most rank 4 and 2^20 elements per
  tensor.
- No views. `reshape` moves the storage to the new shape, and `permute_copy`
  and `transpose_copy` always copy. No strides, in-place methods or `out=`
  variants.
- Axes, dimensions and indices are unsigned 32-bit integers: no negative axes
  or indices, no `-1` in `reshape`, no slicing and no advanced indexing.
- `sum` reduces the whole tensor, with no `dim` or `keepdim`. `matmul` takes
  two rank-2 tensors; it does not promote 1-D inputs and does not batch.
- `sum`, `matmul`, `reduce_to` and `sum_to_size` (when it reduces axes) add in
  a fixed sequential order. PyTorch uses a different order, so on longer
  inputs the two agree within the rounding bound, not word for word.
- `reduce_to` turns `-0.0` into `+0.0` when the input and target shapes are
  equal, as its ordered-fold specification says; PyTorch's autograd returns the
  input unchanged there. `sum_to_size` matches PyTorch word for word in that
  case only, and within the rounding bound when it reduces axes.
  `Gradient.grad` reads a slot that received nothing as `None`, like PyTorch's
  `.grad`; `Gradient.gradient` is the separate read that gives zeros.
- Tensor `backward` covers only the operations in the table above. Its laws
  state the engine's structure, the values each VJP sends and, as the chain
  rule, the gradients of a sweep that succeeds; they say nothing about a sweep
  that fails beyond its refusal laws. Agreement with PyTorch is checked by the
  runtime lanes and the PyTorch record. The prototype tape in `experiments/` is
  not checked by any gate and is not used.
- CPU only. The gates run native code on 8 threads (fewer on a machine with
  fewer CPUs). `matmul` forks by work (cells times inner terms), and the gates
  run products that fork at 1, 8 and 24 threads; no gate case splits the
  parallel elementwise build. `reduce_to` and `sum_to_size` fork only when
  every fiber is contiguous (row sums); column and batch sums, such as bias
  gradients, and the full `sum` run on one thread.

The [differences from PyTorch](docs/torch-differences.md) page lists each
request that PyTorch accepts and bendtorch rejects.

## Roadmap

The road from Bend to PyTorch has nine phases.

| # | Phase | Status | What it delivers |
| --- | --- | --- | --- |
| 1 | Shapes and storage | Done | Validated shapes up to rank 4, row-major indexing, broadcast plans and bounds-checked F32 storage with one owner. |
| 2 | Tensor core and operations | Done | Tensors with `add`, `sub`, `mul`, whole-tensor `sum`, 2-D `matmul`, `reshape`, `permute_copy` and `transpose_copy`, checked against frozen PyTorch outputs. |
| 3 | Live proof contracts | Done | Laws that later proofs can take as inputs, for the facts listed under live proofs above. |
| 4 | Autograd building blocks | Done | `broadcast_to`, `reduce_to`, `sum_to_size` and gradient slots that accumulate in arrival order and read as `None` until they receive a value. |
| 5 | Reverse-mode engine | Done | `backward` on a tape, built on the gradient slots: for scalars (`add`, `sub`, `mul`, `neg`) and for tensors (`add`, `sub`, `mul` with broadcasting, `sum`, `matmul`). |
| 6 | Neural-network modules | In progress | `Linear` is done: `x @ W^T + b` on the tape, `W` read in place, with value laws for its gradients. `relu` is done, with PyTorch's words at `-0.0`, NaN and the gradient at `0`. `mse_loss` and `cross_entropy` with mean reduction are done, in PyTorch's operation order, with value laws for their gradients. |
| 7 | Optimizers and training | In progress | SGD is done: `step` over a list of tape parameters, `p - lr * g` with the learning rate only, on the parallel elementwise build, and `zero_grad`. An experiment trains a 784-128-10 MLP on MNIST for an epoch, its loss at every step within `1e-2` of PyTorch's copy and its test accuracy within a band of one binomial standard error ([MNIST](docs/mnist.md)). Momentum, weight decay and other optimizers remain. |
| 8 | An API closer to PyTorch | Planned | More dtypes, views and strides, `dim` and `keepdim` on reductions, negative axes and indices, slicing and batched `matmul`. |
| 9 | Performance | In progress | Parallel CPU code and GPU kernels. `matmul` and the elementwise operations build large outputs in parallel; reductions and GPU kernels remain. |

Each phase lands as a series of capabilities, each with its laws, proofs and a
gate. No step weakens an existing law, removes a test case or widens a
numerical tolerance.

## Quickstart

bendtorch runs on Ubuntu 24.04 on x86_64. It needs [Bun](https://bun.sh/)
1.3.14, which runs Bend2, and [uv](https://docs.astral.sh/uv/) 0.10.2;
bootstrap checks both versions. Native builds link with the system linker and
C library, so a minimal image, such as the `ubuntu:24.04` container, also
needs:

```sh
sudo apt-get install git ca-certificates zstd binutils libc6-dev libgcc-13-dev libedit2 libxml2
```

Bootstrap fetches the one Bend2 commit pinned in
`verification/references.json`, builds the BendTT kernel, and downloads
checksum-pinned Lean 4.34.0 and Clang 19.1.1 into `.tools/`. It changes
nothing outside the repository. A `bend` already on your `PATH` can be a
different version, and it can reject this code.
[Bumping the Bend pin](docs/bend-pin.md) describes how the pin follows
upstream main. Bootstrap, then run the pinned Bend2 by hand:

```sh
uv sync --locked
uv run --locked python -m tools.bootstrap
export BEND_NO_TELEMETRY=1 BENDTT="$PWD/.tools/kernel/bendtt" LEAN_STACK_SIZE_KB=4194304
alias bend="bun $PWD/.tools/bend/bend2/main.ts"
bend guide
```

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
evidence to a new directory and refuses to overwrite an existing one, so pick
a fresh name for each run:

```sh
bend PROOF.bend --verdict
uv run --locked python -m tools.test
uv run --locked python -m tools.check all --output .evidence/check-run-1
uv run --locked python -m verification.contracts.run --output .evidence/contracts-run-1
uv run --locked python -m verification.families.run --output .evidence/families-run-1
```

`PROOF.bend` checks the laws of every module. `tools.check all` runs one gate
per suite: `toolchain` checks the pinned Bend2 and BendTT themselves, `shape`
and `storage` their modules, `tensor` the tensor laws and the frozen PyTorch
corpus, and `autograd` the autograd laws and the backward pass; name suites to
run only those, as in `tools.check tensor autograd`. The last two commands
check the proved contracts and their negative controls. The oracle has its own
uv project with a CPU-only PyTorch, so PyTorch is never installed in the root
environment; the [oracle](oracle/README.md) and
[tensor oracle](oracle/tensor/README.md) pages explain how to check and
regenerate the references.

## Contributing

[AGENTS.md](AGENTS.md) has the working rules: use the pinned Bend2, add a law
for each rule a later change must not break, never weaken an existing law, and
run the checks above before a pull request. There is no CI: each pull request
lists the commands it ran, their exit codes and the Bend commit.
