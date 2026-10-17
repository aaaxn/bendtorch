# Autograd PyTorch observations

`generate.py` records what PyTorch `2.14.1+cpu` does with every request of the
autograd broadcast, `reduce_to` and gradient subgates, and writes
`fixtures/s3-torch-v1.json`. The autograd gates compare Bend with Python
references in `tools/autograd_*.py`; `tests/test_autograd_torch.py` in the root
suite compares those references with this file. Bend never calls PyTorch, and
the root suite reads only the frozen JSON.

For each request the generator records:

- **broadcast**: `x.expand(target)`, or the exception type when PyTorch rejects
  it.
- **reduce_to**: `x.sum_to_size(target)` and the gradient of a `target`-shaped
  leaf expanded to the contribution's shape, with the contribution as the
  output gradient.
- **gradient**: a leaf of the slot's shape, then one `leaf.backward(c)` per
  contribution `c` in order, recording which ones PyTorch accepts, and the
  final `leaf.grad` (`null` when PyTorch leaves it `None`).

Results hold hex F32 words. A result with more than 4096 words stores its
count and the SHA-256 of its space-separated words instead. The file also
records the SHA-256 of each request list, so the root test fails if a subgate
changes its requests without a regenerated fixture.

Every successful request agrees with PyTorch word for word, except for these
recorded differences:

| Request | bendtorch | PyTorch |
| --- | --- | --- |
| broadcast `target_rank_five`, `target_capacity`; gradient `slot_rank_five`, `slot_capacity` | `RankExceeded` / `CapacityExceeded` | accepts rank 5 and more than 2^20 elements |
| reduce `special_words_rows` | NaN | NaN with another payload |
| gradient slots that never accumulate (17 requests) | `gradient` materializes zeros; `Gradient.grad` is `None` | `.grad` is `None` |

All other rejections agree: PyTorch also rejects every request that bendtorch
rejects with `IncompatibleBroadcast`, `NotBroadcastable`, `NotReducible` or
`ShapeMismatch`.

## Backward

`backward.py` replays every script of the autograd backward subgate on PyTorch
scalars and writes `fixtures/s3-backward-torch-v1.json`. Each script records
leaves and `+`, `-`, `*` and unary `-` in order; every node keeps its gradient
(`retain_grad()`), every `backward` keeps the graph (`retain_graph=True`), and
`zero_grad` sets every `.grad` to `None`. `tests/test_autograd_backward.py` in
the root suite compares the gate's Python reference with this file. 83 of 91
scripts agree line for line. Three scripts use node ids that the tape does not
hold; bendtorch reports `UnknownNode` where the replay raises `IndexError`.
Four scripts forge tapes and are recorded as `unrepresentable`: three with
malformed wiring (an operand that is the node itself, a newer node or a missing
node) and one with sound wiring but saved state that disagrees with its leaves.
In `nan_payloads_sub`, PyTorch's `NaN_a - NaN_b` keeps the second payload and
bendtorch's native lane keeps the first.

## Tensor backward

`tensor_backward.py` replays every script of the autograd tensor backward
subgate on PyTorch tensors and writes `fixtures/tensor-backward-torch.json`.
Each script records leaves (constants as tensors without `requires_grad`) and `+`, `-`, `*`, `@`, `linear(x, W)`
(`matmul_nt`), `linear(x, W) + b` (`linear`, two entries, as the tape records
it; both are computed before either is recorded, so a refused `linear`
records nothing), `torch.relu`, `mse_loss`, `cross_entropy` with class
indices, `.sum()` and `.sum(dim=..., keepdim=...)` in order, and SGD steps
(`torch.optim.SGD(params, lr=lr).step()` over the listed nodes; a rate below
zero is recorded as `ValueError`; an empty list, which `torch.optim.SGD`
refuses, updates nothing); every `backward` seeds `torch.ones_like(output)` and keeps the graph
(`retain_graph=True`), every non-leaf keeps its gradient (`retain_grad()`), and
`zero_grad` sets every `.grad` to `None`. Lines carry decimal F32 words in the
gate's own format. `tests/test_autograd_tensor_backward.py` in the root suite
compares the gate's Python reference with this file, under level 2 of the
[numeric policy](../../AGENTS.md#numeric-policy) (`tools/autograd_tensor_bound.py`
computes each word's rounding-error bound, checked on top of the listed
words): 165 of 217 scripts agree line for
line, and [the differences from PyTorch](../../docs/torch-differences.md#tensor-backward)
explain the other 52.

## Reproduce

Use the oracle environment from `oracle/tensor/README.md`, with one CPU thread:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/generate.py --check
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/backward.py --check
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/tensor_backward.py --check
(cd oracle/autograd && PYTHONDONTWRITEBYTECODE=1 uv run --project .. --frozen --no-sync python -m unittest discover -s tests -v)
uv run --locked python -m unittest tests.test_autograd_torch tests.test_autograd_backward.BackwardTorchOracleTests -v \
  tests.test_autograd_tensor_backward.TensorBackwardTorchOracleTests
```

`--check` regenerates the file in memory and compares bytes. Without
`--check` each generator refuses to overwrite an existing file.
