# S3 PyTorch observations

`generate.py` records what PyTorch `2.14.1+cpu` does with every request of the
S3 broadcast, `reduce_to` and gradient subgates, and writes
`fixtures/s3-torch-v1.json`. The S3 gates compare Bend with Python references in
`tools/s3_*.py`; `tests/test_s3_torch.py` in the root suite compares those
references with this file. Bend never calls PyTorch, and the root suite reads
only the frozen JSON.

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

## Reproduce

Use the oracle environment from `oracle/s2/README.md`, with one CPU thread:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/s3/generate.py --check
(cd oracle/s3 && PYTHONDONTWRITEBYTECODE=1 uv run --project .. --frozen --no-sync python -m unittest discover -s tests -v)
uv run --locked python -m unittest tests.test_s3_torch -v
```

`--check` regenerates the file in memory and compares bytes. Without
`--check` the generator refuses to overwrite an existing file.
