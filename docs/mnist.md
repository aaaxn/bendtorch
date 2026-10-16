# MNIST end to end

`experiments/mnist/train.bend` trains a 784-128-10 MLP on MNIST with the tape:
`linear`, `relu`, `linear`, `cross_entropy` with mean reduction, `backward`
and `Sgd.step`, one mini-batch per step, then counts the test rows it
classifies right. `oracle/mnist/train.py` is PyTorch's copy, and
`oracle/mnist/compare.py` runs both over a few seeds and checks that they
agree. PyTorch runs only in the oracle environment; no gate imports it.

## Data and shared files

The program reads the four uncompressed IDX files of MNIST from a directory
given on its command line, outside the repository. It checks each header:
magic 2051 with 28 x 28 images, magic 2049 for labels, and the same count in
both files of a pair; anything else stops it with exit code 4. A pixel is its
byte divided by 255 with one F32 rounding, as `uint8.float() / 255` is in
PyTorch.

Download the four files and check them:

```sh
mkdir -p /scratch/$USER/datasets/mnist && cd /scratch/$USER/datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sSfL -o $f.gz https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
sha256sum -c - <<'EOF'
440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  train-images-idx3-ubyte.gz
3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  train-labels-idx1-ubyte.gz
8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  t10k-images-idx3-ubyte.gz
f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  t10k-labels-idx1-ubyte.gz
EOF
gunzip -k *.gz
```

These are the published hashes; the MD5 sums equal torchvision's list.

`train.py prepare SEED DIR` writes the two files both sides read:

- `init-SEED.bin`: the initial parameters of PyTorch's default `nn.Linear`
  init after `torch.manual_seed(SEED)`, as little-endian F32 words:
  `weight1 [128, 784]`, `bias1 [128]`, `weight2 [10, 128]`, `bias2 [10]`,
  each row-major. `W` is `[out, in]`, as `linear` reads it.
- `order-SEED-bBATCH.bin`: for each epoch, `torch.randperm(60000)` from a
  generator seeded with `SEED`, its first `937 * 64` indices kept, as
  DataLoader's `drop_last` keeps whole batches, as little-endian U32 words.
  Step `k` takes the `BATCH` indices from word `k * BATCH` on.

## A step

Each step builds a new tape: the four parameters (ids 0 to 3) and the batch
(id 4) as leaves, then `linear`, `relu`, `linear` and `cross_entropy` with the
ids each recording returns, `backward` of the loss, and `Sgd.step` over ids 0
to 3. It reads the loss and the four updated parameters with
`Backward.value` and drops the tape. A new tape starts with every gradient
slot empty, which is what PyTorch's `optimizer.zero_grad()` (with
`set_to_none=True`) leaves, so the program does not call `zero_grad`. The
tape has no way to drop entries, so a single tape across steps would keep
every step's saved values and make each `backward` walk every older entry.

The program prints `loss STEP WORD` for every step, the loss's F32 bits as a
decimal, then `correct N ROWS` for the test set, 1000 rows a batch, the
argmax of each row the first largest logit, as PyTorch's `argmax`. With
`words` it then prints every word of the trained parameters. The training's
wall time goes to stderr as `train_ms`.

## Run

```sh
export BEND_NO_TELEMETRY=1 BENDTT=$PWD/.tools/kernel/bendtt
bun .tools/bend/bend2/main.ts experiments/mnist/train.bend -o train.c
.tools/llvm/usr/bin/clang-19 -std=c11 -O3 train.c -lpthread -lm -o train
cd oracle
uv run --frozen python -I mnist/train.py prepare 0 WORK
../train MNIST WORK/init-0.bin WORK/order-0-b64.bin 64 937 0.1 --threads 24 --gpu off
uv run --frozen python -I mnist/train.py train MNIST WORK/init-0.bin WORK/order-0-b64.bin 64 937 0.1
uv run --frozen python -I mnist/compare.py ../train MNIST WORK --kernel default
```

`compare.py` prepares each seed, runs both, prints the table below and exits
1 when a seed fails a check. `--kernel default` runs PyTorch with
`ATEN_CPU_CAPABILITY=default`; `--kernel native` lets it pick its AVX512
kernels on this machine.

`tests/test_mnist_training.py`, in the unit tests, builds the program and
trains two steps of batch 3 on a tiny IDX set it writes: every loss word, the
test count and every trained parameter word must equal the tape reference in
`tools/autograd_tensor_backward.py`, at one thread and at several, and a label
file with the wrong count must be refused.

## Against PyTorch

One epoch, batch 64, `lr` 0.1, 937 steps, seeds 0, 1 and 2, Bend on 12
threads, PyTorch on one thread. "Equal words" counts the steps whose loss word
is the same on both sides; "rel" is `|bend - torch| / |torch|` of a step's
loss.

| PyTorch kernel | seed | equal words | max rel (step) | max rel, steps 0-99 | max rel, steps 300-936 | correct, Bend | correct, PyTorch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 0 | 116 | 3.6e-4 (886) | 3.5e-7 | 3.6e-4 | 9242 | 9242 |
| `default` | 1 | 190 | 7.0e-7 (366) | 4.1e-7 | 7.0e-7 | 9287 | 9287 |
| `default` | 2 | 192 | 8.3e-7 (908) | 3.4e-7 | 8.3e-7 | 9273 | 9273 |
| AVX512 | 0 | 165 | 1.4e-4 (919) | 3.5e-7 | 1.4e-4 | 9242 | 9242 |
| AVX512 | 1 | 193 | 8.1e-7 (456) | 3.4e-7 | 8.1e-7 | 9287 | 9287 |
| AVX512 | 2 | 48 | 2.7e-3 (886) | 3.4e-7 | 2.7e-3 | 9273 | 9272 |

The loss words differ from the first step: the products add their terms in
another order than PyTorch's BLAS, and PyTorch's AVX512 SGD fuses the
multiply and the add. The first steps differ by one or two units in the last
place, and training carries the difference forward. PyTorch's two kernels
differ from each other the same way: on these seeds, AVX512 against `default`
reaches 3.7e-4, 4.5e-7 and 2.7e-3 in the first epoch, with 9242/9242,
9287/9287 and 9272/9273 correct. Bend is as close to either kernel as the two
kernels are to each other.

The difference grows with training. PyTorch's two kernels on seed 0 over
three epochs: the median step differs by 1.5e-7 in the first epoch, 7.3e-4
in the second and 3.1e-3 in the third, where 45 steps differ by more than
1e-2.Bend against PyTorch's `default` kernel on seed 0 over the same three
epochs: 2.7e-7, 8.6e-4 and 3.0e-3 in median, at most 2.8e-2, with 65 steps
above 1e-2 in the third epoch; 9574 correct against 9575 (AVX512: 9577). A
per-step tolerance therefore holds for a stated number of steps only; past it,
only the final accuracy can be compared.

## Tolerance and band

`compare.py` checks three things for each seed:

- every step's loss within `1e-2` relative, over one epoch;
- the first 100 steps within `1e-4` relative;
- the test count within one binomial standard error of PyTorch's accuracy
  `p` on the `n = 10000` test rows, `ceil(sqrt(p (1 - p) n))` rows, 26 or 27
  at 92 to 93%.

They were fixed from PyTorch's runs before Bend's epoch was compared, not fit
to Bend's numbers. A run that differs from PyTorch only in the order of its
F32 roundings must pass whatever PyTorch's own rounding change passes. That
change, its AVX512 kernels against its `default` ones, reaches 2.7e-3 in one
epoch, and `1e-2` is the next power of ten. The band is the sampling noise of
the test set itself: a smaller difference in accuracy cannot be told apart on
10000 rows.

A per-step tolerance over the whole epoch is a weak test, because rounding
drift late in the epoch is as large as the effect of a bug. PyTorch runs with
a planted change, against PyTorch, seed 0:

| Change | max rel, epoch | max rel, steps 1-9 | correct |
| --- | --- | --- | --- |
| `lr` 0.101 instead of 0.1 | 1.2e-2 | 9.0e-4 | 9246 |
| `lr` 0.11 | 1.0e-1 | 9.0e-3 | 9268 |
| `bias1` never updated | 3.4e-2 | 3.0e-4 | 9241 |
| pixels divided by 256 | 6.6e-3 | 6.8e-4 | 9245 |

The band catches none of them and the epoch tolerance misses the last one.
Each shows by 3e-4 within ten steps, while PyTorch's own rounding change stays
below 2.4e-7 for 100 steps. The early check at `1e-4` sits between the two,
400 times above the rounding drift and 3 times below the smallest change, and
it catches all four.

## Time

Seed 0, batch 64, one epoch of 937 steps, native build, on this repository's
64-core x86-64 machine with other work running beside it (load 2 to 12 for the
rows below). `train_ms` is the program's own clock over the training steps;
the test pass is apart.

| Threads | ms per step | Epoch | Test pass (10000 rows) | Peak memory |
| --- | --- | --- | --- | --- |
| 1 | 923 (100 steps) | 14.4 min, extrapolated | 51 s | 79 MB |
| 8 | 312 | 4.9 min | 19 s | 468 MB |
| 24 | 235 | 3.7 min | 14 s | 524 MB |

PyTorch's copy on one thread takes 0.46 ms a step, 0.44 s an epoch, 2000
times less than Bend on one thread.

The three products of the first `linear` set the time: `x @ W1^T` forward,
`g^T @ x` for `W1` and `g @ W1` for `x`, each 64 x 784 x 128 multiply-adds.
With batch 1 a whole step takes 28 ms on one thread, so the work that scales
with the parameters (the update, the reads and copies of the 101770 words) is
at most 3% of a batch-64 step; the other 895 ms are the 19.5 million
multiply-adds of the products, 46 ns each. The `x` gradient is a
third of them and nothing reads it: the tape has no leaf that takes no
gradient. Threads give 3.9 times at 24, close to the 3.6 times that the
`linear` bench of #13 measured for the same products. Loading a batch takes 19 ms at any thread count.

The output is the same at 1, 8 and 24 threads: for the whole epoch, every
loss word, the test count and every trained word (102708 lines) have one
MD5. `compare.py` ran Bend on 12 threads and an earlier run on 16 gave the
same lines.
