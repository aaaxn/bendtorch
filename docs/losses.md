# Losses

`src/autograd/loss.bend` holds PyTorch's `mse_loss` and `cross_entropy` over
class logits, each with `mean` reduction, and the `mean` they share. The tape
records them as `MseLoss` and `CrossEntropy` nodes
([tensor backward](tensor-backward.md)).

## Forward

Every step is a tensor operation that builds its output over cells in
parallel ([parallel build](parallel-build.md)): `Elementwise.binary`,
`sum_to_size` and a row maximum on `Parallel.build`. No step adds a tree sum:
within one output cell a reduction adds left to right from `+0.0`.

- `mean(x)` is `Reduction.sum_dynamic(x)`, every word added in ascending flat
  order from `+0.0`, then `Divide` by the element count as an F32
  (`F32.div`).
- `mse_loss(input, target)` broadcasts the two shapes as `sub` does and
  computes `d = input - target` and `e = target - input`, then `mean(d * d)`.
  The tape saves `e` and `d`.
- `cross_entropy(logits, targets)` takes logits of `[rows, columns]` and one
  class index below `columns` per row, as a `List<U32>`. Per row it takes
  the stable `log_softmax` in PyTorch's order: `m` = the row maximum,
  `shifted = x - m`, `lse = log` of the sum of `exp(shifted)` from `+0.0`
  (`sum_to_size` to `[rows, 1]`), and `lp = shifted - lse`. The one-hot rows
  are `Matches` of the `[rows, 1]` targets and the `[1, columns]` class
  numbers. The loss is the sum of `-lp` at each row's target (`PickNegated`,
  `+0.0` elsewhere), in row order from `+0.0`, divided by the rows. The tape
  saves `lp` and the one-hot rows.
- The row maximum is PyTorch's `maximum` folded left to right from the row's
  first word: NaN when either word is NaN, else the larger word, and on a tie
  the later word. Base's `F32.max` drops a NaN in its second argument, so it
  is not used. Each output cell walks its own row, and the build forks by the
  work of a row, as `Matrix.splits` does.

Refusals: shapes that do not broadcast (`IncompatibleBroadcast`, 6), logits
that are not of rank 2 (`NotLogits`, 107), a target count other than the rows
or a target at or past `columns` (`TargetOutOfRange`, 108). PyTorch also takes
unbatched `[C]` logits and `[N, C, d...]` logits; the tape refuses both.
With no rows, `[0, columns]` logits have no cell for any number of columns:
the one-hot rows are then empty and the `[1, columns]` class numbers are not
built, so the loss is NaN (`+0.0 / +0.0`) and the gradient empty, as in
PyTorch.

## Backward

The new `Elementwise` operations take `g` (the rank-0 gradient of the loss,
broadcast) on the left and a saved word on the right, as `Mul` does in the
`Mul` VJP, so each VJP runs the `Mul` VJP's steps:

| Node | First contribution | Second contribution |
| --- | --- | --- |
| `MseLoss` | to `a`: `MseGrad{2 / N}(g, d) = (2 / N * d) * g` | to `b`: `MseGrad{2 / N}(g, e) = (2 / N * e) * g` |
| `CrossEntropy` | to `a`: `OneHotGrad{N}(g, onehot) = -(g / N)` at a target, `+0.0` elsewhere | to `a`: `SoftmaxGrad{N, single}(g, lp) = -(exp(lp) * r)`, `r = -(g / N)` for one column, `r = -(g / N) + 0.0` otherwise |

`2 / N` is `F32.div(2.0, N)` for the `N` cells of the broadcast shape;
`N` for `cross_entropy` is the number of rows. Each contribution is reduced to
its operand's shape with `sum_to_size`.

The orders are PyTorch's. `mse_loss_backward` computes `(norm * (a - b)) * g`
with `norm = 2 / N`, and its gradient for the target is the same with the
operands swapped. `nll_loss` backward gives `-(g / N)` at each target and
`+0.0` elsewhere; `log_softmax` backward gives `grad - exp(output) * sum(grad)`
per row. The row sum of that one-hot gradient is `-(g / N) + 0.0` when the
row has two or more columns, and `-(g / N)` itself when it has one: PyTorch's
sum of one word is that word, so at `g = +0.0` it stays `-0.0`, and adding
`+0.0` would turn it into `+0.0`. `SoftmaxGrad` carries `single`, whether the
logits have one column, and the VJP passes `Loss.one_class(input)`. The
`CrossEntropy` VJP sends the two terms as two contributions, and the logits'
working slot adds the second into the first, `gout + (-(exp(lp) * r))`, which
is `gout - exp(lp) * r` word for word. When the logits also feed another node,
the slot already holds that node's gradient and adds the two terms one at a
time, where PyTorch adds the finished `cross_entropy` gradient once.
`cross_entropy_fan_out` exercises that arrival order; on its words the result
equals PyTorch's.

## Laws

`src/tensor/loss-ops-LAWS.bend` pins each new operation to the F32 operations
it applies, in this order. `F32.div`, `F32.exp` and `F32.log` are opaque in
Base: the checker never evaluates them, so a law states which operation runs
on which words and says nothing about how close `exp` and `log` are to the
exact values. On both lanes they are the C library's or JavaScript's binary64
`exp` and `log` of the F32 value, rounded to F32.

`tensor_mse_vjp_values` and `tensor_cross_entropy_vjp_values` in
`src/autograd/tensor-vjp-LAWS.bend` state every cell of both contributions for
replayed saved tensors and a replayed `g`; their proofs follow the `Mul` VJP's
proof. No law states the forward value the tape records, nor the row maximum
or the one-hot rows; the runtime lanes check those against the Python
reference.

## Against PyTorch

PyTorch 2.14.1+cpu, AVX512, one thread. On the bench words (below),
`mse_loss` and its gradient are bit-identical; for `cross_entropy` about 10%
of the logits' gradient words differ by 1 or 2 ulp, and the loss by 1 to 28
ulp:

| output | words | bit-identical | max ulp |
| --- | --- | --- | --- |
| `cross_entropy` `[64, 10]` loss | 1 | 0 | 1 |
| `cross_entropy` `[64, 10]` gradient | 640 | 572 | 2 |
| `cross_entropy` `[128, 10]` loss | 1 | 0 | 4 |
| `cross_entropy` `[128, 10]` gradient | 1280 | 1157 | 2 |
| `cross_entropy` `[4096, 10]` loss | 1 | 0 | 28 |
| `cross_entropy` `[4096, 10]` gradient | 40960 | 36945 | 2 |
| `mse_loss` `[64, 10]` loss and gradient | 641 | 641 | 0 |
| `mse_loss` `[4096, 10]` loss and gradient | 40961 | 40961 | 0 |

The PyTorch record ([tensor backward](tensor-backward.md#pytorch-record))
compares these words under level 2 of the
[numeric policy](../AGENTS.md#numeric-policy): the `mse_loss` words within
the rounding-error bound of their sums, and the `cross_entropy` words, built
on `exp` and `log`, which no bound covers, on pins.

Where they differ, and why:

- The nll sum. PyTorch adds the `-lp` terms in a cascade of partial sums
  (blocks of 16 rows); the tape adds them left to right. An emulation of the
  cascade in F32 gives PyTorch's loss word exactly at every size tried.
- The `mse_loss` sum. PyTorch's `sum` is a cascade sum too. On random
  Gaussian words the left-to-right sum differed by 1 to 10 ulp from 320 to
  40960 cells and by 3851 ulp at 1048576 cells; on the bench words it
  agreed. The gradient, `(2 / N * d) * g`, agreed in every cell.
- `exp` inside PyTorch's `log_softmax` backward is a vectorized `exp` that is
  about 1 ulp from the correctly rounded value in some cells. On Gaussian
  logits of `[64, 10]` the tape's `lp` equals PyTorch's word for word, and
  PyTorch's own formula computed from its `lp` words with `torch.exp` misses
  PyTorch's gradient in 64 of 640 cells, as the tape does: the difference is
  the `exp` inside the backward kernel. The forward's `exp`
  and `log` (vectorized too) give `lp` words equal to PyTorch's in all 640
  cells there and in 1259 of 1280 at `[128, 10]`.
- One class. With logits `[[0.0]]` or `[[2.0], [-7.5]]`, target 0, and the
  loss times `g`, PyTorch's logits gradient is `+0.0` (`0x00000000`) at
  `g = +0.0`, `-0.0` and `1.0`, and NaN at a NaN `g`; the tape gives the same
  words (`cross_entropy_one_class_scaled`). With two classes the words agree
  too (`cross_entropy_two_class_scaled`); at `g = +0.0` the target cell is
  `-0.0` and the other `+0.0`.
- For more than 15 columns PyTorch's row sum of `exp` adds in vector lanes,
  so `lp` differs in the last bits; for 10 columns its partial-vector path
  adds left to right, as the tape does.

No tolerance is applied: the numeric policy of #47 does not exist yet.

## Cost

Forward and backward on the tape, median of 5 alternating rounds, per
iteration, at 1 thread: `cross_entropy` on `[64, 10]` takes 1.24 ms, 1.34
times a `Mul` of the same shape (0.92 ms); on `[4096, 10]` 117 ms against
`Mul`'s 68 ms. Threads do not help at these sizes. What limits the number at
`[4096, 10]` is `Gradient.accumulate`, the one-thread checked loop that adds a
contribution into a held gradient: about 29 ms per add of 40960 cells.
`cross_entropy` adds once into the slot held from the iteration before and
once more for its own second contribution; with `zero_grad` before each
backward it takes 87 ms, and the builds of its forward and backward take the
remaining 58 ms or so. Output words are byte-identical at 1, 8 and 24 threads.
