# Differences from PyTorch

bendtorch implements a bounded subset of PyTorch's tensor API. This page lists
where a request that PyTorch accepts is rejected by bendtorch, where both
reject but report a different first error, and where results agree with
PyTorch only within a numerical envelope. Each row was observed with the
isolated oracle environment (PyTorch `2.14.1+cpu`, one thread,
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`), not inferred.

## Requests PyTorch accepts and bendtorch rejects

Of the 45 expected-error requests in `oracle/tensor/fixtures/s2-v2.json`, PyTorch
raises on 24 and accepts 21. The 21 accepted requests are deliberate
restrictions of the bounded API, not bugs:

| Corpus ids | bendtorch error | PyTorch behavior |
| --- | --- | --- |
| `rank5`, `rank_before_capacity`, `reshape_rank` | `RankExceeded` | accepts rank 5 |
| `capacity_1d`, `capacity_product`, `broadcast_output_capacity`, `matmul_output_capacity` | `CapacityExceeded` | accepts more than 2^20 elements |
| `over_u32_dimension` | `U32Range` | accepts a dimension of 2^32 |
| `coordinate_rank` | `CoordinateRankMismatch` | partial indexing: `a[0]` of `[2,3]` has shape `(3,)` |
| `negative_coordinate` | `U32Range` | wraps: `a[-1,0]` reads row 1 |
| `transpose_negative_axis` | `U32Range` | normalizes: `transpose(-1,0)` of `[2,3]` has shape `(3,2)` |
| `matmul_rank_left`, `matmul_rank_right` | `MatmulRankMismatch` | 1-D promotion: `[3]@[3,2]` and `[2,3]@[3]` have shape `(2,)`; a 1-D dot such as `[2]@[2]` has shape `()` |
| `matmul_batched_unsupported` | `MatmulRankMismatch` | batched: `[1,2,3]@[1,3,2]` has shape `(1,2,2)`, and `[4,2,3]@[3,5]` broadcasts to `(4,2,5)` |
| `dtype_float64`, `dtype_float16`, `dtype_bfloat16`, `dtype_int32`, `dtype_int64`, `dtype_bool`, `dtype_complex64` | `UnsupportedDtype` | accepts the dtype |

These three corpus requests call `Matrix.matmul`, which keeps its rank-2
contract. `Batched.matmul` in `src/tensor/batched-matmul.bend`, PyTorch's
`torch.matmul`, and the tape's
`batched_matmul` accept them with PyTorch's shapes; they refuse a rank-0
operand, as PyTorch does, and an output past 2^20 elements, which PyTorch
accepts.

API forms with no corpus case, which PyTorch accepts and bendtorch does not
offer:

- `reshape` with an inferred `-1` dimension: `reshape(-1,2)` of `[2,3]` has
  shape `(3,2)`. bendtorch dimensions are U32 and must be given in full.
- Negative axes in `permute` and in the tape's `sum_dims`: `permute(-1,0)`
  of `[2,3]` has shape `(3,2)`, and `sum(dim=-1)` of `[2,3]` has shape `(2,)`.
  An axis is a U32, and the word of `-1` is out of bounds.
- Slicing, advanced indexing and negative flat indices.

## Same request, different first error

Both reject `coordinate_rank_before_bounds` (coordinate `[9]` on shape
`[2,3]`), but the first error differs. bendtorch reports
`CoordinateRankMismatch`, because the frozen precedence checks coordinate rank
before bounds. PyTorch treats `[9]` as a partial index and reports
`IndexError: index 9 is out of bounds for dimension 0 with size 2`. The
bendtorch precedence is part of the frozen policy
(`oracle/tensor/numerical-policy.json`, `expected_errors.precedence`).

## Reduction order: parity with PyTorch is envelope-only

`sum`, `matmul` and the autograd `reduce_to` accumulate sequentially in
ascending row-major order from `+0.0`. PyTorch's CPU reduction uses a
vectorized order. The results are word-identical only for short inputs. For
uniform values in `[-8, 8]` (seed 0), PyTorch `sum`, the PyTorch gradient of
`expand` and the sequential fold give:

| n | `torch.sum` | torch `expand` gradient | sequential fold |
| --- | --- | --- | --- |
| 8 | `409203ba` | `409203ba` | `409203ba` |
| 16 | `4226d94c` | `4226d94c` | `4226d94b` |
| 33 | `424790f9` | `424790f9` | `424790fa` |
| 4097 | `43c0366d` | `43c0366d` | `43c03676` |
| 65536 | `c4cd82c4` | `c4cd82c4` | `c4cd829a` |

So from about 16 elements on, agreement with PyTorch is checked only within the
tensor reduction envelope (`oracle/tensor/README.md`). Word-exact checks of
these operations are against the ordered reference
(`oracle/tensor/ORDERED_REFERENCE.md`), not against PyTorch. A future pairwise
or blocked reduction needs its own law; the ordered law stays the reference.

The PyTorch outputs above come from a script run in the oracle environment;
it reproduces with `uv run --project oracle --frozen --no-sync python <script>`
and is not part of any gate.

## Scalar backward

The scalar `backward` in `src/autograd/backward.bend` agrees with PyTorch
scalar autograd word for word on 83 of the 91 scripts in
`oracle/autograd/fixtures/s3-backward-torch-v1.json`, including `None` gradients,
NaN payloads, infinities, `-0.0` and subnormals. The differences:

| Behavior | bendtorch | PyTorch |
| --- | --- | --- |
| Second `backward` on the same graph | Adds to the gradient slots, like `retain_graph=True` | Raises `RuntimeError` unless the first call passed `retain_graph=True` |
| `.grad` of a non-leaf | Every node has a slot | `None` unless `retain_grad()` was called (the oracle calls it on every node) |
| Node id the tape does not hold | `UnknownNode` (103) | No ids; the oracle's Python list raises `IndexError` (3 scripts) |
| Operand that is the node itself, a newer node or a missing node | `DanglingOperand` (104) for a forged tape | Cannot be built (3 scripts recorded as unrepresentable) |
| Saved operands that disagree with the leaves | Gradients use the saved values of a forged tape | Cannot be built (1 script, `forged_sound`, recorded as unrepresentable) |
| `NaN_a - NaN_b` with two payloads | Native keeps `NaN_a`'s payload | Keeps `NaN_b`'s payload (1 script); `add` and `mul` agree |

`src/autograd/backward-LAWS.bend` has the contract.

## Tensor backward

The tensor `backward` in `src/autograd/tensor-backward.bend` agrees with
PyTorch's gradients word for word on every script of
`oracle/autograd/fixtures/tensor-backward-torch.json` that PyTorch can
represent, including arrival order, broadcast reductions, `-0.0`, infinities
and subnormals, except the last bits of some words in the `mse_loss`,
`cross_entropy` and `batched_matmul` (`torch.matmul`) scripts, which the
table below explains. The
record is compared under level 2 of the
[numeric policy](../AGENTS.md#numeric-policy): fixed words bit for bit, the
`mse_loss` words on their pins and within the rounding-error bound, the
forward `sum` and `matmul` words within the bound with exact gradients, and
the `cross_entropy` words, which no bound covers, on their pins. 191 of the 249 scripts agree line for line, the five `linear`
scripts that PyTorch accepts among them, and every `relu` script but four:
`relu_edges` and `relu_nan_gradient` (NaN payloads of `NaN * NaN` in the
product after `relu`), `relu_13` (a one-term matmul) and
`relu_unknown_operand` (an intentional refusal); their gradients agree.
`relu` gives PyTorch's words at `-0.0` (kept), NaN (kept, with its payload),
`-inf` (`+0.0`), and its gradient is `+0.0` at `x = +0.0` and `-0.0` and `g`
at a NaN `x`. 16 of the 39 `mse_loss` and `cross_entropy` scripts agree line
for line; the table below says why the others differ. An SGD step, `p - lr * g`, rounds the product and then
the difference; PyTorch's CPU kernel for `param.add_(grad, alpha=-lr)` rounds
once (a fused multiply-add) when it dispatches to AVX2 or AVX512, as on the
machine that recorded the fixture, and twice, as bendtorch does, on its
`DEFAULT` kernel (`ATEN_CPU_CAPABILITY=default`). Bend2 has no fused F32
multiply-add, and its native lane is compiled without contraction. So every
non-NaN word of a step equals the `DEFAULT` kernel's; the NaN a step gives,
its sign and payload, depends on the kernel. A learning rate below zero is
refused as PyTorch refuses it (`lr < 0.0`); `-0.0`, NaN and `+inf` are
accepted. The differences:

| Behavior | bendtorch | PyTorch |
| --- | --- | --- |
| Seed | Ones of the output's shape, for any output | `backward()` needs a scalar output; the replay passes `torch.ones_like(output)` |
| Node id the tape does not hold | `UnknownNode` (103) | `IndexError` from the replay's Python list (8 scripts) |
| `sum_dims` axis past the rank, or listed twice | `AxisOutOfBounds` (13), `DuplicateAxis` (14) | `IndexError`, `RuntimeError` (1 script) |
| Shapes the operation refuses | The tensor layer's error code | `RuntimeError` (5 scripts) |
| Operand that is the node itself, a newer node or a missing node | `DanglingOperand` (104) for a forged tape | Cannot be built (3 scripts) |
| `NaN * NaN`, and an SGD step with a NaN parameter, gradient or rate | NaN | NaN with another sign or payload (5 scripts) |
| SGD learning rate below zero | `InvalidLearningRate` (110) | `ValueError` (1 script) |
| NaN losses | NaN | NaN with another payload (7 scripts) |
| `cross_entropy` logits | `[rows, columns]` only; a target count other than the rows is `TargetOutOfRange` (108) | Also `[C]` and `[N, C, d...]`; `ValueError` for the count |
| Sums inside `mse_loss` and `cross_entropy` | Left to right from `+0.0` | A cascade of partial sums; the loss can differ in the last bits (14 scripts, up to 3 ulp, every word pinned; the 6 `mse_loss` scripts' words also within the bound) |
| `exp` in the `cross_entropy` backward | Correctly rounded in practice (binary64 `exp`, rounded) | A vectorized `exp`; about 1 gradient word in 10 differs by 1 or 2 ulp |
| Forward `sum`, `sum_dims` and `matmul` | Ordered folds from `+0.0` (see above) | Another order, and `-0.0` from a one-term matmul (4 scripts, within the bound; the gradients agree) |
| `batched_matmul` (`torch.matmul`) refusals: a rank-0 operand, unequal inner extents, batches that do not broadcast, an unknown id | 105, 106, 6, 103 | `RuntimeError`, `IndexError` (`batched_refusals`, 1 script) |
| `batched_matmul` of NaN words | NaN | NaN with another payload (`batched_special_words`, 1 script) |
| Forward `sum` of a `batched_matmul` product | Ordered fold from `+0.0` | Another order (`batched_rounding_01`, 1 script, within the bound; the gradients agree) |
| `batched_matmul` gradients | A product of one term is folded from `+0.0`, so a `-0.0` product gives `+0.0`; a broadcast operand's gradient sums each batch's product in batch order | `-0.0`; the batch folded into one product. 3 scripts, 12 pinned words, each within the bound: 10 signed zeros, a gradient word 1 ulp and a forward `sum` 3 ulp apart |
| SGD step `p - lr * g` | The product rounded, then the difference | One rounding on AVX2 and AVX512, two on the `DEFAULT` kernel; `sgd_two_layers` and `sgd_rounding` match a one-rounding reference word for word (2 scripts; 6 of 337 words differ, by 1 to 6 ulp) |
| SGD step over a node that is not a leaf | `NotLeaf` (109) | `torch.optim.SGD` accepts a non-leaf that retains its gradient, as every non-leaf of the replay does (1 script) |
| `backward` of a constant, or of an operation on constants only | Seeds the output with ones and sends nothing on | `RuntimeError`: the output does not require grad (no script) |

`src/autograd/tensor-backward-LAWS.bend`, `tensor-vjp-LAWS.bend` and
`tensor-chain-LAWS.bend` in `src/autograd/` have the contract.
