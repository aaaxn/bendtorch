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

API forms with no corpus case, which PyTorch accepts and bendtorch does not
offer:

- `reshape` with an inferred `-1` dimension: `reshape(-1,2)` of `[2,3]` has
  shape `(3,2)`. bendtorch dimensions are U32 and must be given in full.
- Negative axes in `permute`: `permute(-1,0)` of `[2,3]` has shape `(3,2)`.
- Per-axis `sum(dim, keepdim)`: `sum(dim=0, keepdim=True)` of `[2,3]` has shape
  `(1,3)`. bendtorch's `sum` reduces every element to a scalar `[]`.
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

[docs/backward.md](backward.md) has the contract.

## Tensor backward

The tensor `backward` in `src/autograd/tensor-backward.bend` agrees with
PyTorch's gradients word for word on every script of
`oracle/autograd/fixtures/tensor-backward-torch.json` that PyTorch can
represent, including arrival order, broadcast reductions, `-0.0`, infinities
and subnormals. 113 of the 138 scripts agree line for line, the five `linear`
scripts that PyTorch accepts among them, and every `relu` script but four:
`relu_edges` and `relu_nan_gradient` (NaN payloads of `NaN * NaN` in the
product after `relu`), `relu_13` (a one-term matmul) and
`relu_unknown_operand` (an intentional refusal); their gradients agree.
`relu` gives PyTorch's words at `-0.0` (kept), NaN (kept, with its payload),
`-inf` (`+0.0`), and its gradient is `+0.0` at `x = +0.0` and `-0.0` and `g`
at a NaN `x`. An SGD step, `p - lr * g`, rounds the product and then the
difference; PyTorch's CPU kernel for `param.add_(grad, alpha=-lr)` rounds
once (a fused multiply-add) when it dispatches to AVX2 or AVX512, as on the
machine that recorded the fixture, and twice, as bendtorch does, on its
`DEFAULT` kernel (`ATEN_CPU_CAPABILITY=default`). Bend2 has no fused F32
multiply-add, and its native lane is compiled without contraction. The
differences:

| Behavior | bendtorch | PyTorch |
| --- | --- | --- |
| Seed | Ones of the output's shape, for any output | `backward()` needs a scalar output; the replay passes `torch.ones_like(output)` |
| Node id the tape does not hold | `UnknownNode` (103) | `IndexError` from the replay's Python list (7 scripts) |
| Shapes the operation refuses | The tensor layer's error code | `RuntimeError` (5 scripts) |
| Operand that is the node itself, a newer node or a missing node | `DanglingOperand` (104) for a forged tape | Cannot be built (3 scripts) |
| `NaN * NaN`, and an SGD step with a NaN parameter and a NaN gradient | NaN | NaN with another payload (4 scripts) |
| Forward `sum` and `matmul` | Ordered folds from `+0.0` (see above) | Another order, and `-0.0` from a one-term matmul (3 scripts; the gradients agree) |
| SGD step `p - lr * g` | The product rounded, then the difference | One rounding on AVX2 and AVX512, two on the `DEFAULT` kernel; `sgd_two_layers` and `sgd_rounding` match a one-rounding reference word for word (2 scripts; 6 of 337 words differ, by 1 to 6 ulp) |
| SGD step over a node that is not a leaf | `NotLeaf` (107) | `torch.optim.SGD` accepts a non-leaf that retains its gradient, as every non-leaf of the replay does (1 script) |

[docs/tensor-backward.md](tensor-backward.md) has the contract.
