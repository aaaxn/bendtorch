# S3.A prototypes (experimental)

Three automatic-differentiation designs over the same cases, compared in
[docs/s3-architecture.md](../../docs/s3-architecture.md). Nothing here is
imported by `src/` or by any gate.

| Design | Scalar core | Tensor version | Cases | Laws |
|---|---|---|---|---|
| A. Explicit reverse tape | `tape.bend` | `tape-tensor.bend` | `tape-cases.bend`, `tape-tensor-cases.bend` | `tape-LAWS.bend`, `tape-PROOF.bend` |
| B. One-shot affine pullback closures | `pullback.bend` | `pullback-tensor.bend` | `pullback-cases.bend`, `pullback-tensor-cases.bend` | `pullback-LAWS.bend`, `pullback-PROOF.bend` |
| C. Forward dual numbers (reference) | `dual.bend` | `dual-tensor.bend` | `dual-cases.bend`, `dual-tensor-cases.bend` | `dual-LAWS.bend`, `dual-PROOF.bend` |

- `tensor-ops.bend`: S2 wrappers shared by the tensor versions (ops that hand
  owners back, leading-axis unbroadcast, the optional-tensor store).
- `compare-LAWS.bend`, `compare-PROOF.bend`: A and B each equal C for every `x`.
- `bench.bend`: the repetition driver, `bench <job> <n>`.
- `kernel-fuel-probe.bend`, `kernel-fuel-adds.bend`: concrete tensor laws that
  pass `--check-only` but run the BendTT kernel out of fuel. These are kept as
  evidence of a known upstream limitation, not as laws.
