# Ordered F32 reference contract

The ordered reference checks a different claim from the frozen numerical
compatibility envelope. A reference backend must round every primitive in the
specified sequence. A later optimized backend may reassociate or fuse operations
if its separately declared contract permits that and the unchanged compatibility
envelope accepts the result. Neither contract implies the other.

For `sum_cancellation_0`, both positive zero and `1.0` pass the existing
`tools.tensor_numerics.verify_envelope`. Only positive zero satisfies the ordered
reference. Accepting `1.0` in the compatibility check is intentional, not a
tolerance defect. Do not tighten that check to make it assert sequence equality.

## Independent primitive model

`ordered_reference.py` is an isolated oracle, not a tensor execution backend.
It imports neither PyTorch nor the existing fixture generator. Finite words
decode to signed integer coefficients times powers of two. Addition aligns those
coefficients, and multiplication multiplies them. Each primitive result rounds
to binary32 with integer quotient, remainder and ties-to-even logic. No host
floating-point arithmetic enters this model.

This implementation differs structurally from the existing rational
single-final-rounding reference. Reusing that reference would not check an
ordered sequence. Host binary64 arithmetic followed by `struct.pack` would add
host rounding to the trusted boundary. The integer model makes the intended
rounding explicit, but its implementation is still trusted reference code, not
a formally proved IEEE implementation.

The supported reference sequence is deliberately narrow.

- Sum starts at positive zero and adds words in logical row-major order.
- Each rank-two matrix cell starts at positive zero and visits increasing inner
  indices. Multiplication rounds before the accumulator addition rounds.
- Matrix product plus bias completes the product sequence and then performs one
  separately rounded bias addition per cell, with ordinary trailing-axis
  broadcasting.

Signed zeros are exact. Exact cancellation under nearest-even produces positive
zero, and adding two negative zeros produces negative zero. Multiplication uses
the XOR of the signs, including a result rounded to zero. Gradual underflow is
required. Subnormal results cannot be replaced by zero.

Infinity signs and invalid-arithmetic NaN classifications are checked outside
any tolerance. The model uses a canonical NaN word only to serialize its result.
Neither the native nor the JavaScript backend must reproduce that payload, sign
or quiet bit. A NaN must remain a NaN. This is a classification-level exception
to exact word comparison, not universal NaN-payload agreement.

## Portable records

`fixtures/ordered-reference.json` contains all 51 successful sum, matmul and
matmul-plus-bias cases in the original frozen corpus. Each record identifies its
case, binds the operation, inputs and parameters with SHA256, and stores the
ordered shape, words and classifications. It never takes its expected result
from `outputs.result`, `independent_reference.result` or the comparison policy.
The hash uses sorted-key compact JSON with the three keys `op`, `inputs` and
`parameters`. Absent parameters mean an empty object.

These portable numerical records contain no CPU provenance. The existing
PyTorch records, compatibility policy, historical evidence and CPU provenance
remain untouched. The synthetic corpus is development-used conformance data,
not independent held-out evidence or an ML accuracy evaluation.

`fixtures/ordered-controls.json` adds two hand-derived controls without editing
any existing case.

| Control | Ordered result | Alternative result |
| --- | --- | --- |
| `[2^24, 1, -2^24]` sum | positive zero | reassociated sum can return `1.0` |
| `[-1, 1+2^-23] @ [1, 1-2^-23]` | positive zero after separate product rounding | fused last multiply-add returns `-2^-46` |

The second result is `a8800000` in binary32. These are mathematical controls,
not claimed PyTorch or Bend observations. The focused tests exercise the actual
model and root comparator. Native execution of the controls remains required
at integration.

## Root comparison boundary

`tools.tensor_ordered` reads portable records and already-parsed Bend observations.
It does not import or execute the oracle. `verify_ordered_lane` checks the full
51-case reference inventory and each input binding. `verify_ordered_case`
requires exact finite and infinity words, including zero signs and subnormals.
It derives classes from words, because the existing native parser does not add
classifications. Supplied classifications must agree with their words.

`tools.tensor.run_gate` applies this comparison to both current reference lanes,
alongside the unchanged `verify_lane` compatibility comparison. Each lane's
acceptance evidence retains `numerical_cases` and adds `ordered_reference` and
`ordered_controls`. Do not apply the ordered check to a future
optimized backend while claiming that only the compatibility envelope is its
contract. Keep the reference backend available for sequence conformance.

The stage and runtime snapshots bind these records, the root comparator, the
independent model and its focused tests. The existing protected hash checks
also cover the new records and controls. The runtime runs the two controls
separately through `decode_case`, `bend_program` and the existing output parser
in native and generated JavaScript executables. They are not inserted into the
historical corpus, and historical evidence is not rewritten.
The controls are valid requests using only `id`, `op` and `inputs`; their
`ordered_result` fields are comparator expectations, not execution inputs.

## Focused verification

Use the existing isolated oracle interpreter and pinned root development tools
on the authorized host, under an allocated CPU slot.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/tensor/ordered_reference.py --check
(cd oracle/tensor && PYTHONDONTWRITEBYTECODE=1 uv run --project .. --frozen --no-sync python -m unittest discover -s tests -p test_ordered_reference.py -v)
uv run --locked python -m unittest discover -s tests -p test_tensor_ordered.py -v
uv run --locked python -m unittest discover -s tests -p test_tensor_numerics.py -v
uv run --locked ruff check --config pyproject.toml oracle/tensor/ordered_reference.py oracle/tensor/tests/test_ordered_reference.py tools/tensor_ordered.py tests/test_tensor_ordered.py
uv run --locked ruff format --check --config pyproject.toml oracle/tensor/ordered_reference.py oracle/tensor/tests/test_ordered_reference.py tools/tensor_ordered.py tests/test_tensor_ordered.py
uv run --locked ty check --python oracle/.venv/bin/python --python-version 3.12 --extra-search-path oracle/tensor --config 'src.include=["oracle/tensor/ordered_reference.py", "oracle/tensor/tests/test_ordered_reference.py", "tools/tensor_ordered.py", "tests/test_tensor_ordered.py"]' --error-on-warning
```

Explicit oracle paths prevent Ruff's directory exclusion from producing a
zero-file check. The ty include override includes the isolated helper and test.
Root lint and type checks still run at integration. Full gates and native runs
are separate, approval-gated checks, not an implication of these stdlib tests.

If the isolated oracle environment is absent, these stdlib-only checks can use
the locked root interpreter without installing PyTorch. This does not verify
the oracle's locked CPython 3.12 environment.

```sh
uv run --locked python -B oracle/tensor/ordered_reference.py --check
PYTHONPATH=oracle/tensor uv run --locked python -B -m unittest discover -s oracle/tensor/tests -p test_ordered_reference.py -v
```

`--check` is read-only and fails on a mismatch. `--write` creates only an absent
output and refuses to overwrite an existing file. A mismatch needs investigation
and retained evidence. It is not a reason to regenerate expectations.
