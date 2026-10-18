# AGENTS.md

bendtorch is a PyTorch-like tensor system written in Bend2. Tensor code lives
in Bend. Python only drives the gates in `tools/` and the separate PyTorch
oracle in `oracle/`.

## Use the pinned Bend2

`verification/references.json` pins the Bend2 commit. Bootstrap puts it, the
BendTT kernel, Lean and Clang in `.tools/`:

```sh
uv sync --locked
uv run --locked python -m tools.bootstrap
```

In this file, `bend` means the pinned Bend2:

```sh
export BEND_NO_TELEMETRY=1 BENDTT=$PWD/.tools/kernel/bendtt
alias bend="bun $PWD/.tools/bend/bend2/main.ts"
```

A global `bend` on your `PATH` can be a different version. Bend 2.0.25
rejects `src/shape/core.bend`. When the two disagree, the pinned one decides.

## Worktrees

Every task runs in its own worktree under `.worktrees/` in the primary
checkout, on a `bendtorch/<task>` branch. The primary checkout stays on a
clean `main`, and its `.tools/` holds the bootstrapped toolchain.

```sh
git worktree add .worktrees/<task> -b bendtorch/<task> main
cd .worktrees/<task> && ln -s ../../.tools .tools && uv sync --locked
```

- Base a stacked PR's branch on its parent PR branch instead of `main`.
  Bring later base commits in with a merge, which keeps the published
  history.
- Continue an existing task worktree rather than creating a second one for
  the same branch; `git worktree list` shows them.
- `.tools/` takes about 4 GB, most of it Lean, so worktrees link the primary
  checkout's copy instead of bootstrapping their own. A branch that changes
  the pin in `verification/references.json` replaces the link with its own
  bootstrap, as `docs/bend-pin.md` describes.
- Each worktree has its own `.venv`, so the gates run that branch's `tools/`
  and follow its `uv.lock`. Keep worktrees on the same filesystem as the uv
  cache: uv hardlinks packages from the cache there and copies them anywhere
  else, which matters for the PyTorch in the oracle environment.
- Gate runs write to the worktree's own `.evidence/`.

## When using Bend

- Run `bend guide` to learn it. Read `bend guide shaders` before you write
  parallel code.
- Use `LAWS.bend` to keep important rules.
- Run `bend PROOF.bend --verdict` before committing.
- Parallelize the code whenever possible.

## Keep rules in LAWS.bend

Each module keeps its own laws: `src/<module>/LAWS.bend` states the laws and
`src/<module>/PROOF.bend` proves each `law name` with a `def Laws.name`. The
root `LAWS.bend` and `PROOF.bend` import every module.
`verification/<module>.json` lists the declarations that the module's gate
requires in the exported proof of `src/<module>/PROOF.bend`.

- Add a law for each rule a later change must not break.
- Never delete, weaken or narrow an existing law to make a proof pass. Adding a
  hypothesis, a `where` clause or a smaller bound counts as weakening. If a law
  looks wrong, stop and say why.
- A law means what the definitions it calls mean. An edit to a definition that
  a law calls, such as `Shape.row_major` or `Model.observe`, changes the law.
- When a PR changes any `LAWS.bend` file, or a definition that a law calls,
  its description must have a section that names each added, changed or
  removed law, what changed in its statement, and why.
- Do not use `@unsafe` or `?TODO` in code that a law covers.

## Check before committing

`bend PROOF.bend --verdict` checks the laws with the Bend checker, then
rechecks them with the BendTT kernel. It checks only what a law states.
Behavior that no law states is checked only by the gates' native and JS
corpora, so run the gate of each module you touched too:

```sh
bend PROOF.bend --verdict
uv run --locked python -m tools.test
uv run --locked ruff check tools tests verification dev
uv run --locked ruff format --check tools tests verification dev
uv run --locked ty check
uv run --locked python -m tools.check <suite> --output .evidence/<new directory>
```

`tools.test` runs each test module in its own process, several at once; it runs
the same tests as `python -m unittest discover -s tests`. `tools.check` runs
several gate commands at once (`--jobs`) and each native Bend program on up to
8 threads (`--threads`). Neither changes what a passing gate accepts, its
observations or the order of the commands in `report.json`; timings, the thread
count in the recorded commands and `identity.json`, and the artifacts a failing
gate leaves behind do change. The families and the contracts run several
commands at once (`--workers`). Every default is a share of the CPUs the
process may use (`os.sched_getaffinity`, which `taskset` narrows), sized so
the checks above, the families and the contracts run together keep every CPU
busy without running many more commands than CPUs:
families 9/16, contracts 5/16, `tools.check` 1/8, `tools.test` 1/8 (36, 20, 8
and 8 on 64 CPUs), never below 1. Each Bend checker or BendTT
command keeps one CPU busy, and the native gate cases finish in milliseconds,
so more native threads would not speed a gate up. Pass the flags to override.
The checks above are independent, so run them
at the same time, each with its own evidence directory. Gate runs refuse to
overwrite an output directory.

While you iterate, `tools.check`, the families and the contracts take
`--cache DIR`, and `python -m tools.cache --cache DIR -- bun ... --verdict`
runs one command the same way. They replay a `--check-only` or `--verdict`
result, or a BendTT kernel run, when the checked file, every file it imports,
`bend2`, Bun, the kernel binary, the libraries they load, the arguments
and the environment are all unchanged (`tools/cache.py`). Entries are sealed
with a secret kept in the git directory, so a cache edited by hand or copied
from another checkout is run again. Exports (`-o`) always run. A command
whose inputs the key cannot pin, such as one under a Bun configuration file,
always runs too. Keep DIR under `.evidence/`. Each
replayed command is marked `"cached": true` in its record. The checks you
report in a PR run without `--cache`: a cold run is the reference. There is
no CI: run the
checks above locally, with the pinned Bend2, before you open a PR. Run the
PyTorch oracle recipes from `oracle/README.md` and `oracle/tensor/README.md` in
their own environment, so PyTorch never enters the gate. Every PR description
lists the commands you ran, their exit codes and the Bend commit.

## Report suspected Bend2 bugs

Bend2 is new, and its checker, compiler, runtime and BendTT kernel can have
bugs. Suspect one when the checker accepts a proof that should fail, rejects a
term that the guide says is valid, or crashes, or when the native and JS lanes
disagree, or when `--check-only` and `--verdict` disagree.

When you suspect a Bend2 bug:

1. Stop the task that depends on it.
2. Reduce it to the smallest `.bend` file that shows it, outside `src/`.
3. Run that file with the pinned Bend2 and with the latest release, and
   record both outputs. The installer puts the latest release at
   `~/.bend/bin/bend`; `~/.bend/bin/bend update` refreshes it.
4. Tell the user before you do anything else: what you expected, what Bend2
   did, the file, the commands and the versions.

Do not work around a suspected bug on your own unless it is listed under
Known Bend2 bugs. Do not weaken a law, add `@unsafe`, edit `.tools/`, change
the pin in `verification/references.json`, or open an upstream issue unless
the user asks.

## Known Bend2 bugs

The user has seen these bugs and approved the workaround each one lists. Use
the workaround; report any case it does not cover.

"A template parameter has no model". `--verdict` exports each `~` parameter
as an opaque constant and needs a model of its type. The model search tries
each constructor in turn, but it fills each field with the first model it
finds and never tries another value. So it finds no model for
`~make: Nat -> Pos` with
`Pos{n: Nat, positive: {Nat.is_lt(0n, n) == True{} : Bool}}`: it picks
`n = 0`, and `Pos{1n, {==}}` is never tried. `--check-only` passes and
`--verdict` fails with `no model for pick~make`. `Nat -> Shape.Shape` fails
the same way. It affects the pin `32d23b43` and release 2.0.37, the latest
release on 2026-10-14. Repro: `bend probes/template_model.bend --verdict`.
Workaround: give an index-map template the type `P -> Nat -> Nat` and state
its bound as a Bool hypothesis, such as `Pointwise.inside(...) == True{}`,
that each instance proves. A `~` parameter whose type is empty also has no
model; that failure is correct and is not this bug.

Fixed: "A match leaves behind proofs about a variable it moves" (`convoy_bind`
dropped a dependent parameter from a match's arms, so `--verdict` failed where
`--check-only` passed) affected `947db722` and was fixed upstream in
`8828cc19`. Repro: `bend probes/convoy_match.bend --verdict`. Law parameters
reordered as its workaround stay as they are.

## Parallelize

Write independent recursive work as a parallel call, `a b = f(xs) f(ys)`, and
split it into halves of equal work. The scheduler is a binary fork-join that
never moves a task after it starts, so an unbalanced split loses the speedup.
A `!` call, `f!(x)`, runs on the GPU in a native build. A `+` value that every
parallel call reads costs one atomic per read.

## Gates

`python -m tools.check` runs the gate of each named suite: `toolchain`,
`shape`, `storage`, `tensor` and `autograd`, or `all` of them. Each suite's
gate is `tools/<suite>.py`. A helper gate, such as `tools/autograd_backward.py`,
runs only if its suite's gate calls it.

The unit of a PR is one independently verifiable capability. Name files,
modules, gates and docs after what they check, not after a roadmap phase.
Experimental work stays in `dev/`, outside `src/`, until it is promoted.
`examples/` holds programs that use the library, such as the MNIST run.

The package is `bendtorch.bend` and what it imports. A new public module
gets an import there. The package ships nothing outside `src/`, the root
`LAWS.bend`, `PROOF.bend` and `LICENSE`, and names files under `src/` with
letters, digits and `_` only (BendHub refuses `-`), no two differing only by
case; `python -m tools.package`
checks it. [VERSIONING](docs/VERSIONING.md) has the version rules and the release
steps. Publishing to BendHub is permanent and only its owner does it.

No PR weakens a law, excludes a case or widens a numerical tolerance.

## Numeric policy

F32 results are compared at three levels. Each check uses one of them, and
its page says which.

**1. Against our own reference, same operations in the same order: bit for
bit.** Every gate lane, native and JavaScript, against its Python reference;
the ordered-fold words of the tensor supplement; the MNIST program against
the tape reference at any thread count. A NaN only has to be a NaN, except
where a check states the word, such as `relu` on the native lane.

**2. Against PyTorch, op by op.**

- An operation that rounds once per output word must give PyTorch's word,
  bit for bit: `add`, `sub`, `mul`, division, negation, comparisons, `max`,
  `relu` and its gradient, and the ops that only move words (copies,
  `reshape`, `transpose`, `permute`, `broadcast_to`). A word that only such
  operations build from the inputs is *fixed*: every correct run gives it.
- A sum, a matmul and every word built on one (`reduce_to`, a reducing
  `sum_to_size`, the backward through them, the losses) must lie within a
  bound on its rounding error, because PyTorch adds in its own order. Both
  PyTorch's word and ours must be within the bound of the exact result.
- The bound. With `u = 2^-24`, `eta = 2^-150` and
  `gamma(k) = k*u / (1 - k*u)`: a sum of `n` terms, in any order, is within
  `gamma(n-1) * sum |x_i|` of the exact sum; a dot product of `K` terms, in
  any order, with or without fused multiply-adds, is within
  `gamma(K) * sum |x_k * y_k| + K*eta / (1 - K*u)` (Higham, *Accuracy and
  Stability of Numerical Algorithms*, 2nd ed., sections 3.1 and 4.2). One
  rounding adds `u * |result|`, plus `eta` for a product or a quotient.
- The bound holds only under these conditions: binary32 arithmetic that
  rounds to nearest, ties to even; gradual underflow, so subnormals are kept
  and an addition is exact in the subnormal range; no intermediate overflow
  in any order; and `k*u < 1` for every count `k`. A word gets a bound only
  where the check can show them: `(1 + gamma(n)) * sum |terms| <= MAX`, the
  largest finite F32, for every sum and dot product, and
  `(1 + u) * (|result| + E) <= MAX` for every other rounding that is not
  fixed. `[MAX, -MAX, MAX]` sums to `MAX` left to right but overflows when
  the two `MAX` come first, so it has no bound. A word without a bound,
  finite or not, goes to the special-word checks below. An empty sum and a
  dot product of no terms are `+0.0` in every order: a fixed word.
- A word built from other words follows the reference's own operations.
  Each word carries its exact value, computed in exact rational arithmetic
  from the input words, and a bound `E` that holds for any run that does the
  same operations with each sum and matmul in any order. An operation first
  carries its operands' bounds through (for a product,
  `|a|*E_b + |b|*E_a + E_a*E_b`; a sum adds them), then adds its own
  rounding. `tools/autograd_tensor_bound.py` computes it for the tensor
  backward.
- The tensor layer's frozen corpus keeps its own published envelope
  (`oracle/tensor/README.md`). It is looser, counting `2K` roundings for a
  `K`-term matmul, and its version 3 adds exact words for `add`, `sub` and
  `mul` and the sign of a zero sum.
- `exp` and `log`. PyTorch's are vectorized approximations whose error this
  repository does not bound, so a word built on them has no bound. Such a
  word is *pinned*: its check lists PyTorch's word and ours, and no other
  word may differ. Today these are the `cross_entropy` words.
- A bound never replaces a stricter check. Where a check already pinned the
  words that may differ, or required some words to agree exactly, it keeps
  doing so, and the bound is checked on top: each pinned word must also lie
  within its bound. A bound counted per script, not per word, would let a
  second word move once one word differs.
- Where PyTorch rounds once and we round twice (an SGD step on AVX2 or
  AVX512 kernels, a fused multiply-add), the check compares with a reference
  that rounds as PyTorch does, word for word.
- Special words. A NaN must be a NaN; its sign and payload are not compared
  with PyTorch. An infinity must be the same word. A fixed zero keeps its
  sign. A word under a bound is compared by value, so a zero of either sign
  passes where the bound reaches zero: a fold from `+0.0` turns a one-term
  `-0.0` sum into `+0.0`.

**3. Training against a PyTorch copy.** The loss at each step must be within
a relative tolerance of PyTorch's, and the final metric within a band, over
a few seeds. Both are fixed from PyTorch against itself before any Bend
result is compared: run PyTorch with another rounding order (its `default`
kernels against its vectorized ones) and take the largest per-step
difference. Over the whole run the tolerance is the next power of ten above
it. An earlier window of steps, where that drift is still far below the
effect of a planted change, may get a tighter tolerance: one that sits
between the window's drift and the smallest planted change's effect in it,
with both numbers stated. The band for an accuracy `p` on `n` test rows is
`ceil(sqrt(p * (1 - p) * n))` rows. Planted changes (another learning rate,
a bias that never updates) must fail the tolerance. `docs/mnist.md` has the
MNIST instance: `1e-2` to the end of the epoch, the next power of ten above
the drift of 2.7e-3; `1e-4` for the first 100 steps, between a drift below
2.4e-7 and planted changes that show by 3e-4; a band of 26 or 27 rows;
seeds 0 to 2.

A bound, a tolerance or a pin is fixed before the result it judges, and is
never widened to let a result pass.
