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

## When using Bend

- Run `bend guide` to learn it. Read `bend guide shaders` before you write
  parallel code.
- Use `LAWS.bend` to keep important rules.
- Run `bend PROOF.bend --verdict` before committing.
- Parallelize the code whenever possible.

## Keep rules in LAWS.bend

Each stage keeps its laws in the modules it adds: `src/<module>/LAWS.bend`
states the laws and `src/<module>/PROOF.bend` proves each `law name` with a
`def Laws.name`. `verification/sN/LAWS.bend` and `verification/sN/PROOF.bend`
import a stage's modules. The root `LAWS.bend` and `PROOF.bend` import the
released stages. `verification/sN.json` lists the declarations the stage gate
requires in the exported proof.

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
Behavior that no law states is checked only by the stage gate's native and JS
corpus, so run the stage gate too:

```sh
bend PROOF.bend --verdict
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check tools tests
uv run --locked ruff format --check tools tests
uv run --locked ty check
uv run --locked python -m tools.sN --output .evidence/<new directory>
```

Gate runs refuse to overwrite an output directory. There is no CI: run the
checks above locally, with the pinned Bend2, before you open a PR. Run the
PyTorch oracle recipes from `oracle/README.md` and `oracle/s2/README.md` in
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

None on the current pin.

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

## Stages

Stages S0 to S7 stay as milestones, but the unit of a PR is one independently
verifiable capability. Each stage keeps a cumulative gate `tools/sN.py` that
dispatches over all of the stage's registered subgates, and one gate run in
`evidence/sN/` when the stage completes. Stage gates are the files matching
`tools/s[0-9].py`, so a helper gate such as `tools/s3_graph.py` must be
registered in `tools/s3.py`.

The R1 track (R1.A, R1.B) is the proof-interface work that precedes S3.
Experimental work, such as S3.A, is labeled experimental and stays outside
`src/` until it is promoted.

No stage advances by weakening laws, excluding cases or widening numerical
tolerances.
