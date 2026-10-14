# Bumping the Bend pin

bendtorch follows the newest upstream Bend2 main branch at
`https://github.com/bendlang/bend` (it moved from `HigherOrderCO/Bend`, which
redirects). `verification/references.json` pins one exact commit, and
`tools/bootstrap.py` fetches that commit into `.tools/bend`. Bump the pin
every time upstream main moves, in a PR of its own titled
`Bend: bump pin to bendlang/bend <short sha>`.

## 1. Pin the new commit

Resolve upstream main and record the full SHA and its date:

```sh
git ls-remote https://github.com/bendlang/bend refs/heads/main
```

In a fresh clone of upstream, outside the repository, compute the new hash of
every file listed under `bend_files`:

```sh
git clone https://github.com/bendlang/bend.git bend && cd bend
git checkout <sha>
sha256sum bend2/bend.ts bend2/comp.ts bend2/main.ts bend2/safe.ts \
  bend2/bendtt.lean bend2/base.bend bend2/pack/bun.lock \
  demos/pure_par_sum/main.bend demos/pure_par_sum/LAWS.bend \
  demos/pure_par_sum/PROOF.bend
```

Edit `verification/references.json`: set `sources.bend.url` and
`sources.bend.commit`, and replace every `bend_files` hash. Check
that upstream still needs the pinned tools: `LEAN_VERSION` in
`bend2/safe.ts` must equal `tools.lean`, and the Bun version Bend asks for
must match `tools.bun`. If one changed, update `tools` and `downloads` too.

`.tools/bend` is fetched only when it is missing, so move the old checkout and
kernel aside before you bootstrap again:

```sh
mv .tools .tools.old-<old short sha>
uv sync --locked
uv run --locked python -m tools.bootstrap
git -C .tools/bend rev-parse HEAD
```

The last command must print the new SHA.

## 2. Verify the toolchain, shape, storage and tensor gates on the new pin

Run every check in AGENTS.md "Check before committing" with the new
`.tools/`. The root verdict, the full unittest discovery and the gates
are heavy verification and need approval for the job. Use new evidence
directory names that include the short SHA:

```sh
export BEND_NO_TELEMETRY=1 BENDTT=$PWD/.tools/kernel/bendtt LEAN_STACK_SIZE_KB=4194304
bun .tools/bend/bend2/main.ts PROOF.bend --verdict
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check tools tests
uv run --locked ruff format --check tools tests
uv run --locked ty check
uv run --locked python -m tools.check toolchain --output .evidence/toolchain-bend-<short sha>
uv run --locked python -m tools.check shape storage --output .evidence/shape-storage-bend-<short sha>
uv run --locked python -m tools.check tensor --output .evidence/tensor-bend-<short sha>
```

Also run the native matmul smoke from `docs/sum-and-matmul.md` and compare its full
output with the documented words. Record each command, its exit code and its
wall time for the PR description.

## 3. Triage every difference

Compare each failure or changed output with the old pin. Put each one in one
of three classes:

- An intended upstream change. Adapt bendtorch without weakening any law. If
  a `LAWS.bend` file or a definition that a law calls changes, the PR
  description names each changed law, as AGENTS.md requires.
- A Bend2 bug. Follow AGENTS.md "Report suspected Bend2 bugs": confirm it
  against `bend guide`, reduce it to the smallest `.bend` file outside `src/`,
  run it on the new commit and on the old pin, record the commands and
  outputs, and stop. Do not work around it.
- A bendtorch bug that the old pin let through. Fix it in bendtorch and
  name any law it touches.

Recheck each entry under AGENTS.md "Known Bend2 bugs" on the new commit with
its example, under both `--check-only` and `--verdict`. If an entry is fixed,
say so in the PR and propose removing it.

The PR description carries the new run's commands and results.

## History

| Commit | Date | Differences found |
| --- | --- | --- |
| `0e14aa8` | 2026-10-12 | Bend 51ab8e8 looks for CUDA at `/usr/local/cuda` and `/opt/cuda` when `$CUDA_HOME` has none, so `bend -o` linked the CUDA lane into the toolchain gate's `!` parallel-sum probe on a CUDA host. The toolchain gate now emits C and compiles it with the pinned clang, as the shape, storage and tensor gates already do. The "match leaves behind proofs" Known Bend2 bug no longer reproduces (fixed upstream in 8828cc1). |
| `63b2c0d` | 2026-10-13 | None. Bend 2.0.37 plus fixes; Lean stays 4.34.0 and `bendtt.lean` is unchanged. `--verdict` now prints the kernel's failure lines (upstream f846cb5), so a kernel failure such as `experiments/autograd-prototypes/kernel-fuel-adds.bend` reads `In probe: out of fuel` under the verdict. Base `F32.round` no longer rounds `a + 0.5` (upstream 63b2c0d); bendtorch does not call it. The kernel fuel limit (bendlang/bend #1322, #1193, both open) is unchanged. |
| `32d23b4` | 2026-10-13 | None. One upstream commit, which changes only Base: `List.any`, `List.all`, `List.find` and `List.contains` now carry the result so far and stop once it is known (upstream #1548). Nothing in bendtorch or elsewhere in Base calls them. The checker, `bendtt.lean`, the compiler and Lean 4.34.0 are unchanged, and so is the kernel fuel limit (bendlang/bend #1322, #1193, both open). |
