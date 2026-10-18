# Versions and releases

## Version numbers

BendHub names a package version with four numbers, `bendtorch-core@A.B.C.D`. The
repository uses the same four numbers, so the tag `vA.B.C.D`, the
`CHANGELOG.md` heading and the import line all name one version.

| Part | Changes when | Example |
| --- | --- | --- |
| `A` | bendtorch is stable and its API changes incompatibly. It stays `0` while bendtorch is an experiment. | `1.0.0.0` |
| `B` | A capability is added, or, while `A` is `0`, a module's public definitions change incompatibly | `0.2.0.0` |
| `C` | A fix, a speed-up, a new law about existing code, or a new Bend pin, with no public definition removed or changed | `0.1.1.0` |
| `D` | The shipped files change without changing code: a comment, the LICENSE, the package entry | `0.1.0.1` |

A published version is public and permanent, and its address is the hash of
its files, so changing one byte of a shipped file needs a new version. Laws are
append-only across versions: a law shipped in one version is not weakened or
removed in a later one (`AGENTS.md` applies the same rule to every PR).

## What the package ships

`bend bendtorch.bend --publish bendtorch-core@A.B.C.D` uploads `bendtorch.bend`
with every file it imports: the modules of `src/`, the root `LAWS.bend` and
`PROOF.bend` with every module's laws and proofs, and the root `LICENSE`.
`tools/`, `tests/`, `verification/`, `oracle/`, `examples/`, `dev/`, `probes/`
and `docs/` stay in the repository. `uv run --locked python -m tools.package`
prints the shipped files and fails if the entry pulls in anything else.

A program imports the module it needs, by name or by hash:

```python
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor-backward.bend as Backward
```

An import is local to the file that makes it, so importing `bendtorch.bend`
does not give access to the modules it imports. `bend bendtorch.bend
--verdict` checks every law the package ships.

## Releasing

1. On `main`, the nine gates and `bend PROOF.bend --verdict` exit 0
   (`make verify`).
2. Work in progress collects under `## A.B.C.D (unreleased)` at the top of
   `CHANGELOG.md`, with the version in the first line of `bendtorch.bend`;
   `make package` checks that the two agree. The release PR replaces
   `unreleased` with the date.
3. After that PR merges, tag the merge commit `vA.B.C.D` and make a GitHub
   release with its `CHANGELOG.md` notes.
4. `bend login`, then `bend bendtorch.bend --publish bendtorch-core@A.B.C.D`. The
   first publish registers the name `bendtorch-core`. This step cannot be undone.
5. Record the printed `0x` hash in the release notes, and import the version
   from an empty directory to check it.
