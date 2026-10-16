# Parallel build

`src/tensor/parallel.bend` holds the one parallel build that elementwise
`binary`, `matmul` and `reduce_to` share. An operation whose output cells are
independent hands it:

- `cell(p, position, state, inside)`: the F32 of output cell `position`, with
  the input state handed back. `inside` is `position < count`; a cell past the
  count is padding.
- `split(state)`: a copy of the state for each half of a fork. The state is a
  pair `A & B`: the two input arrays for elementwise and matmul, the
  contribution array and a `Unit` for `reduce_to`.
- `prepare(q)`: what every cell reads, built from the build's parameters `q`.
  It runs once per subtree below the forks. It builds a new record but does
  not copy the lists inside it, so subtrees may share them; a subtree is not
  tied to one worker thread.

The cells `base .. base + 2^depth` form a balanced tree. The top `splits`
levels fork, each half with its own copy of the state, and each subtree below
the forks fills its cells in index order on one thread. `cell`, `split` and
`prepare` are templates (`~` parameters), so each operation compiles its own
copy of the build, with no function argument passed at run time. That does
not inline their bodies: the cell stays a call of its own.

The bottom level of the tree is `pair`: it calls the cell for `base`, then for
`base + 1` with the state the first hands back, and builds the node of the two
leaves itself. So no leaf pays for an extra return through `leaf`, and the
bottom level, half of the tree's nodes, creates no closure. `leaf` builds
only a one-cell output. For `mul` on `[1024, 1024]` at one thread this makes
the build about 35% faster than the build with one call per leaf.

| Operation | `q` | `prepare(q)` | state | cell |
| --- | --- | --- | --- | --- |
| `Elementwise.build` | `Request{operation, plan}` | `Walk`: operation and index map | both inputs | `Elementwise.cell_at` |
| `Matrix.build` | `Dims{inner, columns}` | a copy of it | both inputs | `Matrix.cell_at` |
| `Reduce.build` | `Fibers{rows, extents}` | a copy of it | contribution, `Unit` | `Reduce.cell_at` |

## Laws

`src/tensor/parallel-LAWS.bend` states three laws for every choice of `cell`,
`split` and `prepare`:

- `parallel_build_balanced`: for every split, the build is a balanced tree of
  its depth. `Parallel.balanced` is the certificate each output storage
  carries.
- `parallel_build_cells`: if every cell hands the state back
  (`snd(cell(prepare(q), position, state, inside)) == state`) and the split
  copies it into both halves, then for every split each slot of the build,
  cell `base + offset`, holds `fst(cell(prepare(q), base + offset, state,
  base + offset < count))`.
- `parallel_build_sequential`: under the same two facts, the build with any
  split and the one-thread build (`splits = 0`) hold the same F32 in every
  slot.

The proofs are in `src/tensor/parallel-proof.bend`, proved once for the
templates. An operation proves only what its cell function computes: that it
hands the state back, that its split copies the state, and the value of an
inside cell. `elementwise-parallel-proof.bend`, `matmul-parallel-proof.bend`
and `reduce-parallel-proof.bend` do that (`cell_keeps`, `split_equal` or
`halves_equal`, `cell_out` or `cell_spec`) and get every cell of the build
from `Generic.build_cells`.

The live replay proofs (`contracts/binary-build-proof.bend`,
`contracts/matmul-parallel-proof.bend`, `reduce-replayed-proof.bend`) still
walk the shared tree themselves: they need the build equal to a spec tree as
a run-time equation, and the generic proof, which takes its hypotheses as
erased functions, gives only erased facts.

A new operation, such as a convolution or a softmax row, writes its cell
function and these three facts; the tree, the forks, the balance certificate
and the per-cell walk come from the generic law.

The mutants `high-fork-reuses-base`, `leaf-writes-zero` and
`pair-writes-zero` of the binary family edit `parallel.bend`; all keep the
build balanced, so the certificate still checks, and the laws reject them,
`parallel_build_cells` among them (`node_cell`, `leaf_facts`,
`pair_join_facts`).

BendTT exports each of the three generic laws once, checked at an opaque
constant for each template parameter, as
`parallel_2d_LAWS.parallel_build_*.qvvvvvvv`; `verification/tensor.json`
requires those names.
