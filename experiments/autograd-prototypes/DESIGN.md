# Autograd prototypes: experimental AD architecture comparison

**Status: EXPERIMENTAL.** This experiment does not establish a verified
autograd foundation, and the proof-interface work is still open. All code lives
in `experiments/autograd-prototypes/`, outside `src/`. No `src/` file and no
`LAWS.bend` under `src/` or `verification/` changed. Nothing here is a
benchmark claim.

- Repository base: `071d9cc` (main after #8, #11 and #12; Bend pin `0e14aa8`). The measurements below are from one run at this revision. An earlier run on `23c1536` agreed within 3%.
- Bend: pinned `bendlang/bend` `0e14aa8b3082e68f419320b2c785762de1d0577b`
  (2.0.36), bootstrapped by `tools.bootstrap`. This is also newest upstream
  main on 2026-10-12. Every check, case file and probe also ran on the old pin
  `947db722` and gave the same results. Nothing worked on only one of the two.
- Host: a local Linux x86_64 machine, in an isolated worktree. All native runs used
  `--threads 1 --gpu off`, pinned to one core with `taskset -c 8`.

## Recommendation

Promote the **explicit reverse tape (A)** as the base of the autograd building
blocks. It handled every case. It keeps each tensor in exactly one owner, the
store. It sends both contributions of `x*x` and `u+u` to one node id. Its
scalar laws compose with the other designs' laws under `--verdict`. The
ownership story was the easiest of the two reverse designs to state: "a value
lives in one store slot; an op takes it out and puts it back." The cost is code
size: about 1.5x the lines of the closure design at the tensor level, mostly
plumbing that the autograd building blocks can factor out.

Keep **forward duals (C)** as the independent reference. It is the smallest
design and the only one with no graph. Its tensor cost grows with the
parameter count: 12 forward passes for 12 parameters, about 13x the reverse
designs here.

Do not promote **pullback closures (B)** as the base. It is correct and as fast
as A. Its proof surface is harder to work with: gradients live inside affine
closures, so a law cannot inspect the graph, the saved values or the
contribution targets without running the closure. Each extra use of a value
needs an explicit `share`, which clones the tensor in the forward pass. Case 4
needed 3 forward clones in B and none in A.

## Cases and results

All three designs ran the same cases. The outputs below are decoded F32 values
from the native binaries. The JS lane output is byte-identical for every case
file.

| Case | Expected | A tape | B closures | C duals |
|---|---|---|---|---|
| 1. `x^2+x` at 3 | 12, d=7 | 12, 7 | 12, 7 | 12, 7 |
| 2. `u=x^2; y=u+u` at 3 | 18, d=12 | 18, 12 | 18, 12 | 18, 12 |
| 3. `x1*x2`, distinct inputs both 3 | 9, (3, 3) | 9, (3, 3) | 9, (3, 3) | 9, (3, 3) |
| 4. `y=sum(W*x+b)`, W:[2,3], x,b:[3] | see below | match | match | match |
| 1 on rank-0 tensors | 12, 7 | 12, 7 | 12, 7 | 12, 7 |

Case 4 uses `W=[[1,2,3],[4,5,6]]`, `x=[0.5,-1,2]` and `b=[1,-1,0.5]`, with
tensor `Elementwise.mul`/`add` (broadcasting `[3]` against `[2,3]`) and
`Reduction.sum_dynamic`. The expected values are `y=14.5`,
`dW=[[0.5,-1,2],[0.5,-1,2]]`, `dx=[5,7,9]` and `db=[2,2,2]`. A and B return
the original input tensors next to their gradients. C returns one tangent per
pass. A dangling operand id on the tape answers `Missing` (error 400) and
builds no alias. Applying a B pullback value or a tensor twice without
`share` or `clone` is rejected by the checker: "x (consumed more than once)".

The reduction of a broadcast cotangent over the leading axis (`dW*x` to `[3]`,
`g` to `[3]`) is `ones[1,r] @ g` through tensor `Matrix.matmul`, then a reshape
(`experiments/autograd-prototypes/tensor-ops.bend`, `unbroadcast`). It supports only that
pattern and answers `Unsupported` otherwise.

## Measurements

Native times come from `experiments/autograd-prototypes/bench.bend`. Each job runs the case
`n` times with inputs that change per iteration, and folds every value and
gradient into a printed checksum, so the work cannot be skipped. The checksums
are identical across the three designs and across the native and JS lanes. Each
value is the median of 5 runs, single thread. The spread was within 3%.
`construct` builds the three case-4 tensors, which is the floor that every
case-4 run pays.

| Measure | A tape | B closures | C duals |
|---|---|---|---|
| Scalar cases 1-3, per iteration (n=2,000,000) | 1.50 µs | 1.15 µs | 0.01 µs |
| Case 4, per iteration (n=20,000) | 31.5 µs | 31.0 µs | 413 µs |
| Case 4 construct floor (n=200,000) | 1.75 µs | 1.75 µs | 1.75 µs |
| Peak RSS, any job | 2.5 MB | 2.5 MB | 2.5 MB |
| Case 4 forward operand clones | 0 | 3 (one per input, so the caller keeps its copy) | 0 per pass, 2 per `mul` |
| Case 4 backward cotangent clones | 2 (one per binary node) | 2 | none |
| Case 4 saved values | 6 (every node value until the sweep pops it) | 2 (`mul` operands in its closure) | none (12 passes) |
| `x*x` handling | 1 operand clone, both contributions to the original id | `share`: 1 clone and a fresh slot per extra use | dual is `Data` at scalar level; at tensor level 2 clones per extra use |
| `--check-only`, prototype + cases | 0.30 s | 0.29 s | 0.29 s |
| `--verdict`, scalar laws | 0.22 s | 0.21 s | 0.19 s |
| Native build of the tensor case file | 4.2 s | 4.2 s | 2.8 s |
| Prototype lines (scalar + tensor, no blanks/comments) | 116 + 297 | 76 + 190 | 33 + 64 |
| Top-level defs (scalar + tensor) | 25 + 55 | 20 + 39 | 10 + 11 |

The shared tensor wrappers and store helpers in `tensor-ops.bend` (190 lines)
are used by all three designs and are not counted above. Check times are close
to the cost of loading the tensor layer itself (0.29 s for `tensor-ops.bend`
alone), so they do not separate the designs.

Notes on the numbers:

- The scalar dual timing is not comparable with the others. The scalar
  prototypes are templates over `T`, and at F32 with no graph, clang folds most
  of the dual arithmetic. It scaled linearly (0.02 s, 0.23 s for 2M, 20M) but
  should not be read as "duals are 100x faster".
- Peak RSS is the runtime's 2 MB floor for every job. At these sizes no design
  shows measurable memory pressure. The memory column is not evidence for or
  against any design.
- Case 4 cost is dominated by tensor operations. Each `Elementwise` call and
  each `matmul` goes through broadcast planning and checked storage. The 1.75
  µs construct floor and about 31 µs per reverse pass (two elementwise ops, one
  sum, the backward ops and two unbroadcast matmuls) make the tape bookkeeping
  itself a small share.

## Laws and proof composition

Each design has a `LAWS.bend`/`PROOF.bend` pair under `experiments/autograd-prototypes/`. The
scalar core of each prototype is a template over `T` with `~plus`/`~times`, so
the same code that runs at F32 in the binaries is instantiated at `U32` and
`Nat` in the laws. F32 literals do not reduce in the checker, for example
`F32.mul(3.0, 3.0) == 9.0` fails, so F32 laws are out of reach by evaluation.

| Law | A | B | C |
|---|---|---|---|
| Case 1 at 3 (U32): `(12, [7])` | pass | pass | pass |
| Case 2 at 3 (U32): `(18, [12])` | pass | pass | pass |
| Case 3, distinct inputs (U32): `(9, [3, 3])` | pass | pass | pass |
| Case 1 for every `x: Nat`: equals hand derivative `(x*x+x, [1+x+x])` | pass, 1 rewrite | pass, 2 lemmas, 2 rewrites | pass, 3 rewrites |
| A and B each equal C for every `x` (`compare-PROOF.bend`) | pass | pass | n/a |

All of these pass `--check-only`, `--verdict` and direct `bendtt` on the `-o`
export. A changed expected value (`(18, [6])`) is rejected. The universal-`x`
law shows that the tape normalizes under a free variable. The result is an
arithmetic term, and proving it is ordinary `Nat` rewriting. `compare-PROOF.bend` checks in 0.34 s with `--verdict`. The cross-design
law composes the three proofs with `Equal.trans`/`Equal.sym` and no new
reasoning.

The arithmetic shape of each design's normal form decides the proof effort. The
tape produced `1+(x+0)+(x+0)`, the closures produced
`(x+0)+((x+0)+1)` and the duals produced `(x+0)+x*1+1`. The tape's
accumulation order gave the shortest proof.

## Design constraint: tensor gradient laws must be generic

Closed evaluations of concrete tensor computations do not pass `--verdict`.
`experiments/autograd-prototypes/kernel-fuel-probe.bend` states whole-run facts about case 4
and rank-0 case 1 (input count, clones, saved values). `--check-only` accepts
them in 0.7 s. The BendTT kernel answers "out of fuel" on each one (its `FUEL`
is 400,000,000 steps). The minimal repro, `kernel-fuel-adds.bend`, is three
chained rank-0 `Elementwise.add` calls. Two chained adds still pass (about 8 s
in the kernel). The old pin behaves the same. This is a known upstream kernel
limitation, not a new bug, and it is not worked around here.

So under `--verdict`, tensor gradient laws must be generic statements over
producers, following the proof-boundary pattern: universal shapes and values,
proved by the actual definitions. Concrete numeric gradients belong to the
native runtime tests and the oracle.

How each prototype fits that style:

- **A tape:** fits best. The graph is first-order data (`Node` ids plus a
  store), so laws can quantify over a tape and state per-node facts. Examples:
  the sweep sends `g` to both operand ids, a `Mul` contribution is `g` times
  the other operand's stored value, and every id below the head is visited
  once. Each of these is a structural induction over the node list. No closure
  needs to be run.
- **B closures:** fits poorly. A pullback is an opaque affine function, so a
  generic law must quantify over closures and can only state facts by applying
  them. The graph, the saved values and the `share` slots are not visible as
  data.
- **C duals:** fits well for per-op laws, such as the tangent of
  `mul(Dual{a,da}, Dual{b,db})` being `da*b + a*db` for all tensors. It is the
  natural generic reference, but it states directional derivatives and not a
  full gradient.

## Open questions for the autograd laws

1. **Generic tensor laws.** The autograd laws must state the tape's correctness
   generically, as the derivative of an op times the cotangent, accumulated per
   id, for all shapes and values. What is the smallest tensor law surface it
   can build on? Today the tensor layer has retention and success laws for
   elementwise ops. Elementwise semantics ("output[i] = op(left[j], right[k])")
   would be needed to state that a contribution equals `g*b` element by
   element.
2. **Unbroadcast.** The prototype reduces only the leading axis, through
   `matmul` with a ones row. The autograd building blocks need a real
   sum-over-axes reduction in the tensor layer (or the autograd layer) with its
   own laws, and a decision on whether it is a tensor capability.
3. **Store representation.** The tape store is a newest-first list of
   `Maybe<Dynamic>` with O(n) take/put. That is fine at four nodes. The first
   autograd implementation should decide between a list, an owned `Array`
   (needs a `Type`-element array API; `Array.get` requires `Data`) or a tree.
   It should also decide whether ids carry an erased proof that they are below
   the head, so a dangling id is a type error instead of a runtime `Missing`.
4. **Saved-value policy.** A keeps every node value until the sweep pops it (6
   saved for case 4). B keeps only what each backward rule needs (2). The first
   autograd implementation can free values the backward pass never reads (`Add`
   and `Sum` operands) at record time. That needs a per-op "needs operand"
   table.
5. **Clone accounting.** Backward clones one cotangent per binary node. A
   borrow API on tensors (read a tensor without consuming it) would remove
   those clones and the take/put round trips. Can that API exist without unsafe
   aliasing?
6. **F32 laws.** The checker does not reduce F32 arithmetic on literals, so
   scalar laws use U32/Nat instances of the same templates. The autograd laws
   must decide whether the gradient spec is stated over an abstract semiring,
   as the templates do, or over F32 with uninterpreted ops.
7. **Kernel fuel.** If upstream lifts the fuel limit, concrete
   tensor laws could become `--verdict` evidence. Until then they are
   `--check-only` probes only, and the doc must not count them as kernel-checked.

## Reproduce

```sh
export BEND_NO_TELEMETRY=1 BENDTT="$PWD/.tools/kernel/bendtt" LEAN_STACK_SIZE_KB=4194304
export CC="$PWD/.tools/llvm/usr/bin/clang-19" LD_LIBRARY_PATH="$PWD/.tools/llvm/usr/lib/x86_64-linux-gnu"
B="bun .tools/bend/bend2/main.ts"
for p in tape pullback dual compare; do $B experiments/autograd-prototypes/$p-PROOF.bend --verdict; done
for f in tape-cases pullback-cases dual-cases tape-tensor-cases pullback-tensor-cases dual-tensor-cases; do
  $B experiments/autograd-prototypes/$f.bend -o "$EVIDENCE/$f" && "$EVIDENCE/$f" --threads 1 --gpu off
done
$B experiments/autograd-prototypes/bench.bend -o "$EVIDENCE/bench"
"$EVIDENCE/bench" --threads 1 --gpu off tape-tensor 20000
$B experiments/autograd-prototypes/kernel-fuel-adds.bend --check-only   # ALL PROOFS CHECK
$B experiments/autograd-prototypes/kernel-fuel-adds.bend --verdict      # exits 1: kernel out of fuel
```

The case binaries print F32 bit words, one per line. `bench` prints one
checksum word.
