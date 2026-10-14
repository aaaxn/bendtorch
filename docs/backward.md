# Scalar backward

`src/autograd/backward.bend` is the reverse-mode engine for F32 scalars. A
program records leaves and operations on a tape, then calls `backward` on the
output it selects. The engine is promoted from the reverse-tape prototype in
`experiments/autograd-prototypes/tape.bend` and fixes the prototype's known bugs: it seeds the
selected output instead of the newest node, and it never reads an unknown id
as another node. Tensors come later.

## Tape and ids

A `Tape` is a list of entries, newest first. Each entry holds a node, its
forward value and its gradient slot. The node is `Leaf`, `Add{a, b}`,
`Sub{a, b}`, `Mul{a, b, left, right}` or `Neg{a}`, where `a` and `b` are
operand ids and a product saves its operand values `left` and `right`.

An id is the number of entries recorded before the node, so the first leaf is
`0`. `Backward.leaf(tape, x)` returns the new tape and the id.
`Backward.add`, `sub`, `mul` and `neg` return the new tape and `Done{id}`.
They read the operands' values through the same lookup as `Backward.value`.

A lookup walks down from the newest entry and compares each entry's id with
the requested one. It never computes an offset by subtraction, so an id that
the tape does not hold reads as `UnknownNode` (error 103) and never as another
node. Recording with an unknown operand returns `Fail{UnknownNode}` and the
tape unchanged.

## backward

`Backward.backward(tape, output)` returns `Done{tape}` or a typed error:

1. If `output` is not an id on the tape, the result is `Fail{UnknownNode}`
   (103).
2. If an entry has an operand id that is not older than the entry itself (the
   entry itself, a newer entry or a missing one), the result is
   `Fail{DanglingOperand}` (104). The public recording functions cannot build
   such a tape; the check protects against a tape built some other way.
3. Otherwise the engine computes one gradient per entry and accumulates it
   into the entry's slot.

The gradients come from one sweep:

- The seed `d(output)/d(output) = 1.0` goes to the selected output. Entries
  newer than the output get nothing, even when they use it.
- The sweep visits the output and then every older entry, newest first. An
  entry that received a gradient `g` sends its vector-Jacobian product to its
  operands, the first operand before the second. An entry that received
  nothing sends nothing.
- The products are `Add`: `(g, g)`; `Sub`: `(g, -g)`; `Mul`:
  `(g * right, g * left)`; `Neg`: `-g`. Each is one F32 operation, and `-g`
  flips the sign bit.
- A node's first contribution moves in unchanged. Each later contribution is
  added with one F32 add, the held total first.

So a node used several times receives its contributions from its consumers in
decreasing id order. For `x` with consumers `x*a`, `x*b` and `x*c` (ids
increasing) and output `(x*a + x*b) + x*c`, the gradient of `x` is
`(c + b) + a`. PyTorch's engine fills its input buffers in the same order on
these graphs, and the oracle below checks it word for word.

After the sweep, each gradient is accumulated into its entry's slot with the
same rule. A slot that receives nothing keeps what it held.

**Repeated backward.** A second `backward` on the returned tape adds to the
slots, like PyTorch's `backward(retain_graph=True)`. The gradients of one
backward depend only on the nodes, never on the slots, so calling
`backward(output)` twice gives each reached slot `g + g`. bendtorch has no
graph freeing, so there is no counterpart of PyTorch's error for a second
backward without `retain_graph`.

**Reads.** `Backward.value(tape, id)` is the forward value.
`Backward.grad(tape, id)` is `Done{None}` for a node that no backward reached,
like PyTorch's `.grad`, and `Done{Some{g}}` otherwise. Both give
`Fail{UnknownNode}` for an unknown id. `Backward.gradients(tape, output)`
returns the gradients of one backward, newest first, without touching the
slots. `Backward.zero_grad(tape)` empties every slot, like PyTorch's
`zero_grad(set_to_none=True)`, so the next contribution moves in unchanged and
a `-0.0` stays `-0.0`.

The sweep is sequential, because the order in which contributions reach a slot
is part of the contract.

## Laws

`src/autograd/backward-LAWS.bend` states 35 laws, and `backward-PROOF.bend`
proves them. The structural laws quantify over every tape, every list of
nodes or slots and every id. The laws about one recorded program quantify
over every F32 input value. F32 arithmetic is opaque to the laws: a law says
which `F32.add`, `F32.mul` or `F32.neg` term each gradient is, not its rounded
value.

- **VJPs:** `vjp_leaf`, `vjp_add`, `vjp_sub`, `vjp_mul` and `vjp_neg` state
  the contribution list of each node for every operand id and gradient.
  `backward_add`, `backward_sub`, `backward_mul` and `backward_neg` record
  `x op y` (or `-x`) through the public API for every `x` and `y`, and state
  the forward value and the gradient of every node.
- **Seed (C1):** `seed_selected_output`: for any nodes `newer`, any node and
  any older nodes `rest`, the gradients are `None` for each newer node, `1.0`
  for the selected node, and the sweep of its VJP over `rest`. The newer nodes
  do not appear on the right-hand side.
  `unused_leaf_none` records `x, y, z, x*y, z+z` and states that backward of
  `x*y` gives `z` and `z+z` the slot `None`.
- **Reachable graph only:** `reached_exactly`: for every graph and output, a
  node has a gradient exactly when the output reaches it, where reachability
  is defined separately in the laws file by marking operands of reached nodes.
- **Fan-out order:** `accumulate_moves_in`, `accumulate_adds_total_first`,
  `accumulate_keeps_without_contribution`, `bump_target` (a contribution to
  node `k` changes exactly slot `k`, for every list of slots) and
  `fan_out_arrival_order` (the three-consumer program above gives
  `(c + b) + a`, for every `x, a, b, c`).
- **Ids (C2):** `lookup_exact` (id `k` reads the element with `k` older
  elements), `lookup_unknown` and `bump_unknown` (an id not below the length
  reads `None` or changes nothing), and `value_unknown`, `grad_unknown`,
  `record_unknown_left`, `record_unknown_right`, `neg_unknown` and
  `backward_unknown`, which state the exact error `UnknownNode` and, for
  recording, the unchanged tape. `backward_dangling` states
  `DanglingOperand` for a tape that fails the wiring check, and
  `backward_success` states the result otherwise.
- **Wiring:** `wired_matches_operand_bounds` proves that the engine's wiring
  check equals a separate specification in the laws file: every operand id of
  a node is below the number of older nodes. `backward_rejects_unbound_operand`
  states `DanglingOperand` for any tape with an entry whose operand is that
  entry, a newer entry or a missing one, at any position and for any output
  the tape holds. `backward_accepts_bound_operands` states that a tape meeting
  the specification is accepted.
- **Repeated backward:** `backward_twice_accumulates` (for every tape and
  output, a second backward accumulates the same gradients again) and
  `backward_twice_mul` (both slots of `x*y` read `g + g`).
- **zero_grad:** `zero_grad_empties` and `zero_grad_keeps_nodes`.

The S3 gate also applies nine proof controls and two error controls to a copy
of the sources, and the proofs reject each one: seeding the newest node, a
lookup or a slot update that matches a larger id, dropping a later
contribution, swapping the `Mul` VJP operands, dropping the `Sub` sign,
overwriting a slot instead of adding, a wiring check that skips the operand
bounds (rejected at `wired_matches_operand_bounds`), a `zero_grad` that keeps
the slot, and each error code reported as the other.

## Runtime lanes

`tools/autograd_backward.py` runs 91 scripts on native Bend CPU
(`--threads 1 --gpu off`) and on the generated JavaScript. Each script records
leaves and operations, runs `backward` on chosen outputs, calls `zero_grad`,
and reads the value and `grad` of every node. The harness
(`src/autograd/backward-harness.bend`) calls the public functions and prints
every id, error and word. A `Forge` step pushes an entry without checks, so
the gate can build tapes with dangling operands.

- 43 handwritten scripts: the unused-node graph with each output selected, the
  four ops and their VJPs, `x*x`, `x-x` and `x+x`, the fan-out program with
  inputs whose order changes the result, a diamond, an interior output,
  repeated backward on one and on two outputs, `zero_grad` with a `-0.0`
  contribution, NaN, infinities, overflow, subnormals, unknown operands and
  outputs, and four forged tapes: three with an operand that is the node
  itself, a newer node or a missing node, and one with sound wiring but a
  forward value and saved operands that disagree with its leaves.
- 8 scripts where two NaNs with different payloads meet: `add`, `mul` and
  `sub` of two quiet NaNs, a quiet and a signaling NaN, a NaN gradient
  through `mul`, a NaN fan-out, and repeated backward through a non-leaf and
  at a leaf.
- 48 generated scripts (fixed seed): 2 to 5 leaves from a pool that includes
  `±0.0`, `1e8` and `0.1`, 3 to 14 random operations, one or two backward
  calls on one of the four newest nodes, and sometimes a `zero_grad` and a
  third backward.

Both lanes must print exactly the lines of a Python reference that applies the
same rules in the same order and rounds to F32 after each operation. The
native lane must match bit for bit, NaN payloads included. When an operand is
NaN, the reference does not use Python arithmetic. It implements the payload
rule of the pinned native lane (clang 19 `-O3` on x86-64): the result is a NaN
operand, quieted. With two NaN operands, `add` and `mul` keep the second
operand's NaN and `sub` keeps the first's. The rule was measured on this
toolchain; it is a property of the compiled code, not of IEEE 754, and the
same C at `-O0` picks differently. The JavaScript lane may return a NaN with
another payload, as in the other S3 subgates.

The subgate also builds four runtime controls from a mutated copy of
`backward.bend` and requires the JavaScript lane to reject each one: a seed at
the newest node (C1), a lookup that aliases an unknown id to the newest node
(C2), a slot that keeps its first contribution and drops the later ones, and a
`Mul` VJP with swapped operands.

## PyTorch record

`oracle/autograd/backward.py` replays every script with PyTorch `2.14.1+cpu` scalars
(`OMP_NUM_THREADS=1`, `retain_graph=True`, `retain_grad()` on every non-leaf)
and writes `oracle/autograd/fixtures/s3-backward-torch-v1.json`. The root test
`tests/test_autograd_backward.py` compares it with the Python reference. Bend never
calls PyTorch.

83 scripts agree with PyTorch line for line: every id, value, gradient word
and `None`, including NaN payloads, infinities, `-0.0` and subnormals. The
nonfinite cases come from PyTorch, not from forward duals: at `x = inf`,
`y = 0`, `x * y` is NaN, but reverse mode gives `dx = 0` and `dy = inf`, where
dual numbers would give NaN. The other 8 differ as listed:

| Scripts | bendtorch | PyTorch |
| --- | --- | --- |
| `unknown_operand`, `unknown_output`, `unknown_output_empty` | `UnknownNode` (103) for each unknown id | `IndexError` from the Python list that the oracle uses as a tape; every other line agrees |
| `dangling_self`, `dangling_newer`, `dangling_mul` | `DanglingOperand` (104) | not representable: PyTorch builds its graph from tensors, so no node can name itself, a newer node or a missing node |
| `forged_sound` | gradients from the forged saved operands | not representable: PyTorch saves the operands it was given, so saved values cannot disagree with the leaves |
| `nan_payloads_sub` | `NaN_a - NaN_b` keeps `NaN_a`'s payload | keeps `NaN_b`'s payload; every other line agrees, including `add` and `mul` of the same NaNs |

## Reproduce

```sh
bend src/autograd/backward-PROOF.bend --verdict
uv run --locked python -m tools.autograd_backward --output .evidence/<new directory>
uv run --locked python -m unittest tests.test_s3_backward -v
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 uv run --project oracle --frozen --no-sync python oracle/autograd/backward.py --check
```
