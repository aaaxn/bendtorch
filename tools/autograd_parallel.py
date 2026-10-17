"""Tape scripts large enough that every parallel build the tensor backward
calls forks: elementwise ops above 2^15 cells, matmul, matmul_nt and
matmul_tn, `torch.matmul` with a batch, the transpose of a matmul's
backward, `reduce_to`, the row maximum of `cross_entropy`, slicing with its
scatter, and index_select and gather with theirs. The gate runs them
natively at each thread count of the tensor probe, requires that the runtime
reports a pool of that many workers, the same bytes at every count and every
word the reference computes, and plants a bug in each build's fork to show
that the scripts take the forked path.
"""

import argparse
import json
import math
import re
import shutil
from functools import partial
from pathlib import Path
from typing import NamedTuple

from tools import common
from tools.autograd_tensor_backward import parse, program, reference, same_row, verify
from tools.common import ROOT, Evidence, GateError, bend, parallel
from tools.tensor_parallel import THREAD_COUNTS


# Word `index` of a spread leaf: a finite F32 in [2^-5, 2^3) of either sign
# whose 23 mantissa bits vary, so sums and products round. `SPREAD` computes
# the same words in Bend, so the program holds no literal list.
def spread_word(index: int, step: int) -> int:
    h = (index + 1) * step % 1000003
    sign = h // 8 % 2
    exponent = 122 + h % 8
    return sign * 2**31 + exponent * 2**23 + h * 8191 % 2**23


def spread(count: int, step: int) -> list[int]:
    return [spread_word(index, step) for index in range(count)]


SPREAD = """def spread_word(+index: Nat, +step: Nat) -> U32:
  +h = Nat.mod(Nat.mul(1n+index, step), 1000003n)
  +sign = Nat.mul(Nat.mod(Nat.div(h, 8n), 2n), 2147483648n)
  +exponent = Nat.mul(Nat.add(122n, Nat.mod(h, 8n)), 8388608n)
  U32.from_nat(Nat.add(sign, Nat.add(exponent, Nat.mod(Nat.mul(h, 8191n), 8388608n))))

def spread_go(fuel: Nat, +index: Nat, +step: Nat, acc: List<&2, U32>) -> List<&2, U32>:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      spread_go(rest, Nat.sub(index, 1n), step, spread_word(index, step) <> acc)

def spread(+count: Nat, +step: Nat) -> List<&2, U32>:
  spread_go(count, Nat.sub(count, 1n), step, Nil{})

def cycle_word(+index: Nat, +step: Nat, +modulus: Nat) -> U32:
  U32.from_nat(Nat.mod(Nat.mul(index, step), modulus))

def cycle_go(fuel: Nat, +index: Nat, +step: Nat, +modulus: Nat,
  acc: List<&2, U32>) -> List<&2, U32>:
  match fuel:
    case 0n:
      acc
    case 1n+rest:
      +word = cycle_word(index, step, modulus)
      cycle_go(rest, Nat.sub(index, 1n), step, modulus, word <> acc)

def cycle(+count: Nat, +step: Nat, +modulus: Nat) -> List<&2, U32>:
  cycle_go(count, Nat.sub(count, 1n), step, modulus, Nil{})
"""


# Entry `index` of a cycled gather index: index * step mod modulus. `cycle`
# computes the same entries in Bend.
def cycle(count: int, step: int, modulus: int) -> list[int]:
    return [index * step % modulus for index in range(count)]


def leaf(shape: list[int], step: int) -> dict:
    count = math.prod(shape)
    return {"op": "leaf", "shape": shape, "words": spread(count, step), "spread": step}


# Each fork of a parallel build hands both halves a copy of its input state,
# through the split function the build passes, which runs only when the build
# forks. Builds share split functions (`Elementwise.split` serves six), so
# each control gives one build a split of its own that writes 2^20 into word
# 0 of the copies, at that build's calls and nowhere else: it changes an
# output only where that build forked. Its unforked twin then passes 0n
# splits at the same calls, so the poisoned split never runs and no output
# may change; that is the gate seeing a build that stopped forking.
WRONG_WORD = "1048576.0"


class Control(NamedTuple):
    path: str
    poison: tuple[tuple[str, str], ...]
    # Applied after `poison`.
    unforked: tuple[tuple[str, str], ...]


def pair_split(halves: str) -> str:
    return (
        "def wrong_split(inputs: Array<F32> & Array<F32>) -> "
        "(Array<F32> & Array<F32>) & (Array<F32> & Array<F32>):\n"
        "  (la, ra) = inputs\n"
        f"  {halves}(Array.clone(F32, Array.set(F32, la, 0, {WRONG_WORD})), "
        "Array.clone(F32, ra))\n\n"
    )


def unit_split(halves: str) -> str:
    return (
        "def wrong_split(state: Array<F32> & Unit) -> "
        "(Array<F32> & Unit) & (Array<F32> & Unit):\n"
        "  (array, none) = state\n"
        f"  {halves}(Array.clone(F32, Array.set(F32, array, 0, {WRONG_WORD})))\n\n"
    )


def index_split(halves: str) -> str:
    return (
        "def wrong_split(state: Array<F32> & Array<U32>) -> "
        "(Array<F32> & Array<U32>) & (Array<F32> & Array<U32>):\n"
        "  (array, index) = state\n"
        f"  {halves}(Array.clone(F32, Array.set(F32, array, 0, {WRONG_WORD})), "
        "Array.clone(U32, index))\n\n"
    )


def own_split(
    path: str,
    before: str,
    definition: str,
    calls: tuple[str, ...],
    split: str,
    unforked: tuple[tuple[str, str], ...] | None = None,
) -> Control:
    """`definition` goes before `before`, and each call names `wrong_split`
    where `split` names its split. By default the unforked twin passes 0n
    where each call passes `depth, splits`."""
    wrong = split.replace(split.strip("~,("), "wrong_split")
    poisoned = tuple(call.replace(split, wrong) for call in calls)
    if unforked is None:
        unforked = tuple(
            (call, re.sub(r", depth, splits(\(depth\))?,", ", depth, 0n,", call))
            for call in poisoned
        )
    return Control(
        path,
        ((before, definition + before), *zip(calls, poisoned, strict=True)),
        unforked,
    )


def folds(head: str) -> tuple[str, ...]:
    """A build's two calls, `Parallel.build` and `Parallel.balanced`, up to
    their splits."""
    return tuple(f"  Parallel.{fold}({head}" for fold in ("build", "balanced"))


CONTROLS = {
    # add, sub, mul, relu and the other elementwise ops.
    "elementwise-fork-copies-a-wrong-word": own_split(
        "src/tensor/elementwise.bend",
        "def build(",
        pair_split("halves"),
        folds(
            "~Request, ~Walk, ~Array<F32>, ~Array<F32>, ~walk, ~split, ~cell_at, "
            "Request{operation, plan}, depth, splits,"
        ),
        "~split,",
    ),
    # matmul, the cell by cell product.
    "matmul-fork-copies-a-wrong-word": own_split(
        "src/tensor/matmul.bend",
        "def build(",
        pair_split("Elementwise.halves"),
        folds(
            "~Dims, ~Dims, ~Array<F32>, ~Array<F32>, ~dims, ~Elementwise.split, "
            "~cell_at, Dims{inner, columns}, depth, splits,"
        ),
        "~Elementwise.split,",
    ),
    # matmul_nt and matmul_tn, whose product build forks through
    # `Parallel.fork` itself, with the splits its one caller passes.
    "product-fork-copies-a-wrong-word": own_split(
        "src/tensor/matmul.bend",
        "def product_build(",
        pair_split("Elementwise.halves"),
        (
            "      Parallel.fork(~Array<F32>, ~Array<F32>, "
            "Elementwise.split((la, ra)),",
            "      Parallel.fork_balanced(~Array<F32>, ~Array<F32>, lower, "
            "Elementwise.split((la, ra)),",
        ),
        "Elementwise.split(",
        tuple(
            (
                f"    {name}(depth, product_splits(depth, inner),",
                f"    {name}(depth, 0n,",
            )
            for name in ("product_build", "product_build_balanced")
        ),
    ),
    # torch.matmul with a batch.
    "batched-matmul-fork-copies-a-wrong-word": own_split(
        "src/tensor/batched-matmul.bend",
        "def build(",
        pair_split("Elementwise.halves"),
        folds(
            "~Layout, ~Layout, ~Array<F32>, ~Array<F32>, ~layout, ~Elementwise.split, "
            "~cell_at, given, depth, splits,"
        ),
        "~Elementwise.split,",
    ),
    # permute and transpose.
    "permute-fork-copies-a-wrong-word": own_split(
        "src/tensor/permute-build.bend",
        "def build(",
        pair_split("halves"),
        folds(
            "~Request, ~Recipe, ~Array<F32>, ~Array<F32>, ~recipe, ~split, ~cell, "
            "request, depth, splits(depth),"
        ),
        "~split,",
    ),
    # reduce_to, so sum_to_size and sum over axes.
    "reduce-fork-copies-a-wrong-word": own_split(
        "src/autograd/reduce.bend",
        "def build(",
        unit_split("halves"),
        folds(
            "~Fibers, ~Fibers, ~Array<F32>, ~Unit, ~fibers, ~split, ~cell_at, "
            "Fibers{rows, extents}, depth, splits,"
        ),
        "~split,",
    ),
    # The row maximum of cross_entropy.
    "row-max-fork-copies-a-wrong-word": own_split(
        "src/autograd/loss.bend",
        "def max_build(",
        unit_split("Reduce.halves"),
        folds(
            "~Width, ~Width, ~Array<F32>, ~Unit, ~width, ~Reduce.split, ~max_cell, "
            "Width{columns}, depth, splits,"
        ),
        "~Reduce.split,",
    ),
    # slice, narrow and select.
    "slice-fork-copies-a-wrong-word": own_split(
        "src/tensor/slice.bend",
        "def build(",
        pair_split("Elementwise.halves"),
        folds(
            "~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, "
            "~cell, plan, depth, splits,"
        ),
        "~Elementwise.split,",
    ),
    # slice_backward's scatter.
    "slice-backward-fork-copies-a-wrong-word": own_split(
        "src/tensor/slice.bend",
        "def scatter_build(",
        pair_split("Elementwise.halves"),
        folds(
            "~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, ~Elementwise.split, "
            "~scatter_cell, plan, depth, splits,"
        ),
        "~Elementwise.split,",
    ),
    # index_select and gather, whose state is the input and the index array.
    "gather-fork-copies-a-wrong-word": own_split(
        "src/tensor/gather.bend",
        "def build(",
        index_split("halves"),
        folds(
            "~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~cell, "
            "plan, depth, splits,"
        ),
        "~split,",
    ),
    # Their backward, index_add and scatter_add.
    "gather-backward-fork-copies-a-wrong-word": own_split(
        "src/tensor/gather.bend",
        "def scatter_build(",
        index_split("halves"),
        folds(
            "~Plan, ~Plan, ~Array<F32>, ~Array<U32>, ~same, ~split, ~scatter_cell, "
            "plan, depth, splits,"
        ),
        "~split,",
    ),
}


def read(*identifiers: int) -> list[dict]:
    return [{"op": "read", "id": identifier} for identifier in identifiers]


# Each script, with the controls that must change its output: the builds it
# forks. [256, 160] holds 40960 cells, above 2^15, so an elementwise op on it
# forks one level and a row of 160 terms forks a reduction two.
def cases() -> list[dict]:
    lr = 0x3DCCCCCD  # 0.1
    return [
        {
            "id": "parallel_mul_relu_step",
            "forks": ("elementwise-fork-copies-a-wrong-word",),
            "steps": [
                leaf([256, 160], 7),
                leaf([160], 11),
                {"op": "mul", "a": 0, "b": 1},
                {"op": "relu", "a": 2},
                {"op": "sum", "a": 3},
                {"op": "backward", "output": 4},
                {"op": "step", "lr": lr, "params": [0, 1]},
                *read(0, 1, 3),
            ],
        },
        {
            # linear records two entries: the product, 3, and the biased
            # output, 4.
            "id": "parallel_linear",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "matmul-fork-copies-a-wrong-word",
                "product-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([130, 32], 7),
                leaf([256, 32], 11),
                leaf([256], 13),
                {"op": "linear", "a": 0, "b": 1, "bias": 2},
                {"op": "sum", "a": 4},
                {"op": "backward", "output": 5},
                *read(0, 1, 2, 3, 4),
            ],
        },
        {
            "id": "parallel_matmul_transpose",
            "forks": (
                "matmul-fork-copies-a-wrong-word",
                "permute-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([8, 520], 7),
                leaf([520, 64], 11),
                {"op": "matmul", "a": 0, "b": 1},
                {"op": "sum", "a": 2},
                {"op": "backward", "output": 3},
                *read(0, 1, 2),
            ],
        },
        {
            "id": "parallel_batched_matmul",
            "forks": ("batched-matmul-fork-copies-a-wrong-word",),
            "steps": [
                leaf([8, 32, 48], 7),
                leaf([48, 64], 11),
                {"op": "batched_matmul", "a": 0, "b": 1},
                {"op": "sum", "a": 2},
                {"op": "backward", "output": 3},
                *read(0, 1, 2),
            ],
        },
        {
            "id": "parallel_sum_dims",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "reduce-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([256, 160], 7),
                {"op": "sum_dims", "a": 0, "dims": [1], "keepdim": False},
                {"op": "sum", "a": 1},
                {"op": "backward", "output": 2},
                *read(0, 1),
            ],
        },
        {
            # [512, 160] sliced to every other column: the slice build holds
            # 40960 cells and slice_backward's scatter 81920.
            "id": "parallel_slice",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "slice-fork-copies-a-wrong-word",
                "slice-backward-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([512, 160], 7),
                {"op": "slice", "a": 0, "dim": 1, "start": 0, "stop": 160, "step": 2},
                {"op": "sum", "a": 1},
                {"op": "backward", "output": 2},
                *read(0, 1),
            ],
        },
        {
            # [256, 160] indexed along its rows by 300 entries, each row at
            # least once and some twice: the index_select build holds 48000
            # cells and its backward 40960, each a sum of one or two words.
            "id": "parallel_index_select",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "gather-fork-copies-a-wrong-word",
                "gather-backward-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([256, 160], 7),
                {
                    "op": "index_select",
                    "a": 0,
                    "dim": 0,
                    "index": [k * 77 % 256 for k in range(300)],
                },
                {"op": "sum", "a": 1},
                {"op": "backward", "output": 2},
                *read(0, 1),
            ],
        },
        {
            # [256, 160] gathered along its columns by an index of its own
            # shape, each entry of a row twice: both builds hold 40960
            # cells.
            "id": "parallel_gather",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "gather-fork-copies-a-wrong-word",
                "gather-backward-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([256, 160], 11),
                {
                    "op": "gather",
                    "a": 0,
                    "dim": -1,
                    "index_shape": [256, 160],
                    "index": cycle(40960, 38, 160),
                    "cycle": (38, 160),
                },
                {"op": "sum", "a": 1},
                {"op": "backward", "output": 2},
                *read(0, 1),
            ],
        },
        {
            "id": "parallel_cross_entropy",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
                "reduce-fork-copies-a-wrong-word",
                "row-max-fork-copies-a-wrong-word",
            ),
            "steps": [
                leaf([256, 160], 7),
                {
                    "op": "cross_entropy",
                    "a": 0,
                    "targets": [row * 37 % 160 for row in range(256)],
                },
                {"op": "backward", "output": 1},
                *read(0, 1),
            ],
        },
    ]


MAIN = "def main() -> IO(Unit):\n  do IO<Unit>:\n"
# The runtime's worker pool, on stderr, apart from the bytes compared.
POOL = (
    "    pool : U32 <- IO.thread_count()\n"
    '    IO.print_err("autograd:threads:" ++ U32.show(pool))\n'
)


def render(selected: list[dict], directory: Path, source_root: Path) -> str:
    """The tensor backward corpus program, with `spread` after its imports,
    and a main that first reports the worker pool."""
    imports, _, rest = program(selected, directory, source_root).partition("\n\n")
    if rest.count(MAIN) != 1:
        raise GateError("Autograd parallel program has no single main")
    return f"{imports}\n\n{SPREAD}\n{rest.replace(MAIN, MAIN + POOL)}"


def pool(threads: int, stderr: str) -> None:
    """A GateError unless the run reported a pool of `threads` workers: a
    thread count that never reached the runtime would otherwise pass, as
    the output does not depend on it."""
    reported = [
        line.removeprefix("autograd:threads:")
        for line in stderr.splitlines()
        if line.startswith("autograd:threads:")
    ]
    if reported != [str(threads)]:
        raise GateError(
            f"Autograd parallel run at {threads} threads reported a pool of {reported}"
        )


def same_bytes(outputs: dict[int, str]) -> str:
    """The output every thread count printed; a GateError if two differ."""
    first = next(iter(outputs.values()))
    for threads, stdout in outputs.items():
        if stdout != first:
            raise GateError(f"Autograd parallel output at {threads} threads differs")
    return first


def mutate(source: Path, edits: tuple[tuple[str, str], ...], name: str) -> None:
    text = source.read_text()
    for original, mutated in edits:
        if text.count(original) != 1:
            raise GateError(f"{name}: control anchor is missing or ambiguous")
        text = text.replace(original, mutated)
    source.write_text(text)


def build(evidence: Evidence, name: str, source: Path) -> Path:
    emitted = source.with_suffix(".c")
    executable = source.with_suffix("")
    evidence.run(f"{name}-c-export", bend(evidence.tool_root, source, "-o", emitted))
    evidence.run(
        f"{name}-native-build",
        [
            str(evidence.tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(emitted),
            "-lpthread",
            "-lm",
            "-o",
            str(executable),
        ],
    )
    return executable


def run_gate(tool_root: Path, output: Path) -> dict:
    tool_root, output = tool_root.resolve(), output.resolve()
    selected = cases()
    identifiers = [case["id"] for case in selected]
    evidence = Evidence(output, tool_root)
    shutil.copytree(ROOT / "src", output / "source/src")
    source = output / "parallel.bend"
    source.write_text(render(selected, output, output / "source"))
    evidence.run(
        "parallel-check",
        bend(tool_root, source, "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    executable = build(evidence, "parallel", source)
    runs = {
        threads: evidence.run(
            f"parallel-threads-{threads}",
            common.native(executable, threads=threads),
            timeout=1800,
        )
        for threads in THREAD_COUNTS
    }
    for threads, result in runs.items():
        pool(threads, result.stderr)
    first = same_bytes({threads: result.stdout for threads, result in runs.items()})
    summary = verify(selected, parse(first, identifiers), "native", native=True)
    expected = {case["id"]: reference(case) for case in selected}
    rejected = parallel(
        *(partial(control, evidence, selected, expected, name) for name in CONTROLS)
    )
    evidence.save()
    return {
        "thread_counts": list(THREAD_COUNTS),
        **summary,
        "controls": dict(zip(CONTROLS, rejected, strict=True)),
    }


def control(
    evidence: Evidence, selected: list[dict], expected: dict[str, list[str]], name: str
) -> list[str]:
    """The scripts whose output the control changes, after checking that it
    changes every script whose `forks` name it, and that its unforked twin
    changes none."""
    path, poison, unforked = CONTROLS[name]
    changed = changes(evidence, selected, expected, name, path, poison)
    fired(name, selected, changed)
    still = changes(
        evidence, selected, expected, f"{name}-unforked", path, poison + unforked
    )
    # Nothing forks the poisoned split now, so `fired` would fail on every
    # script that forks the build: the gate seeing that build stop forking.
    if still:
        raise GateError(f"{name}: the control still changes {still} at 0n splits")
    return changed


def fired(name: str, selected: list[dict], changed: list[str]) -> None:
    """A GateError unless the control changed every script that forks its
    build."""
    missed = [
        case["id"]
        for case in selected
        if name in case["forks"] and case["id"] not in changed
    ]
    if missed:
        raise GateError(f"{name}: the forked build never ran in {missed}")


def changes(
    evidence: Evidence,
    selected: list[dict],
    expected: dict[str, list[str]],
    name: str,
    path: str,
    edits: tuple[tuple[str, str], ...],
) -> list[str]:
    """The scripts whose output a copy of the sources with `edits` changes."""
    copy = evidence.directory / "controls" / name
    shutil.copytree(ROOT / "src", copy / "src")
    mutate(copy / path, edits, name)
    source = copy / "parallel.bend"
    source.write_text(render(selected, copy, copy))
    result = evidence.run(
        f"{name}-native-run", common.native(build(evidence, name, source)), timeout=1800
    )
    observed = parse(result.stdout, [case["id"] for case in selected])
    return [
        row["id"]
        for row in observed
        if len(row["lines"]) != len(expected[row["id"]])
        or not all(map(same_row, row["lines"], expected[row["id"]]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
