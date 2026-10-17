"""Tape scripts large enough that every parallel build the tensor backward
calls forks: elementwise ops above 2^15 cells, products, `torch.matmul` with
a batch, the transpose of a matmul's backward, `reduce_to` and the row
maximum of `cross_entropy`. The gate runs them natively at each thread count
of the tensor probe, requires the same bytes at every count and every word
the reference computes, and plants a bug in each build's fork to show that
the scripts take the forked path.
"""

import argparse
import json
import math
import shutil
from functools import partial
from pathlib import Path

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
"""


def leaf(shape: list[int], step: int) -> dict:
    count = math.prod(shape)
    return {"op": "leaf", "shape": shape, "words": spread(count, step), "spread": step}


# Each fork of a parallel build hands both halves a copy of its input state,
# through the build's split function, which runs only when the build forks.
# Each control makes one split write 2^20 into word 0 of the copies, so it
# changes an output only where that build forked.
WRONG_WORD = "1048576.0"
CONTROLS = {
    # Elementwise ops, matmul, matmul_nt, matmul_tn, torch.matmul and slicing.
    "elementwise-fork-copies-a-wrong-word": (
        "src/tensor/elementwise.bend",
        (
            (
                "  halves(Array.clone(F32, la), Array.clone(F32, ra))",
                f"  halves(Array.clone(F32, Array.set(F32, la, 0, {WRONG_WORD})), "
                "Array.clone(F32, ra))",
            ),
        ),
    ),
    # reduce_to, so sum_to_size and sum over axes.
    "reduce-fork-copies-a-wrong-word": (
        "src/autograd/reduce.bend",
        (
            (
                "  halves(Array.clone(F32, array))",
                f"  halves(Array.clone(F32, Array.set(F32, array, 0, {WRONG_WORD})))",
            ),
        ),
    ),
    # permute and transpose.
    "permute-fork-copies-a-wrong-word": (
        "src/tensor/permute-build.bend",
        (
            (
                "  halves(Array.clone(F32, a), Array.clone(F32, b))",
                f"  halves(Array.clone(F32, Array.set(F32, a, 0, {WRONG_WORD})), "
                "Array.clone(F32, b))",
            ),
        ),
    ),
    # slice, narrow and select, and slice_backward's scatter, borrow the
    # elementwise split, so their control gives them a split of their own.
    "slice-fork-copies-a-wrong-word": (
        "src/tensor/slice.bend",
        (
            (
                "def build(",
                "def slice_split(inputs: Array<F32> & Array<F32>) -> "
                "(Array<F32> & Array<F32>) & (Array<F32> & Array<F32>):\n"
                "  (la, ra) = inputs\n"
                "  Elementwise.halves(Array.clone(F32, "
                f"Array.set(F32, la, 0, {WRONG_WORD})), Array.clone(F32, ra))\n\n"
                "def build(",
            ),
            *(
                (
                    f"  Parallel.{fold}(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, "
                    f"~Elementwise.split, ~{cell},",
                    f"  Parallel.{fold}(~Plan, ~Plan, ~Array<F32>, ~Array<F32>, ~same, "
                    f"~slice_split, ~{cell},",
                )
                for fold in ("build", "balanced")
                for cell in ("cell", "scatter_cell")
            ),
        ),
    ),
    # The row maximum of cross_entropy borrows reduce_to's split, so its
    # control gives it a split of its own; the balance proof takes any split.
    "row-max-fork-copies-a-wrong-word": (
        "src/autograd/loss.bend",
        (
            (
                "def max_build(",
                "def max_split(state: Array<F32> & Unit) -> "
                "(Array<F32> & Unit) & (Array<F32> & Unit):\n"
                "  (array, none) = state\n"
                "  Reduce.halves(Array.clone(F32, "
                f"Array.set(F32, array, 0, {WRONG_WORD})))\n\ndef max_build(",
            ),
            (
                "  Parallel.build(~Width, ~Width, ~Array<F32>, ~Unit, ~width, "
                "~Reduce.split, ~max_cell,",
                "  Parallel.build(~Width, ~Width, ~Array<F32>, ~Unit, ~width, "
                "~max_split, ~max_cell,",
            ),
            (
                "  Parallel.balanced(~Width, ~Width, ~Array<F32>, ~Unit, ~width, "
                "~Reduce.split, ~max_cell,",
                "  Parallel.balanced(~Width, ~Width, ~Array<F32>, ~Unit, ~width, "
                "~max_split, ~max_cell,",
            ),
        ),
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
            "id": "parallel_linear",
            "forks": ("elementwise-fork-copies-a-wrong-word",),
            "steps": [
                leaf([130, 32], 7),
                leaf([256, 32], 11),
                leaf([256], 13),
                {"op": "linear", "a": 0, "b": 1, "bias": 2},
                {"op": "sum", "a": 3},
                {"op": "backward", "output": 4},
                *read(0, 1, 2, 3),
            ],
        },
        {
            "id": "parallel_matmul_transpose",
            "forks": (
                "elementwise-fork-copies-a-wrong-word",
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
            "forks": ("elementwise-fork-copies-a-wrong-word",),
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


def render(selected: list[dict], directory: Path, source_root: Path) -> str:
    """The tensor backward corpus program, with `spread` after its imports."""
    imports, _, rest = program(selected, directory, source_root).partition("\n\n")
    return f"{imports}\n\n{SPREAD}\n{rest}"


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
    first = same_bytes(
        {
            threads: evidence.run(
                f"parallel-threads-{threads}",
                common.native(executable, threads=threads),
                timeout=1800,
            ).stdout
            for threads in THREAD_COUNTS
        }
    )
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
    changes every script whose `forks` name it."""
    path, edits = CONTROLS[name]
    copy = evidence.directory / "controls" / name
    shutil.copytree(ROOT / "src", copy / "src")
    mutate(copy / path, edits, name)
    source = copy / "parallel.bend"
    source.write_text(render(selected, copy, copy))
    result = evidence.run(
        f"{name}-native-run", common.native(build(evidence, name, source)), timeout=1800
    )
    observed = parse(result.stdout, [case["id"] for case in selected])
    changed = [
        row["id"]
        for row in observed
        if len(row["lines"]) != len(expected[row["id"]])
        or not all(map(same_row, row["lines"], expected[row["id"]]))
    ]
    missed = [
        case["id"]
        for case in selected
        if name in case["forks"] and case["id"] not in changed
    ]
    if missed:
        raise GateError(f"{name}: the forked build never ran in {missed}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
