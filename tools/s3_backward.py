import argparse
import json
import os
import shutil
from pathlib import Path

from tools.common import ROOT, Evidence, GateError, bend
from tools.s3_broadcast import is_nan, run_lanes, word
from tools.s3_reduce import bits, f32, value

UNKNOWN_NODE = 103
DANGLING_OPERAND = 104
ONE = 0x3F800000
SIGN = 0x80000000
NEGATIVE_ZERO = 0x80000000
INFINITY = 0x7F800000
QUIET_NAN = 0x7FC01234
OTHER_NAN = 0x7FC05678
THIRD_NAN = 0x7FC0ABCD
SIGNALING_NAN = 0x7FA00001
QUIET = 0x00400000
BINARY = ("add", "sub", "mul")

# Each mutant breaks one contract on a copy of backward.bend; the JavaScript
# lane must reject every one of them.
RUNTIME_CONTROLS = {
    "seed-newest-node": (
        "unit => pass(rest, below, output), Nat.is_eq(below, output))",
        "unit => pass(rest, below, output), True{})",
    ),
    "lookup-aliases-newest": (
        "picked(A, head, unit => find(A, rest, below, id), Nat.is_eq(below, id))",
        "picked(A, head, unit => find(A, rest, below, id), Nat.is_le(below, id))",
    ),
    "fan-out-drops-later-contribution": (
        "      accumulate(slot, Some{contribution}) <> rest",
        "      Maybe.or(&2, F32, slot, Some{contribution}) <> rest",
    ),
    "mul-vjp-swaps-operands": (
        "[Contribution{a, F32.mul(g, right)}, Contribution{b, F32.mul(g, left)}]",
        "[Contribution{a, F32.mul(g, left)}, Contribution{b, F32.mul(g, right)}]",
    ),
}


def leaf(number: float | int) -> dict:
    return {"op": "leaf", "word": number if isinstance(number, int) else word(number)}


def op(name: str, a: int, b: int | None = None) -> dict:
    return {"op": name, "a": a} if b is None else {"op": name, "a": a, "b": b}


def run(output: int) -> dict:
    return {"op": "backward", "output": output}


def reads(count: int) -> list[dict]:
    return [{"op": "read", "id": index} for index in range(count)]


def forge(node: dict, number: float) -> dict:
    return {"op": "forge", "node": node, "word": word(number)}


ZERO_GRAD = {"op": "zero_grad"}


def binary_case(identifier: str, name: str, x: float, y: float) -> dict:
    return {
        "id": identifier,
        "steps": [leaf(x), leaf(y), op(name, 0, 1), run(2), *reads(3)],
    }


def fan_out(identifier: str, x: float, a: float, b: float, c: float) -> dict:
    # x = 0 has three consumers p = x*a, q = x*b and r = x*c; t = (p + q) + r.
    return {
        "id": identifier,
        "steps": [
            leaf(x),
            leaf(a),
            leaf(b),
            leaf(c),
            op("mul", 0, 1),
            op("mul", 0, 2),
            op("mul", 0, 3),
            op("add", 4, 5),
            op("add", 7, 6),
            run(8),
            *reads(9),
        ],
    }


def handwritten() -> list[dict]:
    unused = [leaf(3.0), leaf(2.0), leaf(5.0), op("mul", 0, 1), op("add", 2, 2)]
    return [
        {"id": "seed_selected_output", "steps": [*unused, run(3), *reads(5)]},
        {"id": "seed_newest_when_selected", "steps": [*unused, run(4), *reads(5)]},
        binary_case("add", "add", 1.5, 2.25),
        binary_case("sub", "sub", 1.5, 2.25),
        binary_case("mul", "mul", 3.0, -0.5),
        binary_case("mul_rounds", "mul", 0.1, 3.0),
        {"id": "neg", "steps": [leaf(7.0), op("neg", 0), run(1), *reads(2)]},
        {"id": "square", "steps": [leaf(3.0), op("mul", 0, 0), run(1), *reads(2)]},
        {"id": "sub_self", "steps": [leaf(3.0), op("sub", 0, 0), run(1), *reads(2)]},
        {"id": "add_self", "steps": [leaf(3.0), op("add", 0, 0), run(1), *reads(2)]},
        fan_out("fan_out_loses_one", 1.0, -1e8, 1e8, 1.0),
        fan_out("fan_out_keeps_one", 1.0, 1.0, 1e8, -1e8),
        {
            "id": "diamond",
            "steps": [
                leaf(1.5),
                op("mul", 0, 0),
                op("add", 1, 0),
                op("mul", 2, 1),
                op("sub", 3, 0),
                op("neg", 4),
                run(5),
                *reads(6),
            ],
        },
        {
            "id": "interior_output",
            "steps": [
                leaf(2.0),
                leaf(-3.0),
                op("mul", 0, 1),
                op("add", 2, 0),
                op("mul", 3, 3),
                run(3),
                *reads(5),
            ],
        },
        {"id": "leaf_output", "steps": [*unused, run(1), *reads(5)]},
        {
            "id": "backward_twice",
            "steps": [*unused, run(3), run(3), *reads(5)],
        },
        {
            "id": "backward_two_outputs",
            "steps": [*unused, run(3), run(4), run(3), *reads(5)],
        },
        {
            "id": "zero_grad_then_backward",
            "steps": [*unused, run(3), ZERO_GRAD, *reads(5), run(4), *reads(5)],
        },
        {
            "id": "negative_zero_moves_in",
            "steps": [leaf(4.0), leaf(NEGATIVE_ZERO), op("mul", 0, 1), run(2)]
            + reads(3)
            + [run(2), *reads(3), ZERO_GRAD, run(2), *reads(3)],
        },
        {
            "id": "negative_zero_leaf",
            "steps": [leaf(NEGATIVE_ZERO), op("neg", 0), op("add", 1, 0), run(2)]
            + reads(3),
        },
        binary_case("inf_times_zero", "mul", float("inf"), 0.0),
        binary_case("inf_times_two", "mul", float("inf"), 2.0),
        binary_case("nan_plus_two", "add", QUIET_NAN, 2.0),
        binary_case("nan_times_two", "mul", QUIET_NAN, 2.0),
        {
            "id": "inf_minus_inf",
            "steps": [leaf(float("inf")), op("sub", 0, 0), run(1), *reads(2)],
        },
        binary_case("nan_payloads_add", "add", QUIET_NAN, OTHER_NAN),
        binary_case("nan_payloads_mul", "mul", QUIET_NAN, OTHER_NAN),
        binary_case("nan_payloads_sub", "sub", QUIET_NAN, OTHER_NAN),
        binary_case("nan_signaling_mul", "mul", QUIET_NAN, SIGNALING_NAN),
        {
            "id": "nan_upstream_mul",
            "steps": [
                leaf(QUIET_NAN),
                leaf(OTHER_NAN),
                leaf(THIRD_NAN),
                op("mul", 0, 1),
                op("mul", 3, 2),
                run(4),
                *reads(5),
            ],
        },
        {
            "id": "nan_fan_out",
            "steps": [
                leaf(1.0),
                leaf(QUIET_NAN),
                leaf(OTHER_NAN),
                op("mul", 0, 1),
                op("mul", 0, 2),
                op("add", 3, 4),
                run(5),
                *reads(6),
            ],
        },
        {
            "id": "nan_twice_non_leaf",
            "steps": [
                leaf(QUIET_NAN),
                leaf(OTHER_NAN),
                op("mul", 0, 1),
                op("neg", 2),
                op("add", 3, 2),
                run(4),
                run(4),
                *reads(5),
            ],
        },
        {
            "id": "nan_twice_leaf",
            "steps": [
                leaf(QUIET_NAN),
                leaf(OTHER_NAN),
                op("mul", 0, 1),
                op("neg", 2),
                op("mul", 3, 1),
                run(4),
                run(2),
                *reads(5),
            ],
        },
        binary_case("overflow", "mul", 3e38, 10.0),
        binary_case("subnormal", "mul", 1, 0.5),
        binary_case("subnormal_sum", "add", 1, 0x80000002),
        {
            "id": "unknown_operand",
            "steps": [
                leaf(2.0),
                op("add", 0, 5),
                op("mul", 5, 0),
                op("neg", 1),
                op("sub", 1, 1),
                run(1),
                *reads(2),
                op("neg", 0),
                run(1),
                *reads(2),
            ],
        },
        {"id": "unknown_output_empty", "steps": [run(0), *reads(1)]},
        {
            "id": "unknown_output",
            "steps": [leaf(2.0), op("neg", 0), run(2), run(7), *reads(3)],
        },
        {
            "id": "dangling_self",
            "steps": [
                leaf(2.0),
                forge({"op": "add", "a": 1, "b": 0}, 4.0),
                run(0),
                run(1),
                *reads(2),
            ],
        },
        {
            "id": "dangling_newer",
            "steps": [
                leaf(2.0),
                forge({"op": "neg", "a": 9}, -2.0),
                leaf(3.0),
                run(0),
                run(2),
                *reads(3),
            ],
        },
        {
            "id": "dangling_mul",
            "steps": [
                leaf(2.0),
                leaf(3.0),
                forge({"op": "mul", "a": 0, "b": 2, "left": 2.0, "right": 5.0}, 6.0),
                run(2),
                *reads(3),
            ],
        },
        {
            "id": "forged_sound",
            "steps": [
                leaf(2.0),
                leaf(3.0),
                forge({"op": "mul", "a": 0, "b": 1, "left": 7.0, "right": 5.0}, 1.0),
                run(2),
                *reads(3),
            ],
        },
    ]


POOL = (
    word(1.0),
    word(-2.0),
    word(0.5),
    word(3.0),
    word(0.1),
    word(-0.3),
    word(1e8),
    word(-1e8),
    word(7.25),
    word(1e-3),
    0,
    NEGATIVE_ZERO,
)


def generated() -> list[dict]:
    state = 0x2545F491
    result = []

    def draw(bound: int) -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state >> 8) % bound

    for index in range(48):
        leaves = 2 + draw(4)
        nodes = 3 + draw(12)
        steps = [leaf(POOL[draw(len(POOL))]) for _ in range(leaves)]
        for position in range(leaves, leaves + nodes):
            choice = draw(4)
            if choice == 3:
                steps.append(op("neg", draw(position)))
            else:
                steps.append(op(BINARY[choice], draw(position), draw(position)))
        count = leaves + nodes
        outputs = [count - 1 - draw(min(count, 4)) for _ in range(1 + draw(2))]
        for output in outputs:
            steps.append(run(output))
        steps += reads(count)
        if draw(3) == 0:
            steps += [ZERO_GRAD, run(count - 1), *reads(count)]
        result.append({"id": f"generated_{index:02d}", "steps": steps})
    return result


def cases() -> list[dict]:
    result = handwritten() + generated()
    if len({case["id"] for case in result}) != len(result):
        raise GateError("Duplicate S3 backward case identifier")
    return result


# F32 NaN payloads of the native lane on the pinned toolchain (clang 19 -O3 on
# x86-64): when an operand is NaN, the result is that operand's NaN, quieted.
# With two NaN operands, add and mul return the second operand's NaN and sub
# returns the first's. Python's binary64 arithmetic selects differently, so it
# is never used when an operand is NaN.
def nan_operand(first: int, second: int, second_first: bool) -> int | None:
    for operand in (second, first) if second_first else (first, second):
        if is_nan(operand):
            return operand | QUIET
    return None


def add_word(left: int, right: int) -> int:
    nan = nan_operand(left, right, True)
    return bits(f32(value(left) + value(right))) if nan is None else nan


def mul_word(left: int, right: int) -> int:
    nan = nan_operand(left, right, True)
    return bits(f32(value(left) * value(right))) if nan is None else nan


def sub_word(left: int, right: int) -> int:
    nan = nan_operand(left, right, False)
    return bits(f32(value(left) - value(right))) if nan is None else nan


def neg_word(number: int) -> int:
    return number ^ SIGN


def accumulate(total: int | None, contribution: int | None) -> int | None:
    if total is None:
        return contribution
    if contribution is None:
        return total
    return add_word(total, contribution)


def operands(node: dict) -> list[int]:
    return [node[key] for key in ("a", "b") if key in node]


def vjp(node: dict, gradient: int) -> list[tuple[int, int]]:
    name = node["op"]
    if name == "leaf":
        return []
    if name == "add":
        return [(node["a"], gradient), (node["b"], gradient)]
    if name == "sub":
        return [(node["a"], gradient), (node["b"], neg_word(gradient))]
    if name == "mul":
        return [
            (node["a"], mul_word(gradient, node["right"])),
            (node["b"], mul_word(gradient, node["left"])),
        ]
    return [(node["a"], neg_word(gradient))]


def gradients(nodes: list[dict], output: int) -> list[int | None]:
    found: list[int | None] = [None] * len(nodes)
    found[output] = ONE
    for position in range(output, -1, -1):
        gradient = found[position]
        if gradient is None:
            continue
        for target, contribution in vjp(nodes[position], gradient):
            if target < position:
                found[target] = accumulate(found[target], contribution)
    return found


def reference(case: dict) -> list[str]:
    nodes: list[dict] = []
    values: list[int] = []
    slots: list[int | None] = []
    lines = []

    def push(node: dict, number: int) -> None:
        lines.append(f"S3:id:{len(nodes)}")
        nodes.append(node)
        values.append(number)
        slots.append(None)

    for step in case["steps"]:
        name = step["op"]
        if name == "leaf":
            push({"op": "leaf"}, step["word"])
        elif name in BINARY or name == "neg":
            ids = operands(step)
            if any(identifier >= len(nodes) for identifier in ids):
                lines.append(f"S3:error:{UNKNOWN_NODE}")
                continue
            words = [values[identifier] for identifier in ids]
            if name == "neg":
                push({"op": "neg", "a": step["a"]}, neg_word(words[0]))
                continue
            node = {"op": name, "a": step["a"], "b": step["b"]}
            if name == "mul":
                node |= {"left": words[0], "right": words[1]}
            combine = {"add": add_word, "sub": sub_word, "mul": mul_word}[name]
            push(node, combine(*words))
        elif name == "forge":
            node = dict(step["node"])
            for key in ("left", "right"):
                if key in node:
                    node[key] = word(node[key])
            lines.append("S3:forged")
            nodes.append(node)
            values.append(step["word"])
            slots.append(None)
        elif name == "backward":
            output = step["output"]
            if output >= len(nodes):
                lines.append(f"S3:error:{UNKNOWN_NODE}")
            elif any(
                target >= position
                for position, node in enumerate(nodes)
                for target in operands(node)
            ):
                lines.append(f"S3:error:{DANGLING_OPERAND}")
            else:
                found = gradients(nodes, output)
                slots = [accumulate(a, b) for a, b in zip(slots, found, strict=True)]
                lines.append("S3:backward")
        elif name == "zero_grad":
            slots = [None] * len(nodes)
            lines.append("S3:zero_grad")
        else:
            identifier = step["id"]
            if identifier >= len(nodes):
                lines += [
                    f"S3:value:error:{UNKNOWN_NODE}",
                    f"S3:grad:error:{UNKNOWN_NODE}",
                ]
                continue
            slot = slots[identifier]
            lines += [
                f"S3:value:{values[identifier]}",
                "S3:grad:none" if slot is None else f"S3:grad:{slot}",
            ]
    return lines


def node_source(node: dict) -> str:
    name = node["op"].capitalize()
    fields = [f"{identifier}n" for identifier in operands(node)]
    fields += [
        f"Harness.from_bits({word(node[key])})"
        for key in ("left", "right")
        if key in node
    ]
    return f"Backward.{name}{{{', '.join(fields)}}}"


def step_source(step: dict) -> str:
    name = step["op"]
    if name == "leaf":
        return f"Harness.Leaf{{{step['word']}}}"
    if name == "backward":
        return f"Harness.Run{{{step['output']}n}}"
    if name == "zero_grad":
        return "Harness.ZeroGrad{}"
    if name == "read":
        return f"Harness.Read{{{step['id']}n}}"
    if name == "forge":
        return f"Harness.Forge{{{node_source(step['node'])}, {step['word']}}}"
    fields = ", ".join(f"{identifier}n" for identifier in operands(step))
    return f"Harness.{name.capitalize()}{{{fields}}}"


def program(selected: list[dict], directory: Path, source_root: Path) -> str:
    if not selected:
        raise GateError("Refusing to generate an empty S3 backward corpus")
    lines = ["import Base"]
    for path, alias in (
        ("src/autograd/backward-harness.bend", "Harness"),
        ("src/autograd/backward.bend", "Backward"),
    ):
        relative = os.path.relpath(source_root / path, directory)
        lines.append(f"import ./{relative} as {alias}")
    for index, case in enumerate(selected):
        steps = ", ".join(step_source(step) for step in case["steps"])
        lines += [
            "",
            f"def run_case_{index}() -> IO(Unit):",
            "  do IO<Unit>:",
            f'    IO.print("S3:begin:{case["id"]}")',
            f"    Harness.script([{steps}], Backward.empty())",
            '    IO.print("S3:end")',
        ]
    batches = range(0, len(selected), 16)
    for batch, start in enumerate(batches):
        lines += ["", f"def run_batch_{batch}() -> IO(Unit):", "  do IO<Unit>:"]
        lines += [
            f"    run_case_{index}()"
            for index in range(start, min(start + 16, len(selected)))
        ]
    lines += ["", "def main() -> IO(Unit):", "  do IO<Unit>:"]
    lines += [f"    run_batch_{batch}()" for batch in range(len(batches))]
    return "\n".join(lines) + "\n"


def parse(text: str, identifiers: list[str]) -> list[dict]:
    rows = text.splitlines()
    results = []
    index = 0
    while index < len(rows):
        begin = rows[index]
        if not begin.startswith("S3:begin:"):
            raise GateError(f"Unframed S3 output: {begin!r}")
        try:
            end = rows.index("S3:end", index)
        except ValueError as error:
            raise GateError(f"{begin}: missing S3:end") from error
        results.append(
            {"id": begin.removeprefix("S3:begin:"), "lines": rows[index + 1 : end]}
        )
        index = end + 1
    if [row["id"] for row in results] != identifiers:
        raise GateError("S3 backward inventory or order differs from the requests")
    return results


WORD_PREFIXES = ("S3:value:", "S3:grad:")


# The native lane must match bit for bit. The JavaScript lane may return a NaN
# with another payload, as in the other S3 subgates.
def same_line(seen: str, wanted: str, lane: str) -> bool:
    if seen == wanted:
        return True
    if lane == "native":
        return False
    for prefix in WORD_PREFIXES:
        if seen.startswith(prefix) and wanted.startswith(prefix):
            left, right = seen.removeprefix(prefix), wanted.removeprefix(prefix)
            if left.isdigit() and right.isdigit():
                return is_nan(int(left)) and is_nan(int(right))
    return False


def verify(selected: list[dict], observations: list[dict], lane: str) -> dict:
    if [row["id"] for row in observations] != [case["id"] for case in selected]:
        raise GateError(f"{lane}: incomplete S3 backward case inventory")
    summary = {"cases": len(selected), "backward": 0, "errors": 0, "none_reads": 0}
    for case, row in zip(selected, observations, strict=True):
        expected = reference(case)
        seen = row["lines"]
        if len(seen) != len(expected) or not all(
            same_line(a, b, lane) for a, b in zip(seen, expected, strict=True)
        ):
            raise GateError(f"{lane} {case['id']}: output differs from the reference")
        summary["backward"] += expected.count("S3:backward")
        summary["errors"] += sum(
            line.startswith(("S3:error:", "S3:value:error:")) for line in expected
        )
        summary["none_reads"] += expected.count("S3:grad:none")
    return summary


def run_stage(tool_root: Path, output: Path) -> dict:
    selected = cases()

    def render(directory: Path, source_root: Path) -> str:
        return program(selected, directory, source_root)

    def check(stdout: str, lane: str) -> tuple[list, dict]:
        observations = parse(stdout, [case["id"] for case in selected])
        return observations, verify(selected, observations, lane)

    summary = run_lanes(tool_root, output, "backward", selected, render, check)
    summary["runtime_controls"] = runtime_controls(
        tool_root.resolve(), output.resolve() / "runtime-controls", selected
    )
    return summary


def runtime_controls(tool_root: Path, output: Path, selected: list[dict]) -> list[str]:
    evidence = Evidence(output, tool_root)
    rejected = []
    for name, (original, mutated) in RUNTIME_CONTROLS.items():
        copy = output / name
        shutil.copytree(ROOT / "src", copy / "src")
        target = copy / "src/autograd/backward.bend"
        text = target.read_text()
        if text.count(original) != 1:
            raise GateError(f"{name}: runtime control anchor is missing or ambiguous")
        target.write_text(text.replace(original, mutated))
        source = copy / "backward.bend"
        source.write_text(program(selected, copy, copy))
        script = copy / "backward.js"
        evidence.run(f"{name}-javascript-build", bend(tool_root, source, "-o", script))
        result = evidence.run(
            f"{name}-javascript-run", [bend(tool_root)[0], str(script)], timeout=1800
        )
        try:
            verify(selected, parse(result.stdout, [c["id"] for c in selected]), name)
        except GateError:
            rejected.append(name)
            continue
        raise GateError(f"{name}: the backward lane accepted a broken engine")
    evidence.save()
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.tool_root, args.output), indent=2))


if __name__ == "__main__":
    main()
