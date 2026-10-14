import argparse
from pathlib import Path

from tools import autograd, shape, storage, tensor, toolchain
from tools.common import ROOT, GateError

SUITES = {
    "toolchain": toolchain.run_gate,
    "shape": shape.run_gate,
    "storage": storage.run_gate,
    "tensor": tensor.run_gate,
    "autograd": autograd.run_gate,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gate of each named suite; `all` runs every suite."
    )
    parser.add_argument("suites", nargs="+", choices=[*SUITES, "all"])
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = list(SUITES) if "all" in args.suites else list(dict.fromkeys(args.suites))
    output = args.output.resolve()
    if output.exists():
        raise GateError(f"Refusing to overwrite evidence: {output}")
    output.mkdir(parents=True)
    for name in names:
        SUITES[name](args.tool_root.resolve(), output / name)
    print(f"Gates passed ({', '.join(names)}): {output}")


if __name__ == "__main__":
    main()
