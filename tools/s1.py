import argparse
import json
import sys
from pathlib import Path

from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_inventory,
    check_sources,
)
from tools.s1_shapes import run_stage as shapes
from tools.s1_storage import run_stage as storage


def run_stage(tool_root: Path, output: Path) -> None:
    check_sources(tool_root)
    required = json.loads((ROOT / "verification/s1.json").read_text())[
        "required_exports"
    ]
    evidence = Evidence(output, tool_root)
    root = ROOT / "verification/s1/PROOF.bend"
    for name, mode in (("source", "--check-only"), ("kernel", "--verdict")):
        evidence.run(
            f"s1-{name}-check", bend(tool_root, root, mode), contains="ALL PROOFS CHECK"
        )
    artifact = output / "PROOF.bendtt"
    evidence.run("s1-export", bend(tool_root, root, "-o", artifact))
    check_inventory(artifact.read_text(), required)
    evidence.run(
        "s1-direct-kernel",
        [str(tool_root / "kernel/bendtt"), str(artifact)],
        contains="ALL PROOFS CHECK",
    )
    evidence.run(
        "pytorch-absent",
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('torch') is None; "
            "print('PyTorch is absent from the root execution environment')",
        ],
        expected_output="PyTorch is absent from the root execution environment\n",
    )
    shapes(tool_root, output / "shapes")
    storage(tool_root, output / "storage")
    for name in required:
        missing = artifact.read_text().replace(name + " :", "omitted :", 1)
        try:
            check_inventory(missing, required)
        except GateError:
            continue
        raise GateError(
            f"The inventory accepted an omitted required declaration: {name}"
        )
    (output / "omitted-inventory.json").write_text(
        json.dumps({"rejected_omissions": required}, indent=2) + "\n"
    )
    evidence.save()
    print(f"S1 authoritative gate passed: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_stage(args.tool_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
