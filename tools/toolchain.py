import json
import shutil
from pathlib import Path

from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_inventory,
    check_omissions,
    check_sources,
    native,
)

# 0 / 0 is the x86 default NaN. JavaScript numbers are doubles, so loading the
# signaling NaN word quiets it; native passes the word through unchanged.
F32_PROBE = (
    ("1.5 + 2.25", 0x40700000, 0x40700000),
    ("1.5 - 2.25", 0xBF400000, 0xBF400000),
    ("1.5 * 2.25", 0x40580000, 0x40580000),
    ("1.5 / 2.25", 0x3F2AAAAB, 0x3F2AAAAB),
    ("word 0x80000000, negative zero", 0x80000000, 0x80000000),
    ("word 0x00000001, smallest subnormal", 0x00000001, 0x00000001),
    ("1 / 0", 0x7F800000, 0x7F800000),
    ("0 / 0", 0xFFC00000, 0xFFC00000),
    ("word 0x7f800001, signaling NaN", 0x7F800001, 0x7FC00000),
    ("U32 16777217 to F32", 0x4B800000, 0x4B800000),
    ("F32 3.75 to U32", 3, 3),
    ("largest finite * 2", 0x7F800000, 0x7F800000),
    ("-1 / 0", 0xFF800000, 0xFF800000),
)
LANES = ("native", "javascript")


def run_gate(tool_root: Path, output: Path) -> None:
    check_sources(tool_root)
    gate = json.loads((ROOT / "verification/toolchain.json").read_text())
    evidence = Evidence(output, tool_root)
    fingerprints = tool_root / "fingerprints.json"
    if not fingerprints.is_file():
        raise GateError("Run the pinned bootstrap before checking the toolchain")
    shutil.copyfile(fingerprints, output / "fingerprints.json")
    source = tool_root / "bend/demos/pure_par_sum"
    clean = output / "unchanged"
    shutil.copytree(source, clean)
    root = clean / "PROOF.bend"
    evidence.run(
        "source-check",
        bend(tool_root, root, "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    evidence.run(
        "kernel-check", bend(tool_root, root, "--verdict"), contains="ALL PROOFS CHECK"
    )
    exported = output / "PROOF.bendtt"
    evidence.run("export", bend(tool_root, root, "-o", exported))
    check_inventory(exported.read_text(), gate["required_exports"])
    kernel = str(tool_root / "kernel/bendtt")
    evidence.run(
        "export-kernel", [kernel, str(exported)], expected_output="ALL PROOFS CHECK\n"
    )
    false_export = output / "false-export.bendtt"
    false_export.write_text(
        exported.read_text().replace(
            "(main.seq (main.pow2 x0) +x1) : Nat} =",
            "(main.seq (main.pow2 x0) +(Nat.add x1 x1)) : Nat} =",
        )
    )
    evidence.run(
        "false-export-kernel",
        [kernel, str(false_export)],
        expected_code=1,
        contains=("SOME PROOFS FAIL", "In LAWS.tree_is_seq:"),
    )
    proof = (source / "PROOF.bend").read_text()
    laws = (source / "LAWS.bend").read_text()
    unproven = proof.split("def Laws.tree_is_seq", 1)[0]
    mutations = {
        "false-law": (
            proof,
            laws.replace("{Par.sum(d, i)", "{Nat.add(1n, Par.sum(d, i))"),
            "Location: Laws.tree_is_seq",
        ),
        "open-law": (unproven, laws, "Error: 1 TODO found."),
        "missing-import": (
            unproven.replace("import ./LAWS.bend as Laws\n", ""),
            laws,
            "Error: PROOF.bend must import ./LAWS.bend",
        ),
        "unsafe": (
            unproven + "@unsafe\ndef escape(d: Nat, +i: Nat) -> "
            "{Par.sum(d, i) == Par.seq(Par.pow2(d), i) : Nat}:\n  escape(d, i)\n\n"
            "def Laws.tree_is_seq(d, i):\n  escape(d, i)\n",
            laws,
            "rely on unsafe or foreign code:\n- LAWS.tree_is_seq\n",
        ),
    }
    for name, (changed_proof, changed_laws, reason) in mutations.items():
        directory = output / name
        shutil.copytree(source, directory)
        (directory / "PROOF.bend").write_text(changed_proof)
        (directory / "LAWS.bend").write_text(changed_laws)
        evidence.run(
            name,
            bend(tool_root, directory / "PROOF.bend", "--verdict"),
            expected_code=1,
            contains=("SOME PROOFS FAIL", reason),
        )
    check_omissions(exported.read_text(), gate["required_exports"])
    binary = output / "parallel-sum"
    emitted = output / "parallel-sum.c"
    evidence.run("native-c-export", bend(tool_root, clean / "main.bend", "-o", emitted))
    evidence.run(
        "native-build",
        [
            str(tool_root / "llvm/usr/bin/clang-19"),
            "-std=c11",
            "-O3",
            str(emitted),
            "-lpthread",
            "-lm",
            "-o",
            str(binary),
        ],
    )
    links = evidence.run("native-dependencies", ["ldd", str(binary)])
    if binary.with_suffix(".gpu").exists() or any(
        name in links.stdout.lower() for name in ("cuda", "nvrtc")
    ):
        raise GateError("The native build linked the CUDA lane")
    evidence.run(
        "native-run",
        native(binary),
        expected_output="2147450880\n",
    )
    expected = "1088421888\n0\n1088421888\n1093664768\n"
    for lane in ("native", "javascript"):
        target = output / ("owned-array" if lane == "native" else "owned-array.js")
        evidence.run(
            f"array-{lane}-build",
            bend(tool_root, ROOT / "probes/owned_array.bend", "-o", target),
        )
        command = (
            native(target) if lane == "native" else [bend(tool_root)[0], str(target)]
        )
        evidence.run(f"array-{lane}-run", command, expected_output=expected)
    malformed = ROOT / "probes/unbalanced_array.bend"
    evidence.run(
        "unbalanced-check",
        bend(tool_root, malformed, "--check-only"),
        contains="ALL PROOFS CHECK",
    )
    evidence.run(
        "unbalanced-build",
        bend(tool_root, malformed, "-o", output / "unbalanced"),
    )
    evidence.run(
        "unbalanced-run",
        native(output / "unbalanced"),
        expected_code=1,
        contains="runtime fail-stop",
    )
    observed = {}
    for column, lane in enumerate(LANES, start=1):
        target = output / ("numbers" if lane == "native" else "numbers.js")
        evidence.run(
            f"numbers-{lane}-build",
            bend(tool_root, ROOT / "probes/numbers.bend", "-o", target),
        )
        command = (
            native(target) if lane == "native" else [bend(tool_root)[0], str(target)]
        )
        result = evidence.run(f"numbers-{lane}-run", command)
        bits = [int(line) for line in result.stdout.splitlines()]
        expected = [row[column] for row in F32_PROBE]
        if bits != expected:
            raise GateError(
                f"{lane}: F32 bits {[f'{b:08x}' for b in bits]} "
                f"differ from {[f'{b:08x}' for b in expected]}"
            )
        observed[lane] = bits
    rows = [
        {"probe": row[0], **{lane: f"{observed[lane][i]:08x}" for lane in LANES}}
        for i, row in enumerate(F32_PROBE)
    ]
    (output / "numbers.json").write_text(json.dumps(rows, indent=2) + "\n")
    evidence.save()
    print(f"Toolchain proof and runtime gate passed: {output}")
