import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from tools.common import ROOT, GateError, check_sources, digest, environment, manifest


def bootstrap(tool_root: Path, download_cache: Path | None = None) -> None:
    refs = manifest()
    tool_root.mkdir(parents=True, exist_ok=True)
    downloads = tool_root / "downloads"
    downloads.mkdir(exist_ok=True)
    source = tool_root / "bend"
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                refs["sources"]["bend"]["url"],
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--depth=1",
                "origin",
                refs["sources"]["bend"]["commit"],
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    check_sources(tool_root)
    for artifact in refs["downloads"]:
        target = downloads / artifact["file"]
        cached = None if download_cache is None else download_cache / artifact["file"]
        if not target.exists():
            if cached is not None and cached.is_file():
                shutil.copyfile(cached, target)
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                with urllib.request.urlopen(artifact["url"], timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                temporary.rename(target)
        if digest(target) != artifact["sha256"]:
            raise GateError(f"Artifact checksum mismatch: {target}")
    lean = tool_root / "lean-4.34.0-linux"
    if not lean.exists():
        subprocess.run(
            [
                "tar",
                "--zstd",
                "-xf",
                str(downloads / refs["downloads"][0]["file"]),
                "-C",
                str(tool_root),
            ],
            check=True,
        )
    llvm = tool_root / "llvm"
    if not llvm.exists():
        llvm.mkdir()
        for artifact in refs["downloads"][1:]:
            subprocess.run(
                ["dpkg-deb", "-x", str(downloads / artifact["file"]), str(llvm)],
                check=True,
            )
    kernel = tool_root / "kernel"
    kernel.mkdir(exist_ok=True)
    original = source / "bend2/bendtt.lean"
    if not (kernel / "bendtt").exists():
        shutil.copyfile(original, kernel / "bendtt.lean")
        subprocess.run(
            [str(lean / "bin/lean"), "-c", "bendtt.c", "bendtt.lean"],
            cwd=kernel,
            check=True,
        )
        subprocess.run(
            [str(lean / "bin/leanc"), "-O3", "-DNDEBUG", "bendtt.c", "-o", "bendtt"],
            cwd=kernel,
            check=True,
        )
    if digest(kernel / "bendtt.lean") != digest(original):
        raise GateError("Cached kernel source differs from pinned source")
    tools = {
        "bun": [shutil.which("bun") or "bun", "--version"],
        "uv": [shutil.which("uv") or "uv", "--version"],
        "lean": [str(lean / "bin/lean"), "--version"],
        "clang": [str(llvm / "usr/bin/clang-19"), "--version"],
    }
    fingerprints = {}
    for name, command in tools.items():
        result = subprocess.run(
            command,
            env=environment(tool_root),
            capture_output=True,
            text=True,
            check=True,
        )
        if refs["tools"][name] not in result.stdout:
            raise GateError(f"Wrong {name} version: {result.stdout}")
        fingerprints[name] = {
            "version_output": result.stdout,
            "binary_sha256": digest(Path(command[0]).resolve()),
        }
    fingerprints["bendtt"] = {"binary_sha256": digest(kernel / "bendtt")}
    (tool_root / "fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", type=Path, default=ROOT / ".tools")
    parser.add_argument("--download-cache", type=Path)
    args = parser.parse_args()
    bootstrap(args.tool_root.resolve(), args.download_cache)


if __name__ == "__main__":
    main()
