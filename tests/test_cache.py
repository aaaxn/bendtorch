import gzip
import hashlib
import hmac
import json
import os
import resource
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.cache import Cache, key
from tools.common import ROOT, bend, environment

TOOLS = ROOT / ".tools"
LIB = "import Base\n\ndef two() -> Nat:\n  2n\n"
LAW = (
    "import Base\nimport ./lib.bend as Lib\n\n"
    "def two_is_two() -> {Lib.two() == 2n : Nat}:\n  {==}\n"
)


class CacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.src = self.root / "src"
        self.src.mkdir()
        (self.src / "lib.bend").write_text(LIB)
        (self.src / "law.bend").write_text(LAW)
        self.secret = self.root / "secret"
        self.cache = Cache(self.root / "cache", self.secret)
        self.env = environment(TOOLS)

    def run_bend(self, *args, env=None):
        return self.cache.run(
            bend(TOOLS, *args), cwd=self.root, env=env or self.env, timeout=300
        )

    def entries(self) -> list[Path]:
        return sorted((self.root / "cache").rglob("*.json.gz"))

    def test_a_second_run_replays_the_first(self):
        cold, replayed = self.run_bend(self.src / "law.bend", "--verdict")
        self.assertFalse(replayed)
        self.assertEqual(cold.returncode, 0)
        self.assertIn("ALL PROOFS CHECK", cold.stdout)
        warm, replayed = self.run_bend(self.src / "law.bend", "--verdict")
        self.assertTrue(replayed)
        self.assertEqual(
            (warm.returncode, warm.stdout, warm.stderr),
            (cold.returncode, cold.stdout, cold.stderr),
        )
        self.assertEqual(self.cache.summary()["replayed"], 1)

    def test_a_law_that_fails_cold_fails_warm(self):
        law = self.src / "law.bend"
        self.assertEqual(self.run_bend(law, "--verdict")[0].returncode, 0)
        # The same path now states a false law: the cache must not answer.
        law.write_text(LAW.replace("== 2n", "== 3n"))
        for replayed in (False, True):
            result, hit = self.run_bend(law, "--verdict")
            self.assertEqual(hit, replayed)
            self.assertEqual(result.returncode, 1)
            self.assertIn("SOME PROOFS FAIL", result.stdout + result.stderr)

    def test_a_changed_import_misses(self):
        law = self.src / "law.bend"
        self.assertEqual(self.run_bend(law, "--check-only")[0].returncode, 0)
        (self.src / "lib.bend").write_text(LIB.replace("2n", "3n"))
        result, replayed = self.run_bend(law, "--check-only")
        self.assertFalse(replayed)
        self.assertEqual(result.returncode, 1)

    def test_an_import_moved_elsewhere_misses(self):
        # Bend names an import after its path, so the layout is in the key.
        law = self.src / "law.bend"
        self.run_bend(law, "--check-only")
        (self.src / "sub").mkdir()
        (self.src / "sub/lib.bend").write_text(LIB)
        law.write_text(LAW.replace("./lib.bend", "./sub/lib.bend"))
        self.assertFalse(self.run_bend(law, "--check-only")[1])

    def test_flags_and_environment_are_in_the_key(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--check-only")
        self.assertFalse(self.run_bend(law, "--verdict")[1])
        other = {**self.env, "LEAN_NUM_THREADS": "3"}
        self.assertFalse(self.run_bend(law, "--verdict", env=other)[1])
        self.assertTrue(self.run_bend(law, "--verdict", env=other)[1])

    def test_bend2_and_the_environment_are_in_the_key(self):
        fake = self.root / "bend"
        (fake / "bend2").mkdir(parents=True)
        (fake / "bend2/main.ts").write_text("// one\n")
        command = ["bun", fake / "bend2/main.ts", self.src / "law.bend", "--check-only"]
        first = key(command, self.root, self.env)
        (fake / "bend2/safe.ts").write_text("// a new file\n")
        second = key(command, self.root, self.env)
        self.assertIsNotNone(first)
        self.assertNotEqual(first, second)
        # Bun reads variables of its own, such as NODE_OPTIONS.
        bun_options = {**self.env, "NODE_OPTIONS": "--stack-size=1"}
        self.assertNotEqual(key(command, self.root, bun_options), second)
        # A variable that only names the session is not in the key.
        inert = {**self.env, "SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}
        self.assertEqual(key(command, self.root, inert), second)
        # Where Bend's library resolves decides how imports are named.
        library = self.root / "library"
        library.mkdir()
        here = {**self.env, "BEND_LIB": str(self.root / "lib")}
        (self.root / "lib").symlink_to(library)
        before = key(command, self.root, here)
        (self.root / "lib").unlink()
        (self.root / "lib").symlink_to(self.src)
        self.assertNotEqual(key(command, self.root, here), before)

    def test_bun_configuration_or_a_relative_path_is_not_cached(self):
        fake = self.root / "bend"
        (fake / "bend2").mkdir(parents=True)
        (fake / "bend2/main.ts").write_text("// one\n")
        command = ["bun", fake / "bend2/main.ts", self.src / "law.bend", "--check-only"]
        self.assertIsNotNone(key(command, self.root, self.env))
        home = {**self.env, "HOME": str(self.root / "home")}
        for config in (
            fake / "package.json",
            fake / "tsconfig.json",
            self.root / "bunfig.toml",
            self.root / ".env",
            self.root / ".env.local",
            self.root / "home/.bunfig.toml",
        ):
            config.parent.mkdir(exist_ok=True)
            config.write_text("")
            self.assertIsNone(key(command, self.root, home), config)
            config.unlink()
        self.assertIsNotNone(key(command, self.root, home))
        relative = {**self.env, "PATH": f"bin:{self.env['PATH']}"}
        self.assertIsNone(key(command, self.root, relative))

    def test_the_kernel_that_runs_is_in_the_key(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        fake = self.root / "bin/bendtt"
        fake.parent.mkdir()
        fake.write_bytes((TOOLS / "kernel/bendtt").read_bytes() + b"\0")
        fake.chmod(0o755)
        result, replayed = self.run_bend(
            law, "--verdict", env={**self.env, "BENDTT": str(fake)}
        )
        self.assertFalse(replayed)
        self.assertEqual(result.returncode, 0)
        # A bare name is found on PATH, as the run finds it.
        artifact = self.root / "x.bendtt"
        artifact.write_text("x")
        command = ["bendtt", str(artifact)]
        on_path = {**self.env, "PATH": f"{fake.parent}:{self.env['PATH']}"}
        material = key(command, self.root, on_path)
        assert material is not None
        self.assertEqual(
            material["kernel"][0],
            [str(fake), hashlib.sha256(fake.read_bytes()).hexdigest()],
        )
        # Replaced at the same path, in the same process: a new key.
        before = key(
            bend(TOOLS, law, "--verdict"), self.root, {**self.env, "BENDTT": str(fake)}
        )
        with fake.open("ab") as stream:
            stream.write(b"\0")
        after = key(
            bend(TOOLS, law, "--verdict"), self.root, {**self.env, "BENDTT": str(fake)}
        )
        self.assertIsNotNone(before)
        self.assertNotEqual(before, after)

    def test_an_export_always_runs_and_its_kernel_check_replays(self):
        law = self.src / "law.bend"
        artifact = self.root / "out/law.bendtt"
        artifact.parent.mkdir()
        self.assertIsNone(key(bend(TOOLS, law, "-o", artifact), self.root, self.env))
        for _ in range(2):
            artifact.unlink(missing_ok=True)
            self.assertFalse(self.run_bend(law, "-o", artifact)[1])
            self.assertTrue(artifact.exists())
        text = artifact.read_text()
        kernel = [str(TOOLS / "kernel/bendtt"), str(artifact)]

        def run() -> bool:
            return self.cache.run(kernel, cwd=self.root, env=self.env, timeout=300)[1]

        self.assertFalse(run())
        self.assertTrue(run())
        # A changed artifact is a different kernel input.
        artifact.write_text(text + "\n")
        self.assertFalse(run())

    def test_the_working_directory_and_temporary_paths_are_in_the_key(self):
        command = bend(TOOLS, self.src / "law.bend", "--verdict")
        other = self.root / "other"
        other.mkdir()
        self.assertNotEqual(
            key(command, self.root, self.env), key(command, other, self.env)
        )
        for name in ("TMPDIR", "TMP", "TEMP"):
            relative = {**self.env, name: "tmp"}
            self.assertIsNone(key(command, self.root, relative), name)
        temporary = self.root / "tmp"
        temporary.mkdir()
        absolute = {**self.env, "TMPDIR": str(temporary)}
        present = key(command, self.root, absolute)
        temporary.rmdir()
        self.assertNotEqual(key(command, self.root, absolute), present)

    def test_resource_limits_are_in_the_key(self):
        command = bend(TOOLS, self.src / "law.bend", "--check-only")
        before = key(command, self.root, self.env)
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0 if soft else min(1, hard), hard))
        try:
            self.assertNotEqual(key(command, self.root, self.env), before)
        finally:
            resource.setrlimit(resource.RLIMIT_CORE, (soft, hard))

    def test_the_timeout_is_in_the_key(self):
        law = self.src / "law.bend"
        command = bend(TOOLS, law, "--check-only")
        self.cache.run(command, cwd=self.root, env=self.env, timeout=300)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.cache.run(command, cwd=self.root, env=self.env, timeout=0)

    def test_a_kernel_output_that_names_its_input_is_not_stored(self):
        kernel = self.shell_kernel()
        self.run_kernel(kernel, "echo 'SOME PROOFS FAIL'; echo x.bendtt; exit 1\n")
        self.assertEqual(self.entries(), [])
        self.run_kernel(kernel, "echo 'SOME PROOFS FAIL'; echo /tmp/bendtt-x; exit 1\n")
        self.assertEqual(self.entries(), [])

    def test_an_output_that_names_a_file_is_not_stored(self):
        missing = self.src / "missing.bend"
        missing.write_text("import Base\nimport ./absent.bend as A\n")
        result, _ = self.run_bend(missing, "--check-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn(str(self.src), result.stdout + result.stderr)
        self.assertEqual(self.entries(), [])
        self.run_bend(self.src / "law.bend", "--check-only")
        self.assertEqual(len(self.entries()), 1)

    def test_a_tampered_or_broken_entry_is_run_again(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        (entry,) = self.entries()
        stored = json.loads(gzip.decompress(entry.read_bytes()))
        stored["exit_code"] = 0
        stored["stdout"] = "ALL PROOFS CHECK\nforged\n"
        entry.write_bytes(gzip.compress(json.dumps(stored).encode()))
        result, replayed = self.run_bend(law, "--verdict")
        self.assertFalse(replayed)
        self.assertNotIn("forged", result.stdout)
        self.assertEqual(self.cache.summary()["rejected"], [entry.name])
        # The rerun replaced the entry, which now replays.
        self.assertTrue(self.run_bend(law, "--verdict")[1])
        entry.write_bytes(b"not gzip")
        self.assertFalse(self.run_bend(law, "--verdict")[1])

    def test_a_false_law_resealed_without_the_secret_is_run_again(self):
        law = self.src / "law.bend"
        law.write_text(LAW.replace("== 2n", "== 3n"))
        self.run_bend(law, "--verdict")
        (entry,) = self.entries()
        stored = json.loads(gzip.decompress(entry.read_bytes()))
        stored["exit_code"] = 0
        stored["stdout"], stored["stderr"] = "ALL PROOFS CHECK\n", ""
        for secret in (b"", b"x" * 32):
            body = json.dumps(
                {k: v for k, v in stored.items() if k != "seal"}, sort_keys=True
            ).encode()
            stored["seal"] = hmac.new(secret, body, hashlib.sha256).hexdigest()
            entry.write_bytes(gzip.compress(json.dumps(stored).encode()))
            result, replayed = self.run_bend(law, "--verdict")
            self.assertFalse(replayed)
            self.assertEqual(result.returncode, 1)

    def test_another_checkouts_cache_is_not_trusted(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        other = Cache(self.root / "cache", self.root / "other-secret")
        result, replayed = other.run(
            bend(TOOLS, law, "--verdict"), cwd=self.root, env=self.env, timeout=300
        )
        self.assertFalse(replayed)
        self.assertEqual(len(other.rejected), 1)

    def test_a_proof_beside_unimported_laws_is_keyed_on_them(self):
        proof = self.src / "PROOF.bend"
        proof.write_text(LAW)
        self.assertEqual(self.run_bend(proof, "--check-only")[0].returncode, 0)
        (self.src / "LAWS.bend").write_text("import Base\n")
        result, replayed = self.run_bend(proof, "--check-only")
        self.assertFalse(replayed)
        self.assertEqual(result.returncode, 1)

    def test_a_file_reached_through_a_symlink_is_not_cached(self):
        link = self.root / "PROOF.bend"
        link.symlink_to(self.src / "law.bend")
        self.assertIsNone(key(bend(TOOLS, link, "--check-only"), self.root, self.env))

    def test_a_path_bend_would_read_otherwise_is_not_cached(self):
        # Bend opens `absent/../law.bend` as given, and it does not exist.
        dotted = f"{self.src}/absent/../law.bend"
        self.assertIsNone(key(bend(TOOLS, dotted, "--check-only"), self.root, self.env))
        law = self.src / "law.bend"
        # A relative or empty loader path is read from the working directory.
        for search in ("lib", "/nonexistent:", "::/usr/lib"):
            relative = {**self.env, "LD_LIBRARY_PATH": search}
            self.assertIsNone(
                key(bend(TOOLS, law, "--check-only"), self.root, relative), search
            )
        # An import through a link: swapping two links would keep the bytes.
        (self.src / "alias.bend").symlink_to(self.src / "lib.bend")
        linked = self.src / "linked.bend"
        linked.write_text(LAW.replace("./lib.bend", "./alias.bend"))
        self.assertIsNone(key(bend(TOOLS, linked, "--check-only"), self.root, self.env))
        # A foreign definition may sit on the def's own line.
        inline = self.src / "inline.bend"
        for foreign in (
            'def f() -> U32: import "./f.c"\n',
            'def f() -> U32: import # a comment\n  "./f.js"\n',
        ):
            inline.write_text("import Base\n\n" + foreign)
            self.assertIsNone(
                key(bend(TOOLS, inline, "--check-only"), self.root, self.env), foreign
            )
        # An import under Bend's library, even one not made yet, is a hub file.
        library = {**self.env, "BEND_LIB": str(self.root / "hub")}
        hub = self.src / "hub.bend"
        hub.write_text("import Base\nimport ../hub/x.bend as X\n")
        self.assertIsNone(key(bend(TOOLS, hub, "--check-only"), self.root, library))
        self.assertIsNotNone(key(bend(TOOLS, law, "--check-only"), self.root, library))
        # Named under the library, even if the link leads out of it.
        (self.root / "hub").mkdir()
        (self.root / "hub/x.bend").symlink_to(self.src / "lib.bend")
        self.assertIsNone(key(bend(TOOLS, hub, "--check-only"), self.root, library))

    def shell_kernel(self) -> Path:
        """A real executable named bendtt that runs its input as a script."""
        kernel = self.root / "sh/bendtt"
        kernel.parent.mkdir(exist_ok=True)
        shutil.copy(shutil.which("sh") or "/bin/sh", kernel)
        return kernel

    def run_kernel(self, kernel: Path, script: str):
        artifact = self.root / "x.bendtt"
        artifact.write_text(script)
        return self.cache.run(
            [str(kernel), str(artifact)], cwd=self.root, env=self.env, timeout=60
        )

    def test_an_input_changed_during_the_run_is_not_stored(self):
        kernel = self.shell_kernel()
        self.assertEqual(
            self.run_kernel(kernel, "echo 'ALL PROOFS CHECK'\n")[0].returncode, 0
        )
        self.assertEqual(len(self.entries()), 1)
        # The kernel edits its own input after reading it, then passes.
        artifact = self.root / "x.bendtt"
        changes = f"echo '# changed' >> {artifact}; echo 'ALL PROOFS CHECK'; exit 0\n"
        self.assertEqual(self.run_kernel(kernel, changes)[0].returncode, 0)
        self.assertEqual(len(self.entries()), 1)

    def test_a_forged_pass_for_a_false_law_is_not_replayed(self):
        # An entry moved under another key, with a valid digest of its own,
        # still names the key it was made for.
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        (good,) = self.entries()
        law.write_text(LAW.replace("== 2n", "== 3n"))
        self.run_bend(law, "--verdict")
        (bad,) = [path for path in self.entries() if path != good]
        bad.write_bytes(good.read_bytes())
        result, replayed = self.run_bend(law, "--verdict")
        self.assertFalse(replayed)
        self.assertEqual(result.returncode, 1)

    def test_unpinnable_inputs_are_not_cached(self):
        foreign = self.src / "foreign.bend"
        foreign.write_text('import Base\n\ndef f() -> U32:\n  import "f.c"\n')
        self.assertIsNone(
            key(bend(TOOLS, foreign, "--check-only"), self.root, self.env)
        )
        hub = self.src / "hub.bend"
        hub.write_text("import Base\nimport 0xabc/x.bend as X\n")
        self.assertIsNone(key(bend(TOOLS, hub, "--check-only"), self.root, self.env))
        native = [str(self.root / "a.out"), "--threads", "8"]
        self.assertIsNone(key(native, self.root, self.env))
        self.assertIsNone(key(bend(TOOLS, self.src / "law.bend"), self.root, self.env))

    def test_a_crash_is_not_stored(self):
        kernel = self.shell_kernel()
        self.assertEqual(self.run_kernel(kernel, "exit 1\n")[0].returncode, 1)
        self.assertEqual(len(self.entries()), 1)
        result, _ = self.run_kernel(kernel, "kill -9 $$\n")
        self.assertLess(result.returncode, 0)
        self.assertEqual(len(self.entries()), 1)

    def test_the_loaded_libraries_are_in_the_key(self):
        kernel = self.shell_kernel()
        artifact = self.root / "x.bendtt"
        artifact.write_text("")
        command = [str(kernel), str(artifact)]
        plain = key(command, self.root, self.env)
        library = self.root / "libpreload.so"
        libm = next(
            path
            for path in ("/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib64/libm.so.6")
            if Path(path).exists()
        )
        shutil.copy(libm, library)
        preload = {**self.env, "LD_PRELOAD": str(library)}
        first = key(command, self.root, preload)
        with library.open("ab") as stream:
            stream.write(b"\0")
        second = key(command, self.root, preload)
        self.assertEqual(len({json.dumps(k) for k in (plain, first, second)}), 3)

    def test_a_kernel_that_cannot_run_is_not_cached(self):
        kernel = self.shell_kernel()
        kernel.chmod(0o644)
        artifact = self.root / "x.bendtt"
        artifact.write_text("")
        self.assertIsNone(key([str(kernel), str(artifact)], self.root, self.env))
        law = self.src / "law.bend"
        unrunnable = {**self.env, "BENDTT": str(kernel)}
        self.assertIsNone(key(bend(TOOLS, law, "--verdict"), self.root, unrunnable))

    def test_telemetry_or_a_file_in_the_library_is_not_cached(self):
        law = self.src / "law.bend"
        command = bend(TOOLS, law, "--check-only")
        self.assertIsNotNone(key(command, self.root, self.env))
        telemetry = {k: v for k, v in self.env.items() if k != "BEND_NO_TELEMETRY"}
        self.assertIsNone(key(command, self.root, telemetry))
        inside = {**self.env, "BEND_LIB": str(self.src)}
        self.assertIsNone(key(command, self.root, inside))
        # An empty BEND_LIB is the working directory to Bend, not the default.
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        empty = key(command, elsewhere, {**self.env, "BEND_LIB": ""})
        assert empty is not None
        self.assertEqual(empty["runtime"]["library"][1], str(elsewhere))
        self.assertIsNone(key(command, self.root, {**self.env, "BEND_LIB": ""}))
        options = {**self.env, "BUN_OPTIONS": "--preload ./x.js"}
        self.assertIsNone(key(command, self.root, options))
        relative = {**self.env, "LD_PRELOAD": "./x.so"}
        (self.root / "x.so").write_bytes(
            Path(shutil.which("sh") or "/bin/sh").read_bytes()
        )
        self.assertIsNone(key(command, self.root, relative))

    def test_the_cli_exits_with_the_commands_code(self):
        law = self.src / "law.bend"
        law.write_text(LAW.replace("== 2n", "== 3n"))
        command = [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "tools.cache",
            "--cache",
            str(self.root / "cache"),
            "--",
            *map(str, bend(TOOLS, law, "--verdict")),
        ]
        for _ in range(2):
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, **self.env},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("SOME PROOFS FAIL", result.stdout + result.stderr)
        self.assertIn("cache: replayed", result.stderr)


if __name__ == "__main__":
    unittest.main()
