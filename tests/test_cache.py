import json
import os
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
        self.cache = Cache(self.root / "cache")
        self.env = environment(TOOLS)

    def run_bend(self, *args, env=None):
        return self.cache.run(
            bend(TOOLS, *args), cwd=self.root, env=env or self.env, timeout=300
        )

    def entries(self) -> list[Path]:
        return sorted((self.root / "cache").rglob("*.json"))

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

    def test_the_bend_commit_and_its_clean_tree_are_in_the_key(self):
        fake = self.root / "bend"
        (fake / "bend2").mkdir(parents=True)
        (fake / "bend2/main.ts").write_text("// one\n")
        git = ["git", "-C", str(fake), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "commit", "-qm", "one"], check=True)
        command = ["bun", fake / "bend2/main.ts", self.src / "law.bend", "--check-only"]
        first = key(command, self.root, self.env)
        assert first is not None
        (fake / "bend2/main.ts").write_text("// two\n")
        # A fresh process sees the dirty checkout and does not cache.
        probe = (
            "import sys; from pathlib import Path; from tools.cache import key; "
            "import os; print(key(sys.argv[1:], Path('.'), dict(os.environ)))"
        )
        python = [str(ROOT / ".venv/bin/python"), "-c", probe, *map(str, command)]
        out = subprocess.run(
            python, cwd=ROOT, env=self.env, capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "None")
        subprocess.run([*git, "commit", "-qam", "two"], check=True)
        out = subprocess.run(
            python, cwd=ROOT, env=self.env, capture_output=True, text=True, check=True
        )
        self.assertNotEqual(out.stdout.strip(), "None")
        self.assertNotIn(first["bend"], out.stdout)

    def test_the_kernel_binary_is_in_the_key(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        fake = self.root / "bendtt"
        fake.write_bytes((TOOLS / "kernel/bendtt").read_bytes() + b"\0")
        fake.chmod(0o755)
        result, replayed = self.run_bend(
            law, "--verdict", env={**self.env, "BENDTT": str(fake)}
        )
        self.assertFalse(replayed)
        self.assertEqual(result.returncode, 0)

    def test_an_export_and_its_kernel_check_replay(self):
        law = self.src / "law.bend"
        artifact = self.root / "out/law.bendtt"
        artifact.parent.mkdir()
        self.run_bend(law, "-o", artifact)
        text = artifact.read_text()
        artifact.unlink()
        self.assertTrue(self.run_bend(law, "-o", artifact)[1])
        self.assertEqual(artifact.read_text(), text)
        kernel = [str(TOOLS / "kernel/bendtt"), str(artifact)]
        run = lambda: self.cache.run(kernel, cwd=self.root, env=self.env, timeout=300)  # noqa: E731
        self.assertFalse(run()[1])
        self.assertTrue(run()[1])
        # A changed artifact is a different kernel input.
        artifact.write_text(text + "\n")
        self.assertFalse(run()[1])

    def test_a_tampered_or_broken_entry_is_run_again(self):
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        (entry,) = self.entries()
        stored = json.loads(entry.read_text())
        stored["exit_code"] = 0
        stored["stdout"] = "ALL PROOFS CHECK\nforged\n"
        entry.write_text(json.dumps(stored))
        result, replayed = self.run_bend(law, "--verdict")
        self.assertFalse(replayed)
        self.assertNotIn("forged", result.stdout)
        self.assertEqual(self.cache.summary()["rejected"], [entry.name])
        # The rerun replaced the entry, which now replays.
        self.assertTrue(self.run_bend(law, "--verdict")[1])
        entry.write_text("{not json")
        self.assertFalse(self.run_bend(law, "--verdict")[1])

    def test_a_forged_pass_for_a_false_law_is_not_replayed(self):
        # An entry moved under another key, with a valid digest of its own,
        # still names the key it was made for.
        law = self.src / "law.bend"
        self.run_bend(law, "--verdict")
        (good,) = self.entries()
        law.write_text(LAW.replace("== 2n", "== 3n"))
        self.run_bend(law, "--verdict")
        (bad,) = [path for path in self.entries() if path != good]
        bad.write_text(good.read_text())
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
        script = self.root / "bendtt"
        script.write_text("#!/bin/sh\nkill -9 $$\n")
        script.chmod(0o755)
        artifact = self.root / "x.bendtt"
        artifact.write_text("x")
        result, _ = self.cache.run(
            [str(script), str(artifact)], cwd=self.root, env=self.env, timeout=60
        )
        self.assertLess(result.returncode, 0)
        self.assertEqual(self.entries(), [])

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
