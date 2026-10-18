"""The package check: what `bend bendtorch.bend --publish` would ship."""

import shutil
import tempfile
import unittest
from pathlib import Path

from tools.package import PackageError, check

ROOT = Path(__file__).resolve().parents[1]


class PackageTest(unittest.TestCase):
    def test_repository_package_ships_only_src_and_root_files(self) -> None:
        shipped = check(ROOT)
        self.assertIn("bendtorch.bend", shipped)
        self.assertIn("PROOF.bend", shipped)
        self.assertIn("LAWS.bend", shipped)
        self.assertIn("LICENSE", shipped)
        self.assertIn("src/autograd/tensor-backward.bend", shipped)
        self.assertIn("src/tensor/core-LAWS.bend", shipped)
        self.assertNotIn("src/tensor/matmul-demo.bend", shipped)
        for path in shipped:
            self.assertTrue(
                path.startswith("src/")
                or path in ("bendtorch.bend", "LAWS.bend", "PROOF.bend", "LICENSE"),
                path,
            )

    def mini(self, directory: Path, entry: str, changelog: str = "0.1.0.0") -> None:
        (directory / "src").mkdir()
        (directory / "src/a.bend").write_text("import Base\n")
        (directory / "bendtorch.bend").write_text(
            "# bendtorch: x; e.g. bendtorch@0.1.0.0/src/a.bend.\n" + entry
        )
        (directory / "LICENSE").write_text("SPDX-License-Identifier: MIT\n\nMIT\n")
        (directory / "CHANGELOG.md").write_text(f"# Changelog\n\n## {changelog} (x)\n")

    def test_each_refusal(self) -> None:
        cases = {
            "stray": ("import ./examples/x.bend as X\n", "outside src/"),
            "missing": ("import ./src/none.bend as N\n", "does not exist"),
            "climb": ("import ../outside.bend as O\n", "above the repository"),
        }
        for name, (entry, message) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "repo"
                root.mkdir()
                self.mini(root, "import ./src/a.bend as A\n" + entry)
                (root / "examples").mkdir()
                (root / "examples/x.bend").write_text("import Base\n")
                (Path(tmp) / "outside.bend").write_text("import Base\n")
                with self.assertRaisesRegex(PackageError, message):
                    check(root)

    def test_license_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.mini(root, "import ./src/a.bend as A\n")
            self.assertEqual(check(root), ["LICENSE", "bendtorch.bend", "src/a.bend"])
            (root / "LICENSE").write_text("MIT License\n")
            with self.assertRaisesRegex(PackageError, "SPDX"):
                check(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.mini(root, "import ./src/a.bend as A\n", changelog="0.2.0.0")
            with self.assertRaisesRegex(PackageError, "CHANGELOG.md 0.2.0.0"):
                check(root)

    def test_a_licence_beside_a_shipped_file_ships(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.mini(root, "import ./src/a.bend as A\n")
            shutil.copy(root / "LICENSE", root / "src/LICENSE")
            self.assertIn("src/LICENSE", check(root))


if __name__ == "__main__":
    unittest.main()
