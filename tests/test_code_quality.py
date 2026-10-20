"""Garde-fous d'ingénierie logicielle : compatibilité Python 3.10+ et zéro dépendance tierce."""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest

COLLECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "collector"
OPTIONAL_DEPS = frozenset({"pynvml"})


class CodeQualityGuardTest(unittest.TestCase):
    def test_all_collector_modules_parse_under_python_3_10_syntax(self):
        py_files = sorted(COLLECTOR_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 10)
        for path in py_files:
            with self.subTest(file=path.relative_to(COLLECTOR_DIR.parent)):
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path), feature_version=(3, 10))

    def test_collector_imports_only_stdlib_at_top_level(self):
        stdlib = sys.stdlib_module_names | {"collector"}
        for path in sorted(COLLECTOR_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        self.assertIn(root, stdlib, f"dépendance non-stdlib {root!r} dans {path}")
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    root = node.module.split(".")[0]
                    self.assertIn(root, stdlib | OPTIONAL_DEPS, f"dépendance non-stdlib {root!r} dans {path}")


if __name__ == "__main__":
    unittest.main()
