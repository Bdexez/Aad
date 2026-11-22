from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import __main__ as cli_entry
from collector.main import SCHEMA
from collector.storage import Storage


class UnifiedCliEntryPointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        with Storage(self.db, SCHEMA) as storage:
            storage.ensure_columns(["temp_c"])
            storage.insert_samples([
                {"ts": 1000, "boot_id": "a", "temp_c": 42.0},
                {"ts": 2000, "boot_id": "a", "temp_c": 44.0},
            ])

    def test_help_lists_all_commands(self):
        with redirect_stdout(io.StringIO()) as out:
            cli_entry.main(["--help"])
        text = out.getvalue()
        for cmd in ("run", "doctor", "scan", "label", "report", "analyze", "detect", "export", "features", "backup"):
            self.assertIn(cmd, text)

    def test_dispatches_subcommands_to_modules(self):
        with redirect_stdout(io.StringIO()) as out:
            cli_entry.main(["report", "--db", str(self.db), "--json"])
        self.assertEqual(json.loads(out.getvalue())["samples"]["count"], 2)

        with redirect_stdout(io.StringIO()) as out:
            cli_entry.main(["backup", "--db", str(self.db), "--verify", "--json"])
        self.assertTrue(json.loads(out.getvalue())["ok"])

        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            cli_entry.main(["features", "--db", str(self.db), "--format", "jsonl", "--deltas"])
        self.assertEqual(len(out.getvalue().splitlines()), 2)

        with redirect_stdout(io.StringIO()) as out:
            cli_entry.main(["summary", "--db", str(self.db), "--json"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["healthy"])
        self.assertEqual(payload["samples"]["count"], 2)
        self.assertGreater(payload["storage_projection"]["estimated_mib_per_day"], 0)

    def test_unknown_command_exits_cleanly_with_code_2(self):
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as exc:
            cli_entry.main(["does-not-exist"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("commande inconnue", err.getvalue())


if __name__ == "__main__":
    unittest.main()
