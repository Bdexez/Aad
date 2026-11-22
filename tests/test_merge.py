from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import merge as merge_module
from collector.backup import verify_database
from collector.main import SCHEMA
from collector.merge import merge_databases
from collector.storage import Storage


class MergeDatabasesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.db1 = self.root / "part1.db"
        self.db2 = self.root / "part2.db"

        with Storage(self.db1, SCHEMA) as s1:
            s1.register_boot("b1", 1000, "6.12")
            s1.ensure_columns(["temp_c", "note"], {"note": "TEXT"})
            s1.insert_samples([{"ts": 1000, "boot_id": "b1", "temp_c": 45.0, "note": "a"}])
            s1.add_event(1000, "kmsg", "mce", "error", "err", "b1:1")
            w = s1.start_label(1000, "normal", "ref")
            s1.end_label(2000, w)

        with Storage(self.db2, SCHEMA) as s2:
            s2.register_boot("b1", 1000, "6.12")
            s2.register_boot("b2", 5000, "6.12")
            s2.ensure_columns(["temp_c", "fan_rpm"])
            s2.insert_samples([{"ts": 5000, "boot_id": "b2", "temp_c": 60.0, "fan_rpm": 1200.0}])
            # Même dedup -> ignoré ; nouvel événement -> inséré
            s2.add_event(1000, "kmsg", "mce", "error", "err", "b1:1")
            s2.add_event(5000, "kmsg", "thermal_trip", "critical", "hot", "b2:1")
            # Même label -> ignoré
            w = s2.start_label(1000, "normal", "ref")
            s2.end_label(2000, w)

    def test_merges_heterogeneous_schemas_and_deduplicates_boots_events_and_labels(self):
        dest = self.root / "merged.db"
        res = merge_databases([self.db1, self.db2], dest)
        self.assertEqual(res["samples"], 2)
        self.assertEqual(res["boots"], 2)
        self.assertEqual(res["events_inserted"], 2)
        self.assertEqual(res["events_skipped"], 1)
        self.assertEqual(res["labels_inserted"], 1)
        self.assertEqual(res["labels_skipped"], 1)
        self.assertEqual(res["columns"], ["fan_rpm", "note", "temp_c"])
        self.assertTrue(verify_database(dest)["ok"])

    def test_rejects_type_conflicts_and_existing_destination(self):
        conflict = self.root / "conflict.db"
        with Storage(conflict, SCHEMA) as sc:
            sc.ensure_columns(["temp_c"], {"temp_c": "TEXT"})
        dest = self.root / "out.db"
        with self.assertRaisesRegex(ValueError, "conflit de type"):
            merge_databases([self.db1, conflict], dest)
        self.assertFalse(dest.exists())

        dest.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            merge_databases([self.db1, self.db2], dest)

    def test_cli_json_and_validation(self):
        dest = self.root / "cli_merged.db"
        with redirect_stdout(io.StringIO()) as out:
            merge_module.main(["--output", str(dest), "--json", str(self.db1), str(self.db2)])
        self.assertEqual(json.loads(out.getvalue())["samples"], 2)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            merge_module.main(["--output", str(self.root / "x.db"), str(self.db1)])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
