from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import retention as retention_module
from collector.main import SCHEMA
from collector.retention import prune_boot, prune_database
from collector.storage import Storage


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        with Storage(self.db, SCHEMA) as storage:
            storage.register_boot("old-boot", 100, "6.12")
            storage.register_boot("new-boot", 500, "6.12")
            storage.insert_samples([
                {"ts": 100, "boot_id": "old-boot"},
                {"ts": 200, "boot_id": "old-boot"},
                {"ts": 500, "boot_id": "new-boot"},
            ])
            storage.add_event(150, "kmsg", "mce", "error", "old")
            storage.add_event(550, "kmsg", "mce", "error", "new")
            w = storage.start_label(100, "old_exp")
            storage.end_label(250, w)
            storage.start_label(500, "current_exp")

    def test_dry_run_counts_without_modifying_database(self):
        before = self.db.read_bytes()
        res = prune_database(self.db, before_ts=300, dry_run=True)
        self.assertTrue(res["dry_run"])
        self.assertEqual((res["samples"], res["events"], res["boots"], res["labels"]), (2, 1, 1, 0))
        self.assertEqual(self.db.read_bytes(), before)

    def test_apply_prunes_old_rows_preserves_labels_by_default_and_vacuums(self):
        res = prune_database(self.db, before_ts=300, dry_run=False, vacuum=True)
        self.assertFalse(res["dry_run"])
        self.assertTrue(res["vacuumed"])
        self.assertEqual((res["samples"], res["events"], res["boots"], res["labels"]), (2, 1, 1, 0))
        with Storage(self.db, SCHEMA) as storage:
            self.assertEqual(storage.count_samples(), 1)
            self.assertEqual(len(storage.list_labels(include_closed=True)), 2)

    def test_cli_dry_run_json_and_vacuum_validation(self):
        with redirect_stdout(io.StringIO()) as out:
            retention_module.main(["--db", str(self.db), "--before-ts", "300", "--json"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["samples"], 2)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            retention_module.main(["--db", str(self.db), "--before-ts", "300", "--vacuum"])
        self.assertEqual(exc.exception.code, 2)

    def test_prune_boot_removes_only_selected_boot_and_respects_dry_run(self):
        dry = prune_boot(self.db, boot_id="old-boot", dry_run=True)
        self.assertEqual((dry["samples"], dry["boots"]), (2, 1))
        applied = prune_boot(self.db, boot_id="old-boot", dry_run=False)
        self.assertEqual((applied["samples"], applied["boots"]), (2, 1))
        with Storage(self.db, SCHEMA) as storage:
            self.assertEqual(storage.count_samples(), 1)
            self.assertEqual(storage.conn.execute("SELECT boot_id FROM boots").fetchall(), [("new-boot",)])


if __name__ == "__main__":
    unittest.main()
