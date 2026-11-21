from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest

from collector.database import read_snapshot, sample_filter_sql, validate_time_range
from collector.main import SCHEMA
from collector.storage import Storage


class ReadSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        self.storage = Storage(self.db, SCHEMA)
        self.addCleanup(self.storage.close)

    def test_read_connection_cannot_modify_the_dataset(self):
        with read_snapshot(self.db) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO samples (ts, boot_id) VALUES (1, 'a')")
        self.assertEqual(self.storage.count_samples(), 0)

    def test_snapshot_remains_stable_while_the_collector_writes(self):
        with read_snapshot(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 0)
            self.storage.insert_samples([{"ts": 1, "boot_id": "a"}])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 0)
        with read_snapshot(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)

    def test_exception_closes_the_snapshot_connection(self):
        with self.assertRaisesRegex(ValueError, "consumer failed"):
            with read_snapshot(self.db) as conn:
                raise ValueError("consumer failed")
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_missing_database_does_not_create_files_or_directories(self):
        missing = self.db.parent / "missing" / "dataset.db"
        with self.assertRaises(sqlite3.OperationalError):
            with read_snapshot(missing):
                self.fail("missing database was opened")
        self.assertFalse(missing.parent.exists())

    def test_validate_time_range_and_sample_filter_sql(self):
        validate_time_range(None, None)
        validate_time_range(10, 20)
        for since, until in ((10, 10), (20, 10)):
            with self.subTest(since=since, until=until), self.assertRaises(ValueError):
                validate_time_range(since, until)
        self.assertEqual(sample_filter_sql(), ("", []))
        self.assertEqual(
            sample_filter_sql(boot_id="b", since=10, until=20, alias="s"),
            (" WHERE s.boot_id = ? AND s.ts >= ? AND s.ts < ?", ["b", 10, 20]),
        )
