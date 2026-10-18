from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from collector import export as export_module
from collector.main import SCHEMA
from collector.storage import Storage


class ExportCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        storage = Storage(self.db, SCHEMA)
        try:
            storage.insert_samples([{"ts": 100, "boot_id": "a"}, {"ts": 200, "boot_id": "b"}])
        finally:
            storage.close()

    def cli(self, *options):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            export_module.main(["--db", str(self.db), *options])
        return stdout.getvalue(), stderr.getvalue()

    def test_stdout_contains_only_csv_and_filters_are_available(self):
        output, status = self.cli("--boot", "a", "--since", "0", "--until", "200", "--columns", "ts", "boot_id")
        self.assertEqual(list(csv.DictReader(io.StringIO(output))), [
            {"ts": "100", "boot_id": "a", "labels_json": "[]"},
        ])
        self.assertIn("1 mesure(s)", status)

    def test_new_output_file_can_be_read_as_csv(self):
        path = self.db.with_suffix(".csv")
        output, status = self.cli("--output", str(path))
        self.assertEqual(output, "")
        with path.open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertIn("2 mesure(s)", status)

    def test_existing_destination_and_source_database_are_never_overwritten(self):
        for path in (self.db, self.db.with_suffix(".csv")):
            if path != self.db:
                path.write_text("previous dataset")
            before = path.read_bytes()
            with self.subTest(path=path), self.assertRaises(SystemExit) as error:
                self.cli("--output", str(path))
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_options_do_not_leave_a_misleading_output_file(self):
        path = self.db.with_suffix(".csv")
        for options in (("--columns", "unknown"), ("--since", "10", "--until", "5")):
            with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                self.cli("--output", str(path), *options)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(path.exists())

    def test_partial_output_is_removed_on_failure_or_interruption(self):
        path = self.db.with_suffix(".csv")
        for exception in (OSError("destination full"), KeyboardInterrupt()):
            def fail(db, target, **options):
                target.write("incomplete data")
                raise exception

            with mock.patch.object(export_module, "export_csv", side_effect=fail):
                with self.subTest(exception=type(exception).__name__), \
                     self.assertRaises(SystemExit if isinstance(exception, OSError) else KeyboardInterrupt):
                    self.cli("--output", str(path))
            self.assertFalse(path.exists())

    def test_missing_database_creates_neither_a_database_nor_an_export(self):
        missing = self.db.with_name("missing.db")
        path = self.db.with_suffix(".csv")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            export_module.main(["--db", str(missing), "--output", str(path)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(missing.exists())
        self.assertFalse(path.exists())

    def test_cli_supports_jsonl_and_events_export_and_rejects_mixed_flags(self):
        with Storage(self.db, SCHEMA) as storage:
            storage.add_event(150, "kmsg", "mce", "error", "err")
        output, status = self.cli("--format", "jsonl", "--boot", "a")
        records = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ts"], 100)
        self.assertIn("1 mesure(s)", status)

        ev_out, ev_status = self.cli("--events", "--format", "jsonl", "--kind", "mce")
        ev_records = [json.loads(line) for line in ev_out.splitlines()]
        self.assertEqual(len(ev_records), 1)
        self.assertEqual(ev_records[0]["source"], "kmsg")
        self.assertIn("1 événement(s)", ev_status)

        for bad in (("--events", "--boot", "a"), ("--events", "--bucket-ms", "1000"), ("--source", "kmsg")):
            with self.subTest(bad=bad), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as err:
                export_module.main(["--db", str(self.db), *bad])
            self.assertEqual(err.exception.code, 2)

        ds_out, ds_status = self.cli("--bucket-ms", "100", "--format", "jsonl")
        self.assertEqual(len(ds_out.splitlines()), 2)
        self.assertIn("2 fenêtre(s)", ds_status)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as err:
            export_module.main(["--db", str(self.db), "--last-minutes", "5", "--since", "0"])
        self.assertEqual(err.exception.code, 2)
