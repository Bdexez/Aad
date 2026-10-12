from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import pathlib
import tempfile
import unittest

from collector import html_report as html_module
from collector.html_report import render_html_report, write_html_report
from collector.main import SCHEMA
from collector.storage import Storage


class HtmlReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        self.storage = Storage(self.db, SCHEMA)
        self.addCleanup(self.storage.close)
        self.storage.ensure_columns(["temp_c", "stuck"])

    def test_renders_self_contained_html_with_svg_sparklines_and_escapes_title(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 42.0, "stuck": 1.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 55.0, "stuck": 1.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 48.0, "stuck": 1.0},
        ])
        self.storage.add_event(2000, "kmsg", "thermal_trip", "critical", "hot")
        doc = render_html_report(self.db, title="<script>alert(1)</script>")
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", doc)
        self.assertNotIn("<script>", doc)
        self.assertIn("<polyline", doc)
        self.assertIn("temp_c", doc)
        self.assertIn("constante", doc)
        self.assertIn("thermal_trip", doc)

    def test_write_and_cli_refuse_to_overwrite_existing_file(self):
        self.storage.insert_samples([{"ts": 1000, "boot_id": "a", "temp_c": 40.0}])
        out_path = self.db.with_suffix(".html")
        with redirect_stdout(io.StringIO()) as out:
            html_module.main(["--db", str(self.db), "--output", str(out_path)])
        self.assertTrue(out_path.exists())
        self.assertIn(str(out_path), out.getvalue())

        with self.assertRaises(FileExistsError):
            write_html_report(self.db, out_path)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            html_module.main(["--db", str(self.db), "--output", str(out_path)])
        self.assertEqual(exc.exception.code, 2)

    def test_embeds_detected_episodes_when_baseline_label_is_provided(self):
        rows = [
            {"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": 49.0 + (i % 3)} for i in range(6)
        ] + [
            {"ts": 10000 + 1000 * i, "boot_id": "a", "temp_c": 90.0} for i in range(4)
        ]
        self.storage.insert_samples(rows)
        norm = self.storage.start_label(1000, "normal")
        self.storage.end_label(7000, norm)
        doc = render_html_report(self.db, baseline_label="normal", since=8000, threshold=4.0, min_samples=3)
        self.assertIn("Épisodes anormaux détectés (1)", doc)
        self.assertIn("temp_c (z=", doc)


if __name__ == "__main__":
    unittest.main()
