from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import pathlib
import sqlite3
import tempfile
import unittest

from collector import analyze as analyze_module
from collector.analyze import analyze_dataset, compare_labels
from collector.main import SCHEMA
from collector.storage import Storage


class AnalyzeDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        self.storage = Storage(self.db, SCHEMA)
        self.addCleanup(self.storage.close)
        self.storage.ensure_columns(["temp_c", "stuck", "note"], {"note": "TEXT"})

    def test_numeric_stats_compute_exact_mean_std_and_flag_constant_sensors(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "stuck": 5.0, "note": "x"},
            {"ts": 2000, "boot_id": "a", "temp_c": 50.0, "stuck": 5.0, "note": "y"},
            {"ts": 3000, "boot_id": "a", "temp_c": 60.0, "stuck": 5.0},
            {"ts": 4000, "boot_id": "a", "temp_c": None, "stuck": 5.0},
        ])
        report = analyze_dataset(self.db)
        by_col = {m["column"]: m for m in report["metrics"]}
        # Les colonnes TEXT sont exclues des statistiques numériques.
        self.assertNotIn("note", by_col)
        temp = by_col["temp_c"]
        self.assertEqual(temp["present"], 3)
        self.assertEqual(temp["missing"], 1)
        self.assertAlmostEqual(temp["coverage_pct"], 75.0)
        self.assertEqual(temp["min"], 40.0)
        self.assertEqual(temp["max"], 60.0)
        self.assertAlmostEqual(temp["mean"], 50.0)
        self.assertAlmostEqual(temp["std"], 10.0)
        self.assertFalse(temp["is_constant"])

        stuck = by_col["stuck"]
        self.assertTrue(stuck["is_constant"])
        self.assertAlmostEqual(stuck["std"], 0.0)

    def test_cadence_detects_gaps_duplicates_and_clock_regressions(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a"},
            {"ts": 2000, "boot_id": "a"},
            {"ts": 3000, "boot_id": "a"},
            {"ts": 3000, "boot_id": "a"},  # doublon
            {"ts": 2500, "boot_id": "a"},  # retour d'horloge
            {"ts": 3500, "boot_id": "a"},
            {"ts": 8500, "boot_id": "a"},  # trou de 5000 ms (> 2 * 1000 ms)
            {"ts": 100, "boot_id": "b"},   # boot à 1 seul sample
        ])
        report = analyze_dataset(self.db)
        cadence = {item["boot_id"]: item for item in report["cadence"]}
        boot_a = cadence["a"]
        self.assertEqual(boot_a["samples"], 7)
        self.assertEqual(boot_a["intervals"], 6)
        self.assertEqual(boot_a["duplicate_timestamps"], 1)
        self.assertEqual(boot_a["clock_regressions"], 1)
        self.assertEqual(boot_a["min_dt_ms"], 1000)
        self.assertAlmostEqual(boot_a["median_dt_ms"], 1000.0)
        self.assertEqual(boot_a["max_dt_ms"], 5000)
        self.assertEqual(boot_a["gaps"], 1)

        boot_b = cadence["b"]
        self.assertEqual(boot_b["samples"], 1)
        self.assertEqual(boot_b["intervals"], 0)
        self.assertIsNone(boot_b["median_dt_ms"])
        self.assertEqual(boot_b["gaps"], 0)

    def test_empty_and_single_value_datasets_return_none_for_undefined_stats(self):
        empty = analyze_dataset(self.db)
        self.assertEqual(empty["samples"], 0)
        self.assertEqual(empty["cadence"], [])
        self.assertTrue(all(m["std"] is None and m["mean"] is None for m in empty["metrics"]))

        self.storage.insert_samples([{"ts": 1000, "boot_id": "a", "temp_c": 42.0}])
        single = analyze_dataset(self.db)
        temp = next(m for m in single["metrics"] if m["column"] == "temp_c")
        self.assertEqual(temp["mean"], 42.0)
        self.assertIsNone(temp["std"])
        self.assertTrue(temp["is_constant"])

    def test_label_summary_counts_windows_samples_events_and_deduplicates_overlaps(self):
        self.storage.insert_samples([
            {"ts": 100, "boot_id": "a", "temp_c": 40.0},
            {"ts": 150, "boot_id": "a", "temp_c": 41.0},
            {"ts": 250, "boot_id": "a", "temp_c": 70.0},
        ])
        w1 = self.storage.start_label(100, "normal")
        self.storage.end_label(200, w1)
        w2 = self.storage.start_label(120, "normal")
        self.storage.end_label(180, w2)
        self.storage.start_label(200, "airflow_blocked")
        self.storage.add_event(150, "kmsg", "mce", "error")
        self.storage.add_event(250, "kmsg", "thermal_trip", "critical")

        report = analyze_dataset(self.db)
        labels = {item["label"]: item for item in report["labels"]}
        self.assertEqual(labels["normal"], {
            "label": "normal", "windows": 2, "open_windows": 0,
            "closed_duration_ms": 160, "samples": 2, "events": 1,
        })
        self.assertEqual(labels["airflow_blocked"], {
            "label": "airflow_blocked", "windows": 1, "open_windows": 1,
            "closed_duration_ms": 0, "samples": 1, "events": 1,
        })

    def test_compare_labels_ranks_metrics_by_standardized_shift(self):
        self.storage.insert_samples([
            {"ts": 100, "boot_id": "a", "temp_c": 48.0, "stuck": 10.0},
            {"ts": 200, "boot_id": "a", "temp_c": 50.0, "stuck": 10.0},
            {"ts": 300, "boot_id": "a", "temp_c": 52.0, "stuck": 10.0},
            {"ts": 400, "boot_id": "a", "temp_c": 70.0, "stuck": 10.0},
            {"ts": 500, "boot_id": "a", "temp_c": 74.0, "stuck": 10.0},
        ])
        base = self.storage.start_label(100, "normal")
        self.storage.end_label(350, base)
        target = self.storage.start_label(300, "airflow_blocked")
        self.storage.end_label(600, target)

        diff = compare_labels(self.db, "normal", "airflow_blocked")
        self.assertEqual(diff["baseline"], {"label": "normal", "samples": 3})
        self.assertEqual(diff["target"], {"label": "airflow_blocked", "samples": 3})
        self.assertEqual(diff["overlap_samples"], 1)
        # temp_c a un z_shift défini et passe devant stuck (baseline_std=0 -> z_shift=None).
        self.assertEqual(diff["metrics"][0]["column"], "temp_c")
        self.assertAlmostEqual(diff["metrics"][0]["baseline_mean"], 50.0)
        self.assertAlmostEqual(diff["metrics"][0]["baseline_std"], 2.0)
        self.assertAlmostEqual(diff["metrics"][0]["target_mean"], (52.0 + 70.0 + 74.0) / 3)
        self.assertAlmostEqual(diff["metrics"][0]["z_shift"], ((52.0 + 70.0 + 74.0) / 3 - 50.0) / 2.0)
        self.assertIsNone(diff["metrics"][1]["z_shift"])

    def test_legacy_database_is_analyzed_without_modification(self):
        legacy = self.db.with_name("legacy.db")
        with sqlite3.connect(legacy) as conn:
            conn.executescript(
                "CREATE TABLE samples (ts INTEGER PRIMARY KEY, boot_id TEXT NOT NULL, temp REAL);"
                "CREATE TABLE labels (id INTEGER PRIMARY KEY, ts_start INTEGER, ts_end INTEGER, label TEXT, notes TEXT);"
                "CREATE TABLE events (ts INTEGER, source TEXT, kind TEXT, severity TEXT);"
                "INSERT INTO samples VALUES (1000, 'b', 40.0), (2000, 'b', 50.0);"
            )
        conn.close()
        before = legacy.read_bytes()
        report = analyze_dataset(legacy)
        self.assertEqual(report["samples"], 2)
        self.assertAlmostEqual(report["metrics"][0]["mean"], 45.0)
        self.assertAlmostEqual(report["metrics"][0]["std"], math.sqrt(50.0))
        self.assertEqual(legacy.read_bytes(), before)

    def test_validation_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            analyze_dataset(self.db, since=200, until=100)
        with self.assertRaises(ValueError):
            analyze_dataset(self.db, gap_factor=1.0)
        with self.assertRaises(ValueError):
            compare_labels(self.db, " ", "target")

    def test_cli_text_and_json_modes(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "stuck": 1.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 50.0, "stuck": 1.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 70.0, "stuck": 1.0},
        ])
        w = self.storage.start_label(1000, "normal")
        self.storage.end_label(2500, w)
        self.storage.start_label(2500, "hot")

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--json"])
        self.assertEqual(json.loads(out.getvalue())["samples"], 3)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db)])
        self.assertIn("Mesures analysées : 3", out.getvalue())
        self.assertIn("[constante]", out.getvalue())

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--compare", "normal", "hot"])
        self.assertIn("Comparaison : normal", out.getvalue())
        self.assertIn("temp_c", out.getvalue())

        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as exc:
            analyze_module.main(["--db", str(self.db), "--gap-factor", "0.5"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--gap-factor", err.getvalue())


if __name__ == "__main__":
    unittest.main()
