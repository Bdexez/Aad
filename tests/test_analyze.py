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
from collector.analyze import (
    analyze_by_group,
    analyze_dataset,
    analyze_drift,
    audit_labels,
    check_physical_bounds,
    compare_boots,
    compare_labels,
    correlate_metric,
    detect_changepoints,
    detect_clipping,
    detect_collinearity,
    detect_missing_blocks,
    detect_outliers,
    detect_stuck_runs,
    lag_correlate,
    thermal_profile,
)
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

    def test_correlate_metric_computes_exact_pairwise_pearson_and_handles_nulls(self):
        self.storage.ensure_columns(["power_w", "fan_rpm"])
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "power_w": 20.0, "fan_rpm": 1200.0, "stuck": 5.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 50.0, "power_w": 30.0, "fan_rpm": 1000.0, "stuck": 5.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 60.0, "power_w": 40.0, "fan_rpm": 800.0, "stuck": 5.0},
            {"ts": 4000, "boot_id": "a", "temp_c": 70.0, "power_w": None, "fan_rpm": 600.0, "stuck": 5.0},
        ])
        res = correlate_metric(self.db, "temp_c")
        by_col = {item["column"]: item for item in res["correlations"]}
        self.assertAlmostEqual(by_col["power_w"]["r"], 1.0)
        self.assertEqual(by_col["power_w"]["pairs"], 3)
        self.assertAlmostEqual(by_col["fan_rpm"]["r"], -1.0)
        self.assertEqual(by_col["fan_rpm"]["pairs"], 4)
        self.assertIsNone(by_col["stuck"]["r"])
        self.assertNotIn("note", by_col)

        with self.assertRaises(ValueError):
            correlate_metric(self.db, "note")
        with self.assertRaises(ValueError):
            correlate_metric(self.db, "temp_c", min_pairs=1)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--correlate", "temp_c"])
        self.assertIn("Corrélations de Pearson avec temp_c", out.getvalue())
        self.assertIn("power_w : r=1.000", out.getvalue())

    def test_detect_outliers_flags_extreme_spikes_and_handles_zero_mad(self):
        values = [48.0, 49.0, 50.0, 50.0, 51.0, 52.0, 50.0, 49.0, 51.0, 120.0]
        self.storage.insert_samples([
            {"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": v, "stuck": 5.0}
            for i, v in enumerate(values)
        ])
        res = detect_outliers(self.db)
        self.assertEqual(res["metrics"][0]["column"], "temp_c")
        temp = res["metrics"][0]
        self.assertEqual(temp["present"], 10)
        self.assertEqual(temp["outlier_count"], 1)
        self.assertEqual(temp["max_outlier"], 120.0)
        self.assertAlmostEqual(temp["median"], 50.0)
        self.assertGreater(temp["mad"], 0.0)

        stuck = res["metrics"][1]
        self.assertEqual(stuck["mad"], 0.0)
        self.assertEqual(stuck["outlier_count"], 0)

        with self.assertRaises(ValueError):
            detect_outliers(self.db, z_threshold=0.0)
        with self.assertRaises(ValueError):
            detect_outliers(self.db, columns=["note"])

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--outliers", "--columns", "temp_c"])
        self.assertIn("outliers=1/10", out.getvalue())

    def test_audit_labels_detects_empty_open_sampleless_and_cross_overlaps(self):
        self.storage.insert_samples([
            {"ts": 100, "boot_id": "a", "temp_c": 40.0},
            {"ts": 150, "boot_id": "a", "temp_c": 50.0},
        ])
        w_normal = self.storage.start_label(100, "normal")
        self.storage.end_label(200, w_normal)
        w_hot = self.storage.start_label(150, "airflow_blocked")
        self.storage.end_label(250, w_hot)
        w_empty = self.storage.start_label(300, "oops")
        self.storage.end_label(300, w_empty)
        w_open = self.storage.start_label(500, "ongoing")

        audit = audit_labels(self.db)
        self.assertEqual(audit["total_windows"], 4)
        self.assertEqual([w["id"] for w in audit["empty_windows"]], [w_empty])
        self.assertEqual([w["id"] for w in audit["open_windows"]], [w_open])
        self.assertEqual([w["id"] for w in audit["sampleless_windows"]], [w_open])
        self.assertEqual(len(audit["overlaps"]), 1)
        self.assertFalse(audit["overlaps"][0]["same_label"])
        self.assertEqual(audit["overlaps"][0]["overlap_start"], 150)
        self.assertEqual(audit["overlaps"][0]["overlap_end"], 200)
        self.assertEqual(audit["cross_label_overlap_samples"], 1)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--check-labels"])
        self.assertIn("Fenêtres vides : 1", out.getvalue())
        self.assertIn("1 mesure(s) multi-labels", out.getvalue())

    def test_analyze_drift_detects_upward_trend_between_first_and_last_slices(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "stuck": 5.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 42.0, "stuck": 5.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 44.0, "stuck": 5.0},
            {"ts": 4000, "boot_id": "a", "temp_c": 60.0, "stuck": 5.0},
            {"ts": 5000, "boot_id": "a", "temp_c": 62.0, "stuck": 5.0},
            {"ts": 6000, "boot_id": "a", "temp_c": 64.0, "stuck": 5.0},
        ])
        drift = analyze_drift(self.db, slices=2)
        self.assertEqual(drift["first_slice"]["samples"], 3)
        self.assertEqual(drift["last_slice"]["samples"], 3)
        self.assertEqual(drift["metrics"][0]["column"], "temp_c")
        self.assertAlmostEqual(drift["metrics"][0]["delta_mean"], 20.0)
        self.assertAlmostEqual(drift["metrics"][0]["z_drift"], 10.0)
        self.assertIsNone(drift["metrics"][1]["z_drift"])

        with self.assertRaises(ValueError):
            analyze_drift(self.db, slices=1)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--drift", "--slices", "2"])
        self.assertIn("Dérive temporelle", out.getvalue())
        self.assertIn("temp_c", out.getvalue())

    def test_analyze_by_group_splits_stats_by_context_including_null(self):
        self.storage.ensure_columns(["is_gaming"])
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 42.0},
            {"ts": 2000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 44.0},
            {"ts": 3000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 74.0},
            {"ts": 4000, "boot_id": "a", "is_gaming": None, "temp_c": 50.0},
        ])
        res = analyze_by_group(self.db, "is_gaming", columns=["temp_c"])
        by_val = {g["value"]: g for g in res["groups"]}
        self.assertAlmostEqual(by_val[0.0]["metrics"][0]["mean"], 43.0)
        self.assertAlmostEqual(by_val[1.0]["metrics"][0]["mean"], 74.0)
        self.assertAlmostEqual(by_val[None]["metrics"][0]["mean"], 50.0)

        with self.assertRaises(ValueError):
            analyze_by_group(self.db, "ts", max_groups=2)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--by", "is_gaming", "--columns", "temp_c"])
        self.assertIn("[0.0]", out.getvalue())
        self.assertIn("[NULL]", out.getvalue())

    def test_check_physical_bounds_flags_implausible_sensor_readings(self):
        self.storage.ensure_columns(["cpu_util", "k10temp_tctl", "rapl_pkg_watts"])
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "cpu_util": 0.5, "k10temp_tctl": 60.0, "rapl_pkg_watts": 45.0},
            {"ts": 2000, "boot_id": "a", "cpu_util": 1.5, "k10temp_tctl": 250.0, "rapl_pkg_watts": -5.0},
        ])
        res = check_physical_bounds(self.db)
        self.assertEqual(res["checked_metrics"], 3)
        self.assertEqual(res["total_violations"], 3)
        by_col = {v["column"]: v for v in res["violations"]}
        self.assertEqual(by_col["cpu_util"]["above_max"], 1)
        self.assertEqual(by_col["k10temp_tctl"]["above_max"], 1)
        self.assertEqual(by_col["rapl_pkg_watts"]["below_min"], 1)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--check-bounds"])
        self.assertIn("3 violation(s)", out.getvalue())

    def test_detect_stuck_runs_identifies_frozen_sensor_segments_per_boot(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 55.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 55.0},
            {"ts": 4000, "boot_id": "a", "temp_c": 55.0},
            {"ts": 5000, "boot_id": "a", "temp_c": 55.0},
            {"ts": 6000, "boot_id": "b", "temp_c": 55.0},
            {"ts": 7000, "boot_id": "b", "temp_c": 55.0},
        ])
        res = detect_stuck_runs(self.db, min_run=3, columns=["temp_c"])
        self.assertEqual(len(res["stuck_metrics"]), 1)
        m = res["stuck_metrics"][0]
        self.assertEqual(m["longest_run"], 4)
        self.assertEqual(m["longest_value"], 55.0)
        self.assertEqual(m["longest_boot"], "a")

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--stuck-runs", "--min-run", "3", "--columns", "temp_c"])
        self.assertIn("max=4", out.getvalue())

    def test_detect_changepoints_locates_step_shift(self):
        rows = [
            {"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": 45.0 + (i % 3) * 0.5} for i in range(10)
        ] + [
            {"ts": 1000 * (i + 11), "boot_id": "a", "temp_c": 75.0 + (i % 3) * 0.5} for i in range(10)
        ]
        self.storage.insert_samples(rows)
        res = detect_changepoints(self.db, "temp_c", window=5, threshold=4.0)
        self.assertEqual(len(res["changepoints"]), 1)
        cp = res["changepoints"][0]
        self.assertEqual(cp["ts"], 11000)
        self.assertAlmostEqual(cp["delta"], 30.0, delta=1.0)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--changepoints", "temp_c", "--window", "5"])
        self.assertIn("ts=11000", out.getvalue())

    def test_detect_collinearity_finds_redundant_sensor_pairs(self):
        self.storage.ensure_columns(["temp_dup", "noise"])
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "temp_dup": 42.0, "noise": 1.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 50.0, "temp_dup": 52.0, "noise": 5.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 60.0, "temp_dup": 62.0, "noise": 2.0},
            {"ts": 4000, "boot_id": "a", "temp_c": 70.0, "temp_dup": 72.0, "noise": 4.0},
        ])
        res = detect_collinearity(self.db, r_threshold=0.95)
        self.assertEqual(len(res["collinear_pairs"]), 1)
        pair = res["collinear_pairs"][0]
        self.assertEqual((pair["column_a"], pair["column_b"]), ("temp_c", "temp_dup"))
        self.assertAlmostEqual(pair["r"], 1.0)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--collinear"])
        self.assertIn("temp_c ↔ temp_dup", out.getvalue())

    def test_markdown_output_renders_gfm_tables_and_rejects_submodes(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 42.0, "stuck": 1.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 44.0, "stuck": 1.0},
        ])
        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--markdown"])
        md = out.getvalue()
        self.assertIn("# Analyse du dataset AAD (2 mesures)", md)
        self.assertIn("| `temp_c` |", md)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            analyze_module.main(["--db", str(self.db), "--drift", "--markdown"])
        self.assertEqual(exc.exception.code, 2)

    def test_detect_missing_blocks_identifies_contiguous_null_dropouts(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 45.0},
            {"ts": 2000, "boot_id": "a", "temp_c": None},
            {"ts": 3000, "boot_id": "a", "temp_c": None},
            {"ts": 4000, "boot_id": "a", "temp_c": None},
            {"ts": 5000, "boot_id": "a", "temp_c": 46.0},
            {"ts": 6000, "boot_id": "b", "temp_c": None},
            {"ts": 7000, "boot_id": "b", "temp_c": None},
        ])
        res = detect_missing_blocks(self.db, min_run=3, columns=["temp_c"])
        self.assertEqual(len(res["missing_blocks"]), 1)
        b = res["missing_blocks"][0]
        self.assertEqual(b["longest_block"], 3)
        self.assertEqual(b["longest_boot"], "a")

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--missing-blocks", "--min-missing", "3", "--columns", "temp_c"])
        self.assertIn("max=3", out.getvalue())

    def test_lag_correlate_detects_thermal_response_delay(self):
        self.storage.ensure_columns(["power_w"])
        # temp_c[t] suit exactement power_w[t - 2] (latence thermique de 2 ticks)
        powers = [20.0, 20.0, 80.0, 90.0, 40.0, 100.0, 30.0, 70.0, 25.0, 85.0]
        rows = []
        for i, p in enumerate(powers):
            t = powers[i - 2] if i >= 2 else 20.0
            rows.append({"ts": 1000 * (i + 1), "boot_id": "a", "power_w": p, "temp_c": t})
        self.storage.insert_samples(rows)
        res = lag_correlate(self.db, "temp_c", "power_w", max_lag=4)
        self.assertEqual(res["best"]["lag"], 2)
        self.assertAlmostEqual(res["best"]["r"], 1.0)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--lag-correlate", "temp_c", "power_w", "--max-lag", "4"])
        self.assertIn("meilleur lag=2", out.getvalue())

    def test_detect_clipping_flags_sensors_pinned_at_ceiling_and_ignores_constants(self):
        # temp_c plafonne à 95°C sur 50 % des ticks ; stuck est constante (ignorée).
        values = [50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 95.0, 95.0, 95.0, 95.0]
        self.storage.insert_samples([
            {"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": v, "stuck": 5.0}
            for i, v in enumerate(values)
        ])
        res = detect_clipping(self.db, min_pct=40.0)
        self.assertEqual([m["column"] for m in res["clipped_metrics"]], ["temp_c"])
        self.assertAlmostEqual(res["clipped_metrics"][0]["at_max_pct"], 50.0)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--clipping", "--clip-pct", "40"])
        self.assertIn("50.0 %", out.getvalue())

    def test_thermal_profile_and_compare_boots_quantify_cooling_degradation(self):
        self.storage.ensure_columns(["power_w"])
        # Boot A (propre) : temp = 30 + 0.5 * power ; Boot B (encrassé) : temp = 30 + 0.9 * power
        rows = [
            {"ts": 1000, "boot_id": "clean", "power_w": 20.0, "temp_c": 40.0},
            {"ts": 2000, "boot_id": "clean", "power_w": 40.0, "temp_c": 50.0},
            {"ts": 3000, "boot_id": "clean", "power_w": 60.0, "temp_c": 60.0},
            {"ts": 4000, "boot_id": "clogged", "power_w": 20.0, "temp_c": 48.0},
            {"ts": 5000, "boot_id": "clogged", "power_w": 40.0, "temp_c": 66.0},
            {"ts": 6000, "boot_id": "clogged", "power_w": 60.0, "temp_c": 84.0},
        ]
        self.storage.insert_samples(rows)
        prof = thermal_profile(self.db, "temp_c", "power_w")
        by_boot = {b["boot_id"]: b for b in prof["per_boot"]}
        self.assertAlmostEqual(by_boot["clean"]["r_th_c_per_w"], 0.5)
        self.assertAlmostEqual(by_boot["clean"]["t_idle_c"], 30.0)
        self.assertAlmostEqual(by_boot["clogged"]["r_th_c_per_w"], 0.9)

        diff = compare_boots(self.db, "clean", "clogged")
        self.assertEqual(diff["metrics"][0]["column"], "temp_c")
        self.assertAlmostEqual(diff["metrics"][0]["delta_mean"], 16.0)

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--thermal-profile", "temp_c", "power_w"])
        self.assertIn("R_th=0.500 °C/W", out.getvalue())

        with redirect_stdout(io.StringIO()) as out:
            analyze_module.main(["--db", str(self.db), "--compare-boots", "clean", "clogged"])
        self.assertIn("Comparaison boots : clean", out.getvalue())


if __name__ == "__main__":
    unittest.main()
