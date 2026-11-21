from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import detector as detector_module
from collector.detector import detect_anomalies
from collector.main import SCHEMA
from collector.storage import Storage


class BaselineDetectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        self.storage = Storage(self.db, SCHEMA)
        self.addCleanup(self.storage.close)
        self.storage.ensure_columns(["temp_c", "fan_rpm", "stuck", "note"], {"note": "TEXT"})

    def seed_experiment(self):
        # 10 mesures normales autour de 50°C / 1200 RPM
        rows = []
        temps = [48.0, 49.0, 50.0, 51.0, 52.0, 49.0, 50.0, 51.0, 48.0, 52.0]
        fans = [1180.0, 1190.0, 1200.0, 1210.0, 1220.0, 1195.0, 1205.0, 1200.0, 1185.0, 1215.0]
        for i, (t, f) in enumerate(zip(temps, fans)):
            rows.append({"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": t, "fan_rpm": f, "stuck": 5.0})
        # 1 pic isolé de 1 tick (doit être filtré par min_samples=3)
        rows.append({"ts": 11000, "boot_id": "a", "temp_c": 90.0, "fan_rpm": 1200.0, "stuck": 5.0})
        # 2 ticks normaux
        rows.append({"ts": 12000, "boot_id": "a", "temp_c": 50.0, "fan_rpm": 1200.0, "stuck": 5.0})
        rows.append({"ts": 13000, "boot_id": "a", "temp_c": 51.0, "fan_rpm": 1200.0, "stuck": 5.0})
        # Épisode d'anomalie thermique de 4 ticks (flux d'air obstrué)
        for i, t in enumerate((82.0, 85.0, 88.0, 86.0)):
            rows.append({"ts": 14000 + i * 1000, "boot_id": "a", "temp_c": t, "fan_rpm": 1200.0, "stuck": 5.0})
        self.storage.insert_samples(rows)

        norm = self.storage.start_label(1000, "normal")
        self.storage.end_label(11000, norm)
        anom = self.storage.start_label(14000, "airflow_blocked")
        self.storage.end_label(18000, anom)

    def test_detects_episode_explains_top_contributor_and_evaluates_f1(self):
        self.seed_experiment()
        res = detect_anomalies(
            self.db,
            baseline_label="normal",
            since=11000,
            eval_label="airflow_blocked",
            threshold=4.0,
            min_samples=3,
        )
        self.assertEqual(res["baseline"]["samples"], 10)
        self.assertEqual(res["baseline"]["features_used"], 2)
        self.assertEqual(res["baseline"]["features_skipped"], ["stuck"])
        self.assertEqual(len(res["episodes"]), 1)

        ep = res["episodes"][0]
        self.assertEqual(ep["boot_id"], "a")
        self.assertEqual(ep["ts_start"], 14000)
        self.assertEqual(ep["ts_end"], 17000)
        self.assertEqual(ep["sample_count"], 4)
        self.assertEqual(ep["top_contributors"][0]["column"], "temp_c")
        self.assertGreater(ep["top_contributors"][0]["peak_z"], 10.0)
        self.assertEqual(ep["top_contributors"][0]["observed_at_peak"], 88.0)

        ev = res["evaluation"]
        self.assertEqual((ev["tp"], ev["fp"], ev["fn"], ev["tn"]), (4, 0, 0, 3))
        self.assertAlmostEqual(ev["precision"], 1.0)
        self.assertAlmostEqual(ev["recall"], 1.0)
        self.assertAlmostEqual(ev["f1"], 1.0)

    def test_episodes_attach_correlated_events_within_margin(self):
        self.seed_experiment()
        self.storage.add_event(1000, "kmsg", "unrelated", "info", "too early")
        self.storage.add_event(15500, "kmsg", "thermal_trip", "critical", "core temp high")
        res = detect_anomalies(
            self.db, baseline_label="normal", since=11000, threshold=4.0, min_samples=3, event_margin_ms=2000,
        )
        events = res["episodes"][0]["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "thermal_trip")
        self.assertEqual(events[0]["ts"], 15500)

    def test_boot_boundary_splits_episodes_and_std_fallback_works_when_mad_is_zero(self):
        # 5 zéros et 1 valeur à 1.0 -> median=0, mad=0, std>0 (repli sur std).
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 0.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 0.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 0.0},
            {"ts": 4000, "boot_id": "a", "temp_c": 0.0},
            {"ts": 5000, "boot_id": "a", "temp_c": 1.0},
            # 2 pics sur boot a et 2 pics sur boot b -> séparés en 2 épisodes distincts.
            {"ts": 6000, "boot_id": "a", "temp_c": 10.0},
            {"ts": 7000, "boot_id": "a", "temp_c": 10.0},
            {"ts": 8000, "boot_id": "b", "temp_c": 10.0},
            {"ts": 9000, "boot_id": "b", "temp_c": 10.0},
        ])
        res = detect_anomalies(
            self.db,
            baseline_since=1000,
            baseline_until=6000,
            since=6000,
            threshold=3.0,
            min_samples=2,
        )
        self.assertEqual([ep["boot_id"] for ep in res["episodes"]], ["a", "b"])

    def test_validation_and_cli_modes(self):
        self.seed_experiment()
        with self.assertRaises(ValueError):
            detect_anomalies(self.db, baseline_label="unknown")
        with self.assertRaises(ValueError):
            detect_anomalies(self.db, baseline_label="normal", threshold=0.0)
        with self.assertRaises(ValueError):
            detect_anomalies(self.db, baseline_label="normal", columns=["note"])

        with redirect_stdout(io.StringIO()) as out:
            detector_module.main([
                "--db", str(self.db), "--baseline-label", "normal",
                "--since", "11000", "--eval-label", "airflow_blocked",
            ])
        self.assertIn("Épisodes  : 1", out.getvalue())
        self.assertIn("f1=1.000", out.getvalue())

        with redirect_stdout(io.StringIO()) as out:
            detector_module.main([
                "--db", str(self.db), "--baseline-label", "normal",
                "--target-label", "airflow_blocked", "--json",
            ])
        payload = json.loads(out.getvalue())
        self.assertEqual(len(payload["episodes"]), 1)

        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as exc:
            detector_module.main(["--db", str(self.db)])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("référence", err.getvalue())

    def test_save_and_load_baseline_model_roundtrip_and_no_overwrite(self):
        self.seed_experiment()
        model_path = self.db.with_name("baseline.json")
        first = detect_anomalies(
            self.db, baseline_label="normal", since=11000, save_model=model_path,
        )
        self.assertTrue(model_path.exists())
        # Refus d'écraser un modèle existant
        with self.assertRaises(FileExistsError):
            detect_anomalies(self.db, baseline_label="normal", since=11000, save_model=model_path)

        second = detect_anomalies(
            self.db, load_model=model_path, since=11000, eval_label="airflow_blocked",
        )
        self.assertEqual(len(second["episodes"]), len(first["episodes"]))
        self.assertAlmostEqual(second["evaluation"]["f1"], 1.0)
        self.assertEqual(second["baseline"]["loaded_from"], str(model_path))


if __name__ == "__main__":
    unittest.main()
