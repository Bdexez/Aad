from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import detector as detector_module
from collector.detector import benchmark_configs, detect_anomalies, tune_threshold
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

    def test_control_for_scores_residuals_so_hot_under_load_is_normal_and_hot_at_idle_is_anomalous(self):
        # Relation normale : temp_c ≈ 40 + 40 * cpu_util ± 1°C
        self.storage.ensure_columns(["cpu_util"])
        baseline_rows = []
        loads = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.4, 0.6, 0.9, 0.2]
        noise = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
        for i, (u, eps) in enumerate(zip(loads, noise)):
            baseline_rows.append({
                "ts": 1000 * (i + 1), "boot_id": "a", "cpu_util": u, "temp_c": 40.0 + 40.0 * u + eps,
            })
        # Cible : 3 ticks chauds MAIS sous forte charge (normal !) puis 3 ticks chauds AU REPOS (anomalie !)
        target_rows = [
            {"ts": 20000, "boot_id": "a", "cpu_util": 1.0, "temp_c": 80.5},
            {"ts": 21000, "boot_id": "a", "cpu_util": 1.0, "temp_c": 79.5},
            {"ts": 22000, "boot_id": "a", "cpu_util": 1.0, "temp_c": 80.0},
            {"ts": 23000, "boot_id": "a", "cpu_util": 0.05, "temp_c": 72.0},
            {"ts": 24000, "boot_id": "a", "cpu_util": 0.05, "temp_c": 74.0},
            {"ts": 25000, "boot_id": "a", "cpu_util": 0.05, "temp_c": 73.0},
        ]
        self.storage.insert_samples(baseline_rows + target_rows)
        norm = self.storage.start_label(1000, "normal")
        self.storage.end_label(15000, norm)
        idle_hot = self.storage.start_label(23000, "cooling_fail")
        self.storage.end_label(26000, idle_hot)

        model_path = self.db.with_name("cond_model.json")
        res = detect_anomalies(
            self.db,
            baseline_label="normal",
            since=20000,
            columns=["temp_c"],
            control_for="cpu_util",
            eval_label="cooling_fail",
            threshold=4.0,
            min_samples=3,
            save_model=model_path,
        )
        self.assertEqual(len(res["episodes"]), 1)
        self.assertEqual(res["episodes"][0]["ts_start"], 23000)
        self.assertAlmostEqual(res["evaluation"]["f1"], 1.0)
        # Rechargement du modèle conditionnel
        reloaded = detect_anomalies(
            self.db, load_model=model_path, since=20000, eval_label="cooling_fail",
            threshold=4.0, min_samples=3,
        )
        self.assertAlmostEqual(reloaded["evaluation"]["f1"], 1.0)

    def test_tune_threshold_selects_optimal_f1_threshold(self):
        self.seed_experiment()
        tuned = tune_threshold(
            self.db,
            baseline_label="normal",
            eval_label="airflow_blocked",
            since=11000,
            thresholds=[1.0, 4.0, 50.0],
            min_samples=3,
        )
        self.assertEqual(tuned["best"]["threshold"], 4.0)
        self.assertAlmostEqual(tuned["best"]["f1"], 1.0)

        with redirect_stdout(io.StringIO()) as out:
            detector_module.main([
                "--db", str(self.db), "--baseline-label", "normal",
                "--since", "11000", "--eval-label", "airflow_blocked",
                "--tune-threshold", "--thresholds", "2.0", "4.0",
            ])
        self.assertIn("meilleur seuil", out.getvalue())

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            detector_module.main(["--db", str(self.db), "--baseline-label", "normal", "--tune-threshold"])
        self.assertEqual(exc.exception.code, 2)

    def test_group_by_fits_separate_regime_baselines_and_persists_in_model(self):
        self.storage.ensure_columns(["is_gaming"])
        # Référence : repos (is_gaming=0) ≈ 45°C ± 1, jeu (is_gaming=1) ≈ 75°C ± 1
        rows = [
            {"ts": 1000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 44.0},
            {"ts": 2000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 45.0},
            {"ts": 3000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 46.0},
            {"ts": 4000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 45.0},
            {"ts": 5000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 74.0},
            {"ts": 6000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 75.0},
            {"ts": 7000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 76.0},
            {"ts": 8000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 75.0},
            # Cible : 3 ticks à 75°C EN JEU (normal) puis 3 ticks à 75°C AU REPOS (anomalie)
            {"ts": 10000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 75.5},
            {"ts": 11000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 74.5},
            {"ts": 12000, "boot_id": "a", "is_gaming": 1.0, "temp_c": 75.0},
            {"ts": 13000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 75.0},
            {"ts": 14000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 76.0},
            {"ts": 15000, "boot_id": "a", "is_gaming": 0.0, "temp_c": 74.0},
        ]
        self.storage.insert_samples(rows)
        norm = self.storage.start_label(1000, "normal")
        self.storage.end_label(9000, norm)
        anom = self.storage.start_label(13000, "idle_hot")
        self.storage.end_label(16000, anom)

        model_path = self.db.with_name("regime_model.json")
        res = detect_anomalies(
            self.db, baseline_label="normal", since=10000, columns=["temp_c"],
            group_by="is_gaming", eval_label="idle_hot", threshold=4.0, min_samples=3,
            save_model=model_path,
        )
        self.assertEqual(len(res["episodes"]), 1)
        self.assertEqual(res["episodes"][0]["ts_start"], 13000)
        self.assertAlmostEqual(res["evaluation"]["f1"], 1.0)

        reloaded = detect_anomalies(
            self.db, load_model=model_path, since=10000, eval_label="idle_hot",
            threshold=4.0, min_samples=3,
        )
        self.assertAlmostEqual(reloaded["evaluation"]["f1"], 1.0)
        self.assertAlmostEqual(reloaded["evaluation"]["episode_f1"], 1.0)
        self.assertEqual(reloaded["evaluation"]["matched_gt_windows"], 1)

        tuned = tune_threshold(
            self.db, load_model=model_path, since=10000, eval_label="idle_hot",
            thresholds=[2.0, 4.0, 40.0], min_samples=3,
        )
        self.assertEqual(tuned["best"]["threshold"], 4.0)
        self.assertAlmostEqual(tuned["best"]["f1"], 1.0)

        out_file = self.db.with_name("episodes.json")
        detector_module.main([
            "--db", str(self.db), "--load-model", str(model_path),
            "--since", "10000", "--json", "--output", str(out_file),
        ])
        self.assertEqual(len(json.loads(out_file.read_text(encoding="utf-8"))["episodes"]), 1)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            detector_module.main([
                "--db", str(self.db), "--load-model", str(model_path),
                "--since", "10000", "--json", "--output", str(out_file),
            ])
        self.assertEqual(exc.exception.code, 2)

    def test_rms_score_mode_filters_single_sensor_glitch_and_catches_multivariate_shift(self):
        # 2 capteurs : temp_c (~50 ± 1) et fan_rpm (~1200 ± 10)
        self.seed_experiment()
        # Ajouter un épisode de 3 ticks où seul temp_c dévie de +5 sigma (max=5, rms=5/sqrt(2)≈3.53)
        # puis 3 ticks où les DEUX capteurs dévient de +5 sigma (max=5, rms=5.0).
        self.storage.insert_samples([
            {"ts": 20000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1200.0},
            {"ts": 21000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1200.0},
            {"ts": 22000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1200.0},
            {"ts": 23000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1275.0},
            {"ts": 24000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1275.0},
            {"ts": 25000, "boot_id": "b", "temp_c": 57.5, "fan_rpm": 1275.0},
        ])
        res_rms = detect_anomalies(
            self.db, baseline_label="normal", boot_id="b",
            score_mode="rms", threshold=4.2, min_samples=3,
        )
        self.assertEqual(res_rms["score_mode"], "rms")
        self.assertEqual(len(res_rms["episodes"]), 1)
        self.assertEqual(res_rms["episodes"][0]["ts_start"], 23000)

        with self.assertRaises(ValueError):
            detect_anomalies(self.db, baseline_label="normal", score_mode="invalid")

    def test_export_scores_writes_per_sample_csv_and_refuses_overwrite(self):
        import csv

        self.seed_experiment()
        scores_csv = self.db.with_name("scores.csv")
        detect_anomalies(
            self.db, baseline_label="normal", since=11000,
            threshold=4.0, min_samples=3, export_scores=scores_csv,
        )
        with scores_csv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[-1]["top_sensor"], "temp_c")
        self.assertEqual(rows[-1]["exceeds_threshold"], "1")
        with self.assertRaises(FileExistsError):
            detect_anomalies(self.db, baseline_label="normal", since=11000, export_scores=scores_csv)

    def test_roc_and_pr_auc_handle_perfect_tied_and_single_class_cases(self):
        from collector.detector import _roc_pr_auc

        self.assertEqual(_roc_pr_auc([(1.0, False), (2.0, False), (5.0, True), (6.0, True)]), (1.0, 1.0))
        roc_tied, _ = _roc_pr_auc([(3.0, False), (3.0, True)])
        self.assertAlmostEqual(roc_tied, 0.5)
        self.assertEqual(_roc_pr_auc([(1.0, False), (2.0, False)]), (None, None))

        self.seed_experiment()
        res = detect_anomalies(
            self.db, baseline_label="normal", since=11000, eval_label="airflow_blocked",
        )
        self.assertIsNotNone(res["evaluation"]["roc_auc"])
        self.assertIsNotNone(res["evaluation"]["pr_auc"])
        # 1 pic négatif isolé (90°C) devance les 4 positifs (82..88°C) -> AP exacte = (1/2+2/3+3/4+4/5)/4
        self.assertAlmostEqual(res["evaluation"]["pr_auc"], (0.5 + 2 / 3 + 0.75 + 0.8) / 4)

    def test_benchmark_configs_ranks_strategies_by_auc_and_f1(self):
        self.seed_experiment()
        bench = benchmark_configs(
            self.db, baseline_label="normal", since=11000,
            eval_label="airflow_blocked", thresholds=[2.0, 4.0], min_samples=3,
        )
        self.assertEqual(len(bench["runs"]), 2)
        self.assertIsNotNone(bench["best"])
        with redirect_stdout(io.StringIO()) as out:
            detector_module.main([
                "--db", str(self.db), "--baseline-label", "normal",
                "--since", "11000", "--eval-label", "airflow_blocked",
                "--benchmark", "--thresholds", "2.0", "4.0",
            ])
        self.assertIn("Benchmark (airflow_blocked", out.getvalue())


if __name__ == "__main__":
    unittest.main()
