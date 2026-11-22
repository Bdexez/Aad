"""Test d'intégration bout-en-bout de toute la chaîne AAD sur un dataset multi-boots."""

from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest

from collector.analyze import (
    analyze_by_group,
    analyze_dataset,
    analyze_drift,
    audit_labels,
    check_physical_bounds,
    compare_labels,
    correlate_metric,
    detect_outliers,
)
from collector.backup import backup_database, verify_database
from collector.detector import detect_anomalies, tune_threshold
from collector.export import export_csv, export_downsampled, export_events, export_jsonl
from collector.features import export_features
from collector.html_report import write_html_report
from collector.main import SCHEMA
from collector.report import dataset_report
from collector.storage import Storage


class EndToEndPipelineTest(unittest.TestCase):
    def test_full_multiboot_campaign_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            db = root / "campaign.db"
            with Storage(db, SCHEMA) as storage:
                storage.register_boot("boot-1", 1000, "6.12-arch1")
                storage.register_boot("boot-2", 50000, "6.12-arch1")
                storage.ensure_columns(
                    ["cpu_util", "k10temp_tctl", "rapl_pkg_watts", "is_gaming", "gpu_throttle_mask", "active_window"],
                    {"active_window": "TEXT"},
                )
                rows = []
                # Boot 1 : 20 ticks de référence (repos + jeu)
                for i in range(10):
                    u = 0.05 + 0.01 * (i % 3)
                    rows.append({
                        "ts": 1000 + i * 1000, "boot_id": "boot-1",
                        "cpu_util": u, "k10temp_tctl": 42.0 + 30.0 * u + (i % 2 - 0.5),
                        "rapl_pkg_watts": 15.0 + 40.0 * u, "is_gaming": 0.0,
                        "gpu_throttle_mask": 0x01, "active_window": "kitty|shell",
                    })
                for i in range(10):
                    u = 0.80 + 0.02 * (i % 3)
                    rows.append({
                        "ts": 11000 + i * 1000, "boot_id": "boot-1",
                        "cpu_util": u, "k10temp_tctl": 42.0 + 30.0 * u + (i % 2 - 0.5),
                        "rapl_pkg_watts": 15.0 + 40.0 * u, "is_gaming": 1.0,
                        "gpu_throttle_mask": 0x04, "active_window": "steam_app|game",
                    })
                # Boot 2 : 5 ticks normaux puis 5 ticks d'anomalie thermique au repos
                for i in range(5):
                    rows.append({
                        "ts": 50000 + i * 1000, "boot_id": "boot-2",
                        "cpu_util": 0.05, "k10temp_tctl": 43.0 + (i % 3) * 0.5, "rapl_pkg_watts": 17.0,
                        "is_gaming": 0.0, "gpu_throttle_mask": 0x01, "active_window": "kitty|shell",
                    })
                for i in range(5):
                    rows.append({
                        "ts": 55000 + i * 1000, "boot_id": "boot-2",
                        "cpu_util": 0.05, "k10temp_tctl": 88.0 + i, "rapl_pkg_watts": 17.0,
                        "is_gaming": 0.0, "gpu_throttle_mask": 0x40, "active_window": "kitty|shell",
                    })
                storage.insert_samples(rows)

                norm = storage.start_label(1000, "normal", "baseline boot-1")
                storage.end_label(21000, norm)
                anom = storage.start_label(55000, "airflow_blocked", "intake blocked")
                storage.end_label(60000, anom)
                storage.add_event(56500, "kmsg", "thermal_trip", "critical", "core temp high", "boot-2:1")

            # 1. Backup & verify
            backup_path = backup_database(db, root / "backup.db")
            ver = verify_database(backup_path)
            self.assertTrue(ver["ok"])
            self.assertEqual(ver["counts"]["samples"], 30)

            # 2. Report & Analyze
            rep = dataset_report(backup_path, boot_id="boot-2")
            self.assertEqual(rep["samples"]["count"], 10)
            ana = analyze_dataset(backup_path)
            self.assertEqual(len(ana["cadence"]), 2)
            self.assertEqual(check_physical_bounds(backup_path)["total_violations"], 0)
            self.assertEqual(audit_labels(backup_path)["cross_label_overlap_samples"], 0)
            self.assertGreater(correlate_metric(backup_path, "k10temp_tctl")["correlations"][0]["pairs"], 20)
            self.assertEqual(len(analyze_by_group(backup_path, "is_gaming")["groups"]), 2)
            self.assertGreater(compare_labels(backup_path, "normal", "airflow_blocked")["metrics"][0]["delta_mean"], 0)
            self.assertGreater(analyze_drift(backup_path, boot_id="boot-2")["metrics"][0]["delta_mean"], 40.0)
            self.assertGreater(detect_outliers(backup_path, columns=["k10temp_tctl"])["metrics"][0]["present"], 20)

            # 3. Export & Feature Engineering
            self.assertEqual(export_csv(backup_path, io.StringIO()), 30)
            self.assertEqual(export_jsonl(backup_path, io.StringIO()), 30)
            self.assertGreater(export_downsampled(backup_path, io.StringIO(), bucket_ms=5000), 0)
            self.assertEqual(export_events(backup_path, io.StringIO(), fmt="jsonl"), 1)
            feat_out = io.StringIO()
            self.assertEqual(
                export_features(
                    backup_path, feat_out, fmt="jsonl", deltas=True, rolling=3,
                    decode_throttle=True, split_ratios=(0.6, 0.2, 0.2),
                ),
                30,
            )
            feat_rows = [json.loads(line) for line in feat_out.getvalue().splitlines()]
            self.assertEqual(feat_rows[-1]["gpu_throttle_hw_thermal"], 1)
            self.assertEqual(feat_rows[-1]["split"], "test")

            # 4. Detector (fit, save, reload, tune, episode events)
            model_path = root / "model.json"
            det = detect_anomalies(
                backup_path, baseline_label="normal", boot_id="boot-2",
                columns=["k10temp_tctl"], control_for="cpu_util", group_by="is_gaming",
                eval_label="airflow_blocked", save_model=model_path,
            )
            self.assertAlmostEqual(det["evaluation"]["f1"], 1.0)
            self.assertAlmostEqual(det["evaluation"]["episode_f1"], 1.0)
            self.assertEqual(det["episodes"][0]["events"][0]["kind"], "thermal_trip")

            tuned = tune_threshold(
                backup_path, load_model=model_path, boot_id="boot-2",
                eval_label="airflow_blocked", thresholds=[2.0, 4.0, 100.0],
            )
            self.assertAlmostEqual(tuned["best"]["f1"], 1.0)

            # 5. HTML report
            html_path = write_html_report(
                backup_path, root / "campaign.html", load_model=model_path, boot_id="boot-2",
            )
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Épisodes anormaux détectés (1)", html_text)
            self.assertIn("<polyline", html_text)


if __name__ == "__main__":
    unittest.main()
