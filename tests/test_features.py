from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import pathlib
import tempfile
import unittest

from collector import features as features_module
from collector.features import decode_throttle_mask, export_features
from collector.main import SCHEMA
from collector.storage import Storage


class FeatureEngineeringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = pathlib.Path(self.tmp.name) / "metrics.db"
        self.storage = Storage(self.db, SCHEMA)
        self.addCleanup(self.storage.close)
        self.storage.ensure_columns(["temp_c", "gpu_throttle_mask", "note"], {"note": "TEXT"})

    def test_decode_throttle_mask_handles_flags_and_nulls(self):
        decoded = decode_throttle_mask(0x04 | 0x40)  # power_cap + hw_thermal
        self.assertEqual(decoded["gpu_throttle_power_cap"], 1)
        self.assertEqual(decoded["gpu_throttle_hw_thermal"], 1)
        self.assertEqual(decoded["gpu_throttle_idle"], 0)
        self.assertTrue(all(v is None for v in decode_throttle_mask(None).values()))

    def test_deltas_and_rolling_reset_across_boots_without_imputing_nulls(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0, "gpu_throttle_mask": 0.0},
            {"ts": 3000, "boot_id": "a", "temp_c": 50.0, "gpu_throttle_mask": 0x40},
            {"ts": 4000, "boot_id": "a", "temp_c": None, "gpu_throttle_mask": None},
            {"ts": 1000, "boot_id": "b", "temp_c": 30.0, "gpu_throttle_mask": 0x01},
        ])
        out = io.StringIO()
        count = export_features(
            self.db, out, fmt="jsonl", rolling=2, deltas=True, decode_throttle=True,
        )
        self.assertEqual(count, 4)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        # Tick 0 (boot a): pas de delta ni de std glissant
        self.assertIsNone(rows[0]["temp_c_delta"])
        self.assertEqual(rows[0]["temp_c_roll2_mean"], 40.0)
        self.assertIsNone(rows[0]["temp_c_roll2_std"])
        self.assertEqual(rows[0]["gpu_throttle_hw_thermal"], 0)
        # Tick 1 (boot a): +10°C en 2s => rate=5.0 °C/s
        self.assertAlmostEqual(rows[1]["temp_c_delta"], 10.0)
        self.assertAlmostEqual(rows[1]["temp_c_rate_s"], 5.0)
        self.assertAlmostEqual(rows[1]["temp_c_roll2_mean"], 45.0)
        self.assertEqual(rows[1]["gpu_throttle_hw_thermal"], 1)
        # Tick 2 (boot a): temp_c=None -> delta=None, mask=None -> flag=None
        self.assertIsNone(rows[2]["temp_c_delta"])
        self.assertIsNone(rows[2]["gpu_throttle_hw_thermal"])
        # Tick 3 (boot b): état réinitialisé au changement de boot
        self.assertIsNone(rows[3]["temp_c_delta"])
        self.assertEqual(rows[3]["temp_c_roll2_mean"], 30.0)
        self.assertEqual(rows[3]["gpu_throttle_idle"], 1)

    def test_cli_exports_csv_and_rejects_invalid_rolling_window(self):
        self.storage.insert_samples([
            {"ts": 1000, "boot_id": "a", "temp_c": 40.0},
            {"ts": 2000, "boot_id": "a", "temp_c": 45.0},
        ])
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            features_module.main(["--db", str(self.db), "--columns", "temp_c", "--deltas", "--rolling", "2"])
        self.assertIn("temp_c_delta,temp_c_rate_s,temp_c_roll2_mean,temp_c_roll2_std,labels_json", out.getvalue())
        self.assertIn("2 mesure(s)", err.getvalue())

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            features_module.main(["--db", str(self.db), "--rolling", "1"])
        self.assertEqual(exc.exception.code, 2)

    def test_chronological_split_ratios_assign_ordered_train_val_test(self):
        self.storage.insert_samples([
            {"ts": 1000 * (i + 1), "boot_id": "a", "temp_c": 40.0 + i} for i in range(10)
        ])
        out = io.StringIO()
        self.assertEqual(
            export_features(self.db, out, fmt="jsonl", columns=["temp_c"], split_ratios=(0.6, 0.2, 0.2)),
            10,
        )
        splits = [json.loads(line)["split"] for line in out.getvalue().splitlines()]
        self.assertEqual(splits, ["train"] * 6 + ["val"] * 2 + ["test"] * 2)
        with self.assertRaises(ValueError):
            export_features(self.db, io.StringIO(), split_ratios=(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
