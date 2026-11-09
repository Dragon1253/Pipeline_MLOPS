"""Jeu de données de prédiction : configurations figées, absence de fuite, cibles cohérentes."""
import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from mlops_churn.workspace import sha256
from monitoring.dataset import HORIZONS, SPLITS, build_split, stream_config, stream_rows
from monitoring.service import ReferenceContext
from monitoring.stream import StreamSpec

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = ReferenceContext()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, **overrides):
        config = {**stream_config(42, "standard"), "n_windows": 14, **overrides}
        return config, stream_rows(config, self.reference, self.out)

    def test_split_seed_ranges_are_disjoint_and_configs_reproducible(self):
        ranges = [set(range(first, first + count)) for first, count, _ in SPLITS.values()]
        for i, a in enumerate(ranges):
            for b in ranges[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(stream_config(7, "standard"), stream_config(7, "standard"))
        self.assertEqual(SPLITS["generalization"][2], "heldout")
        self.assertEqual({SPLITS[s][2] for s in ("train", "validation", "test")}, {"standard"})

    def test_known_metrics_only_use_labels_already_received(self):
        for delay in (1, 2, 3):
            with self.subTest(delay=delay):
                _, rows = self.rows(scenario="data_corruption", onset=5, ramp=2, max_intensity=1.0,
                                    label_delay=delay)
                by_index = {r["index"]: r for r in rows}
                for row in rows:
                    t = row["index"]
                    self.assertEqual(row["known_windows"], max(0, t - delay + 1))
                    if t >= delay:
                        self.assertEqual(row["known_log_loss_lag0"], by_index[t - delay]["label_log_loss"])
                        self.assertEqual(row["known_last_window_age"], delay)
                    else:
                        self.assertIsNone(row["known_log_loss_lag0"])

    def test_targets_match_future_log_loss(self):
        _, rows = self.rows(scenario="concept_drift", onset=6, ramp=0, max_intensity=1.0, label_delay=1)
        threshold = self.reference.calibration["threshold"]
        losses = [r["label_log_loss"] for r in rows]
        for row in rows:
            t = row["index"]
            self.assertEqual(row["target_now"], int(losses[t] > threshold))
            for h in HORIZONS:
                future = losses[t + 1: t + 1 + h]
                expected = int(any(v > threshold for v in future)) if len(future) == h else None
                self.assertEqual(row[f"target_h{h}"], expected)
        # Incident brutal en w006 : la fenêtre w005 doit annoncer une dégradation à h=1.
        self.assertEqual(rows[5]["target_h1"], 1)

    def test_features_do_not_contain_truth_or_targets(self):
        config, rows = self.rows()
        features = [k for k in rows[0] if k.startswith(("sig_", "known_")) or k == "label_delay"]
        self.assertTrue(features)
        self.assertFalse([k for k in features if "truth" in k or "target" in k or k == "label_log_loss"])
        self.assertEqual(rows[0]["truth_scenario"], config["scenario"])

    def test_heldout_variant_changes_the_mechanism(self):
        standard = StreamSpec("data_corruption", 3, n_windows=6, onset=2, ramp=0)
        heldout = StreamSpec("data_corruption", 3, n_windows=6, onset=2, ramp=0, variant="heldout")
        a = stream_rows({**stream_config(3, "standard"), **standard.__dict__, "label_delay": 1}, self.reference, self.out)
        b = stream_rows({**stream_config(3, "heldout"), **heldout.__dict__, "label_delay": 1}, self.reference, self.out)
        self.assertGreater(a[3]["sig_psi_monthly_charges"], 1.0)
        self.assertLess(b[3]["sig_psi_monthly_charges"], 0.2)
        self.assertGreater(b[3]["sig_psi_usage_hours_month"], 1.0)
        with self.assertRaises(ValueError):
            StreamSpec("data_corruption", 3, variant="inconnue")

    def test_build_split_writes_csv_and_manifest(self):
        path = build_split("validation", output_dir=self.out / "ds", n_streams=2)
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["rows"], len(rows))
        self.assertEqual(manifest["streams"], 2)
        self.assertEqual(manifest["csv_sha256"], sha256(path))
        self.assertEqual({r["stream_id"] for r in rows}, {"10001", "10002"})


if __name__ == "__main__":
    unittest.main()
