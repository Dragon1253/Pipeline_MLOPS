"""Prédiction de l'alerte : pas de fuite dans les entrées, seuils à budget égal, chaîne complète."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.dataset import build_split
from monitoring.prediction import (FORBIDDEN_PREFIXES, METHODS, add_trend_features, event_metrics,
                                   feature_columns, load_split, run, threshold_for_budget)

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "dataset"
        for split, streams in (("train", 12), ("validation", 6), ("test", 6), ("generalization", 6)):
            build_split(split, output_dir=cls.data, n_streams=streams)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_inputs_never_include_truth_targets_or_reactive_decisions(self):
        frame = load_split("validation", self.data)
        for groups in (("sig",), ("sig", "known"), ("sig", "known", "trend")):
            columns = feature_columns(frame, groups)
            self.assertTrue(columns)
            self.assertFalse([c for c in columns if c.startswith(FORBIDDEN_PREFIXES)])

    def test_trend_features_only_use_past_windows(self):
        frame = load_split("validation", self.data)
        stream = frame[frame["stream_id"] == frame["stream_id"].iloc[0]].reset_index(drop=True)
        cut = 10
        full = add_trend_features(stream.drop(columns=[c for c in stream if c.startswith("trend_")]))
        truncated = add_trend_features(stream.drop(columns=[c for c in stream if c.startswith("trend_")]).iloc[:cut])
        trend = [c for c in full if c.startswith("trend_")]
        pd.testing.assert_frame_equal(full.loc[: cut - 1, trend], truncated[trend])

    def test_threshold_respects_false_alarm_budget(self):
        frame = pd.DataFrame({"truth_incident_active": [0] * 100 + [1] * 20})
        scores = np.r_[np.linspace(0, 1, 100), np.ones(20)]
        for budget in (0, 5, 20):
            threshold = threshold_for_budget(scores, frame, budget)
            self.assertLessEqual((scores[:100] >= threshold).mean() * 100, budget)

    def test_event_lead_is_measured_against_the_reactive_alert(self):
        frame = pd.DataFrame({
            "stream_id": [1] * 6, "index": range(6), "truth_scenario": ["data_corruption"] * 6,
            "truth_incident_active": [0, 0, 1, 1, 1, 1], "target_now": [0, 0, 1, 1, 1, 1],
            "truth_onset": [2] * 6, "react_alert_at_step": [0, 0, 0, 1, 0, 0],
            "react_alert_window": [None, None, None, "w002", None, None]})
        events = event_metrics(frame, np.array([False, False, True, True, False, False]))
        self.assertEqual(events.loc[0, "detected_step"], 2)
        self.assertEqual(events.loc[0, "reactive_step"], 3)

    def test_full_run_writes_a_comparison_of_all_methods(self):
        folder, report = run("target_now", 10, dataset_dir=self.data, output_dir=Path(self.tmp.name) / "pred",
                             save_models=True)
        self.assertTrue((folder / "report.md").is_file() and (folder / "models.joblib").is_file())
        for split in ("validation", "test", "generalization"):
            self.assertEqual(set(report["splits"][split]), set(METHODS))
        # Seuils choisis sur la validation : le budget y est respecté pour chaque méthode à score.
        for method in METHODS[1:]:
            self.assertLessEqual(report["splits"]["validation"][method]["false_alarms_per_100_clean_windows"], 10)


if __name__ == "__main__":
    unittest.main()
