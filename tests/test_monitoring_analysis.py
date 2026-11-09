"""Consolidation : entrée de position exclue, bootstrap cohérent, explications des alertes."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from monitoring.analysis import bootstrap, explain_alert, reference_profile, run
from monitoring.dataset import build_split
from monitoring.prediction import feature_columns, load_split

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "dataset"
        for split, streams in (("train", 14), ("validation", 8), ("test", 8), ("generalization", 6)):
            build_split(split, output_dir=cls.data, n_streams=streams)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_position_in_stream_is_not_an_input(self):
        columns = feature_columns(load_split("validation", self.data))
        self.assertNotIn("known_windows", columns)
        self.assertNotIn("index", columns)

    def test_bootstrap_interval_contains_the_point_estimate(self):
        frame = load_split("test", self.data).dropna(subset=["target_h2"]).reset_index(drop=True)
        scores = frame["sig_score_psi"].fillna(0).to_numpy()
        result = bootstrap(frame, scores, 0.2, "target_h2", n_boot=50)
        for key, entry in result.items():
            if np.isnan(entry["value"]):
                continue
            low, high = entry["ci95"]
            self.assertLessEqual(low, entry["value"] + 1e-9, key)
            self.assertGreaterEqual(high, entry["value"] - 1e-9, key)

    def test_full_analysis_explains_alerts_with_measured_signals(self):
        folder, report = run("target_h2", 10, dataset_dir=self.data, output_dir=Path(self.tmp.name) / "an",
                             n_boot=50, ablate=True)
        self.assertTrue((folder / "report.md").is_file())
        self.assertEqual(len(report["ablation"]), 6)
        self.assertTrue(report["importance"])
        for explanation in report["explanations"].values():
            self.assertLessEqual(len(explanation["signals"]), 3)
            for signal in explanation["signals"]:
                self.assertTrue(signal["signal"].startswith(("sig_", "known_")))

    def test_explanation_reports_signals_far_from_usual(self):
        from monitoring.prediction import fit_models
        train = load_split("train", self.data).dropna(subset=["target_h2"])
        validation = load_split("validation", self.data).dropna(subset=["target_h2"])
        columns = feature_columns(train)
        model = fit_models(train, validation, "target_h2", columns, kinds=("boosting",))[0]["boosting"]
        row = validation.iloc[0].copy()
        row["sig_score_psi"] = 50.0
        explanation = explain_alert(model, row, columns, reference_profile(train, columns))
        self.assertEqual(explanation["signals"][0]["signal"], "sig_score_psi")


if __name__ == "__main__":
    unittest.main()
