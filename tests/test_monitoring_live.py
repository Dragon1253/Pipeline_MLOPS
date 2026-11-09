"""Alerte prédictive en direct : mêmes entrées qu'à l'entraînement, alerte avant l'alerte confirmée."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.dataset import build_split, stream_config, stream_rows
from monitoring.prediction import add_trend_features
from monitoring.predictor import load_predictor, train_predictor
from monitoring.service import ReferenceContext, run_stream
from monitoring.stream import StreamSpec

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


class LivePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        for split, streams in (("train", 40), ("validation", 12), ("test", 8), ("generalization", 4)):
            build_split(split, output_dir=root / "dataset", n_streams=streams)
        cls.path, cls.info = train_predictor("target_h2", 5, dataset_dir=root / "dataset",
                                             path=root / "predictor" / "predictor.joblib")
        cls.predictor = load_predictor(cls.path)
        cls.reference = ReferenceContext()
        cls.out = root / "runs"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_live_inputs_equal_training_inputs(self):
        config = {**stream_config(5, "standard"), "scenario": "data_corruption", "onset": 6, "ramp": 3,
                  "max_intensity": 1.0, "label_delay": 2, "n_windows": 12}
        offline = add_trend_features(pd.DataFrame(stream_rows(config, self.reference, self.out)))
        live = []
        spec = StreamSpec("data_corruption", 5, n_windows=12, onset=6, ramp=3)
        run_stream(spec, reference=self.reference, output_dir=self.out, label_delay=2, package=False,
                   predictor=self.predictor,
                   on_event=lambda e: live.append(e["features"]) if e["event"] == "risk_recorded" else None)
        live = pd.DataFrame(live)[self.predictor.columns].astype(float)
        np.testing.assert_allclose(live.to_numpy(), offline[self.predictor.columns].astype(float).to_numpy(),
                                   rtol=0, atol=1e-12, equal_nan=True)

    def test_predictive_alert_precedes_the_confirmed_alert_and_is_handed_over(self):
        spec = StreamSpec("data_corruption", 7, n_windows=14, onset=8, ramp=4)
        summary = run_stream(spec, reference=self.reference, output_dir=self.out, label_delay=2,
                             predictor=self.predictor)
        critical = {a["window_id"] for a in summary["alerts"] if a["severity"] == "critical"}
        confirmed = [a for a in summary["predictive_alerts"]
                     if a["status"] == "confirmed" and a["confirmed_window"] in critical]
        self.assertTrue(confirmed, summary["predictive_alerts"])
        alert = confirmed[0]
        # L'alerte prédictive annonce l'incident (début en w008) au plus tard sur sa première
        # fenêtre, et avant l'alerte confirmée qui attend les labels.
        self.assertGreaterEqual(int(alert["confirmed_window"][1:]), 8)
        self.assertLessEqual(int(alert["window_id"][1:]), int(alert["confirmed_window"][1:]))
        self.assertGreaterEqual(alert["lead_steps"], 1)
        self.assertTrue(alert["top_signals"])
        # Fichier de vigilance, sans le nom du scénario.
        files = list((Path(summary["run_dir"]) / "predictive_alerts").glob("*.json"))
        self.assertTrue(files)
        for file in files:
            self.assertNotIn("corruption", file.read_text(encoding="utf-8"))
        # Le paquet de l'alerte confirmée mentionne l'alerte prédictive qui l'a précédée.
        episode = Path([e for e in summary["episodes"] if alert["confirmed_window"] in e][0])
        handed = json.loads((episode / "alert.json").read_text(encoding="utf-8"))
        self.assertEqual(handed["preceded_by_predictive_alert"]["window_id"], alert["window_id"])
        context = json.loads((episode / "monitoring_context.json").read_text(encoding="utf-8"))
        self.assertIn("prediction", context)
        self.assertIsNotNone(context["windows"][0]["risk"])

    def test_without_predictor_the_monitoring_is_unchanged(self):
        summary = run_stream(StreamSpec("data_corruption", 7, n_windows=10, onset=5, ramp=2),
                             reference=self.reference, output_dir=self.out, package=False)
        self.assertEqual(summary["predictive_alerts"], [])
        self.assertFalse(any(e["event"] == "risk_recorded" for e in summary["events"]))


if __name__ == "__main__":
    unittest.main()
