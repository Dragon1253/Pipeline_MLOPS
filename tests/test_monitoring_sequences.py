"""Modèles séquentiels : séquences sans fuite entre flux ni vers le futur ; entraînement si PyTorch est installé."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.sequence_models import Preprocessor, build_sequences, torch

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


def toy_frame():
    rows = [{"stream_id": s, "index": i, "sig_a": s * 100 + i, "known_b": np.nan if i < 2 else float(i)}
            for s in (1, 2) for i in range(8)]
    return pd.DataFrame(rows)


class SequenceTests(unittest.TestCase):
    def test_sequences_end_on_current_window_and_never_mix_streams(self):
        frame = toy_frame()
        matrix = frame[["sig_a"]].to_numpy(dtype=float)
        seq = build_sequences(frame, matrix, length=4)
        self.assertEqual(seq.shape, (16, 4, 2))
        # Fenêtre 5 du flux 1 : fenêtres 2, 3, 4, 5 dans l'ordre, sans remplissage.
        self.assertEqual(seq[5, :, 0].tolist(), [102, 103, 104, 105])
        self.assertEqual(seq[5, :, 1].tolist(), [0, 0, 0, 0])
        # Fenêtre 1 du flux 2 : deux positions de remplissage, rien du flux 1.
        self.assertEqual(seq[9, :, 1].tolist(), [1, 1, 0, 0])
        self.assertEqual(seq[9, 2:, 0].tolist(), [200, 201])
        self.assertTrue((seq[9, :2, 0] == 0).all())

    def test_future_windows_do_not_change_past_sequences(self):
        frame = toy_frame()
        matrix = frame[["sig_a"]].to_numpy(dtype=float)
        full = build_sequences(frame, matrix, length=3)
        cut = frame[frame["index"] < 5].reset_index(drop=True)
        partial = build_sequences(cut, cut[["sig_a"]].to_numpy(dtype=float), length=3)
        np.testing.assert_array_equal(full[frame["index"].to_numpy() < 5], partial)

    def test_unsorted_rows_are_refused(self):
        frame = toy_frame().iloc[[0, 2, 1]].reset_index(drop=True)
        with self.assertRaises(ValueError):
            build_sequences(frame, frame[["sig_a"]].to_numpy(dtype=float))

    def test_preprocessor_uses_training_statistics_and_flags_missing_metrics(self):
        train = toy_frame()
        pre = Preprocessor().fit(train, ["sig_a", "known_b"])
        other = toy_frame().assign(sig_a=1e6)
        matrix = pre.transform(other)
        self.assertFalse(np.isnan(matrix).any())
        self.assertAlmostEqual(pre.mean[0], train["sig_a"].mean())
        self.assertEqual(matrix[0, -1], 1.0)   # known_b manquant en début de flux
        self.assertEqual(matrix[5, -1], 0.0)

    @unittest.skipIf(torch is None, "PyTorch non installé : pip install -r requirements-monitoring.txt")
    def test_gru_and_lstm_train_and_join_the_comparison(self):
        from monitoring.dataset import build_split
        from monitoring.prediction import run
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "dataset"
            for split, streams in (("train", 12), ("validation", 6), ("test", 6), ("generalization", 6)):
                build_split(split, output_dir=data, n_streams=streams)
            folder, report = run("target_h2", 10, dataset_dir=data, output_dir=Path(tmp) / "pred",
                                 sequence_models=("gru", "lstm"), sequence_epochs=2)
            self.assertTrue((folder / "sequence_models.pt").is_file())
            for split in ("validation", "test", "generalization"):
                for cell in ("gru", "lstm"):
                    result = report["splits"][split][cell]
                    self.assertGreaterEqual(result["window"]["pr_auc"], 0)
                    self.assertIn("detection_rate", result["events"])
            self.assertIn("M3 GRU", (folder / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
