"""Étape 2.5 : le modèle de prédiction de l'alerte utilisé en direct par le monitoring.

``train_predictor`` entraîne une fois le modèle retenu (M2 gradient boosting, cible
``target_h2``) avec le protocole des étapes 2.2–2.4 et l'enregistre dans
``monitoring_runs/predictor/predictor.joblib`` avec son seuil d'alarme et le profil des
valeurs habituelles (pour expliquer les alertes).

``AlertPredictor`` calcule, à chaque fenêtre d'un flux, les mêmes entrées que le jeu de
données (``dataset.window_features`` + tendances de ``prediction.add_trend_features``), le
risque de dégradation dans les ``horizon`` prochaines fenêtres et, au-delà du seuil, les
signaux qui expliquent l'alerte.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd

from mlops_churn.workspace import sha256, write_json

from .analysis import _prepare, explain_alert, fit_boosting, method_summary, reference_profile
from .prediction import DATASET_DIR, add_trend_features, feature_columns
from .service import DEFAULT_OUTPUT

DEFAULT_PREDICTOR = DEFAULT_OUTPUT / "predictor" / "predictor.joblib"
HORIZONS = {"target_now": 0, "target_h1": 1, "target_h2": 2, "target_h3": 3}


class AlertPredictor:
    def __init__(self, model, columns, threshold, median, spread, target, info, version=None):
        self.model, self.columns, self.threshold = model, list(columns), float(threshold)
        self.median, self.spread = median, spread
        self.target, self.horizon, self.info = target, HORIZONS[target], info
        self.version = version

    def features(self, history):
        """Entrées de la dernière fenêtre de ``history`` (lignes de ``window_features`` du flux,
        dans l'ordre) ; les tendances n'utilisent que ces fenêtres passées et la courante."""
        frame = pd.DataFrame(history)
        frame["stream_id"] = 0
        frame["index"] = range(len(frame))
        frame = add_trend_features(frame)
        return frame.iloc[[-1]][self.columns].astype(float)

    def risk(self, features):
        return float(self.model.predict_proba(features)[:, 1][0])

    def explain(self, features, top=3):
        return explain_alert(self.model, features.iloc[0], self.columns, (self.median, self.spread), top=top)


def train_predictor(target="target_h2", budget=5.0, dataset_dir=DATASET_DIR, path=DEFAULT_PREDICTOR):
    splits = _prepare(target, dataset_dir)
    columns = feature_columns(splits["train"], ("sig", "known", "trend"))
    model, threshold, params = fit_boosting(splits, target, columns, budget)
    median, spread = reference_profile(splits["train"], columns)
    test = method_summary(splits["test"], model.predict_proba(splits["test"][columns])[:, 1], threshold, target)
    manifests = {}
    for split in ("train", "validation"):
        manifest = Path(dataset_dir) / f"{split}.manifest.json"
        if manifest.exists():
            manifests[split] = json.loads(manifest.read_text(encoding="utf-8")).get("csv_sha256")
    info = {"created_at": datetime.now(timezone.utc).isoformat(), "model": "M2 gradient boosting",
            "target": target, "horizon_windows": HORIZONS[target], "params": params,
            "false_alarm_budget_per_100_clean_windows": budget, "threshold": threshold,
            "n_features": len(columns), "dataset_sha256": manifests,
            "test": {k: v for k, v in test.items() if k != "by_scenario"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "threshold": threshold, "median": median,
                 "spread": spread, "target": target, "info": info}, path)
    write_json(path.with_suffix(".json"), {**info, "file_sha256": sha256(path)})
    return path, info


def load_predictor(path=DEFAULT_PREDICTOR):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prédicteur absent : {path}. Lancer d'abord « python -m monitoring train-predictor ».")
    bundle = joblib.load(path)
    return AlertPredictor(bundle["model"], bundle["columns"], bundle["threshold"], bundle["median"],
                          bundle["spread"], bundle["target"], bundle["info"], version="sha256:" + sha256(path))

