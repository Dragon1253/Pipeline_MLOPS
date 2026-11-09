
"""Comparaison de versions pour préparer un rollback simulé."""

import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error

from ml_pipeline import make_data

OUT = Path("artifacts_v3/rollback")
MODELS = Path("artifacts_v3/pipeline")


def prepare_rollback():
    OUT.mkdir(parents=True, exist_ok=True)

    # Chargement de modèles locaux de confiance
    v1 = joblib.load(MODELS / "v1_reference.joblib")
    v2 = joblib.load(MODELS / "v2_candidate.joblib")

    X, y = make_data()

    # Évaluation sur la période après dérive
    X_eval, y_eval = X[2500:], y[2500:]

    mae_v1 = mean_absolute_error(y_eval, v1.predict(X_eval))
    mae_v2 = mean_absolute_error(y_eval, v2.predict(X_eval))

    # Simulation : v1 est la version récente problématique,
    # v2 est la version antérieure stable.
    current_mae = float(mae_v1)
    previous_mae = float(mae_v2)

    rollback_recommended = previous_mae < current_mae

    report = {
        "scenario": "S4_simulated_version_regression",
        "current_version": "v1_reference.joblib",
        "previous_version": "v2_candidate.joblib",
        "current_mae": current_mae,
        "previous_mae": previous_mae,
        "rollback_recommended": rollback_recommended,
        "status": (
            "pending_human_approval"
            if rollback_recommended
            else "rollback_not_recommended"
        ),
        "rollback_executed": False,
        "production_modified": False,
    }

    (OUT / "rollback_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"MAE version récente : {current_mae:.2f}")
    print(f"MAE version précédente : {previous_mae:.2f}")
    print("Rollback recommandé :", rollback_recommended)
    print("Rapport :", OUT / "rollback_report.json")
    print("Aucune modification de production.")

    return report


if __name__ == "__main__":
    prepare_rollback()
