
import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error

from agent_v3 import CorrectionAgentV3
from ml_pipeline import make_data, fit

OUT = Path("artifacts_v3/pipeline")


def run():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Données et modèle initial
    X, y = make_data()
    v1 = fit(X[:1000], y[:1000])

    baseline_mae = mean_absolute_error(
        y[1000:1200], v1.predict(X[1000:1200])
    )
    current_mae = mean_absolute_error(
        y[2000:2500], v1.predict(X[2000:2500])
    )

    print(f"MAE initiale : {baseline_mae:.2f}")
    print(f"MAE après dérive : {current_mae:.2f}")

    # 2. Construire l'incident à partir des métriques
    if current_mae <= baseline_mae * 1.2:
        print("Aucune dégradation suffisante détectée.")
        return

    incident = {
        "incident_id": "PIPELINE-V3-001",
        "type": "derive_donnees",
        "confidence": 0.95,
        "metrics": {
            "baseline_mae": float(baseline_mae),
            "current_mae": float(current_mae),
        },
        "details": {
            "dataset": "synthetic_bike_demand",
            "data_quality_validated": True,
        },
    }

    # 3. Décision du vrai LLM
    agent = CorrectionAgentV3(
        mode="llm",
        out_dir="artifacts_v3"
    )
    decision = agent.run(incident)

    proposed = decision["proposed_plan"]["action"]
    final = decision["plan"]["action"]

    print(f"Action proposée par LLM : {proposed}")
    print(f"Action validée : {final}")

    # 4. Aucun réentraînement sans autorisation
    if not decision["verification"]["safe"]:
        print("Arrêt : vérification de sécurité échouée.")
        return

    if not decision["guard_result"]["accepted"]:
        print("Arrêt : garde-fou a refusé le plan.")
        return

    if final != "prepare_retraining":
        print("Pas de réentraînement autorisé.")
        return

    # 5. Entraînement réel du candidat v2
    print("Entraînement du modèle candidat v2...")
    v2 = fit(X[:2000], y[:2000])

    # 6. Évaluation indépendante
    val_v1 = mean_absolute_error(
        y[2000:2500], v1.predict(X[2000:2500])
    )
    val_v2 = mean_absolute_error(
        y[2000:2500], v2.predict(X[2000:2500])
    )

    print(f"Validation MAE v1 : {val_v1:.2f}")
    print(f"Validation MAE v2 : {val_v2:.2f}")

    if val_v2 >= val_v1:
        print("Candidat rejeté : aucune amélioration.")
        return

    holdout_v1 = mean_absolute_error(
        y[2500:], v1.predict(X[2500:])
    )
    holdout_v2 = mean_absolute_error(
        y[2500:], v2.predict(X[2500:])
    )

    print(f"Holdout MAE v1 : {holdout_v1:.2f}")
    print(f"Holdout MAE v2 : {holdout_v2:.2f}")

    # 7. Enregistrer le candidat sans déploiement
    joblib.dump(v1, OUT / "v1_reference.joblib")
    joblib.dump(v2, OUT / "v2_candidate.joblib")

    report = {
        "incident_id": incident["incident_id"],
        "llm_proposed_action": proposed,
        "final_action": final,
        "validation_mae_v1": float(val_v1),
        "validation_mae_v2": float(val_v2),
        "holdout_mae_v1": float(holdout_v1),
        "holdout_mae_v2": float(holdout_v2),
        "candidate_better_on_holdout": bool(
            holdout_v2 < holdout_v1
        ),
        "status": "pending_human_approval",
        "production_modified": False,
    }

    (OUT / "evaluation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("Rapport :", OUT / "evaluation.json")
    print("Aucun déploiement automatique.")


if __name__ == "__main__":
    run()
