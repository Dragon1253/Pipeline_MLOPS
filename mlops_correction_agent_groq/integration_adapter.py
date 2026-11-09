
from datetime import datetime, timezone
from uuid import uuid4

CAUSE_TO_TYPE = {
    "data_corruption": "qualite_donnees",
    "data_drift": "derive_donnees",
    "concept_drift": "derive_donnees",
    "deployment_regression": "inconnu",
    "no_incident": "fausse_alerte",
    "undetermined": "inconnu",
}


def adapt_amen_diagnosis(result):
    if not isinstance(result, dict):
        raise ValueError("Le diagnostic doit etre un objet JSON.")

    # Accepte la sortie complete de investigate()
    # ou directement son champ final_diagnosis.
    if "final_diagnosis" in result:
        if result.get("status") != "completed":
            raise ValueError(
                "Investigation incomplete : correction non autorisee."
            )
        diagnosis = result["final_diagnosis"]
    else:
        diagnosis = result

    if not isinstance(diagnosis, dict):
        raise ValueError("Diagnostic final absent ou invalide.")

    cause = diagnosis.get("cause_probable")
    if cause not in CAUSE_TO_TYPE:
        raise ValueError(f"Cause inconnue : {cause}")

    summary = diagnosis.get("resume")
    proofs = diagnosis.get("preuves")

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Resume manquant.")

    if not isinstance(proofs, list) or not all(
        isinstance(p, str) for p in proofs
    ):
        raise ValueError("Preuves invalides.")

    # Amen ne fournit pas de confiance numerique.
    # On force une revue humaine sans inventer de score.
    confidence = 0.0

    return {
        "incident_id": "AMEN-" + uuid4().hex[:10],
        "type": CAUSE_TO_TYPE[cause],
        "confidence": confidence,
        "metrics": {},
        "details": {
            "source": "amen_investigation",
            "cause_probable": cause,
            "description": summary,
            "evidence_ids": proofs,
            "confidence_available": False,
            "requires_human_review": True,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }
