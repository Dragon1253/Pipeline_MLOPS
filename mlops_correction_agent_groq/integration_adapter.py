
from datetime import datetime, timezone
from uuid import uuid4

CONFIDENCE_THRESHOLD = 0.7

CAUSE_TO_TYPE = {
    "data_corruption": "qualite_donnees",
    "data_drift": "derive_donnees",
    "concept_drift": "derive_donnees",
    "deployment_regression": "inconnu",
    "no_incident": "fausse_alerte",
    "undetermined": "inconnu",
}


def _successful_evidence(result):
    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        return {}
    return {
        item["evidence_id"]: item
        for item in evidence
        if isinstance(item, dict)
        and isinstance(item.get("evidence_id"), str)
        and isinstance(item.get("result"), dict)
        and item["result"].get("status") == "ok"
    }


def _observations(item):
    payload = item["result"].get("observations", item["result"])
    return payload if isinstance(payload, dict) else {}


def is_data_transformation_anomaly(observations):
    """Detect a large source-to-input change, not population drift."""
    if not isinstance(observations, dict):
        return False
    rows = observations.get("matched_rows", 0)
    columns = observations.get("columns", [])
    if not isinstance(rows, int) or rows < 100 or not isinstance(columns, list):
        return False
    for column in columns:
        if not isinstance(column, dict):
            continue
        changed = max(column.get("changed_values", 0), column.get("missing_transitions", 0))
        ratio = column.get("median_right_over_left")
        scaling = isinstance(ratio, (int, float)) and (ratio >= 3 or 0 < ratio <= 1 / 3)
        if changed / rows >= 0.2 and (scaling or column.get("missing_transitions", 0) / rows >= 0.2
                                      or column.get("changed_values", 0) / rows >= 0.8):
            return True
    return False


def _supports_model_regression(observations):
    bootstrap = observations.get("paired_bootstrap", {})
    delta = observations.get("candidate_minus_reference_log_loss")
    lower = bootstrap.get("lower") if isinstance(bootstrap, dict) else None
    return (isinstance(delta, (int, float)) and delta > 0
            and isinstance(lower, (int, float)) and lower > 0)


def _supports_distribution_drift(observations):
    columns = observations.get("columns", [])
    return isinstance(columns, list) and any(
        isinstance(column, dict)
        and isinstance(column.get("q_value_bh"), (int, float))
        and column["q_value_bh"] < 0.05
        for column in columns
    )


def _supports_concept_drift(observations):
    likelihood = observations.get("likelihood_ratio", {})
    validation = observations.get("cross_validation", {})
    p_value = likelihood.get("p_value") if isinstance(likelihood, dict) else None
    gain = validation.get("loss_gain") if isinstance(validation, dict) else None
    positive_folds = validation.get("positive_gain_folds", 0) if isinstance(validation, dict) else 0
    folds = validation.get("folds", 0) if isinstance(validation, dict) else 0
    return (isinstance(p_value, (int, float)) and p_value < 0.05
            and isinstance(gain, (int, float)) and gain > 0
            and isinstance(positive_folds, int) and isinstance(folds, int)
            and folds > 0 and positive_folds / folds >= 0.6)


def score_amen_diagnosis(result, monitoring_alerts=()):
    """Return an explainable corroboration score; it is not a calibrated probability."""
    diagnosis = result.get("final_diagnosis", {}) if isinstance(result, dict) else {}
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    cause = diagnosis.get("cause_probable")
    proofs = diagnosis.get("preuves", [])
    proofs = proofs if isinstance(proofs, list) else []
    successful = _successful_evidence(result if isinstance(result, dict) else {})
    valid_proofs = [proof for proof in dict.fromkeys(proofs) if proof in successful]
    cited = [successful[proof] for proof in valid_proofs]

    table_evidence = [item for item in cited if item.get("tool") == "compare_tables"
                      and {item.get("arguments", {}).get("left"), item.get("arguments", {}).get("right")}
                      == {"source", "inputs"}]
    model_evidence = [item for item in cited if item.get("tool") == "compare_model_versions"]
    distribution_evidence = [item for item in cited if item.get("tool") == "compare_distributions"]
    conditional_evidence = [item for item in cited if item.get("tool") == "compare_conditional_relationship"]

    table_support = next((item for item in table_evidence
                          if is_data_transformation_anomaly(_observations(item))), None)
    model_support = next((item for item in model_evidence
                          if _supports_model_regression(_observations(item))), None)
    distribution_support = next((item for item in distribution_evidence
                                 if _supports_distribution_drift(_observations(item))), None)
    conditional_support = next((item for item in conditional_evidence
                                if _supports_concept_drift(_observations(item))), None)

    incident_type = CAUSE_TO_TYPE.get(cause, "inconnu")
    supporting_tool = None
    supported = False
    if cause == "data_corruption" and table_support:
        incident_type, supporting_tool, supported = "qualite_donnees", "compare_tables", True
    elif cause == "data_drift" and distribution_support:
        incident_type, supporting_tool, supported = "derive_donnees", "compare_distributions", True
    elif cause == "concept_drift" and conditional_support:
        incident_type, supporting_tool, supported = "derive_donnees", "compare_conditional_relationship", True
    elif cause == "deployment_regression":
        if table_support:
            incident_type, supporting_tool, supported = "qualite_donnees", "compare_tables", True
        elif model_support:
            incident_type, supporting_tool, supported = "regression_version", "compare_model_versions", True
        else:
            incident_type = "inconnu"

    alerts = monitoring_alerts if isinstance(monitoring_alerts, (list, tuple)) else []
    confirmed = any(alert.get("alert_type") == "confirmed" for alert in alerts if isinstance(alert, dict))
    critical = any(alert.get("severity") == "critical" for alert in alerts if isinstance(alert, dict))
    if supported:
        confidence = 0.75 + (0.1 if confirmed else 0) + (0.05 if critical else 0)
        confidence += 0.05 if len(valid_proofs) >= 2 else 0
        confidence = min(confidence, 0.95)
    elif incident_type != "inconnu":
        confidence = 0.35
    else:
        confidence = 0.0

    return {
        "confidence": round(confidence, 2),
        "incident_type": incident_type,
        "supported": supported,
        "supporting_tool": supporting_tool,
        "evidence_ids": [item["evidence_id"] for item in cited],
        "method": "heuristique de corroboration; non calibree comme probabilite",
        "requires_human_review": confidence < CONFIDENCE_THRESHOLD or incident_type == "inconnu",
    }


def adapt_amen_diagnosis(result, *, confidence_assessment=None):
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

    assessment = confidence_assessment or {}
    confidence = assessment.get("confidence", 0.0)
    incident_type = assessment.get("incident_type", CAUSE_TO_TYPE[cause])
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1):
        raise ValueError("Score de confiance invalide.")
    if incident_type not in {"qualite_donnees", "derive_donnees", "regression_version", "fausse_alerte", "inconnu"}:
        raise ValueError("Type d'incident invalide.")

    return {
        "incident_id": "AMEN-" + uuid4().hex[:10],
        "type": incident_type,
        "confidence": float(confidence),
        "metrics": {},
        "details": {
            "source": "amen_investigation",
            "cause_probable": cause,
            "description": summary,
            "evidence_ids": proofs,
            "confidence_available": confidence_assessment is not None,
            "confidence_method": assessment.get("method"),
            "confidence_supported": assessment.get("supported", False),
            "confidence_tool": assessment.get("supporting_tool"),
            "confidence_evidence_ids": assessment.get("evidence_ids", []),
            "requires_human_review": assessment.get("requires_human_review", True),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }
