
from agent_v3 import CorrectionAgentV3

agent = CorrectionAgentV3(
    mode="llm",
    out_dir="artifacts_v3"
)

incidents = [
    {
        "incident_id": "TEST-DONNEES-SUSPECTES",
        "type": "qualite_donnees",
        "confidence": 0.95,
        "metrics": {
            "missing_rate": 0.35,
            "baseline_missing_rate": 0.01,
            "temperature_max": 110
        },
        "details": {
            "description": (
                "35% de valeurs manquantes et "
                "temperatures anormales. "
                "Les donnees ne doivent pas etre "
                "utilisees pour le reentrainement."
            ),
            "batch_id": "BATCH-2026-001",
            "data_quality_validated": False
        }
    },
    {
        "incident_id": "TEST-HUMAIN",
        "type": "inconnu",
        "confidence": 0.45,
        "metrics": {
            "baseline_mae": 20,
            "current_mae": 55
        },
        "details": {
            "description": (
                "Degradation importante du modele, "
                "mais cause inconnue. "
                "Une intervention humaine est necessaire."
            )
        }
    }
]

for incident in incidents:
    print("\n" + "=" * 55)
    print("INCIDENT :", incident["incident_id"])

    result = agent.run(incident)

    print(
        "Action proposee par Groq :",
        result["proposed_plan"]["action"]
    )
    print(
        "Decision apres garde-fou :",
        result["plan"]["action"]
    )
    print(
        "Proposition acceptee :",
        result["guard_result"]["accepted"]
    )
    print(
        "Resultat :",
        result["tool_result"]
    )
    print(
        "Verification :",
        result["verification"]
    )
