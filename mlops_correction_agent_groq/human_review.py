
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_v3 import CorrectionAgentV3


SCENARIOS = {
    "uncertain": {
        "incident_id": "HUMAN-001",
        "type": "inconnu",
        "confidence": 0.40,
        "metrics": {
            "baseline_mae": 20,
            "current_mae": 55,
        },
        "details": {
            "description": "Degradation du modele, cause inconnue."
        },
    },
    "retraining": {
        "incident_id": "HUMAN-002",
        "type": "derive_donnees",
        "confidence": 0.95,
        "metrics": {
            "baseline_mae": 8.8,
            "current_mae": 91.25,
        },
        "details": {
            "description": "Degradation apres changement des donnees."
        },
    },
    "rollback": {
        "incident_id": "HUMAN-003",
        "type": "regression_version",
        "confidence": 0.95,
        "metrics": {
            "current_mae": 91.17,
            "previous_mae": 54.70,
        },
        "details": {
            "description": "La version actuelle est moins performante.",
            "run_local_rollback_demo": True,
        },
    },
}


def ask_human(report):
    print("\n=== VALIDATION HUMAINE ===")
    print("Incident :", report["incident_id"])
    print("Action :", report["plan"]["action"])
    print("Resultat :", report["tool_result"])

    print("\n1 - Approuver la recommandation")
    print("2 - Demander une investigation")
    print("3 - Refuser la recommandation")

    while True:
        choice = input("\nVotre choix (1/2/3) : ").strip()
        if choice in ("1", "2", "3"):
            break
        print("Choix invalide.")

    decisions = {
        "1": "approved",
        "2": "investigation_requested",
        "3": "rejected",
    }

    decision = {
        "incident_id": report["incident_id"],
        "recommended_action": report["plan"]["action"],
        "human_decision": decisions[choice],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "production_modified": False,
    }

    path = Path("artifacts_v3/human_decisions.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")

    print("\nDecision humaine :", decisions[choice])
    print("Production modifiee : False")
    print("Decision enregistree :", path)

    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["llm", "rules"],
        default="llm",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default="uncertain",
    )
    args = parser.parse_args()

    agent = CorrectionAgentV3(
        mode=args.mode,
        out_dir="artifacts_v3",
    )

    incident = SCENARIOS[args.scenario]
    report = agent.run(incident)

    print("\n=== RESULTAT DE L'AGENT ===")
    print("Action Groq :", report["proposed_plan"]["action"])
    print("Action finale :", report["plan"]["action"])
    print("Garde-fou :", report["guard_result"])
    print("Verification :", report["verification"])

    ask_human(report)


if __name__ == "__main__":
    main()
