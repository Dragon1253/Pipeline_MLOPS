import argparse
import json
from agent_v3 import CorrectionAgentV3

SCENARIOS = [
    {"incident_id":"S1", "type":"derive_donnees", "confidence":0.95,
     "metrics":{"mae_before":38,"mae_current":51,"psi":0.34}, "details":{"batch_id":"week14"}},
    {"incident_id":"S2", "type":"fausse_alerte", "confidence":0.86,
     "metrics":{"mae_before":38,"mae_current":39,"psi":0.27}, "details":{}},
    {"incident_id":"S3", "type":"qualite_donnees", "confidence":0.93,
     "metrics":{"missing_rate":0.31,"psi":4.8}, "details":{"batch_id":"week30"}},
    {"incident_id":"S4", "type":"regression_version", "confidence":0.84,
     "metrics":{"mae_old":40,"mae_new":49}, "details":{"version":"v2"}},
    {"incident_id":"S5", "type":"inconnu", "confidence":0.45,
     "metrics":{}, "details":{}},
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rules","llm"], default="rules")
    args = parser.parse_args()
    agent = CorrectionAgentV3(mode=args.mode)
    for incident in SCENARIOS:
        result = agent.run(incident)
        print(f"{result['incident_id']} | {result['mode']} | {result['plan']['action']} | "
              f"{result['tool_result']['status']} | safe={result['verification']['safe']}")
    print("Traces: artifacts_v3/agent_traces.jsonl")
