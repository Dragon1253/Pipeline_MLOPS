from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

ALLOWED = {'fausse_alerte','derive_donnees','qualite_donnees','regression_version'}

@dataclass
class Incident:
    incident_id: str
    type: str
    confidence: float
    metrics: dict
    details: dict

class CorrectionAgent:
    def __init__(self, log_path='decisions.jsonl'):
        self.log_path = Path(log_path)

    def decide(self, incident: Incident) -> dict:
        if incident.type not in ALLOWED:
            raise ValueError(f'Type inconnu: {incident.type}')
        if not 0 <= incident.confidence <= 1:
            raise ValueError('confidence doit être entre 0 et 1')
        if incident.confidence < 0.7:
            action, status, reason = 'human_review', 'pending_approval', 'Diagnostic insuffisamment fiable'
        elif incident.type == 'qualite_donnees':
            action, status, reason = 'quarantine_batch', 'executed', 'Données suspectes : interdire le réentraînement'
        elif incident.type == 'fausse_alerte':
            action, status, reason = 'close_alert', 'executed', 'Aucune correction nécessaire'
        elif incident.type == 'derive_donnees':
            action, status, reason = 'request_retraining', 'pending_approval', 'Préparer un candidat, évaluer puis demander validation avant déploiement'
        else:
            action, status, reason = 'request_rollback', 'pending_approval', 'Rollback uniquement après comparaison des versions et validation humaine'
        result = {'incident_id': incident.incident_id, 'action': action, 'status': status, 'reason': reason, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result
