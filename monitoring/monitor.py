"""Monitoring réactif réel : métriques calculées à l'arrivée des labels, seuil, alerte.

Remplace la relecture scénarisée de la plateforme (alerte toujours posée sur
``focus``) par une décision mesurée fenêtre après fenêtre :

1. chaque fenêtre est scorée dès son arrivée (prédictions + signaux sans labels) ;
2. ses labels arrivent ``label_delay`` fenêtres plus tard ;
3. les métriques (log loss, accuracy, précision, recall, F1, ROC-AUC, AP) sont alors
   calculées avec ``mlops_churn.scenarios.metrics`` ;
4. une alerte est confirmée quand la log loss dépasse le seuil calibré sur
   ``persistence`` fenêtres étiquetées consécutives ;
5. l'incident reste ouvert, sans nouvelle alerte, jusqu'au retour sous le seuil
   pendant ``recovery`` fenêtres.

Le seuil est celui de ``scenario_data/demo/calibration.json`` (quantile 95 % de
30 fenêtres normales + marge), le même que celui utilisé par l'agent et la plateforme.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_churn.scenarios import metrics, monitor_performance

ALERT_SCHEMA_VERSION = 1


def load_calibration(reference_dir):
    calibration = json.loads((Path(reference_dir) / "calibration.json").read_text(encoding="utf-8"))
    for key in ("threshold", "reference_value", "min_labeled_rows"):
        if key not in calibration:
            raise ValueError(f"calibration.json : champ {key} absent.")
    return calibration


@dataclass
class WindowRecord:
    window: object
    probabilities: np.ndarray = field(repr=False)
    model_version: str
    scored_at: int
    signals: dict = field(default_factory=dict)
    labeled_at: int | None = None
    metrics: dict | None = None
    monitoring: dict | None = None

    @property
    def window_id(self):
        return self.window.window_id

    def predictions_frame(self):
        threshold = self.window.active["threshold"]
        return pd.DataFrame({"customer_id": self.window.inputs["customer_id"],
                             "probability": self.probabilities,
                             "prediction": (self.probabilities >= threshold).astype(int),
                             "model_version": self.model_version})

    def feedback_frame(self):
        if self.labeled_at is None:
            raise ValueError(f"Labels de {self.window_id} pas encore disponibles.")
        return pd.DataFrame({"customer_id": self.window.inputs["customer_id"], "churn": self.window.labels})


@dataclass
class Incident:
    alert: dict
    focus: str
    opened_at: int
    confirmations: list = field(default_factory=list)
    below_streak: int = 0
    closed_at: int | None = None
    close_reason: str | None = None
    packaged: bool = False


class PerformanceMonitor:
    """Décide des alertes à partir des seules observations disponibles à chaque instant."""

    def __init__(self, calibration, *, label_delay=1, persistence=1, recovery=2, confirmations=2,
                 critical_factor=3.0, run_id="stream"):
        if label_delay < 0 or persistence < 1 or recovery < 1 or confirmations < 0 or critical_factor <= 1:
            raise ValueError("Paramètres de monitoring invalides.")
        self.calibration = calibration
        self.critical_factor = critical_factor
        self.label_delay, self.persistence = label_delay, persistence
        self.recovery, self.confirmations = recovery, confirmations
        self.run_id = run_id
        self.records: dict[str, WindowRecord] = {}
        self.order: list[str] = []
        self.above_streak: list[str] = []
        self.incidents: list[Incident] = []

    @property
    def threshold(self):
        return self.calibration["threshold"]

    @property
    def critical_threshold(self):
        """Niveau critique : écart à la référence ``critical_factor`` fois celui du seuil d'alerte."""
        reference = self.calibration["reference_value"]
        return reference + self.critical_factor * (self.threshold - reference)

    @property
    def open_incident(self):
        return self.incidents[-1] if self.incidents and self.incidents[-1].closed_at is None else None

    def score(self, window, model_version, step, signals=None):
        """Scoring à l'arrivée de la fenêtre : aucune métrique avec labels n'est disponible."""
        features = window.active["features"]
        probabilities = np.asarray(window.active["pipeline"].predict_proba(window.inputs[features])[:, 1], dtype=float)
        record = WindowRecord(window, probabilities, model_version, step, signals or {})
        self.records[window.window_id] = record
        self.order.append(window.window_id)
        return record

    def advance(self, step):
        """Fait arriver les labels dus à ``step`` et applique la règle d'alerte. Retourne les événements."""
        events = []
        for window_id in self.order:
            record = self.records[window_id]
            if record.labeled_at is None and record.scored_at + self.label_delay <= step:
                events.extend(self._label(record, step))
        return events

    def _label(self, record, step):
        record.labeled_at = step
        labels = record.window.labels
        measures = metrics(labels, record.probabilities, record.window.active["threshold"])
        decision = monitor_performance(measures, len(labels), self.calibration, record.window_id)
        record.metrics = measures
        record.monitoring = {"metrics": measures, "labeled_rows": int(len(labels)), **decision}
        events = [{"event": "metric_recorded", "window": record.window_id, "step": step,
                   "metrics": measures, "labeled_rows": int(len(labels)), "status": decision["status"]}]
        above = decision["alert"] is not None
        incident = self.open_incident
        if incident is not None and record.model_version != self.records[incident.focus].model_version:
            # Une nouvelle version servie est un nouveau contexte : l'incident précédent
            # ne doit pas absorber une éventuelle régression de cette version.
            incident.closed_at, incident.close_reason = step, "model_version_changed"
            events.append({"event": "incident_closed", "window": record.window_id, "step": step,
                           "alert_id": incident.alert["alert_id"], "reason": "model_version_changed"})
            self.above_streak, incident = [], None
        if (incident is not None and incident.alert["severity"] == "warning"
                and measures["log_loss"] >= self.critical_threshold):
            # Escalade : une alerte faible (souvent une fluctuation) ne doit pas absorber une
            # dégradation critique ; celle-ci reçoit sa propre alerte, centrée sur sa fenêtre.
            incident.closed_at, incident.close_reason = step, "escalated"
            events.append({"event": "incident_closed", "window": record.window_id, "step": step,
                           "alert_id": incident.alert["alert_id"], "reason": "escalated"})
            self.above_streak, incident = [], None
        if incident is not None:
            incident.confirmations.append(record.window_id)
            incident.below_streak = 0 if above else incident.below_streak + 1
            reason = None
            if incident.below_streak >= self.recovery:
                reason = "recovered"
            elif len(incident.confirmations) == self.confirmations and not all(
                    self.records[w].monitoring["alert"] for w in incident.confirmations):
                # Les confirmations repassent sous le seuil : l'alerte n'était pas persistante.
                # Elle est transmise telle quelle et la surveillance reprend, pour qu'un
                # incident ultérieur reçoive sa propre alerte au lieu d'être absorbé.
                reason = "not_persistent"
            if reason is None:
                return events
            incident.closed_at, incident.close_reason = step, reason
            events.append({"event": "incident_closed", "window": record.window_id, "step": step,
                           "alert_id": incident.alert["alert_id"], "reason": reason})
            trailing = []
            for window_id in incident.confirmations:
                trailing = trailing + [window_id] if self.records[window_id].monitoring["alert"] else []
            self.above_streak = trailing if reason == "not_persistent" else []
        else:
            self.above_streak = self.above_streak + [record.window_id] if above else []
        if self.above_streak and len(self.above_streak) >= self.persistence:
            focus = self.above_streak[0]
            base = self.records[focus].monitoring["alert"] or decision["alert"]
            alert = {**base,
                     "alert_id": f"{self.run_id}-{focus}-alert",
                     "window_id": focus,
                     "alert_type": "confirmed",
                     "severity": "critical" if base["observed_value"] >= self.critical_threshold else "warning",
                     "critical_threshold": self.critical_threshold,
                     "alert_schema_version": ALERT_SCHEMA_VERSION,
                     "persistence_windows": self.persistence,
                     "windows_above_threshold": list(self.above_streak),
                     "label_delay_windows": self.label_delay,
                     "detected_at_step": step}
            incident = Incident(alert=alert, focus=focus, opened_at=step,
                                confirmations=list(self.above_streak[1:]))
            self.incidents.append(incident)
            self.above_streak = []
            events.append({"event": "alert_triggered", "window": focus, "step": step, "alert": alert})
        return events

    def ready_for_packaging(self, incident, stream_finished=False):
        """Le paquet part quand les confirmations attendues sont étiquetées (ou en fin de flux)."""
        if incident.packaged:
            return False
        return (len(incident.confirmations) >= self.confirmations or incident.closed_at is not None
                or stream_finished)

    def history(self, until_window=None, last=8):
        """Historique des fenêtres étiquetées, au format ``performance_history`` de context.json."""
        rows = []
        for window_id in self.order:
            record = self.records[window_id]
            if record.metrics is None:
                continue
            rows.append({"window": window_id, "metrics": record.metrics,
                         "labeled_rows": record.monitoring["labeled_rows"],
                         "alert": record.monitoring["alert"] is not None})
            if window_id == until_window:
                break
        return rows[-last:] if last else rows
