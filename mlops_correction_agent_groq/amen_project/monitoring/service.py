"""Orchestration du monitoring : flux → scoring et signaux → labels retardés → alerte → paquet.

``run_stream`` est le point d'entrée réutilisable par la CLI, les tests, l'évaluation
et, plus tard, la plateforme (``on_event`` reçoit les mêmes événements que l'interface).
La vérité de simulation est écrite à part, dans ``evaluation/``, hors du dossier d'épisode.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import joblib
import pandas as pd

from mlops_churn.workspace import ROOT, write_json

from .monitor import PerformanceMonitor, load_calibration
from .packager import model_version, save_active_model, write_episode
from .signals import input_signals, prediction_signals
from .stream import StreamSpec, simulate_stream

DEFAULT_REFERENCE = ROOT / "scenario_data" / "demo"
DEFAULT_OUTPUT = ROOT / "monitoring_runs"


class ReferenceContext:
    """Artefacts figés partagés avec l'agent : modèle, population et calibration de référence."""

    def __init__(self, reference_dir=DEFAULT_REFERENCE):
        self.dir = Path(reference_dir)
        self.model_path = self.dir / "models" / "reference.joblib"
        self.bundle = joblib.load(self.model_path)
        self.calibration = load_calibration(self.dir)
        self.frame = pd.read_csv(self.dir / "reference_data.csv")
        self.quality = json.loads((self.dir / "quality_reference.json").read_text(encoding="utf-8"))["profile"]
        features = self.bundle["features"]
        self.probabilities = self.bundle["pipeline"].predict_proba(self.frame[features])[:, 1]


def run_stream(spec, *, reference=None, output_dir=DEFAULT_OUTPUT, label_delay=1, persistence=1,
               recovery=2, confirmations=2, package=True, on_event=None, run_id=None):
    """Exécute un flux complet et retourne un résumé JSON-sérialisable."""
    reference = reference or ReferenceContext()
    # Identifiant neutre : il apparaît dans alert.json et les identifiants clients transmis à
    # l'agent, il ne doit donc jamais contenir le nom du scénario.
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / run_id
    events, truth_rows = [], []

    def emit(event):
        events.append(event)
        if on_event:
            on_event(event)

    monitor = PerformanceMonitor(reference.calibration, label_delay=label_delay, persistence=persistence,
                                 recovery=recovery, confirmations=confirmations, run_id=run_id)
    model_paths, deployment_events, episodes = {}, [], []

    def package_ready(stream_finished):
        for incident in monitor.incidents:
            if package and monitor.ready_for_packaging(incident, stream_finished):
                focus = monitor.records[incident.focus]
                episode_dir = write_episode(run_dir / "episodes" / f"episode-{incident.focus}",
                                            reference.dir, monitor, incident, model_paths[focus.window_id],
                                            deployment_events=[e for e in deployment_events
                                                               if e["window_index"] <= focus.window.index])
                episodes.append(str(episode_dir))
                emit({"event": "episode_packaged", "alert_id": incident.alert["alert_id"],
                      "episode_dir": str(episode_dir)})

    step = -1
    for window, truth in simulate_stream(spec, reference.bundle, run_id=run_id):
        step = window.index
        truth_rows.append(truth)
        path = save_active_model(window.active if window.model_changed else None, run_dir / "models", reference.dir)
        version = model_version(path)
        if window.model_changed and not deployment_events:
            deployment_events.append({"event": "model_activated", "window": window.window_id,
                                      "window_index": window.index, "version": version})
        model_paths[window.window_id] = path
        features = window.active["features"]
        probabilities = window.active["pipeline"].predict_proba(window.inputs[features])[:, 1]
        signals = {**prediction_signals(probabilities, reference.probabilities),
                   **input_signals(window.inputs, reference.frame, reference.quality)}
        monitor.score(window, version, step, signals)
        emit({"event": "window_scored", "window": window.window_id, "step": step,
              "model_version": version, "signals": signals})
        for event in monitor.advance(step):
            emit(event)
        package_ready(stream_finished=False)
    # Après la dernière fenêtre, le temps passe encore : les derniers labels arrivent.
    for extra in range(1, label_delay + 1):
        for event in monitor.advance(step + extra):
            emit(event)
        package_ready(stream_finished=False)
    package_ready(stream_finished=True)

    alerts = [i.alert for i in monitor.incidents]
    summary = {
        "run_id": run_id, "run_dir": str(run_dir), "spec": spec.__dict__,
        "label_delay": label_delay, "persistence": persistence, "threshold": monitor.threshold,
        "alerts": alerts, "episodes": episodes,
        "incidents": [{"alert_id": i.alert["alert_id"], "focus": i.focus, "severity": i.alert["severity"],
                       "confirmations": i.confirmations, "close_reason": i.close_reason}
                      for i in monitor.incidents],
        "critical_threshold": monitor.critical_threshold,
        "first_alert_window": alerts[0]["window_id"] if alerts else None,
        "first_alert_step": alerts[0]["detected_at_step"] if alerts else None,
        "timeline": [{"window": w, "scored_at": r.scored_at, "labeled_at": r.labeled_at,
                      "log_loss": (r.metrics or {}).get("log_loss"),
                      "above_threshold": None if r.monitoring is None else r.monitoring["alert"] is not None}
                     for w, r in monitor.records.items()],
    }
    if package:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation").mkdir(exist_ok=True)
        # Vérité de simulation : réservée à l'évaluation, jamais copiée dans un épisode.
        write_json(run_dir / "evaluation" / "truth.json", {"spec": spec.__dict__, "windows": truth_rows})
        write_json(run_dir / "summary.json", summary)
    summary["events"] = events
    return summary


def default_spec(scenario, seed, **overrides):
    return StreamSpec(scenario=scenario, seed=seed, **overrides)
