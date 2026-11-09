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
               recovery=2, confirmations=2, package=True, on_event=None, run_id=None,
               predictor=None, predictive_recovery=2):
    """Exécute un flux complet et retourne un résumé JSON-sérialisable.

    Avec ``predictor`` (``predictor.load_predictor()``), chaque fenêtre reçoit aussi un risque
    de dégradation, calculé sans attendre ses labels, et une alerte ``predictive`` est levée
    au-delà du seuil du prédicteur. Elle met le système en vigilance (fichier dans
    ``predictive_alerts/``) ; l'enquête de l'agent reste déclenchée par l'alerte confirmée,
    qui mentionne l'alerte prédictive qui l'a précédée.
    """
    from .dataset import window_features
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
    history, predictive, open_predictive = [], [], None

    def predictive_context():
        return {"predictor_version": predictor.version, "predictor_threshold": predictor.threshold,
                "horizon_windows": predictor.horizon,
                "predictive_alerts": [{k: v for k, v in a.items() if k != "features"} for a in predictive]} \
            if predictor else None

    def package_ready(stream_finished):
        for incident in monitor.incidents:
            if package and monitor.ready_for_packaging(incident, stream_finished):
                focus = monitor.records[incident.focus]
                episode_dir = write_episode(run_dir / "episodes" / f"episode-{incident.focus}",
                                            reference.dir, monitor, incident, model_paths[focus.window_id],
                                            deployment_events=[e for e in deployment_events
                                                               if e["window_index"] <= focus.window.index],
                                            extra_context=predictive_context())
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
        confirmed_now = []
        for event in monitor.advance(step):
            emit(event)
            if event["event"] == "alert_triggered":
                confirmed_now.append(event["alert"])
        if predictor is not None:
            open_predictive = _predict_step(predictor, monitor, window, step, signals, history, predictive,
                                            open_predictive, confirmed_now, label_delay, predictive_recovery,
                                            run_id, emit, window_features,
                                            run_dir / "predictive_alerts" if package else None)
        package_ready(stream_finished=False)
    # Après la dernière fenêtre, le temps passe encore : les derniers labels arrivent.
    for extra in range(1, label_delay + 1):
        for event in monitor.advance(step + extra):
            emit(event)
            if event["event"] == "alert_triggered" and _follows(open_predictive, event["alert"]):
                _link(open_predictive, event["alert"], step + extra)
                open_predictive = None
        package_ready(stream_finished=False)
    if open_predictive is not None:
        open_predictive["status"] = "open_at_end"
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
        "predictive_alerts": [{k: v for k, v in a.items() if k != "features"} for a in predictive],
        "risk_timeline": [{"window": w, "risk": getattr(r, "risk", None)} for w, r in monitor.records.items()]
        if predictor else [],
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


def _follows(predictive_alert, confirmed):
    """L'alerte confirmée valide l'alerte prédictive si elle porte sur une fenêtre de l'intervalle
    annoncé : de la fenêtre prédite jusqu'à ``horizon_windows`` fenêtres après. Plus ancienne, elle
    concerne un autre épisode ; plus tardive, la prédiction avait expiré."""
    if predictive_alert is None:
        return False
    start = int(predictive_alert["window_id"][1:])
    return start <= int(confirmed["window_id"][1:]) <= start + predictive_alert["horizon_windows"]


def _link(predictive_alert, confirmed, step):
    """Relie une alerte prédictive à l'alerte confirmée qui la suit."""
    predictive_alert.update(status="confirmed", confirmed_alert_id=confirmed["alert_id"],
                            confirmed_window=confirmed["window_id"], confirmed_at_step=step,
                            lead_steps=step - predictive_alert["detected_at_step"])
    confirmed["preceded_by_predictive_alert"] = {
        "alert_id": predictive_alert["alert_id"], "window_id": predictive_alert["window_id"],
        "risk": predictive_alert["risk"], "lead_steps": predictive_alert["lead_steps"],
        "top_signals": predictive_alert["top_signals"]}


def _predict_step(predictor, monitor, window, step, signals, history, predictive, open_predictive,
                  confirmed_now, label_delay, recovery, run_id, emit, window_features, folder):
    """Risque de la fenêtre courante et cycle de vie des alertes prédictives."""
    known = [{"window": r.window_id, "metrics": r.metrics} for r in monitor.records.values()
             if r.labeled_at is not None and r.labeled_at <= step]
    history.append(window_features(step, signals, known, label_delay, monitor.threshold))
    features = predictor.features(history)
    risk = predictor.risk(features)
    monitor.records[window.window_id].risk = risk
    above = risk >= predictor.threshold
    emit({"event": "risk_recorded", "window": window.window_id, "step": step, "risk": risk,
          "threshold": predictor.threshold, "above_threshold": above,
          "features": dict(zip(predictor.columns, features.iloc[0].tolist()))})
    for confirmed in confirmed_now:
        if _follows(open_predictive, confirmed):
            _link(open_predictive, confirmed, step)
            open_predictive = None
    if open_predictive is not None:
        # Les labels de toute la période annoncée sont arrivés sans alerte confirmée : expirée.
        horizon_end = int(open_predictive["window_id"][1:]) + open_predictive["horizon_windows"]
        if step - horizon_end > label_delay:
            # Pendant un incident déjà confirmé et ouvert, aucune nouvelle alerte confirmée ne peut
            # valider la prédiction : c'est un rappel de vigilance, pas une fausse alerte.
            during = monitor.open_incident is not None
            open_predictive["status"] = "during_open_incident" if during else "expired"
            emit({"event": "predictive_alert_closed", "alert_id": open_predictive["alert_id"], "step": step,
                  "reason": "incident_already_open" if during else "no_confirmed_degradation_within_horizon"})
            open_predictive = None
    if open_predictive is not None:
        open_predictive["below_streak"] = 0 if above else open_predictive["below_streak"] + 1
        if open_predictive["below_streak"] >= recovery:
            open_predictive["status"] = "not_confirmed"
            emit({"event": "predictive_alert_closed", "alert_id": open_predictive["alert_id"], "step": step,
                  "reason": "risk_back_below_threshold"})
            open_predictive = None
    # Une alerte confirmée « warning » encore ouverte (souvent une fluctuation) ne doit pas
    # masquer un risque élevé : seul un incident critique déjà confirmé rend l'alerte inutile.
    # Une alerte prédictive couvre sa fenêtre et les ``horizon`` suivantes : la suivante ne
    # peut partir qu'après cette période (pas de doublon pour une même dégradation).
    incident = monitor.open_incident
    critical_open = incident is not None and incident.alert["severity"] == "critical"
    covered = bool(predictive) and step <= int(predictive[-1]["window_id"][1:]) + predictor.horizon
    if open_predictive is None and above and not critical_open and not covered:
        explanation = predictor.explain(features)
        labeled = [r.window_id for r in monitor.records.values() if r.labeled_at is not None and r.labeled_at <= step]
        alert = {"alert_id": f"{run_id}-{window.window_id}-predictive", "alert_type": "predictive",
                 "alert_schema_version": 1, "window_id": window.window_id, "detected_at_step": step,
                 "risk": risk, "threshold": predictor.threshold, "horizon_windows": predictor.horizon,
                 "labels_available_until": labeled[-1] if labeled else None,
                 "top_signals": explanation["signals"],
                 "risk_if_signals_usual": explanation["risk_if_signals_usual"],
                 "predictor_version": predictor.version, "status": "open", "below_streak": 0,
                 "note": "Alerte de vigilance sans labels : signaux mesurés, aucune cause désignée."}
        predictive.append(alert)
        open_predictive = alert
        emit({"event": "predictive_alert", "alert": {k: v for k, v in alert.items() if k != "below_streak"}})
        if folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            write_json(Path(folder) / f"{alert['alert_id']}.json",
                       {k: v for k, v in alert.items() if k not in ("below_streak", "status")})
    return open_predictive


def default_spec(scenario, seed, **overrides):
    return StreamSpec(scenario=scenario, seed=seed, **overrides)
