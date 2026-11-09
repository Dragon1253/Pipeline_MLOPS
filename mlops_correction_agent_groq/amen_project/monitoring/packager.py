"""Écrit le paquet d'alerte au format exact attendu par ``mlops_churn.agent.investigate``.

Contrat (voir README de l'agent) :

    episode_dir/
    ├── alert.json
    ├── context.json            # historique, seuil, empreintes des versions
    ├── active.joblib           # bundle réellement servi
    ├── monitoring_context.json # ajout du monitoring : signaux et chronologie (non lu par l'agent)
    ├── focus/{source,inputs,predictions,feedback}.csv + monitoring.json
    └── confirmation-*/…        # mêmes fichiers ; l'agent lit monitoring.json

Règles : aucune vérité de simulation n'est écrite dans ce dossier ; les empreintes
SHA-256 de context.json correspondent aux fichiers de modèle, sinon l'agent refuse
la comparaison de versions.
"""
from pathlib import Path
import shutil

import joblib

from mlops_churn.workspace import sha256, utc_now, write_json


def model_version(path):
    return "sha256:" + sha256(path)


def _write_window(folder, record):
    folder.mkdir(parents=True, exist_ok=False)
    record.window.source.to_csv(folder / "source.csv", index=False)
    record.window.inputs.to_csv(folder / "inputs.csv", index=False)
    record.predictions_frame().to_csv(folder / "predictions.csv", index=False)
    record.feedback_frame().to_csv(folder / "feedback.csv", index=False)
    write_json(folder / "monitoring.json", record.monitoring)


def write_episode(destination, reference_dir, monitor, incident, active_model_path, *, deployment_events=None):
    """Crée ``destination`` (nouveau dossier) et y écrit le paquet de l'incident."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Le dossier d'épisode existe déjà : {destination}")
    reference_model = Path(reference_dir) / "models" / "reference.joblib"
    focus = monitor.records[incident.focus]
    confirmations = [monitor.records[w] for w in incident.confirmations[: monitor.confirmations]
                     if monitor.records[w].labeled_at is not None]
    destination.mkdir(parents=True)
    shutil.copyfile(active_model_path, destination / "active.joblib")
    active_version = model_version(destination / "active.joblib")
    if active_version != focus.model_version:
        raise ValueError("La version servie ne correspond pas aux prédictions de la fenêtre.")
    _write_window(destination / "focus", focus)
    for index, record in enumerate(confirmations, 1):
        _write_window(destination / f"confirmation-{index}", record)
    roles = {focus.window_id: "focus", **{r.window_id: f"confirmation-{i}" for i, r in enumerate(confirmations, 1)}}
    last = confirmations[-1].window_id if confirmations else focus.window_id
    history = [{**row, "role": roles.get(row["window"], "before_alert")}
               for row in monitor.history(until_window=last, last=8 + len(confirmations))]
    reference_version = model_version(reference_model)
    write_json(destination / "alert.json", incident.alert)
    write_json(destination / "context.json", {
        "episode_id": destination.name,
        "threshold": monitor.threshold,
        "performance_history": history,
        "deployment": {
            "reference_version": reference_version,
            "active_version": active_version,
            "model_changed": active_version != reference_version,
            "preprocessing_changed": False,
            "events": deployment_events or [],
        },
    })
    write_json(destination / "monitoring_context.json", {
        "created_at": utc_now(),
        "producer": "monitoring",
        "alert_schema_version": incident.alert["alert_schema_version"],
        "label_delay_windows": monitor.label_delay,
        "persistence_windows": monitor.persistence,
        "windows": [{"window": r.window_id, "role": roles.get(r.window_id, "other"),
                     "scored_at_step": r.scored_at, "labeled_at_step": r.labeled_at,
                     "log_loss": (r.metrics or {}).get("log_loss"), "signals": r.signals}
                    for r in (monitor.records[w] for w in monitor.order)],
        "note": "Signaux calculés sans labels au moment du scoring ; ils ne désignent aucune cause.",
    })
    incident.packaged = True
    return destination


def save_active_model(bundle, directory, reference_dir):
    """Chemin du fichier modèle servi : la référence elle-même, ou une copie sauvegardée du candidat."""
    reference_model = Path(reference_dir) / "models" / "reference.joblib"
    if bundle is None:
        return reference_model
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "candidate.joblib"
    if not path.exists():
        joblib.dump(bundle, path)
    return path
