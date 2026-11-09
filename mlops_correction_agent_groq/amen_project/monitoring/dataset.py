"""Jeu de données pour prédire l'alerte : une ligne par fenêtre de flux simulé.

Chaque flux est produit par ``run_stream`` (étape 1). Pour chaque fenêtre ``t`` :

* ``sig_*``   : signaux sans labels calculés au scoring de ``t`` (connus à ``t``) ;
* ``known_*`` : métriques des fenêtres dont les labels sont arrivés au plus tard à ``t``
  (reconstruites à partir des événements ``metric_recorded`` de pas <= t) ;
* ``react_*`` : décision du monitoring réactif de l'étape 1 au pas ``t`` (référence R1) ;
* ``label_log_loss`` et ``target_*`` : ce qu'on veut prédire (futur, jamais une entrée) ;
* ``truth_*`` : scénario et intensité, réservés à l'analyse des résultats.

Cibles :
* ``target_now``  : la fenêtre ``t`` elle-même dépasse le seuil (le monitoring réactif ne le
  saura qu'à ``t + label_delay``) ;
* ``target_h{k}`` : au moins une des fenêtres ``t+1 … t+k`` dépasse le seuil.

Les configurations de flux sont tirées à partir de la graine du flux seulement, avant
toute simulation ; les plages de graines des découpages sont disjointes.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import math
from pathlib import Path
import tempfile

import numpy as np

from mlops_churn.workspace import sha256, write_json

from .service import DEFAULT_OUTPUT, ReferenceContext, run_stream
from .stream import INCIDENTS, StreamSpec

SPLITS = {
    # nom : (première graine, nombre de flux, variante des mécanismes)
    "train": (1, 2000, "standard"),
    "validation": (10_001, 500, "standard"),
    "test": (20_001, 500, "standard"),
    "generalization": (30_001, 300, "heldout"),
}
GRID = {
    "normal_share": 0.4,
    "scenarios": list(INCIDENTS),
    "max_intensity": [0.25, 0.5, 1.0],
    "ramp": [0, 2, 4, 8],
    "onset": [6, 14],          # bornes incluses, tirage uniforme
    "label_delay": [1, 2, 3],
    "n_windows": 24,
}
HORIZONS = (1, 2, 3)
KNOWN_METRICS = ("log_loss", "accuracy", "precision", "recall", "f1", "roc_auc")


def stream_config(seed, variant, grid=GRID):
    """Configuration d'un flux, tirée uniquement à partir de sa graine."""
    rng = np.random.default_rng([seed, 2026])
    normal = rng.random() < grid["normal_share"]
    return {
        "seed": int(seed), "variant": variant,
        "scenario": "normal" if normal else str(rng.choice(grid["scenarios"])),
        "max_intensity": float(rng.choice(grid["max_intensity"])),
        "ramp": int(rng.choice(grid["ramp"])),
        "onset": int(rng.integers(grid["onset"][0], grid["onset"][1] + 1)),
        "label_delay": int(rng.choice(grid["label_delay"])),
        "n_windows": int(grid["n_windows"]),
    }


def _slope(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def stream_rows(config, reference, output_dir, split=""):
    """Simule un flux et le transforme en lignes (une par fenêtre)."""
    spec = StreamSpec(config["scenario"], config["seed"], n_windows=config["n_windows"],
                      onset=config["onset"], ramp=config["ramp"],
                      max_intensity=config["max_intensity"], variant=config["variant"])
    summary = run_stream(spec, reference=reference, output_dir=output_dir,
                         label_delay=config["label_delay"], package=False, run_id=f"ds-{config['seed']}")
    truth = {row["window"]: row for row in _truth_rows(spec)}
    threshold = summary["threshold"]
    scored = {e["window"]: e for e in summary["events"] if e["event"] == "window_scored"}
    recorded = [e for e in summary["events"] if e["event"] == "metric_recorded"]
    alerts_at = {e["step"]: e["alert"] for e in summary["events"] if e["event"] == "alert_triggered"}
    losses = [row["log_loss"] for row in summary["timeline"]]
    rows = []
    for index, timeline in enumerate(summary["timeline"]):
        window = timeline["window"]
        # Uniquement les labels arrivés au plus tard au pas courant.
        known = sorted((e for e in recorded if e["step"] <= index), key=lambda e: int(e["window"][1:]))
        recent = known[-3:][::-1]
        row = {"split": split, "stream_id": config["seed"], "window": window, "index": index,
               "label_delay": config["label_delay"],
               **{f"sig_{k}": v for k, v in scored[window]["signals"].items()},
               "known_windows": len(known),
               "known_last_window_age": index - int(known[-1]["window"][1:]) if known else None}
        for lag in range(3):
            row[f"known_log_loss_lag{lag}"] = recent[lag]["metrics"]["log_loss"] if lag < len(recent) else None
        for name in KNOWN_METRICS[1:]:
            row[f"known_{name}"] = recent[0]["metrics"][name] if recent else None
        row["known_log_loss_slope3"] = _slope([e["metrics"]["log_loss"] for e in known[-3:]])
        row["known_above_threshold_last3"] = sum(e["metrics"]["log_loss"] > threshold for e in known[-3:])
        alert = alerts_at.get(index)
        row["react_alert_at_step"] = int(alert is not None)
        row["react_alert_window"] = alert["window_id"] if alert else None
        row["react_alert_severity"] = alert["severity"] if alert else None
        row["label_log_loss"] = losses[index]
        row["target_now"] = int(losses[index] > threshold)
        for horizon in HORIZONS:
            future = losses[index + 1: index + 1 + horizon]
            row[f"target_h{horizon}"] = int(any(v > threshold for v in future)) if len(future) == horizon else None
        t = truth[window]
        row.update({"truth_scenario": spec.scenario, "truth_variant": spec.variant,
                    "truth_intensity": t["intensity"], "truth_incident_active": int(t["incident_active"]),
                    "truth_onset": spec.onset if spec.scenario != "normal" else None,
                    "truth_ramp": spec.ramp, "truth_max_intensity": spec.max_intensity})
        rows.append(row)
    return rows


def _truth_rows(spec):
    from .stream import intensity
    return [{"window": f"w{i:03d}", "intensity": intensity(spec, i), "incident_active": intensity(spec, i) > 0}
            for i in range(spec.n_windows)]


_WORKER = {}


def _init_worker():
    _WORKER["reference"] = ReferenceContext()
    _WORKER["scratch"] = tempfile.mkdtemp(prefix="monitoring-dataset-")


def _work(job):
    config, split = job
    return stream_rows(config, _WORKER["reference"], _WORKER["scratch"], split)


def build_split(split, *, output_dir=DEFAULT_OUTPUT / "dataset", n_streams=None, workers=1, progress=None):
    """Génère un découpage complet et son manifeste. Retourne le chemin du CSV."""
    first, count, variant = SPLITS[split]
    count = n_streams or count
    configs = [stream_config(seed, variant) for seed in range(first, first + count)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}.csv"
    jobs = [(config, split) for config in configs]
    writer, n_rows, positives = None, 0, {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        if workers > 1:
            executor = ProcessPoolExecutor(workers, initializer=_init_worker)
            results = executor.map(_work, jobs, chunksize=4)
        else:
            _init_worker()
            executor, results = None, map(_work, jobs)
        try:
            for done, rows in enumerate(results, 1):
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                n_rows += len(rows)
                for row in rows:
                    for key in ("target_now", *(f"target_h{h}" for h in HORIZONS)):
                        if row[key] is not None:
                            total, ones = positives.get(key, (0, 0))
                            positives[key] = (total + 1, ones + row[key])
                if progress:
                    progress(done, len(jobs))
        finally:
            if executor:
                executor.shutdown()
    scenarios = {}
    for config in configs:
        scenarios[config["scenario"]] = scenarios.get(config["scenario"], 0) + 1
    write_json(output_dir / f"{split}.manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "split": split, "variant": variant,
        "seeds": [first, first + count - 1], "streams": count, "rows": n_rows,
        "streams_per_scenario": scenarios, "grid": GRID, "horizons": list(HORIZONS),
        "positive_rate": {k: ones / total for k, (total, ones) in positives.items()},
        "csv_sha256": sha256(path),
        "columns": {"features": "sig_*, known_*, label_delay", "reactive_reference": "react_*",
                    "targets": "label_log_loss, target_*", "evaluation_only": "truth_*"},
    })
    return path
