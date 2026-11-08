"""Matrice de détection : le monitoring déclenche-t-il une alerte pour chaque scénario ?

Pour chaque scénario et chaque graine, un flux complet est simulé, puis comparé à
la vérité réservée à l'évaluation (début de l'incident). Mesures par scénario :

* ``detection_rate`` : part des flux où une alerte est posée sur une fenêtre touchée ;
* ``mean_delay_windows`` : fenêtres entre le début de l'incident et la fenêtre en alerte ;
* ``mean_detection_lag_steps`` : délai réel de détection, retard des labels compris ;
* ``false_alerts_per_100_windows`` : alertes posées sur des fenêtres sans incident.

Les graines sont fixées avant l'exécution ; aucun flux n'est retiré après coup.
"""
from datetime import datetime, timezone
from pathlib import Path
import statistics
import tempfile

from mlops_churn.workspace import write_json

from .service import DEFAULT_OUTPUT, ReferenceContext, run_stream
from .stream import STREAM_SCENARIOS, StreamSpec


def evaluate(scenarios=STREAM_SCENARIOS, seeds=range(1, 31), *, n_windows=20, onset=8, ramp=4,
             max_intensity=1.0, label_delay=1, persistence=1, reference=None):
    reference = reference or ReferenceContext()
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for scenario in scenarios:
            for seed in seeds:
                spec = StreamSpec(scenario, seed, n_windows=n_windows, onset=onset, ramp=ramp,
                                  max_intensity=max_intensity)
                summary = run_stream(spec, reference=reference, output_dir=scratch, label_delay=label_delay,
                                     persistence=persistence, package=False, run_id=f"eval-{seed}")
                incident = scenario != "normal"
                position = lambda window_id: int(window_id[1:])
                details = {i["alert_id"]: i for i in summary["incidents"]}
                true = [a for a in summary["alerts"] if incident and position(a["window_id"]) >= onset]
                # Alerte posée avant le début mais restée ouverte sur des fenêtres touchées :
                # la dégradation est surveillée, mais la fenêtre principale du paquet est saine.
                merged = [a for a in summary["alerts"] if incident and position(a["window_id"]) < onset
                          and any(position(w) >= onset for w in details[a["alert_id"]]["confirmations"])
                          and details[a["alert_id"]]["close_reason"] != "not_persistent"]
                false = [a for a in summary["alerts"] if a not in true and a not in merged]
                clean_windows = n_windows if not incident else onset
                first = true[0] if true else None
                rows.append({
                    "scenario": scenario, "seed": seed, "detected": bool(first),
                    "alert_window": first["window_id"] if first else None,
                    "severity": first["severity"] if first else None,
                    "delay_windows": position(first["window_id"]) - onset if first else None,
                    "detection_lag_steps": first["detected_at_step"] - onset if first else None,
                    "merged_with_earlier_alert": bool(merged) and not first,
                    "false_alerts": len(false), "clean_windows": clean_windows,
                })
    return {"config": {"scenarios": list(scenarios), "seeds": list(seeds), "n_windows": n_windows,
                       "onset": onset, "ramp": ramp, "max_intensity": max_intensity,
                       "label_delay": label_delay, "persistence": persistence,
                       "threshold": reference.calibration["threshold"]},
            "summary": summarize(rows, scenarios), "runs": rows}


def summarize(rows, scenarios):
    table = []
    for scenario in scenarios:
        group = [r for r in rows if r["scenario"] == scenario]
        detected = [r for r in group if r["detected"]]
        clean = sum(r["clean_windows"] for r in group)
        table.append({
            "scenario": scenario, "runs": len(group),
            "detected": len(detected) if scenario != "normal" else None,
            "detection_rate": len(detected) / len(group) if group and scenario != "normal" else None,
            "merged_with_earlier_alert": sum(r["merged_with_earlier_alert"] for r in group),
            "critical_alerts": sum(r["severity"] == "critical" for r in detected),
            "mean_delay_windows": statistics.mean(r["delay_windows"] for r in detected) if detected else None,
            "mean_detection_lag_steps": statistics.mean(r["detection_lag_steps"] for r in detected) if detected else None,
            "false_alerts": sum(r["false_alerts"] for r in group),
            "false_alerts_per_100_windows": 100 * sum(r["false_alerts"] for r in group) / clean if clean else None,
        })
    return table


def markdown(result):
    c = result["config"]
    lines = ["# Matrice de détection du monitoring", "",
             f"Seuil log loss : {c['threshold']:.4f} · {len(c['seeds'])} flux par scénario · "
             f"{c['n_windows']} fenêtres · début de l'incident : fenêtre {c['onset']} · montée sur {c['ramp']} fenêtres · "
             f"intensité max {c['max_intensity']} · retard des labels : {c['label_delay']} · persistance : {c['persistence']}", "",
             "| Scénario | Détection | Dont critiques | Fusionnées avec une alerte antérieure | Délai moyen (fenêtres) | Délai réel avec labels (pas) | Fausses alertes / 100 fenêtres saines |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    for row in result["summary"]:
        incident = row["detected"] is not None
        detection = f"{row['detected']} / {row['runs']}" if incident else "—"
        critical = row["critical_alerts"] if incident else "—"
        merged = row["merged_with_earlier_alert"] if incident else "—"
        lines.append(f"| {row['scenario']} | {detection} | {critical} | {merged} | {fmt(row['mean_delay_windows'])} | "
                     f"{fmt(row['mean_detection_lag_steps'])} | {fmt(row['false_alerts_per_100_windows'], 1)} |")
    lines += ["", "Le flux « normal » ne contient aucun incident : toute alerte y est une fausse alerte "
              "(c'est le scénario false_alert de l'agent). Le seuil étant le quantile 95 % de fenêtres "
              "normales, environ 5 fausses alertes pour 100 fenêtres saines sont attendues avec une persistance de 1."]
    return "\n".join(lines) + "\n"


def save(result, output_dir=DEFAULT_OUTPUT):
    folder = Path(output_dir) / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "detection_matrix.json", result)
    (folder / "detection_matrix.md").write_text(markdown(result), encoding="utf-8")
    return folder
