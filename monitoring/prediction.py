"""Étape 2.2 : prédire l'alerte avant l'arrivée des labels, et comparer aux références.

Méthodes comparées sur les mêmes flux :

* R1 ``reactive``      : monitoring de l'étape 1 (alerte quand les labels montrent la dégradation) ;
* R2 ``static_rules``  : règle classique sans labels, max(PSI des scores, PSI max des entrées) ;
* R3 ``expected_loss`` : log loss estimée sans labels (entropie moyenne des probabilités) ;
* M1 ``logistic``      : régression logistique sur les signaux et leurs tendances ;
* M2 ``boosting``      : gradient boosting (HistGradientBoosting) sur les mêmes entrées.

Protocole, fixé avant les résultats :
1. apprentissage sur ``train``, choix des hyperparamètres et du seuil d'alarme sur ``validation`` ;
2. seuil d'alarme de chaque méthode à score (R2, R3, M1, M2) choisi pour un même budget de
   fausses alarmes par 100 fenêtres saines : comparaison à coût égal ;
3. mesure finale, une seule fois, sur ``test`` puis sur ``generalization`` (mécanismes jamais vus).

Les entrées sont uniquement ``sig_*``, ``known_*``, ``label_delay`` et des tendances calculées
sur les fenêtres passées du même flux ; ``truth_*``, ``target_*``, ``react_*`` et
``label_log_loss`` n'en font jamais partie.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_churn.workspace import write_json

from .service import DEFAULT_OUTPUT

DATASET_DIR = DEFAULT_OUTPUT / "dataset"
TREND_SIGNALS = ["sig_mean_probability", "sig_predicted_positive_rate", "sig_expected_log_loss",
                 "sig_score_psi", "sig_psi_max", "sig_out_of_range_max", "sig_share_extreme",
                 "sig_share_uncertain", "sig_tv_max", "sig_std_probability"]
FORBIDDEN_PREFIXES = ("truth_", "target_", "react_", "label_log_loss")
METHODS = ("reactive", "static_rules", "expected_loss", "logistic", "boosting")
LABELS = {"reactive": "R1 monitoring réactif (étape 1)", "static_rules": "R2 seuils fixes sans labels",
          "expected_loss": "R3 log loss estimée sans labels", "logistic": "M1 régression logistique",
          "boosting": "M2 gradient boosting"}


def load_split(name, dataset_dir=DATASET_DIR):
    frame = pd.read_csv(Path(dataset_dir) / f"{name}.csv")
    return add_trend_features(frame.sort_values(["stream_id", "index"]).reset_index(drop=True))


def add_trend_features(frame):
    """Tendances par flux, construites uniquement avec la fenêtre courante et les précédentes."""
    grouped = frame.groupby("stream_id", sort=False)
    new = {}
    for column in TREND_SIGNALS:
        series = grouped[column]
        new[f"trend_{column}_diff1"] = series.diff()
        new[f"trend_{column}_mean3"] = series.transform(lambda s: s.rolling(3, min_periods=1).mean())
        past = series.transform(lambda s: s.shift(1).expanding().mean())
        new[f"trend_{column}_vs_past"] = frame[column] - past
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


def feature_columns(frame, groups=("sig", "known", "trend")):
    prefixes = tuple(f"{g}_" for g in groups)
    columns = [c for c in frame.columns if c.startswith(prefixes)]
    if "known" in groups:
        columns.append("label_delay")
    assert not [c for c in columns if c.startswith(FORBIDDEN_PREFIXES)], "fuite de colonnes"
    return columns


def baseline_scores(frame):
    return {
        "static_rules": np.fmax(frame["sig_score_psi"].fillna(0), frame["sig_psi_max"].fillna(0)).to_numpy(),
        "expected_loss": frame["sig_expected_log_loss"].to_numpy(),
    }


def make_model(kind, params):
    if kind == "logistic":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=3000, class_weight="balanced", **params))])
    return HistGradientBoostingClassifier(random_state=0, class_weight="balanced", **params)


GRIDS = {"logistic": [{"C": c} for c in (0.03, 0.3, 3.0)],
         "boosting": [{"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 300}
                      for lr in (0.05, 0.1) for leaves in (15, 31)]}


def fit_models(train, validation, target, columns, kinds=("logistic", "boosting")):
    """Choisit les hyperparamètres sur la validation (PR-AUC), puis garde le meilleur modèle."""
    models, tuning = {}, {}
    y_train, y_val = train[target], validation[target]
    for kind in kinds:
        results = []
        for params in GRIDS[kind]:
            model = make_model(kind, params).fit(train[columns], y_train)
            score = average_precision_score(y_val, model.predict_proba(validation[columns])[:, 1])
            results.append((score, params, model))
        best = max(results, key=lambda r: r[0])
        models[kind] = best[2]
        tuning[kind] = {"chosen": best[1], "validation_pr_auc": [{"params": p, "pr_auc": s} for s, p, _ in results]}
    return models, tuning


def all_scores(frame, models, columns):
    scores = baseline_scores(frame)
    for kind, model in models.items():
        scores[kind] = model.predict_proba(frame[columns])[:, 1]
    return scores


def clean_mask(frame):
    return frame["truth_incident_active"].to_numpy() == 0


def threshold_for_budget(scores, frame, budget):
    """Plus petit seuil donnant au plus ``budget`` fausses alarmes pour 100 fenêtres saines."""
    clean = np.sort(scores[clean_mask(frame)])[::-1]
    allowed = int(np.floor(budget / 100 * len(clean)))
    if allowed >= len(clean):
        return float(clean[-1])
    return float(np.nextafter(clean[allowed], np.inf))


def _window_position(window_id):
    return int(str(window_id)[1:])


def event_metrics(frame, alarms):
    """Détection par flux : premier pas où l'incident est signalé, avance sur le monitoring réactif.

    Pour une méthode sans labels, l'alarme de la fenêtre ``t`` est disponible au pas ``t``.
    Pour R1, le pas est celui où l'alerte est réellement levée (labels arrivés).
    """
    rows = []
    for stream_id, group in frame.groupby("stream_id", sort=False):
        group = group.sort_values("index")
        scenario = group["truth_scenario"].iloc[0]
        active = group["truth_incident_active"].to_numpy() == 1
        degraded = bool((active & (group["target_now"].to_numpy() == 1)).any())
        alarm = alarms[group.index.to_numpy()]
        hits = group["index"].to_numpy()[active & alarm]
        react = group[(group["react_alert_at_step"] == 1)]
        onset = group["truth_onset"].iloc[0]
        react_hits = [] if scenario == "normal" else [
            int(r["index"]) for _, r in react.iterrows() if _window_position(r["react_alert_window"]) >= onset]
        rows.append({"stream_id": stream_id, "scenario": scenario, "degraded": degraded,
                     "detected_step": int(hits[0]) if len(hits) else None,
                     "reactive_step": react_hits[0] if react_hits else None,
                     "onset": None if scenario == "normal" else int(onset)})
    return pd.DataFrame(rows)


def summarize_events(events, scenario=None):
    group = events[events["degraded"]]
    if scenario:
        group = group[group["scenario"] == scenario]
    if group.empty:
        return {"degraded_streams": 0}
    detected = group["detected_step"].notna()
    both = group.dropna(subset=["detected_step", "reactive_step"])
    lead = (both["reactive_step"] - both["detected_step"]).astype(float)
    delay = (group.loc[detected, "detected_step"] - group.loc[detected, "onset"]).astype(float)
    return {"degraded_streams": int(len(group)), "detection_rate": float(detected.mean()),
            "mean_delay_from_onset": float(delay.mean()) if len(delay) else None,
            "mean_lead_vs_reactive": float(lead.mean()) if len(lead) else None,
            "share_earlier_than_reactive": float((lead > 0).mean()) if len(lead) else None}


def reactive_alarms(frame):
    """Alarmes de R1, rattachées à la fenêtre alertée (pour compter ses fausses alarmes)."""
    flagged = np.zeros(len(frame), dtype=bool)
    lookup = {(s, i): k for k, (s, i) in enumerate(zip(frame["stream_id"], frame["index"]))}
    for _, row in frame[frame["react_alert_at_step"] == 1].iterrows():
        position = lookup.get((row["stream_id"], _window_position(row["react_alert_window"])))
        if position is not None:
            flagged[position] = True
    return flagged


def evaluate_split(frame, scores, thresholds, target):
    """Métriques par fenêtre et par flux pour chaque méthode."""
    results = {}
    y = frame[target].to_numpy()
    clean = clean_mask(frame)
    for method in METHODS:
        if method == "reactive":
            alarms = reactive_alarms(frame)
            window = {}
            events = event_metrics(frame, np.zeros(len(frame), dtype=bool))
            events["detected_step"] = events["reactive_step"]
        else:
            alarms = scores[method] >= thresholds[method]
            window = {"pr_auc": float(average_precision_score(y, scores[method])),
                      "roc_auc": float(roc_auc_score(y, scores[method]))}
            if method in ("logistic", "boosting"):
                window["brier"] = float(brier_score_loss(y, scores[method]))
            events = event_metrics(frame, alarms)
        results[method] = {
            "window": window,
            "false_alarms_per_100_clean_windows": float(100 * alarms[clean].mean()),
            "events": summarize_events(events),
            "by_scenario": {s: summarize_events(events, s) for s in sorted(events["scenario"].unique()) if s != "normal"},
            "threshold": thresholds.get(method),
        }
    return results


def run(target="target_now", budget=5.0, dataset_dir=DATASET_DIR, output_dir=DEFAULT_OUTPUT / "prediction",
        groups=("sig", "known", "trend"), save_models=True):
    train, validation = load_split("train", dataset_dir), load_split("validation", dataset_dir)
    columns = feature_columns(train, groups)
    train, validation = train.dropna(subset=[target]), validation.dropna(subset=[target])
    models, tuning = fit_models(train, validation, target, columns)
    val_scores = all_scores(validation, models, columns)
    thresholds = {m: threshold_for_budget(s, validation, budget) for m, s in val_scores.items()}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "target": target,
              "false_alarm_budget_per_100_clean_windows": budget, "feature_groups": list(groups),
              "n_features": len(columns), "tuning": tuning, "thresholds": thresholds, "splits": {}}
    for split in ("validation", "test", "generalization"):
        frame = validation if split == "validation" else load_split(split, dataset_dir).dropna(subset=[target])
        scores = val_scores if split == "validation" else all_scores(frame, models, columns)
        report["splits"][split] = evaluate_split(frame.reset_index(drop=True), scores, thresholds, target)
    folder = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{target}")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "results.json", report)
    (folder / "report.md").write_text(markdown(report), encoding="utf-8")
    if save_models:
        joblib.dump({"models": models, "columns": columns, "thresholds": thresholds, "target": target},
                    folder / "models.joblib")
    return folder, report


def markdown(report):
    fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    pct = lambda v: "—" if v is None else f"{100 * v:.0f} %"
    lines = [f"# Prédiction de l'alerte — cible `{report['target']}`", "",
             f"Seuils d'alarme choisis sur la validation pour au plus "
             f"{report['false_alarm_budget_per_100_clean_windows']:g} fausses alarmes / 100 fenêtres saines "
             f"(R1 a son propre taux, non réglable). {report['n_features']} entrées.", ""]
    for split, results in report["splits"].items():
        title = {"validation": "Validation", "test": "Test (mesure finale)",
                 "generalization": "Généralisation (mécanismes jamais vus)"}[split]
        lines += [f"## {title}", "",
                  "| Méthode | PR-AUC | Détection | Délai depuis le début (fenêtres) | Avance sur R1 (fenêtres) | Plus tôt que R1 | Fausses alarmes / 100 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for method in METHODS:
            r = results[method]
            e = r["events"]
            lines.append(f"| {LABELS[method]} | {fmt(r['window'].get('pr_auc'))} | {pct(e.get('detection_rate'))} | "
                         f"{fmt(e.get('mean_delay_from_onset'))} | "
                         f"{fmt(e.get('mean_lead_vs_reactive')) if method != 'reactive' else '—'} | "
                         f"{pct(e.get('share_earlier_than_reactive')) if method != 'reactive' else '—'} | "
                         f"{fmt(r['false_alarms_per_100_clean_windows'], 1)} |")
        lines += ["", "Par scénario (meilleur modèle M2 : détection · avance sur R1) :", ""]
        for scenario, e in results["boosting"]["by_scenario"].items():
            lines.append(f"- {scenario} : {pct(e.get('detection_rate'))} · {fmt(e.get('mean_lead_vs_reactive'))} fenêtre(s) "
                         f"sur {e.get('degraded_streams', 0)} flux dégradés")
        lines.append("")
    return "\n".join(lines)


def load_report(folder):
    return json.loads((Path(folder) / "results.json").read_text(encoding="utf-8"))
