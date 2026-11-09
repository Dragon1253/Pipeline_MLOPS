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
# known_windows (nombre de fenêtres déjà étiquetées) révèle la position dans le flux ; comme les
# incidents simulés commencent entre les fenêtres 6 et 14, le modèle apprendrait ce calendrier
# de simulation, qui n'existe pas en production. Exclu des entrées.
EXCLUDED_FEATURES = ("known_windows",)
METHODS = ("reactive", "static_rules", "expected_loss", "logistic", "boosting")
SEQUENCE_METHODS = ("gru", "lstm")
LABELS = {"reactive": "R1 monitoring réactif (étape 1)", "static_rules": "R2 seuils fixes sans labels",
          "expected_loss": "R3 log loss estimée sans labels", "logistic": "M1 régression logistique",
          "boosting": "M2 gradient boosting", "gru": "M3 GRU (séquences)", "lstm": "M4 LSTM (séquences)"}


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
    columns = [c for c in frame.columns if c.startswith(prefixes) and c not in EXCLUDED_FEATURES]
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


def evaluate_split(frame, scores, thresholds, target, methods=METHODS):
    """Métriques par fenêtre et par flux pour chaque méthode."""
    results = {}
    y = frame[target].to_numpy()
    clean = clean_mask(frame)
    for method in methods:
        if method == "reactive":
            alarms = reactive_alarms(frame)
            window = {}
            events = event_metrics(frame, np.zeros(len(frame), dtype=bool))
            events["detected_step"] = events["reactive_step"]
        else:
            alarms = scores[method] >= thresholds[method]
            window = {"pr_auc": float(average_precision_score(y, scores[method])),
                      "roc_auc": float(roc_auc_score(y, scores[method]))}
            if method in ("logistic", "boosting", *SEQUENCE_METHODS):
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
        groups=("sig", "known", "trend"), save_models=True, sequence_models=(), sequence_epochs=30):
    """Entraîne, règle sur la validation et compare toutes les méthodes. ``sequence_models`` :
    ``("gru", "lstm")`` pour ajouter les réseaux récurrents (PyTorch requis)."""
    full = {"train": load_split("train", dataset_dir), "validation": load_split("validation", dataset_dir)}
    columns = feature_columns(full["train"], groups)
    train, validation = (full[k].dropna(subset=[target]) for k in ("train", "validation"))
    models, tuning = fit_models(train, validation, target, columns)
    sequence, methods = {}, METHODS + tuple(sequence_models)
    if sequence_models:
        from .sequence_models import fit_sequence_models
        sequence_columns = feature_columns(full["train"], ("sig", "known"))
        sequence, sequence_tuning = fit_sequence_models(full["train"], full["validation"], target, sequence_columns,
                                                        cells=tuple(sequence_models), epochs=sequence_epochs)
        tuning.update(sequence_tuning)

    def scored(frame):
        # Les séquences ont besoin du flux complet : score sur toutes les fenêtres, puis
        # évaluation seulement là où la cible est définie.
        scores = all_scores(frame, models, columns)
        for name, model in sequence.items():
            scores[name] = model.predict_frame(frame)
        mask = frame[target].notna().to_numpy()
        return {k: v[mask] for k, v in scores.items()}, frame[mask].reset_index(drop=True)

    val_scores, val_frame = scored(full["validation"])
    thresholds = {m: threshold_for_budget(s, val_frame, budget) for m, s in val_scores.items()}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "target": target,
              "false_alarm_budget_per_100_clean_windows": budget, "feature_groups": list(groups),
              "n_features": len(columns), "methods": list(methods), "tuning": tuning,
              "thresholds": thresholds, "splits": {}}
    for split in ("validation", "test", "generalization"):
        scores, frame = (val_scores, val_frame) if split == "validation" else scored(load_split(split, dataset_dir))
        report["splits"][split] = evaluate_split(frame, scores, thresholds, target, methods)
    suffix = f"-{target}" + ("-" + "-".join(sequence_models) if sequence_models else "")
    folder = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + suffix)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "results.json", report)
    (folder / "report.md").write_text(markdown(report), encoding="utf-8")
    if save_models:
        joblib.dump({"models": models, "columns": columns, "thresholds": thresholds, "target": target},
                    folder / "models.joblib")
        if sequence:
            import torch
            torch.save({name: {"state_dict": m.network.state_dict(), "preprocessor": m.preprocessor.__dict__,
                               "length": m.length, "hidden": tuning[name]["chosen"]["hidden"]}
                        for name, m in sequence.items()}, folder / "sequence_models.pt")
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
        for method in report.get("methods", METHODS):
            r = results[method]
            e = r["events"]
            lines.append(f"| {LABELS[method]} | {fmt(r['window'].get('pr_auc'))} | {pct(e.get('detection_rate'))} | "
                         f"{fmt(e.get('mean_delay_from_onset'))} | "
                         f"{fmt(e.get('mean_lead_vs_reactive')) if method != 'reactive' else '—'} | "
                         f"{pct(e.get('share_earlier_than_reactive')) if method != 'reactive' else '—'} | "
                         f"{fmt(r['false_alarms_per_100_clean_windows'], 1)} |")
        learned = [m for m in report.get("methods", METHODS) if m in ("boosting", *SEQUENCE_METHODS)]
        scenarios = list(results["boosting"]["by_scenario"])
        lines += ["", "Avance sur R1 par scénario, en fenêtres (détection entre parenthèses) :", "",
                  "| Scénario | " + " | ".join(LABELS[m] for m in learned) + " |",
                  "| --- |" + " --- |" * len(learned)]
        for scenario in scenarios:
            cells = [f"{fmt(results[m]['by_scenario'][scenario].get('mean_lead_vs_reactive'))} "
                     f"({pct(results[m]['by_scenario'][scenario].get('detection_rate'))})" for m in learned]
            lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def load_report(folder):
    return json.loads((Path(folder) / "results.json").read_text(encoding="utf-8"))
