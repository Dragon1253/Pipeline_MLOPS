"""Étape 2.4 : consolider la prédiction de l'alerte (modèle retenu : M2 gradient boosting).

1. **Ablation** : M2 réentraîné avec différents groupes d'entrées, pour savoir lesquels comptent.
2. **Intervalles de confiance** : bootstrap sur les flux (tirage avec remise des flux du test et
   de la généralisation) pour la détection, l'avance sur R1, les fausses alarmes et le PR-AUC.
3. **Explication** : importance globale par permutation (validation) et, pour chaque alerte,
   les signaux qui l'ont déclenchée (remplacer un signal par sa valeur habituelle fait-il
   retomber le risque ?). L'explication décrit des signaux, jamais une cause.

Même protocole que ``prediction.py`` : apprentissage sur ``train``, réglages et seuil sur
``validation``, budget de fausses alarmes identique, mesures sur ``test`` et ``generalization``.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mlops_churn.workspace import write_json

from .prediction import (DATASET_DIR, DEFAULT_OUTPUT, clean_mask, event_metrics, feature_columns,
                         fit_models, load_split, reactive_alarms, summarize_events, threshold_for_budget)

ABLATIONS = {
    "prédictions seules (sig_ sur les scores)": ("sig_pred",),
    "signaux sans labels (sig_)": ("sig",),
    "signaux sans labels + tendances": ("sig", "trend"),
    "métriques connues seules (known_)": ("known",),
    "signaux + métriques connues": ("sig", "known"),
    "tout (sig_ + known_ + tendances)": ("sig", "known", "trend"),
}
PREDICTION_SIGNALS = ["sig_mean_probability", "sig_std_probability", "sig_q10_probability",
                      "sig_q50_probability", "sig_q90_probability", "sig_predicted_positive_rate",
                      "sig_expected_log_loss", "sig_share_uncertain", "sig_share_extreme", "sig_score_psi"]


def columns_for(frame, groups):
    if groups == ("sig_pred",):
        return list(PREDICTION_SIGNALS)
    return feature_columns(frame, groups)


def _prepare(target, dataset_dir):
    splits = {name: load_split(name, dataset_dir).dropna(subset=[target]).reset_index(drop=True)
              for name in ("train", "validation", "test", "generalization")}
    return splits


def fit_boosting(splits, target, columns, budget):
    models, tuning = fit_models(splits["train"], splits["validation"], target, columns, kinds=("boosting",))
    model = models["boosting"]
    threshold = threshold_for_budget(model.predict_proba(splits["validation"][columns])[:, 1],
                                     splits["validation"], budget)
    return model, threshold, tuning["boosting"]["chosen"]


def method_summary(frame, scores, threshold, target):
    alarms = scores >= threshold
    events = event_metrics(frame, alarms)
    summary = summarize_events(events)
    return {"pr_auc": float(average_precision_score(frame[target], scores)),
            "detection_rate": summary.get("detection_rate"),
            "mean_lead_vs_reactive": summary.get("mean_lead_vs_reactive"),
            "false_alarms_per_100_clean_windows": float(100 * alarms[clean_mask(frame)].mean()),
            "by_scenario": {s: summarize_events(events, s).get("detection_rate")
                            for s in sorted(events["scenario"].unique()) if s != "normal"}}


def ablation(splits, target, budget):
    rows = []
    for label, groups in ABLATIONS.items():
        columns = columns_for(splits["train"], groups)
        model, threshold, params = fit_boosting(splits, target, columns, budget)
        row = {"inputs": label, "groups": list(groups), "n_features": len(columns), "params": params}
        for split in ("test", "generalization"):
            frame = splits[split]
            row[split] = method_summary(frame, model.predict_proba(frame[columns])[:, 1], threshold, target)
        rows.append(row)
    return rows


def _stream_tables(frame, scores, threshold, target):
    """Tables par flux réutilisées par le bootstrap (aucun recalcul du modèle)."""
    alarms = scores >= threshold
    clean = clean_mask(frame)
    reactive = reactive_alarms(frame)
    events = event_metrics(frame, alarms).set_index("stream_id")
    per_stream = pd.DataFrame({"stream_id": frame["stream_id"], "clean": clean,
                               "alarm_clean": alarms & clean, "reactive_clean": reactive & clean})
    counts = per_stream.groupby("stream_id")[["clean", "alarm_clean", "reactive_clean"]].sum()
    return events.join(counts), frame.groupby("stream_id").indices


def bootstrap(frame, scores, threshold, target, n_boot=1000, seed=0):
    """Intervalles à 95 % par tirage avec remise des flux."""
    table, positions = _stream_tables(frame, scores, threshold, target)
    streams = table.index.to_numpy()
    y, s = frame[target].to_numpy(), np.asarray(scores)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("detection_rate", "mean_lead_vs_reactive", "false_alarms_per_100_clean_windows",
                             "reactive_false_alarms_per_100_clean_windows", "pr_auc")}

    def stats(sample):
        sub = table.loc[sample]
        degraded = sub[sub["degraded"]]
        detected = degraded["detected_step"].notna()
        both = degraded.dropna(subset=["detected_step", "reactive_step"])
        clean = sub["clean"].sum()
        rows = np.concatenate([positions[i] for i in sample])
        return {"detection_rate": detected.mean() if len(degraded) else np.nan,
                "mean_lead_vs_reactive": (both["reactive_step"] - both["detected_step"]).mean() if len(both) else np.nan,
                "false_alarms_per_100_clean_windows": 100 * sub["alarm_clean"].sum() / clean if clean else np.nan,
                "reactive_false_alarms_per_100_clean_windows": 100 * sub["reactive_clean"].sum() / clean if clean else np.nan,
                "pr_auc": average_precision_score(y[rows], s[rows]) if y[rows].any() else np.nan}

    point = stats(streams)
    for _ in range(n_boot):
        for key, value in stats(rng.choice(streams, size=len(streams), replace=True)).items():
            draws[key].append(value)
    return {key: {"value": float(point[key]), "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
            for key, v in draws.items()}


def permutation_importance(model, frame, columns, target, repeats=3, seed=0, top=12):
    """Baisse du PR-AUC de validation quand on mélange une entrée (les autres restant intactes)."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(frame[target], model.predict_proba(frame[columns])[:, 1])
    drops = []
    for column in columns:
        losses = []
        for _ in range(repeats):
            shuffled = frame[columns].copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(base - average_precision_score(frame[target], model.predict_proba(shuffled)[:, 1]))
        drops.append({"feature": column, "pr_auc_drop": float(np.mean(losses))})
    return sorted(drops, key=lambda d: d["pr_auc_drop"], reverse=True)[:top]


def reference_profile(frame, columns):
    """Valeurs habituelles : médiane et dispersion (écart interquartile) sur les fenêtres saines
    de l'entraînement."""
    clean = frame.loc[clean_mask(frame), columns]
    median = clean.median()
    spread = (clean.quantile(0.75) - clean.quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(clean.std().replace(0, np.nan)).fillna(1.0)
    return median, spread


def explain_alert(model, row, columns, profile, top=3):
    """Signaux qui ont déclenché une alerte.

    Plusieurs signaux disent souvent la même chose : en remettre un seul à sa valeur habituelle
    ne fait pas baisser le risque. On désigne donc les signaux directs (``sig_``, ``known_``) qui
    s'écartent le plus de leur valeur habituelle (en nombre d'écarts interquartiles), puis on
    vérifie qu'en les remettant **ensemble** à leur valeur habituelle (avec leurs tendances),
    le risque retombe.
    """
    median, spread = profile
    base = pd.DataFrame([row[columns]])
    risk = float(model.predict_proba(base)[:, 1][0])
    direct = [c for c in columns if c.startswith(("sig_", "known_"))]
    deviation = ((base[direct].iloc[0] - median[direct]) / spread[direct]).abs().dropna()
    chosen = list(deviation.sort_values(ascending=False).index[:top])
    restored = base.copy()
    for column in chosen:
        related = [c for c in columns if c == column or c.startswith(f"trend_{column}_")]
        for c in related:
            restored[c] = median[c]
    risk_after = float(model.predict_proba(restored)[:, 1][0])
    return {"risk": risk, "risk_if_signals_usual": risk_after, "signals": [
        {"signal": c, "value": None if pd.isna(row[c]) else float(row[c]),
         "usual_value": float(median[c]), "deviation_in_iqr": float(deviation[c])} for c in chosen]}


def example_explanations(model, frame, columns, threshold, profile):
    """Première alerte sur une fenêtre touchée, pour un flux dégradé par scénario."""
    frame = frame.sort_values(["stream_id", "index"])
    scores = model.predict_proba(frame[columns])[:, 1]
    examples = {}
    for scenario in sorted(s for s in frame["truth_scenario"].unique() if s != "normal"):
        rows = frame[(frame["truth_scenario"] == scenario) & (frame["truth_incident_active"] == 1)
                     & (scores >= threshold)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        examples[scenario] = {"stream_id": int(row["stream_id"]), "window": row["window"],
                              **explain_alert(model, row, columns, profile)}
    return examples


def run(target="target_h2", budget=5.0, dataset_dir=DATASET_DIR, output_dir=DEFAULT_OUTPUT / "analysis",
        n_boot=1000, ablate=True):
    splits = _prepare(target, dataset_dir)
    columns = columns_for(splits["train"], ("sig", "known", "trend"))
    model, threshold, params = fit_boosting(splits, target, columns, budget)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "target": target,
              "model": "M2 gradient boosting", "params": params, "threshold": threshold,
              "false_alarm_budget_per_100_clean_windows": budget, "confidence": {}}
    for split in ("test", "generalization"):
        frame = splits[split]
        report["confidence"][split] = bootstrap(frame, model.predict_proba(frame[columns])[:, 1], threshold,
                                                target, n_boot=n_boot)
    report["importance"] = permutation_importance(model, splits["validation"], columns, target)
    profile = reference_profile(splits["train"], columns)
    report["explanations"] = example_explanations(model, splits["test"], columns, threshold, profile)
    if ablate:
        report["ablation"] = ablation(splits, target, budget)
    folder = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{target}")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "results.json", report)
    (folder / "report.md").write_text(markdown(report), encoding="utf-8")
    return folder, report


def markdown(report):
    fmt = lambda v, d=2: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    pct = lambda v: "—" if v is None else f"{100 * v:.0f} %"
    ci = lambda e, d=2, p=False: (f"{100 * e['value']:.0f} % [{100 * e['ci95'][0]:.0f} ; {100 * e['ci95'][1]:.0f}]" if p
                                  else f"{e['value']:.{d}f} [{e['ci95'][0]:.{d}f} ; {e['ci95'][1]:.{d}f}]")
    lines = [f"# Consolidation de la prédiction — {report['model']}, cible `{report['target']}`", "",
             f"Seuil d'alarme {report['threshold']:.3f} (budget {report['false_alarm_budget_per_100_clean_windows']:g} "
             "fausses alarmes / 100 fenêtres saines, fixé sur la validation).", "",
             "## Intervalles de confiance à 95 % (bootstrap sur les flux)", "",
             "| Mesure | Test | Généralisation |", "| --- | --- | --- |"]
    names = {"detection_rate": ("Détection des flux dégradés", True), "mean_lead_vs_reactive": ("Avance sur R1 (fenêtres)", False),
             "false_alarms_per_100_clean_windows": ("Fausses alarmes / 100 fenêtres saines", False),
             "reactive_false_alarms_per_100_clean_windows": ("R1 : fausses alarmes / 100 fenêtres saines", False),
             "pr_auc": ("PR-AUC", False)}
    for key, (name, as_pct) in names.items():
        t, g = report["confidence"]["test"][key], report["confidence"]["generalization"][key]
        lines.append(f"| {name} | {ci(t, p=as_pct)} | {ci(g, p=as_pct)} |")
    if "ablation" in report:
        lines += ["", "## Ablation : quelles entrées comptent ?", "",
                  "| Entrées | Nb | Test : PR-AUC | Test : détection | Test : avance | Généralisation : détection | Généralisation : avance | Fausses alarmes (test) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in report["ablation"]:
            t, g = row["test"], row["generalization"]
            lines.append(f"| {row['inputs']} | {row['n_features']} | {fmt(t['pr_auc'])} | {pct(t['detection_rate'])} | "
                         f"{fmt(t['mean_lead_vs_reactive'])} | {pct(g['detection_rate'])} | {fmt(g['mean_lead_vs_reactive'])} | "
                         f"{fmt(t['false_alarms_per_100_clean_windows'], 1)} |")
    lines += ["", "## Entrées les plus importantes (permutation, validation)", "",
              "| Entrée | Baisse du PR-AUC si on la mélange |", "| --- | --- |"]
    lines += [f"| `{d['feature']}` | {d['pr_auc_drop']:.3f} |" for d in report["importance"]]
    lines += ["", "## Exemples d'alertes expliquées (test)", ""]
    for scenario, e in report["explanations"].items():
        lines.append(f"- **{scenario}** (flux {e['stream_id']}, {e['window']}) : risque {e['risk']:.2f}, "
                     f"{e['risk_if_signals_usual']:.2f} si ces signaux étaient habituels — " + "; ".join(
            f"`{s['signal']}` = {fmt(s['value'], 3)} (habituel {fmt(s['usual_value'], 3)})" for s in e["signals"]))
    lines += ["", "Les explications désignent des signaux mesurés, pas une cause : le diagnostic reste "
              "le travail de l'agent d'analyse.", ""]
    return "\n".join(lines)
