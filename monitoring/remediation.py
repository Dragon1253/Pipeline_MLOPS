"""Application d'une correction dans la simulation, puis vérification par le monitoring.

L'agent de correction du groupe (``agent_v3.py``) choisit l'action et la valide (plan → garde-fou →
exécution → vérification). Ce module applique l'effet de cette action au monde simulé, puis mesure
les fenêtres suivantes comme le ferait le monitoring :

* ``quarantine_batch`` : la transformation d'entrée fautive est corrigée (les entrées servies au
  modèle redeviennent égales aux données source) et les lots corrompus sont exclus du réentraînement ;
* ``prepare_rollback`` : la version de référence du modèle est remise en service ;
* ``prepare_retraining`` : un nouveau modèle est entraîné sur les dernières fenêtres étiquetées,
  comparé au modèle actuel sur un lot récent, et mis en service seulement s'il fait mieux ;
* ``close_alert`` : rien ne change ;
* ``recalibrate_threshold`` (proposée seulement si le modèle est bien calibré) : le modèle est déjà le
  meilleur possible pour la nouvelle population ; le seuil d'alerte est adapté à cette population.

Si la vérification échoue, ``propose_solutions`` teste les autres actions sur le dernier lot étiqueté
(sans rien mettre en service) et renvoie les solutions classées, à faire approuver par un humain.

Le monde simulé continue avec l'incident (le scénario reste actif) : seule la correction change ce
qui est servi. Une correction inadaptée ne fait donc pas baisser l'erreur, et la vérification échoue.
Le nom du scénario sert uniquement à simuler le monde ; il n'intervient dans aucune décision.
"""
import warnings

import numpy as np
import pandas as pd

from mlops_churn.scenarios import FEATURES, build_pipeline, metrics

from .stream import intensity, make_progressive_window

PLANNED = {
    "qualite_donnees": ("quarantine_batch", "Corriger la transformation des données d'entrée (valeurs reçues à la mauvaise échelle) "
                        "et bloquer les lots corrompus pour le réentraînement. Le modèle n'est pas modifié."),
    "derive_donnees": ("prepare_retraining", "Réentraîner le modèle sur les derniers clients étiquetés, le comparer au modèle "
                       "actuel sur un lot récent, et le mettre en service seulement s'il fait mieux."),
    "regression_version": ("prepare_rollback", "Remettre en service la version de référence du modèle ; la version "
                           "actuellement déployée est retirée."),
    "fausse_alerte": ("close_alert", "Clore l'alerte sans modifier le modèle ni les données."),
}
EFFECTS = {
    "quarantine_batch": "Transformation d'entrée corrigée ; lots corrompus exclus du réentraînement.",
    "prepare_rollback": "Version de référence du modèle remise en service.",
    "prepare_retraining": "Nouveau modèle entraîné sur les dernières fenêtres étiquetées.",
    "close_alert": "Alerte close ; modèle et données inchangés.",
    "recalibrate_threshold": "Seuil d'alerte adapté à la nouvelle population ; modèle inchangé.",
}
PLAIN = {
    "quarantine_batch": "Réparer les données d'entrée et mettre de côté les lots abîmés",
    "prepare_retraining": "Réentraîner le modèle sur les clients récents",
    "prepare_rollback": "Remettre l'ancienne version du modèle",
    "recalibrate_threshold": "Adapter le seuil d'alerte à la nouvelle population",
}
MANUAL = [
    ("Collecter de nouvelles informations sur ces clients",
     "Ajouter des variables qui distinguent ceux qui partent (historique d'usage détaillé, satisfaction…), puis réentraîner."),
    ("Traiter ces clients à part",
     "Les confier à une équipe de rétention ou à une règle métier, au lieu de se fier au modèle pour eux."),
]


def planned_correction(incident_type):
    """(action, phrase) annoncée à l'humain avant son approbation."""
    return PLANNED.get(incident_type, (None, "Aucune correction possible : la cause n'est pas déterminée. "
                                             "Une approbation sera bloquée par l'agent de correction."))


def _world(spec, index, reference_bundle):
    """Fenêtre du monde simulé (l'incident continue) : source, entrées reçues, labels, modèle servi."""
    seed = spec.seed * 1_000 + index
    source, inputs, labels, active, _ = make_progressive_window(
        spec.scenario, intensity(spec, index), reference_bundle, spec.batch_size, seed, spec.variant)
    return source, inputs, labels, active


def _log_loss(bundle, frame, labels):
    probabilities = bundle["pipeline"].predict_proba(frame[FEATURES])[:, 1]
    return metrics(labels, probabilities, bundle.get("threshold", 0.5)), probabilities


def _changed_rows(source, inputs):
    """Lignes dont une valeur reçue diffère de la source (transformation fautive)."""
    a, b = source[FEATURES], inputs[FEATURES]
    different = ~((a == b) | (a.isna() & b.isna()))
    return int(different.any(axis=1).sum())


def _entropy(probabilities):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def _retrain(reference_bundle, windows):
    X = pd.concat([inputs[FEATURES] for _, inputs, _, _ in windows], ignore_index=True)
    y = np.concatenate([labels for _, _, labels, _ in windows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence de lbfgs sur des données corrompues
        pipeline = build_pipeline().fit(X, y)
    return {**reference_bundle, "pipeline": pipeline,
            "versions": {**reference_bundle.get("versions", {}), "retrained_on_rows": int(len(y))}}, int(len(y))


def calibration_check(spec, reference_bundle, *, n_recent=4):
    """Erreur observée et erreur attendue (si le modèle était juste) sur les derniers lots étiquetés."""
    recent = [_world(spec, i, reference_bundle) for i in range(spec.n_windows - n_recent, spec.n_windows)]
    observed, expected = [], []
    for _, inputs, labels, active in recent:
        measures, probabilities = _log_loss(active, inputs, labels)
        observed.append(measures["log_loss"])
        expected.append(_entropy(probabilities))
    return float(np.mean(observed)), float(np.mean(expected)), max(observed)


def propose_solutions(spec, reference_bundle, threshold, reference_value, *, tried=(), n_recent=4):
    """Teste chaque action sur le dernier lot étiqueté (rien n'est mis en service) et les classe."""
    recent = [_world(spec, i, reference_bundle) for i in range(spec.n_windows - n_recent, spec.n_windows)]
    current = recent[-1][3]
    source, inputs, labels, _ = recent[-1]
    window = f"w{spec.n_windows - 1:03d}"
    current_loss = _log_loss(current, inputs, labels)[0]["log_loss"]
    options = []

    def add(action, loss, why, **extra):
        if action in tried or (action != "recalibrate_threshold" and abs(loss - current_loss) < 1e-9):
            return  # déjà essayée, ou sans aucun effet sur ce lot
        options.append({"action": action, "label": PLAIN[action], "expected_log_loss": round(loss, 4),
                        "would_fix": bool(loss <= extra.get("threshold", threshold)), "why": why,
                        "tested_on": window, **extra})

    add("quarantine_batch", _log_loss(current, source, labels)[0]["log_loss"],
        "Test : on sert au modèle les données d'origine au lieu des données reçues.")
    add("prepare_rollback", _log_loss(reference_bundle, inputs, labels)[0]["log_loss"],
        "Test : on remet la version de référence du modèle.")
    candidate, rows = _retrain(reference_bundle, recent[:-1])
    add("prepare_retraining", _log_loss(candidate, inputs, labels)[0]["log_loss"],
        f"Test : nouveau modèle entraîné sur {rows} clients récents.")
    observed, expected, worst = calibration_check(spec, reference_bundle, n_recent=n_recent)
    if abs(observed - expected) <= 0.03 and observed > threshold:
        new_threshold = round(worst + (threshold - reference_value), 4)
        add("recalibrate_threshold", observed,
            (f"Le modèle n'est pas cassé : ses prédictions sont bien calibrées (erreur attendue {expected:.3f}, "
             f"erreur observée {observed:.3f}). L'erreur élevée vient de clients imprévisibles, pas du modèle. "
             f"Adapter le seuil de {threshold:.3f} à {new_threshold:.3f} permet au monitoring de continuer à détecter "
             "les vraies dégradations sur cette population."),
            threshold=new_threshold, previous_threshold=float(threshold))
    options.sort(key=lambda o: (not o["would_fix"], o["action"] == "recalibrate_threshold", o["expected_log_loss"]))
    return {"automatic": options, "manual": [{"label": a, "why": b} for a, b in MANUAL],
            "calibration": {"observed": round(observed, 4), "expected": round(expected, 4)}}


def apply_correction(action, spec, reference_bundle, threshold, *, n_after=6, n_recent=4, start=None,
                     new_threshold=None):
    """Applique ``action`` après la fin du flux ``spec`` et mesure ``n_after`` fenêtres.

    Les ``n_recent`` dernières fenêtres du flux (labels déjà arrivés) servent de mesure « avant » et,
    pour un réentraînement, de données d'entraînement (toutes sauf la dernière) et de lot de comparaison
    (la dernière). Retourne un résumé JSON-sérialisable.
    """
    recent = [_world(spec, i, reference_bundle) for i in range(spec.n_windows - n_recent, spec.n_windows)]
    current = recent[-1][3]                      # modèle servi au moment de l'incident
    before = [_log_loss(active, inputs, labels)[0]["log_loss"] for _, inputs, labels, active in recent]
    fix_inputs, served, details = False, None, {}
    start = spec.n_windows if start is None else start
    used_threshold = float(threshold)

    if action == "quarantine_batch":
        fix_inputs = True
        changed = [_changed_rows(source, inputs) for source, inputs, _, _ in recent]
        details.update(quarantined_windows=[f"w{i:03d}" for i, n in zip(range(spec.n_windows - n_recent, spec.n_windows), changed) if n],
                       quarantined_rows=int(sum(changed)))
    elif action == "prepare_rollback":
        served = reference_bundle
        details.update(rolled_back_to="version de référence", removed_version="version déployée pendant l'incident")
    elif action == "recalibrate_threshold":
        if new_threshold is None:
            raise ValueError("recalibrate_threshold demande new_threshold.")
        used_threshold = float(new_threshold)
        details.update(previous_threshold=float(threshold), new_threshold=used_threshold)
    elif action == "prepare_retraining":
        candidate, rows = _retrain(reference_bundle, recent[:-1])
        _, holdout_inputs, holdout_labels, _ = recent[-1]
        candidate_loss = _log_loss(candidate, holdout_inputs, holdout_labels)[0]["log_loss"]
        current_loss = _log_loss(current, holdout_inputs, holdout_labels)[0]["log_loss"]
        deployed = candidate_loss < current_loss
        served = candidate if deployed else None
        details.update(training_rows=rows,
                       training_windows=[f"w{i:03d}" for i in range(spec.n_windows - n_recent, spec.n_windows - 1)],
                       comparison_window=f"w{spec.n_windows - 1:03d}",
                       candidate_log_loss=round(candidate_loss, 4), current_log_loss=round(current_loss, 4),
                       candidate_deployed=bool(deployed))
    elif action != "close_alert":
        raise ValueError(f"Action inconnue : {action}")

    windows, corrected_after = [], 0
    for index in range(start, start + n_after):
        source, inputs, labels, active = _world(spec, index, reference_bundle)
        model = served or active
        frame = source if fix_inputs else inputs
        if fix_inputs:
            corrected_after += _changed_rows(source, inputs)
        measures, _ = _log_loss(model, frame, labels)
        windows.append({"window": f"w{index:03d}", "labeled_rows": int(len(labels)), "metrics": measures})

    if fix_inputs:
        details["corrected_rows_after"] = int(corrected_after)
    after = [w["metrics"]["log_loss"] for w in windows]
    after_mean = float(np.mean(after))
    return {
        "action": action, "label": PLAIN.get(action, action), "effect": EFFECTS[action], "details": details,
        "threshold": used_threshold, "before_mean": float(np.mean(before)), "after_mean": after_mean,
        "windows_above_threshold": int(sum(v > used_threshold for v in after)), "windows": windows,
        "passed": bool(after_mean <= used_threshold),
    }
