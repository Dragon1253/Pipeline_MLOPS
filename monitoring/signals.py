"""Signaux disponibles dès le scoring d'une fenêtre, sans attendre les labels.

Ces signaux décrivent la forme des prédictions et l'état des entrées. Ils sont
enregistrés à chaque fenêtre pour alimenter le futur modèle de prédiction des
alertes. Ils ne déclenchent pas l'alerte confirmée, qui reste fondée sur les
métriques calculées avec les labels.
"""
import numpy as np
import pandas as pd

from mlops_churn.scenarios import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-15


def psi(reference, current, bins=10):
    """Population Stability Index entre deux échantillons numériques (quantiles de la référence)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference, current = reference[np.isfinite(reference)], current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def total_variation(reference, current):
    """Distance de variation totale entre deux variables catégorielles (manquants inclus)."""
    ref = pd.Series(reference).fillna("<NA>").value_counts(normalize=True)
    cur = pd.Series(current).fillna("<NA>").value_counts(normalize=True)
    return float(0.5 * ref.subtract(cur, fill_value=0).abs().sum())


def prediction_signals(probabilities, reference_probabilities):
    """Forme de la distribution des probabilités prédites."""
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1 - EPS)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {
        "mean_probability": float(p.mean()),
        "std_probability": float(p.std()),
        "q10_probability": float(np.quantile(p, 0.10)),
        "q50_probability": float(np.quantile(p, 0.50)),
        "q90_probability": float(np.quantile(p, 0.90)),
        "predicted_positive_rate": float((p >= 0.5).mean()),
        # Log loss attendue si le modèle était bien calibré : estimation sans labels.
        "expected_log_loss": float(entropy.mean()),
        "share_uncertain": float(((p > 0.4) & (p < 0.6)).mean()),
        "share_extreme": float(((p < 0.02) | (p > 0.98)).mean()),
        "score_psi": psi(reference_probabilities, p),
    }


def input_signals(inputs, reference_frame, quality_profile):
    """Écarts des entrées servies par rapport à la population de référence."""
    signals = {}
    psis, out_of_range = [], []
    for column in NUMERIC_FEATURES:
        value = psi(reference_frame[column], inputs[column])
        signals[f"psi_{column}"] = value
        if value is not None:
            psis.append(value)
        quantiles = quality_profile["features"][column]["quantiles"]
        low, high = quantiles["0"], quantiles["1"]
        observed = inputs[column].dropna()
        share = float(((observed < low) | (observed > high)).mean()) if len(observed) else 0.0
        signals[f"out_of_range_{column}"] = share
        out_of_range.append(share)
    tvs = []
    for column in CATEGORICAL_FEATURES:
        value = total_variation(reference_frame[column], inputs[column])
        signals[f"tv_{column}"] = value
        tvs.append(value)
    missing = inputs[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().mean()
    signals.update({
        "psi_max": float(max(psis)) if psis else None,
        "tv_max": float(max(tvs)) if tvs else None,
        "out_of_range_max": float(max(out_of_range)),
        "missing_rate_max": float(missing.max()),
    })
    return signals
