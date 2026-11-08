"""Flux simulé de fenêtres de production, avec des incidents progressifs.

Le flux réutilise le générateur et les injections de ``mlops_churn.scenarios``
sans les modifier. Chaque fenêtre est rendue en deux parties séparées :

* ``Window`` : ce que la production observe réellement (source, entrées,
  modèle servi, labels qui n'arriveront que plus tard) ;
* ``truth`` : le scénario et son intensité, réservés à l'évaluation. Le
  monitoring et l'agent d'analyse ne doivent jamais les recevoir.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlops_churn.scenarios import FEATURES, deployment_candidate, generate_churn

INCIDENTS = ("data_drift", "concept_drift", "data_corruption", "deployment_regression")
STREAM_SCENARIOS = (*INCIDENTS, "normal")
BATCH_SIZE = 500
MISSING_RATE = 0.03


@dataclass(frozen=True)
class StreamSpec:
    """Paramètres d'un flux, fixés avant la simulation.

    ``onset`` : indice de la première fenêtre touchée par l'incident.
    ``ramp`` : nombre de fenêtres pour atteindre l'intensité maximale
    (0 = incident brutal). ``max_intensity`` : intensité finale, entre 0 et 1.
    """
    scenario: str
    seed: int
    n_windows: int = 20
    onset: int = 8
    ramp: int = 4
    max_intensity: float = 1.0
    batch_size: int = BATCH_SIZE

    def __post_init__(self):
        if self.scenario not in STREAM_SCENARIOS:
            raise ValueError(f"Scénario inconnu : {self.scenario}")
        if self.n_windows < 1 or self.batch_size < 100:
            raise ValueError("n_windows doit être >= 1 et batch_size >= 100.")
        if self.scenario != "normal" and not 0 <= self.onset < self.n_windows:
            raise ValueError("onset doit être un indice de fenêtre du flux.")
        if self.ramp < 0 or not 0 <= self.max_intensity <= 1:
            raise ValueError("ramp doit être >= 0 et max_intensity dans [0, 1].")


def intensity(spec, index):
    """Intensité de l'incident à la fenêtre ``index`` : 0 avant onset, puis montée linéaire."""
    if spec.scenario == "normal" or index < spec.onset:
        return 0.0
    if spec.ramp == 0:
        return float(spec.max_intensity)
    return float(spec.max_intensity * min(1.0, (index - spec.onset + 1) / spec.ramp))


@dataclass
class Window:
    """Observations de production d'une fenêtre (aucune information sur l'injection)."""
    index: int
    window_id: str
    source: pd.DataFrame
    inputs: pd.DataFrame
    labels: np.ndarray = field(repr=False)
    active: dict = field(repr=False)
    model_changed: bool = False


def _boundary_population(bundle, n, seed):
    """Clients proches de la frontière du modèle, sélectionnés sur X et le score (jamais sur y)."""
    pool = generate_churn(n * 20, seed)
    scores = bundle["pipeline"].predict_proba(pool[FEATURES])[:, 1]
    selected = np.argsort(np.abs(scores - 0.5), kind="stable")[:n]
    return pool.iloc[selected]


def make_progressive_window(scenario, alpha, bundle, n, seed):
    """Construit une fenêtre pour un scénario à l'intensité ``alpha`` (0 = normal, 1 = incident complet).

    * data_drift : une part ``alpha`` des clients vient d'une population proche de la frontière ;
    * concept_drift : le signe du logit générateur passe progressivement de +1 à -1 ;
    * data_corruption : une part ``alpha`` des lignes reçoit monthly_charges x 100 après la source ;
    * deployment_regression : une mise en production est un saut, la version inversée sert
      toute la fenêtre dès que ``alpha`` > 0.
    Retourne (source, inputs, labels, modèle servi, modèle changé).
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit appartenir à [0, 1].")
    rng = np.random.default_rng(seed + 7_919)
    if scenario == "data_drift" and alpha > 0:
        shifted = int(round(alpha * n))
        parts = [_boundary_population(bundle, shifted, seed + 500_000)]
        if shifted < n:
            parts.append(generate_churn(max(n - shifted, 100), seed).iloc[: n - shifted])
        frame = pd.concat(parts, ignore_index=True)
    elif scenario == "concept_drift":
        frame = generate_churn(n, seed, MISSING_RATE, concept_scale=1.0 - 2.0 * alpha)
    else:
        frame = generate_churn(n, seed, MISSING_RATE)
    frame = frame.sample(frac=1.0, random_state=seed % (2**32)).reset_index(drop=True)
    source = frame.drop(columns="churn")
    labels = frame["churn"].to_numpy()
    inputs = source.copy()
    active, changed = bundle, False
    if scenario == "data_corruption" and alpha > 0:
        corrupted = rng.random(n) < alpha
        inputs.loc[corrupted, "monthly_charges"] = inputs.loc[corrupted, "monthly_charges"] * 100
    elif scenario == "deployment_regression" and alpha > 0:
        active, changed = deployment_candidate(bundle), True
    return source, inputs, labels, active, changed


def simulate_stream(spec, bundle, run_id="stream"):
    """Génère le flux fenêtre par fenêtre : (Window, truth).

    Les graines de chaque fenêtre dépendent seulement de ``spec.seed`` et de l'indice,
    pas des résultats du monitoring : aucune sélection a posteriori.
    """
    candidate = None
    for index in range(spec.n_windows):
        alpha = intensity(spec, index)
        seed = spec.seed * 1_000 + index
        source, inputs, labels, active, changed = make_progressive_window(
            spec.scenario, alpha, bundle, spec.batch_size, seed)
        if changed:
            # Une seule nouvelle version pour tout le flux, comme un vrai déploiement.
            candidate = candidate or active
            active = candidate
        window_id = f"w{index:03d}"
        ids = [f"{run_id}-{window_id}-{i:05d}" for i in range(len(source))]
        source = source.assign(customer_id=ids)
        inputs = inputs.assign(customer_id=ids)
        window = Window(index, window_id, source, inputs, labels, active, changed)
        truth = {"window_id": window_id, "index": index, "scenario": spec.scenario,
                 "intensity": alpha, "incident_active": alpha > 0, "seed": seed}
        yield window, truth
