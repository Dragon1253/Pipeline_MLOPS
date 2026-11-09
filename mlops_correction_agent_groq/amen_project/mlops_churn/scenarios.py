"""Génération des cinq scénarios, modèle prédictif et déclenchement des alertes."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, log_loss,
)
import copy
import json
import shutil
import time
from pathlib import Path
import joblib
from .workspace import ROOT, sha256, write_json, utc_now
from types import SimpleNamespace
import uuid

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "support_tickets_3m",
    "late_payments_12m", "usage_hours_month",
]


CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]


FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


TARGET = "churn"


def generate_churn(n_samples=5000, random_state=42, missing_rate=0.03, *, concept_scale=1.0):
    """Échantillonne les attributs, puis une cible probabiliste liée aux attributs.

    Les valeurs manquantes sont injectées après création de la cible.
    Les relations choisies sont des hypothèses de simulation, pas des faits métier.
    """
    if n_samples < 100:
        raise ValueError("n_samples doit être supérieur ou égal à 100.")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate doit appartenir à [0, 1[.")
    rng = np.random.default_rng(random_state)
    contract = rng.choice(["monthly", "annual", "two_year"], n_samples, p=[.55, .30, .15])
    payment = rng.choice(["automatic", "card", "manual"], n_samples, p=[.45, .30, .25])
    internet = rng.choice(["fiber", "dsl", "none"], n_samples, p=[.5, .4, .1])
    tenure = rng.integers(1, 73, n_samples)
    charges = np.clip(rng.normal(65, 20, n_samples) + 15 * (internet == "fiber"), 15, 140)
    tickets = rng.poisson(1.2, n_samples)
    late = rng.poisson(.7, n_samples)
    usage = np.clip(rng.normal(45, 18, n_samples), 0, 120)
    logit = (
        -1.5 - .025 * tenure + .018 * (charges - 65)
        + .40 * tickets + .35 * late - .012 * (usage - 45)
        + .9 * (contract == "monthly") - .7 * (contract == "two_year")
        + .45 * (payment == "manual") - .25 * (payment == "automatic")
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-concept_scale * logit)))
    frame = pd.DataFrame({
        "customer_id": [f"SYN-{i:06d}" for i in range(n_samples)],
        "tenure_months": tenure, "monthly_charges": charges.round(2),
        "support_tickets_3m": tickets, "late_payments_12m": late,
        "usage_hours_month": usage.round(2), "contract_type": contract,
        "payment_method": payment, "internet_service": internet, "churn": churn,
    })
    for column in FEATURES:
        mask = rng.random(n_samples) < missing_rate
        if column in NUMERIC_FEATURES:
            frame[column] = frame[column].astype(float)
        frame.loc[mask, column] = np.nan
    return frame


def validate_inference_data(frame):
    """Contrat structurel à l'inférence ; la cible n'est pas requise."""
    required = ["customer_id", *FEATURES]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Colonnes absentes : {missing}")
    if frame["customer_id"].isna().any() or frame["customer_id"].duplicated().any():
        raise ValueError("customer_id doit être renseigné et unique.")
    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"La colonne {column} doit être numérique.")
        if not np.isfinite(frame[column].dropna().to_numpy()).all():
            raise ValueError(f"La colonne {column} contient des valeurs infinies.")
    for column in FEATURES:
        if frame[column].isna().all():
            raise ValueError(f"La colonne {column} est entièrement vide.")
    for column in CATEGORICAL_FEATURES:
        if not frame[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"La colonne {column} doit contenir des chaînes.")


def validate_labels(frame):
    """Un lot de feedback peut légitimement ne contenir qu'une classe."""
    if not {"prediction_id", TARGET}.issubset(frame.columns):
        raise ValueError("Labels : prediction_id et churn sont requis.")
    if frame.prediction_id.isna().any() or frame.prediction_id.duplicated().any():
        raise ValueError("prediction_id doit être renseigné et unique.")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Les labels doivent appartenir à {0, 1} sans valeur manquante.")


def validate_data(frame):
    """Contrat historique d'entraînement, conservé pour les commandes existantes."""
    if TARGET not in frame:
        raise ValueError(f"Colonnes absentes : ['{TARGET}']")
    validate_inference_data(frame)
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("churn doit contenir les deux classes 0 et 1 sans valeur manquante.")


def build_pipeline(random_state=42):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ], remainder="drop")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    two_classes = len(set(y_true)) == 2
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if two_classes else None,
        "average_precision": float(average_precision_score(y_true, probabilities)) if two_classes else None,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def monitor_performance(measures, labeled_rows, calibration, window_id):
    if labeled_rows < calibration["min_labeled_rows"]:
        return {"status": "insufficient_labels", "alert": None}
    value = measures.get("log_loss")
    if value is None or not np.isfinite(value):
        return {"status": "metric_unavailable", "alert": None}
    alert = None
    if value > calibration["threshold"]:
        alert = {
            "alert_id": f"{window_id}-alert", "window_id": window_id,
            "metric": "log_loss", "observed_value": value,
            "direction": "higher_is_worse",
            "reference_value": calibration["reference_value"],
            "threshold": calibration["threshold"], "labeled_rows": labeled_rows,
        }
    return {"status": "alert" if alert else "normal", "alert": alert}


CAUSES = ["data_drift", "concept_drift", "data_corruption", "deployment_regression", "no_incident"]


def deployment_candidate(bundle):
    candidate = copy.deepcopy(bundle)
    classifier = candidate["pipeline"].named_steps["classifier"]
    classifier.coef_ *= -1
    classifier.intercept_ *= -1
    return candidate


def make_window(scenario, bundle, n, seed, missing_rate):
    """Retourne les entrées reçues, les labels réels, le modèle servi et l'injection privée."""
    frame = generate_churn(n, seed, missing_rate, concept_scale=-1 if scenario == "concept_drift" else 1)
    source = frame.drop(columns="churn").copy()
    labels = frame.churn.copy()
    active = bundle
    details = {}
    if scenario == "data_drift":
        # Sélection sur X et les scores seuls : la loi conditionnelle de y est conservée.
        pool = generate_churn(n * 20, seed, missing_rate)
        probabilities = bundle["pipeline"].predict_proba(pool[FEATURES])[:, 1]
        selected = np.argsort(np.abs(probabilities - .5), kind="stable")[:n]
        frame = pool.iloc[selected].reset_index(drop=True)
        source, labels = frame.drop(columns="churn").copy(), frame.churn.copy()
        details = {"mechanism": "Population concentrée près de la frontière de décision ; sélection indépendante de y."}
    inputs = source.copy()
    if scenario == "data_corruption":
        inputs["monthly_charges"] *= 100
        details = {"feature": "monthly_charges", "mechanism": "Conversion d'unité incorrecte après réception ; labels inchangés."}
    elif scenario == "concept_drift":
        details = {"mechanism": "Signe du logit générateur inversé ; génération de X inchangée."}
    elif scenario == "deployment_regression":
        active = deployment_candidate(bundle)
        details = {"mechanism": "Coefficients et intercept de la version servie inversés."}
    elif scenario not in {"normal", "false_alert", "data_drift"}:
        raise ValueError(f"Scénario inconnu : {scenario}")
    if scenario == "false_alert":
        details = {"mechanism": "Lot normal de queue de distribution, sélectionné conditionnellement au franchissement ; aucune corruption."}
    return source, inputs, labels.to_numpy(), active, details


def generate_one(args):
    episode_id = {'concept_drift': 'episode-002', 'data_drift': 'episode-001',
                  'data_corruption': 'episode-003', 'deployment_regression': 'episode-004', 'false_alert': 'episode-005'}[args.scenario]
    previous = ROOT / args.reference
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + args.label + '-' + uuid.uuid4().hex[:8]
    destination = ROOT / args.output / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'evaluation').mkdir()
    (destination / 'models').mkdir()
    episode = destination / 'episodes' / episode_id
    episode.mkdir(parents=True)
    # Graines choisies avant génération ; pas de sélection en fonction du diagnostic.
    seeds = ([args.seed, args.seed + 1001, args.seed + 1002] if args.scenario == 'false_alert'
             else [args.seed, args.seed + 1, args.seed + 2])
    plan = {'created_at': utc_now(), 'production_seeds': seeds, 'batch_size': 500,
            'missing_rate': 0.03, 'reference_dataset': str(previous),
            'data_origin': 'generate_churn, données entièrement synthétiques',
            'selection_attempts': 1, 'max_focus_selection_attempts': 1000 if args.scenario == 'false_alert' else 1,
            'reference_and_calibration_preserved': True}
    write_json(destination / 'evaluation/generation_plan.json', plan)
    for name in ['reference_data.csv', 'quality_reference.json', 'calibration.json']:
        shutil.copyfile(previous / name, destination / name)
    shutil.copyfile(previous / 'models/reference.joblib', destination / 'models/reference.joblib')
    shutil.copyfile(previous / 'models/reference.joblib', episode / 'active.joblib')
    bundle = joblib.load(destination / 'models/reference.joblib')
    calibration = json.loads((destination / 'calibration.json').read_text(encoding='utf-8'))
    version = 'sha256:' + sha256(destination / 'models/reference.joblib')
    history = []
    private = None
    for index, seed in enumerate(seeds):
        name = 'focus' if index == 0 else f'confirmation-{index}'
        folder = episode / name
        folder.mkdir()
        scenario = 'normal' if args.scenario == 'false_alert' and index else args.scenario
        if args.scenario == 'false_alert' and index == 0:
            for attempt in range(1000):
                seed = args.seed + attempt
                source, inputs, labels, active, details = make_window('false_alert', bundle, 500, seed, 0.03)
                scores = active['pipeline'].predict_proba(inputs[FEATURES])[:, 1]
                test_metrics = metrics(labels, scores, active['threshold'])
                if monitor_performance(test_metrics, 500, calibration, f'{episode_id}-focus')['alert']:
                    break
            else:
                raise RuntimeError('Aucune fausse alerte en 1000 lots normaux ; seuil inchangé.')
            seeds[0] = seed
            plan['selected_focus_seed'] = seed
            plan['selection_attempts'] = attempt + 1
            write_json(destination / 'evaluation/generation_plan.json', plan)
        else:
            source, inputs, labels, active, details = make_window(scenario, bundle, 500, seed, 0.03)
        if index == 0 and args.scenario == 'deployment_regression':
            joblib.dump(active, episode / 'active.joblib')
            version = 'sha256:' + sha256(episode / 'active.joblib')
        probabilities = active['pipeline'].predict_proba(inputs[FEATURES])[:, 1]
        measures = metrics(labels, probabilities, active['threshold'])
        signal = monitor_performance(measures, 500, calibration, f'{episode_id}-{name}')
        ids = [f'{run_id}-{episode_id}-{name}-{i:05d}' for i in range(500)]
        source['customer_id'] = ids
        inputs['customer_id'] = ids
        source.to_csv(folder / 'source.csv', index=False)
        inputs.to_csv(folder / 'inputs.csv', index=False)
        pd.DataFrame({'customer_id': ids, 'churn': labels}).to_csv(folder / 'feedback.csv', index=False)
        pd.DataFrame({'customer_id': ids, 'probability': probabilities,
                      'prediction': (probabilities >= active['threshold']).astype(int),
                      'model_version': version}).to_csv(folder / 'predictions.csv', index=False)
        write_json(folder / 'monitoring.json', {'metrics': measures, 'labeled_rows': 500, **signal})
        history.append({'window': name, 'metrics': measures, 'labeled_rows': 500, 'alert': bool(signal['alert'])})
        if index == 0:
            if not signal['alert']:
                raise RuntimeError('Le nouveau lot ne déclenche pas d’alerte ; aucune investigation lancée.')
            write_json(episode / 'alert.json', signal['alert'])
            private = details
    write_json(episode / 'context.json', {'episode_id': episode_id, 'threshold': calibration['threshold'],
        'performance_history': history, 'deployment': {'reference_version': 'sha256:' + sha256(destination / 'models/reference.joblib'), 'active_version': version,
        'model_changed': args.scenario == 'deployment_regression', 'preprocessing_changed': False,
        'events': [{'event': 'model_activated', 'window': 'focus', 'version': version}] if args.scenario == 'deployment_regression' else []}})
    write_json(destination / 'evaluation/ground_truth.json', [{'episode_id': episode_id,
        'scenario': args.scenario, 'expected_cause': 'no_incident' if args.scenario == 'false_alert' else args.scenario, 'is_incident': args.scenario != 'false_alert',
        'selected_seed': seeds[0], 'confirmation_seeds': seeds[1:], 'selection_attempts': plan['selection_attempts'],
        'injection': private, 'focus_alert_verified': True}])
    old_data = pd.read_csv(previous / 'episodes' / episode_id / 'focus/source.csv')
    new_data = pd.read_csv(episode / 'focus/source.csv')
    checks = {'features_differ_from_previous': not old_data[FEATURES].equals(new_data[FEATURES]),
              'model_unchanged': sha256(destination / 'models/reference.joblib') == sha256(previous / 'models/reference.joblib'),
              'reference_unchanged': sha256(destination / 'reference_data.csv') == sha256(previous / 'reference_data.csv'),
              'calibration_unchanged': sha256(destination / 'calibration.json') == sha256(previous / 'calibration.json'),
              'focus_window_alerted': history[0]['alert']}
    if args.scenario != 'false_alert':
        checks['all_windows_alerted'] = all(row['alert'] for row in history)
    write_json(destination / 'evaluation/generation_checks.json', checks)
    assert all(value for key,value in checks.items() if key != 'features_differ_from_previous'), checks
    print(json.dumps({'experiment': str(destination), 'scenario': args.scenario,
          'production_seeds': seeds, 'focus_log_loss': history[0]['metrics']['log_loss'],
          'checks': checks}, ensure_ascii=False, indent=2))
    return destination

SCENARIOS = {'data_drift': 'episode-001', 'concept_drift': 'episode-002',
             'data_corruption': 'episode-003', 'deployment_regression': 'episode-004',
             'false_alert': 'episode-005'}


def generate_scenarios(scenario='all', seed=100001, reference_dir='scenario_data/demo',
                       output_dir='scenario_data/generated', label='generated'):
    """Generate independent episodes without invoking a language model."""
    if scenario not in [*SCENARIOS, 'all']:
        raise ValueError('Scénario inconnu : ' + scenario)
    if not label or Path(label).name != label or label in {'.', '..'}:
        raise ValueError('label doit être un nom simple sans chemin.')
    cases = list(SCENARIOS) if scenario == 'all' else [scenario]
    return [generate_one(SimpleNamespace(scenario=case, seed=seed + i * 2000,
            reference=str(reference_dir), output=str(output_dir), label=label + '-' + case))
            for i, case in enumerate(cases)]
