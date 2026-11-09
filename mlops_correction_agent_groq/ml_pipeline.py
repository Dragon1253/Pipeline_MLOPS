"""Démonstration MLOps autonome avec données synthétiques (pas Bike Sharing réel)."""
from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from agent import CorrectionAgent, Incident

OUT = Path('artifacts')
FEATURES = ['hour', 'temperature', 'humidity', 'working_day']

def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    temperature = rng.uniform(0, 35, n)
    humidity = rng.uniform(20, 95, n)
    working_day = rng.integers(0, 2, n)
    X = np.column_stack([hour, temperature, humidity, working_day])
    # Changement de comportement à t=1200 (simulation de concept drift).
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    base = 70 + 2.5 * temperature - 0.45 * humidity + 35 * rush * working_day
    after = np.arange(n) >= 1200
    y = base + after * (45 + 1.7 * temperature + 35 * working_day)
    y += rng.normal(0, 6, n)
    return X, y

def fit(X, y, seed=42):
    model = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    return model

def run():
    OUT.mkdir(exist_ok=True)
    X, y = make_data()
    # Ordre temporel : 0:1000 entraînement initial, 1200:2000 données récentes,
    # 2000:2500 validation, 2500:3000 holdout final indépendant.
    v1 = fit(X[:1000], y[:1000])
    baseline_mae = mean_absolute_error(y[1000:1200], v1.predict(X[1000:1200]))
    degraded_mae = mean_absolute_error(y[2000:2500], v1.predict(X[2000:2500]))
    print(f'MAE avant dérive (v1): {baseline_mae:.2f}')
    print(f'MAE après dérive (v1): {degraded_mae:.2f}')
    if degraded_mae <= baseline_mae * 1.2:
        raise RuntimeError('La dérive simulée n’a pas déclenché le seuil.')
    agent = CorrectionAgent(OUT / 'decisions.jsonl')
    decision = agent.decide(Incident('DEMO-001', 'derive_donnees', 0.95,
                                     {'baseline_mae': baseline_mae, 'current_mae': degraded_mae},
                                     {'dataset': 'synthetic_bike_demand'}))
    print('Décision agent:', decision['action'])
    if decision['action'] != 'request_retraining':
        return
    # Entraînement sur données passées disponibles, sans toucher au holdout final.
    v2 = fit(X[:2000], y[:2000])
    validation_v1 = mean_absolute_error(y[2000:2500], v1.predict(X[2000:2500]))
    validation_v2 = mean_absolute_error(y[2000:2500], v2.predict(X[2000:2500]))
    print(f'Validation MAE v1: {validation_v1:.2f} | v2: {validation_v2:.2f}')
    if validation_v2 >= validation_v1:
        print('Candidat rejeté : aucune amélioration sur validation.')
        return
    holdout_v1 = mean_absolute_error(y[2500:], v1.predict(X[2500:]))
    holdout_v2 = mean_absolute_error(y[2500:], v2.predict(X[2500:]))
    print(f'Holdout MAE v1: {holdout_v1:.2f} | v2: {holdout_v2:.2f}')
    joblib.dump(v1, OUT / 'v1_production.joblib')
    joblib.dump(v2, OUT / 'v2_candidate.joblib')
    report = {'dataset': 'synthetic_bike_demand', 'status': 'pending_human_approval',
              'baseline_mae': baseline_mae, 'degraded_mae': degraded_mae,
              'validation_mae_v1': validation_v1, 'validation_mae_v2': validation_v2,
              'holdout_mae_v1': holdout_v1, 'holdout_mae_v2': holdout_v2,
              'candidate_better_on_holdout': bool(holdout_v2 < holdout_v1),
              'production_model': 'v1_production.joblib', 'candidate_model': 'v2_candidate.joblib'}
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Rapport:', OUT / 'evaluation.json')
    print('Aucun déploiement automatique : approbation humaine nécessaire.')

if __name__ == '__main__':
    run()
