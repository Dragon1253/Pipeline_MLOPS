"""Vérifie les contrats utiles aux futurs scénarios d'agents."""

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from mlops_churn.scenarios import FEATURES, generate_churn, validate_data
from mlops_churn.scenarios import build_pipeline
from mlops_churn.scenarios import generate_scenarios, SCENARIOS
from mlops_churn.workspace import ROOT, sha256
import contextlib
import io
import json
import tempfile


class PipelineTests(unittest.TestCase):
    def test_reproducible_data_and_schema_rejection(self):
        frame = generate_churn(500, 42)
        self.assertTrue(frame.equals(generate_churn(500, 42)))
        validate_data(frame)
        with self.assertRaisesRegex(ValueError, "Colonnes absentes"):
            validate_data(frame.drop(columns="monthly_charges"))
        invalid = frame.copy()
        invalid["monthly_charges"] = "invalid"
        with self.assertRaisesRegex(ValueError, "numérique"):
            validate_data(invalid)

    def test_unseen_category_and_missing_values_at_inference(self):
        frame = generate_churn(600, 42)
        train, test = train_test_split(frame, test_size=.2, random_state=42, stratify=frame.churn)
        model = build_pipeline().fit(train[FEATURES], train.churn)
        probe = test[FEATURES].copy()
        probe["contract_type"] = "never_seen_contract"
        probe["monthly_charges"] = np.nan
        probabilities = model.predict_proba(probe)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(probabilities.shape, (len(test), 2))
        self.assertTrue(set(train.customer_id).isdisjoint(test.customer_id))

    def test_generates_all_five_scenarios_with_reference_and_threshold_preserved(self):
        baseline = ROOT / 'scenario_data/demo'
        before = {str(p):sha256(p) for p in baseline.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            generated = generate_scenarios(seed=100001, output_dir=tmp)
            self.assertEqual(len(generated), 5)
            for experiment, (scenario, episode_id) in zip(generated, SCENARIOS.items()):
                truth = json.loads((experiment/'evaluation/ground_truth.json').read_text(encoding='utf-8'))[0]
                self.assertEqual(truth['scenario'], scenario)
                checks = json.loads((experiment/'evaluation/generation_checks.json').read_text(encoding='utf-8'))
                self.assertTrue(all(checks.values()), checks)
                episode = experiment/'episodes'/episode_id
                self.assertTrue((episode/'focus/feedback.csv').is_file())
                self.assertTrue((episode/'confirmation-2/monitoring.json').is_file())
        self.assertEqual(before, {str(p):sha256(p) for p in baseline.rglob('*') if p.is_file()})



