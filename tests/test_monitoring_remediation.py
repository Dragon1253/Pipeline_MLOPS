"""Correction appliquée dans la simulation : l'erreur redescend seulement si l'action est adaptée."""
import unittest
import warnings

from monitoring.remediation import apply_correction, planned_correction, propose_solutions
from monitoring.service import ReferenceContext
from monitoring.stream import StreamSpec

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


class RemediationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = ReferenceContext()
        cls.threshold = cls.reference.calibration["threshold"]

    def run_action(self, scenario, action):
        return apply_correction(action, StreamSpec(scenario, 42), self.reference.bundle, self.threshold, n_after=3)

    def test_adapted_actions_bring_error_back_under_threshold(self):
        for scenario, action in [("deployment_regression", "prepare_rollback"), ("data_corruption", "quarantine_batch"),
                                 ("concept_drift", "prepare_retraining")]:
            with self.subTest(scenario=scenario):
                result = self.run_action(scenario, action)
                self.assertGreater(result["before_mean"], self.threshold)
                self.assertTrue(result["passed"])
                self.assertEqual(result["windows_above_threshold"], 0)

    def test_inadapted_action_fails_verification(self):
        self.assertFalse(self.run_action("concept_drift", "prepare_rollback")["passed"])
        self.assertFalse(self.run_action("data_corruption", "close_alert")["passed"])

    def test_retraining_is_deployed_only_if_better(self):
        details = self.run_action("concept_drift", "prepare_retraining")["details"]
        self.assertTrue(details["candidate_deployed"])
        self.assertLess(details["candidate_log_loss"], details["current_log_loss"])

    def test_failed_correction_gets_working_proposal(self):
        spec = StreamSpec("concept_drift", 42)
        self.assertFalse(self.run_action("concept_drift", "prepare_rollback")["passed"])
        proposals = propose_solutions(spec, self.reference.bundle, self.threshold,
                                      self.reference.calibration["reference_value"], tried=["prepare_rollback"])
        best = proposals["automatic"][0]
        self.assertEqual(best["action"], "prepare_retraining")
        self.assertTrue(best["would_fix"])

    def test_unpredictable_population_proposes_threshold_recalibration(self):
        spec = StreamSpec("data_drift", 42)
        proposals = propose_solutions(spec, self.reference.bundle, self.threshold,
                                      self.reference.calibration["reference_value"], tried=["prepare_retraining"])
        recalibration = next(o for o in proposals["automatic"] if o["action"] == "recalibrate_threshold")
        result = apply_correction("recalibrate_threshold", spec, self.reference.bundle, self.threshold,
                                  n_after=3, new_threshold=recalibration["threshold"])
        self.assertGreater(result["threshold"], self.threshold)
        self.assertTrue(result["passed"])

    def test_planned_correction_matches_incident_type(self):
        self.assertEqual(planned_correction("regression_version")[0], "prepare_rollback")
        self.assertEqual(planned_correction("derive_donnees")[0], "prepare_retraining")
        self.assertIsNone(planned_correction("inconnu")[0])


if __name__ == "__main__":
    unittest.main()
