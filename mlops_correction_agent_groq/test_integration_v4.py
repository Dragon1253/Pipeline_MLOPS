
import tempfile
import unittest
from pathlib import Path

from agent_v3 import CorrectionAgentV3


class IntegrationV4Tests(unittest.TestCase):

    def test_rollback_with_local_models(self):
        # Les modèles sont générés par ml_pipeline_v3.py
        models = Path("artifacts_v3/pipeline")
        self.assertTrue(
            (models / "v1_reference.joblib").exists()
        )
        self.assertTrue(
            (models / "v2_candidate.joblib").exists()
        )

        incident = {
            "incident_id": "S4-INTEGRATION",
            "type": "regression_version",
            "confidence": 0.95,
            "metrics": {},
            "details": {
                "run_local_rollback_demo": True
            },
        }

        with tempfile.TemporaryDirectory() as folder:
            agent = CorrectionAgentV3(
                mode="rules",
                out_dir=folder,
            )
            result = agent.run(incident)

        self.assertEqual(
            result["plan"]["action"],
            "prepare_rollback",
        )
        self.assertTrue(result["guard_result"]["accepted"])
        self.assertTrue(result["verification"]["safe"])
        self.assertTrue(
            result["tool_result"]["rollback_recommended"]
        )
        self.assertFalse(
            result["tool_result"]["rollback_executed"]
        )

    def test_low_confidence_blocks_rollback(self):
        incident = {
            "incident_id": "S4-LOW-CONFIDENCE",
            "type": "regression_version",
            "confidence": 0.4,
            "metrics": {},
            "details": {
                "run_local_rollback_demo": True
            },
        }

        with tempfile.TemporaryDirectory() as folder:
            agent = CorrectionAgentV3(
                mode="rules",
                out_dir=folder,
            )
            result = agent.run(incident)

        self.assertEqual(
            result["plan"]["action"],
            "human_review",
        )
        self.assertTrue(
            result["verification"]["production_unchanged"]
        )


if __name__ == "__main__":
    unittest.main()
