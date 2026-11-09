import tempfile
import unittest
from agent_v3 import CorrectionAgentV3
from demo_v3 import SCENARIOS

class AgentV3Tests(unittest.TestCase):
    def test_scenarios(self):
        expected = ["prepare_retraining", "close_alert", "quarantine_batch",
                    "prepare_rollback", "human_review"]
        with tempfile.TemporaryDirectory() as folder:
            agent = CorrectionAgentV3(mode="rules", out_dir=folder)
            for incident, action in zip(SCENARIOS, expected):
                with self.subTest(incident=incident["incident_id"]):
                    result = agent.run(incident)
                    self.assertEqual(result["plan"]["action"], action)
                    self.assertTrue(result["verification"]["safe"])
                    self.assertTrue(result["verification"]["production_unchanged"])

    def test_llm_cannot_override_guard(self):
        class FakeLLM:
            def with_structured_output(self, schema, **kwargs):
                class Structured:
                    def invoke(self, prompt):
                        return schema(action="prepare_retraining", reason="Mauvaise proposition volontaire")
                return Structured()
        with tempfile.TemporaryDirectory() as folder:
            agent = CorrectionAgentV3(mode="llm", out_dir=folder, llm=FakeLLM())
            result = agent.run(SCENARIOS[2])
            self.assertEqual(result["plan"]["action"], "human_review")
            self.assertTrue(result["verification"]["safe"])

if __name__ == "__main__":
    unittest.main()
