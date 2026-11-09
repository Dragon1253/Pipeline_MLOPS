import tempfile
import unittest
from pathlib import Path
from agent import CorrectionAgent, Incident

class AgentTests(unittest.TestCase):
    def test_scenarios(self):
        expected = {'fausse_alerte':'close_alert','derive_donnees':'request_retraining','qualite_donnees':'quarantine_batch','regression_version':'request_rollback'}
        with tempfile.TemporaryDirectory() as d:
            agent = CorrectionAgent(Path(d)/'log.jsonl')
            for kind, action in expected.items():
                result = agent.decide(Incident(kind, kind, 0.9, {}, {}))
                self.assertEqual(result['action'], action)
            self.assertEqual(agent.decide(Incident('low', 'derive_donnees', 0.5, {}, {}))['action'], 'human_review')
            self.assertEqual(len((Path(d)/'log.jsonl').read_text().splitlines()), 5)

if __name__ == '__main__':
    unittest.main()
