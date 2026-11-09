import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as review_app
from human_review_store import HumanReviewStore


class HumanReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "reviews.sqlite3"
        self.run_dir = self.root / "run-1"
        self.run_dir.mkdir()
        self.result_file = self.run_dir / "pipeline_result.json"
        self.result_file.write_text(
            json.dumps({"status": "human_review", "human_review": {}}),
            encoding="utf-8",
        )
        self.env_patch = patch.dict(os.environ, {
            "HUMAN_REVIEW_DB": str(self.database),
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        review_app.app.config.update(TESTING=True)
        self.client = review_app.app.test_client()

    def create_review(self, *, incident_type="qualite_donnees"):
        incident = {
            "incident_id": "REVIEW-TEST",
            "type": incident_type,
            "confidence": 0.35,
            "metrics": {},
            "details": {
                "confidence_supported": False,
                "requires_human_review": True,
                "batch_id": "BATCH-REVIEW-TEST",
            },
        }
        created = HumanReviewStore(self.database).create_review(
            request={
                "incident_id": incident["incident_id"],
                "incident_type": incident_type,
                "confidence": incident["confidence"],
                "cause_probable": incident_type,
                "description": "Incident de test.",
                "evidence_ids": [],
                "reason": "Confiance insuffisante.",
            },
            incident=incident,
            correction_mode="rules",
            run_dir=self.run_dir,
            pipeline_result_file=self.result_file,
        )
        return created

    def test_explicit_approval_runs_only_the_deterministic_simulation(self):
        created = self.create_review()
        page = self.client.get(
            f"/human-review/{created['review_id']}?token={created['token']}"
        )
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Approuver", page.data)

        response = self.client.post(
            f"/human-review/{created['review_id']}/decision",
            data={"token": created["token"], "decision": "approved"},
        )
        self.assertEqual(response.status_code, 302)
        review = HumanReviewStore(self.database).get_review(
            created["review_id"], created["token"]
        )
        self.assertEqual(review["status"], "completed")
        self.assertEqual(review["execution"]["plan"]["action"], "quarantine_batch")
        self.assertEqual(review["execution"]["tool_result"]["status"], "simulated")
        self.assertTrue(review["execution"]["verification"]["production_unchanged"])

        saved_result = json.loads(self.result_file.read_text(encoding="utf-8"))
        self.assertEqual(saved_result["status"], "completed_after_human_approval")
        self.assertEqual(saved_result["correction"]["final_plan"]["action"], "quarantine_batch")
        self.assertEqual(
            self.client.post(
                f"/human-review/{created['review_id']}/decision",
                data={"token": created["token"], "decision": "approved"},
            ).status_code,
            409,
        )

    def test_rejection_does_not_run_correction(self):
        created = self.create_review()
        response = self.client.post(
            f"/human-review/{created['review_id']}/decision",
            data={"token": created["token"], "decision": "rejected"},
        )
        self.assertEqual(response.status_code, 302)
        review = HumanReviewStore(self.database).get_review(
            created["review_id"], created["token"]
        )
        self.assertEqual(review["status"], "rejected")
        self.assertFalse((self.run_dir / "human_approved_correction").exists())

    def test_invalid_token_cannot_read_or_resolve_review(self):
        created = self.create_review()
        self.assertEqual(
            self.client.get(
                f"/human-review/{created['review_id']}?token=wrong-token"
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                f"/human-review/{created['review_id']}/decision",
                data={"token": "wrong-token", "decision": "approved"},
            ).status_code,
            403,
        )

    def test_unknown_incident_approval_is_recorded_but_action_is_blocked(self):
        created = self.create_review(incident_type="inconnu")
        response = self.client.post(
            f"/human-review/{created['review_id']}/decision",
            data={"token": created["token"], "decision": "approved"},
        )
        self.assertEqual(response.status_code, 302)
        review = HumanReviewStore(self.database).get_review(
            created["review_id"], created["token"]
        )
        self.assertEqual(review["status"], "blocked")
        self.assertIn("type inconnu", review["execution"]["error"])
        self.assertFalse((self.run_dir / "human_approved_correction").exists())


if __name__ == "__main__":
    unittest.main()
