import tempfile
import unittest
import os
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from human_review_store import HumanReviewStore
from orchestrator_v4 import OrchestratorAgentV4, load_local_groq_settings


class FakeStructuredLLM:
    def __init__(self, route):
        self.route = route

    def with_structured_output(self, schema, **_kwargs):
        route = self.route

        class Structured:
            def invoke(self, _prompt):
                return schema(route=route, reason="Route choisie par le test du LLM.")

        return Structured()


class FakeUsageStructuredLLM:
    def with_structured_output(self, schema, **kwargs):
        class RawResponse:
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                }
            }

        class Structured:
            def invoke(self, _prompt):
                decision = schema(route="correction", reason="Route choisie par le test du LLM.")
                if kwargs.get("include_raw"):
                    return {"raw": RawResponse(), "parsed": decision, "parsing_error": None}
                return decision

        return Structured()


def incident(confidence, *, supported=True, requires_human_review=False):
    return {
        "incident_id": "ORCH-TEST",
        "type": "qualite_donnees",
        "confidence": confidence,
        "metrics": {},
        "details": {
            "batch_id": "BATCH-TEST",
            "confidence_supported": supported,
            "requires_human_review": requires_human_review,
        },
    }


class OrchestratorAgentTests(unittest.TestCase):
    def test_loads_groq_settings_from_env_without_overwriting_shell_values(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = __import__("pathlib").Path(folder) / ".env"
            env_file.write_text(
                '# local credentials\nGROQ_API_KEY="test-key"\nGROQ_MODEL=openai/gpt-oss-120b\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GROQ_API_KEY": "", "GROQ_MODEL": "shell-model"}):
                load_local_groq_settings(env_file)
                self.assertEqual(os.environ["GROQ_API_KEY"], "test-key")
                self.assertEqual(os.environ["GROQ_MODEL"], "shell-model")

    def test_llm_routes_supported_incident_to_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = OrchestratorAgentV4(
                mode="llm",
                llm=FakeStructuredLLM("correction"),
                out_dir=folder,
            )
            result = agent.run(incident(0.9))

        self.assertEqual(result["decision"]["route"], "correction")
        self.assertTrue(result["guard_result"]["accepted"])
        self.assertEqual(result["correction_result"]["plan"]["action"], "quarantine_batch")
        self.assertIsNone(result["human_review"])

    def test_preserves_token_usage_from_raw_structured_response(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = OrchestratorAgentV4(
                mode="llm",
                llm=FakeUsageStructuredLLM(),
                out_dir=folder,
            )
            result = agent.run(incident(0.9))

        self.assertEqual(
            result["proposed_decision"]["usage"],
            {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )

    def test_guard_overrides_llm_when_confidence_is_too_low(self):
        notification = {"status": "not_configured", "missing_settings": ["SMTP_HOST"]}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "HUMAN_REVIEW_DB": os.path.join(folder, "reviews.sqlite3"),
            "HUMAN_REVIEW_BASE_URL": "http://127.0.0.1:5000",
        }), patch(
            "human_notifications.send_human_review_email",
            return_value=notification,
        ) as send_email:
            agent = OrchestratorAgentV4(
                mode="llm",
                llm=FakeStructuredLLM("correction"),
                out_dir=folder,
            )
            result = agent.run(incident(0.35, supported=False, requires_human_review=True))
            email_request = send_email.call_args.args[0]
            token = parse_qs(urlsplit(email_request["review_url"]).query)["token"][0]
            stored_review = HumanReviewStore(
                os.path.join(folder, "reviews.sqlite3")
            ).get_review(result["human_review"]["review_id"], token)

        self.assertEqual(stored_review["status"], "pending")
        self.assertNotIn("token", result["human_review"])
        self.assertEqual(result["proposed_decision"]["route"], "correction")
        self.assertEqual(result["decision"]["route"], "human_review")
        self.assertFalse(result["guard_result"]["accepted"])
        self.assertIsNone(result["correction_result"])
        self.assertTrue(result["human_review"]["required"])
        self.assertEqual(result["human_review"]["notification"], notification)
        send_email.assert_called_once()


if __name__ == "__main__":
    unittest.main()