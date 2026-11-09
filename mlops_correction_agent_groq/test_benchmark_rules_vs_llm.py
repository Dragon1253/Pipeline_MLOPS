import json
import tempfile
import unittest
from unittest.mock import patch

from benchmark_rules_vs_llm import benchmark_rules_vs_llm, make_seeded_incident


class FakeStructuredLLM:
    def __init__(self, route):
        self.route = route

    def with_structured_output(self, schema, **_kwargs):
        route = self.route

        class Structured:
            def invoke(self, _prompt):
                return schema(route=route, reason="route via benchmark test")

        return Structured()


class RateLimitError(Exception):
    pass


class BenchmarkRulesVsLLMTests(unittest.TestCase):
    def test_benchmark_reports_modes_and_false_corrections(self):
        summary = benchmark_rules_vs_llm(
            seeds=(1, 2),
            incident_types=("fausse_alerte", "qualite_donnees"),
            modes=("rules", "llm"),
            scenarios=("standard",),
            llm_factory=lambda incident: FakeStructuredLLM(
                "correction" if incident["details"].get("confidence_supported") else "human_review"
            ),
        )

        self.assertEqual(set(summary["modes"]), {"rules", "llm"})
        self.assertIn("totals", summary)
        self.assertGreaterEqual(summary["totals"]["runs"], 4)
        self.assertIn("false_corrections", summary["totals"])
        self.assertIn("wrong_routes", summary["totals"])
        self.assertIn("avg_latency_ms", summary["totals"])
        self.assertIn("mode_summary", summary)
        self.assertTrue(all("latency_ms" in record for record in summary["records"]))
        self.assertIn("csv_path", summary)

    def test_threshold_near_scenarios_and_cost_metrics_are_exported(self):
        threshold_incident = make_seeded_incident(7, "fausse_alerte", scenario="near_threshold")
        self.assertAlmostEqual(threshold_incident["confidence"], 0.7, delta=0.08)

        summary = benchmark_rules_vs_llm(
            seeds=(7,),
            incident_types=("fausse_alerte",),
            modes=("rules",),
            scenarios=("near_threshold",),
            llm_factory=lambda incident: FakeStructuredLLM(
                "correction" if incident["details"].get("confidence_supported") else "human_review"
            ),
        )

        self.assertIn("total_cost_usd", summary["totals"])
        self.assertIn("avg_cost_usd_per_run", summary["totals"])
        self.assertIn("total_cost_usd", summary["mode_summary"]["rules"])
        self.assertGreaterEqual(summary["totals"]["total_cost_usd"], 0.0)

    def test_benchmark_human_review_does_not_send_email(self):
        with patch("human_notifications.send_human_review_email") as send_email:
            summary = benchmark_rules_vs_llm(
                seeds=(2,),
                incident_types=("fausse_alerte",),
                modes=("rules",),
                scenarios=("standard",),
                output_dir=None,
            )

        self.assertEqual(summary["records"][0]["actual_route"], "human_review")
        send_email.assert_not_called()

    def test_rate_limit_saves_partial_json_and_csv_reports(self):
        class RateLimitedAgent:
            def __init__(self, **_kwargs):
                pass

            def run(self, _incident):
                raise RateLimitError("daily limit")

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "benchmark_rules_vs_llm.OrchestratorAgentV4", RateLimitedAgent
        ):
            summary = benchmark_rules_vs_llm(
                seeds=(1, 2),
                incident_types=("fausse_alerte",),
                modes=("llm",),
                scenarios=("standard",),
                llm_factory=lambda _incident: object(),
                output_dir=output_dir,
            )

            with open(summary["report_path"], encoding="utf-8") as report_file:
                saved_report = json.load(report_file)

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["totals"]["runs"], 0)
        self.assertEqual(saved_report["status"], "partial")
        self.assertTrue(summary["csv_path"])


if __name__ == "__main__":
    unittest.main()
