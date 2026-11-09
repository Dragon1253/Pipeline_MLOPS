import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integration_adapter import score_amen_diagnosis
from pipeline_v4 import run_pipeline


class FullPipelineTests(unittest.TestCase):
    def test_monitoring_fixture_analysis_and_correction_are_connected(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_pipeline(
                scenario="data_corruption",
                seed=5,
                analysis_mode="fixture",
                output_dir=folder,
                n_windows=12,
                onset=5,
                ramp=2,
            )

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["monitoring"]["alerts"])
            self.assertTrue(Path(result["monitoring"]["episode_dir"]).is_dir())
            self.assertEqual(result["analysis"]["mode"], "fixture")
            self.assertEqual(result["analysis"]["diagnosis"]["cause_probable"], "data_corruption")
            self.assertGreaterEqual(result["analysis"]["confidence_assessment"]["confidence"], 0.7)
            self.assertEqual(result["correction"]["final_plan"]["action"], "quarantine_batch")
            self.assertEqual(result["correction"]["tool_result"]["status"], "simulated")
            self.assertTrue(result["correction"]["verification"]["production_unchanged"])
            saved = json.loads(Path(result["result_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["correction"]["final_plan"]["action"], "quarantine_batch")

    def test_incomplete_analysis_stops_before_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "monitoring" / "run-1"
            episode_dir = run_dir / "episodes" / "episode-1"
            episode_dir.mkdir(parents=True)
            summary = {
                "run_id": "run-1",
                "run_dir": str(run_dir),
                "alerts": [{"alert_id": "A1"}],
                "episodes": [str(episode_dir)],
            }
            with patch("pipeline_v4.run_stream", return_value=summary), patch(
                "pipeline_v4.OrchestratorAgentV4"
            ) as orchestrator:
                result = run_pipeline(
                    output_dir=folder,
                    analysis_runner=lambda **_kwargs: {
                        "status": "failed_or_incomplete",
                        "final_diagnosis": None,
                    },
                )

            self.assertEqual(result["status"], "analysis_incomplete")
            self.assertIsNone(result["correction"])
            orchestrator.assert_not_called()

    def test_named_cause_without_matching_successful_evidence_stays_below_threshold(self):
        assessment = score_amen_diagnosis({
            "final_diagnosis": {
                "cause_probable": "data_corruption",
                "resume": "Cause annoncee mais preuve non liee.",
                "preuves": ["E0001"],
            },
            "evidence": [{
                "evidence_id": "E0001",
                "tool": "inspect_files",
                "result": {"status": "ok"},
            }],
        }, [{"alert_type": "confirmed", "severity": "critical"}])

        self.assertEqual(assessment["incident_type"], "qualite_donnees")
        self.assertEqual(assessment["confidence"], 0.35)
        self.assertTrue(assessment["requires_human_review"])

    def test_model_regression_evidence_allows_rollback_recommendation(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "monitoring" / "run-2"
            episode_dir = run_dir / "episodes" / "episode-2"
            episode_dir.mkdir(parents=True)
            summary = {
                "run_id": "run-2",
                "run_dir": str(run_dir),
                "alerts": [{"alert_id": "A2", "alert_type": "confirmed", "severity": "critical"}],
                "episodes": [str(episode_dir)],
            }
            analysis = {
                "status": "completed",
                "final_diagnosis": {
                    "cause_probable": "deployment_regression",
                    "resume": "Le modele actif regresse sur le lot confirme.",
                    "preuves": ["E0001"],
                },
                "evidence": [{
                    "evidence_id": "E0001",
                    "tool": "compare_model_versions",
                    "result": {
                        "status": "ok",
                        "observations": {
                            "candidate_minus_reference_log_loss": 0.8,
                            "paired_bootstrap": {"lower": 0.4, "upper": 1.1},
                        },
                    },
                }],
            }
            with patch("pipeline_v4.run_stream", return_value=summary):
                result = run_pipeline(
                    output_dir=folder,
                    analysis_runner=lambda **_kwargs: analysis,
                )

        self.assertEqual(result["analysis"]["confidence_assessment"]["incident_type"], "regression_version")
        self.assertGreaterEqual(result["analysis"]["confidence_assessment"]["confidence"], 0.7)
        self.assertEqual(result["correction"]["final_plan"]["action"], "prepare_rollback")
        self.assertFalse(result["correction"]["tool_result"]["rollback_executed"])


if __name__ == "__main__":
    unittest.main()