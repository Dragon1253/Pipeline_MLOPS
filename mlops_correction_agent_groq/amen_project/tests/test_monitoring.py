"""Monitoring : détection par scénario, absence de fuite et contrat avec l'agent d'analyse.

Aucun appel à un LLM : le contrat est vérifié avec la préparation et les outils
réels de l'agent (``prepare_workspace`` et ``AnalyticalMethods``).
"""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from mlops_churn.tools import AnalyticalMethods
from mlops_churn.workspace import ROOT, prepare_workspace, sha256
from monitoring.monitor import PerformanceMonitor
from monitoring.service import ReferenceContext, run_stream
from monitoring.stream import INCIDENTS, StreamSpec, intensity, simulate_stream

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


def position(window_id):
    return int(window_id[1:])


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = ReferenceContext()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_spec(self, scenario, seed, **kwargs):
        spec = StreamSpec(scenario, seed, n_windows=kwargs.pop("n_windows", 12),
                          onset=kwargs.pop("onset", 5), ramp=kwargs.pop("ramp", 2))
        return run_stream(spec, reference=self.reference, output_dir=self.out, **kwargs)

    def test_intensity_schedule(self):
        spec = StreamSpec("data_drift", 1, n_windows=10, onset=4, ramp=4)
        self.assertEqual([intensity(spec, i) for i in range(4)], [0.0] * 4)
        self.assertEqual([intensity(spec, i) for i in range(4, 9)], [0.25, 0.5, 0.75, 1.0, 1.0])
        self.assertEqual(intensity(StreamSpec("normal", 1), 15), 0.0)
        self.assertEqual(intensity(StreamSpec("concept_drift", 1, onset=3, ramp=0), 3), 1.0)

    def test_stream_is_reproducible(self):
        spec = StreamSpec("data_corruption", 4, n_windows=3, onset=1, ramp=1)
        first = [w.inputs for w, _ in simulate_stream(spec, self.reference.bundle, run_id="r")]
        second = [w.inputs for w, _ in simulate_stream(spec, self.reference.bundle, run_id="r")]
        for a, b in zip(first, second):
            self.assertTrue(a.equals(b))

    def test_labels_arrive_only_after_delay(self):
        monitor = PerformanceMonitor(self.reference.calibration, label_delay=2)
        spec = StreamSpec("normal", 2, n_windows=4)
        for window, _ in simulate_stream(spec, self.reference.bundle):
            monitor.score(window, "sha256:test", window.index)
            monitor.advance(window.index)
            for record in monitor.records.values():
                labeled = record.scored_at + 2 <= window.index
                self.assertEqual(record.metrics is not None, labeled, record.window_id)

    def test_every_incident_scenario_triggers_an_alert_on_affected_windows(self):
        for scenario in INCIDENTS:
            for seed in (1, 2, 3):
                with self.subTest(scenario=scenario, seed=seed):
                    summary = self.run_spec(scenario, seed, package=False)
                    affected = [a for a in summary["alerts"] if position(a["window_id"]) >= 5]
                    self.assertTrue(affected, summary["timeline"])
                    self.assertGreater(affected[0]["observed_value"], summary["threshold"])

    def test_escalation_gives_the_incident_its_own_alert_after_an_earlier_fluctuation(self):
        # Graine 9 : fausse alerte naturelle en w007, juste avant une corruption en w008.
        summary = run_stream(StreamSpec("data_corruption", 9), reference=self.reference,
                             output_dir=self.out, package=False)
        reasons = {i["focus"]: i["close_reason"] for i in summary["incidents"]}
        self.assertEqual(reasons.get("w007"), "escalated")
        critical = [a for a in summary["alerts"] if a["severity"] == "critical"]
        self.assertEqual(critical[0]["window_id"], "w008")

    def test_normal_streams_rarely_alert(self):
        alerts = windows = 0
        for seed in range(1, 11):
            summary = run_stream(StreamSpec("normal", seed, n_windows=20), reference=self.reference,
                                 output_dir=self.out, package=False, persistence=2)
            alerts += len(summary["alerts"])
            windows += 20
        self.assertLessEqual(alerts / windows, 0.05)

    def test_packaged_episodes_are_accepted_by_the_analysis_agent(self):
        demo = ROOT / "scenario_data" / "demo"
        before = {str(p): sha256(p) for p in demo.rglob("*") if p.is_file()}
        for scenario in INCIDENTS:
            with self.subTest(scenario=scenario):
                summary = self.run_spec(scenario, 3)
                episode = Path(summary["episodes"][-1])
                for name in ("alert.json", "context.json", "active.joblib", "focus/feedback.csv",
                             "focus/monitoring.json", "confirmation-1/monitoring.json"):
                    self.assertTrue((episode / name).is_file(), name)
                # La vérité de simulation reste hors du dossier transmis.
                self.assertFalse(any("truth" in p.name or "ground" in p.name for p in episode.rglob("*")))
                self.assertNotIn(scenario, (episode / "alert.json").read_text(encoding="utf-8"))
                _, workspace = prepare_workspace(dict(
                    source_experiment=str(demo), episode_dir=str(episode), episode_id=episode.name,
                    output_dir=str(self.out / "results"), model_version_comparison=True))
                ids = {r["id"] for r in workspace.manifest["resources"]}
                self.assertTrue({"alert", "context", "feedback", "model_catalog"} <= ids)
        self.assertEqual(before, {str(p): sha256(p) for p in demo.rglob("*") if p.is_file()})

    def test_agent_tools_see_the_injected_problem_in_the_packaged_files(self):
        def workspace_for(scenario):
            episode = Path(self.run_spec(scenario, 5, onset=4, ramp=0)["episodes"][-1])
            return episode, prepare_workspace(dict(
                source_experiment=str(self.reference.dir), episode_dir=str(episode),
                episode_id=episode.name, output_dir=str(self.out / "results"),
                model_version_comparison=True))[1]

        _, corruption = workspace_for("data_corruption")
        column = AnalyticalMethods(corruption).compare_tables("source", "inputs", columns=["monthly_charges"])["columns"][0]
        self.assertGreater(column["changed_values"], 400)
        self.assertAlmostEqual(column["median_right_over_left"], 100)

        episode, regression = workspace_for("deployment_regression")
        deployment = json.loads((episode / "context.json").read_text(encoding="utf-8"))["deployment"]
        self.assertTrue(deployment["model_changed"])
        self.assertEqual(deployment["active_version"], "sha256:" + sha256(episode / "active.joblib"))
        versions = AnalyticalMethods(regression).compare_model_versions(bootstrap_samples=100)
        self.assertGreater(versions["paired_bootstrap"]["lower"], 0)


if __name__ == "__main__":
    unittest.main()
