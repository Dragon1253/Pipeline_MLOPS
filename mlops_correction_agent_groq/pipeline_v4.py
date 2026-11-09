"""Run monitoring, Amen investigation, and guarded correction in sequence."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
AMEN_ROOT = HERE / "amen_project"
sys.path.insert(0, str(AMEN_ROOT))

from integration_adapter import (
    adapt_amen_diagnosis,
    is_data_transformation_anomaly,
    score_amen_diagnosis,
)
from monitoring.service import ReferenceContext, run_stream
from monitoring.stream import STREAM_SCENARIOS, StreamSpec
from orchestrator_v4 import OrchestratorAgentV4


def _fixture_analysis(*, episode_dir, reference_dir, output_dir, **_kwargs):
    from mlops_churn.tools import AnalyticalMethods
    from mlops_churn.workspace import prepare_workspace

    _, workspace = prepare_workspace({
        "source_experiment": str(reference_dir),
        "episode_dir": str(episode_dir),
        "episode_id": Path(episode_dir).name,
        "output_dir": str(Path(output_dir) / "fixture_workspace"),
        "model_version_comparison": True,
    })
    comparison = AnalyticalMethods(workspace).compare_tables("source", "inputs")
    evidence = [{
        "evidence_id": "E0001",
        "tool": "compare_tables",
        "arguments": {"left": "source", "right": "inputs"},
        "result": {"status": "ok", "observations": comparison},
    }]
    anomalies = [
        column for column in comparison["columns"]
        if isinstance(column.get("median_right_over_left"), (int, float))
        and column.get("changed_values", 0) / max(comparison["matched_rows"], 1) >= 0.2
        and (column["median_right_over_left"] >= 3 or column["median_right_over_left"] <= 1 / 3)
    ]
    if is_data_transformation_anomaly(comparison):
        column = anomalies[0]["column"] if anomalies else "entree"
        ratio = anomalies[0].get("median_right_over_left") if anomalies else None
        diagnosis = {
            "cause_probable": "data_corruption",
            "resume": f"Comparaison source/inputs: transformation anormale sur {column} (ratio median {ratio}).",
            "preuves": ["E0001"],
        }
    else:
        diagnosis = {
            "cause_probable": "undetermined",
            "resume": "Aucune corruption source/inputs forte detectee par le test hors ligne.",
            "preuves": ["E0001"],
        }
    return {
        "status": "completed",
        "final_diagnosis": diagnosis,
        "evidence": evidence,
    }


def run_pipeline(*, scenario="data_corruption", seed=42, analysis_mode="fixture",
                 correction_mode="rules", orchestrator_mode="rules", output_dir="artifacts_v4",
                 n_windows=20, onset=8, ramp=4, analysis_runner=None, orchestrator_llm=None):
    """Run all stages and save a compact result beside the monitoring run."""
    if analysis_mode not in {"fixture", "nvidia", "local"}:
        raise ValueError("analysis_mode doit etre fixture, nvidia ou local.")

    output_root = Path(output_dir)
    reference = ReferenceContext()
    monitoring = run_stream(
        StreamSpec(scenario, seed, n_windows=n_windows, onset=onset, ramp=ramp),
        reference=reference,
        output_dir=output_root / "monitoring_runs",
    )
    run_dir = Path(monitoring["run_dir"]).resolve()
    episodes = [str(Path(episode).resolve()) for episode in monitoring["episodes"]]
    result = {
        "status": "no_alert" if not monitoring["alerts"] else "no_episode",
        "monitoring": {
            "run_id": monitoring["run_id"],
            "run_dir": str(run_dir),
            "alerts": monitoring["alerts"],
            "episode_dir": episodes[-1] if episodes else None,
        },
        "analysis": None,
        "correction": None,
    }

    if not monitoring["alerts"] or not episodes:
        _save_result(run_dir, result)
        return result

    episode_dir = Path(episodes[-1])
    if analysis_runner is not None:
        analysis_result = analysis_runner(
            episode_dir=str(episode_dir),
            reference_dir=str(reference.dir),
            provider=analysis_mode,
            output_dir=str(run_dir / "analysis_results"),
        )
    elif analysis_mode == "fixture":
        analysis_result = _fixture_analysis(
            episode_dir=str(episode_dir),
            reference_dir=str(reference.dir),
            output_dir=str(run_dir / "analysis_results"),
        )
    else:
        from mlops_churn.agent import investigate

        analysis_result = investigate(
            episode_dir=str(episode_dir),
            reference_dir=str(reference.dir),
            provider=analysis_mode,
            output_dir=str(run_dir / "analysis_results"),
        )

    diagnosis = analysis_result.get("final_diagnosis")
    result["analysis"] = {
        "mode": analysis_mode,
        "status": analysis_result.get("status"),
        "diagnosis": diagnosis,
        "run_directory": analysis_result.get("run_directory"),
    }
    if analysis_result.get("status") != "completed" or not isinstance(diagnosis, dict):
        result["status"] = "analysis_incomplete"
        _save_result(run_dir, result)
        return result

    try:
        confidence_assessment = score_amen_diagnosis(analysis_result, monitoring["alerts"])
        result["analysis"]["confidence_assessment"] = confidence_assessment
        incident = adapt_amen_diagnosis(
            analysis_result,
            confidence_assessment=confidence_assessment,
        )
    except (TypeError, ValueError) as exc:
        result["status"] = "analysis_invalid"
        result["error"] = str(exc)
        _save_result(run_dir, result)
        return result

    workflow = OrchestratorAgentV4(
        mode=orchestrator_mode,
        correction_mode=correction_mode,
        out_dir=run_dir,
        llm=orchestrator_llm,
    ).run(incident)
    result["orchestration"] = {
        "mode": orchestrator_mode,
        "proposed_decision": workflow["proposed_decision"],
        "decision": workflow["decision"],
        "guard": workflow["guard_result"],
    }
    if workflow["correction_result"] is None:
        result["status"] = "human_review"
        result["human_review"] = workflow["human_review"]
        result["correction"] = None
        _save_result(run_dir, result)
        return result

    correction = workflow["correction_result"]
    result["status"] = "completed"
    result["correction"] = {
        "incident": incident,
        "proposed_plan": correction["proposed_plan"],
        "final_plan": correction["plan"],
        "guard": correction["guard_result"],
        "tool_result": correction["tool_result"],
        "verification": correction["verification"],
    }
    _save_result(run_dir, result)
    return result


def _save_result(run_dir, result):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "pipeline_result.json"
    result["result_file"] = str(result_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=STREAM_SCENARIOS, default="data_corruption")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-mode", choices=("fixture", "nvidia", "local"), default="fixture",
                        help="fixture est hors ligne; nvidia/local appellent le modele d'analyse")
    parser.add_argument("--orchestrator-mode", choices=("rules", "llm"), default="rules",
                        help="llm utilise Groq pour router vers correction ou revue humaine")
    parser.add_argument("--correction-mode", choices=("rules", "llm"), default="rules")
    parser.add_argument("--output-dir", default="artifacts_v4")
    args = parser.parse_args(argv)
    result = run_pipeline(
        scenario=args.scenario,
        seed=args.seed,
        analysis_mode=args.analysis_mode,
        orchestrator_mode=args.orchestrator_mode,
        correction_mode=args.correction_mode,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] in {"completed", "human_review", "no_alert"} else 1


if __name__ == "__main__":
    raise SystemExit(main())