"""Flux complet du projet : monitoring → agent d'analyse → orchestrateur → agent de correction.

Script d'intégration : il ne modifie aucun fichier du groupe. Il réutilise tel quel le
pipeline de ``mlops_correction_agent_groq/pipeline_v4.py`` (analyse, score de confiance,
orchestrateur LangGraph, agent de correction, revue humaine), en lui fournissant le
monitoring à jour du dossier ``monitoring/`` (alerte confirmée + alerte prédictive) au lieu
de la copie figée de ``amen_project/``.

    python run_full_pipeline.py --scenario data_corruption
    python run_full_pipeline.py --scenario data_corruption --predict --label-delay 2
    python run_full_pipeline.py --all --predict
    python run_full_pipeline.py --scenario concept_drift --analysis-mode nvidia   # clé NVIDIA requise

Modes : ``--analysis-mode fixture`` (hors ligne, sans clé : ne reconnaît que la corruption),
``nvidia`` ou ``local`` (vrai agent d'analyse) ; ``--orchestrator-mode rules`` (sans clé) ou
``llm`` (Groq, clé dans ``mlops_correction_agent_groq/.env``). Les corrections restent simulées.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parent
GROUP_PIPELINE = ROOT / "mlops_correction_agent_groq"
OUTPUT = ROOT / "monitoring_runs" / "integration"
SCENARIOS = ("data_corruption", "data_drift", "concept_drift", "deployment_regression", "normal")

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
# Les demandes de revue humaine sont enregistrées ici, pas dans le dossier du groupe
# (sauf si HUMAN_REVIEW_DB est déjà défini).
os.environ.setdefault("HUMAN_REVIEW_DB", str(OUTPUT / "human_reviews.sqlite3"))

# 1. Charger d'abord le monitoring et l'agent d'analyse de la racine : Python les garde en
#    mémoire, et pipeline_v4 réutilisera ces versions au lieu de celles de amen_project/.
sys.path.insert(0, str(ROOT))
import monitoring.service as monitoring_service  # noqa: E402
import mlops_churn  # noqa: E402,F401
from monitoring.stream import StreamSpec  # noqa: E402

sys.path.insert(1, str(GROUP_PIPELINE))
import pipeline_v4  # noqa: E402  (code du groupe, inchangé)

assert pipeline_v4.run_stream.__module__ == "monitoring.service"
assert Path(monitoring_service.__file__).resolve().parent == ROOT / "monitoring", \
    "Le pipeline doit utiliser le monitoring de la racine."


def run(scenario, *, seed=42, predict=False, label_delay=1, analysis_mode="fixture",
        orchestrator_mode="rules", correction_mode="rules", n_windows=20, onset=8, ramp=4):
    """Lance le pipeline du groupe avec le monitoring à jour ; retourne son résultat enrichi."""
    predictor = None
    if predict:
        from monitoring.predictor import load_predictor
        predictor = load_predictor()
    # Injection : même fonction run_stream, avec en plus le prédicteur et le retard des labels.
    pipeline_v4.run_stream = partial(monitoring_service.run_stream, predictor=predictor, label_delay=label_delay)
    result = pipeline_v4.run_pipeline(scenario=scenario, seed=seed, analysis_mode=analysis_mode,
                                      orchestrator_mode=orchestrator_mode, correction_mode=correction_mode,
                                      output_dir=str(OUTPUT), n_windows=n_windows, onset=onset, ramp=ramp)
    summary_file = Path(result["monitoring"]["run_dir"]) / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    result["monitoring"]["predictive_alerts"] = summary.get("predictive_alerts", [])
    result["monitoring"]["label_delay"] = label_delay
    return result


def report(scenario, result):
    m = result["monitoring"]
    lines = [f"\n=== {scenario} ===", "1. Monitoring"]
    for alert in m["alerts"]:
        lines.append(f"   alerte confirmée {alert['window_id']} ({alert.get('severity')}), "
                     f"log loss {alert['observed_value']:.3f}, levée au pas {alert.get('detected_at_step')}")
    for alert in m.get("predictive_alerts", []):
        lead = f", {alert['lead_steps']} pas avant l'alerte confirmée" if alert.get("lead_steps") is not None else ""
        signals = ", ".join(s["signal"] for s in alert.get("top_signals", []))
        lines.append(f"   alerte prédictive {alert['window_id']} (risque {alert['risk']:.2f}, {alert['status']}{lead}) : {signals}")
    if not m["alerts"]:
        lines.append("   aucune alerte confirmée : rien n'est transmis à l'analyse")
    else:
        lines.append(f"   dossier transmis à l'analyse : {Path(m['episode_dir']).name if m['episode_dir'] else '—'}")
    analysis = result.get("analysis")
    if analysis:
        diagnosis = analysis.get("diagnosis") or {}
        confidence = (analysis.get("confidence_assessment") or {})
        lines += ["2. Agent d'analyse", f"   cause : {diagnosis.get('cause_probable')} — {diagnosis.get('resume', '')}",
                  f"   confiance : {confidence.get('confidence')} (type {confidence.get('incident_type')}, "
                  f"preuve {confidence.get('supporting_tool')})"]
    orchestration = result.get("orchestration")
    if orchestration:
        lines += ["3. Orchestrateur", f"   décision : {orchestration['decision'].get('route')} — "
                                      f"{orchestration['decision'].get('reason', '')}"]
    correction = result.get("correction")
    if correction:
        plan, tool = correction["final_plan"], correction["tool_result"]
        lines += ["4. Agent de correction", f"   action : {plan.get('action')} ({tool.get('status')}), "
                  f"production inchangée : {correction['verification'].get('production_unchanged')}"]
    elif result.get("human_review"):
        lines += ["4. Revue humaine", f"   demande enregistrée : {result['human_review'].get('status', 'oui')}"]
    lines.append(f"Statut final : {result['status']} — résultat : {result.get('result_file')}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=SCENARIOS, default="data_corruption")
    parser.add_argument("--all", action="store_true", help="enchaîner les 5 scénarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", action="store_true", help="alerte prédictive (python -m monitoring train-predictor avant)")
    parser.add_argument("--label-delay", type=int, default=1)
    parser.add_argument("--analysis-mode", choices=("fixture", "nvidia", "local"), default="fixture")
    parser.add_argument("--orchestrator-mode", choices=("rules", "llm"), default="rules")
    parser.add_argument("--correction-mode", choices=("rules", "llm"), default="rules")
    args = parser.parse_args(argv)
    for scenario in (SCENARIOS if args.all else (args.scenario,)):
        result = run(scenario, seed=args.seed, predict=args.predict, label_delay=args.label_delay,
                     analysis_mode=args.analysis_mode, orchestrator_mode=args.orchestrator_mode,
                     correction_mode=args.correction_mode)
        print(report(scenario, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
