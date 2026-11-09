
import json
from datetime import datetime, timezone
from pathlib import Path
from integration_adapter import adapt_amen_diagnosis
from amen_runner import run_amen_investigation

from flask import Flask, render_template, request, redirect, url_for, session
from agent_v3 import CorrectionAgentV3

app = Flask(__name__)
app.secret_key = "dev-only-change-this-key"

ARTIFACTS = Path("artifacts_v3")
ARTIFACTS.mkdir(exist_ok=True)

REPORT_FILE = ARTIFACTS / "flask_s3_reports.json"
DECISIONS_FILE = ARTIFACTS / "human_decisions.jsonl"


def save_report(report):
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_report():
    if REPORT_FILE.exists():
        with REPORT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.route("/", methods=["GET"])
def index():
    report = load_report()
    return render_template("index.html", report=report)


@app.route("/analyze", methods=["POST"])
def analyze():
    batch_id = request.form.get("batch_id", "").strip()
    missing_rate = float(request.form.get("missing_rate", 35))
    temperature_max = float(request.form.get("temperature_max", 110))
    confidence = float(request.form.get("confidence", 0.95))

    if not batch_id or not 0 <= missing_rate <= 100:
        return "Parametres invalides", 400

    if not 0 <= confidence <= 1:
        return "Confiance invalide", 400

    incident = {
        "incident_id": "S3-" + batch_id,
        "type": "qualite_donnees",
        "confidence": confidence,
        "metrics": {
            "missing_rate": missing_rate / 100,
            "baseline_missing_rate": 0.01,
            "temperature_max": temperature_max
        },
        "details": {
            "description": (
                "Lot suspect : valeurs manquantes "
                "et temperatures anormales."
            ),
            "batch_id": batch_id,
            "data_quality_validated": False
        }
    }

    agent = CorrectionAgentV3(
        mode="llm",
        out_dir="artifacts_v3"
    )

    result = agent.run(incident)

    report = {
        "incident": incident,
        "proposed_action": result["proposed_plan"]["action"],
        "final_action": result["plan"]["action"],
        "guard": result["guard_result"],
        "tool_result": result["tool_result"],
        "verification": result["verification"],
        "human_decision": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    save_report(report)
    return redirect(url_for("index"))


@app.route("/decision", methods=["POST"])
def decision():
    choice = request.form.get("choice")

    choices = {
        "approve": "approved",
        "investigate": "investigation_requested",
        "reject": "rejected"
    }

    if choice not in choices:
        return "Decision invalide", 400

    report = load_report()
    if not report:
        return "Aucun incident analyse", 400

    if report["human_decision"] is not None:
        return "Decision deja enregistree", 409

    human_decision = {
        "incident_id": report["incident"]["incident_id"],
        "recommended_action": report["final_action"],
        "human_decision": choices[choice],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "production_modified": False
    }

    with DECISIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(human_decision, ensure_ascii=False) + "\n")

    report["human_decision"] = choices[choice]
    save_report(report)

    return redirect(url_for("index"))

@app.route("/analyze-amen", methods=["POST"])
def analyze_amen():
    try:
        raw_json = request.form.get("diagnosis_json", "")
        amen_result = json.loads(raw_json)

        incident = adapt_amen_diagnosis(amen_result)

        agent = CorrectionAgentV3(
            mode="llm",
            out_dir="artifacts_v3",
        )
        result = agent.run(incident)

        report = {
            "incident": incident,
            "amen_diagnosis": amen_result,
            "proposed_action": result["proposed_plan"]["action"],
            "final_action": result["plan"]["action"],
            "guard": result["guard_result"],
            "tool_result": result["tool_result"],
            "verification": result["verification"],
            "human_decision": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        save_report(report)
        return redirect(url_for("index"))

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"Diagnostic invalide : {exc}", 400
    except Exception:
        app.logger.exception("Erreur pendant l'integration")
        return "Erreur interne pendant l'analyse.", 500



@app.route("/run-amen", methods=["POST"])
def run_amen():
    try:
        episode = request.form.get(
            "episode", "episode-003"
        )

        amen_result = run_amen_investigation(
            episode=episode,
            provider="nvidia",
        )

        incident = adapt_amen_diagnosis(amen_result)

        agent = CorrectionAgentV3(
            mode="llm",
            out_dir="artifacts_v3",
        )

        result = agent.run(incident)

        report = {
            "incident": incident,
            "amen_diagnosis": amen_result,
            "proposed_action": result["proposed_plan"]["action"],
            "final_action": result["plan"]["action"],
            "guard": result["guard_result"],
            "tool_result": result["tool_result"],
            "verification": result["verification"],
            "human_decision": None,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_report(report)
        return redirect(url_for("index"))

    except Exception:
        app.logger.exception("Echec de l'investigation")
        return "Erreur pendant l'investigation Amen.", 500



if __name__ == "__main__":
    app.run(debug=True)
