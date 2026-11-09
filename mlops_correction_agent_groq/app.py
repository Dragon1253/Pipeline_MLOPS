
import json
from datetime import datetime, timezone
from pathlib import Path
from integration_adapter import adapt_amen_diagnosis
from amen_runner import run_amen_investigation

from flask import Flask, render_template, request, redirect, url_for, session
from agent_v3 import CorrectionAgentV3
from human_review_store import (
    HumanReviewStore,
    InvalidReviewToken,
    ReviewAlreadyResolved,
    ReviewNotFound,
    review_database_path,
)

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


def get_review_store():
    return HumanReviewStore(review_database_path())


def update_pipeline_result(review):
    result_path = Path(review["pipeline_result_file"])
    if not result_path.is_file():
        return

    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    review_result = result.setdefault("human_review", {})
    review_result["decision"] = review["decision"]
    review_result["decided_at"] = review["decided_at"]
    review_result["status"] = review["status"]
    review_result["execution"] = review["execution"]

    if review["status"] == "completed":
        result["status"] = "completed_after_human_approval"
        execution = review["execution"]
        result["correction"] = {
            "incident": review["incident"],
            "proposed_plan": execution["plan"],
            "final_plan": execution["plan"],
            "guard": execution["guard_result"],
            "tool_result": execution["tool_result"],
            "verification": execution["verification"],
            "human_approval_id": execution["approval_id"],
        }
    elif review["status"] == "rejected":
        result["status"] = "rejected_by_human"
    else:
        result["status"] = "human_review_blocked"

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


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


@app.route("/human-review/<review_id>", methods=["GET"])
def human_review_page(review_id):
    token = request.args.get("token", "")
    try:
        review = get_review_store().get_review(review_id, token)
    except ReviewNotFound:
        return "Demande de revue introuvable.", 404
    except InvalidReviewToken:
        return "Lien de revue invalide.", 403

    return render_template("human_review.html", review=review, token=token)


@app.route("/human-review/<review_id>/decision", methods=["POST"])
def human_review_decision(review_id):
    token = request.form.get("token", "")
    decision_value = request.form.get("decision", "")
    store = get_review_store()
    try:
        review = store.resolve(review_id, token, decision_value)
    except ReviewNotFound:
        return "Demande de revue introuvable.", 404
    except InvalidReviewToken:
        return "Lien de revue invalide.", 403
    except ReviewAlreadyResolved:
        return "Cette demande a deja ete traitee.", 409
    except ValueError:
        return "Decision invalide.", 400

    if decision_value == "approved":
        if not store.claim_execution(review_id):
            return "Cette demande est deja en cours ou terminee.", 409
        try:
            CorrectionAgentV3.human_approved_action(review["incident"])
            agent = CorrectionAgentV3(
                mode="rules",
                out_dir=Path(review["run_dir"]) / "human_approved_correction",
            )
            execution = agent.run_human_approved(
                review["incident"], approval_id=review_id
            )
            execution_status = (
                "completed" if execution["verification"]["safe"] else "blocked"
            )
            store.complete_execution(
                review_id, status=execution_status, result=execution
            )
        except ValueError as exc:
            execution = {"error": str(exc), "production_modified": False}
            store.complete_execution(review_id, status="blocked", result=execution)
        except Exception:
            app.logger.exception("Echec de l'action approuvee")
            execution = {
                "error": "Erreur pendant l'action simulee.",
                "production_modified": False,
            }
            store.complete_execution(review_id, status="failed", result=execution)
    else:
        execution = None

    try:
        review = store.get_review(review_id, token)
        update_pipeline_result(review)
    except (OSError, ValueError, KeyError):
        app.logger.exception("Impossible de mettre a jour le resultat du pipeline")

    return redirect(url_for(
        "human_review_page", review_id=review_id, token=token
    ))


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
    app.run(host="127.0.0.1", port=5000, debug=False)
