"""Sentinel (interface d'Amen) branché sur le flux complet du projet.

    python sentinel_app.py                 puis ouvrir http://127.0.0.1:8000
    python sentinel_app.py --orchestrator rules --no-predict --port 8001

Aucun fichier du groupe n'est modifié. Ce script :

* charge le monitoring à jour (``monitoring/``) et le pipeline du groupe via ``run_full_pipeline`` ;
* importe l'API Sentinel (``platform_api/app.py``) et son interface compilée (``frontend/dist``) ;
* remplace uniquement sa fonction ``run_incident``, qui rejouait des épisodes figés et affichait une
  correction de démonstration, par le vrai flux : monitoring en direct (alertes confirmées et
  prédictives) → agent d'analyse Nemotron → score de confiance → orchestrateur → agent de correction
  ou revue humaine (``pipeline_v4.run_pipeline``, inchangé).

Après la décision (automatique, ou approbation humaine notifiée dans la page, sans e-mail), l'action
choisie par l'agent de correction est appliquée dans le système simulé (``monitoring/remediation.py``) et
le monitoring vérifie sur les fenêtres suivantes que l'erreur repasse sous le seuil.

C'est l'évolution prévue dans PLATFORM.md (« Le futur agent de correction remplacera la section de
simulation de run_incident … Un connecteur de monitoring pourra ensuite alimenter le même stockage »).
Les données Sentinel de ce mode sont rangées à part, dans ``monitoring_runs/sentinel``.
"""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
# Base SQLite séparée de celle d'Amen (platform_data/), à fixer avant d'importer l'API.
os.environ.setdefault("SENTINEL_DATA_DIR", str(ROOT / "monitoring_runs" / "sentinel"))

# L'API Sentinel de la racine est importée en premier : pipeline_v4 ajoute ensuite à sys.path la
# copie amen_project/, qui contient une autre version de platform_api.
sys.path.insert(0, str(ROOT))
import platform_api.app as sentinel  # noqa: E402  (API d'Amen, inchangée)
from fastapi import Request  # noqa: E402
import run_full_pipeline  # noqa: E402,F401  (monitoring de la racine + pipeline du groupe)
import monitoring.service as monitoring_service  # noqa: E402
import pipeline_v4  # noqa: E402
import human_notifications  # noqa: E402  (e-mail de revue humaine, code du groupe)

assert Path(sentinel.__file__).resolve() == ROOT / "platform_api" / "app.py", "Sentinel doit venir de la racine."

SETTINGS = {"predict": True, "label_delay": 2, "orchestrator_mode": "rules", "speed": 0.6, "always_approve": False}
# Sentinel appelle « false_alert » le scénario que le monitoring appelle « normal ».
STREAM_SCENARIO = {"false_alert": "normal"}
ACTIONS = {
    "quarantine_batch": "Mise en quarantaine du lot",
    "prepare_retraining": "Préparation d'un réentraînement",
    "prepare_rollback": "Préparation du retour à la version de référence",
    "close_alert": "Clôture de l'alerte",
    "human_review": "Revue humaine",
}


# Le lien de la page de revue (avec son jeton) n'existe que dans l'e-mail. On le garde aussi
# pour l'afficher dans Sentinel, que l'e-mail soit configuré ou non.
REVIEW_LINK = {}
_send_review_email = human_notifications.send_human_review_email


def _send_and_keep_link(review):
    """Pas d'e-mail : la demande est notifiée dans Sentinel (bandeau et panneau de revue)."""
    REVIEW_LINK["url"] = review.get("review_url")
    return {"status": "in_app"}


human_notifications.send_human_review_email = _send_and_keep_link


def planned_correction(incident_type):
    from monitoring.remediation import planned_correction as planned
    return planned(incident_type)


def _effect(action):
    from monitoring.remediation import EFFECTS
    return EFFECTS.get(action, "")


_score_diagnosis = pipeline_v4.score_amen_diagnosis


def _score_with_policy(result, monitoring_alerts=()):
    """Score du groupe inchangé ; avec --approbation-toujours, toute correction exige un humain."""
    assessment = _score_diagnosis(result, monitoring_alerts)
    if SETTINGS["always_approve"]:
        assessment = {**assessment, "requires_human_review": True,
                      "policy": "approbation humaine exigée pour toute correction (--approbation-toujours)"}
    return assessment


pipeline_v4.score_amen_diagnosis = _score_with_policy


def _pipeline(incident_id, scenario, loop):
    """Exécuté dans un thread : le pipeline du groupe, avec des crochets d'affichage."""

    def update(**changes):
        # Les mises à jour de l'incident passent par la boucle de Sentinel (comme dans son code).
        asyncio.run_coroutine_threadsafe(_update(incident_id, changes), loop).result()

    def emit(kind, payload):
        sentinel.emit(incident_id, kind, payload)

    def on_monitoring(event):
        kind = event["event"]
        if kind == "window_scored":
            time.sleep(SETTINGS["speed"])
        elif kind == "metric_recorded":
            sentinel.add_metric(incident_id, event["metrics"], event["window"], event["labeled_rows"], source="monitoring")
        elif kind == "predictive_alert":
            a = event["alert"]
            signals = ", ".join(s["signal"] for s in a.get("top_signals", []))
            emit("Alerte prédictive", {"message": f"{a['window_id']} · risque {a['risk']:.2f} au pas {a['detected_at_step']}, "
                                                 f"avant l'arrivée des labels · {signals}"})
        elif kind == "alert_triggered":
            a = event["alert"]
            emit("Alerte confirmée", {"message": f"{a['window_id']} · {a.get('severity')} · log loss "
                                                 f"{a['observed_value']:.3f} (seuil {a['threshold']:.3f}) au pas {a.get('detected_at_step')}"})
            update(alert=a)
        elif kind == "incident_closed":
            emit("Incident clos par le monitoring", {"message": f"{event.get('window')} · {event.get('reason')}"})
        elif kind == "episode_packaged":
            emit("Dossier d'alerte écrit", {"message": Path(event["episode_dir"]).name})

    predictor = None
    if SETTINGS["predict"]:
        from monitoring.predictor import load_predictor
        predictor = load_predictor()

    def stream(*args, **kwargs):
        emit("Monitoring en direct", {"message": "Fenêtres de 500 clients générées et évaluées une par une."})
        return monitoring_service.run_stream(*args, predictor=predictor, label_delay=SETTINGS["label_delay"],
                                             on_event=on_monitoring, **kwargs)

    def analysis(**kwargs):
        import json
        from mlops_churn.agent import investigate
        episode = Path(kwargs["episode_dir"])
        alert = json.loads((episode / "alert.json").read_text(encoding="utf-8"))
        context = json.loads((episode / "context.json").read_text(encoding="utf-8"))
        update(alert=alert, deployment=context["deployment"], status="investigating", phase=1)
        emit("alert_triggered", alert)

        def on_event(event):
            payload = {k: v for k, v in event.items() if k not in ("event", "result")} \
                if event["event"] == "investigation_completed" else {k: v for k, v in event.items() if k != "event"}
            emit(event["event"], payload)
        result = investigate(on_event=on_event, **kwargs)
        update(agent_result=result)
        if result.get("status") == "completed":
            update(phase=2, status="diagnosed", diagnosis=result["final_diagnosis"])
            emit("diagnosis_ready", result["final_diagnosis"])
        return result

    pipeline_v4.run_stream = stream
    REVIEW_LINK.clear()
    return pipeline_v4.run_pipeline(scenario=scenario, seed=42, analysis_mode="nvidia",
                                    orchestrator_mode=SETTINGS["orchestrator_mode"], correction_mode="rules",
                                    output_dir=str(ROOT / "monitoring_runs" / "sentinel"), analysis_runner=analysis)


def _redact(text):
    """Masque les clés éventuellement présentes dans un message d'erreur."""
    for name in ("NVIDIA_API_KEY", "GROQ_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[CLÉ MASQUÉE]")
    return text


async def _update(incident_id, changes):
    sentinel.update(incident_id, **changes)


def _flow(result):
    """Ce que chaque étape a produit, conservé dans l'incident pour le rapport."""
    import json
    monitoring = dict(result.get("monitoring") or {})
    summary_file = Path(monitoring.get("run_dir", "")) / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    analysis = result.get("analysis") or {}
    return {
        "settings": dict(SETTINGS),
        "monitoring": {
            "run_id": monitoring.get("run_id"), "threshold": summary.get("threshold"),
            "critical_threshold": summary.get("critical_threshold"), "label_delay": summary.get("label_delay"),
            "n_windows": len(summary.get("timeline", [])), "alerts": monitoring.get("alerts", []),
            "predictive_alerts": summary.get("predictive_alerts", []),
            "incidents": summary.get("incidents", []),
            "episode": Path(monitoring["episode_dir"]).name if monitoring.get("episode_dir") else None,
        },
        "confidence": analysis.get("confidence_assessment"),
        "orchestration": result.get("orchestration"),
        "correction": {k: v for k, v in (result.get("correction") or {}).items() if k != "incident"} or None,
        "human_review": result.get("human_review"),
        "status": result.get("status"),
        "result_file": result.get("result_file"),
    }


def _email_status(notification):
    notification = notification or {}
    status = notification.get("status")
    if status == "sent":
        return f"envoyé à {notification.get('recipient')}"
    if status == "not_configured":
        return "non configuré (SMTP absent du .env)"
    if status == "failed":
        return f"échec ({notification.get('error_type')} {notification.get('smtp_code') or ''})".strip()
    if status == "in_app":
        return "panneau « Revue humaine » de Sentinel (pas d'e-mail)"
    return str(status or "—")


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pairs(data):
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in (data or {}).items() if not isinstance(v, (dict, list)))


def build_report(item, metrics):
    """Rapport Markdown du flux complet : une section par étape, avec les valeurs réellement produites."""
    flow = item.get("flow") or {}
    mon, settings = flow.get("monitoring") or {}, flow.get("settings") or {}
    diagnosis, events = item.get("diagnosis") or {}, item.get("events", [])
    duration = "—"
    if item.get("ended_at"):
        from datetime import datetime
        seconds = (datetime.fromisoformat(item["ended_at"]) - datetime.fromisoformat(item["created_at"])).total_seconds()
        duration = f"{seconds:.0f} s"
    lines = [
        "# Rapport Sentinel — flux complet", "",
        f"* Incident : {item['id']}", f"* Scénario simulé : {item['title']}",
        f"* Début : {item['created_at']} · durée : {duration}",
        f"* Résultat du parcours : **{flow.get('status', item.get('outcome'))}**",
        f"* Données : flux synthétique généré en direct ({mon.get('n_windows', '—')} fenêtres de 500 clients), "
        f"labels reçus avec {mon.get('label_delay', '—')} fenêtre(s) de retard. Aucune donnée de production.",
        "", "Chaîne : monitoring → agent d'analyse → score de confiance → orchestrateur → agent de correction ou revue humaine.",
        "", "## 1. Monitoring (`monitoring/`)", "",
        f"Seuil d'alerte (warning) : {_fmt(mon.get('threshold'), 4)} · seuil critique : {_fmt(mon.get('critical_threshold'), 4)} · "
        f"alerte prédictive : {'activée' if settings.get('predict') else 'désactivée'}.", "",
        "### Log loss par fenêtre (à l'arrivée des labels)", "", "| Fenêtre | Log loss | ROC AUC | Clients étiquetés |", "| --- | --- | --- | --- |",
        *[f"| {m['window']} | {_fmt(m['metrics'].get('log_loss'))} | {_fmt(m['metrics'].get('roc_auc'))} | {m['labeled_rows']} |" for m in metrics],
        "", "### Alertes prédictives (sans labels)", "",
    ]
    predictive = mon.get("predictive_alerts") or []
    if predictive:
        lines += ["| Fenêtre | Pas | Risque | Seuil | Statut | Avance | Signaux principaux |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for a in predictive:
            signals = ", ".join(f"{s['signal']} = {_fmt(s.get('value'))} (habituel {_fmt(s.get('usual_value'))})" for s in a.get("top_signals", []))
            lead = f"{a['lead_steps']} pas" if a.get("lead_steps") is not None else "—"
            lines.append(f"| {a['window_id']} | {a.get('detected_at_step')} | {_fmt(a.get('risk'))} | {_fmt(a.get('threshold'))} | "
                         f"{a.get('status')} | {lead} | {signals} |")
    else:
        lines.append("Aucune alerte prédictive." + ("" if settings.get("predict") else " (prédiction désactivée)"))
    lines += ["", "### Alertes confirmées (avec labels)", ""]
    alerts = mon.get("alerts") or []
    if alerts:
        lines += ["| Fenêtre | Sévérité | Log loss | Pas de détection |", "| --- | --- | --- | --- |",
                  *[f"| {a['window_id']} | {a.get('severity')} | {_fmt(a.get('observed_value'), 4)} | {a.get('detected_at_step')} |" for a in alerts]]
        closed = [f"{i['focus']} ({i['close_reason']})" for i in mon.get("incidents", []) if i.get("close_reason")]
        if closed:
            lines.append(f"\nIncidents clos par le monitoring : {', '.join(closed)}.")
        lines.append(f"\nDossier d'alerte transmis à l'analyse : `{mon.get('episode')}`.")
    else:
        lines.append("Aucune alerte confirmée : rien n'a été transmis à l'analyse.")

    tools = [e for e in events if e["kind"] == "tool_completed"]
    calls = len([e for e in events if e["kind"] == "model_started"])
    lines += ["", "## 2. Agent d'analyse (`mlops_churn/agent.py`, Nemotron via NVIDIA)", ""]
    if diagnosis:
        lines += [f"* Cause probable : **{diagnosis.get('cause_probable')}**", f"* Résumé : {diagnosis.get('resume', '')}",
                  f"* Preuves citées : {', '.join(map(str, diagnosis.get('preuves', [])))}",
                  f"* {calls} appels au modèle, {len(tools)} outils exécutés :", "",
                  "| Preuve | Outil | Statut | Durée |", "| --- | --- | --- | --- |",
                  *[f"| {e['payload'].get('evidence_id')} | {e['payload'].get('tool')} | {(e['payload'].get('result') or {}).get('status')} | "
                    f"{_fmt(e['payload'].get('duration_seconds'), 2)} s |" for e in tools]]
    else:
        lines.append("Pas de diagnostic." + (f" Erreur : {item['error']}" if item.get("error") else ""))

    confidence = flow.get("confidence")
    lines += ["", "## 3. Score de confiance (`integration_adapter.py`)", ""]
    if confidence:
        lines += [f"* Confiance : **{confidence.get('confidence')}** (seuil de correction automatique : 0,7)",
                  f"* Type d'incident : {confidence.get('incident_type')} · preuve qui l'appuie : {confidence.get('supporting_tool') or 'aucune'}",
                  f"* Diagnostic appuyé par une preuve cohérente : {confidence.get('supported')} · revue humaine exigée : {confidence.get('requires_human_review')}",
                  f"* Méthode : {confidence.get('method', '')}"]
    else:
        lines.append("Non calculé.")

    orchestration = flow.get("orchestration")
    lines += ["", "## 4. Orchestrateur (`orchestrator_v4.py`, LangGraph)", ""]
    if orchestration:
        proposed, decision, guard = orchestration.get("proposed_decision") or {}, orchestration.get("decision") or {}, orchestration.get("guard") or {}
        lines += [f"* Mode : {orchestration.get('mode')} ({'Groq propose la route' if orchestration.get('mode') == 'llm' else 'règles fixes'})",
                  f"* Route proposée : {proposed.get('route')}", f"* Garde-fou : {_pairs(guard)}",
                  f"* **Route finale : {decision.get('route')}**", f"* Raison : {decision.get('reason', '')}"]
    else:
        lines.append("Non exécuté.")

    correction, review = flow.get("correction"), flow.get("human_review")
    section = 5
    if review:
        execution = review.get("execution") or {}
        lines += ["", f"## {section}. Revue humaine (`human_review_store.py`, `app.py`)", "",
                  f"* Demande : {review.get('review_id')} · statut : **{review.get('status')}**",
                  f"* Raison : {review.get('reason', '')}",
                  f"* Correction proposée à l'humain : {review.get('planned_correction') or '—'}",
                  f"* Notification : {_email_status(review.get('notification'))}",
                  f"* Décision humaine : {review.get('decision') or 'en attente'}" + (f" ({review['decided_at']})" if review.get("decided_at") else ""),
                  *([f"* Exécution bloquée : {execution['error']}"] if execution.get("error") else []),
                  "* Si l'humain approuve, la seule action correspondant au type d'incident est lancée, puis appliquée et vérifiée (section suivante)."]
        section += 1
    if correction:
        plan = correction.get("final_plan") or {}
        origin = " après validation humaine" if correction.get("human_approval_id") else ""
        lines += ["", f"## {section}. Décision de l'agent de correction{origin} (`agent_v3.py` : plan → garde-fou → exécution → vérification)", "",
                  f"* Action proposée : {(correction.get('proposed_plan') or {}).get('action')}",
                  f"* Garde-fou : {_pairs(correction.get('guard'))}",
                  f"* **Action retenue : {plan.get('action')}** — {plan.get('reason', '')}",
                  f"* Enregistrement de l'agent (avant application) : {_pairs(correction.get('tool_result'))}",
                  f"* Contrôle de sûreté de l'agent : {_pairs(correction.get('verification'))}"]
        section += 1
    attempts = flow.get("remediation_attempts") or ([flow["remediation"]] if flow.get("remediation") else [])
    if attempts:
        lines += ["", f"## {section}. Correction appliquée et vérification par le monitoring (`monitoring/remediation.py`)", ""]
        for k, attempt in enumerate(attempts, 1):
            details = attempt.get("details") or {}
            origin = "solution proposée par l'agent, approuvée par un humain" if attempt.get("origin") == "proposal" else "action de l'agent de correction"
            lines += [f"### Tentative {k} : {attempt.get('label', attempt['action'])} (`{attempt['action']}`, {origin})", "",
                      f"* Effet : {attempt['effect']}",
                      *([f"* Détails : {_pairs(details)}"] if _pairs(details) else []),
                      f"* Erreur moyenne : {_fmt(attempt['before_mean'])} avant → **{_fmt(attempt['after_mean'])}** après "
                      f"(seuil {_fmt(attempt['threshold'], 4)})",
                      f"* Résultat : **{'corrigé' if attempt['passed'] else 'pas corrigé'}**", "",
                      "| Lot après correction | Log loss | ROC AUC |", "| --- | --- | --- |",
                      *[f"| {w['window']} | {_fmt(w['log_loss'])} | {_fmt(w.get('roc_auc'))} |" for w in attempt.get("windows", [])], ""]
        last = attempts[-1]
        if not last["passed"] and last.get("proposals"):
            proposals = last["proposals"]
            lines += ["### Solutions proposées (en attente d'une décision humaine)", "",
                      *[f"* **{o['label']}** — erreur estimée {_fmt(o['expected_log_loss'])} sur {o['tested_on']}"
                        f"{' (corrigerait)' if o['would_fix'] else ''} : {o['why']}" for o in proposals["automatic"]],
                      *[f"* {o['label']} (action humaine) : {o['why']}" for o in proposals["manual"]]]
        section += 1
    if not review and not correction:
        lines += ["", f"## {section}. Correction", "", "Aucune action : pas d'alerte confirmée ou analyse incomplète."]
        section += 1

    lines += ["", f"## {section}. Limites", "",
              "* La correction est appliquée dans le système simulé (données et modèle de la simulation), pas sur un "
              "système de production réel.",
              "* La vérification porte sur 6 fenêtres après la correction ; le monde simulé continue avec l'incident.",
              "* Le score de confiance est une heuristique de corroboration, pas une probabilité calibrée.",
              "", "## Journal", "",
              *[f"- {e['at'][11:19]} — {e['kind']}" + (f" : {e['payload']['message']}" if e["payload"].get("message") else "")
                for e in events if e["kind"] not in ("metric_recorded", "tool_started", "replay_started")],
              "", f"Résultat complet : `{flow.get('result_file')}`"]
    return "\n".join(lines) + "\n"


def report(incident_id: str):
    from fastapi import HTTPException
    from fastapi.responses import Response
    from sqlalchemy import text
    item = sentinel.detail(incident_id)
    if item["status"] != "completed":
        raise HTTPException(409, "Le rapport final sera disponible en fin de parcours.")
    import json
    with sentinel.engine.connect() as conn:
        rows = conn.execute(text("SELECT body FROM metrics WHERE incident_id=:id ORDER BY id"), {"id": incident_id}).all()
    metrics = [json.loads(r[0]) for r in rows]
    return Response(build_report(item, metrics), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="sentinel-{incident_id}.md"'})


async def run_incident(incident_id):
    record = sentinel.get_incident(incident_id)
    scenario = STREAM_SCENARIO.get(record["scenario"], record["scenario"])
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _pipeline, incident_id, scenario, loop)
        status = result["status"]
        if result.get("human_review") and REVIEW_LINK.get("url"):
            result["human_review"]["review_url"] = REVIEW_LINK["url"]
        if result.get("human_review"):
            action_, text_ = planned_correction(((result["human_review"].get("request") or {}).get("incident_type")))
            result["human_review"]["planned_action"], result["human_review"]["planned_correction"] = action_, text_
        sentinel.update(incident_id, flow=_flow(result))
        if status == "no_alert":
            sentinel.update(incident_id, phase=5, status="completed", ended_at=sentinel.now(), outcome="no_alert",
                            correction=dict(simulated=True, action="Aucune alerte confirmée", applied=False,
                                            message="Le monitoring n'a confirmé aucune dégradation : rien n'est transmis à l'analyse."))
            sentinel.emit(incident_id, "report_ready", {"message": "Aucune alerte confirmée."})
            return
        if status in ("analysis_incomplete", "analysis_invalid", "no_episode"):
            raise RuntimeError(result.get("error") or "L'agent d'analyse n'a pas rendu de diagnostic exploitable.")

        assessment = result["analysis"].get("confidence_assessment") or {}
        sentinel.emit(incident_id, "Score de confiance", {"message": (
            f"{assessment.get('confidence')} · type {assessment.get('incident_type')} · "
            f"preuve {assessment.get('supporting_tool') or 'aucune'} (seuil de correction automatique : 0,7)")})
        decision = result["orchestration"]["decision"]
        sentinel.emit(incident_id, "Décision de l'orchestrateur", {"message": (
            f"{decision.get('route')} ({result['orchestration']['mode']}) · {decision.get('reason', '')}")})
        await asyncio.sleep(1)

        if result.get("correction"):
            c = result["correction"]
            plan = c["final_plan"]
            action = plan.get("action")
            correction = dict(simulated=False, applied=True, action=f"{ACTIONS.get(action, action)} ({action})",
                              message=(f"Correction automatique : confiance {assessment.get('confidence')}, route décidée par "
                                       f"l'orchestrateur : {decision.get('route')}. Agent de correction : garde-fou "
                                       f"{'accepté' if c['guard'].get('accepted') else 'refusé'}, {plan.get('reason') or ''}. "
                                       f"Effet : {_effect(action)}").strip())
            outcome = "corrected"
        else:
            review = result.get("human_review") or {}
            planned_action, planned_text = planned_correction((review.get("request") or {}).get("incident_type"))
            correction = dict(simulated=False, applied=False, action="Approbation humaine requise",
                              message=(f"{decision.get('reason', '')} Si vous approuvez, l'agent de correction appliquera : "
                                       f"{planned_text}"))
            validation = dict(simulated=True, passed=None, before=None, after=None,
                              message="Approuvez ou refusez dans le panneau « Revue humaine » ci-dessous.")
            # L'agent de correction n'est PAS exécuté : le parcours s'arrête à l'étape Correction et attend
            # la décision humaine (phase 3). Le statut « completed » libère seulement Sentinel pour un autre
            # scénario ; le rapport est provisoire jusqu'à la décision.
            sentinel.update(incident_id, phase=3, status="completed", ended_at=sentinel.now(),
                            outcome="awaiting_human_decision", correction=correction, validation=validation,
                            pipeline_result=result.get("result_file"))
            sentinel.emit(incident_id, "Approbation demandée", {"message": f"Correction proposée : {planned_text}"})
            sentinel.emit(incident_id, "En attente de la décision humaine", {"message": (
                "L'agent de correction n'a rien appliqué. Il agira seulement si un humain approuve.")})
            return
        sentinel.update(incident_id, phase=3, status="correcting", correction=correction)
        sentinel.emit(incident_id, "correction_started", correction)
        remediation = await loop.run_in_executor(None, _remediate, incident_id, action)
        if not remediation["passed"]:
            outcome = "correction_insufficient"
        sentinel.update(incident_id, phase=5, status="completed", ended_at=sentinel.now(), outcome=outcome,
                        pipeline_result=result.get("result_file"))
        sentinel.emit(incident_id, "report_ready", {"message": f"Parcours terminé : {outcome}."})
    except asyncio.CancelledError:
        sentinel.update(incident_id, status="interrupted", ended_at=sentinel.now(), error="Serveur arrêté pendant le traitement.")
        raise
    except Exception as exc:
        message = _redact(f"{type(exc).__name__} : {exc}")
        sentinel.update(incident_id, status="failed", error=message, ended_at=sentinel.now())
        sentinel.emit(incident_id, "failed", {"message": message})


_REFERENCE = {}


def _remediate(incident_id, action, *, new_threshold=None, origin="agent"):
    """Applique une action dans la simulation et vérifie avec le monitoring.

    Exécuté dans un thread : réentraînement éventuel, puis 6 fenêtres mesurées après la correction,
    ajoutées au graphique de l'incident. Si la vérification échoue, les autres solutions sont testées
    sur le dernier lot étiqueté et proposées à l'humain. Chaque tentative est conservée.
    """
    from monitoring.remediation import apply_correction, propose_solutions
    from monitoring.service import ReferenceContext
    from monitoring.stream import StreamSpec
    record = sentinel.get_incident(incident_id)
    scenario = STREAM_SCENARIO.get(record["scenario"], record["scenario"])
    if "ctx" not in _REFERENCE:
        _REFERENCE["ctx"] = ReferenceContext()
    reference = _REFERENCE["ctx"]
    threshold, reference_value = reference.calibration["threshold"], reference.calibration["reference_value"]
    flow = record.get("flow") or {}
    attempts = flow.get("remediation_attempts", [])
    spec = StreamSpec(scenario, 42)
    start = spec.n_windows + 6 * len(attempts)          # les tentatives se suivent dans le temps
    label = (ACTIONS | {"recalibrate_threshold": "Adaptation du seuil d'alerte"}).get(action, action)
    who = "proposée par l'agent et approuvée par un humain" if origin == "proposal" else "choisie par l'agent de correction"
    sentinel.emit(incident_id, f"Tentative {len(attempts) + 1} : correction appliquée",
                  {"message": f"{label} ({action}), {who}."})
    result = apply_correction(action, spec, reference.bundle, threshold, start=start, new_threshold=new_threshold)
    details = result["details"]
    if action == "prepare_retraining":
        sentinel.emit(incident_id, "Réentraînement", {"message": (
            f"Nouveau modèle entraîné sur {details['training_rows']} clients étiquetés ; sur {details['comparison_window']} : "
            f"log loss {details['candidate_log_loss']} (nouveau) contre {details['current_log_loss']} (actuel) → "
            f"{'mis en service' if details['candidate_deployed'] else 'non déployé, le modèle actuel est conservé'}.")})
    sentinel.update(incident_id, phase=4, status="validating")
    for window in result["windows"]:
        time.sleep(SETTINGS["speed"])
        sentinel.add_metric(incident_id, window["metrics"], window["window"], window["labeled_rows"], source="after_correction")
    verdict = "réussie" if result["passed"] else "échouée"
    message = (f"Vérification par le monitoring {verdict} : log loss moyenne {result['before_mean']:.3f} avant la correction, "
               f"{result['after_mean']:.3f} sur les {len(result['windows'])} fenêtres suivantes (seuil {result['threshold']:.3f}, "
               f"{result['windows_above_threshold']} fenêtre(s) au-dessus).")
    summary = {k: v for k, v in result.items() if k != "windows"} | {
        "origin": origin, "attempt": len(attempts) + 1,
        "windows": [{"window": w["window"], "log_loss": w["metrics"]["log_loss"], "roc_auc": w["metrics"].get("roc_auc")}
                    for w in result["windows"]]}
    if not result["passed"]:
        tried = [a["action"] for a in attempts] + [action]
        summary["proposals"] = propose_solutions(spec, reference.bundle, threshold, reference_value, tried=tried)
        n = len(summary["proposals"]["automatic"])
        message += (f" La correction n'a pas suffi : {n} solution(s) testée(s) et proposée(s) à l'humain."
                    if n else " La correction n'a pas suffi et aucune autre action automatique ne corrige : décision humaine nécessaire.")
    flow = sentinel.get_incident(incident_id).get("flow") or {}
    flow["remediation_attempts"] = attempts + [summary]
    flow["remediation"] = summary
    validation = dict(simulated=False, passed=result["passed"], before=result["before_mean"], after=result["after_mean"], message=message)
    sentinel.update(incident_id, flow=flow, validation=validation)
    sentinel.emit(incident_id, "Vérification par le monitoring", {"message": message})
    if not result["passed"] and summary["proposals"]["automatic"]:
        sentinel.emit(incident_id, "Solutions proposées", {"message": " ; ".join(
            f"{o['label']} (erreur estimée {o['expected_log_loss']:.3f}{', corrigerait' if o['would_fix'] else ''})"
            for o in summary["proposals"]["automatic"])})
    return result


async def apply_solution(incident_id: str, request: Request):
    """L'humain choisit une des solutions proposées après un échec : elle est appliquée puis vérifiée."""
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    body = await request.json()
    record = sentinel.get_incident(incident_id)
    remediation = (record.get("flow") or {}).get("remediation") or {}
    options = {o["action"]: o for o in (remediation.get("proposals") or {}).get("automatic", [])}
    choice = options.get(body.get("action"))
    if remediation.get("passed") or choice is None:
        raise HTTPException(409, "Cette solution n'est pas proposée pour cet incident.")
    if record["status"] not in ("completed", "failed", "interrupted"):
        raise HTTPException(409, "Une correction est déjà en cours.")
    sentinel.update(incident_id, phase=3, status="correcting")
    sentinel.emit(incident_id, "Solution approuvée par un humain", {"message": f"{choice['label']} ({choice['action']})."})
    result = await run_in_threadpool(_remediate, incident_id, choice["action"],
                                     new_threshold=choice.get("threshold"), origin="proposal")
    outcome = "corrected_after_proposal" if result["passed"] else "correction_insufficient"
    sentinel.update(incident_id, phase=5, status="completed", ended_at=sentinel.now(), outcome=outcome)
    sentinel.emit(incident_id, "report_ready", {"message": f"Rapport mis à jour : {outcome}."})
    return {"passed": result["passed"], "after_mean": result["after_mean"], "threshold": result["threshold"]}


# Seul point d'accroche : launch() de Sentinel appelle run_incident par son nom.
sentinel.run_incident = run_incident
app = sentinel.app
# Rapport : même adresse que celui de Sentinel, contenu complété avec toutes les étapes.
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/api/incidents/{incident_id}/report"]
app.add_api_route("/api/incidents/{incident_id}/report", report, methods=["GET"])


# ---------------------------------------------------------------------------------------------
# Revue humaine dans la même page : la logique de décision reste celle de app.py (inchangé),
# appelée à travers son client de test Flask, puis l'incident Sentinel est mis à jour.
_REVIEW = {}
REVIEW_JS = r"""// Ajouts à la page Sentinel (sans modifier son code React) :
//  * panneau « Revue humaine » dans la page de l'incident ;
//  * textes de validation mis à jour quand la correction a été réellement appliquée et vérifiée.
(() => {
  const ACTIONS = {quarantine_batch: 'Correction des entrées et quarantaine', prepare_retraining: 'Réentraînement du modèle',
    prepare_rollback: 'Retour à la version de référence', close_alert: 'Clôture de l’alerte'};
  const STATUS = {pending: ['En attente de décision', 'orange'], approved: ['Approuvée', 'blue'], executing: ['Exécution…', 'blue'],
    completed: ['Approuvée · correction appliquée', 'green'], rejected: ['Refusée', 'red'], blocked: ['Approuvée · action bloquée', 'red'], failed: ['Échec', 'red']};
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const num = v => v == null ? '—' : Number(v).toFixed(3);
  let busy = false, message = '', state = null;

  async function decide(r, decision, buttons) {
    buttons.forEach(x => x.disabled = true);
    const token = new URL(r.review_url).searchParams.get('token');
    message = '';
    const res = await fetch(`/human-review/${r.review_id}/decision`, {method: 'POST', redirect: 'manual',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: new URLSearchParams({token, decision})});
    if (res.status >= 400) message = (await res.text()) || 'Décision refusée.';
    busy = false; refresh();
  }



  function remediationHtml(i) {
    const m = i.flow && i.flow.remediation;
    if (!m) return '';
    return `<div class="validation" style="margin-top:12px"><div><strong>${m.passed ? 'Correction réussie' : 'Correction insuffisante'}</strong>
      <p>Log loss moyenne : ${num(m.before_mean)} avant → ${num(m.after_mean)} après (seuil ${num(m.threshold)}).</p></div></div>`;
  }

  function panelHtml(i, r) {
    const req = r.request || {}, [label, tone] = STATUS[r.status] || [r.status, 'neutral'];
    let body = `<dl class="hr-grid">
      <dt>Raison</dt><dd>${esc(r.reason)}</dd>
      <dt>Type d’incident</dt><dd>${esc(req.incident_type)} · cause proposée ${esc(req.cause_probable)} · confiance ${esc(req.confidence)}</dd>
      <dt>Correction prévue</dt><dd><strong>${esc(ACTIONS[r.planned_action] || 'Aucune')}</strong><br>${esc(r.planned_correction)}</dd></dl>`;
    if (r.status === 'pending') {
      body += `<div class="hr-actions"><button class="button primary" data-decision="approved">Approuver la correction</button>
        <button class="button outline" data-decision="rejected">Refuser</button></div>
        <p class="muted hr-note">Rien n’est appliqué sans votre approbation. La décision est enregistrée une seule fois.</p>`;
    } else {
      const ex = r.execution || {};
      body += `<p><strong>Décision : ${esc(r.decision || r.status)}</strong>${r.decided_at ? ' · ' + esc(new Date(r.decided_at).toLocaleTimeString('fr-FR')) : ''}
        ${ex.error ? ' · ' + esc(ex.error) : ''}</p>` + remediationHtml(i);
    }
    if (message) body += `<p class="hr-note" style="color:#f87171">${esc(message)}</p>`;
    return `<div class="panel-heading"><h2>Revue humaine</h2><span class="badge ${tone}">${esc(label)}</span></div><div class="section-body">${body}</div>`;
  }


  const pairs = o => Object.entries(o || {}).filter(([, v]) => typeof v !== 'object').map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(' · ');

  function attemptsOf(i) { const f = i.flow || {}; return f.remediation_attempts || (f.remediation ? [f.remediation] : []); }

  function chart(i) {
    const base = state.threshold;
    const attempts = attemptsOf(i);
    const thrOf = {};
    attempts.forEach(a => (a.windows || []).forEach(w => { thrOf[w.window] = a.threshold; }));
    const pts = state.metrics.filter(m => m.incident_id === i.id && m.metrics && m.metrics.log_loss != null)
      .map(m => ({w: m.window, v: m.metrics.log_loss, after: m.source === 'after_correction'}));
    if (pts.length < 2) return '';
    const W = 760, H = 250, L = 48, R = 16, T = 22, B = 34;
    const maxThr = Math.max(base, ...attempts.map(a => a.threshold));
    const lo = Math.log10(Math.min(0.3, ...pts.map(p => p.v))), hi = Math.log10(Math.max(maxThr * 1.5, ...pts.map(p => p.v)) * 1.15);
    const x = k => L + k * (W - L - R) / (pts.length - 1), y = v => T + (hi - Math.log10(v)) * (H - T - B) / (hi - lo);
    const ticks = [0.3, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(t => Math.log10(t) >= lo && Math.log10(t) <= hi);
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Erreur du modèle avant et après chaque correction" style="display:block">`;
    svg += ticks.map(t => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#202a39" stroke-dasharray="3 4"/><text x="${L - 8}" y="${y(t) + 4}" text-anchor="end" fill="#78869b" font-size="11">${t}</text>`).join('');
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(base)}" y2="${y(base)}" stroke="#d9922e" stroke-dasharray="6 4"/><text x="${L + 4}" y="${y(base) - 6}" fill="#d9922e" font-size="11">seuil ${base.toFixed(3)} : il faut être en dessous</text>`;
    attempts.forEach((a, n) => {
      const k = pts.findIndex(p => p.after && p.w === (a.windows || [{}])[0].window);
      if (k <= 0) return;
      const cx = (x(k - 1) + x(k)) / 2;
      svg += `<line x1="${cx}" x2="${cx}" y1="${T}" y2="${H - B}" stroke="#c7d2e3" stroke-dasharray="4 3"/><text x="${cx + 5}" y="${T - 6}" fill="#c7d2e3" font-size="10.5">tentative ${n + 1}</text>`;
      if (Math.abs(a.threshold - base) > 1e-6) {
        svg += `<line x1="${cx}" x2="${W - R}" y1="${y(a.threshold)}" y2="${y(a.threshold)}" stroke="#f0b35a" stroke-dasharray="2 3"/><text x="${W - R}" y="${y(a.threshold) - 6}" text-anchor="end" fill="#f0b35a" font-size="10.5">nouveau seuil ${a.threshold.toFixed(3)}</text>`;
      }
    });
    svg += `<polyline fill="none" stroke="#6d92ff" stroke-width="2.2" points="${pts.map((p, k) => `${x(k)},${y(p.v)}`).join(' ')}"/>`;
    svg += pts.map((p, k) => { const t = thrOf[p.w] || base; return `<circle cx="${x(k)}" cy="${y(p.v)}" r="${p.after ? 4.6 : 3.2}" fill="${p.after ? (p.v <= t ? '#4ade80' : '#f87171') : '#6d92ff'}"><title>${p.w} · erreur ${p.v.toFixed(3)}</title></circle>`; }).join('');
    const step = Math.ceil(pts.length / 14);
    svg += pts.map((p, k) => k % step === 0 ? `<text x="${x(k)}" y="${H - 12}" text-anchor="middle" fill="#78869b" font-size="10">${p.w}</text>` : '').join('');
    svg += `</svg>`;
    return svg + `<div class="hr-legend"><span><i style="background:#6d92ff"></i>erreur du modèle par lot (log loss, échelle log)</span><span><i class="dot" style="background:#4ade80"></i>après correction : corrigé</span><span><i class="dot" style="background:#f87171"></i>après correction : pas corrigé</span></div>`;
  }

  const ACTION_PLAIN = {quarantine_batch: 'Réparer les données d’entrée et mettre de côté les lots abîmés',
    prepare_retraining: 'Réentraîner le modèle (lui réapprendre à partir des clients récents)',
    prepare_rollback: 'Remettre l’ancienne version du modèle, celle qui marchait', close_alert: 'Fermer l’alerte sans rien changer',
    recalibrate_threshold: 'Adapter le seuil d’alerte à la nouvelle population'};

  function proposalsHtml(i, m) {
    const p = m.proposals;
    if (!p) return '';
    const auto = (p.automatic || []).map(o => `<li class="hr-prop ${o.would_fix ? 'good' : ''}"><div><strong>${esc(o.label)}</strong>
        <span class="badge ${o.would_fix ? 'green' : 'neutral'}">${o.would_fix ? 'corrigerait' : 'ne suffirait pas'}</span></div>
        <p>${esc(o.why)} Erreur estimée sur ${esc(o.tested_on)} : <b>${num(o.expected_log_loss)}</b> (seuil ${num(o.threshold || m.threshold)}).</p>
        <button class="button ${o.would_fix ? 'primary' : 'outline'}" data-solution="${esc(o.action)}">Appliquer cette solution</button></li>`).join('');
    const manual = (p.manual || []).map(o => `<li class="hr-prop"><div><strong>${esc(o.label)}</strong><span class="badge neutral">action humaine</span></div><p>${esc(o.why)}</p></li>`).join('');
    return `<h3 class="hr-sub">Solutions proposées par l’agent</h3>
      <p class="hr-help">L’agent a testé les autres actions sur le dernier lot dont on connaît la réponse, sans rien mettre en service. Choisissez une solution : elle sera appliquée, puis vérifiée par le monitoring sur 6 nouveaux lots.</p>
      <ul class="hr-props">${auto}${manual}</ul>`;
  }

  function correctionPanel(i) {
    const f = i.flow || {}, m = f.remediation, c = f.correction || {};
    if (!m) return '';
    const attempts = attemptsOf(i);
    const conf = f.confidence || {}, g = c.guard || {}, d = m.details || {};
    const cause = (i.diagnosis || {}).cause_probable;
    const nAfter = (m.windows || []).length;
    const strong = Number(conf.confidence) >= 0.7;
    const why = c.human_approval_id
      ? (strong ? `L’agent d’analyse a trouvé la cause « ${esc(cause)} » avec une preuve solide (confiance ${esc(conf.confidence)}), mais une approbation humaine était exigée. Un humain a approuvé la correction.`
        : `L’agent d’analyse a trouvé la cause « ${esc(cause)} », mais sa preuve était faible (confiance ${esc(conf.confidence)}). Un humain a approuvé la correction.`)
      : `L’agent d’analyse a trouvé la cause « ${esc(cause)} » avec une preuve solide (confiance ${esc(conf.confidence)}, au-dessus de 0,7). La correction est donc automatique.`;
    let data = '';
    if (m.action === 'prepare_retraining') data = `Pour réapprendre, il a pris les <strong>${esc(d.training_rows)} clients des lots ${esc((d.training_windows || []).join(', '))}</strong> : ce sont les derniers clients dont on connaît déjà la vraie réponse (parti ou resté). Il a ensuite testé le nouveau modèle sur les 500 clients du lot ${esc(d.comparison_window)}, qu’il n’avait pas vus.`;
    if (m.action === 'quarantine_batch') data = `Il a repéré les valeurs abîmées en comparant, pour chaque client, les données d’origine et les données reçues par le modèle. Lots mis de côté : <strong>${esc((d.quarantined_windows || []).join(', '))}</strong> (${esc(d.quarantined_rows)} lignes abîmées, interdites pour tout futur réentraînement).`;
    if (m.action === 'prepare_rollback') data = `Aucune donnée n’est nécessaire : il remet simplement en service la <strong>version de référence</strong> du modèle et retire la version déployée pendant l’incident.`;
    if (m.action === 'recalibrate_threshold') data = `Sur les derniers lots étiquetés, l’erreur observée est égale à l’erreur attendue d’un modèle juste : le modèle n’est pas cassé, ces clients sont imprévisibles. Le seuil d’alerte passe de <strong>${num(d.previous_threshold)} à ${num(d.new_threshold)}</strong>, calculé sur ces lots avec la même marge que le seuil d’origine. Le modèle n’est pas modifié.`;
    let test = '';
    if (m.action === 'prepare_retraining') {
      const gain = d.current_log_loss - d.candidate_log_loss;
      test = `Sur le lot ${esc(d.comparison_window)}, l’erreur du nouveau modèle est de ${num(d.candidate_log_loss)}, contre ${num(d.current_log_loss)} pour l’ancien. ` +
        (d.candidate_deployed ? (gain < 0.02 ? `Il est mis en service car il fait un peu mieux, mais <strong>la différence est minime</strong>.` : `Il fait nettement mieux, donc <strong>il est mis en service</strong>.`)
          : `Il ne fait pas mieux, donc <strong>l’ancien modèle est gardé</strong>.`);
    }
    const result = `Ensuite, ${nAfter} nouveaux lots de 500 clients sont arrivés (${esc(m.windows[0].window)} à ${esc(m.windows[nAfter - 1].window)}). ` +
      `Erreur moyenne : <strong>${num(m.before_mean)} avant → ${num(m.after_mean)} après</strong>, pour un seuil de ${num(m.threshold)}${m.action === 'recalibrate_threshold' ? ' (nouveau seuil)' : ''}. ` +
      (m.passed ? `<strong class="ok">Le problème est corrigé</strong> : l’erreur est sous le seuil.` : `<strong class="ko">Le problème n’est pas corrigé</strong> : ${esc(m.windows_above_threshold)} lot(s) sur ${nAfter} restent au-dessus du seuil.`);
    let next = '';
    if (!m.passed) {
      const auc = m.windows.reduce((a, w) => a + (w.roc_auc || 0), 0) / nAfter;
      next = (m.action === 'prepare_retraining' && auc < 0.6
        ? `Même un modèle réentraîné n’y arrive pas : sa capacité à reconnaître les clients qui partent (ROC AUC ≈ ${num(auc)}) est proche du hasard (0,5). `
        : `Cette action n’a pas fait baisser l’erreur. `) + `L’agent a donc testé d’autres solutions et vous les propose ci-dessous.`;
    }
    const history = attempts.length > 1 ? `<ol class="hr-history">${attempts.map(a => `<li>${a.origin === 'proposal' ? 'Solution approuvée' : 'Action de l’agent'} : <strong>${esc(ACTION_PLAIN[a.action] || a.action)}</strong> → <span class="${a.passed ? 'ok' : 'ko'}">${a.passed ? 'corrigé' : 'pas corrigé'}</span> (erreur ${num(a.after_mean)}, seuil ${num(a.threshold)})</li>`).join('')}</ol>` : '';
    const steps = [['Pourquoi corriger', why]];
    if (history) steps.push(['Les tentatives', history]);
    steps.push([attempts.length > 1 ? 'Dernière tentative' : 'Ce que l’agent a décidé', `<strong>${esc(ACTION_PLAIN[m.action] || m.action)}</strong> (action <code>${esc(m.action)}</code>).`]);
    if (data) steps.push([m.action === 'recalibrate_threshold' ? 'Le diagnostic de l’agent' : 'Les données utilisées', data]);
    if (test) steps.push(['Test avant la mise en service', test]);
    steps.push(['Résultat sur les lots suivants', result]);
    if (next) steps.push(['Et maintenant ?', next]);
    const rows = (m.windows || []).map(w => `<tr><td>${esc(w.window)}</td><td>500</td><td>${num(w.log_loss)}</td><td>${num(w.roc_auc)}</td><td class="${w.log_loss <= m.threshold ? 'ok' : 'ko'}">${w.log_loss <= m.threshold ? '✓ corrigé' : '✗ encore en erreur'}</td></tr>`).join('');
    return `<div class="panel-heading"><h2>Correction : ce qui s’est passé</h2><span class="badge ${m.passed ? 'green' : 'red'}">${m.passed ? 'Problème corrigé' : 'Pas corrigé · solutions proposées'}</span></div>
      <div class="section-body">${chart(i)}
      <ol class="hr-steps">${steps.map(([t, v]) => `<li><strong>${t}</strong><span>${v}</span></li>`).join('')}</ol>
      ${m.passed ? '' : proposalsHtml(i, m)}
      <h3 class="hr-sub">Les lots arrivés après la dernière correction</h3>
      <p class="hr-help">Chaque ligne est un nouveau lot de 500 clients. <b>Erreur</b> (log loss) : plus elle est basse, mieux c’est ; il faut être sous ${num(m.threshold)}. <b>ROC AUC</b> : capacité du modèle à distinguer les clients qui partent de ceux qui restent (1 = parfait, 0,5 = au hasard).</p>
      <table class="hr-table"><thead><tr><th>Lot</th><th>Clients</th><th>Erreur (log loss)</th><th>ROC AUC</th><th>État</th></tr></thead><tbody>${rows}</tbody></table>
      <details class="hr-tech"><summary>Détails techniques de l’agent de correction</summary>
      <p><b>Garde-fou</b> : ${g.approved_by_human ? 'approuvé par un humain ; seule l’action du type d’incident est autorisée' : `action attendue ${esc(g.expected_action)}, proposée ${esc(g.proposed_action)}, ${g.accepted ? 'acceptée' : 'refusée'}`}.<br>
      <b>Enregistrement de l’agent</b> : ${pairs(c.tool_result)}.<br><b>Effet appliqué</b> : ${esc(m.effect)}</p></details></div>`;
  }

  function renderCorrection(i, main) {
    let panel = document.getElementById('hr-correction');
    const content = i ? correctionPanel(i) : '';
    if (!content) { if (panel) panel.remove(); return; }
    if (!panel) {
      panel = document.createElement('section');
      panel.id = 'hr-correction'; panel.className = 'panel'; panel.style.marginBottom = '20px';
      const anchor = document.getElementById('hr-panel') || [...main.children].find(sct => /Plan de correction/.test(sct.textContent.slice(0, 60)));
      anchor ? anchor.after(panel) : main.prepend(panel);
    }
    if (panel.dataset.content !== content) {
      panel.innerHTML = content; panel.dataset.content = content;
      panel.querySelectorAll('[data-solution]').forEach(b => b.addEventListener('click', async () => {
        panel.querySelectorAll('[data-solution]').forEach(x => x.disabled = true);
        b.textContent = 'Application et vérification en cours…';
        const res = await fetch(`/api/incidents/${i.id}/solution`, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: b.dataset.solution})});
        if (!res.ok) { message = 'Solution refusée : ' + ((await res.json()).detail || res.status); }
        busy = false; refresh();
      }));
    }
  }

  function renderPanel() {
    const tag = document.querySelector('.incident-summary .eyebrow');
    const id = tag && (tag.textContent.match(/INC-([0-9A-F]{6})/) || [])[1];
    const old = document.getElementById('hr-panel');
    const i = id && state && state.incidents.find(x => x.id.slice(0, 6).toUpperCase() === id);
    if (i && i.validation && i.validation.simulated === false) {
      // Textes fixes de Sentinel remplacés quand la correction a vraiment été vérifiée.
      document.querySelectorAll('.validation strong').forEach(el => {
        if (/Validation réelle en attente/.test(el.textContent)) el.textContent = i.validation.passed ? 'Vérification par le monitoring : réussie' : 'Vérification par le monitoring : échouée';
      });
    }
    // Le panneau « Plan de correction » de Sentinel fait doublon avec « Revue humaine » et
    // « Correction : ce qui s'est passé » : il est masqué dès que ces panneaux existent.
    const ours = i && i.flow && (i.flow.remediation || (i.flow.human_review && i.flow.human_review.review_url));
    document.querySelectorAll('.detail-main > .panel').forEach(p => {
      if (/Plan de correction/.test(p.textContent.slice(0, 40))) p.style.display = ours ? 'none' : '';
    });
    if (i && i.flow && i.flow.remediation) {
      document.querySelectorAll('.report-callout p').forEach(el => {
        if (/correction réelle à réaliser/.test(el.textContent)) el.textContent = i.flow.remediation.passed
          ? 'Correction appliquée et vérifiée par le monitoring.' : 'Correction appliquée mais insuffisante : intervention humaine nécessaire.';
      });
    }
    const main = document.querySelector('.detail-main');
    if (!main) { if (old) old.remove(); return; }
    const r = i && i.flow && i.flow.human_review;
    if (!r || !r.review_url) { if (old) old.remove(); renderCorrection(i, main); return; }
    let panel = old;
    if (!panel) {
      panel = document.createElement('section');
      panel.id = 'hr-panel'; panel.className = 'panel'; panel.style.marginBottom = '20px';
      const correction = [...main.children].find(sct => /Plan de correction/.test(sct.textContent.slice(0, 60)));
      correction ? correction.after(panel) : main.prepend(panel);
    }
    const content = panelHtml(i, r);
    if (panel.dataset.content !== content) {
      panel.innerHTML = content; panel.dataset.content = content;
      panel.querySelectorAll('[data-decision]').forEach(b => b.addEventListener('click',
        () => decide(r, b.dataset.decision, [...panel.querySelectorAll('button')])));
    }
    renderCorrection(i, main);
  }

  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      state = await (await fetch('/api/state')).json();
      renderPanel();
      relabel();
    } catch (e) { /* le serveur redémarre : on réessaie au prochain tour */ } finally { busy = false; }
  }
  // Dates à la place des numéros de lot : chaque lot (w000, w001…) représente une journée de clients,
  // à partir du jour où le parcours a été lancé. Les axes « F1, F2… » de Sentinel deviennent aussi des dates.
  const DAY = 86400000;
  function contextIncident() {
    if (!state || !state.incidents.length) return null;
    const tag = document.querySelector('.incident-summary .eyebrow');
    const id = tag && (tag.textContent.match(/INC-([0-9A-F]{6})/) || [])[1];
    return (id && state.incidents.find(x => x.id.slice(0, 6).toUpperCase() === id)) || state.incidents[0];
  }
  function dateOf(index, short) {
    const inc = contextIncident();
    if (!inc) return null;
    const start = new Date(inc.created_at); start.setHours(0, 0, 0, 0);
    const d = new Date(start.getTime() + index * DAY);
    return short ? d.toLocaleDateString('fr-FR', {day: '2-digit', month: '2-digit'})
      : d.toLocaleDateString('fr-FR', {day: 'numeric', month: 'short'});
  }
  const SKIP = new Set(['PRE', 'CODE', 'SCRIPT', 'STYLE', 'TEXTAREA', 'INPUT']);
  function relabel() {
    const root = document.getElementById('root');
    if (!root || !contextIncident()) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const parent = node.parentElement;
      if (!parent || SKIP.has(parent.tagName) || parent.closest('pre, code')) continue;
      const text = node.nodeValue;
      const tick = text.match(/^F(\d+)$/);
      if (tick && parent.closest('.recharts-surface')) {
        const label = dateOf(Number(tick[1]) - 1, true);
        if (label) node.nodeValue = label;
        continue;
      }
      if (/\bw\d{3}\b/.test(text)) {
        const short = !!parent.closest('svg');
        node.nodeValue = text.replace(/\bw(\d{3})\b/g, (m, n) => dateOf(Number(n), short) || m);
      }
    }
    // Les dates sont plus longues que « F1 » : on masque les étiquettes qui se chevaucheraient.
    root.querySelectorAll('.recharts-xAxis-tick-labels').forEach(group => {
      let last = -Infinity;
      [...group.querySelectorAll('text')].map(t => [t, t.getBBox()]).sort((a, b) => a[1].x - b[1].x).forEach(([t, box]) => {
        const keep = box.x >= last + 6;
        t.style.visibility = keep ? '' : 'hidden';
        if (keep) last = box.x + box.width;
      });
    });
  }
  let pending = false;
  new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; relabel(); });
  }).observe(document.documentElement, {subtree: true, childList: true, characterData: true});

  const style = document.createElement('style');
  style.textContent = `.hr-grid{display:grid;grid-template-columns:minmax(140px,.3fr) 1fr;gap:10px 16px;margin:0 0 16px}.hr-grid dt{color:#78869b}.hr-grid dd{margin:0}
    .hr-actions{display:flex;gap:10px;flex-wrap:wrap}.hr-note{font-size:12px;margin-top:10px}
    .ko-card{border-color:#f87171}.hr-props{list-style:none;padding:0;margin:8px 0 6px;display:grid;gap:10px}.hr-prop{border:1px solid #263244;border-radius:10px;padding:12px 14px}.hr-prop.good{border-color:#2f6b47}.hr-prop div{display:flex;justify-content:space-between;gap:10px;align-items:center}.hr-prop p{font-size:13px;color:#b7c1d1;margin:6px 0 10px;line-height:1.5}.hr-history{margin:4px 0 0;padding-left:18px;font-size:13px;color:#b7c1d1}.hr-x{background:none;border:0;color:#9aa6b8;font-size:18px;cursor:pointer;line-height:1}
    .hr-steps{margin:16px 0;padding-left:20px;display:grid;gap:10px}.hr-steps li{padding-left:4px}.hr-steps>li>strong{display:block;font-size:13px;color:#e6ebf3}.hr-steps span strong{color:#e6ebf3}.hr-steps span{font-size:13px;color:#b7c1d1;line-height:1.5}
    .hr-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#9aa6b8;margin:6px 0 4px}.hr-legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle;border-radius:2px}.hr-legend i.dot{width:9px;height:9px;border-radius:50%}
    .ok{color:#4ade80}.ko{color:#f87171}.hr-sub{font-size:14px;margin:18px 0 4px}.hr-help{font-size:12.5px;color:#9aa6b8;line-height:1.5;margin:0 0 6px}.hr-tech{margin-top:14px;font-size:12.5px;color:#9aa6b8}.hr-tech summary{cursor:pointer;color:#c7d2e3}
    .hr-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}.hr-table th,.hr-table td{text-align:left;padding:7px 8px;border-bottom:1px solid #202a39}.hr-table th{color:#78869b;font-weight:500}
    @media(max-width:600px){.hr-grid{grid-template-columns:1fr}}`;
  document.head.appendChild(style);
  refresh();
  setInterval(refresh, 1500);
})();
"""


def _review_client():
    if "client" not in _REVIEW:
        import importlib.util
        group = ROOT / "mlops_correction_agent_groq"
        spec = importlib.util.spec_from_file_location("review_app", group / "app.py")
        module = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        os.chdir(group)  # app.py crée artifacts_v3/ dans le dossier courant
        try:
            spec.loader.exec_module(module)
        finally:
            os.chdir(cwd)
        _REVIEW["client"] = module.app.test_client()
    return _REVIEW["client"]


def _incident_for_review(review_id):
    for incident in sentinel.incidents():
        review = ((incident.get("flow") or {}).get("human_review") or {})
        if review.get("review_id") == review_id:
            return incident
    return None


def _sync_review(review_id):
    """Recopie dans l'incident Sentinel la décision enregistrée par app.py."""
    from urllib.parse import parse_qs, urlparse
    from human_review_store import HumanReviewStore, review_database_path
    incident = _incident_for_review(review_id)
    if incident is None:
        return
    flow = incident["flow"]
    old = flow["human_review"]
    token = parse_qs(urlparse(old["review_url"]).query).get("token", [""])[0]
    review = HumanReviewStore(review_database_path()).get_review(review_id, token)
    if review["status"] == old.get("status"):
        return
    flow["human_review"] = {**old, **{k: review.get(k) for k in ("status", "decision", "decided_at", "execution")}}
    execution = review.get("execution") or {}
    if review["status"] == "completed":
        plan, tool, check = execution["plan"], execution["tool_result"], execution["verification"]
        flow["correction"] = {"proposed_plan": plan, "final_plan": plan, "guard": execution["guard_result"],
                              "tool_result": tool, "verification": check, "human_approval_id": review_id}
        correction = dict(simulated=False, applied=True,
                          action=f"Approuvée par un humain : {ACTIONS.get(plan['action'], plan['action'])} ({plan['action']})",
                          message=(f"Agent de correction lancé après la validation {review_id[:8]} (garde-fou : approuvé par un "
                                   f"humain). Effet : {_effect(plan['action'])}"))
        sentinel.update(incident["id"], flow=flow, correction=correction, phase=3, status="correcting",
                        outcome="correcting_after_human_approval")
        sentinel.emit(incident["id"], "Décision humaine : approuvée", {"message": correction["message"]})
        return plan["action"]
    elif review["status"] == "rejected":
        correction = dict(simulated=True, applied=False, action="Refusée par un humain",
                          message="Aucune action exécutée : la correction proposée a été refusée.")
        validation = dict(simulated=True, passed=None, before=None, after=None, message="Incident clos sans correction.")
        outcome, kind = "rejected_by_human", "Décision humaine : refusée"
    else:
        correction = dict(simulated=True, applied=False, action="Approuvée, mais action bloquée",
                          message=execution.get("error") or f"Statut : {review['status']}.")
        validation = dict(simulated=True, passed=None, before=None, after=None,
                          message="Aucune modification : l'agent de correction a refusé d'exécuter l'action.")
        outcome, kind = "human_review_blocked", "Décision humaine : approuvée, action bloquée"
    sentinel.update(incident["id"], flow=flow, correction=correction, validation=validation, outcome=outcome,
                    phase=5, ended_at=sentinel.now())
    sentinel.emit(incident["id"], kind, {"message": correction["message"]})
    sentinel.emit(incident["id"], "report_ready", {"message": f"Rapport mis à jour : {outcome}."})
    return None


def _plain(response):
    from fastapi.responses import Response
    headers = {}
    if "Location" in response.headers:
        from urllib.parse import urlsplit
        target = urlsplit(response.headers["Location"])
        headers["Location"] = target.path + (f"?{target.query}" if target.query else "")
    return Response(response.data, status_code=response.status_code, headers=headers,
                    media_type=response.mimetype or "text/html")


async def review_page(review_id: str, request: Request):
    from starlette.concurrency import run_in_threadpool
    query = request.url.query
    return _plain(await run_in_threadpool(lambda: _review_client().get(f"/human-review/{review_id}", query_string=query)))


async def review_decision(review_id: str, request: Request):
    from starlette.concurrency import run_in_threadpool
    body, content_type = await request.body(), request.headers.get("content-type", "application/x-www-form-urlencoded")
    response = await run_in_threadpool(lambda: _review_client().post(
        f"/human-review/{review_id}/decision", data=body, content_type=content_type))
    try:
        action = _sync_review(review_id)
        if action:  # approuvée : l'agent applique la correction, le monitoring vérifie
            incident = _incident_for_review(review_id)
            result = await run_in_threadpool(_remediate, incident["id"], action)
            outcome = "corrected_after_human_approval" if result["passed"] else "correction_insufficient"
            sentinel.update(incident["id"], phase=5, status="completed", ended_at=sentinel.now(), outcome=outcome)
            sentinel.emit(incident["id"], "report_ready", {"message": f"Rapport mis à jour : {outcome}."})
    except Exception as exc:  # la décision reste enregistrée par app.py
        print("Synchronisation de la revue impossible :", _redact(str(exc)))
    return _plain(response)


def index():
    from fastapi.responses import HTMLResponse
    page = (ROOT / "frontend" / "dist" / "index.html").read_text(encoding="utf-8")
    import hashlib
    version = hashlib.sha256(REVIEW_JS.encode("utf-8")).hexdigest()[:10]  # nouvelle URL à chaque mise à jour
    return HTMLResponse(page.replace("</body>", f'<script src="/addon/review.js?v={version}" defer></script></body>'),
                        headers={"Cache-Control": "no-store"})


def review_script():
    from fastapi.responses import Response
    return Response(REVIEW_JS, media_type="text/javascript; charset=utf-8", headers={"Cache-Control": "no-store"})


app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
app.add_api_route("/human-review/{review_id}", review_page, methods=["GET"])
app.add_api_route("/human-review/{review_id}/decision", review_decision, methods=["POST"])
app.add_api_route("/addon/review.js", review_script, methods=["GET"])
app.add_api_route("/api/incidents/{incident_id}/solution", apply_solution, methods=["POST"])
app.add_api_route("/", index, methods=["GET"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--orchestrator", choices=("rules", "llm"), default="rules",
                        help="llm : Groq (clé dans mlops_correction_agent_groq/.env)")
    parser.add_argument("--no-predict", action="store_true", help="désactiver l'alerte prédictive")
    parser.add_argument("--label-delay", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.6, help="secondes par fenêtre")
    parser.add_argument("--approbation-toujours", action="store_true",
                        help="demander l'approbation humaine avant toute correction, même quand la confiance est élevée")
    args = parser.parse_args(argv)
    SETTINGS.update(predict=not args.no_predict, label_delay=args.label_delay, always_approve=args.approbation_toujours,
                    orchestrator_mode=args.orchestrator, speed=args.speed)
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("Interface non compilée : dans frontend, lancer « npm install » puis « npm run build ».")
    if SETTINGS["predict"] and not (ROOT / "monitoring_runs" / "predictor" / "predictor.joblib").exists():
        raise SystemExit("Prédicteur absent : lancer « python -m monitoring train-predictor » ou ajouter --no-predict.")
    # Le lien de l'e-mail de revue ouvre la page servie par Sentinel (plus besoin de lancer app.py).
    os.environ["HUMAN_REVIEW_BASE_URL"] = f"http://127.0.0.1:{args.port}"
    import uvicorn
    print(f"Sentinel (flux complet) : http://127.0.0.1:{args.port}  · orchestrateur {args.orchestrator}"
          f" · alerte prédictive {'oui' if SETTINGS['predict'] else 'non'}"
          f" · approbation {'toujours' if SETTINGS['always_approve'] else 'si doute'}  · Ctrl+C pour arrêter")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
