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

SETTINGS = {"predict": True, "label_delay": 2, "orchestrator_mode": "rules", "speed": 0.6}
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
    REVIEW_LINK["url"] = review.get("review_url")
    return _send_review_email(review)


human_notifications.send_human_review_email = _send_and_keep_link


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
                  f"* E-mail : {_email_status(review.get('notification'))}",
                  f"* Décision humaine : {review.get('decision') or 'en attente'}" + (f" ({review['decided_at']})" if review.get("decided_at") else ""),
                  *([f"* Exécution bloquée : {execution['error']}"] if execution.get("error") else []),
                  "* Si l'humain approuve, l'agent de correction exécute, en simulation, la seule action correspondant au type d'incident."]
        section += 1
    if correction:
        plan = correction.get("final_plan") or {}
        origin = " après validation humaine" if correction.get("human_approval_id") else ""
        lines += ["", f"## {section}. Agent de correction{origin} (`agent_v3.py` : plan → garde-fou → exécution → vérification)", "",
                  f"* Action proposée : {(correction.get('proposed_plan') or {}).get('action')}",
                  f"* Garde-fou : {_pairs(correction.get('guard'))}",
                  f"* **Action retenue : {plan.get('action')}** — {plan.get('reason', '')}",
                  f"* Exécution : {_pairs(correction.get('tool_result'))}",
                  f"* Vérification : {_pairs(correction.get('verification'))}"]
        section += 1
    if not review and not correction:
        lines += ["", f"## {section}. Correction", "", "Aucune action : pas d'alerte confirmée ou analyse incomplète."]
        section += 1

    lines += ["", f"## {section}. Limites", "",
              "* Les actions de correction sont simulées ou en attente d'évaluation : aucun modèle n'est réentraîné, "
              "remplacé ni redéployé, et aucune donnée n'est modifiée.",
              "* Le gain de performance après correction n'est pas mesuré dans ce parcours.",
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
            plan, tool, check = c["final_plan"], c["tool_result"], c["verification"]
            action = plan.get("action")
            correction = dict(simulated=True, applied=False, action=f"{ACTIONS.get(action, action)} ({action})",
                              message=(f"Agent de correction : statut « {tool.get('status')} ». "
                                       f"Confiance {assessment.get('confidence')}, route décidée par l'orchestrateur : "
                                       f"{decision.get('route')}. {plan.get('reason') or ''}").strip())
            validation = dict(simulated=True, passed=check.get("production_unchanged"), before=None, after=None,
                              message=("Vérification de l'agent : production inchangée = "
                                       f"{check.get('production_unchanged')}. L'action reste à valider avant toute "
                                       "modification réelle du modèle ou des données."))
            outcome = "correction_planned"
        else:
            review = result.get("human_review") or {}
            correction = dict(simulated=True, applied=False, action="Revue humaine requise",
                              message=(f"{decision.get('reason', '')} Demande {review.get('review_id', '')[:8]} "
                                       f"enregistrée ({review.get('status', 'pending')})."))
            validation = dict(simulated=True, passed=None, before=None, after=None,
                              message=(f"E-mail : {_email_status(review.get('notification'))}. Décision à prendre dans le "
                                       "panneau « Revue humaine » ci-dessous (ou avec le lien de l'e-mail)."))
            # L'agent de correction n'est PAS exécuté : le parcours s'arrête à l'étape Correction et attend
            # la décision humaine (phase 3). Le statut « completed » libère seulement Sentinel pour un autre
            # scénario ; le rapport est provisoire jusqu'à la décision.
            sentinel.update(incident_id, phase=3, status="completed", ended_at=sentinel.now(),
                            outcome="awaiting_human_decision", correction=correction, validation=validation,
                            pipeline_result=result.get("result_file"))
            sentinel.emit(incident_id, "Revue humaine demandée", {"message": validation["message"]})
            sentinel.emit(incident_id, "En attente de la décision humaine", {"message": (
                "L'agent de correction n'a rien exécuté. Il agira seulement si un humain approuve.")})
            return
        sentinel.update(incident_id, phase=3, status="correcting", correction=correction)
        sentinel.emit(incident_id, "correction_started", correction)
        await asyncio.sleep(1.5)
        sentinel.update(incident_id, phase=4, status="validating", validation=validation)
        sentinel.emit(incident_id, "validation_started", validation)
        await asyncio.sleep(1)
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
REVIEW_JS = r"""// Panneau « Revue humaine » ajouté à la page d'incident de Sentinel (sans modifier son code React).
(() => {
  const ACTIONS = {quarantine_batch: 'Mise en quarantaine du lot', prepare_retraining: 'Préparation d’un réentraînement',
    prepare_rollback: 'Préparation du retour à la version de référence', close_alert: 'Clôture de l’alerte'};
  const BY_TYPE = {fausse_alerte: 'close_alert', qualite_donnees: 'quarantine_batch', derive_donnees: 'prepare_retraining', regression_version: 'prepare_rollback'};
  const STATUS = {pending: ['En attente de décision', 'orange'], approved: ['Approuvée', 'blue'], executing: ['Exécution…', 'blue'],
    completed: ['Approuvée · correction exécutée', 'green'], rejected: ['Refusée', 'red'], blocked: ['Approuvée · action bloquée', 'red'], failed: ['Échec', 'red']};
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const email = n => !n ? '—' : n.status === 'sent' ? `envoyé à ${n.recipient}` : n.status === 'not_configured' ? 'non configuré (SMTP absent du .env)' : `échec (${n.error_type || n.status})`;
  let busy = false, message = '';

  function html(i, r) {
    const req = r.request || {}, [label, tone] = STATUS[r.status] || [r.status, 'neutral'];
    const action = BY_TYPE[req.incident_type];
    const ex = r.execution || {}, tool = ex.tool_result || {}, check = ex.verification || {};
    let body = `<dl class="hr-grid">
      <dt>Raison</dt><dd>${esc(r.reason)}</dd>
      <dt>Type d’incident</dt><dd>${esc(req.incident_type)} · cause proposée ${esc(req.cause_probable)} · confiance ${esc(req.confidence)}</dd>
      <dt>Si vous approuvez</dt><dd>${action ? `${esc(ACTIONS[action])} (<code>${action}</code>), exécutée en simulation par l’agent de correction` : 'aucune action possible : type inconnu, l’approbation sera bloquée'}</dd>
      <dt>E-mail</dt><dd>${esc(email(r.notification))}</dd></dl>`;
    if (r.status === 'pending') {
      body += `<div class="hr-actions"><button class="button primary" data-decision="approved">Approuver</button>
        <button class="button outline" data-decision="rejected">Refuser</button></div>
        <p class="muted hr-note">La décision est enregistrée une seule fois. Aucune action n’est exécutée sans approbation.</p>`;
    } else {
      body += `<div class="validation"><div><strong>Décision : ${esc(r.decision || r.status)}${r.decided_at ? ' · ' + esc(new Date(r.decided_at).toLocaleTimeString('fr-FR')) : ''}</strong>
        <p>${ex.error ? esc(ex.error) : tool.status ? `Action ${esc((ex.plan || {}).action)} · statut « ${esc(tool.status)} » · production inchangée : ${esc(check.production_unchanged)}` : 'Aucune action exécutée.'}</p></div></div>`;
    }
    if (message) body += `<p class="hr-note" style="color:#f87171">${esc(message)}</p>`;
    return `<div class="panel-heading"><h2>Revue humaine</h2><span class="badge ${tone}">${esc(label)}</span></div><div class="section-body">${body}</div>`;
  }

  async function decide(r, decision) {
    const token = new URL(r.review_url).searchParams.get('token');
    message = '';
    const res = await fetch(`/human-review/${r.review_id}/decision`, {method: 'POST', redirect: 'manual',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body: new URLSearchParams({token, decision})});
    if (res.status >= 400) message = (await res.text()) || 'Décision refusée.';
  }

  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      const tag = document.querySelector('.incident-summary .eyebrow');
      const id = tag && (tag.textContent.match(/INC-([0-9A-F]{6})/) || [])[1];
      const old = document.getElementById('hr-panel');
      if (!id) { if (old) old.remove(); return; }
      const state = await (await fetch('/api/state')).json();
      const i = state.incidents.find(x => x.id.slice(0, 6).toUpperCase() === id);
      const r = i && i.flow && i.flow.human_review;
      if (!r || !r.review_url) { if (old) old.remove(); return; }
      const main = document.querySelector('.detail-main');
      if (!main) return;
      let panel = old;
      if (!panel) {
        panel = document.createElement('section');
        panel.id = 'hr-panel'; panel.className = 'panel'; panel.style.marginBottom = '20px';
        const correction = [...main.children].find(s => /Plan de correction/.test(s.textContent.slice(0, 60)));
        correction ? correction.after(panel) : main.prepend(panel);
      }
      const content = html(i, r);
      if (panel.dataset.content !== content) {
        panel.innerHTML = content; panel.dataset.content = content;
        panel.querySelectorAll('[data-decision]').forEach(b => b.addEventListener('click', async () => {
          panel.querySelectorAll('button').forEach(x => x.disabled = true);
          await decide(r, b.dataset.decision); busy = false; refresh();
        }));
      }
    } catch (e) { /* le serveur redémarre : on réessaie au prochain tour */ } finally { busy = false; }
  }
  const style = document.createElement('style');
  style.textContent = `.hr-grid{display:grid;grid-template-columns:minmax(140px,.3fr) 1fr;gap:10px 16px;margin:0 0 16px}.hr-grid dt{color:#78869b}.hr-grid dd{margin:0}
    .hr-actions{display:flex;gap:12px;flex-wrap:wrap}.hr-note{font-size:12px;margin-top:10px}@media(max-width:600px){.hr-grid{grid-template-columns:1fr}}`;
  document.head.appendChild(style);
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
        correction = dict(simulated=True, applied=False,
                          action=f"Approuvée par un humain : {ACTIONS.get(plan['action'], plan['action'])} ({plan['action']})",
                          message=(f"Agent de correction exécuté après la validation {review_id[:8]} : statut "
                                   f"« {tool.get('status')} ». {_pairs(tool)}"))
        validation = dict(simulated=True, passed=check.get("production_unchanged"), before=None, after=None,
                          message=f"Vérification de l'agent : {_pairs(check)}.")
        outcome, kind = "completed_after_human_approval", "Décision humaine : approuvée"
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
        _sync_review(review_id)
    except Exception as exc:  # la décision reste enregistrée par app.py
        print("Synchronisation de la revue impossible :", _redact(str(exc)))
    return _plain(response)


def index():
    from fastapi.responses import HTMLResponse
    page = (ROOT / "frontend" / "dist" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(page.replace("</body>", '<script src="/addon/review.js" defer></script></body>'))


def review_script():
    from fastapi.responses import Response
    return Response(REVIEW_JS, media_type="text/javascript; charset=utf-8")


app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
app.add_api_route("/human-review/{review_id}", review_page, methods=["GET"])
app.add_api_route("/human-review/{review_id}/decision", review_decision, methods=["POST"])
app.add_api_route("/addon/review.js", review_script, methods=["GET"])
app.add_api_route("/", index, methods=["GET"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--orchestrator", choices=("rules", "llm"), default="rules",
                        help="llm : Groq (clé dans mlops_correction_agent_groq/.env)")
    parser.add_argument("--no-predict", action="store_true", help="désactiver l'alerte prédictive")
    parser.add_argument("--label-delay", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.6, help="secondes par fenêtre")
    args = parser.parse_args(argv)
    SETTINGS.update(predict=not args.no_predict, label_delay=args.label_delay,
                    orchestrator_mode=args.orchestrator, speed=args.speed)
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("Interface non compilée : dans frontend, lancer « npm install » puis « npm run build ».")
    if SETTINGS["predict"] and not (ROOT / "monitoring_runs" / "predictor" / "predictor.joblib").exists():
        raise SystemExit("Prédicteur absent : lancer « python -m monitoring train-predictor » ou ajouter --no-predict.")
    # Le lien de l'e-mail de revue ouvre la page servie par Sentinel (plus besoin de lancer app.py).
    os.environ["HUMAN_REVIEW_BASE_URL"] = f"http://127.0.0.1:{args.port}"
    import uvicorn
    print(f"Sentinel (flux complet) : http://127.0.0.1:{args.port}  · orchestrateur {args.orchestrator}"
          f" · alerte prédictive {'oui' if SETTINGS['predict'] else 'non'}  · Ctrl+C pour arrêter")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
