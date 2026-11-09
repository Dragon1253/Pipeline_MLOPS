"""Interface web du flux complet : monitoring → analyse → orchestrateur → correction.

    python integration_app.py        puis ouvrir http://127.0.0.1:8050

Le pipeline lancé est exactement celui de ``run_full_pipeline.py`` : le code du groupe
(``mlops_correction_agent_groq/pipeline_v4.py``) est réutilisé sans modification, avec le
monitoring à jour de ``monitoring/``. La page suit chaque étape en direct : log loss et
risque prédit fenêtre par fenêtre, alertes, outils appelés par l'agent d'analyse,
diagnostic, score de confiance, décision de l'orchestrateur et action de correction.

Clés : NVIDIA (analyse réelle) dans ``.env`` à la racine ; Groq (orchestrateur LLM) dans
``mlops_correction_agent_groq/.env``. Elles ne sont jamais affichées.
"""
import os
from pathlib import Path
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, request

import run_full_pipeline as integration  # charge monitoring/ puis le pipeline du groupe
import monitoring.service as monitoring_service
import pipeline_v4

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
RUNS = {}
BUSY = threading.Lock()


def key_configured(env_file, name, placeholder=""):
    if os.environ.get(name):
        return True
    path = Path(env_file)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().partition("=")
        value = value.strip().strip('"').strip("'")
        if sep and key.strip() == name and value and value != placeholder:
            return True
    return False


def _summary(alert):
    return {k: alert.get(k) for k in ("window_id", "severity", "observed_value", "detected_at_step", "risk")}


def _worker(run_id, params):
    """Toute erreur, même au chargement d'un module, s'affiche dans la page et libère la place."""
    run = RUNS[run_id]
    try:
        _execute(run_id, params)
    except Exception as exc:
        run.update(status="error", error=_redact(f"{type(exc).__name__} : {exc}"),
                   trace=_redact(traceback.format_exc()))
        run["events"].append({"kind": "stage", "stage": "error", "t": round(time.time() - run["started"], 1)})
    finally:
        BUSY.release()


def _execute(run_id, params):
    run = RUNS[run_id]

    def emit(kind, **payload):
        run["events"].append({"kind": kind, "t": round(time.time() - run["started"], 1), **payload})

    def stage(name):
        run["stage"] = name
        emit("stage", stage=name)

    def on_monitoring(event):
        kind = event["event"]
        if kind == "window_scored":
            emit("window", window=event["window"], step=event["step"])
            time.sleep(params["speed"])
        elif kind == "metric_recorded":
            m = event["metrics"]
            emit("metric", window=event["window"], step=event["step"], log_loss=m["log_loss"],
                 accuracy=m["accuracy"], recall=m["recall"], status=event["status"])
        elif kind == "risk_recorded":
            emit("risk", window=event["window"], step=event["step"], risk=event["risk"], threshold=event["threshold"])
        elif kind == "alert_triggered":
            emit("alert", **_summary(event["alert"]))
        elif kind == "predictive_alert":
            a = event["alert"]
            emit("predictive", window_id=a["window_id"], step=a["detected_at_step"], risk=a["risk"],
                 signals=[s["signal"] for s in a["top_signals"]])
        elif kind == "incident_closed":
            emit("closed", window=event["window"], reason=event.get("reason"))
        elif kind == "episode_packaged":
            emit("episode", episode=Path(event["episode_dir"]).name)

    predictor = None
    if params["predict"]:
        from monitoring.predictor import load_predictor
        predictor = load_predictor()

    def stream(*args, **kwargs):
        stage("monitoring")
        summary = monitoring_service.run_stream(*args, predictor=predictor, label_delay=params["label_delay"],
                                                on_event=on_monitoring, **kwargs)
        stage("analysis")
        return summary

    # L'agent d'analyse (LangChain, NVIDIA, Ollama) n'est chargé que pour une vraie analyse.
    analysis_agent = None
    if params["analysis_mode"] != "fixture":
        import mlops_churn.agent as analysis_agent
        original_investigate = analysis_agent.investigate
    original_fixture = pipeline_v4._fixture_analysis

    def investigate(**kwargs):
        def on_event(event):
            kind = event["event"]
            if kind in ("tool_started", "tool_completed"):
                emit("tool", event=kind, tool=event.get("tool"), evidence_id=event.get("evidence_id"),
                     arguments=event.get("arguments"),
                     status=(event.get("result") or {}).get("status"))
            elif kind in ("model_started", "model_retry"):
                emit("model", event=kind, call=event.get("call"), reason=event.get("reason"))
        return original_investigate(on_event=on_event, **kwargs)

    def fixture(**kwargs):
        emit("tool", event="tool_started", tool="compare_tables", evidence_id="E0001",
             arguments={"left": "source", "right": "inputs"}, status="running")
        result = original_fixture(**kwargs)
        emit("tool", event="tool_completed", tool="compare_tables", evidence_id="E0001", status="ok")
        return result

    pipeline_v4.run_stream = stream
    pipeline_v4._fixture_analysis = fixture
    if analysis_agent is not None:
        analysis_agent.investigate = investigate
    try:
        result = pipeline_v4.run_pipeline(
            scenario=params["scenario"], seed=params["seed"], analysis_mode=params["analysis_mode"],
            orchestrator_mode=params["orchestrator_mode"], correction_mode="rules",
            output_dir=str(integration.OUTPUT), n_windows=params["windows"], onset=params["onset"], ramp=params["ramp"])
        summary_file = Path(result["monitoring"]["run_dir"]) / "summary.json"
        if summary_file.exists():
            import json
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            result["monitoring"]["predictive_alerts"] = summary.get("predictive_alerts", [])
        stage("done")
        run.update(status="done", result=result)
    finally:
        pipeline_v4._fixture_analysis = original_fixture
        if analysis_agent is not None:
            analysis_agent.investigate = original_investigate


def _redact(text):
    """Masque les clés éventuellement présentes dans un message d'erreur."""
    for name in ("NVIDIA_API_KEY", "GROQ_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[CLÉ MASQUÉE]")
    return text


@app.post("/api/runs")
def start_run():
    body = request.get_json(force=True)
    params = {
        "scenario": body.get("scenario", "data_corruption"), "seed": int(body.get("seed", 42)),
        "analysis_mode": body.get("analysis_mode", "fixture"),
        "orchestrator_mode": body.get("orchestrator_mode", "rules"),
        "predict": bool(body.get("predict", True)), "label_delay": int(body.get("label_delay", 2)),
        "speed": float(body.get("speed", 0.25)), "windows": int(body.get("windows", 20)),
        "onset": int(body.get("onset", 8)), "ramp": int(body.get("ramp", 4)),
    }
    if params["scenario"] not in integration.SCENARIOS or params["analysis_mode"] not in ("fixture", "nvidia", "local") \
            or params["orchestrator_mode"] not in ("rules", "llm") or not 0 <= params["speed"] <= 2:
        return jsonify(error="Paramètres invalides."), 422
    if params["predict"] and not (ROOT / "monitoring_runs" / "predictor" / "predictor.joblib").exists():
        return jsonify(error="Prédicteur absent : lancer « python -m monitoring train-predictor »."), 409
    if not BUSY.acquire(blocking=False):
        return jsonify(error="Un parcours est déjà en cours."), 409
    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {"id": run_id, "params": params, "status": "running", "stage": "start",
                    "events": [], "result": None, "error": None, "started": time.time()}
    threading.Thread(target=_worker, args=(run_id, params), daemon=True).start()
    return jsonify(id=run_id)


@app.get("/api/runs/<run_id>")
def get_run(run_id):
    run = RUNS.get(run_id)
    if run is None:
        return jsonify(error="Parcours inconnu."), 404
    after = int(request.args.get("after", 0))
    return jsonify(id=run_id, status=run["status"], stage=run["stage"], events=run["events"][after:],
                   next=len(run["events"]), result=run["result"] if run["status"] != "running" else None,
                   error=run["error"])


@app.get("/api/config")
def config():
    return jsonify(
        nvidia=key_configured(ROOT / ".env", "NVIDIA_API_KEY", "nvapi-REMPLACER_ICI"),
        groq=key_configured(ROOT / "mlops_correction_agent_groq" / ".env", "GROQ_API_KEY"),
        predictor=(ROOT / "monitoring_runs" / "predictor" / "predictor.joblib").exists(),
        threshold=monitoring_service.ReferenceContext().calibration["threshold"])


@app.get("/")
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipeline MLOps</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--muted:#667085;--line:#e4e7ec;--accent:#3559e0;
--ok:#1a7f4b;--warn:#b26a00;--crit:#c0362c;--pred:#7a3fd1}
@media (prefers-color-scheme:dark){:root{--bg:#11141a;--card:#1a1f27;--ink:#e6e9ef;--muted:#98a2b3;--line:#2a313c;--accent:#7b96ff}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:19px;margin:0}.badges span{font-size:12px;padding:3px 8px;border-radius:99px;border:1px solid var(--line);margin-right:6px}
.on{color:var(--ok)}.off{color:var(--muted)}
main{display:grid;grid-template-columns:300px 1fr;gap:16px;padding:16px 24px;max-width:1400px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}select,input{width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.row{display:flex;gap:8px}.row>div{flex:1}button{margin-top:16px;width:100%;padding:11px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.hint{font-size:12px;color:var(--muted);margin-top:8px}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.step{padding:10px;border-radius:10px;border:1px solid var(--line);font-size:13px;text-align:center;color:var(--muted)}
.step b{display:block;color:var(--ink);font-size:14px}.step.active{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent) inset}
.step.done{border-color:var(--ok)}.step.done b::after{content:" ✓";color:var(--ok)}
svg{width:100%;height:280px}.legend{font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.log{max-height:300px;overflow:auto;font:12.5px/1.5 ui-monospace,Consolas,monospace}
.log div{padding:2px 0;border-bottom:1px dashed var(--line)}.k{color:var(--muted)}
.result h3{margin:0 0 6px;font-size:14px;color:var(--muted);font-weight:600}.result p{margin:0 0 12px}
.big{font-size:18px;font-weight:600}.tag{padding:2px 8px;border-radius:99px;font-size:12px;border:1px solid currentColor}
.err{color:var(--crit);white-space:pre-wrap;font-size:13px}
</style></head><body>
<header><h1>Pipeline MLOps · monitoring → analyse → orchestrateur → correction</h1>
<div class="badges" id="badges"></div></header>
<main>
<section class="card">
<label>Scénario</label><select id="scenario">
<option value="data_corruption">Corruption des données</option><option value="data_drift">Data drift</option>
<option value="concept_drift">Concept drift</option><option value="deployment_regression">Régression de déploiement</option>
<option value="normal">Normal (aucune panne)</option></select>
<label>Agent d'analyse</label><select id="analysis_mode">
<option value="fixture">Hors ligne (sans clé, corruption seulement)</option><option value="nvidia">Nemotron via NVIDIA (clé)</option>
<option value="local">Qwen local via Ollama</option></select>
<label>Orchestrateur</label><select id="orchestrator_mode"><option value="rules">Règles (sans clé)</option><option value="llm">LLM Groq (clé)</option></select>
<div class="row"><div><label>Graine</label><input id="seed" type="number" value="42"></div>
<div><label>Retard des labels</label><input id="label_delay" type="number" min="1" max="3" value="2"></div></div>
<div class="row"><div><label>Début de la panne</label><input id="onset" type="number" value="8"></div>
<div><label>Montée (fenêtres)</label><input id="ramp" type="number" value="4"></div></div>
<label><input id="predict" type="checkbox" checked style="width:auto"> Alerte prédictive</label>
<label>Vitesse de lecture (s par fenêtre)</label><input id="speed" type="number" step="0.05" min="0" max="2" value="0.25">
<button id="go">Lancer le parcours</button>
<div class="hint" id="hint"></div>
</section>
<section>
<div class="steps">
<div class="step" id="s-monitoring"><b>1. Monitoring</b>alertes</div>
<div class="step" id="s-analysis"><b>2. Analyse</b>cause</div>
<div class="step" id="s-orchestration"><b>3. Orchestrateur</b>décision</div>
<div class="step" id="s-correction"><b>4. Correction</b>ou revue humaine</div></div>
<div class="card"><svg id="chart" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
<div class="legend"><span><i style="background:var(--accent)"></i>log loss (échelle log)</span><span><i style="background:var(--pred)"></i>risque prédit (0–1)</span>
<span><i style="background:var(--muted)"></i>seuil</span><span><i style="background:var(--warn)"></i>alerte warning</span>
<span><i style="background:var(--crit)"></i>alerte critical</span><span><i style="background:var(--pred);border-radius:50%"></i>alerte prédictive</span></div></div>
<div class="grid2"><div class="card"><h3 style="margin:0 0 8px;font-size:14px">Journal en direct</h3><div class="log" id="log"></div></div>
<div class="card result" id="result"><h3>Résultat</h3><p class="k">Lancer un parcours pour voir le diagnostic, la décision et l'action.</p></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);let run=null,cursor=0,threshold=0.5412,W={},alerts=[],preds=[],nWindows=20;
const label={data_corruption:"corruption",data_drift:"data drift",concept_drift:"concept drift",deployment_regression:"régression de déploiement",no_incident:"pas d'incident",undetermined:"indéterminé"};
fetch('/api/config').then(r=>r.json()).then(c=>{threshold=c.threshold;
 $('badges').innerHTML=`<span class="${c.nvidia?'on':'off'}">clé NVIDIA ${c.nvidia?'configurée':'absente'}</span><span class="${c.groq?'on':'off'}">clé Groq ${c.groq?'configurée':'absente'}</span><span class="${c.predictor?'on':'off'}">prédicteur ${c.predictor?'prêt':'absent'}</span>`;draw();});
function log(t){const d=document.createElement('div');d.innerHTML=t;$('log').prepend(d);}
function setStep(s){const order=['monitoring','analysis','orchestration','correction'];const i=order.indexOf(s);
 order.forEach((n,j)=>{$('s-'+n).className='step'+(j<i||s==='done'?' done':j===i?' active':'');});}
function draw(){const svg=$('chart'),w=900,h=280,p=40,ix=i=>p+(w-2*p)*i/Math.max(nWindows-1,1);
 const ly=v=>{const lo=Math.log10(0.3),hi=Math.log10(40);return h-p-(h-2*p)*(Math.log10(Math.max(v,0.3))-lo)/(hi-lo)};
 const ry=v=>h-p-(h-2*p)*v;let s=`<line x1="${p}" y1="${ly(threshold)}" x2="${w-p}" y2="${ly(threshold)}" stroke="var(--muted)" stroke-dasharray="5 4"/>`;
 s+=`<text x="${w-p}" y="${ly(threshold)-5}" font-size="11" text-anchor="end" fill="var(--muted)">seuil ${threshold.toFixed(3)}</text>`;
 [0.5,1,5,20].forEach(v=>s+=`<text x="4" y="${ly(v)+4}" font-size="10" fill="var(--muted)">${v}</text>`);
 [0,0.5,1].forEach(v=>s+=`<text x="${w-4}" y="${ry(v)+4}" font-size="10" text-anchor="end" fill="var(--pred)">${v}</text>`);
 s+=`<text x="4" y="14" font-size="10" fill="var(--accent)">log loss</text><text x="${w-4}" y="14" font-size="10" text-anchor="end" fill="var(--pred)">risque</text>`;
 const keys=Object.keys(W).sort();const loss=keys.filter(k=>W[k].log_loss!=null),risk=keys.filter(k=>W[k].risk!=null);
 if(loss.length)s+=`<polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${loss.map(k=>ix(+k.slice(1))+','+ly(W[k].log_loss)).join(' ')}"/>`;
 if(risk.length)s+=`<polyline fill="none" stroke="var(--pred)" stroke-width="2" stroke-dasharray="6 3" points="${risk.map(k=>ix(+k.slice(1))+','+ry(W[k].risk)).join(' ')}"/>`;
 keys.forEach(k=>{const x=ix(+k.slice(1));s+=`<text x="${x}" y="${h-12}" font-size="10" text-anchor="middle" fill="var(--muted)">${k}</text>`;});
 alerts.forEach(a=>{const x=ix(+a.window_id.slice(1));s+=`<line x1="${x}" y1="${p-10}" x2="${x}" y2="${h-p}" stroke="${a.severity==='critical'?'var(--crit)':'var(--warn)'}" stroke-width="2"/>`;});
 preds.forEach(a=>{const x=ix(+a.window_id.slice(1));s+=`<circle cx="${x}" cy="${p-14}" r="7" fill="var(--pred)"/>`;});svg.innerHTML=s;}
function handle(e){const w=e.window||e.window_id;
 if(e.kind==='stage'){if(e.stage==='analysis')log('<b>→ Analyse</b> : le dossier d\'alerte est transmis à l\'agent');setStep(e.stage==='error'?'':e.stage);}
 else if(e.kind==='window'){W[w]=W[w]||{};}
 else if(e.kind==='metric'){(W[w]=W[w]||{}).log_loss=e.log_loss;log(`<span class="k">pas ${e.step}</span> labels de ${w} arrivés : log loss ${e.log_loss.toFixed(3)}`);}
 else if(e.kind==='risk'){(W[w]=W[w]||{}).risk=e.risk;}
 else if(e.kind==='alert'){alerts.push(e);log(`<b style="color:${e.severity==='critical'?'var(--crit)':'var(--warn)'}">ALERTE CONFIRMÉE ${e.severity}</b> sur ${e.window_id} (log loss ${e.observed_value.toFixed(3)}, pas ${e.detected_at_step})`);}
 else if(e.kind==='predictive'){preds.push(e);log(`<b style="color:var(--pred)">ALERTE PRÉDICTIVE</b> sur ${e.window_id} (risque ${e.risk.toFixed(2)}, pas ${e.step}) : ${e.signals.join(', ')}`);}
 else if(e.kind==='closed'){log(`<span class="k">incident ${w} clos (${e.reason||''})</span>`);}
 else if(e.kind==='episode'){log(`dossier d'alerte écrit : ${e.episode}`);}
 else if(e.kind==='tool'){log(e.event==='tool_started'?`<span class="k">${e.evidence_id}</span> outil <b>${e.tool}</b> ${e.arguments?JSON.stringify(e.arguments):''}`:`<span class="k">${e.evidence_id} ${e.tool} → ${e.status}</span>`);}
 else if(e.kind==='model'){log(`<span class="k">appel au modèle d'analyse n° ${e.call||''} ${e.reason||''}</span>`);}
 draw();}
function showResult(r){const res=r.result,a=res.analysis||{},d=a.diagnosis||{},c=a.confidence_assessment||{},o=res.orchestration||{},cor=res.correction;
 let h='<h3>Diagnostic de l\'agent d\'analyse</h3>';
 h+=a.status?`<p><span class="big">${label[d.cause_probable]||d.cause_probable||'—'}</span><br>${d.resume||''}<br><span class="k">preuves : ${(d.preuves||[]).join(', ')}</span></p>`:'<p class="k">Aucune analyse (pas d\'alerte confirmée).</p>';
 if(c.confidence!=null)h+=`<h3>Score de confiance</h3><p><span class="big">${c.confidence}</span> · type ${c.incident_type} · preuve ${c.supporting_tool||'aucune'}</p>`;
 if(o.decision)h+=`<h3>Orchestrateur</h3><p><span class="tag">${o.decision.route==='correction'?'correction':'revue humaine'}</span> ${o.decision.reason||''}</p>`;
 if(cor)h+=`<h3>Agent de correction</h3><p><span class="big">${cor.final_plan.action}</span> (${cor.tool_result.status})<br><span class="k">production inchangée : ${cor.verification.production_unchanged}</span></p>`;
 else if(res.human_review)h+=`<h3>Revue humaine</h3><p>Demande enregistrée (${res.human_review.status||'en attente'}) : un humain doit valider l'action.</p>`;
 h+=`<h3>Statut final</h3><p><span class="tag">${res.status}</span><br><span class="k">${res.result_file||''}</span></p>`;
 $('result').innerHTML=h;setStep(cor?'done':(o.decision?'correction':'orchestration'));if(cor||res.human_review)['monitoring','analysis','orchestration','correction'].forEach(n=>$('s-'+n).className='step done');}
async function poll(){const r=await (await fetch(`/api/runs/${run}?after=${cursor}`)).json();r.events.forEach(handle);cursor=r.next;
 if(r.status==='running')return setTimeout(poll,400);$('go').disabled=false;
 if(r.status==='error'){$('result').innerHTML=`<h3>Erreur</h3><p class="err">${r.error}</p><p class="k">Vérifier la clé dans .env et la connexion.</p>`;log('<b style="color:var(--crit)">Erreur</b> '+r.error);}
 else{showResult(r);log('<b>Parcours terminé</b> : '+r.result.status);}}
$('go').onclick=async()=>{W={};alerts=[];preds=[];cursor=0;$('log').innerHTML='';nWindows=20;draw();
 const body={scenario:$('scenario').value,analysis_mode:$('analysis_mode').value,orchestrator_mode:$('orchestrator_mode').value,
  seed:+$('seed').value,label_delay:+$('label_delay').value,onset:+$('onset').value,ramp:+$('ramp').value,predict:$('predict').checked,speed:+$('speed').value};
 $('result').innerHTML='<h3>Résultat</h3><p class="k">Parcours en cours…</p>';
 const r=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();
 if(!r.ok){$('hint').textContent=j.error;return;}$('hint').textContent=body.analysis_mode==='fixture'?'':'L\'analyse par LLM peut prendre plusieurs minutes.';
 run=j.id;$('go').disabled=true;setStep('monitoring');poll();};
</script></body></html>"""


if __name__ == "__main__":
    print("Interface : http://127.0.0.1:8050  (Ctrl+C pour arrêter)")
    app.run(host="127.0.0.1", port=8050, debug=False, threaded=True)
