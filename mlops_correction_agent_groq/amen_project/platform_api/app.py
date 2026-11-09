"""Run with .venv-cloud/Scripts/python -m uvicorn platform_api.app:app."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('SENTINEL_DATA_DIR', ROOT / 'platform_data'))
DATA.mkdir(parents=True, exist_ok=True)
engine = create_engine('sqlite:///' + (DATA / 'monitoring.sqlite3').as_posix(), connect_args={'check_same_thread': False, 'timeout': 30})
SCENARIOS = {
    'data_drift': ('episode-001', 'Dérive des données', 'compare_distributions', {'left': 'reference_clients', 'right': 'source', 'columns': ['monthly_charges', 'tenure_months']}, 'Examiner la population et adapter les données de référence.'),
    'concept_drift': ('episode-002', 'Dérive du concept', 'compare_conditional_relationship', {}, 'Préparer un réentraînement sur des observations récentes.'),
    'data_corruption': ('episode-003', 'Corruption des données', 'compare_tables', {'left': 'source', 'right': 'inputs', 'columns': ['monthly_charges']}, 'Corriger la transformation de monthly_charges et contrôler les unités.'),
    'deployment_regression': ('episode-004', 'Régression de déploiement', 'compare_model_versions', {'bootstrap_samples': 100}, 'Préparer un retour à la version de référence du modèle.'),
    'false_alert': ('episode-005', 'Fausse alerte', 'compare_windows', {}, 'Clôturer sans modifier le modèle et maintenir la surveillance.'),
}
REFERENCE = ROOT / 'scenario_data/demo'
calibration = json.loads((REFERENCE / 'calibration.json').read_text(encoding='utf-8'))
tasks: set[asyncio.Task] = set()
start_lock = asyncio.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, body TEXT NOT NULL)'))
        conn.execute(text('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT, at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)'))
        conn.execute(text('CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT, at TEXT NOT NULL, body TEXT NOT NULL)'))
    for incident in incidents():
        if incident['status'] not in ('completed', 'failed', 'interrupted'):
            update(incident['id'], status='interrupted', error='Exécution interrompue par le redémarrage du serveur.', ended_at=now())
            emit(incident['id'], 'interrupted', {'message': 'Le serveur a redémarré ; relancez un scénario.'})


def incidents():
    with engine.connect() as conn:
        return [json.loads(row[0]) for row in conn.execute(text('SELECT body FROM incidents ORDER BY rowid DESC'))]


def get_incident(incident_id):
    with engine.connect() as conn:
        body = conn.execute(text('SELECT body FROM incidents WHERE id=:id'), {'id': incident_id}).scalar()
    if body is None:
        raise HTTPException(404, 'Incident introuvable')
    return json.loads(body)


def update(incident_id, **changes):
    # Updates are serialized by the event loop; tool threads only append events.
    body = get_incident(incident_id)
    body.update(changes)
    with engine.begin() as conn:
        conn.execute(text('UPDATE incidents SET body=:body WHERE id=:id'), {'id': incident_id, 'body': json.dumps(body, ensure_ascii=False)})
    return body


def emit(incident_id, kind, payload):
    with engine.begin() as conn:
        conn.execute(text('INSERT INTO events(incident_id,at,kind,payload) VALUES (:incident,:at,:kind,:payload)'), {'incident': incident_id, 'at': now(), 'kind': kind, 'payload': json.dumps(payload, ensure_ascii=False, default=str)})


def events_after(after=0):
    with engine.connect() as conn:
        rows = conn.execute(text('SELECT id,incident_id,at,kind,payload FROM events WHERE id>:after ORDER BY id LIMIT 200'), {'after': after}).all()
    return [dict(id=r[0], incident_id=r[1], at=r[2], kind=r[3], payload=json.loads(r[4])) for r in rows]


def add_metric(incident_id, metrics, window, labeled_rows, source='replay'):
    body = dict(metrics=metrics, window=window, labeled_rows=labeled_rows, source=source)
    with engine.begin() as conn:
        conn.execute(text('INSERT INTO metrics(incident_id,at,body) VALUES (:incident,:at,:body)'), {'incident': incident_id, 'at': now(), 'body': json.dumps(body)})
    emit(incident_id, 'metric_recorded', body)


def snapshot():
    with engine.connect() as conn:
        rows = conn.execute(text('SELECT id,incident_id,at,body FROM metrics ORDER BY id DESC LIMIT 180')).all()
        cursor = conn.execute(text('SELECT COALESCE(MAX(id),0) FROM events')).scalar()
    return dict(incidents=incidents(), metrics=[dict(id=r[0], incident_id=r[1], at=r[2], **json.loads(r[3])) for r in reversed(rows)], cursor=cursor,
                reference=calibration['reference_value'], threshold=calibration['threshold'],
                scenarios=[dict(id=k, label=v[1]) for k, v in SCENARIOS.items()])


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title='Sentinel · MLOps', lifespan=lifespan)


class Launch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scenario: str = 'deployment_regression'


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/state')
def state():
    return snapshot()


@app.get('/api/incidents/{incident_id}')
def detail(incident_id: str):
    incident = get_incident(incident_id)
    with engine.connect() as conn:
        rows = conn.execute(text('SELECT id,at,kind,payload FROM events WHERE incident_id=:id ORDER BY id'), {'id': incident_id}).all()
    return dict(**incident, events=[dict(id=r[0], at=r[1], kind=r[2], payload=json.loads(r[3])) for r in rows])


@app.post('/api/incidents', status_code=202)
async def launch(body: Launch):
    if body.scenario not in SCENARIOS:
        raise HTTPException(422, 'Scénario inconnu')
    async with start_lock:
        if any(i['status'] not in ('completed', 'failed', 'interrupted') for i in incidents()):
            raise HTTPException(409, 'Une investigation est déjà en cours.')
        incident_id = uuid.uuid4().hex[:12]
        episode = REFERENCE / 'episodes' / SCENARIOS[body.scenario][0]
        alert = json.loads((episode / 'alert.json').read_text(encoding='utf-8'))
        context = json.loads((episode / 'context.json').read_text(encoding='utf-8'))
        record = dict(id=incident_id, scenario=body.scenario, title=SCENARIOS[body.scenario][1], mode='nvidia', status='monitoring', phase=0, created_at=now(), alert=alert, deployment=context['deployment'], diagnosis=None, correction=None, validation=None, error=None)
        with engine.begin() as conn:
            conn.execute(text('INSERT INTO incidents(id,body) VALUES (:id,:body)'), {'id': incident_id, 'body': json.dumps(record)})
        emit(incident_id, 'replay_started', {'message': 'Relecture accélérée de fenêtres synthétiques étiquetées.'})
        task = asyncio.create_task(run_incident(incident_id))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    return record


async def run_incident(incident_id):
    record = get_incident(incident_id)
    spec = SCENARIOS[record['scenario']]
    episode = REFERENCE / 'episodes' / spec[0]
    try:
        # Real measured calibration points, replayed at the current presentation time.
        for index, metric in enumerate(calibration['calibration_metrics'][-8:]):
            add_metric(incident_id, metric, f'calibration-{index+1}', 500)
            await asyncio.sleep(.35)
        for window in ('focus', 'confirmation-1', 'confirmation-2'):
            monitoring = json.loads((episode / window / 'monitoring.json').read_text(encoding='utf-8'))
            add_metric(incident_id, monitoring['metrics'], window, monitoring['labeled_rows'])
            if window == 'focus':
                update(incident_id, status='investigating', phase=1)
                emit(incident_id, 'alert_triggered', record['alert'])
            await asyncio.sleep(.8)
        from mlops_churn.agent import investigate
        loop = asyncio.get_running_loop()

        def on_event(event):
            # Thread-safe durable delivery, including parallel analytical tools.
            emit(incident_id, event['event'], {k: v for k, v in event.items() if k not in ('event', 'result')} if event['event'] == 'investigation_completed' else {k: v for k, v in event.items() if k != 'event'})
        options = dict(episode_dir=episode, reference_dir=REFERENCE, output_dir=ROOT / 'results', provider='nvidia', on_event=on_event)
        result = await loop.run_in_executor(None, lambda: investigate(**options))
        # Keep evidence, usage and raw output even when the provider fails to conclude.
        update(incident_id, agent_result=result)
        if result['status'] != 'completed':
            raise RuntimeError((result.get('error') or {}).get('message') or (
                'Le modèle a terminé sans diagnostic JSON exploitable. '
                f"{result['tool_calls']} analyses conservées ; "
                'consultez la réponse finale et les traces de l’investigation.'))
        update(incident_id, phase=2, status='diagnosed', diagnosis=result['final_diagnosis'], agent_result=result)
        emit(incident_id, 'diagnosis_ready', result['final_diagnosis'])
        await asyncio.sleep(1.5)
        cause = result['final_diagnosis'].get('cause_probable')
        action = SCENARIOS[cause][4] if cause in SCENARIOS else (SCENARIOS['false_alert'][4] if cause in ('no_incident', 'false_alert') else 'Diagnostic à examiner : aucune correction automatique proposée.')
        correction = dict(simulated=True, action=action, applied=False, message='Présentation du plan uniquement. Aucune donnée ni version de modèle modifiée.')
        update(incident_id, phase=3, status='correcting', correction=correction)
        emit(incident_id, 'correction_started', correction)
        await asyncio.sleep(2)
        validation = dict(simulated=True, passed=None, before=record['alert']['observed_value'], after=None, message='Validation à réaliser par le futur agent de correction. Aucun gain mesuré.')
        update(incident_id, phase=4, status='validating', validation=validation)
        emit(incident_id, 'validation_started', validation)
        await asyncio.sleep(1.5)
        update(incident_id, phase=5, status='completed', ended_at=now(), outcome='closed_without_action' if cause in ('no_incident', 'false_alert') else 'awaiting_real_correction')
        emit(incident_id, 'report_ready', {'message': 'Rapport disponible. Parcours de présentation terminé.'})
    except asyncio.CancelledError:
        update(incident_id, status='interrupted', ended_at=now(), error='Serveur arrêté pendant le traitement.')
        raise
    except Exception as exc:
        from mlops_churn.models import redact
        message = redact(exc)
        update(incident_id, status='failed', error=message, ended_at=now())
        emit(incident_id, 'failed', {'message': message})


@app.websocket('/api/events')
async def stream(socket: WebSocket, after: int = 0):
    await socket.accept()
    try:
        while True:
            batch = events_after(after)
            if batch:
                await socket.send_json({'events': batch})
                after = batch[-1]['id']
            else:
                await socket.send_json({'events': [], 'heartbeat': now()})
            await asyncio.sleep(.5)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get('/api/incidents/{incident_id}/report')
def report(incident_id: str):
    item = detail(incident_id)
    if item['status'] != 'completed':
        raise HTTPException(409, 'Le rapport final sera disponible en fin de parcours.')
    diagnosis = item.get('diagnosis') or {}
    contents = '\n'.join([
        '# Rapport Sentinel', '', f"Incident : {item['id']}", f"Scénario : {item['title']}",
        f"Diagnostic : {'scénarisé' if item['mode'] == 'demo' else 'agent ' + item['mode']}",
        'Données : fenêtres synthétiques étiquetées rejouées.', '', '## Alerte',
        f"Log loss : {item['alert']['observed_value']:.4f} ; seuil : {item['alert']['threshold']:.4f}",
        '', '## Diagnostic', json.dumps(diagnosis, ensure_ascii=False, indent=2),
        '', '## Correction simulée', (item.get('correction') or {}).get('action', ''),
        'Aucune modification appliquée au modèle ou aux données.', '', '## Validation',
        'Non réalisée. Aucun gain de performance mesuré.', '', '## Événements',
        *[f"- {e['at']} — {e['kind']}" for e in item['events']]])
    return Response(contents, media_type='text/markdown', headers={'Content-Disposition': f'attachment; filename="sentinel-{incident_id}.md"'})


# Built React app is served by the same local API, with no separate Node server.
dist = ROOT / 'frontend/dist'
if dist.exists():
    app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')

    @app.get('/')
    def index():
        return FileResponse(dist / 'index.html')
