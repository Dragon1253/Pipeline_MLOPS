"""Offline platform integration checks: persistence, stream, real tools, simulated correction."""
import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from mlops_churn import investigate as real_investigate

_storage = tempfile.TemporaryDirectory()
os.environ['SENTINEL_DATA_DIR'] = _storage.name
from platform_api import app as platform
os.environ.pop('SENTINEL_DATA_DIR', None)


def tearDownModule():
    platform.engine.dispose()
    _storage.cleanup()


class PlatformTests(unittest.TestCase):
    def test_end_to_end_all_scenarios_and_durable_events(self):
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(0.001)

        def offline_investigation(**options):
            # The platform must select NVIDIA; fake responses exist only in this test.
            self.assertEqual(options['provider'], 'nvidia')
            self.assertNotIn('model', options)
            scenario, spec = next((key, value) for key, value in platform.SCENARIOS.items()
                                  if value[0] == options['episode_dir'].name)

            class ScriptedModel(FakeMessagesListChatModel):
                def bind_tools(self, tools, **kwargs):
                    return self.bind(**kwargs)

            import json
            diagnosis = {'cause_probable': 'no_incident' if scenario == 'false_alert' else scenario,
                         'resume': 'Diagnostic simulé uniquement pour les tests.', 'preuves': ['E0001','E0002']}
            return real_investigate(**options, model=ScriptedModel(responses=[
                AIMessage(content='', tool_calls=[dict(name='evaluate_predictions', args={}, id='test1')]),
                AIMessage(content='', tool_calls=[dict(name=spec[2], args=spec[3], id='test2')]),
                AIMessage(content=json.dumps(diagnosis))]))

        with TestClient(platform.app) as client, patch.object(platform.asyncio, 'sleep', fast_sleep), patch('time.sleep', lambda _: None), patch('mlops_churn.agent.investigate', side_effect=offline_investigation):
            self.assertEqual(client.get('/api/health').status_code, 200)
            self.assertEqual(client.post('/api/incidents', json={'scenario': 'unknown'}).status_code, 422)
            for mode in ('demo','local','nvidia'):
                self.assertEqual(client.post('/api/incidents', json={'mode':mode}).status_code,422)
            self.assertEqual(client.get('/api/incidents/missing').status_code, 404)
            for scenario in platform.SCENARIOS:
                with self.subTest(scenario=scenario):
                    response = client.post('/api/incidents', json={'scenario': scenario})
                    self.assertEqual(response.status_code, 202, response.text)
                    self.assertEqual(response.json()['mode'],'nvidia')
                    incident_id = response.json()['id']
                    self.assertEqual(client.get(f'/api/incidents/{incident_id}/report').status_code, 409)
                    self.assertEqual(client.post('/api/incidents', json={'scenario': scenario}).status_code, 409)
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        item = client.get(f'/api/incidents/{incident_id}').json()
                        if item['status'] in ('completed', 'failed'):
                            break
                        # Unpatched event-based wait lets the worker run.
                        import threading
                        threading.Event().wait(.03)
                    self.assertEqual(item['status'], 'completed', item.get('error'))
                    self.assertEqual(item['phase'], 5)
                    self.assertEqual(item['agent_result']['tool_calls'], 2)
                    self.assertTrue(item['agent_result']['originals_preserved'])
                    self.assertTrue(all(e['result']['status'] == 'ok' for e in item['agent_result']['evidence']), item['agent_result']['evidence'])
                    self.assertFalse(item['correction']['applied'])
                    self.assertIsNone(item['validation']['after'])
                    kinds = [e['kind'] for e in item['events']]
                    self.assertIn('tool_started', kinds)
                    self.assertIn('tool_completed', kinds)
                    self.assertEqual(kinds[-1], 'report_ready')
                    report = client.get(f'/api/incidents/{incident_id}/report')
                    self.assertEqual(report.status_code, 200)
                    self.assertIn('Aucun gain de performance', report.text)
            snapshot = client.get('/api/state').json()
            self.assertEqual(len(snapshot['incidents']), 5)
            self.assertEqual(len(snapshot['metrics']), 55)
            with client.websocket_connect('/api/events?after=0') as socket:
                batch = socket.receive_json()['events']
                self.assertTrue(batch)
                self.assertEqual(batch[0]['kind'], 'replay_started')
            platform.initialize()
            self.assertTrue(all(i['status'] == 'completed' for i in platform.incidents()))

    def test_failed_investigation_preserves_result_and_does_not_correct(self):
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(.001)

        failure = {'status': 'failed_or_incomplete', 'error': {'message': 'Réponse vide du modèle.'},
                   'model_calls': 3, 'tool_calls': 10, 'raw_final_text': '',
                   'final_diagnosis': None, 'evidence': [], 'empty_response_retries': 2}
        with TestClient(platform.app) as client, patch.object(platform.asyncio, 'sleep', fast_sleep), patch('mlops_churn.agent.investigate', return_value=failure):
            response = client.post('/api/incidents', json={'scenario': 'data_drift'})
            self.assertEqual(response.status_code, 202)
            incident_id = response.json()['id']
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                item = client.get(f'/api/incidents/{incident_id}').json()
                if item['status'] == 'failed':
                    break
                import threading
                threading.Event().wait(.03)
            self.assertEqual(item['status'], 'failed')
            self.assertEqual(item['agent_result'], failure)
            self.assertEqual(item['error'], 'Réponse vide du modèle.')
            self.assertIsNone(item['correction'])
            self.assertIsNone(item['diagnosis'])

    def test_restart_marks_unfinished_incident_interrupted(self):
        with platform.engine.begin() as conn:
            import json
            conn.execute(platform.text('INSERT INTO incidents(id,body) VALUES (:id,:body)'), {'id': 'restart-test', 'body': json.dumps({'id': 'restart-test', 'status': 'investigating'})})
        platform.initialize()
        self.assertEqual(platform.get_incident('restart-test')['status'], 'interrupted')
