"""Offline coordination tests. Scripted answers do not evaluate LLM reasoning."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from mlops_churn import investigate
from mlops_churn.agent import build_memory, parse_final
from mlops_churn.workspace import ROOT


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        assert 'submit_conclusion' not in [t.name for t in tools]
        return self.bind(**kwargs)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_agent(self, responses, episode='episode-004', **kwargs):
        return investigate(ROOT / 'scenario_data/demo/episodes' / episode,
            output_dir=self.root, model=ToolModel(responses=responses), **kwargs)

    def test_full_tool_cycle_for_all_five_incidents(self):
        cases = [
            ('episode-001', 'data_drift', 'compare_distributions',
             {'left': 'reference_clients', 'right': 'source', 'columns': ['monthly_charges']}),
            ('episode-002', 'concept_drift', 'compare_conditional_relationship', {}),
            ('episode-003', 'data_corruption', 'compare_tables',
             {'left': 'source', 'right': 'inputs', 'columns': ['monthly_charges']}),
            ('episode-004', 'deployment_regression', 'compare_model_versions', {'bootstrap_samples': 100}),
            ('episode-005', 'no_incident', 'compare_windows', {})]
        for episode, cause, method, arguments in cases:
            with self.subTest(episode=episode):
                final = json.dumps({'cause_probable': cause, 'preuves': ['E0001']})
                events = []
                result = self.run_agent([
                    AIMessage(content='', tool_calls=[{'name': method, 'args': arguments, 'id': 'c1'}],
                              usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}),
                    AIMessage(content=final,
                              usage_metadata={'input_tokens': 20, 'output_tokens': 5, 'total_tokens': 25})],
                    episode, on_event=events.append)
                self.assertEqual(result['status'], 'completed', result['error'])
                self.assertEqual(result['raw_final_text'], final)
                self.assertEqual(result['evidence'][0]['result']['status'], 'ok', result['evidence'])
                self.assertEqual(result['model_calls'], 2)
                self.assertEqual(result['remaining_model_calls'], 10)
                self.assertEqual(result['remaining_tool_calls'], 15)
                self.assertEqual(result['usage']['total_tokens'], 40)
                self.assertTrue(result['originals_preserved'])
                self.assertEqual(events[-1]['event'], 'investigation_completed')
                run = Path(result['run_directory'])
                self.assertEqual(json.loads((run/'result.json').read_text(encoding='utf-8'))['status'], 'completed')
                self.assertFalse((run/'conclusion_control.json').exists())
                public = (run/'initial_observations.json').read_text(encoding='utf-8')
                self.assertNotIn('expected_cause', public)
                self.assertNotIn('selected_seed', public)

    def test_external_incident_without_any_simulation_truth(self):
        ref = self.root / 'reference'
        (ref/'models').mkdir(parents=True)
        shutil.copyfile(ROOT/'scenario_data/demo/reference_data.csv', ref/'reference_data.csv')
        shutil.copyfile(ROOT/'scenario_data/demo/models/reference.joblib', ref/'models/reference.joblib')
        episode = self.root / 'incident-from-another-agent'
        shutil.copytree(ROOT/'scenario_data/demo/episodes/episode-004', episode)
        final = '{"cause_probable":"undetermined","limites":["Test de coordination"]}'
        with patch('mlops_churn.agent.make_model', side_effect=AssertionError('Aucun réseau')):
            result = investigate(episode, ref, model=ToolModel(responses=[AIMessage(content=final)]),
                                 output_dir=self.root/'output', provider='local')
        self.assertEqual(result['status'], 'completed')
        self.assertNotIn('expected_cause_evaluator_only', result)

    def test_truncated_and_non_json_final_are_incomplete(self):
        for message in [AIMessage(content='incomplet', response_metadata={'finish_reason':'length'}),
                        AIMessage(content='Je vais comparer les modèles')]:
            result = self.run_agent([message])
            self.assertEqual(result['status'], 'failed_or_incomplete')

    def test_model_budget_stops_without_inventing_final_diagnosis(self):
        result = self.run_agent([AIMessage(content='', tool_calls=[
            {'name':'compare_model_versions','args':{'bootstrap_samples':100},'id':'c1'}])],
            options={'max_model_calls':1})
        self.assertEqual(result['status'], 'failed_or_incomplete')
        self.assertIsNone(result['final_diagnosis'])
        self.assertEqual(result['remaining_model_calls'], 0)

    def test_error_is_not_a_measurement_in_memory(self):
        result = build_memory([{'evidence_id':'E0001','tool':'compare_tables','arguments':{},
                               'result':{'status':'error','error':'Ressource inconnue'}}])
        self.assertEqual(result['successful_evidence'], [])
        self.assertFalse(result['failed_or_cached_operations'][0]['analysis_completed'])
        self.assertNotIn('cause_probable', str(result))

    def test_final_parser_accepts_json_with_trailing_explanation(self):
        self.assertEqual(parse_final('```json\n{"cause_probable":"no_incident"}\n```\nExplication')['cause_probable'], 'no_incident')
        self.assertIsNone(parse_final('Une hypothèse sans objet JSON'))
