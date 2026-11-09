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
from mlops_churn.agent import build_memory, final_json_syntax_error, parse_final
from mlops_churn.workspace import ROOT


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        assert 'submit_conclusion' not in [t.name for t in tools]
        return self.bind(**kwargs)


class AgentTests(unittest.TestCase):
    def test_malformed_json_is_reformatted_with_evidence_and_cause_preserved(self):
        malformed = 'Diagnostic :\n{\n"cause_probable":"data_drift",\n"preuves":["E0001"],\n"limites":["Lot limité." }\n}'
        corrected = '{"cause_probable":"data_drift","resume":"Population différente.","preuves":["E0001"]}'
        events = []
        result = self.run_agent([
            AIMessage(content='', tool_calls=[{'name':'evaluate_predictions','args':{},'id':'c1'}]),
            AIMessage(content=malformed), AIMessage(content=corrected)], on_event=events.append)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['final_diagnosis']['cause_probable'], 'data_drift')
        self.assertNotIn('limites',result['final_diagnosis'])
        self.assertEqual(set(result['final_diagnosis']),{'cause_probable','resume','preuves'})
        self.assertEqual(result['raw_final_text'], corrected)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['json_format_retries'], 1)
        self.assertEqual(result['empty_response_retries'], 0)
        retries = [e for e in events if e['event'] == 'model_retry']
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]['reason'], 'invalid_json')
        self.assertGreater(retries[0]['format_error']['line'], 1)
        trace = json.loads((Path(result['run_directory'])/'trace.json').read_text(encoding='utf-8'))
        self.assertTrue(any(t['message']['data'].get('content') == malformed for t in trace))
        prompts = [t['message']['data']['content'] for t in trace if t['message']['type']=='human']
        self.assertTrue(any('Conserve la même cause_probable' in text for text in prompts))
        self.assertTrue(any('Supprime le champ limites' in text for text in prompts))

    def test_malformed_json_retries_are_bounded_and_raw_output_is_kept(self):
        malformed = '{"cause_probable":"concept_drift","limites":["Limite"}}'
        result = self.run_agent([AIMessage(content=malformed) for _ in range(3)])
        self.assertEqual(result['status'], 'failed_or_incomplete')
        self.assertEqual(result['json_format_retries'], 2)
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['raw_final_text'], malformed)
        self.assertIn('Diagnostic JSON invalide', result['error']['message'])
        self.assertIsNone(result['final_diagnosis'])

    def test_json_format_retry_respects_budget_and_can_be_disabled(self):
        malformed = '{"cause_probable":"concept_drift","limites":["Limite"}}'
        for options in ({'max_model_calls':1}, {'max_json_format_retries':0}):
            with self.subTest(options=options):
                result = self.run_agent([AIMessage(content=malformed)], options=options)
                self.assertEqual(result['json_format_retries'], 0)
                self.assertEqual(result['model_calls'], 1)
                self.assertEqual(result['raw_final_text'], malformed)
                self.assertIn('Diagnostic JSON invalide', result['error']['message'])

    def test_format_recovery_does_not_judge_cause(self):
        for cause in ('unexpected_cause', 'undetermined', 'no_incident'):
            result = self.run_agent([AIMessage(content=json.dumps({'cause_probable':cause}))])
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['json_format_retries'], 0)
            self.assertEqual(result['model_calls'], 1)
    def test_prose_without_diagnosis_retries_then_stops_with_explicit_error(self):
        result = self.run_agent([AIMessage(content='Je vais comparer les modèles.') for _ in range(3)])
        self.assertEqual(result['status'], 'failed_or_incomplete')
        self.assertEqual(result['json_format_retries'], 2)
        self.assertEqual(result['model_calls'], 3)
        self.assertIn('Diagnostic JSON absent', result['error']['message'])

    def test_prose_resumes_native_tools_and_finishes_with_three_fields(self):
        events = []
        diagnosis = {'cause_probable':'data_drift','resume':'Population différente.','preuves':['E0001']}
        result = self.run_agent([
            AIMessage(content='Je lance les comparaisons.'),
            AIMessage(content='',tool_calls=[{'name':'evaluate_predictions','args':{},'id':'c1'}]),
            AIMessage(content=json.dumps(diagnosis))], on_event=events.append)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['final_diagnosis'], diagnosis)
        self.assertNotIn('limites',result['final_diagnosis'])
        self.assertEqual(result['tool_calls'],1)
        self.assertEqual(result['json_format_retries'],1)
        self.assertEqual(next(e for e in events if e['event']=='model_retry')['reason'],'missing_json')
        prompt = json.loads((Path(result['run_directory'])/'runtime.json').read_text(encoding='utf-8'))['prompt']
        self.assertIn('exactement trois champs',prompt)

    def test_missing_diagnosis_recovery_does_not_exceed_budget(self):
        result = self.run_agent([AIMessage(content='Je vais comparer les modèles.')],options={'max_model_calls':1})
        self.assertEqual(result['model_calls'],1)
        self.assertEqual(result['json_format_retries'],0)
        self.assertIn('Diagnostic JSON absent',result['error']['message'])

    def test_syntax_error_detects_fenced_diagnosis_without_repairing_it(self):
        malformed = '```json\n{"cause_probable":"data_drift","limites":["Limite"}}\n```'
        self.assertIsNotNone(final_json_syntax_error(malformed))
        self.assertIsNone(parse_final(malformed))
        self.assertIsNone(final_json_syntax_error('Je vais analyser les données.'))
        self.assertIsNone(final_json_syntax_error('{"resume":"sans cause"}'))
        self.assertIsNone(final_json_syntax_error('{"cause_probable":"undetermined"}'))

    def test_reasoning_only_response_retries_with_evidence_preserved(self):
        events = []
        result = self.run_agent([
            AIMessage(content='', tool_calls=[{'name': 'evaluate_predictions', 'args': {}, 'id': 'c1'}]),
            AIMessage(content='', additional_kwargs={'reasoning_content': 'Internal deliberation.'}),
            AIMessage(content='', tool_calls=[{'name': 'inspect_deployment', 'args': {}, 'id': 'c2'}]),
            AIMessage(content='{"cause_probable":"undetermined","preuves":["E0001","E0002"]}')],
            on_event=events.append)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['tool_calls'], 2)
        self.assertEqual(result['empty_response_retries'], 1)
        self.assertEqual(result['evidence'][0]['evidence_id'], 'E0001')
        self.assertTrue(any(e['event'] == 'model_retry' for e in events))

    def test_repeated_empty_responses_stop_with_explicit_error(self):
        result = self.run_agent([AIMessage(content='') for _ in range(3)])
        self.assertEqual(result['status'], 'failed_or_incomplete')
        self.assertEqual(result['model_calls'], 3)
        self.assertEqual(result['empty_response_retries'], 2)
        self.assertIn('réponse vide', result['error']['message'])
        self.assertIsNone(result['final_diagnosis'])

    def test_empty_response_does_not_exceed_model_budget(self):
        result = self.run_agent([AIMessage(content='')], options={'max_model_calls': 1})
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(result['empty_response_retries'], 0)
        self.assertIn('réponse vide', result['error']['message'])

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
