"""Tests hors réseau : transport officiel, schémas et compteurs."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from mlops_churn.models import make_nvidia_langchain_model
from mlops_churn.models import make_model, compatible_schema, response_truncated
from langchain_core.messages import AIMessage, ToolMessage


try:
    import langchain_nvidia_ai_endpoints
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False


@unittest.skipUnless(HAS_NVIDIA, 'Connecteur NVIDIA absent de cet environnement local')
class NvidiaConnectorTests(unittest.TestCase):
    def test_payload_and_native_decision_are_preserved_without_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi-offline-test'}):
            model = make_nvidia_langchain_model({'model': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
                'request_interval_seconds': 0, 'max_model_calls': 1}, tmp)
            bound = model.bind_tools([{'type': 'function', 'function': {'name': 'probe',
                'description': 'Probe', 'parameters': {'type': 'object', 'properties': {
                    'limit': {'type': 'integer', 'enum': [1, 2]}}, 'required': ['limit']}}}])
            payload = model._get_payload([{'role': 'user', 'content': 'test'}], stop=None, **bound.kwargs)
            self.assertTrue(payload['chat_template_kwargs']['enable_thinking'])
            self.assertTrue(payload['parallel_tool_calls'])
            self.assertEqual(payload['max_tokens'], 16384)
            self.assertEqual(payload['tools'][0]['function']['parameters']['properties']['limit']['enum'], [1, 2])
            chunk = ChatGenerationChunk(message=AIMessageChunk(content='',
                tool_call_chunks=[{'name': 'probe', 'args': '{"limit":1}', 'id': 'call-1', 'index': 0}],
                response_metadata={'finish_reason': 'tool_calls'},
                usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}))
            with patch('langchain_nvidia_ai_endpoints.ChatNVIDIA._stream', return_value=iter([chunk])):
                response = bound.invoke('test')
            self.assertEqual(response.tool_calls[0]['args'], {'limit': 1})
            self.assertEqual(response.usage_metadata['total_tokens'], 15)
            with self.assertRaisesRegex(RuntimeError, 'Plafond'):
                bound.invoke('test')
            for path in Path(tmp).glob('*.json'):
                self.assertNotIn('nvapi-offline-test', path.read_text(encoding='utf-8'))


class LocalConnectorTests(unittest.TestCase):
    def test_nullable_schema_preserves_enum_and_does_not_mutate_original(self):
        original = {'anyOf': [{'type': 'integer', 'enum': [1, 2]}, {'type': 'null'}], 'default': None}
        self.assertEqual(compatible_schema(original), {'type': 'integer', 'enum': [1, 2]})
        self.assertIn('anyOf', original)
        with self.assertRaises(ValueError):
            compatible_schema({'anyOf': [{'type': 'string'}, {'type': 'integer'}]})

    def test_native_tool_result_is_linked_to_its_name_without_network(self):
        model = make_model({'provider': 'ollama', 'model': 'qwen3:8b-q4_K_M',
            'host': 'http://localhost:11434', 'num_ctx': 16384, 'num_predict': 1200,
            'timeout_seconds': 600, 'tool_protocol': 'native_compatible'})
        messages = [AIMessage(content='', tool_calls=[{'name':'compare_tables','args':{},'id':'c1'}]),
                    ToolMessage(content='preuve', tool_call_id='c1')]
        params = model._chat_params(messages)
        self.assertEqual(params['messages'][-1]['tool_name'], 'compare_tables')
        self.assertEqual(params['options']['num_ctx'], 16384)

    def test_truncation_is_recognized_for_both_providers(self):
        for metadata in [{'done_reason':'length'}, {'finish_reason':'length'}]:
            self.assertTrue(response_truncated(AIMessage(content='', response_metadata=metadata)))



