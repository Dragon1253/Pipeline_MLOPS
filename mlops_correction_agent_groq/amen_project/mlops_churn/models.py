"""Connexion à Qwen local ou NVIDIA, clé locale et audit des échanges."""
import os
import re
from pathlib import Path
from copy import deepcopy
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from ollama._types import ChatRequest
from ollama._client import _copy_messages, _copy_tools
from pydantic import PrivateAttr
from .workspace import write_json

def load_local_key():
    if os.environ.get('NVIDIA_API_KEY'):
        return
    path = Path(__file__).resolve().parents[1] / '.env'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.strip().partition('=')
            if sep and name.strip() == 'NVIDIA_API_KEY':
                value = value.strip().strip('"').strip("'")
                if value and value != 'nvapi-REMPLACER_ICI':
                    os.environ['NVIDIA_API_KEY'] = value
                    return
    raise ValueError('Configurer NVIDIA_API_KEY dans le fichier .env ou dans l’environnement.')


def redact(error):
    text = str(error)
    if os.environ.get('NVIDIA_API_KEY'):
        text = text.replace(os.environ['NVIDIA_API_KEY'], '[CLÉ MASQUÉE]')
    return re.sub(r'nvapi-[A-Za-z0-9_-]+', '[CLÉ MASQUÉE]', text)


def validate_model_mode(metadata, config):
    finetune = str(metadata.get('model_info', {}).get('general.finetune', '')).lower()
    if 'thinking' in finetune and config.get('reasoning') is False:
        raise ValueError('Ce checkpoint est une variante Thinking. think=false ne fournit pas un mode Instruct supporté. '
                         'Choisir un tag Instruct ou hybride explicite, ou prévoir un budget de raisonnement adapté.')


def compatible_schema(value):
    """Le client Ollama 0.6.3 ignore anyOf dans les propriétés d'outils.

    Un paramètre nullable et facultatif reste facultatif : conserver son type
    non-null et omettre le paramètre lorsqu'il n'est pas utilisé.
    """
    if isinstance(value, list):
        return [compatible_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = deepcopy(value)
    if 'anyOf' in value:
        branches = [item for item in value['anyOf'] if item.get('type') != 'null']
        if len(branches) != 1 or len(value['anyOf']) != 2:
            raise ValueError('Union JSON non prise en charge par ce client Ollama ; ne pas perdre silencieusement le type.')
        value = {**branches[0], **{key: item for key, item in value.items() if key not in {'anyOf', 'default'}}}
    return {key: compatible_schema(item) for key, item in value.items()}


class AuditedChatOllama(ChatOllama):
    """Conserve les paramètres assemblés pour le client Ollama et les sorties brutes."""

    _audit_directory: Path | None = PrivateAttr(default=None)
    _request_count: int = PrivateAttr(default=0)
    repair_tool_names: bool = False
    repair_tool_schema: bool = False

    def _chat_params(self, messages, stop=None, **kwargs):
        params = super()._chat_params(messages, stop, **kwargs)
        if self.repair_tool_schema and params.get('tools'):
            params['tools'] = compatible_schema(params['tools'])
        if self.repair_tool_names:
            names = {call['id']: call['name'] for message in messages
                     for call in getattr(message, 'tool_calls', [])}
            for original, converted in zip(messages, params['messages']):
                if isinstance(original, ToolMessage):
                    name = original.name or names.get(original.tool_call_id)
                    if not name:
                        raise ValueError('Impossible de relier le résultat à son outil.')
                    converted['tool_name'] = name
        self._request_count += 1
        if self._audit_directory:
            write_json(self._audit_directory / f'{self._request_count:03d}_request.json', params)
            wire = dict(params, messages=list(_copy_messages(params['messages'])),
                        tools=list(_copy_tools(params.get('tools'))))
            write_json(self._audit_directory / f'{self._request_count:03d}_wire_request.json',
                       ChatRequest(**wire).model_dump(mode='json', exclude_none=True))
        return params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            result = super()._generate(messages, stop, run_manager, **kwargs)
        except Exception as exc:
            if self._audit_directory:
                write_json(self._audit_directory / f'{self._request_count:03d}_error.json',
                           {'type': type(exc).__name__, 'message': str(exc)})
            raise
        if self._audit_directory:
            write_json(self._audit_directory / f'{self._request_count:03d}_response.json', {
                'generations': [{'message': generation.message.model_dump(mode='json'),
                                 'generation_info': generation.generation_info} for generation in result.generations]})
        return result


def make_model(config, audit_directory=None):
    provider = config.get('provider', 'ollama')
    if provider == 'nvidia_langchain':
        return make_nvidia_langchain_model(config, audit_directory)
    if provider != 'ollama':
        raise ValueError('Fournisseur inconnu : ' + provider)
    backend = AuditedChatOllama(
        model=config['model'], base_url=config['host'], temperature=config.get('temperature', 0),
        num_ctx=config['num_ctx'], num_predict=config['num_predict'], reasoning=config.get('reasoning', False),
        seed=config.get('seed', 42), client_kwargs={'timeout': config['timeout_seconds']},
        **{key: config[key] for key in ('top_p', 'top_k', 'repeat_penalty') if key in config})
    if audit_directory:
        backend._audit_directory = Path(audit_directory)
        backend._audit_directory.mkdir(parents=True, exist_ok=True)
    protocol = config.get('tool_protocol', 'native')
    if protocol == 'native_compatible':
        backend.repair_tool_names = True
        backend.repair_tool_schema = True
        return backend
    if protocol == 'native_named':
        backend.repair_tool_names = True
        return backend
    if protocol != 'native':
        raise ValueError('Protocole d’outils inconnu.')
    return backend


def message_text(message):
    """Texte visible, sans convertir les blocs structurés en repr Python."""
    if isinstance(message.content, str):
        return message.content
    return '\n'.join(block if isinstance(block, str) else block.get('text', '')
                     for block in message.content
                     if isinstance(block, str) or (isinstance(block, dict)
                        and block.get('type') == 'text' and not block.get('thought')))


def response_truncated(message):
    metadata = message.response_metadata
    return (metadata.get('done_reason') == 'length'
            or str(metadata.get('finish_reason', '')).upper() in {'MAX_TOKENS', 'LENGTH'})

import threading
import time
from langchain_core.language_models.chat_models import generate_from_stream
_lock = threading.Lock()
_last_request = 0.0

def _nvidia_model_class():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    class AuditedChatNVIDIA(ChatNVIDIA):
        _audit_directory: Path | None = PrivateAttr(default=None)
        _request_count: int = PrivateAttr(default=0)
        _max_requests: int = PrivateAttr(default=12)
        _request_interval: float = PrivateAttr(default=3)
        _index: int = PrivateAttr(default=0)
    
        def _get_payload(self, inputs, **kwargs):
            payload = super()._get_payload(inputs, **kwargs)
            if self._audit_directory:
                write_json(self._audit_directory / f'{self._index:03d}_request.json', payload)
            return payload
    
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            global _last_request
            with _lock:
                if self._request_count >= self._max_requests:
                    raise RuntimeError('Plafond de requêtes NVIDIA atteint.')
                delay = max(0, self._request_interval - (time.monotonic() - _last_request))
                if delay:
                    time.sleep(delay)
                _last_request = time.monotonic()
                self._request_count += 1
                self._index = self._request_count
            index = self._index
            started = time.monotonic()
            timing = {'request': index, 'first_chunk_seconds': None, 'first_text_seconds': None}
            print(f'NVIDIA appel {index}/{self._max_requests} : génération en cours', flush=True)
    
            def measured_stream():
                for chunk in super(AuditedChatNVIDIA, self)._stream(messages, stop, run_manager, **kwargs):
                    elapsed = round(time.monotonic() - started, 3)
                    if timing['first_chunk_seconds'] is None:
                        timing['first_chunk_seconds'] = elapsed
                    if chunk.message.content and timing['first_text_seconds'] is None:
                        timing['first_text_seconds'] = elapsed
                    yield chunk
    
            try:
                result = generate_from_stream(measured_stream())
                timing['duration_seconds'] = round(time.monotonic() - started, 3)
                for generation in result.generations:
                    generation.message.response_metadata.update(timing)
                if self._audit_directory:
                    write_json(self._audit_directory / f'{index:03d}_response.json', {
                        **timing, 'generations': [
                            {'message': item.message.model_dump(mode='json'), 'generation_info': item.generation_info}
                            for item in result.generations]})
                names = [call['name'] for g in result.generations for call in g.message.tool_calls]
                print(f'NVIDIA appel {index} terminé en {timing["duration_seconds"]:.2f} s ; outils demandés : {names}', flush=True)
                return result
            except Exception as exc:
                timing.update(duration_seconds=round(time.monotonic() - started, 3),
                              error_type=type(exc).__name__, error=redact(exc))
                if self._audit_directory:
                    write_json(self._audit_directory / f'{index:03d}_error.json', timing)
                raise RuntimeError(timing['error']) from None
    return AuditedChatNVIDIA

def make_nvidia_langchain_model(config, audit_directory=None):
    key = os.environ.get('NVIDIA_API_KEY')
    if not key:
        raise ValueError('NVIDIA_API_KEY absent.')
    model = _nvidia_model_class()(
        base_url='https://integrate.api.nvidia.com/v1', model=config['model'], api_key=key,
        temperature=config.get('temperature', 1), top_p=config.get('top_p', 0.95),
        max_completion_tokens=config.get('max_output_tokens', 16384), seed=config.get('seed', 42),
        model_kwargs={'chat_template_kwargs': {'enable_thinking': config.get('reasoning', True)},
                      'parallel_tool_calls': True}, usage_telemetry_enabled=False)
    model._client.timeout = config.get('timeout_seconds', 180)
    model._max_requests = config.get('max_model_calls', 12)
    model._request_interval = config.get('request_interval_seconds', 3)
    if audit_directory:
        model._audit_directory = Path(audit_directory)
        model._audit_directory.mkdir(parents=True, exist_ok=True)
    return model
