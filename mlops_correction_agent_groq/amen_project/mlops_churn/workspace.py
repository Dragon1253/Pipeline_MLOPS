"""Préparation du catalogue public et vérification de l’intégrité des ressources."""
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InvestigationWorkspace:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        self.resources = {r['id']: r for r in self.manifest['resources']}
        if len(self.resources) != len(self.manifest['resources']):
            raise ValueError('Identifiants de ressources dupliqués.')
        for resource_id in self.resources:
            self.path(resource_id)

    def path(self, resource_id):
        if resource_id not in self.resources:
            raise ValueError(
                'Ressource inconnue. Utiliser un identifiant exact du catalogue dans resource_id : '
                + ', '.join(sorted(self.resources))
                + '. Les noms de fichiers (.csv/.json) et les chemins ne sont pas des identifiants.')
        item = self.resources[resource_id]
        path = (self.directory / item['file']).resolve()
        if not path.is_relative_to(self.directory) or path.suffix not in {'.csv', '.json'}:
            raise ValueError('Chemin de ressource hors du périmètre autorisé.')
        if sha256(path) != item['sha256']:
            raise ValueError('Le contenu de la ressource a changé depuis sa préparation.')
        return path

    def inspect(self, operation='list', resource_id='', columns=None, offset=0, limit=5):
        if operation == 'list':
            return {'resources': list(self.resources.values()),
                    'path_policy': 'Accès par identifiant du catalogue uniquement.'}
        if operation != 'read':
            raise ValueError('Opération autorisée : list ou read.')
        if offset < 0 or not 1 <= limit <= 20:
            raise ValueError('offset >= 0 et 1 <= limit <= 20 sont requis.')
        path = self.path(resource_id)
        item = self.resources[resource_id]
        if item['kind'] == 'csv':
            frame = pd.read_csv(path).iloc[offset:offset + limit]
            if columns:
                missing = set(columns) - set(frame.columns)
                if missing:
                    raise ValueError(f'Colonnes inconnues : {sorted(missing)}')
                frame = frame[columns]
            return {'resource_id': resource_id, 'sha256': item['sha256'], 'total_rows': item['rows'],
                    'offset': offset, 'returned_rows': len(frame), 'columns': list(frame.columns),
                    'records': json.loads(frame.to_json(orient='records'))}
        content = path.read_text(encoding='utf-8')
        # Pour JSON : offset en caractères et 250 caractères par unité de limit.
        snippet = content[offset:offset + limit * 250]
        return {'resource_id': resource_id, 'text': snippet, 'offset_unit': 'characters',
                'next_offset': offset + len(snippet) if offset + len(snippet) < len(content) else None,
                'sha256': item['sha256']}

    def verify(self):
        for resource_id in self.resources:
            self.path(resource_id)
        if (self.directory / 'model_manifest.json').is_file():
            for model_id in json.loads((self.directory / 'model_manifest.json').read_text(encoding='utf-8'))['models']:
                self.model_path(model_id)
        return True

    def model_path(self, model_id):
        """Modèles locaux enregistrés par le programme, jamais un chemin du LLM."""
        manifest = self.directory / 'model_manifest.json'
        if not manifest.is_file():
            raise ValueError('Aucun modèle local enregistré pour cette investigation.')
        models = json.loads(manifest.read_text(encoding='utf-8'))['models']
        if model_id not in models:
            raise ValueError('Identifiant de modèle inconnu ; consulter model_catalog.')
        item = models[model_id]
        path = (self.directory / item['file']).resolve()
        if not path.is_relative_to((self.directory / 'models').resolve()) or path.suffix != '.joblib':
            raise ValueError('Modèle extérieur au registre local autorisé.')
        if sha256(path) != item['sha256']:
            raise ValueError('Le modèle a changé depuis son enregistrement.')
        return path


def prepare_workspace(config):
    source_root = resolve_path(config['source_experiment']).resolve()
    episode_id = config['episode_id']
    source_episode = resolve_path(config.get('episode_dir', source_root / 'episodes' / episode_id)).resolve()
    if not source_episode.is_dir():
        raise FileNotFoundError(f'Épisode absent : {source_episode}')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run = resolve_path(config['output_dir']) / run_id
    public = run / 'workspace'
    (public / 'data').mkdir(parents=True, exist_ok=False)
    resources, originals = [], {}

    def copy_resource(resource_id, source, description):
        source = source.resolve()
        if not any(source.is_relative_to(allowed) for allowed in (source_root, source_episode)) or not source.is_file():
            raise ValueError('Source absente ou extérieure à l’expérience autorisée.')
        kind = source.suffix.lstrip('.')
        if kind not in {'csv', 'json'}:
            raise ValueError('Type de fichier non autorisé.')
        destination = public / 'data' / f'{resource_id}.{kind}'
        shutil.copyfile(source, destination)
        checksum = sha256(source)
        if sha256(destination) != checksum:
            raise ValueError('La source a changé pendant sa copie.')
        resource = {'id': resource_id, 'file': destination.relative_to(public).as_posix(),
                    'container_path': '/data/' + destination.relative_to(public).as_posix(),
                    'kind': kind, 'description': description, 'sha256': checksum}
        if kind == 'csv':
            frame = pd.read_csv(destination)
            resource.update(rows=len(frame), columns=list(frame.columns), dtypes={k: str(v) for k, v in frame.dtypes.items()})
        resources.append(resource)
        originals[str(source)] = checksum

    for name in ('source', 'inputs', 'predictions', 'feedback'):
        copy_resource(name, source_episode / 'focus' / f'{name}.csv', {
            'source': 'Capture source avant les transformations de la fenêtre.',
            'inputs': 'Données effectivement reçues par le modèle.',
            'predictions': 'Prédictions et version servie.',
            'feedback': 'Résultats réels disponibles au moment de cette investigation.',
        }[name])
    for name in ('alert', 'context'):
        copy_resource(name, source_episode / f'{name}.json', 'Observations publiques : ' + name)
    copy_resource('metrics', source_episode / 'focus/monitoring.json', 'Métriques et alerte de la fenêtre courante.')
    copy_resource('reference_clients', source_root / 'reference_data.csv', 'Population de référence indépendante.')
    for index, confirmation in enumerate(sorted(source_episode.glob('confirmation-*')), 1):
        copy_resource(f'confirmation_{index}_metrics', confirmation / 'monitoring.json', 'Métriques de confirmation déjà disponibles.')
    if config.get('model_version_comparison'):
        # Artéfacts créés par notre pipeline locale de confiance ; aucun chemin
        # arbitraire fourni par un outil ou par la réponse du modèle distant.
        model_dir = public / 'models'
        model_dir.mkdir()
        registry = {}
        deployment = json.loads((source_episode / 'context.json').read_text(encoding='utf-8'))['deployment']
        for model_id, source, version_key in [
                ('reference_model', source_root / 'models/reference.joblib', 'reference_version'),
                ('active_model', source_episode / 'active.joblib', 'active_version')]:
            source = source.resolve()
            if not any(source.is_relative_to(allowed) for allowed in (source_root, source_episode)) or not source.is_file():
                raise ValueError('Modèle source absent ou hors expérience.')
            digest = sha256(source)
            if deployment.get(version_key) != 'sha256:' + digest:
                raise ValueError('Empreinte du modèle incohérente avec les métadonnées de déploiement.')
            target = model_dir / (model_id + '.joblib')
            shutil.copyfile(source, target)
            if sha256(target) != digest:
                raise ValueError('Modèle modifié pendant la copie.')
            originals[str(source)] = digest
            registry[model_id] = {'file': target.relative_to(public).as_posix(), 'sha256': digest}
        write_json(public / 'model_manifest.json', {'models': registry})
        catalog = public / 'data/model_catalog.json'
        write_json(catalog, {'models': [{'id': k, 'version': 'sha256:' + v['sha256']} for k, v in registry.items()],
                             'execution': 'Inférence locale en lecture seule via compare_model_versions ; binaires non transmis au LLM.'})
        resources.append({'id': 'model_catalog', 'file': catalog.relative_to(public).as_posix(),
                          'container_path': '/data/data/model_catalog.json', 'kind': 'json',
                          'description': 'Identifiants des modèles locaux disponibles pour une comparaison sur les mêmes observations.',
                          'sha256': sha256(catalog)})
    write_json(public / 'manifest.json', {'schema_version': 1, 'resources': resources,
                                         'feedback_policy': 'Instant d’investigation après réception des confirmations de cet épisode.'})
    write_json(run / 'setup.json', {'created_at': utc_now(), 'config': config, 'source_hashes': originals,
                                    'policy': 'Seules les observations publiques sont copiées ; aucune vérité de simulation.'})
    return run, InvestigationWorkspace(public)
