# Agent de correction MLOps — V3

## Ce que contient la V3

- **LangGraph**: workflow `plan -> guard -> execute -> verify`.
- **LLM optionnel**: plan structuré via `ChatOpenAI` (ou endpoint compatible OpenAI).
- **Baseline**: mode `rules` sans API.
- **Garde-fous**: décision LLM incompatible avec le diagnostic -> `human_review`.
- **Outils**: clôture/quarantaine simulées, préparation de réentraînement/rollback; aucune modification de production.
- **Traces**: `artifacts_v3/agent_traces.jsonl`.
- **Tests**: 5 incidents, y compris confiance faible, et tentative LLM interdite.

## Installation — PowerShell VS Code

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python demo_v3.py --mode rules
python -m unittest -v test_v3.py
```

## Activer le vrai LLM

Dans PowerShell, uniquement si vous disposez d'une clé API valide :

```powershell
$env:OPENAI_API_KEY = "VOTRE_CLE_API"
$env:OPENAI_MODEL = "gpt-4o-mini"
python demo_v3.py --mode llm
```

Pour un service compatible OpenAI, définir aussi `OPENAI_BASE_URL` et le nom de modèle approprié.
**Ne jamais committer une clé API.** Des coûts d'API peuvent s'appliquer.

## Limites honnêtes

Le mode `rules` est un orchestrateur déterministe, **pas** un agent LLM.
Le mode `llm` planifie avec un LLM mais les garde-fous sont déterministes.
Les outils sont des simulations, pas un réentraînement/rollback de production.
Le modèle Random Forest de `ml_pipeline.py` est un banc d'essai indépendant.
Les scénarios sont inspirés de la présentation et ne constituent pas une évaluation sur des incidents réels.

## Intégration future

Amen peut envoyer les incidents au schéma `IncidentV3` (`incident_id`, `type`, `confidence`, `metrics`, `details`).
Le monitoring proactif doit envoyer son alerte à l'analyse, puis à l'agent de correction.
Brancher ensuite des outils réels de réentraînement, registre MLflow et vérification des performances avant approbation humaine.
