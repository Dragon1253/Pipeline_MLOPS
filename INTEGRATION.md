# Intégration du projet MLOps : le flux complet

Ce document décrit comment les parties du groupe s'enchaînent, et comment lancer le flux
complet pour vérifier que tout fonctionne. Le script `run_full_pipeline.py` **ne modifie aucun
fichier du groupe** : il réutilise le pipeline existant en lui fournissant le monitoring à jour.

## 1. Le flux

```text
Flux de fenêtres de 500 clients (simulé)
        │
        ▼
1. MONITORING                     monitoring/                       (partie monitoring)
   métriques à l'arrivée des labels → alerte confirmée warning / critical
   risque prédit sans labels        → alerte prédictive (vigilance, environ 2 pas plus tôt)
   écrit un dossier d'alerte : alert.json, context.json, focus/, confirmation-*/
        │  alerte confirmée
        ▼
2. AGENT D'ANALYSE                mlops_churn/agent.py              (partie analyse)
   investigate(episode_dir) : LLM (Nemotron ou Qwen) + 10 outils analytiques
   → diagnostic { cause_probable, resume, preuves E0001… }
        │
        ▼
   SCORE DE CONFIANCE             mlops_correction_agent_groq/integration_adapter.py
   la cause est-elle appuyée par une preuve cohérente ? alerte confirmée (+0,10) et critique (+0,05)
   → incident { type, confidence, details }
        │
        ▼
3. ORCHESTRATEUR                  mlops_correction_agent_groq/orchestrator_v4.py
   LangGraph : règles ou LLM Groq propose une route, un garde-fou la vérifie
   (confiance ≥ 0,7, type connu, preuve cohérente) → correction ou revue humaine
        │                                   │
        ▼                                   ▼
4. AGENT DE CORRECTION            REVUE HUMAINE (Flask app.py + e-mail)
   agent_v3.py : plan → guard → execute → verify
   close_alert · quarantine_batch · prepare_retraining · prepare_rollback   (toutes simulées)
```

| Cause diagnostiquée | Type d'incident | Action de correction |
| --- | --- | --- |
| `data_corruption` | qualité des données | `quarantine_batch` (lot bloqué pour tout réentraînement) |
| `data_drift`, `concept_drift` | dérive | `prepare_retraining` |
| `deployment_regression` (preuve de version) | régression de version | `prepare_rollback` |
| `no_incident` | fausse alerte | `close_alert` |
| `undetermined` ou preuve insuffisante | inconnu | revue humaine |

## 2. Ce que le script d'intégration change (et ne change pas)

* Le pipeline du groupe (`pipeline_v4.py`) importe le monitoring depuis sa copie
  `mlops_correction_agent_groq/amen_project/monitoring/`, figée à l'étape 2.2.
* `run_full_pipeline.py` charge d'abord le **vrai** dossier `monitoring/` de la racine, puis
  appelle `pipeline_v4.run_pipeline` sans le modifier : l'analyse, le score, l'orchestrateur et
  la correction sont exactement ceux du groupe.
* Il ajoute l'alerte prédictive (`--predict`) et le retard des labels (`--label-delay`).
* Les sorties vont dans `monitoring_runs/integration/` (ignoré par Git), y compris la base des
  revues humaines (`HUMAN_REVIEW_DB`), pour ne rien écrire dans le dossier du groupe.

## 3. Lancer le flux complet

Sur une branche `integration` créée à partir de `agent_investigation`, avec la branche
`monitoring` fusionnée (voir le guide pas à pas plus bas), depuis la racine du projet :

```powershell
# Dépendances du pipeline du groupe (orchestrateur LangGraph)
.\.venv\Scripts\python.exe -m pip install -r mlops_correction_agent_groq\requirements.txt

# Prédicteur de l'alerte (une fois ; nécessite le jeu de données monitoring_runs/dataset)
.\.venv\Scripts\python.exe -m monitoring train-predictor

# Flux complet, hors ligne et sans clé
.\.venv\Scripts\python.exe run_full_pipeline.py --scenario data_corruption --predict --label-delay 2
.\.venv\Scripts\python.exe run_full_pipeline.py --all --predict --label-delay 2

# Avec le vrai agent d'analyse (clé NVIDIA dans .env à la racine) ou Ollama local
.\.venv\Scripts\python.exe run_full_pipeline.py --scenario deployment_regression --predict --analysis-mode nvidia

# Avec l'orchestrateur LLM (clé Groq dans mlops_correction_agent_groq\.env)
.\.venv\Scripts\python.exe run_full_pipeline.py --scenario data_corruption --orchestrator-mode llm
```

## 3 bis. L'interface web

`integration_app.py` lance le même pipeline et l'affiche en direct dans le navigateur :
log loss et risque prédit fenêtre par fenêtre, alertes confirmées et prédictives, outils
appelés par l'agent d'analyse, diagnostic, score de confiance, décision de l'orchestrateur,
action de correction ou revue humaine.

```powershell
.\.venv\Scripts\python.exe -m pip install -r mlops_correction_agent_groq\requirements-integration.txt
.\.venv\Scripts\python.exe integration_app.py
```

Puis ouvrir http://127.0.0.1:8050 . En haut de la page, trois indicateurs montrent si la clé
NVIDIA, la clé Groq et le prédicteur sont prêts (les clés ne sont jamais affichées). Un seul
parcours à la fois ; toute erreur s'affiche dans la page.

Clé NVIDIA (analyse réelle) : créer une clé sur https://build.nvidia.com (bouton « Get API
Key », elle commence par `nvapi-`), copier `.env.example` vers `.env` à la racine et y mettre
`NVIDIA_API_KEY=nvapi-…`. Ne jamais committer `.env` (déjà ignoré par Git).

## 3 ter. Le flux complet dans Sentinel (interface d'Amen)

`sentinel_app.py` lance l'interface Sentinel (`platform_api/` + `frontend/`) en remplaçant
uniquement sa fonction `run_incident` : au lieu de rejouer des épisodes figés avec une correction
de démonstration, chaque scénario lance le vrai flux (monitoring en direct avec alerte prédictive,
agent d'analyse Nemotron, score de confiance, orchestrateur, agent de correction ou revue humaine).
C'est l'évolution prévue dans `PLATFORM.md`. Aucun fichier d'Amen n'est modifié ; ses données sont
rangées dans `monitoring_runs/sentinel` (base séparée de `platform_data/`).

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
.\.venv\Scripts\python.exe sentinel_app.py                       # orchestrateur à règles
.\.venv\Scripts\python.exe sentinel_app.py --orchestrator llm     # orchestrateur Groq
```

Puis ouvrir http://127.0.0.1:8000 et cliquer sur « Lancer un scénario ». Quand l'orchestrateur demande une revue humaine, un panneau « Revue humaine » apparaît dans la page de l'incident avec les boutons Approuver / Refuser ; la décision passe par la logique de `app.py` (inchangé), servie par le même serveur, et le lien de l'e-mail ouvre aussi cette adresse. Il n'est plus nécessaire de lancer `app.py`. L'interface compilée
(`frontend/dist`, ignorée par Git) se régénère avec `npm ci` puis `npm run build` dans `frontend`.

## 4. Résultat attendu en mode hors ligne (`fixture`, sans clé)

| Scénario | Monitoring | Analyse (fixture) | Orchestrateur | Correction |
| --- | --- | --- | --- | --- |
| data_corruption | prédictive w008 (pas 8) puis confirmée critical (pas 10) | data_corruption, confiance 0,90 | correction | `quarantine_batch` simulée |
| data_drift | prédictive w008 puis confirmées warning et critical | undetermined | revue humaine | — |
| concept_drift | confirmée critical (pas de prédictive : invisible sans labels) | undetermined | revue humaine | — |
| deployment_regression | prédictive w008 puis confirmée critical | undetermined | revue humaine | — |
| normal | fluctuation warning, prédictive non confirmée | undetermined | revue humaine | — |

Le mode `fixture` est une analyse hors ligne qui ne reconnaît que la corruption ; les autres
scénarios partent donc en revue humaine. Avec `--analysis-mode nvidia` (ou `local`), le vrai
agent d'analyse diagnostique tous les scénarios et l'orchestrateur peut choisir les autres actions.

## 5. Vérifier

```powershell
# Tests du monitoring (racine)
.\.venv\Scripts\python.exe -m unittest tests.test_monitoring tests.test_monitoring_dataset tests.test_monitoring_prediction tests.test_monitoring_sequences tests.test_monitoring_analysis tests.test_monitoring_live

# Tests du pipeline du groupe (depuis leur dossier)
Set-Location mlops_correction_agent_groq
..\.venv\Scripts\python.exe -m unittest test_pipeline_v4 test_orchestrator_v4 test_v3
Set-Location ..
```

## 6. Points à décider avec le groupe

1. **Copie figée** : `amen_project/` contient une copie de `mlops_churn/` (identique) et de
   `monitoring/` (étape 2.2). Une fois la branche `monitoring` fusionnée, `pipeline_v4.py` pourrait
   importer la racine au lieu de la copie ; en attendant, `run_full_pipeline.py` le fait sans
   toucher à leur code.
2. **Alerte prédictive** : elle est produite et transmise (`alert.json` → `preceded_by_predictive_alert`,
   `predictive_alerts/`), mais l'orchestrateur ne l'utilise pas encore. Une action préventive sûre
   (quarantaine simulée dès l'alerte prédictive) serait une évolution de l'orchestrateur.
3. **Choix du dossier transmis** : `pipeline_v4.py` prend le dernier dossier d'alerte ; prendre
   celui de l'alerte `critical` serait plus sûr si plusieurs alertes se suivent.
