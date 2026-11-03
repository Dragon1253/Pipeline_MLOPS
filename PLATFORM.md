# Sentinel — plateforme locale MLOps

Interface React/TypeScript, Tailwind, composants accessibles Radix (approche shadcn/ui), Recharts et icônes Lucide. API FastAPI, WebSocket et stockage SQLite via SQLAlchemy. L'agent Python existant est appelé dans un thread de travail pour garder l'API réactive.

## Lancer l'application

Depuis la racine du projet :

```powershell
.\start-platform.ps1
```

Ouvrir http://127.0.0.1:8000. Le même serveur fournit l'API et l'interface compilée. Aucun serveur Node n'est nécessaire après compilation. Arrêter avec Ctrl+C.

Si PowerShell bloque les scripts, lancer directement :

```powershell
.\.venv-cloud\Scripts\python.exe -m uvicorn platform_api.app:app --host 127.0.0.1 --port 8000
```

## Installation et compilation

Les dépendances de plateforme sont ajoutées à l'environnement cloud existant ; l'environnement local Qwen reste intact.

```powershell
.\.venv-cloud\Scripts\python.exe -m pip install -r requirements-platform.txt
Set-Location frontend
npm ci
npm run build
Set-Location ..
```

En développement, lancer l'API puis `npm run dev` dans frontend. Vite fournit l'interface sur le port annoncé (5173 par défaut), avec proxy HTTP/WebSocket vers l'API sur 8000. Recompiler après modification pour utiliser le serveur unique.

## Démonstration

1. Cliquer sur **Lancer un scénario** et choisir un des cinq scénarios. Le diagnostic utilise automatiquement **Nemotron via NVIDIA**.
2. Les huit dernières fenêtres de calibration, puis focus et deux confirmations sont rejouées. Les métriques sont celles des fichiers existants, sans fabrication de performances. Les heures représentent les heures de relecture, pas une collecte historique de production.
3. L'alerte ouvre une investigation réelle avec Nemotron. Le modèle choisit ses outils et sa conclusion ; les preuves restent consultables. Aucun sélecteur de modèle ni diagnostic scénarisé n'est disponible pour les nouvelles investigations.
4. NVIDIA utilise la clé locale `.env` et consomme des appels à chaque investigation. Aucun appel réel n'est effectué au simple lancement de la plateforme. Les anciens incidents conservent l'indication du modèle réellement utilisé.
5. La correction présente un plan simulé selon la cause retournée. Elle ne modifie aucun fichier source et ne réentraîne ni ne déploie aucun modèle.
6. La validation indique son absence de résultat réel. Aucune amélioration ni résolution réelle n'est inventée. Une fausse alerte peut proposer une clôture sans modification.
7. Le rapport Markdown est téléchargeable à la fin. Il reste accessible dans les rapports et dans l'historique des incidents.

## Persistance et limites

Les métriques, incidents et événements sont conservés dans `platform_data/monitoring.sqlite3` (ignoré par Git). Les traces originales de l'agent restent sous `results/`. Les événements WebSocket sont relus depuis la base ; l'interface se reconnecte automatiquement et recharge l'état. Un redémarrage marque les traitements inachevés comme interrompus ; il ne les relance pas automatiquement.

Cette version fonctionne sur une seule instance serveur et autorise un parcours actif à la fois. Elle écoute uniquement sur la machine locale et ne contient pas d'authentification. Elle n'est pas encore connectée à un service de prédiction en production : le flux est une relecture explicite des scénarios étiquetés. Les performances de churn en production nécessiteront la réception des labels réels.

Le futur agent de correction remplacera la section de simulation de `run_incident` dans `platform_api/app.py`, en conservant les événements et le rapport. Un connecteur de monitoring pourra ensuite alimenter le même stockage et déclencher l'investigation à réception de nouvelles fenêtres.

Le diagnostic du prototype demande uniquement `cause_probable`, `resume` et `preuves` (identifiants E). Le champ `limites` n'est plus demandé ni affiché dans le panneau de diagnostic ; les anciennes traces restent intactes.

Une réponse du modèle sans texte visible ni appel d'outil natif est relancée au maximum deux fois, dans le budget d'appels existant et avec le même historique. Une tentative de diagnostic contenant `cause_probable` mais mal formée en JSON reçoit l'erreur et sa position, avec une demande de conserver la cause, le résumé et les preuves, et de supprimer le champ `limites`. Une réponse visible qui annonce une analyse sans appeler d'outil ni fournir de diagnostic reçoit une demande de poursuite : le modèle peut appeler un outil ou conclure avec les trois champs. Ces deux derniers cas partagent une limite de deux relances. Aucune réparation locale des caractères ni sélection de cause n'est appliquée. Le journal distingue les trois cas. Si l'investigation reste incomplète, ses preuves, compteurs et réponse brute sont conservés, et aucune correction ne démarre.

Les options Python `max_json_format_retries` et `max_empty_response_retries` valent 2 par défaut et peuvent être ramenées à 0. Elles n'augmentent pas `max_model_calls`. Le résultat expose séparément `json_format_retries` et `empty_response_retries` ; les messages originaux et les demandes de correction restent dans `trace.json`.

## Vérification

```powershell
.\.venv-cloud\Scripts\python.exe -m unittest discover -s tests -v
Set-Location frontend
npm run build
```

Les tests de plateforme parcourent les cinq scénarios sans réseau en injectant un modèle simulé uniquement dans les tests : calculs d'outils, événements de démarrage et de fin, rapport, limites de simulation, historique et interruption après redémarrage. Ils vérifient que la plateforme appelle toujours le fournisseur NVIDIA et refuse les anciens paramètres de sélection de modèle.
