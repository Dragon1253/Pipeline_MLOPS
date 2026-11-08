# Monitoring du modèle de churn

Premier maillon du workflow : **monitoring → agent d'analyse → agent de correction**.
Le monitoring reçoit des fenêtres de production, mesure la performance du modèle quand
les labels arrivent, décide d'une alerte, puis écrit un paquet d'alerte au format exact
qu'attend `mlops_churn.agent.investigate`. Il ne diagnostique pas la cause et ne corrige rien.

Il remplace la relecture scénarisée de la plateforme (`run_incident`, où l'alerte est
toujours posée sur la fenêtre `focus`) par une décision réellement mesurée.

## Fonctionnement

```text
Flux de fenêtres (500 clients)            monitoring/stream.py
   │  incident progressif : début, montée, intensité
   ▼
Scoring immédiat + signaux sans labels    monitoring/signals.py
   │  (forme des prédictions, PSI des entrées) — enregistrés pour la future prédiction
   ▼
Labels retardés (label_delay fenêtres)    monitoring/monitor.py
   │  log loss, accuracy, précision, recall, F1, ROC-AUC, AP
   ▼
Règle d'alerte
   │  log loss > seuil calibré sur `persistence` fenêtres consécutives
   │  gravité : warning, ou critical si l'écart à la référence dépasse 3 × celui du seuil
   ▼
Paquet d'alerte pour l'agent              monitoring/packager.py
      alert.json, context.json, active.joblib, focus/, confirmation-*/, monitoring_context.json
```

Le seuil est celui de `scenario_data/demo/calibration.json` (quantile 95 % de 30 fenêtres
normales + marge = 0,5412), le même que pour l'agent et la plateforme.

### Règles de gestion des incidents

| Situation | Décision |
| --- | --- |
| Log loss au-dessus du seuil | Alerte, avec la fenêtre concernée comme `focus` |
| Incident ouvert, confirmations au-dessus | Même incident, pas de nouvelle alerte |
| Confirmations repassent sous le seuil | Incident clos `not_persistent` (probable fausse alerte), surveillance reprise |
| Alerte `warning` ouverte puis fenêtre `critical` | Incident clos `escalated`, nouvelle alerte centrée sur la fenêtre critique |
| Nouvelle version du modèle servie | Incident clos `model_version_changed`, nouveau contexte de surveillance |
| `recovery` fenêtres sous le seuil | Incident clos `recovered` |

Les deux règles d'escalade et de non-persistance évitent qu'une fluctuation naturelle,
survenue juste avant un vrai incident, « absorbe » cet incident : sans elles, le paquet
transmis à l'agent aurait une fenêtre principale saine.

## Scénarios du flux

| Scénario | Montée progressive |
| --- | --- |
| `data_drift` | Part croissante de clients proches de la frontière du modèle |
| `concept_drift` | Signe du logit générateur qui passe de +1 à −1 |
| `data_corruption` | Part croissante des lignes avec `monthly_charges` × 100 après la source |
| `deployment_regression` | Saut : la version inversée sert toute la fenêtre dès le déploiement |
| `normal` | Aucun incident ; une alerte y est une fausse alerte (scénario `false_alert` de l'agent) |

Les injections réutilisent `mlops_churn.scenarios` sans le modifier. La vérité
(scénario, intensité par fenêtre) est écrite dans `monitoring_runs/<run>/evaluation/truth.json`,
**jamais** dans un dossier d'épisode. L'identifiant du flux est neutre (date + hasard)
pour que le nom du scénario n'apparaisse ni dans `alert.json` ni dans les identifiants clients.

## Commandes

Depuis la racine du projet, avec l'environnement existant (`.venv-cloud` ou `.venv-agent`) :

```powershell
# Un flux : affiche la log loss par fenêtre et les alertes, écrit les paquets
.\.venv-cloud\Scripts\python.exe -m monitoring run --scenario data_corruption --seed 9

# Même chose, puis transmission du dernier paquet à l'agent d'analyse (consomme des appels NVIDIA)
.\.venv-cloud\Scripts\python.exe -m monitoring run --scenario deployment_regression --seed 2 --investigate nvidia

# Matrice de détection : 30 flux par scénario
.\.venv-cloud\Scripts\python.exe -m monitoring evaluate --seeds 30
.\.venv-cloud\Scripts\python.exe -m monitoring evaluate --seeds 30 --persistence 2

# Tests (aucun appel réseau ni LLM)
.\.venv-cloud\Scripts\python.exe -m unittest tests.test_monitoring -v
```

Options communes : `--windows`, `--onset`, `--ramp`, `--intensity`, `--label-delay`, `--persistence`.
Les sorties vont dans `monitoring_runs/` (ignoré par Git).

## Résultats actuels

30 flux par scénario, 20 fenêtres, incident à partir de la fenêtre 8, montée sur 4 fenêtres,
labels en retard d'une fenêtre.

| Scénario | Persistance 1 : détection | Fausses alertes / 100 fenêtres saines | Persistance 2 : détection | Fausses alertes / 100 fenêtres saines |
| --- | --- | --- | --- | --- |
| data_drift | 30 / 30 (5 critiques) | 5,8 | 30 / 30 | 0,4 |
| concept_drift | 30 / 30 | 7,1 | 30 / 30 | 0,8 |
| data_corruption | 30 / 30 | 7,1 | 30 / 30 | 0,8 |
| deployment_regression | 30 / 30 | 7,1 | 30 / 30 | 0,8 |
| normal (aucun incident) | — | 9,2 | — | 0,7 |

Lecture : toutes les dégradations sont détectées, avec un délai réel d'environ une fenêtre
(le temps que les labels arrivent). La persistance 2 divise les fausses alertes par dix au
prix d'une fenêtre de délai supplémentaire. Avec la persistance 1, le taux de fausses alertes
(7 à 9 %) dépasse les 5 % nominaux : le seuil est estimé sur seulement 30 fenêtres.

Limites : le data drift partiel (25 % de la population) reste souvent sous le seuil ; la
log loss ne distingue pas concept drift et régression de version (mêmes valeurs ici), ce
qui est le rôle de l'agent ; les incidents sont simulés.

## Contrat avec l'agent d'analyse

Chaque paquet respecte le contrat du README de l'agent et est vérifié par les tests avec
ses propres fonctions (`prepare_workspace`, `AnalyticalMethods`) :

* `feedback.csv` présent pour `focus` et les confirmations ;
* `context.json` : `deployment.reference_version` et `active_version` égaux aux SHA-256
  des fichiers de modèle, sinon la comparaison de versions est refusée ;
* `alert.json` : champs historiques (`metric`, `observed_value`, `threshold`, …) plus
  `alert_type`, `severity`, `critical_threshold`, `persistence_windows`,
  `windows_above_threshold`, `label_delay_windows`, `detected_at_step`, `alert_schema_version` ;
* `monitoring_context.json` : chronologie et signaux sans labels (non lu par l'agent actuel).

Intégration dans la plateforme (à valider avec la personne de l'agent) : remplacer la
boucle de relecture de `run_incident` par `monitoring.service.run_stream(..., on_event=...)`,
dont les événements `metric_recorded` et `alert_triggered` sont déjà connus de l'interface,
puis appeler `investigate(episode_dir=<paquet>)`.

## Suite : prédire l'alerte

`monitoring_context.json` enregistre déjà, à chaque fenêtre, les signaux disponibles
avant les labels. Ils serviront à entraîner le modèle qui prédit une dégradation à
venir (régression logistique, gradient boosting, puis GRU / LSTM), comparé au monitoring
réactif ci-dessus sur les mêmes flux.
