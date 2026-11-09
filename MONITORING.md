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

## Étape 2 : prédire l'alerte

Question : **peut-on anticiper une alerte avant l'arrivée des labels, et pour quels incidents ?**
Le monitoring réactif ci-dessus sert de référence (R1) : il ne voit une fenêtre dégradée
qu'après `label_delay` fenêtres.

### 2.1 Jeu de données (`monitoring/dataset.py`)

Des milliers de flux sont simulés avec `run_stream`, en variant les réglages ; chaque
fenêtre devient une ligne.

| Groupe de colonnes | Contenu | Rôle |
| --- | --- | --- |
| `sig_*` | Forme des prédictions (proba moyenne, taux prédit positif, entropie, PSI des scores…) et état des entrées (PSI par variable, hors plage, manquants) | Entrées du modèle, connues au moment `t` |
| `known_*`, `label_delay` | Métriques des fenêtres dont les labels sont **déjà arrivés** à `t` (log loss des 3 dernières, accuracy, recall, F1, AUC, pente) | Entrées du modèle |
| `react_*` | Décision du monitoring réactif à ce pas | Référence R1 |
| `label_log_loss`, `target_now`, `target_h1…h3` | Log loss réelle de `t` ; `t` dépasse le seuil ; une des fenêtres `t+1 … t+h` dépasse le seuil | Cibles, jamais des entrées |
| `truth_*` | Scénario, variante, intensité, début | Analyse des résultats uniquement |

`target_now` fait gagner `label_delay` fenêtres sur le monitoring réactif ; `target_h{k}`
annonce en plus une dégradation **future**.

Grille, tirée à partir de la seule graine du flux (avant toute simulation) : 40 % de flux
normaux, sinon un des 4 incidents ; intensité max 25 / 50 / 100 % ; montée 0, 2, 4 ou 8
fenêtres ; début entre les fenêtres 6 et 14 ; retard des labels 1, 2 ou 3 ; 24 fenêtres.

| Découpage | Graines | Flux | Mécanismes |
| --- | --- | --- | --- |
| `train` | 1 – 2 000 | 2 000 | standard |
| `validation` | 10 001 – 10 500 | 500 | standard |
| `test` | 20 001 – 20 500 | 500 | standard (utilisé une seule fois, à la fin) |
| `generalization` | 30 001 – 30 300 | 300 | **jamais vus** : drift vers le segment mensuel + paiement manuel, lien X → churn qui s'efface au lieu de s'inverser, `usage_hours_month` × 10, version trop sûre d'elle |

Chaque fichier a un manifeste (`*.manifest.json`) : grille, graines, nombre de lignes,
part des cibles à 1 et empreinte SHA-256 du CSV.

```powershell
# Tout générer (environ 10 minutes avec 4 processus ; sans --workers : environ 30 minutes)
.\.venv\Scripts\python.exe -m monitoring dataset --workers 4

# Un essai rapide
.\.venv\Scripts\python.exe -m monitoring dataset --split train --streams 50
```

Sorties : `monitoring_runs/dataset/{train,validation,test,generalization}.csv`.

Contrôles (tests `tests/test_monitoring_dataset.py`) : plages de graines disjointes,
configurations reproductibles, colonnes `known_*` construites **uniquement** à partir des
labels déjà reçus (vérifié pour des retards de 1, 2 et 3), cibles cohérentes avec la log
loss future, aucune colonne de vérité ou de cible parmi les entrées, variante `heldout`
réellement différente.

Observation sur un premier échantillon (200 flux) : 9 % des fenêtres **sans incident**
dépassent le seuil par simple fluctuation. Ces cas sont imprévisibles par nature et
fixent une limite au score atteignable par tout modèle de prédiction.

### 2.2 Références et premiers modèles (`monitoring/prediction.py`)

| Méthode | Principe |
| --- | --- |
| R1 monitoring réactif | Étape 1 : alerte quand les labels montrent la dégradation |
| R2 seuils fixes | Règle classique sans labels : max(PSI des scores, PSI max des entrées) |
| R3 log loss estimée | Entropie moyenne des probabilités (log loss attendue si le modèle était calibré) |
| M1 régression logistique | 69 entrées : signaux, métriques déjà connues, tendances par flux |
| M2 gradient boosting | `HistGradientBoostingClassifier`, mêmes entrées |

Protocole, fixé avant les résultats : apprentissage sur `train` ; hyperparamètres (PR-AUC)
et seuil d'alarme choisis sur `validation` ; **même budget de fausses alarmes pour toutes
les méthodes à score** (5 pour 100 fenêtres saines) ; mesure unique sur `test`, puis sur
`generalization`. Les tendances n'utilisent que la fenêtre courante et les précédentes
(vérifié par un test) ; `truth_*`, `target_*`, `react_*` ne sont jamais des entrées.
`known_windows` (nombre de fenêtres déjà étiquetées) est aussi exclu : il révèle la position
dans le flux, et le modèle apprendrait que les incidents simulés commencent entre les
fenêtres 6 et 14, un calendrier qui n'existe pas en production (biais repéré à l'étape 2.4).

Mesures par flux dégradé : **détection** (l'incident est-il signalé ?), **délai** depuis
le début de l'incident, **avance sur R1** (de combien de fenêtres l'alarme précède l'alerte
réactive), fausses alarmes pour 100 fenêtres saines.

```powershell
.\.venv\Scripts\python.exe -m monitoring predict --target target_now
.\.venv\Scripts\python.exe -m monitoring predict --target target_h2
.\.venv\Scripts\python.exe -m monitoring predict --target target_h2 --groups sig    # ablation
```

#### Résultats sur le test (500 flux, mesure finale)

Cible `target_h2` (dégradation dans les 2 prochaines fenêtres) :

| Méthode | PR-AUC | Détection | Délai depuis le début | Avance sur R1 | Fausses alarmes / 100 |
| --- | --- | --- | --- | --- | --- |
| R1 monitoring réactif | — | 99 % | 2,30 | — | 7,9 |
| R2 seuils fixes | 0,83 | 86 % | 0,79 | 1,50 | 4,8 |
| R3 log loss estimée | 0,60 | 45 % | 2,02 | 0,58 | 5,0 |
| M1 régression logistique | 0,89 | 100 % | 0,53 | 1,78 | 4,5 |
| **M2 gradient boosting** | **0,89** | **100 %** | **0,56** | **1,75** | **4,4** |

Avance de M2 sur R1 par scénario : data drift 2,78 · régression de déploiement 2,10 ·
corruption 1,93 · **concept drift 0,15** fenêtre.

Généralisation (mécanismes jamais vus) : M2 détecte 100 % des flux dégradés avec 1,85
fenêtre d'avance et 4,9 fausses alarmes / 100 ; la régression logistique 100 % avec 2,01.
Avec la cible `target_now`, la régression logistique chute à 78 % de détection sur ces
mécanismes : prédire un horizon de 2 fenêtres généralise mieux.

#### Lecture

* Les modèles appris détectent tous les incidents **avec environ 1,8 fenêtre d'avance** sur
  le monitoring réactif, tout en faisant **moins** de fausses alarmes que lui (≈ 5 contre 8).
* R3 seule échoue (45 %) : la log loss estimée ne voit ni la corruption (modèle sûr de lui
  et faux) ni la régression (probabilités inversées, même entropie). R2 rate les incidents
  qui ne changent pas les distributions. Combiner les signaux est donc nécessaire.
* **Concept drift : presque aucune avance** (0,15 fenêtre). Les entrées et la forme des
  prédictions restent normales ; seule la relation avec la réalité change. Il n'est détecté
  qu'à l'arrivée des labels. C'est l'hypothèse H2, confirmée.
* Limite : 9 % des fenêtres saines dépassent le seuil par fluctuation ; ces cas sont
  imprévisibles et plafonnent le PR-AUC.

### 2.3 Modèles séquentiels : GRU et LSTM (`monitoring/sequence_models.py`)

Question : un réseau qui **lit la suite des fenêtres** fait-il mieux que le gradient
boosting (M2), qui reçoit des tendances fabriquées à la main ?

* Entrée de la fenêtre `t` : les 40 signaux (`sig_*`, `known_*`, `label_delay`) des 6
  dernières fenêtres de son flux, plus un canal « remplissage » en début de flux et un
  indicateur « métriques pas encore connues ». Pas de tendances calculées à la main :
  le réseau les apprend.
* Réseau : une couche GRU ou LSTM (32 ou 64 unités, choisi sur la validation), dropout
  0,1, une sortie. Perte pondérée pour le déséquilibre des classes, Adam, arrêt précoce
  sur le PR-AUC de validation (patience 4, 30 époques maximum).
* Même protocole que 2.2 : cible `target_h2`, même budget de fausses alarmes, même test
  et même généralisation ; le rapport ajoute deux lignes, M3 GRU et M4 LSTM, et un
  tableau d'avance par scénario pour M2, M3 et M4.

Contrôles (`tests/test_monitoring_sequences.py`) : chaque séquence se termine sur la
fenêtre courante, ne contient jamais une fenêtre future ni une fenêtre d'un autre flux ;
la normalisation n'utilise que les statistiques d'entraînement ; un entraînement court
de bout en bout (ignoré si PyTorch n'est pas installé).

```powershell
# Une seule fois : PyTorch (environ 200 Mo, version processeur sous Windows)
.\.venv\Scripts\python.exe -m pip install -r requirements-monitoring.txt

# Tests puis expérience (quelques minutes)
.\.venv\Scripts\python.exe -m unittest tests.test_monitoring_sequences -v
.\.venv\Scripts\python.exe -m monitoring predict --target target_h2 --sequence gru lstm
```

#### Résultats (cible `target_h2`, 69 entrées, sans `known_windows`)

| Méthode | Test : PR-AUC | Test : détection | Test : avance sur R1 | Généralisation : détection | Généralisation : avance |
| --- | --- | --- | --- | --- | --- |
| M2 gradient boosting | 0,89 | 100 % | 1,75 | **100 %** | 1,85 |
| M3 GRU | 0,89 | 100 % | 1,72 | **83 %** | 1,79 |
| M4 LSTM | 0,89 | 100 % | 1,78 | **74 %** | 1,97 |

Sur la corruption jamais vue (`usage_hours_month` × 10) : M2 100 %, GRU **43 %**, LSTM **12 %**.
Un premier entraînement (avec `known_windows`) donnait l'inverse : GRU 12 %, LSTM 39 %.

* Sur les incidents connus, lire la suite des fenêtres n'apporte rien : les tendances
  calculées à la main suffisent, et M2 était déjà à 100 % de détection.
* Sur les incidents jamais vus, les réseaux **surapprennent** : ils ont associé « corruption »
  à `monthly_charges` ; M2 s'appuie aussi sur des signaux généraux (PSI max, hors plage max)
  et généralise.
* Les réseaux sont aussi **instables** : d'un entraînement à l'autre, leur détection de la
  corruption jamais vue passe de 12 % à 43 % (et inversement), alors que M2 reste à 100 %.
* **Modèle retenu : M2**, aussi bon, plus robuste, plus stable, plus rapide et explicable.
  Piste : entraîner les réseaux sur des pannes plus variées ou sur les seuls signaux agrégés.

### 2.4 Consolidation du modèle retenu (`monitoring/analysis.py`)

```powershell
.\.venv\Scripts\python.exe -m monitoring analyze --target target_h2
```

#### Intervalles de confiance à 95 % (bootstrap sur 1 000 tirages des flux)

| Mesure (M2, `target_h2`) | Test | Généralisation |
| --- | --- | --- |
| Détection des flux dégradés | 100 % [100 ; 100] | 100 % [100 ; 100] |
| Avance sur R1 (fenêtres) | 1,75 [1,60 ; 1,90] | 1,85 [1,58 ; 2,13] |
| Fausses alarmes / 100 fenêtres saines | 4,4 [3,9 ; 4,9] | 4,9 [4,3 ; 5,6] |
| R1 : fausses alarmes / 100 fenêtres saines | 7,9 [7,3 ; 8,4] | 7,4 [6,8 ; 8,2] |
| PR-AUC | 0,89 [0,88 ; 0,90] | 0,85 [0,82 ; 0,87] |

Les intervalles de M2 et de R1 pour les fausses alarmes ne se chevauchent pas : M2 en fait
significativement moins que le monitoring réactif, tout en prévenant 1,6 à 1,9 fenêtre plus tôt.

#### Ablation : quelles entrées comptent ?

| Entrées de M2 | Nb | Test : détection | Test : avance sur R1 | Généralisation : détection |
| --- | --- | --- | --- | --- |
| Forme des prédictions seule | 10 | 87 % | 1,71 | 89 % |
| Signaux sans labels (prédictions + entrées) | 27 | 84 % | 1,75 | 90 % |
| Signaux sans labels + tendances | 57 | 89 % | 1,43 | 92 % |
| Métriques déjà connues seules | 12 | 100 % | **0,47** | 99 % |
| Signaux + métriques connues | 39 | 100 % | 1,75 | 99 % |
| Tout | 69 | 100 % | 1,75 | 100 % |

Lecture : **les signaux sans labels donnent la vitesse** (environ 1,7 fenêtre d'avance) mais
ratent les incidents qui ne changent pas les distributions (le concept drift) ; **les
métriques déjà connues donnent l'exhaustivité** (100 %) mais peu d'avance. Il faut les
deux. Les tendances ajoutent peu une fois les métriques connues présentes.

#### Explication des alertes

Importance par permutation (validation) : la dernière log loss connue domine, puis la
proba moyenne prédite, le PSI des scores, la pente de la log loss connue et l'accuracy connue.

Pour une alerte, on désigne les signaux directs les plus éloignés de leur valeur habituelle
(médiane des fenêtres saines, en écarts interquartiles), et on vérifie qu'en les remettant
ensemble à cette valeur, le risque retombe. Exemples sur le test :

| Scénario | Signaux désignés | Risque → si signaux habituels |
| --- | --- | --- |
| Régression de déploiement | PSI des scores 5,17 (habituel 0,02), q10 des probas 0,46 (0,07), proba moyenne 0,73 (0,28) | 1,00 → 0,16 |
| Concept drift | dernière log loss connue 1,44 (0,51), accuracy connue 0,23 (0,75), pente 0,47 (0) | 1,00 → 0,38 |
| Data drift | PSI des scores 1,11 (0,02), part de probas incertaines 0,59 (0,17), médiane des probas 0,48 (0,23) | 1,00 → 0,97 |
| Corruption | hors plage `monthly_charges` 53 % (0 %), part de probas extrêmes 52 % (1 %) | 1,00 → 1,00 |

Pour le data drift et la corruption, le risque reste élevé même sans ces trois signaux :
beaucoup d'autres signaux concordent. L'explication désigne des signaux, jamais une cause ;
le diagnostic reste le travail de l'agent d'analyse.

### 2.5 Alerte prédictive en direct (`monitoring/predictor.py`, `service.py`)

Le modèle retenu (M2, cible `target_h2`) est entraîné une fois puis utilisé fenêtre par
fenêtre par le monitoring, à côté de l'alerte confirmée de l'étape 1.

```powershell
# Une fois : entraîner et enregistrer le prédicteur (monitoring_runs/predictor/predictor.joblib)
.\.venv\Scripts\python.exe -m monitoring train-predictor

# Un flux avec le risque prédit et les alertes prédictives
.\.venv\Scripts\python.exe -m monitoring run --scenario data_corruption --seed 7 --label-delay 2 --predict
```

À chaque fenêtre : les entrées sont calculées **exactement** comme dans le jeu de données
(fonction partagée `dataset.window_features`, vérifié par un test à 10⁻¹² près), puis le
risque de dégradation dans les 2 prochaines fenêtres. Événements émis : `risk_recorded`,
`predictive_alert`, `predictive_alert_closed` (en plus de `metric_recorded` et `alert_triggered`).

Exemple (corruption, début en w008, labels en retard de 2 fenêtres) :

```text
w006  log loss 0.543  risque 0.30  ALERTE CONFIRMÉE warning (pas 8)      ← fluctuation
w007  log loss 0.499  risque 0.29
w008  log loss 7.041  risque 1.00  ALERTE PRÉDICTIVE (pas 8) · ALERTE CONFIRMÉE critical (pas 10)
Alerte prédictive w008 : risque 1.00, statut confirmed, 2 pas avant l'alerte confirmée
  signaux : sig_out_of_range_monthly_charges = 0.276 (habituel 0.000), sig_share_extreme = 0.270 (habituel 0.006)
```

#### Règles de l'alerte prédictive

| Situation | Décision |
| --- | --- |
| Risque ≥ seuil du prédicteur | Alerte `predictive` : risque, horizon, signaux désignés, dernière fenêtre étiquetée |
| Une alerte confirmée porte sur une fenêtre de l'intervalle annoncé (fenêtre prédite … + 2) | Statut `confirmed`, avance mesurée ; l'`alert.json` confirmé contient `preceded_by_predictive_alert` |
| Labels de tout l'intervalle arrivés sans confirmation | Statut `expired` (ou `during_open_incident` si un incident était déjà ouvert : rappel de vigilance) |
| Risque retombé sous le seuil 2 fenêtres de suite | Statut `not_confirmed` |
| Incident **critique** déjà confirmé et ouvert | Pas d'alerte prédictive (inutile) |
| Alerte prédictive émise il y a ≤ 2 fenêtres | Pas de nouvelle alerte (pas de doublon) |

Une alerte `warning` ouverte (souvent une fluctuation) ne bloque **pas** l'alerte prédictive :
sinon une fluctuation masquerait le vrai incident, comme à l'étape 1. Une prédiction expirée
ne peut plus être reliée à un incident ultérieur.

#### Transmission dans le workflow

* **Vigilance** : chaque alerte prédictive est écrite dans
  `monitoring_runs/<run>/predictive_alerts/<alert_id>.json` (sans labels ni cause).
* **Enquête** : l'agent d'analyse reste déclenché par l'alerte confirmée, car il a besoin de
  `feedback.csv`. Son `alert.json` indique l'alerte prédictive qui l'a précédée (fenêtre,
  risque, avance, signaux) ; `monitoring_context.json` contient le risque de chaque fenêtre
  et toutes les alertes prédictives.

#### Bilan sur 40 flux par scénario (début en w008, montée sur 4 fenêtres, labels en retard de 2)

| Scénario | Incidents annoncés par une alerte prédictive | Avance moyenne sur l'alerte confirmée |
| --- | --- | --- |
| Corruption | 39 / 40 | 2,1 pas |
| Data drift | 40 / 40 | 2,4 pas |
| Régression de déploiement | 39 / 40 | 2,1 pas |
| Concept drift | 1 / 40 | — (invisible sans labels, attendu) |
| Flux normaux | — | 35 alertes prédictives sur 800 fenêtres (4,4 / 100, budget 5) |

### Suite prévue

Branchement dans l'interface Sentinel avec la personne de l'agent d'analyse : courbe du
risque à côté de la log loss, alertes `predictive`, `warning` et `critical`, puis enquête.
