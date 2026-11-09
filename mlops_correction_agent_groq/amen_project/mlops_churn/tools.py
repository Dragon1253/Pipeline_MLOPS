"""Outils analytiques en lecture seule : qualité, dérives, performances et versions."""
import json
import math
import inspect
import functools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from .scenarios import metrics
from .workspace import sha256

def create_analytical_tools(workspace, record, include_conditional=False, include_model_comparison=False):
    from langchain_core.tools import StructuredTool
    methods = AnalyticalMethods(workspace)
    descriptions = {
        'describe_resource': 'Décrire un CSV complet (types, valeurs manquantes, statistiques) ou lire un document JSON public complet. resource_id vient du catalogue.',
        'check_data_quality': 'Contrôler valeurs manquantes, doublons, infinis et plages numériques choisies. ranges associe une colonne à [minimum, maximum]. Ne détecte pas seul un changement d’échelle.',
        'compare_tables': 'Comparer deux CSV pour les mêmes identifiants uniques. right/left donne le sens du ratio. Par défaut toutes les colonnes communes hors clé ; la clé présente dans columns est explicitement exclue des mesures. Valeurs manquantes communes identiques. Retourne nombres modifiés et ratio médian sur valeurs modifiées finies, left non nul.',
        'compare_distributions': 'Comparer les distributions marginales de deux populations. KS et distance Wasserstein pour numériques, chi2 et variation totale pour catégories, correction BH. Choisir les colonnes, exclure les identifiants. Ne démontre aucune cause.',
        'evaluate_predictions': 'Recalculer log loss, accuracy, AUC et matrice de confusion par clé unique. predictions et feedback sont des IDENTIFIANTS DE FICHIERS (défauts predictions et feedback). target, probability et prediction sont des NOMS DE COLONNES (défauts churn, probability et prediction). Omettre les paramètres pour utiliser ces défauts. Option group_by/group_resource pour sous-groupes ; décisions existantes, pas de nouvelle inférence.',
        'compare_windows': 'Lire performance_history d’un document contexte et comparer une métrique entre fenêtres. Retourne les alertes et valeurs observées, sans déclarer une fausse alerte.',
        'inspect_deployment': 'Lire les métadonnées publiques deployment : versions, changements et événements. Omettre fields pour lire tous les champs existants ; ne pas deviner leurs noms. Un changement déclaré ne prouve pas une régression causale.',
    }
    if include_conditional:
        descriptions['compare_conditional_relationship']='Comparer la relation cible/caractéristiques entre référence étiquetée et lot courant + feedback. reference=reference_clients, current=source, feedback=feedback par défaut. Toutes les variables explicatives par défaut, sans cible ni identifiant. Test logistique conditionnel avec période/interactions et gain de log loss en validation croisée. folds de 3 à 10. Ne démontre pas une causalité ni la qualité des labels.'
    if include_model_comparison:
        descriptions['compare_model_versions']='Exécuter deux modèles locaux enregistrés sur exactement les mêmes entrées et labels. Défauts : reference_model=reference_model, candidate_model=active_model, inputs=inputs, feedback=feedback. Identifiants disponibles dans model_catalog. Retourne performances, écart candidat moins référence de log loss et intervalle bootstrap apparié. Ne modifie aucun modèle et ne choisit aucune cause.'
    tools = []
    for name, description in descriptions.items():
        method = getattr(methods, name)
        def build(function, tool_name):
            @functools.wraps(function)
            def execute(*args, **kwargs):
                arguments = inspect.signature(function).bind(*args, **kwargs)
                arguments.apply_defaults()
                def operation(evidence_id):
                    workspace.verify()
                    try:
                        result = function(**arguments.arguments)
                    except Exception as exc:
                        related = {str(v) for k, v in arguments.arguments.items()
                                   if k in {'resource_id','left','right','predictions','feedback','group_resource','context_resource','reference','current'}
                                   and isinstance(v, str) and v in workspace.resources}
                        workspace.verify()
                        return {'status': 'error', 'analysis_completed': False,
                                'error_type': type(exc).__name__, 'error': str(exc),
                                'error_scope': 'Analyse non effectuée : ne constitue pas une anomalie mesurée des données.',
                                'selected_resources': [{'resource_id': name, 'columns': workspace.resources[name].get('columns')}
                                                       for name in sorted(related)],
                                'next_action_hint': 'Vérifier ressources, colonnes et population commune ; choisir une autre méthode ou d’autres paramètres.'}
                    workspace.verify()
                    return {'status': 'ok', 'observations': clean(result),
                            'provenance': 'Ressources publiques du catalogue, vérifiées par SHA256 ; calcul déterministe en lecture seule.'}
                return record(tool_name, dict(arguments.arguments), operation)
            return execute
        tools.append(StructuredTool.from_function(build(method, name), name=name, description=description))
    return tools


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class AnalyticalMethods:
    def __init__(self, workspace):
        self.workspace = workspace

    def frame(self, resource):
        if self.workspace.resources.get(resource, {}).get('kind') != 'csv':
            raise ValueError('Choisir une ressource CSV du catalogue.')
        return pd.read_csv(self.workspace.path(resource), float_precision='round_trip')

    def document(self, resource):
        if self.workspace.resources.get(resource, {}).get('kind') != 'json':
            raise ValueError('Choisir une ressource JSON du catalogue.')
        return json.loads(self.workspace.path(resource).read_text(encoding='utf-8'))

    def compare_model_versions(self, reference_model: str = 'reference_model', candidate_model: str = 'active_model',
                               inputs: str = 'inputs', feedback: str = 'feedback', key: str = 'customer_id',
                               target: str = 'churn', bootstrap_samples: int = 1000, confidence: float = 0.95, seed: int = 42):
        return clean(compare_versions(self, reference_model, candidate_model, inputs, feedback,
                                      key, target, bootstrap_samples, confidence, seed))

    def compare_conditional_relationship(self, reference: str = 'reference_clients', current: str = 'source',
                                         feedback: str = 'feedback', key: str = 'customer_id', target: str = 'churn',
                                         columns: list[str] | None = None, folds: int = 5, seed: int = 42):
        return clean(compare_relationship(self,reference,current,feedback,key,target,columns,folds,seed))

    @staticmethod
    def columns(frame, columns):
        selected = list(frame.columns) if columns is None else columns
        if not selected or len(selected) > 32 or len(set(selected)) != len(selected):
            raise ValueError('Sélectionner de 1 à 32 colonnes distinctes.')
        if set(selected) - set(frame.columns):
            raise ValueError('Colonnes absentes : ' + str(sorted(set(selected) - set(frame.columns))))
        return selected

    @staticmethod
    def unique_key(frame, key):
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('La clé doit exister et être unique, sans valeur manquante ; aucune jointure multiple implicite.')

    def describe_resource(self, resource_id: str, columns: list[str] | None = None):
        item = self.workspace.resources.get(resource_id)
        if item is None:
            raise ValueError('Ressource absente du catalogue.')
        if item['kind'] == 'json':
            return {'kind': 'json', 'document': self.document(resource_id)}
        f = self.frame(resource_id)
        out = []
        for c in self.columns(f, columns):
            s = f[c]; numeric = pd.api.types.is_numeric_dtype(s)
            row = {'column': c, 'dtype': str(s.dtype), 'missing': int(s.isna().sum()), 'unique': int(s.nunique())}
            if numeric:
                finite = s[np.isfinite(s)]
                row.update(mean=finite.mean(), median=finite.median(), minimum=finite.min(), maximum=finite.max())
            else:
                row['top_values'] = s.value_counts().head(5).to_dict()
            out.append(row)
        return clean({'rows': len(f), 'columns': out})

    def check_data_quality(self, resource_id: str, columns: list[str] | None = None, key: str | None = None, ranges: dict[str, list[float]] | None = None):
        f = self.frame(resource_id); selected = self.columns(f, columns)
        ranges = ranges or {}
        if set(ranges) - set(selected):
            raise ValueError('Les plages doivent concerner les colonnes sélectionnées.')
        out = []
        for c in selected:
            s = f[c]; row = {'column': c, 'missing': int(s.isna().sum()), 'non_finite': 0}
            if pd.api.types.is_numeric_dtype(s):
                row['non_finite'] = int((s.notna() & ~np.isfinite(s)).sum())
            if c in ranges:
                bounds = ranges[c]
                if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                    raise ValueError('Une plage contient [minimum, maximum].')
                number = pd.to_numeric(s, errors='coerce')
                row['invalid_numeric'] = int((s.notna() & number.isna()).sum())
                row['outside_requested_range'] = int(((number < bounds[0]) | (number > bounds[1])).sum())
            out.append(row)
        result = {'rows': len(f), 'duplicate_rows': int(f.duplicated().sum()), 'columns': out,
                  'limit': 'Une mise à l’échelle peut rester numérique et sans défaut de schéma ; ces contrôles seuls ne prouvent pas son absence.'}
        if key:
            self.columns(f, [key])
            result.update(missing_keys=int(f[key].isna().sum()), duplicate_keys=int(f[key].duplicated().sum()))
        return clean(result)

    def compare_tables(self, left: str, right: str, key: str = 'customer_id', columns: list[str] | None = None, rtol: float = 1e-7, atol: float = 1e-9):
        if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0:
            raise ValueError('Tolérances finies positives ou nulles.')
        a, b = self.frame(left), self.frame(right)
        self.unique_key(a, key); self.unique_key(b, key)
        selected = self.columns(a, columns if columns is not None else [c for c in a if c != key and c in b])
        self.columns(b, selected)
        excluded=[key] if key in selected else []
        selected=[c for c in selected if c!=key]
        x, y = a.set_index(key), b.set_index(key)
        common = x.index.intersection(y.index); x, y = x.loc[common], y.loc[common]
        out = []
        for c in selected:
            u, v = x[c], y[c]
            both_missing = u.isna() & v.isna()
            numeric = pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v)
            equal = pd.Series(np.isclose(u, v, rtol=rtol, atol=atol, equal_nan=True), index=common) if numeric else ((u == v) | both_missing)
            changed = ~equal
            row = {'column': c, 'changed_values': int(changed.sum()), 'common_missing_values': int(both_missing.sum()),
                   'missing_transitions': int((u.isna() ^ v.isna()).sum())}
            if numeric:
                valid = changed & np.isfinite(u) & np.isfinite(v) & (u != 0)
                row.update(ratio_count=int(valid.sum()), median_right_over_left=(v[valid] / u[valid]).median(),
                           mean_left=u[np.isfinite(u)].mean(), mean_right=v[np.isfinite(v)].mean())
            out.append(row)
        return clean({'left': left, 'right': right, 'matched_rows': len(common), 'left_only': len(a)-len(common),
                      'right_only': len(b)-len(common), 'rtol': rtol, 'atol': atol, 'columns': out,
                      'excluded_from_measurements':excluded})

    def compare_distributions(self, left: str, right: str, columns: list[str] | None = None):
        a, b = self.frame(left), self.frame(right)
        selected = self.columns(a, columns if columns is not None else [c for c in a if c in b and c != 'customer_id'])
        self.columns(b, selected); out = []
        for c in selected:
            u, v = a[c].dropna(), b[c].dropna()
            row = {'column': c, 'left_nonmissing': len(u), 'right_nonmissing': len(v),
                   'left_missing_rate': float(a[c].isna().mean()), 'right_missing_rate': float(b[c].isna().mean())}
            if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
                u, v = u[np.isfinite(u)], v[np.isfinite(v)]
                if len(u) and len(v):
                    test = stats.ks_2samp(u, v)
                    row.update(test='KS', statistic=float(test.statistic), p_value=float(test.pvalue),
                               wasserstein=float(stats.wasserstein_distance(u, v)), mean_left=float(u.mean()), mean_right=float(v.mean()))
                else:
                    row.update(test='KS', p_value=None, reason='Échantillon fini vide.')
            else:
                cats = pd.Index(u.astype(str).unique()).union(pd.Index(v.astype(str).unique()))
                if len(cats) > 100:
                    row.update(test='categorical', p_value=None, reason='Plus de 100 modalités : exclure identifiants ou sélectionner une autre variable.')
                elif len(u) and len(v):
                    cu = u.astype(str).value_counts().reindex(cats, fill_value=0)
                    cv = v.astype(str).value_counts().reindex(cats, fill_value=0)
                    p = stats.chi2_contingency(np.array([cu, cv]), correction=False)[1] if len(cats) > 1 else 1.
                    row.update(test='chi2', p_value=float(p), total_variation=float(abs(cu/len(u)-cv/len(v)).sum()/2))
                else:
                    row.update(test='categorical', p_value=None, reason='Échantillon vide.')
            out.append(row)
        valid = [(i, r['p_value']) for i, r in enumerate(out) if r.get('p_value') is not None]
        ordered = sorted(valid, key=lambda item: item[1]); previous = 1.
        for rank in range(len(ordered)-1, -1, -1):
            i, p = ordered[rank]; previous = min(previous, p * len(ordered)/(rank+1)); out[i]['q_value_bh'] = previous
        return clean({'left_rows': len(a), 'right_rows': len(b), 'columns': out,
                      'limit': 'Tests marginaux et correction BH sur les colonnes sélectionnées ; ni preuve causale, ni test de concept drift. Chi2 approximatif si faibles effectifs.'})

    def evaluate_predictions(self, predictions: str = 'predictions', feedback: str = 'feedback', key: str = 'customer_id', target: str = 'churn', probability: str = 'probability', prediction: str = 'prediction', group_by: str | None = None, group_resource: str | None = None):
        a, b = self.frame(predictions), self.frame(feedback)
        self.unique_key(a, key); self.unique_key(b, key)
        self.columns(a, [probability, prediction]); self.columns(b, [target])
        if {target} & set(a.columns):
            raise ValueError('La cible ne doit pas être déjà présente dans les prédictions.')
        f = a.merge(b[[key, target]], on=key, validate='one_to_one')
        grouping_error = None
        if group_by:
            try:
                if group_resource:
                    g = self.frame(group_resource); self.unique_key(g, key); self.columns(g, [group_by])
                    if group_by in f:
                        raise ValueError('Colonne de groupe déjà présente ; omettre group_resource.')
                    if not f[key].isin(g[key]).any():
                        raise ValueError('Aucun client commun avec la ressource de groupe ; utiliser une ressource décrivant les mêmes clients, pas une population indépendante.')
                    f = f.merge(g[[key, group_by]], on=key, how='left', validate='one_to_one')
                self.columns(f, [group_by])
            except (ValueError, KeyError) as exc:
                grouping_error = str(exc)
        valid = f[target].notna() & f[probability].notna() & f[prediction].notna()
        used = f.loc[valid]
        if not len(used):
            raise ValueError('Aucune prédiction avec cible et probabilité disponibles.')
        if not used[target].isin([0, 1]).all() or not used[prediction].isin([0, 1]).all():
            raise ValueError('Cible et décision binaires 0/1 requises.')
        if not np.isfinite(used[probability]).all() or not used[probability].between(0, 1).all():
            raise ValueError('Probabilités finies entre 0 et 1 requises.')
        def measure(g):
            y, p = g[target], g[probability]
            return {'rows': len(g), 'positive_labels': int(y.sum()), 'positive_predictions': int(g[prediction].sum()),
                    'log_loss': float(log_loss(y, p, labels=[0, 1])), 'accuracy': float(accuracy_score(y, g[prediction])),
                    'roc_auc': float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
                    'confusion_matrix': confusion_matrix(y, g[prediction], labels=[0, 1]).tolist()}
        result = {'prediction_rows': len(a), 'feedback_rows': len(b), 'matched_rows': len(f), 'excluded_missing': int((~valid).sum()), 'metrics': measure(used)}
        if group_by and grouping_error is None:
            groups = [(k, g) for k, g in used.groupby(group_by, dropna=False)]
            result['groups'] = [{'group': str(k), **measure(g)} for k, g in groups[:20]]
            result['groups_truncated'] = len(groups) > 20
            result['group_values_missing'] = int(used[group_by].isna().sum())
            result['grouping_status'] = 'ok'
        elif group_by:
            result['grouping_status'] = 'not_performed'
            result['grouping_error'] = grouping_error
            result['grouping_limit'] = 'Les métriques globales sont valides ; le regroupement demandé n’a pas été effectué.'
        return clean(result)

    def compare_windows(self, context_resource: str = 'context', metric: str = 'log_loss', threshold: float | None = None):
        doc = self.document(context_resource)
        rows = []
        for window in doc.get('performance_history', []):
            value = window.get('metrics', {}).get(metric)
            rows.append({'window': window.get('window'), 'value': value, 'reported_alert': window.get('alert'),
                         'labeled_rows': window.get('labeled_rows')})
        if not rows:
            raise ValueError('Aucun historique performance_history dans cette ressource.')
        result = {'metric': metric, 'windows': rows, 'reported_alert_count': sum(r['reported_alert'] is True for r in rows),
                  'limit': 'La persistance ou la disparition d’une alerte ne prouve pas, seule, une cause ou une fausse alerte.'}
        if threshold is not None:
            result['above_requested_threshold'] = sum(r['value'] is not None and r['value'] > threshold for r in rows)
        if metric == 'log_loss':
            result['context_threshold'] = doc.get('threshold')
        return clean(result)

    def inspect_deployment(self, context_resource: str = 'context', fields: list[str] | None = None):
        doc = self.document(context_resource)
        deployment = doc.get('deployment')
        if not isinstance(deployment, dict):
            raise ValueError('Aucune métadonnée deployment disponible.')
        fields = list(deployment) if fields is None else fields
        if set(fields) - set(deployment):
            raise ValueError('Champs de déploiement absents : '+str(sorted(set(fields)-set(deployment)))+
                             '. Champs disponibles : '+str(sorted(deployment))+'. Omettre fields pour tout lire.')
        result={'deployment': {k: deployment[k] for k in fields},
                'limit': 'Métadonnées déclaratives : sans prédictions comparables des deux versions, la régression causale n’est pas prouvée.'}
        return result


def preprocessor(frame):
    numeric=list(frame.select_dtypes(include='number').columns)
    categorical=[c for c in frame if c not in numeric]
    transforms=[]
    if numeric:
        transforms.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                                               ('scale',StandardScaler())]),numeric))
    if categorical:
        transforms.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='__missing__')),
            ('encode',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),categorical))
    return ColumnTransformer(transforms)


def fit_logistic(x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(penalty=None,solver='lbfgs',max_iter=3000,tol=1e-8).fit(x,y)
    return model


def expanded(x,period):
    return np.column_stack([x,period,x*period[:,None]])


def compare_relationship(methods, reference, current, feedback, key, target, columns, folds, seed):
    if not 3<=folds<=10:raise ValueError('folds doit être compris entre 3 et 10.')
    if reference==current:raise ValueError('Deux populations indépendantes distinctes sont requises.')
    a,b,labels=methods.frame(reference),methods.frame(current),methods.frame(feedback)
    for frame in [a,b,labels]:methods.unique_key(frame,key)
    methods.columns(a,[target]);methods.columns(labels,[target])
    if target in b:raise ValueError('La cible courante doit venir du feedback séparé.')
    if len(set(a[key]) & set(b[key])):raise ValueError('Populations supposées indépendantes : identifiants communs non pris en charge.')
    if set(b[key])!=set(labels[key]):raise ValueError('Le feedback doit couvrir exactement toutes les clés courantes.')
    selected=methods.columns(a,columns if columns is not None else [c for c in b if c in a and c not in {key,target}])
    if set(selected)&{key,target}:raise ValueError('Exclure identifiant et cible des variables explicatives.')
    methods.columns(b,selected)
    joined=b.merge(labels[[key,target]],on=key,validate='one_to_one')
    for frame in [a,joined]:
        if len(frame)<100:raise ValueError('Au moins 100 observations par période sont nécessaires.')
        if frame[target].isna().any() or not frame[target].isin([0,1]).all():raise ValueError('Labels binaires complets requis.')
        if frame[target].value_counts().reindex([0,1],fill_value=0).min()<max(10,folds):
            raise ValueError('Au moins 10 exemples de chaque classe par période sont requis.')
    x=pd.concat([a[selected],joined[selected]],ignore_index=True)
    if any(x[c].isna().all() for c in x):raise ValueError('Une variable explicative est entièrement manquante.')
    for c in x.select_dtypes(include='number'):
        if not np.isfinite(x[c].dropna()).all():raise ValueError('Variables numériques non finies.')
    if any(x[c].nunique()>100 for c in x.select_dtypes(exclude='number')):
        raise ValueError('Plus de 100 modalités : exclure les identifiants et variables de cardinalité excessive.')
    y=np.r_[a[target].to_numpy(),joined[target].to_numpy()].astype(int)
    period=np.r_[np.zeros(len(a)),np.ones(len(joined))]
    # Transformations ajustées sans cible ; le LRT compare deux modèles emboîtés sur les mêmes lignes.
    z=preprocessor(x).fit_transform(x);z_alt=expanded(z,period)
    rank_shared=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(z)),z])))
    rank_alt=int(np.linalg.matrix_rank(np.column_stack([np.ones(len(z)),z_alt])))
    df=rank_alt-rank_shared
    if df<=0 or len(x)<=10*rank_alt:raise ValueError('Modèles non identifiables ou trop de paramètres pour les observations.')
    shared=fit_logistic(z,y);alternative=fit_logistic(z_alt,y)
    null_loss=float(log_loss(y,shared.predict_proba(z)[:,1],labels=[0,1]))
    alt_loss=float(log_loss(y,alternative.predict_proba(z_alt)[:,1],labels=[0,1]))
    lr=2*len(y)*(null_loss-alt_loss)
    if lr < -1e-6:raise ValueError('Ajustement incohérent : le modèle emboîtant ajuste moins bien les données.')
    lr=max(0.,lr)
    # Validation croisée stratifiée sur période ET classe, prétraitement réajusté dans chaque pli.
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    p_shared=np.zeros(len(y));p_alt=np.zeros(len(y));fold_results=[]
    for i,(train,test) in enumerate(splitter.split(x,period*2+y),1):
        transform=preprocessor(x.iloc[train]);train_x=transform.fit_transform(x.iloc[train]);test_x=transform.transform(x.iloc[test])
        first=fit_logistic(train_x,y[train]);second=fit_logistic(expanded(train_x,period[train]),y[train])
        p_shared[test]=first.predict_proba(test_x)[:,1]
        p_alt[test]=second.predict_proba(expanded(test_x,period[test]))[:,1]
        loss1=float(log_loss(y[test],p_shared[test],labels=[0,1]));loss2=float(log_loss(y[test],p_alt[test],labels=[0,1]))
        fold_results.append({'fold':i,'test_rows':len(test),'loss_shared':loss1,'loss_period_interactions':loss2,'loss_gain':loss1-loss2})
    loss1=float(log_loss(y,p_shared,labels=[0,1]));loss2=float(log_loss(y,p_alt,labels=[0,1]))
    return {'method':'conditional_logistic_period_interactions',
            'tested_quantity':'Relation cible/caractéristiques conditionnellement à X selon la période ; pas distribution marginale des caractéristiques.',
            'null_hypothesis':'Même relation logistique cible/X dans les deux périodes.',
            'alternative_hypothesis':'La période modifie l’interception et/ou les effets de X sur la cible.',
            'reference_resource':reference,'current_resource':current,
            'feedback_resource':feedback,'features':selected,'reference_rows':len(a),'current_rows':len(joined),
            'matched_feedback_rows':len(joined),'excluded_labels':0,'reference_positive_rate':float(a[target].mean()),
            'current_positive_rate':float(joined[target].mean()),
            'likelihood_ratio':{'statistic':lr,'degrees_of_freedom':df,'p_value':float(stats.chi2.sf(lr,df)),
                                'rank_shared':rank_shared,'rank_alternative':rank_alt},
            'cross_validation':{'folds':folds,'seed':seed,'loss_shared':loss1,'loss_period_interactions':loss2,
                                'loss_gain':loss1-loss2,'positive_gain_folds':sum(r['loss_gain']>0 for r in fold_results),
                                'fold_results':fold_results},
            'assumptions':['Échantillons indépendants et labels fiables.',
                           'Logit linéaire après imputation/encodage ; alternative avec période et interactions période×variables.',
                           'Approximation asymptotique chi2 ; une mauvaise spécification ou un support insuffisant peut tromper le test.'],
            'limit':'Preuve conditionnelle paramétrique, pas test universel de P(y|X), ni preuve de causalité ou de fiabilité des labels. Les modèles auxiliaires ne remplacent pas le modèle servi.'}


def compare_versions(methods, reference_model, candidate_model, inputs, feedback,
                     key, target, bootstrap_samples, confidence, seed):
    if reference_model == candidate_model:
        raise ValueError('Choisir deux identifiants de modèle distincts.')
    if not 100 <= bootstrap_samples <= 5000 or not 0.8 <= confidence <= 0.99:
        raise ValueError('bootstrap_samples entre 100 et 5000 ; confidence entre 0.8 et 0.99.')
    data, labels = methods.frame(inputs), methods.frame(feedback)
    for frame in (data, labels):
        methods.unique_key(frame, key)
    methods.columns(labels, [target])
    if target in data:
        raise ValueError('La cible doit provenir du feedback séparé, jamais des entrées.')
    if set(data[key]) != set(labels[key]):
        raise ValueError('Le feedback doit couvrir exactement les mêmes clients que les entrées.')
    if not 2 <= len(data) <= 100000:
        raise ValueError('La comparaison exige de 2 à 100000 clients.')
    joined = data[[key]].merge(labels[[key, target]], on=key, how='left', validate='one_to_one', sort=False)
    y = joined[target].to_numpy()
    if joined[target].isna().any() or not joined[target].isin([0, 1]).all():
        raise ValueError('Labels binaires complets requis.')
    paths = {name: methods.workspace.model_path(name) for name in (reference_model, candidate_model)}
    hashes = {name: sha256(path) for name, path in paths.items()}
    bundles = {name: joblib.load(path) for name, path in paths.items()}
    if bundles[reference_model]['features'] != bundles[candidate_model]['features']:
        raise ValueError('Les deux modèles doivent utiliser les mêmes variables explicatives ordonnées.')
    features = list(bundles[reference_model]['features'])
    if key in features or target in features:
        raise ValueError('Identifiant ou cible présent dans les variables du modèle.')
    methods.columns(data, features)
    scores, measures, versions = {}, {}, {}
    for name, bundle in bundles.items():
        threshold = bundle['threshold']
        if not 0 <= threshold <= 1:
            raise ValueError('Seuil de décision invalide.')
        classes = list(bundle['pipeline'].classes_)
        if set(classes) != {0, 1}:
            raise ValueError('Modèle binaire 0/1 requis.')
        probabilities = np.asarray(bundle['pipeline'].predict_proba(data[features])[:, classes.index(1)], dtype=float)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Probabilités non finies ou hors [0,1].')
        scores[name] = probabilities
        measures[name] = metrics(y, probabilities, threshold)
        versions[name] = {'sha256': hashes[name], 'threshold': threshold}
    def losses(p):
        p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    delta = losses(scores[candidate_model]) - losses(scores[reference_model])
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 32):
        end = min(start + 32, bootstrap_samples)
        indices = rng.integers(0, len(data), size=(end - start, len(data)))
        samples[start:end] = delta[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    for name in paths:
        if sha256(paths[name]) != hashes[name]:
            raise ValueError('Modèle modifié pendant la comparaison.')
    methods.workspace.verify()
    return {'input_resource': inputs, 'feedback_resource': feedback, 'key': key, 'target': target,
            'matched_rows': len(data), 'excluded_rows': 0, 'features': features,
            'reference_model': reference_model, 'candidate_model': candidate_model,
            'model_versions': versions, 'metrics': measures,
            'candidate_minus_reference_log_loss': float(delta.mean()),
            'paired_bootstrap': {'samples': bootstrap_samples, 'seed': seed, 'confidence': confidence,
                                 'lower': float(low), 'upper': float(high),
                                 'method': 'percentile bootstrap des clients appariés, mêmes indices pour les deux modèles'},
            'changed_probability_rows': int((scores[reference_model] != scores[candidate_model]).sum()),
            'direction': 'Un écart de log loss positif signifie que le candidat est moins performant sur ce lot.',
            'limitations': ['Comparaison conditionnelle au lot et aux labels disponibles ; pas une garantie de généralisation.',
                            'Bootstrap supposant des clients indépendants ; aucun réentraînement ni modification du déploiement.']}
