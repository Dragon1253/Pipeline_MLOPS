import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from mlops_churn.tools import AnalyticalMethods
from mlops_churn.workspace import InvestigationWorkspace
from mlops_churn.workspace import sha256, write_json


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        frames = {
            'source': pd.DataFrame({'customer_id': ['a','b','c','d'], 'amount': [10.,20.,None,40.], 'category':['x','x','y','y']}),
            'inputs': pd.DataFrame({'customer_id': ['d','c','a','b'], 'amount': [400.,None,100.,20.], 'category':['y','y','x','x']}),
            'predictions': pd.DataFrame({'customer_id':['a','b','c','d'], 'probability':[.1,.8,.7,.2], 'prediction':[0,1,1,0]}),
            'feedback': pd.DataFrame({'customer_id':['d','c','b','a'], 'churn':[0,1,1,0]}),
            'duplicates': pd.DataFrame({'customer_id':['a','a'], 'amount':[1.,2.]}),
            'invalid': pd.DataFrame({'customer_id':['a'], 'probability':[1.2], 'prediction':[1]}),
        }
        resources=[]
        for name,frame in frames.items():
            p=self.root/(name+'.csv');frame.to_csv(p,index=False)
            resources.append({'id':name,'file':p.name,'kind':'csv','sha256':sha256(p),'rows':len(frame),'columns':list(frame)})
        p=self.root/'context.json'
        write_json(p,{'threshold':.5,'performance_history':[{'window':'focus','metrics':{'log_loss':.7},'alert':True},{'window':'next','metrics':{'log_loss':.4},'alert':False}], 'deployment':{'model_changed':True,'events':[]}})
        resources.append({'id':'context','file':p.name,'kind':'json','sha256':sha256(p)})
        write_json(self.root/'manifest.json',{'resources':resources})
        self.workspace=InvestigationWorkspace(self.root)
        self.m=AnalyticalMethods(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def test_deployment_error_lists_real_fields_and_default_reads_all(self):
        with self.assertRaisesRegex(ValueError,'Champs disponibles.*model_changed'):
            self.m.inspect_deployment(fields=['invented'])
        result=self.m.inspect_deployment()
        self.assertEqual(result['deployment'],{'model_changed':True,'events':[]})

    def test_comparison_aligns_and_counts_common_missing(self):
        r=self.m.compare_tables('source','inputs',columns=['amount'])
        self.assertEqual(r['matched_rows'],4)
        self.assertEqual(r['columns'][0]['changed_values'],2)
        self.assertEqual(r['columns'][0]['common_missing_values'],1)
        self.assertEqual(r['columns'][0]['median_right_over_left'],10.)
        self.assertEqual(self.m.compare_tables('inputs','source',columns=['amount'])['columns'][0]['median_right_over_left'],.1)

    def test_extra_rows_are_not_silently_counted_as_paired_changes(self):
        p=self.root/'inputs.csv';f=pd.read_csv(p)
        f.loc[f.customer_id=='d','customer_id']='extra';f.to_csv(p,index=False)
        self.workspace.resources['inputs']['sha256']=sha256(p)
        r=self.m.compare_tables('source','inputs',columns=['amount'])
        self.assertEqual((r['matched_rows'],r['left_only'],r['right_only']),(3,1,1))
        self.assertEqual(r['columns'][0]['changed_values'],1)

    def test_key_requested_as_measure_is_explicitly_excluded(self):
        r=self.m.compare_tables('source','inputs',columns=['customer_id','amount'])
        self.assertEqual(r['excluded_from_measurements'],['customer_id'])
        self.assertEqual(r['columns'][0]['column'],'amount')
        self.assertEqual(r['columns'][0]['changed_values'],2)

    def test_each_tool_dispatches_to_its_own_method(self):
        from mlops_churn.agent import create_tools
        tools,evidence=create_tools(self.workspace,self.root,include_conditional=False,include_model_comparison=False)
        arguments={'describe_resource':{'resource_id':'source'},'check_data_quality':{'resource_id':'source'},
                   'compare_tables':{'left':'source','right':'inputs','columns':['amount']},
                   'compare_distributions':{'left':'source','right':'inputs','columns':['amount']},
                   'evaluate_predictions':{},'compare_windows':{},'inspect_deployment':{}}
        for t in tools:
            if t.name in arguments:
                value=json.loads(t.invoke(arguments[t.name]))
                self.assertEqual(value['tool'],t.name)
                self.assertEqual(value['result']['status'],'ok',value)
        self.assertEqual(len(evidence),7)

    def test_missing_transition_is_a_change_not_a_ratio(self):
        path=self.root/'inputs.csv';f=pd.read_csv(path);f.loc[f.customer_id=='a','amount']=None;f.to_csv(path,index=False)
        self.workspace.resources['inputs']['sha256']=sha256(path)
        c=self.m.compare_tables('source','inputs',columns=['amount'])['columns'][0]
        self.assertEqual(c['changed_values'],2); self.assertEqual(c['missing_transitions'],1);self.assertEqual(c['ratio_count'],1)

    def test_duplicates_and_bad_columns_rejected(self):
        with self.assertRaises(ValueError):self.m.compare_tables('source','duplicates',columns=['amount'])
        with self.assertRaises(ValueError):self.m.describe_resource('source',['secret'])
        with self.assertRaises(ValueError):self.m.describe_resource('../../evaluation/ground_truth')

    def test_description_quality_and_ranges(self):
        r=self.m.describe_resource('source',['amount']);self.assertEqual(r['columns'][0]['missing'],1)
        r=self.m.check_data_quality('inputs',['amount'],key='customer_id',ranges={'amount':[0,50]})
        self.assertEqual(r['columns'][0]['outside_requested_range'],2)
        self.assertEqual(r['duplicate_keys'],0)

    def test_distributions_identical_and_multiple_testing(self):
        r=self.m.compare_distributions('source','source',['amount','category'])
        self.assertTrue(all(c['p_value']==1 for c in r['columns']))
        self.assertTrue(all(c['q_value_bh']==1 for c in r['columns']))
        self.assertEqual(r['columns'][1]['total_variation'],0)

    def test_prediction_join_single_class_groups_and_invalid_probabilities(self):
        r=self.m.evaluate_predictions(group_by='category',group_resource='source')
        self.assertEqual(r['metrics']['accuracy'],1.)
        self.assertEqual(r['metrics']['confusion_matrix'],[[2,0],[0,2]])
        self.assertEqual(len(r['groups']),2)
        with self.assertRaises(ValueError):self.m.evaluate_predictions(predictions='invalid')

    def test_missing_group_is_reported_without_pandas_category_error(self):
        p=self.root/'source.csv';f=pd.read_csv(p);f.loc[f.customer_id=='a','category']=None;f.to_csv(p,index=False)
        self.workspace.resources['source']['sha256']=sha256(p)
        r=self.m.evaluate_predictions(group_by='category',group_resource='source')
        self.assertEqual(r['group_values_missing'],1)
        self.assertEqual(sum(g['rows'] for g in r['groups']),4)

    def test_unrelated_group_population_is_rejected_explicitly(self):
        p=self.root/'source.csv';f=pd.read_csv(p);f['customer_id']='other-'+f.customer_id;f.to_csv(p,index=False)
        self.workspace.resources['source']['sha256']=sha256(p)
        result=self.m.evaluate_predictions(group_by='category',group_resource='source')
        self.assertEqual(result['grouping_status'],'not_performed')
        self.assertIn('Aucun client commun',result['grouping_error'])
        self.assertEqual(result['metrics'],self.m.evaluate_predictions()['metrics'])
        self.assertNotIn('groups',result)

    def test_probability_precision_survives_csv_roundtrip(self):
        p=self.root/'predictions.csv';f=pd.read_csv(p);value=.9999999999999997
        f.loc[0,'probability']=value;f.to_csv(p,index=False)
        self.workspace.resources['predictions']['sha256']=sha256(p)
        self.assertEqual(self.m.frame('predictions').loc[0,'probability'],value)

    def test_windows_and_deployment_do_not_classify_cause(self):
        r=self.m.compare_windows(threshold=.5)
        self.assertEqual(r['reported_alert_count'],1)
        self.assertEqual(r['above_requested_threshold'],1)
        self.assertNotIn('cause',r)
        self.assertTrue(self.m.inspect_deployment()['deployment']['model_changed'])

    def test_langchain_tools_record_evidence_and_capture_errors(self):
        from mlops_churn.agent import create_tools
        tools,evidence=create_tools(self.workspace,self.root,include_conditional=False,include_model_comparison=False)
        self.assertEqual(len(tools),8)
        result=json.loads(next(t for t in tools if t.name=='compare_tables').invoke({'left':'source','right':'inputs','columns':['amount']}))
        self.assertEqual(result['evidence_id'],'E0001')
        self.assertEqual(result['result']['observations']['columns'][0]['changed_values'],2)
        result=json.loads(next(t for t in tools if t.name=='compare_tables').invoke({'left':'source','right':'duplicates','columns':['amount']}))
        self.assertEqual(result['result']['status'],'error')
        self.assertFalse(result['result']['analysis_completed'])
        self.assertIn('ne constitue pas',result['result']['error_scope'])
        self.assertIn('selected_resources',result['result'])
        self.assertEqual(len(evidence),2)
        self.assertTrue(self.workspace.verify())




import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_churn.tools import AnalyticalMethods
from mlops_churn.workspace import InvestigationWorkspace
from mlops_churn.workspace import sha256,write_json


class ConditionalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,change=False,shift=0.):
        rng=np.random.default_rng(812)
        def sample(n,prefix,sign,offset):
            x=rng.normal(size=n)+offset;category=rng.choice(['a','b'],size=n)
            y=rng.binomial(1,1/(1+np.exp(-sign*(-.3+1.4*x+.6*(category=='b')))))
            return pd.DataFrame({'customer_id':[prefix+str(i) for i in range(n)],'amount':x,'category':category,'churn':y})
        ref=sample(800,'r',1,0.);cur=sample(400,'c',-1 if change else 1,shift)
        frames={'reference_clients':ref,'source':cur.drop(columns='churn'),'feedback':cur[['customer_id','churn']]}
        resources=[]
        for name,frame in frames.items():
            path=self.root/(name+'.csv');frame.to_csv(path,index=False)
            resources.append({'id':name,'file':path.name,'kind':'csv','sha256':sha256(path)})
        write_json(self.root/'manifest.json',{'resources':resources})
        self.workspace=InvestigationWorkspace(self.root)
        self.methods=AnalyticalMethods(self.workspace)

    def rewrite(self,name,mutate):
        path=self.workspace.path(name);frame=pd.read_csv(path);mutate(frame);frame.to_csv(path,index=False)
        self.workspace.resources[name]['sha256']=sha256(path)

    def test_changed_conditional_relation_has_heldout_gain(self):
        self.fixture(change=True);result=self.methods.compare_conditional_relationship()
        self.assertLess(result['likelihood_ratio']['p_value'],.01)
        self.assertGreater(result['cross_validation']['loss_gain'],.02)
        self.assertEqual(result['cross_validation']['positive_gain_folds'],5)
        self.assertNotIn('cause',result)
        self.assertEqual(result['features'],['amount','category'])
        self.assertTrue(self.workspace.verify())

    def test_same_relation_and_covariate_shift_do_not_create_concept_signal(self):
        for shift in [0.,1.]:
            with self.subTest(shift=shift):
                self.fixture(shift=shift);result=self.methods.compare_conditional_relationship()
                self.assertFalse(result['likelihood_ratio']['p_value']<=.01 and result['cross_validation']['loss_gain']>=.02)

    def test_feedback_order_is_aligned_by_key(self):
        self.fixture(change=True);before=self.methods.compare_conditional_relationship()
        path=self.workspace.path('feedback');frame=pd.read_csv(path).iloc[::-1];frame.to_csv(path,index=False)
        self.workspace.resources['feedback']['sha256']=sha256(path)
        after=self.methods.compare_conditional_relationship()
        self.assertEqual(before,after)

    def test_target_identifier_partial_feedback_and_invalid_folds_rejected(self):
        self.fixture()
        for columns in [['amount','churn'],['customer_id']]:
            with self.assertRaises(ValueError):self.methods.compare_conditional_relationship(columns=columns)
        with self.assertRaises(ValueError):self.methods.compare_conditional_relationship(folds=2)
        self.rewrite('feedback',lambda f:f.drop(index=f.index[-1],inplace=True))
        with self.assertRaisesRegex(ValueError,'exactement'):self.methods.compare_conditional_relationship()

    def test_missing_labels_nonfinite_features_and_overlap_rejected(self):
        self.fixture();self.rewrite('feedback',lambda f:f.loc.__setitem__((0,'churn'),None))
        with self.assertRaisesRegex(ValueError,'binaires'):self.methods.compare_conditional_relationship()
        self.fixture();self.rewrite('source',lambda f:f.loc.__setitem__((0,'amount'),float('inf')))
        with self.assertRaisesRegex(ValueError,'non finies'):self.methods.compare_conditional_relationship()
        self.fixture();self.rewrite('reference_clients',lambda f:f.loc.__setitem__((0,'customer_id'),'c0'))
        with self.assertRaisesRegex(ValueError,'indépendantes'):self.methods.compare_conditional_relationship()

    def test_conditional_tool_is_opt_in_and_records_actual_method(self):
        from mlops_churn.agent import create_tools
        self.fixture(change=True)
        tools,evidence=create_tools(self.workspace,self.root,include_conditional=True,include_model_comparison=False)
        tool=next(t for t in tools if t.name=='compare_conditional_relationship')
        result=json.loads(tool.invoke({}))
        self.assertEqual(result['result']['status'],'ok')
        self.assertEqual(evidence[0]['tool'],'compare_conditional_relationship')





"""Comparaisons locales : appariement, sens de l'écart et intégrité des modèles."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_churn.tools import AnalyticalMethods
from mlops_churn.workspace import InvestigationWorkspace
from mlops_churn.workspace import sha256, write_json


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'models').mkdir()
        x = np.tile(np.array([-3., -2., -1., 1., 2., 3.]), 12)
        self.inputs = pd.DataFrame({'customer_id': [f'c{i}' for i in range(len(x))], 'x': x})
        self.labels = pd.DataFrame({'customer_id': self.inputs.customer_id, 'churn': (x > 0).astype(int)})
        model = Pipeline([('classifier', LogisticRegression())]).fit(self.inputs[['x']], self.labels.churn)
        baseline = {'pipeline': model, 'features': ['x'], 'threshold': .5}
        active = copy.deepcopy(baseline)
        active['pipeline'].named_steps['classifier'].coef_ *= -1
        active['pipeline'].named_steps['classifier'].intercept_ *= -1
        registry = {}
        for name, bundle in [('reference_model', baseline), ('active_model', active)]:
            p = self.root / 'models' / (name + '.joblib')
            joblib.dump(bundle, p)
            registry[name] = {'file': 'models/' + p.name, 'sha256': sha256(p)}
        write_json(self.root / 'model_manifest.json', {'models': registry})
        resources = []
        for name, frame in [('inputs', self.inputs), ('feedback', self.labels.iloc[::-1])]:
            p = self.root / (name + '.csv')
            frame.to_csv(p, index=False)
            resources.append({'id': name, 'file': p.name, 'kind': 'csv', 'sha256': sha256(p)})
        write_json(self.root / 'manifest.json', {'resources': resources})
        self.workspace = InvestigationWorkspace(self.root)
        self.methods = AnalyticalMethods(self.workspace)

    def rewrite_labels(self, frame):
        p = self.root / 'feedback.csv'
        frame.to_csv(p, index=False)
        self.workspace.resources['feedback']['sha256'] = sha256(p)

    def test_reversed_feedback_is_aligned_and_regression_has_positive_interval(self):
        result = self.methods.compare_model_versions(bootstrap_samples=200)
        self.assertEqual(result['matched_rows'], len(self.inputs))
        self.assertEqual(result['metrics']['reference_model']['accuracy'], 1)
        self.assertEqual(result['metrics']['active_model']['accuracy'], 0)
        self.assertGreater(result['candidate_minus_reference_log_loss'], 0)
        self.assertGreater(result['paired_bootstrap']['lower'], 0)
        self.rewrite_labels(self.labels)
        ordered = self.methods.compare_model_versions(bootstrap_samples=200)
        self.assertEqual(result['candidate_minus_reference_log_loss'], ordered['candidate_minus_reference_log_loss'])

    def test_swapping_models_reverses_sign(self):
        first = self.methods.compare_model_versions(bootstrap_samples=100)
        other = self.methods.compare_model_versions('active_model', 'reference_model', bootstrap_samples=100)
        self.assertAlmostEqual(first['candidate_minus_reference_log_loss'], -other['candidate_minus_reference_log_loss'])
        self.assertAlmostEqual(first['paired_bootstrap']['lower'], -other['paired_bootstrap']['upper'])

    def test_incomplete_or_duplicate_labels_are_rejected(self):
        for labels in [self.labels.iloc[:-1], pd.concat([self.labels, self.labels.iloc[:1]])]:
            self.rewrite_labels(labels)
            with self.assertRaises(ValueError):
                self.methods.compare_model_versions(bootstrap_samples=100)

    def test_unknown_model_and_tampered_file_never_deserialized(self):
        with patch('mlops_churn.tools.joblib.load') as loader:
            with self.assertRaises(ValueError):
                self.methods.compare_model_versions(candidate_model='../../secret.joblib')
            loader.assert_not_called()
            with (self.root / 'models/active_model.joblib').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'changé'):
                self.methods.compare_model_versions()
            loader.assert_not_called()

    def test_tool_is_optional_and_returns_recorded_evidence(self):
        from mlops_churn.agent import create_tools
        tools, evidence = create_tools(self.workspace, self.root, include_model_comparison=True)
        method = next(t for t in tools if t.name == 'compare_model_versions')
        value = json.loads(method.invoke({'bootstrap_samples': 100}))
        self.assertEqual(value['result']['status'], 'ok', value)
        self.assertEqual(evidence[0]['tool'], 'compare_model_versions')
        tools, _ = create_tools(self.workspace, self.root, include_model_comparison=False)
        self.assertNotIn('compare_model_versions', [t.name for t in tools])

    def test_uncontrolled_prompt_has_no_submission_or_acceptance_instructions(self):
        from mlops_churn.agent import OPEN_INVESTIGATION_PROMPT, OPEN_MEMORY_INSTRUCTIONS
        self.assertNotIn('submit_conclusion', OPEN_INVESTIGATION_PROMPT + OPEN_MEMORY_INSTRUCTIONS)
        self.assertIn('compare_model_versions', OPEN_INVESTIGATION_PROMPT)



