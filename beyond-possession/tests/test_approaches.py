"""Adversarial checks for the three slide proposals; all labels here are fixtures."""
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from beyond_possession.agents import ALTERNATIVES, catalog
from beyond_possession.analytics import WEIGHTS, summarize
from beyond_possession.data import DataError, demo, validate_events
from beyond_possession.gemini_ai import GeminiBackend
from beyond_possession.narrative import fact_check, validate_decisions
from beyond_possession.pipeline import analyze
from beyond_possession.tci_learning import examples, fit_tci, evaluate_tci, validate_model


def training_fixture(prefix='train', count=3):
    rows=[]
    for match in range(count):
        for interval in range(4):
            for team in ['A','B']:
                for i in range(20):
                    rows.append({'match_id':f'{prefix}-{match}','event_id':f'{interval}-{team}-p{i}',
                        'period':1,'time_seconds':interval*900+i*20,'team':team,'player':team+'1',
                        'receiver':team+'2','type':'pass','outcome':'success' if i<(17 if team=='A' else 12) else 'fail',
                        'x':75 if i<(15 if team=='A' else 5) else 35,'y':50,
                        'end_x':90 if team=='A' else 40,'end_y':50})
                rows.append({'match_id':f'{prefix}-{match}','event_id':f'{interval}-{team}-shot',
                    'period':1,'time_seconds':interval*900+800,'team':team,'player':team+'1',
                    'receiver':'','type':'shot','outcome':'fail','x':90,'y':50,'end_x':100,'end_y':50,
                    'xg':(.7 if team=='A' else .1)+match*.01})
    return rows


def prose(item, text=None, quote=None):
    quote=quote or f"{item['value']*100:.2f}%"
    return {'evidence_id':item['evidence_id'],'evidence_ids':[item['evidence_id']],'value':item['value'],
        'text':text or f"{item['team']} completed {quote} of its passes in this interval.",
        'numbers':[{'quote':quote,'evidence_id':item['evidence_id'],'field':'value'}],
        'alternative_explanation':ALTERNATIVES[0]}


class FakeAnalyst:
    model='fixture-analyst'
    def __init__(self, wrong=False, fail=False, omit=False):
        self.calls=[];self.inputs=[];self.wrong=wrong;self.fail=fail;self.omit=omit
    def select(self,role,evidence,seed):
        item=next((e for e in evidence if e['metric']=='completion_rate'),evidence[0])
        return [{'evidence_id':item['evidence_id'],'value':item['value']+(1 if self.wrong else 0),'alternative_explanation':ALTERNATIVES[0]}]
    def generate_task(self,task,payload,seed):
        self.inputs.append((task,copy.deepcopy(payload)))
        self.calls.append({'role':payload['role']+'_'+task,'model':self.model,'input_tokens':1,'output_tokens':1})
        if task=='write':
            item=next((e for e in payload['evidence'] if e['metric']=='completion_rate'),payload['evidence'][0])
            claim=prose(item)
            if item['metric']!='completion_rate':
                claim['text']=f"{item['team']} recorded {claim['numbers'][0]['quote']} for {item['metric'].replace('_',' ')}."
            if self.wrong and not payload.get('feedback'):
                claim=prose(item,quote='999.00%')
            return {'claims':[claim]}
        decisions=[{'claim_id':c['claim_id'],'verdict':'fail' if self.fail else 'pass',
                    'reason':'Fixture tactical assessment.'} for c in payload['claims']]
        return {'decisions':[] if self.omit else decisions}


class TCILearningTests(unittest.TestCase):
    def test_fit_and_disjoint_test_preserve_simplex_and_provenance(self):
        rows=training_fixture();model=fit_tci(rows)
        self.assertAlmostEqual(sum(model['weights'].values()),1)
        self.assertTrue(all(w>=0 for w in model['weights'].values()))
        self.assertLessEqual(model['training_scores']['learned']['mse'],model['training_scores']['fixed']['mse']+1e-10)
        self.assertEqual(model['features'],list(WEIGHTS))
        self.assertEqual(validate_model(model),model)
        scores=evaluate_tci(model,training_fixture('test',1))
        self.assertEqual(scores['learned']['matches'],1)
        self.assertIsNone(analyze(demo()[0])['summary']['tci_model']['model_id'])
        report=analyze(demo()[0],tci_model=model)
        self.assertEqual(report['summary']['tci_weights'],model['weights'])
        self.assertEqual(report['config']['tci_model']['model_id'],model['model_id'])

    def test_future_labels_never_enter_historical_features(self):
        rows=validate_events(training_fixture())
        before,_=examples(rows)
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['type']=='shot' and 900<=row['time_seconds']<1800:
                row['xg']=.01 if row['team']=='A' else .99
        after,_=examples(changed)
        old=next(e for e in before if e['team']=='A' and e['feature_start']==0)
        new=next(e for e in after if e['team']=='A' and e['feature_start']==0)
        self.assertEqual(old['x'],new['x']);self.assertNotEqual(old['y'],new['y'])
        self.assertEqual(new['target_start'],new['feature_end'])
        self.assertEqual(new['target_end']-new['target_start'],900)

    def test_missing_xg_cannot_be_imputed_or_treated_as_zero(self):
        rows=training_fixture();del next(e for e in rows if e['type']=='shot')['xg']
        with self.assertRaisesRegex(ValueError,'every shot/goal'):fit_tci(rows)
        with self.assertRaisesRegex(ValueError,'three independent'):fit_tci(training_fixture(count=1))

    def test_overlapping_test_tampered_artifact_and_window_are_rejected(self):
        model=fit_tci(training_fixture())
        with self.assertRaisesRegex(ValueError,'overlap'):evaluate_tci(model,training_fixture())
        wrong=copy.deepcopy(model);wrong['weights']['shot_share']=.99
        with self.assertRaisesRegex(ValueError,'fingerprint'):validate_model(wrong)
        with self.assertRaisesRegex(ValueError,'same analysis window'):analyze(demo()[0],tci_model=model,window_minutes=10)

    def test_known_test_or_validation_never_used_for_fitting(self):
        from beyond_possession.dataset_splits import read_manifest
        manifest=read_manifest()
        if not manifest:self.skipTest('No frozen manifest')
        for split in ['test','validation']:
            match=next(k for k,v in manifest['match_splits'].items() if v==split)
            rows=training_fixture()
            for row in rows:
                if row['match_id']=='train-0':row['match_id']=match
            with self.subTest(split=split),self.assertRaises(ValueError):fit_tci(rows)

    def test_optional_xg_validation_and_legacy_fingerprint_compatibility(self):
        rows,_=demo();self.assertNotIn('xg',rows[0])
        for value in [True,-.01,1.1,float('nan')]:
            bad=training_fixture();next(e for e in bad if e['type']=='shot')['xg']=value
            with self.subTest(value=value),self.assertRaises(DataError):validate_events(bad)
        bad=training_fixture();bad[0]['xg']=.5
        with self.assertRaises(DataError):validate_events(bad)

    def test_zero_chance_horizons_and_period_boundaries_are_excluded(self):
        rows=training_fixture()
        for row in rows:
            if row['type']=='shot':row['xg']=0
        self.assertEqual(examples(validate_events(rows))[0],[])
        rows=training_fixture()
        for row in rows:row['period']=1 if row['time_seconds']<1800 else 2
        self.assertEqual(examples(validate_events(rows))[0],[])


class NarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,_=demo();cls.evidence=catalog(summarize(cls.rows))
        cls.item=next(e for e in cls.evidence.values() if e['metric']=='completion_rate')

    def test_all_numbers_bound_with_rounding_percent_context_and_minutes(self):
        claim=prose(self.item)
        self.assertTrue(fact_check(claim,self.evidence)[0])
        claim['text']+=' The interval began at 0 minutes with the score 0 to 0.'
        claim['numbers'] += [{'quote':'0','evidence_id':self.item['evidence_id'],'field':field} for field in ['start_minute','score_for','score_against']]
        self.assertTrue(fact_check(claim,self.evidence)[0])

    def test_unbound_wrong_repeated_and_spelled_numbers_rejected(self):
        for extra in [' There were 99 shots.',' There were two substitutions.',' They had 100% possession.']:
            claim=prose(self.item);claim['text']+=extra
            with self.subTest(extra=extra):self.assertFalse(fact_check(claim,self.evidence)[0])
        self.assertFalse(fact_check(prose(self.item,quote='99.99%'),self.evidence)[0])
        claim=prose(self.item);claim['numbers']*=2
        self.assertFalse(fact_check(claim,self.evidence)[0])

    def test_cross_metric_wrong_unit_and_unknown_citation_fail(self):
        claim=prose(self.item);claim['numbers'][0]['evidence_id']='invented'
        self.assertFalse(fact_check(claim,self.evidence)[0])
        item={**self.item,'metric':'tci','value':75};claim=prose(item)
        self.assertFalse(fact_check(claim,{item['evidence_id']:item})[0])

    def test_number_errors_return_once_for_correction_and_remain_audited(self):
        model=FakeAnalyst(wrong=True)
        report=analyze(self.rows,system='single',model_backend=model,generation_mode='free')
        result=report['analysis']
        self.assertEqual(result['correction_calls'],1)
        self.assertEqual(len(result['claims']),1)
        self.assertEqual(result['rejected'][0]['stage'],'draft_fact_check')
        self.assertTrue(result['claims'][0]['number_check']['passed'])
        self.assertTrue(model.inputs[1][1]['feedback'])

    def test_hybrid_tactical_rejection_and_independent_reviewer(self):
        writer=FakeAnalyst();reviewer=FakeAnalyst(fail=True);reviewer.model='fixture-independent-reviewer'
        report=analyze(self.rows,system='single',model_backend=writer,generation_mode='free',critic_mode='hybrid',critic_model_backend=reviewer)
        self.assertEqual(report['analysis']['claims'],[])
        self.assertEqual(report['analysis']['critic_model'],'fixture-independent-reviewer')
        self.assertEqual(report['analysis']['rejected'][0]['stage'],'ai')
        self.assertEqual([task for task,_ in writer.inputs],['write'])
        self.assertEqual([task for task,_ in reviewer.inputs],['review'])

    def test_numeric_hybrid_blocks_wrong_template_value_before_ai(self):
        model=FakeAnalyst(wrong=True)
        report=analyze(self.rows,system='single',model_backend=model,critic_mode='hybrid')
        self.assertFalse(report['analysis']['claims']);self.assertFalse(model.inputs)

    def test_ai_only_is_explicitly_not_marked_numerically_verified(self):
        model=FakeAnalyst(wrong=True)
        report=analyze(self.rows,system='single',model_backend=model,critic_mode='ai')
        claim=report['analysis']['claims'][0]
        self.assertEqual(claim['verification'],'ai_reviewed')
        self.assertFalse(claim['number_check']['applied']);self.assertFalse(claim['number_check']['passed'])

    def test_missing_duplicate_unknown_or_invalid_ai_decisions_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'omitted'):
            analyze(self.rows,system='single',model_backend=FakeAnalyst(omit=True),critic_mode='hybrid')
        decision={'claim_id':'a','verdict':'pass','reason':'Supported.'}
        for decisions in [[decision,decision],[{**decision,'claim_id':'b'}],[{**decision,'verdict':True}]]:
            with self.assertRaises(ValueError):validate_decisions({'decisions':decisions},{'a'})

    def test_no_game_state_removes_context_from_writer_and_reviewer(self):
        class Scoped(FakeAnalyst):
            def generate_task(self,task,payload,seed):
                if task=='write' and not any(e['metric']=='completion_rate' for e in payload['evidence']):
                    self.inputs.append((task,copy.deepcopy(payload)));return {'claims':[]}
                return super().generate_task(task,payload,seed)
        model=Scoped()
        report=analyze(self.rows,system='no_game_state',model_backend=model,generation_mode='free',critic_mode='hybrid')
        for _,payload in model.inputs:
            self.assertTrue(all('context' not in e for e in payload['evidence']))
        self.assertFalse(report['analysis']['context_enabled'])

    def test_no_critic_and_original_comparison_stay_available(self):
        result=analyze(self.rows,system='no_critic',model_backend=FakeAnalyst(wrong=True),generation_mode='free')['analysis']
        self.assertEqual(result['critic_mode'],'none');self.assertEqual(result['claims'][0]['verification'],'unchecked')
        self.assertEqual(result['correction_calls'],0)
        original=analyze(self.rows,system='single')['analysis']
        self.assertEqual(original['generation_mode'],'template');self.assertEqual(original['critic_mode'],'numeric')
        for options in [{'generation_mode':'free'},{'critic_mode':'hybrid'}]:
            with self.assertRaisesRegex(ValueError,'need an AI backend'):analyze(self.rows,**options)

    def test_google_native_writer_and_optional_critic_model_request(self):
        class Response:
            def __init__(self,key):self.key=key
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({self.key:[]})}]}}]}).encode()
        with patch.dict(os.environ,{'GEMINI_API_KEY':'fixture-only','GEMINI_CRITIC_MODEL':'gemini-fixture-critic'},clear=True),patch('beyond_possession.gemini_ai.load_environment'):
            model=GeminiBackend()
            with patch('urllib.request.urlopen',return_value=Response('claims')) as call:
                model.generate_task('write',{'role':'single_analyst','evidence':[],'feedback':[]},42)
                body=json.loads(call.call_args.args[0].data)
                self.assertIn('numbers',body['generationConfig']['responseJsonSchema']['properties']['claims']['items']['properties'])
            with patch('urllib.request.urlopen',return_value=Response('decisions')) as call:
                model.generate_task('review',{'role':'tactical_critic','claims':[],'evidence':[]},42)
                self.assertIn('/models/gemini-fixture-critic:',call.call_args.args[0].full_url)
            self.assertNotIn('fixture-only',json.dumps(model.calls))

    def test_comparison_packets_blind_approaches_and_keep_human_ratings_blank(self):
        from beyond_possession.evaluation import run_experiment
        with tempfile.TemporaryDirectory() as directory:
            result=run_experiment(self.rows,[],directory,repeats=1,model_backend_factory=FakeAnalyst,compare_approaches=True)
            self.assertEqual(len(result['records']),5)
            self.assertTrue(all(row['status']=='ok' for row in result['records']))
            for file in (Path(directory)/'blinded').glob('*.md'):
                text=file.read_text();self.assertNotIn('fixture-analyst',text);self.assertNotIn('critic_mode',text)

    def test_http_training_model_upload_and_shared_writing_reach_real_pipeline(self):
        from beyond_possession import server
        captured={}
        class FakeServer:
            def __init__(self,address,handler):captured['handler']=handler;self.server_address=address
            def serve_forever(self):pass
            def server_close(self):pass
        with patch.object(server,'ThreadingHTTPServer',FakeServer),patch('builtins.print'):
            server.serve()
        def request(method,path,payload=None,token=''):
            handler=object.__new__(captured['handler']);handler.path=path
            data=json.dumps(payload or {}).encode()
            handler.headers={'Content-Length':str(len(data)),'X-Session-Token':token};handler.rfile=io.BytesIO(data)
            handler.send=lambda status,body,kind='application/json':(status,body)
            return getattr(handler,'do_'+method)()
        token=request('GET','/api/config')[1]['token']
        payload={'training_json':json.dumps(training_fixture()),'test_json':json.dumps(training_fixture('test',1))}
        self.assertEqual(request('POST','/api/train-tci',payload)[0],403)
        status,body=request('POST','/api/train-tci',payload,token)
        self.assertEqual(status,200);self.assertEqual(body['test']['learned']['matches'],1)
        with patch.object(server,'SharedBackend',return_value=FakeAnalyst()):
            status,report=request('POST','/api/analyze',{'source':'demo','system':'single','backend':'shared',
                'generation_mode':'free','critic_mode':'hybrid','tci_model':body['model']},token)
        self.assertEqual(status,200)
        self.assertEqual(report['report']['analysis']['claims'][0]['verification'],'numeric_and_ai_reviewed')
        self.assertEqual(report['report']['summary']['tci_model']['kind'],'learned')
        self.assertIn('AI review: pass',report['markdown'])


if __name__=='__main__':unittest.main()
