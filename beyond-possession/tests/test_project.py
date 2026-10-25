import copy
import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from beyond_possession.data import demo, validate_events, validate_tracking, DataError, fingerprint
from beyond_possession.analytics import network, window_metrics, summarize, control_surface
from beyond_possession.agents import critic, catalog, ModelError, OllamaBackend
from beyond_possession.pipeline import analyze, save_report
from beyond_possession.evaluation import run_experiment
from beyond_possession.tools import MatchTools

class DataTests(unittest.TestCase):
    def setUp(self):self.rows,self.frames=demo()
    def test_reproducible_demo(self):
        self.assertEqual(fingerprint(*demo()),fingerprint(*demo()))
        self.assertNotEqual(fingerprint(*demo()),fingerprint(*demo(43)))
    def test_duplicate_and_nonfinite_rejected(self):
        with self.assertRaises(DataError):validate_events(self.rows+[self.rows[0]])
        bad=copy.deepcopy(self.rows);bad[0]['x']=float('nan')
        with self.assertRaises(DataError):validate_events(bad)
    def test_invalid_coordinates_and_teams(self):
        bad=copy.deepcopy(self.rows);bad[0]['x']=101
        with self.assertRaises(DataError):validate_events(bad)
        with self.assertRaises(DataError):validate_events([r for r in self.rows if r['team']==self.rows[0]['team']])
    def test_tracking_requires_full_teams(self):
        bad=copy.deepcopy(self.frames[:1]);bad[0]['players']=bad[0]['players'][:2]
        with self.assertRaises(DataError):validate_tracking(bad,self.rows)
    def test_duplicate_tracking_times(self):
        with self.assertRaises(DataError):validate_tracking([self.frames[0],self.frames[0]],self.rows)

class AnalyticTests(unittest.TestCase):
    def setUp(self):self.rows,self.frames=demo()
    def test_half_open_windows(self):
        a=window_metrics(self.rows,[],self.rows[0]['team'],0,900)
        b=window_metrics(self.rows,[],self.rows[0]['team'],900,1800)
        combined=window_metrics(self.rows,[],self.rows[0]['team'],0,1800)
        self.assertEqual(a['n_events']+b['n_events'],combined['n_events'])
    def test_known_passing_graph(self):
        rows=[dict(type='pass',outcome='success',player=a,receiver=b,x=30,y=50,end_x=50,end_y=50)
              for a,b in [('A','B'),('A','B'),('B','C'),('C','A')]]
        net=network(rows)
        self.assertAlmostEqual(net['density'],.5)
        self.assertAlmostEqual(net['hub_share'],.5)
        self.assertEqual(sum(e['count'] for e in net['edges']),4)
    def test_event_only_has_no_spatial_metrics(self):
        result=analyze(self.rows)
        self.assertFalse(result['summary']['tracking_available'])
        self.assertTrue(all('central_control' not in w['metrics'] for w in result['summary']['windows']))
    def test_control_complement(self):
        frame=self.frames[0]
        a=control_surface(frame,frame['home_team']);b=control_surface(frame,frame['away_team'])
        self.assertTrue(all(x['control']+y['control']==1 for x,y in zip(a,b)))
    def test_tci_formula_and_sparse_null(self):
        w=window_metrics(self.rows,self.frames,self.rows[0]['team'],0,900)
        m=w['metrics']
        self.assertAlmostEqual(m['tci'],100*(.3*m['territory_share']+.25*m['progressive_pass_rate']+.2*m['completion_rate']+.25*m['shot_share']))
        sparse=window_metrics(self.rows,[],self.rows[0]['team'],0,1)
        self.assertIsNone(sparse['metrics']['tci'])
        with self.assertRaises(ValueError):summarize(self.rows,[],0)
    def test_game_state_excludes_future_goal(self):
        team='Northbridge FC'
        self.assertEqual(window_metrics(self.rows,[],team,1800,2700)['contexts']['score_at_start'],[0,0])
        self.assertEqual(window_metrics(self.rows,[],team,2700,3600)['contexts']['score_at_start'],[1,0])
    def test_tool_queries(self):
        tools=MatchTools(self.rows,self.frames)
        self.assertTrue(tools.calculate_pitch_control('Northbridge FC',0,15)['available'])
        self.assertEqual(len(tools.retrieve_supporting_events(event_type='goal')),2)
        self.assertIn('differences',tools.compare_game_periods('Northbridge FC',(0,15),(15,30)))

class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows,cls.frames=demo()
    def test_critic_rejects_false_numbers_and_unknown_refs(self):
        evidence=catalog(summarize(self.rows))
        item=next(v for v in evidence.values() if v['sample_count']>=10)
        c={'evidence_id':item['evidence_id'],'value':item['value'],'alternative_explanation':'No causal inference is supported.'}
        self.assertTrue(critic(c,evidence)[0])
        c['value']+=.2;self.assertFalse(critic(c,evidence)[0])
        c['evidence_id']='invented';self.assertFalse(critic(c,evidence)[0])
    def test_injected_model_rejection_and_ablation(self):
        class FakeModel:
            model='test-model';calls=[]
            def select(self,role,evidence,seed):
                e=evidence[0]
                return [{'evidence_id':e['evidence_id'],'value':e['value']+1,'alternative_explanation':'No causal inference is supported.'}]
        guarded=analyze(self.rows,system='multi',backend='ollama',model_backend=FakeModel())
        unchecked=analyze(self.rows,system='no_critic',backend='ollama',model_backend=FakeModel())
        self.assertEqual(len(guarded['analysis']['claims']),0)
        self.assertEqual(len(guarded['analysis']['rejected']),4)
        self.assertTrue(all(c['verification']=='unchecked' for c in unchecked['analysis']['claims']))
        self.assertGreater(len(unchecked['analysis']['claims']),0)
    def test_context_ablated_from_model_inputs_and_claims(self):
        class FakeModel:
            model='test-model';calls=[]
            def select(self,role,evidence,seed):
                assert all('context' not in e for e in evidence)
                return []
        report=analyze(self.rows,system='no_game_state',backend='ollama',model_backend=FakeModel())
        self.assertFalse(report['analysis']['context_enabled'])
        offline=analyze(self.rows,system='no_game_state')
        self.assertTrue(all('context' not in c['evidence'] for c in offline['analysis']['claims']))
    def test_model_failure_is_not_fake_offline_success(self):
        class Broken:
            model='broken';calls=[]
            def select(self,*args):raise ModelError('offline')
        with self.assertRaises(ModelError):analyze(self.rows,backend='ollama',model_backend=Broken())
    def test_model_payload_and_usage_with_mocked_transport(self):
        from unittest.mock import patch
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps({'message':{'content':'{"claims": []}'},'prompt_eval_count':10,'eval_count':4}).encode()
        backend=OllamaBackend('fake')
        with patch('urllib.request.urlopen',return_value=Response()) as call:
            self.assertEqual(backend.select('passing',[],42),[])
            payload=json.loads(call.call_args.args[0].data)
            self.assertFalse(payload['stream']);self.assertEqual(payload['format']['type'],'object')
        self.assertEqual(backend.calls[0]['input_tokens'],10)

class IntegrationTests(unittest.TestCase):
    def test_exports_and_sqlite(self):
        rows,frames=demo()
        with tempfile.TemporaryDirectory() as tmp:
            report=analyze(rows,frames)
            save_report(report,tmp,Path(tmp)/'runs.sqlite')
            self.assertEqual(json.loads((Path(tmp)/'report.json').read_text())['run_id'],report['run_id'])
            with closing(sqlite3.connect(Path(tmp)/'runs.sqlite')) as db:self.assertEqual(db.execute('select count(*) from runs').fetchone()[0],1)
            self.assertIn('Limitations',(Path(tmp)/'report.md').read_text())
    def test_experiment_blinding_and_repeats(self):
        rows,_=demo()
        with tempfile.TemporaryDirectory() as tmp:
            result=run_experiment(rows,[],tmp,repeats=2)
            self.assertEqual(len(result['records']),12)
            self.assertTrue(all(r['status']=='ok' for r in result['records']))
            self.assertEqual(result['reproducibility']['multi']['mean_evidence_jaccard'],1)
            self.assertTrue((Path(tmp)/'blinded/ratings.csv').exists())
            for file in (Path(tmp)/'blinded').glob('*.md'):
                text=file.read_text();self.assertNotIn('backend',text.lower());self.assertNotIn('critic',text.lower())
            with self.assertRaises(ValueError):run_experiment(rows,[],tmp)


class HttpHandlerTests(unittest.TestCase):
    def test_routes_upload_and_token_without_network_socket(self):
        import io
        from unittest.mock import patch
        from beyond_possession import server
        captured={}
        class FakeServer:
            def __init__(self,address,handler):captured['handler']=handler;self.server_address=address
            def serve_forever(self):pass
            def server_close(self):pass
        with patch.object(server,'ThreadingHTTPServer',FakeServer),patch('builtins.print'):
            server.serve()
        handler=captured['handler']
        def request(method,path,payload=None,token=''):
            obj=object.__new__(handler);obj.path=path
            encoded=json.dumps(payload or {}).encode()
            obj.headers={'Content-Length':str(len(encoded)),'X-Session-Token':token}
            obj.rfile=io.BytesIO(encoded)
            obj.send=lambda status,body,kind='application/json':(status,body,kind)
            return getattr(obj,'do_'+method)()
        self.assertEqual(request('GET','/api/health')[0],200)
        self.assertEqual(request('GET','/../../README.md')[0],404)
        self.assertIn(b'Beyond Possession',request('GET','/')[1])
        self.assertEqual(request('POST','/api/analyze',{})[0],403)
        token=request('GET','/api/config')[1]['token']
        code,body,_=request('POST','/api/analyze',{'source':'demo'},token)
        self.assertEqual(code,200);self.assertTrue(body['report']['analysis']['claims'])
        code,body,_=request('POST','/api/analyze',{'source':'upload','events_csv':'bad'},token)
        self.assertEqual(code,400)
        self.assertIn('error',body)

if __name__=='__main__':unittest.main()
