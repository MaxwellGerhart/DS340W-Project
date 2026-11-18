import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from html.parser import HTMLParser

from beyond_possession.agents import CONFIGS
from beyond_possession.analytics import summarize
from beyond_possession.data import demo
from beyond_possession.outlook import team_outlook, ai_payload, validate_ai, explain_outlook, _project
from beyond_possession.pipeline import analyze, markdown
from test_approaches import FakeAnalyst


class Coach(FakeAnalyst):
    def generate_task(self,task,payload,seed):
        if task!='outlook':return super().generate_task(task,payload,seed)
        self.inputs.append((task,copy.deepcopy(payload)))
        self.calls.append({'role':'manager_analyst_outlook','model':self.model,'input_tokens':10,'output_tokens':10})
        return {'summary':'If recent patterns continue, progression could remain a useful focus for this team.',
            'evidence_ids':[payload['evidence'][0]['evidence_id']],
            'recommendations':[{'priority_id':p['priority_id'],'rationale':'The cited recent metrics support testing this priority.',
                'match_action':p['match_action'],'practice_action':p['practice_action'],'watch_for':p['watch_for']}
                for p in payload['priorities'][:2]]}


class OutlookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,_=demo();cls.summary=summarize(cls.rows)
        cls.team=cls.summary['teams'][0]
    def outlook(self,start=1800):return team_outlook(copy.deepcopy(self.summary),self.team,start)

    def test_forecast_is_bounded_and_tci_uses_selected_weights(self):
        self.assertAlmostEqual(_project([.3,.6]),.65)
        self.assertLessEqual(_project([.9,1]),1)
        outlook=self.outlook()
        rates={p['metric']:p for p in outlook['projections']}
        self.assertEqual(outlook['status'],'ready')
        for metric,p in rates.items():self.assertTrue(0<=p['estimate']<=(100 if metric=='tci' else 1))
        if 'tci' in rates:
            expected=100*sum(self.summary['tci_weights'][k]*rates[k]['estimate'] for k in self.summary['tci_weights'])
            self.assertAlmostEqual(rates['tci']['estimate'],expected)
        self.assertNotIn('win_probability',outlook)

    def test_future_values_never_change_an_earlier_outlook_or_prompt(self):
        before=self.outlook(900)
        changed=copy.deepcopy(self.summary)
        for window in changed['windows']:
            if window['start']>=1800:
                for key,value in window['metrics'].items():
                    if value is not None:window['metrics'][key]=999999
                window['contexts']['score_at_start']=[99,99]
        after=team_outlook(changed,self.team,900)
        self.assertEqual(before,after)
        payload=ai_payload(after)
        self.assertNotIn('duration_seconds',payload)
        self.assertTrue(all(e['team']==self.team and e['end']<=after['as_of_seconds'] for e in payload['evidence']))
        self.assertLessEqual(len(payload['evidence']),24)

    def test_partial_sparse_missing_data_and_first_window_are_explicit(self):
        self.assertEqual(self.outlook(0)['status'],'insufficient_history')
        summary=copy.deepcopy(self.summary)
        selected=next(w for w in summary['windows'] if w['team']==self.team and w['start']==1800)
        selected['end']=1900
        out=team_outlook(summary,self.team,1800)
        self.assertEqual(out['as_of_seconds'],1800)
        self.assertTrue(any('partial' in warning for warning in out['warnings']))
        for window in summary['windows']:
            if window['team']==self.team:window['metrics']['completion_rate']=None
        out=team_outlook(summary,self.team,1800)
        self.assertNotIn('completion_rate',[p['metric'] for p in out['projections']])
        self.assertNotIn('tci',[p['metric'] for p in out['projections']])

    def test_unknown_team_or_cutoff_and_tracking_inferences_are_rejected(self):
        for team,start in [('Unknown',0),(self.team,-1),(self.team,True)]:
            with self.assertRaises(ValueError):team_outlook(self.summary,team,start)
        out=self.outlook()
        self.assertNotIn('central_space',[p['priority_id'] for p in out['priorities']])

    def test_ai_uses_only_allowed_priorities_and_exports_its_provenance(self):
        out=self.outlook();model=Coach();result=explain_outlook(out,model)
        self.assertEqual(result['ai']['model'],'fixture-analyst')
        self.assertEqual(len(result['ai']['model_calls']),1)
        self.assertTrue(result['ai']['recommendations'])
        self.assertIsNone(out['ai'])
        report=analyze(self.rows,system='single')
        for i,o in enumerate(report['outlooks']):
            if o['team']==result['team'] and o['selected_interval_start']==result['selected_interval_start']:
                report['outlooks'][i]=result
        # Markdown exports the latest outlook for each team, and JSON retains
        # every interval including the explicitly requested historical outlook.
        self.assertIn('Team outlooks and actions',markdown(report))

    def test_numeric_inventions_guarantees_bad_refs_and_unknown_priorities_fail(self):
        out=self.outlook();model=Coach();valid=model.generate_task('outlook',ai_payload(out),42)
        self.assertTrue(validate_ai(valid,out))
        for key,value in [('summary','The team will win with 90% certainty.'),('evidence_ids',['future:invented'])]:
            candidate=copy.deepcopy(valid);candidate[key]=value
            with self.assertRaises(ValueError):validate_ai(candidate,out)
        bad=copy.deepcopy(valid);bad['recommendations'][0]['priority_id']='invented_formation'
        with self.assertRaises(ValueError):validate_ai(bad,out)
        bad=copy.deepcopy(valid);bad['recommendations'][0]['practice_action']='The injured player definitely needs treatment.'
        with self.assertRaises(ValueError):validate_ai(bad,out)

    def test_all_research_modes_remain_in_python_and_only_two_are_in_the_menu(self):
        self.assertEqual(len(CONFIGS),6)
        for system in CONFIGS:
            self.assertEqual(analyze(self.rows,system=system)['analysis']['system'],system)
        class Parser(HTMLParser):
            def __init__(self):super().__init__();self.inside=False;self.options=[]
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if tag=='select' and attrs.get('id')=='system':self.inside=True
                if tag=='option' and self.inside:self.options.append(attrs.get('value'))
            def handle_endtag(self,tag):
                if tag=='select':self.inside=False
        parser=Parser();parser.feed((Path(__file__).parents[1]/'beyond_possession/static/index.html').read_text())
        self.assertEqual(parser.options,['single','multi'])

    def test_http_outlook_handles_cache_filters_and_exports(self):
        from beyond_possession import server
        captured={}
        class FakeServer:
            def __init__(self,address,handler):captured['handler']=handler;self.server_address=address
            def serve_forever(self):pass
            def server_close(self):pass
        with patch.object(server,'ThreadingHTTPServer',FakeServer),patch('builtins.print'):server.serve()
        def request(method,path,payload=None,token=''):
            obj=object.__new__(captured['handler']);obj.path=path;raw=json.dumps(payload or {}).encode()
            obj.headers={'Content-Length':str(len(raw)),'X-Session-Token':token};obj.rfile=io.BytesIO(raw)
            obj.send=lambda status,body,kind='application/json':(status,body)
            return getattr(obj,'do_'+method)()
        token=request('GET','/api/config')[1]['token'];model=Coach()
        with patch.object(server,'SharedBackend',return_value=model):
            status,data=request('POST','/api/analyze',{'source':'demo','backend':'shared','system':'single'},token)
            self.assertEqual(status,200)
            payload={'report_handle':data['report_handle'],'team':self.team,'interval_start':1800}
            self.assertEqual(request('POST','/api/outlook',payload)[0],403)
            status,result=request('POST','/api/outlook',payload,token)
            self.assertEqual(status,200);self.assertTrue(result['outlook']['ai'])
            self.assertIn('AI outlook (fixture-analyst)',result['markdown'])
            self.assertEqual(request('POST','/api/outlook',payload,token)[1]['cached'],True)
            self.assertEqual(sum(task=='outlook' for task,_ in model.inputs),1)
            self.assertEqual(request('POST','/api/outlook',{**payload,'team':'Other'},token)[0],400)
            self.assertEqual(request('POST','/api/outlook',{**payload,'report_handle':'old'},token)[0],404)
        # Restarted servers cannot accidentally approve a client-submitted
        # summary: only opaque handles for stored successful reports are used.
        self.assertNotIn('summary',payload)


if __name__=='__main__':unittest.main()
