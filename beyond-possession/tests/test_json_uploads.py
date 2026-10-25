import json
import unittest
from beyond_possession.data import DataError, demo, parse_json_events

class JsonUploadTests(unittest.TestCase):
    def test_normalized_event_json_matches_existing_data(self):
        rows,_=demo()
        self.assertEqual(parse_json_events(json.dumps(rows)),rows)
        self.assertEqual(parse_json_events(json.dumps({'events':rows})),rows)
    def test_catalogue_has_actionable_error(self):
        for data in [[{'wyId':123,'teamsData':{}}],{'matches':[{'wyId':123}]}]:
            with self.assertRaisesRegex(DataError,'[Mm]atch'):
                parse_json_events(json.dumps(data))
    def test_invalid_json_and_wrong_shape(self):
        for text in ['bad json','3','{}']:
            with self.assertRaises(DataError):parse_json_events(text)
    def test_uploaded_match_files_merge_via_http_handler(self):
        import io
        from unittest.mock import patch
        from beyond_possession import server
        captured={}
        class FakeServer:
            def __init__(self,address,handler):captured['handler']=handler;self.server_address=address
            def serve_forever(self):pass
            def server_close(self):pass
        with patch.object(server,'ThreadingHTTPServer',FakeServer),patch('builtins.print'):server.serve()
        Handler=captured['handler']
        def request(path,method='GET',data=None,token=''):
            obj=object.__new__(Handler);obj.path=path
            body=json.dumps(data or {}).encode();obj.rfile=io.BytesIO(body)
            obj.headers={'Content-Length':str(len(body)),'X-Session-Token':token}
            obj.send=lambda status,body,kind='application/json':(status,body)
            return getattr(obj,'do_'+method)()
        token=request('/api/config')[1]['token']
        payload={'files':[{'name':'England.json','text':'[{"wyId":1,"label":"A vs B"}]'},{'name':'France.json','text':'[{"wyId":2,"label":"C vs D"}]'}]}
        status,body=request('/api/combine-matches','POST',payload,token)
        self.assertEqual(status,200)
        self.assertEqual(body['catalog']['summary']['unique_matches'],2)
        self.assertEqual(body['catalog']['sources'][0]['filename'],'England.json')
        rows,_=demo()
        status,body=request('/api/analyze','POST',{'source':'upload','events_json':json.dumps(rows)},token)
        self.assertEqual(status,200)
        self.assertFalse(body['report']['summary']['tracking_available'])
