import json
import tempfile
import unittest
from pathlib import Path
from beyond_possession.matches import write_catalog, load_catalog, EXPECTED_FILES

class MatchCatalogTests(unittest.TestCase):
    def test_seven_inputs_preserve_records_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            for i,name in enumerate(EXPECTED_FILES):
                p=root/name;p.write_text(json.dumps([{'wyId':i+1,'label':name,'teamsData':{'a':{'lineup':[1,2]}}}]))
                paths.append(p)
            output=root/'combined.json';result=write_catalog(paths,output)
            self.assertEqual(result['summary']['unique_matches'],7)
            self.assertEqual(len(result['sources']),7)
            self.assertEqual(len(load_catalog(output)),7)
            self.assertEqual(result['matches'][0]['teamsData'],{'a':{'lineup':[1,2]}})
            self.assertEqual(len(result['provenance']['1']),1)
    def test_identical_duplicates_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.json';b=root/'b.json'
            a.write_text('[{"wyId":1,"label":"A"}]');b.write_text(a.read_text())
            result=write_catalog([a,b],root/'out.json')
            self.assertEqual(result['summary']['identical_duplicates_removed'],1)
            self.assertEqual(len(result['provenance']['1']),2)
    def test_conflict_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.json';b=root/'b.json';output=root/'out.json'
            a.write_text('[{"id":1,"score":2}]');b.write_text('[{"id":1,"score":3}]');output.write_text('previous')
            with self.assertRaises(ValueError):write_catalog([a,b],output)
            self.assertEqual(output.read_text(),'previous')
    def test_missing_ids_and_invalid_structure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.json'
            for text in ['[{"label":"no id"}]','{}','[5]','[{"wyId":1,"value":NaN}]']:
                a.write_text(text)
                with self.assertRaises(ValueError):write_catalog([a],root/'out.json')
    def test_wrapper_supported_and_input_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.json';a.write_text('{"matches":[{"match_id":"x"}]}')
            result=write_catalog([a],root/'out.json')
            self.assertEqual(result['matches'][0]['match_id'],'x')
            with self.assertRaises(ValueError):write_catalog([a],a)

if __name__=='__main__':unittest.main()
