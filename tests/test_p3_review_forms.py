"""Review import integrity with explicitly synthetic labels."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.p3_review_forms import export,merge,sha

class ReviewFormsTest(unittest.TestCase):
    def test_named_review_hashes_and_disagreements_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            write=lambda p,x:p.write_text(json.dumps(x),encoding='utf-8')
            modes=['with','without','red_team']
            for mode in modes:
                write(raw/f'{mode}.json',dict(execution_complete=True,records=[dict(case_id='synthetic',rubric=['Synthetic only.'],trace=dict(request='Diagnostic.',input_image='never-copy-portrait.png',vision=None,status='refused',message='Diagnostic.',attempts=[dict(index=0,critique=dict(decision='accept'))]))]))
            write(raw/'manifest.json',dict(status='completed'))
            export(raw,root/'forms')
            self.assertFalse(list((root/'forms/images').iterdir()))
            hashes={m:sha(raw/f'{m}.json') for m in modes}
            form=lambda name,label:dict(raw_execution_hashes=hashes,independent_human_review=True,reviewer_name=name,
                records=[dict(mode=m,case_id='synthetic',pass_=label,attempts=[dict(index=0,human_accept=label)]) for m in modes])
            a,b=form('Synthetic A',True),form('Synthetic B',False)
            for f in [a,b]:
                for r in f['records']:r['pass']=r.pop('pass_')
            write(root/'a.json',a);write(root/'b.json',b)
            merge(raw,root/'a.json',root/'b.json',root/'merged')
            result=json.loads((root/'merged/with.json').read_text())['records'][0]
            self.assertIsNone(result['adjudicated_pass']);self.assertIsNone(result['trace']['attempts'][0]['human_accept'])
            self.assertEqual(len(json.loads((root/'merged/adjudication_needed.json').read_text())),6)
            b['reviewer_name']='Synthetic A';write(root/'b.json',b)
            with self.assertRaises(ValueError):merge(raw,root/'a.json',root/'b.json',root/'duplicate')
            b['reviewer_name']='Synthetic B';b['raw_execution_hashes']={};write(root/'b.json',b)
            with self.assertRaises(ValueError):merge(raw,root/'a.json',root/'b.json',root/'wrong-run')

if __name__=='__main__':unittest.main()
