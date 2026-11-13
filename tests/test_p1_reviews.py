"""Synthetic-only checks that human review files cannot bind to another run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tests.test_p1_evaluation import fixture_spec,fixture_run
from src.p1_evaluation import make_review_template

module_path=Path(__file__).resolve().parents[1]/'scripts/p1_review_forms.py'
loader=importlib.util.spec_from_file_location('p1_review_forms',module_path)
review_forms=importlib.util.module_from_spec(loader)
loader.loader.exec_module(review_forms)


class HumanReviewImportTest(unittest.TestCase):
    def test_synthetic_complete_import_and_wrong_hash_rejection(self):
        spec=fixture_spec();run=fixture_run(spec);review=make_review_template(run,spec)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def save(name,value):
                p=root/name;p.write_text(json.dumps(value),encoding='utf-8');return p
            paths=[save('run.json',run),save('spec.json',spec),save('review.json',review)]
            forms=[]
            for label in ['A','B']:
                forms.append(dict(run_sha256=run['run_sha256'],mode='final',
                    reviewer_name=f'SYNTHETIC TEST {label}',independent_human_review=True,
                    records=[dict(question_id=c['question_id'],groundedness=4) for c in spec['cases']]))
            a,b=save('a.json',forms[0]),save('b.json',forms[1]);out=root/'merged.json'
            review_forms.merge(*paths,a,b,out)
            merged=json.loads(out.read_text(encoding='utf-8'))
            self.assertTrue(all(r['reviewer_a_groundedness']==4 for r in merged['records']))
            self.assertIsNone(merged['records'][0]['answer_correct'])
            forms[1]['run_sha256']='another run';b=save('b.json',forms[1])
            with self.assertRaisesRegex(ValueError,'another run'):review_forms.merge(*paths,a,b,root/'rejected.json')
            self.assertFalse((root/'rejected.json').exists())

    def test_same_reviewer_cannot_count_as_both_members(self):
        spec=fixture_spec();run=fixture_run(spec);review=make_review_template(run,spec)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            values=[run,spec,review]
            form=dict(run_sha256=run['run_sha256'],mode='final',reviewer_name='SYNTHETIC TEST',
                      independent_human_review=True,records=[dict(question_id=c['question_id'],groundedness=3) for c in spec['cases']])
            values.extend([form,form])
            paths=[]
            for i,value in enumerate(values):
                p=root/f'{i}.json';p.write_text(json.dumps(value),encoding='utf-8');paths.append(p)
            with self.assertRaisesRegex(ValueError,'same person'):review_forms.merge(*paths,root/'rejected.json')
            self.assertFalse((root/'rejected.json').exists())


if __name__=='__main__':unittest.main()
