"""Content review evidence binding; synthetic labels are never saved as scores."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p1_content_forms', ROOT / 'scripts/p1_content_review_forms.py')
forms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forms)


class ContentReviewContractTest(unittest.TestCase):
    def setUp(self):
        self.expected = forms.contract(forms.DEFAULT_RESULTS, ROOT / 'eval/yuyao_ding/p1_golden_v2.json')

    def synthetic_complete(self):
        value = copy.deepcopy(self.expected)
        value['content_reviewer'] = 'SYNTHETIC CONTRACT TEST ONLY'
        value['content_review_method'] = 'independent_human_source_review'
        for mode in value['modes'].values():
            for row in mode['records']:
                row['answer_correct'] = False
                row['citation_support'] = {k: False for k in row['citation_support']}
        return value

    def test_no_prefilled_judgments_or_groundedness(self):
        self.assertEqual([len(m['records']) for m in self.expected['modes'].values()], [25, 25])
        self.assertEqual([sum(len(r['citation_support']) for r in m['records']) for m in self.expected['modes'].values()], [75, 47])
        for mode in self.expected['modes'].values():
            for row in mode['records']:
                self.assertIsNone(row['answer_correct'])
                self.assertTrue(all(v is None for v in row['citation_support'].values()))
                self.assertFalse(any('groundedness' in k for k in row))
        with self.assertRaises(ValueError):
            forms.validate_submission(self.expected, self.expected)

    def test_valid_complete_review_and_evidence_mutation_rejected(self):
        complete = self.synthetic_complete()
        forms.validate_submission(complete, self.expected)
        for mutate in (
            lambda v: v.update(predictions_sha256='wrong'),
            lambda v: v['modes']['final'].update(run_sha256='wrong'),
            lambda v: v['modes']['final']['records'][0]['candidate'].update(answer='changed answer'),
            lambda v: v['modes']['final']['records'].pop(),
            lambda v: v['modes']['final']['records'][0].update(answer_correct=1),
            lambda v: v['modes']['dense_only']['records'][0]['citation_support'].update(extra=True),
        ):
            with self.subTest(mutation=mutate):
                value = copy.deepcopy(complete)
                mutate(value)
                with self.assertRaises(ValueError):
                    forms.validate_submission(value, self.expected)


if __name__ == '__main__':
    unittest.main()
