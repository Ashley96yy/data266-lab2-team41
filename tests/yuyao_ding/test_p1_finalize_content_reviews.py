"""Recorded import recomputation and synthetic corruption checks; no inference."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location('p1_finalizer', ROOT / 'scripts/yuyao_ding/finalize_p1_content_review.py')
forms = importlib.util.module_from_spec(loader)
loader.loader.exec_module(forms)
RESULTS = ROOT / 'eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z'
CONTENT = RESULTS / 'content_review_20261115'
SPEC = ROOT / 'eval/yuyao_ding/p1_golden_v2.json'


class FinalizeContentReviewTest(unittest.TestCase):
    def inputs(self):
        original = forms.read(CONTENT / 'part1-content-review.original.json')
        corrections = forms.read(CONTENT / 'corrections.json')
        corrections['verified_source_review_sha256'] = forms.digest(CONTENT / 'part1-content-review.original.json')
        groundedness = forms.read(RESULTS / 'human_review_20261114/ratings_and_metrics.json')
        return original, corrections, groundedness

    def test_original_preserved_confirmed_correction_and_groundedness_reused(self):
        original, corrections, groundedness = self.inputs()
        before = copy.deepcopy(original)
        out, effective = forms.aggregate(RESULTS, SPEC, original, corrections, groundedness)
        self.assertEqual(original, before)
        self.assertEqual(out, forms.read(CONTENT / 'metrics.json'))
        self.assertTrue(original['modes']['dense_only']['records'][10]['answer_correct'])
        self.assertFalse(effective['modes']['dense_only']['records'][10]['answer_correct'])
        a, b = out['metrics']
        self.assertEqual((a['answer_correct_out_of_25'], b['answer_correct_out_of_25']), (7, 19))
        self.assertEqual((a['supported_citations'], b['supported_citations']), (14, 28))
        self.assertEqual((a['citation_denominator'], b['citation_denominator']), (75, 47))
        self.assertIsNone(a['groundedness_mean'])
        self.assertEqual(b['groundedness_mean'], groundedness['metrics']['groundedness_mean'])
        # A signed decrease is not automatically overruled by token-only matching.
        discrepancy = [r for r in out['records'] if r['numeric_token_discrepancy_requires_context']]
        self.assertEqual(len(discrepancy), 1)
        self.assertTrue(discrepancy[0]['counted_correct'])

    def test_unauthorized_correction_or_wrong_groundedness_rejected(self):
        for failure in ('missing_authority', 'wrong_original', 'wrong_run', 'same_reviewer'):
            a, b, g = self.inputs()
            if failure == 'missing_authority':
                b['corrections'][0]['user_confirmation'] = ''
            elif failure == 'wrong_original':
                b['verified_source_review_sha256'] = 'wrong'
            elif failure == 'wrong_run':
                g['source_run_sha256'] = 'wrong'
            else:
                g['reviewer_b_name'] = g['reviewer_a_name']
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                forms.aggregate(RESULTS, SPEC, a, b, g)


if __name__ == '__main__':
    unittest.main()
