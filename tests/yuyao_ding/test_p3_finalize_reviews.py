"""Scoring provenance and rejection checks using synthetic records only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.yuyao_ding.finalize_p3_reviews import finalize
from scripts.yuyao_ding.p3_review_forms import sha, write


class FinalizeReviewsTest(unittest.TestCase):
    def fixture(self, root):
        run = root / 'synthetic-run'
        run.mkdir()
        reviews = [dict(reviewer_name=name, independent_human_review=True, records=[])
                   for name in ('Synthetic A', 'Synthetic B')]
        for mode in ('with', 'without', 'red_team'):
            count = 10 if mode == 'red_team' else 15
            prefix = 'P3-RT' if count == 10 else 'P3-S'
            records = []
            for i in range(1, count + 1):
                case = f'{prefix}{i:02d}'
                generated = mode != 'red_team' and i == 1
                run_id = f'{mode}-{i}'
                attempts = []
                if generated:
                    image = run / 'requests' / run_id / 'output.png'
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b'synthetic hash fixture; not a real inference image')
                    attempts = [dict(index=0, generation=dict(image_path=str(image)),
                                     critique=dict(decision='accept'))]
                records.append(dict(case_id=case, trace=dict(status='completed',
                    run_id=run_id, elapsed_seconds=1.0, attempts=attempts)))
                for index, review in enumerate(reviews):
                    label = not (mode == 'with' and i == 1 and index == 1)
                    review['records'].append(dict(mode=mode, case_id=case,
                        **{'pass': label}, note='', attempts=[dict(index=0, human_accept=label)] if generated else []))
            write(run / f'{mode}.json', dict(set_id='p3_red_team_10' if count == 10 else 'p3_primary_15',
                critique_enabled=mode != 'without', execution_complete=True, records=records))
        write(run / 'manifest.json', dict(source_commit='synthetic', gpu='synthetic CPU fixture', seed=266))
        hashes = {mode: sha(run / f'{mode}.json') for mode in ('with', 'without', 'red_team')}
        for label, review in zip(('A', 'B'), reviews):
            review['raw_execution_hashes'] = hashes
            write(root / f'{label}.json', review)
        joint = dict(schema_version=1, raw_execution_hashes=hashes,
            reviewer_names=['Synthetic A', 'Synthetic B'], joint_human_adjudication=True,
            resolutions=[dict(mode='with', case_id='P3-S01', kind='case', value=False,
                              resolution_note='Synthetic shared decision.'),
                         dict(mode='with', case_id='P3-S01', kind='attempt', index=0,
                              value=False, resolution_note='Synthetic image decision.')])
        write(root / 'joint.json', joint)
        return run, joint

    def invoke(self, root, run):
        return finalize(run, root/'A.json', root/'B.json', root/'joint.json',
                        root/'private', root/'public')

    def test_joint_decisions_preserve_original_ratings_and_hashes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, _ = self.fixture(root)
            originals = {p: sha(p) for p in [root/'A.json', root/'B.json', root/'joint.json', *run.glob('*.json')]}
            metrics = self.invoke(root, run)
            self.assertEqual(metrics['modes']['with']['passes'], 14)
            self.assertEqual(metrics['modes']['with']['reviewer_agreement'], 14/15)
            self.assertEqual(metrics['modes']['with']['critic_agreement'], 0.0)
            self.assertTrue(metrics['paired_generated_images'][0]['identical'])
            self.assertFalse(metrics['inference_rerun'])
            self.assertEqual(json.loads((root/'private/adjudication_needed.json').read_text()), [])
            case = json.loads((root/'public/with.reviewed.json').read_text())['records'][0]
            self.assertTrue(case['reviewer_a_pass'])
            self.assertFalse(case['reviewer_b_pass'])
            self.assertFalse(case['adjudicated_pass'])
            self.assertNotIn('generation', case['trace']['attempts'][0])
            for path, expected in originals.items():
                self.assertEqual(sha(path), expected)

    def test_wrong_run_duplicate_missing_and_unexplained_decisions_rejected(self):
        for issue in ('hash', 'duplicate', 'missing', 'blank_note'):
            with self.subTest(issue=issue), TemporaryDirectory() as temp:
                root = Path(temp)
                run, joint = self.fixture(root)
                if issue == 'hash': joint['raw_execution_hashes']['with'] = 'wrong'
                if issue == 'duplicate': joint['resolutions'].append(joint['resolutions'][0])
                if issue == 'missing': joint['resolutions'].pop()
                if issue == 'blank_note': joint['resolutions'][0]['resolution_note'] = ' '
                write(root/'joint.json', joint)
                with self.assertRaises(ValueError):
                    self.invoke(root, run)
                self.assertFalse((root/'public').exists())


if __name__ == '__main__':
    unittest.main()
