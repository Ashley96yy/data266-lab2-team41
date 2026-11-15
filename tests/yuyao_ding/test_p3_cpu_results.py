"""Verify recorded CPU aggregation, unchanged input hashes and prior GPU cells."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.yuyao_ding.p3_evaluation import summarize
from src.yuyao_ding.repository_paths import recorded_notebook_snapshot_digest, member_migration

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = ROOT / 'eval/yuyao_ding/part3/results/t4_20261114T060249Z'
NOTEBOOK = ROOT / 'notebooks/Yuyao_Ding/p3_agents.ipynb'


class CpuReviewResultsTest(unittest.TestCase):
    def test_saved_table_matches_supplied_reviews_and_original_metric_record(self):
        computed = json.loads((REVIEWS / 'cpu_result_table.json').read_text())
        existing = json.loads((REVIEWS / 'metrics.json').read_text())
        for name, digest in computed['input_hashes'].items():
            self.assertEqual(hashlib.sha256((REVIEWS / name).read_bytes()).hexdigest(), digest)
        for mode in ('with', 'without', 'red_team'):
            reviewed = json.loads((REVIEWS / f'{mode}.reviewed.json').read_text())
            actual = summarize(reviewed)
            self.assertEqual(actual, existing['modes'][mode])
            for key, value in actual.items():
                self.assertEqual(computed['modes'][mode][key], value)
            self.assertIs(computed['modes'][mode]['human_review_complete'], True)

    def test_actual_cpu_output_is_saved_and_prior_snapshot_is_preserved(self):
        notebook = json.loads(NOTEBOOK.read_text())
        proof = notebook['metadata']['data266_cpu_result_append']
        original = next(n for n in member_migration(ROOT)['notebooks'] if n['path'] == str(NOTEBOOK.relative_to(ROOT)))
        self.assertEqual(recorded_notebook_snapshot_digest(NOTEBOOK), original['new_notebook_sha256'])
        cell = notebook['cells'][proof['aggregation_cell_index']]
        self.assertEqual(hashlib.sha256(''.join(cell['source']).encode()).hexdigest(), proof['analysis_source_sha256'])
        self.assertIsNotNone(cell['execution_count'])
        self.assertFalse(any(o['output_type'] == 'error' for o in cell['outputs']))
        self.assertTrue(any('text/markdown' in o.get('data', {}) for o in cell['outputs']))
        text = ''.join(''.join(o.get('text', [])) for o in cell['outputs'])
        self.assertIn("human_review_complete: {'with': True, 'without': True, 'red_team': True}", text)
        self.assertIn('no model or GPU inference', text)

    def test_snapshot_check_rejects_modified_prior_gpu_output(self):
        nb = json.loads(NOTEBOOK.read_text())
        old_code = next(c for c in nb['cells'][:nb['metadata']['data266_cpu_result_append']['original_cell_count']] if c['cell_type'] == 'code')
        old_code['outputs'] = []
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'tampered.ipynb'
            path.write_text(json.dumps(nb))
            with self.assertRaisesRegex(ValueError, 'Original notebook'):
                recorded_notebook_snapshot_digest(path)
