"""Verify directory-only migration preserves frozen evidence and executed outputs."""
from pathlib import Path
import ast
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'eval/yuyao_ding'


class EvaluationRelocationTest(unittest.TestCase):
    def test_frozen_and_recorded_payload_bytes_are_unchanged(self):
        manifest = json.loads((EVAL / 'relocation_manifest.json').read_text())
        self.assertFalse(manifest['new_gpu_inference'])
        for row in manifest['files']:
            path = ROOT / row['new_path']
            if path.suffix in ('.json', '.csv'):
                self.assertTrue(row['byte_identical'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['original_sha256'])
            self.assertTrue(path.exists())
            self.assertFalse((ROOT / row['old_path']).exists())

    def test_only_eval_prefix_changed_in_recorded_notebook_code(self):
        manifest = json.loads((EVAL / 'relocation_manifest.json').read_text())
        for notebook in manifest['notebooks']:
            nb = json.loads((ROOT / notebook['path']).read_text())
            self.assertEqual(hashlib.sha256((ROOT / notebook['path']).read_bytes()).hexdigest(), notebook['new_notebook_sha256'])
            for row in notebook['changed_code_cells']:
                cell = nb['cells'][row['cell_index']]
                source = ''.join(cell['source'])
                # No model prompt, numerical rule or computation is changed.
                restored = source.replace('eval/yuyao_ding/', 'eval/')
                self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(), row['old_source_sha256'])
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), row['new_source_sha256'])
                self.assertEqual(hashlib.sha256(json.dumps(cell['outputs'], sort_keys=True).encode()).hexdigest(), row['outputs_sha256'])
                self.assertIsNotNone(cell['execution_count'])
                ast.parse(source)

    def test_moved_script_root_resolves_to_repository(self):
        for name in ('verify_p3_fixture_manifest.py','validate_p3_revision.py','build_p3_red_team_inputs.py'):
            path = EVAL / name
            tree = ast.parse(path.read_text())
            value = next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets))
            resolved = eval(compile(ast.Expression(value),str(path),'eval'),{'Path':Path,'__file__':str(path)})
            self.assertEqual(resolved,ROOT)
