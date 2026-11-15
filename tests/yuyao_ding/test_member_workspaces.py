"""Verify member namespace relocation preserves evidence, imports and history."""
from pathlib import Path
import hashlib
import importlib
import json
import unittest

from src.yuyao_ding.repository_paths import member_migration, recorded_file, restore_member_source
ROOT = Path(__file__).resolve().parents[2]


class MemberWorkspaceTest(unittest.TestCase):
    def test_original_payloads_and_notebook_outputs_are_preserved(self):
        manifest = member_migration(ROOT)
        self.assertFalse(manifest['new_gpu_inference'])
        for row in manifest['files']:
            p = ROOT / row['new_path']
            if p.suffix in ('.json','.csv','.txt','.bundle','.png'):
                self.assertTrue(row['byte_identical'])
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['original_sha256'])
        for row in manifest['notebooks']:
            p = ROOT / row['path']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['new_notebook_sha256'])
            nb = json.loads(p.read_text())
            for proof in row['changed_code_cells']:
                c = nb['cells'][proof['cell_index']]
                source = ''.join(c['source'])
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),proof['new_source_sha256'])
                self.assertEqual(hashlib.sha256(restore_member_source(source,ROOT).encode()).hexdigest(),proof['old_source_sha256'])
                self.assertEqual(hashlib.sha256(json.dumps(c['outputs'],sort_keys=True).encode()).hexdigest(),proof['outputs_sha256'])
                self.assertIsNotNone(c['execution_count'])

    def test_archived_specification_paths_resolve_without_changing_content(self):
        earlier = json.loads((ROOT/'eval/yuyao_ding/relocation_manifest.json').read_text())
        for row in earlier['files']:
            if row['old_path'].endswith('.json') and row['old_path'].startswith('eval/p3'):
                path = recorded_file(ROOT,row['old_path'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['original_sha256'])
        for bad in ('../private.json','/etc/passwd','C:/private/file.json','a\\b.json'):
            with self.subTest(path=bad),self.assertRaises(ValueError):recorded_file(ROOT,bad)
        with self.assertRaises(FileNotFoundError):recorded_file(ROOT,'unknown/missing.json')

    def test_member_packages_import_without_loading_models(self):
        for name in ('src.yuyao_ding.p1_evaluation','src.yuyao_ding.p2_partner_baseline',
                     'src.yuyao_ding.p3_orchestration','src.pratiksha_kaushik',
                     'scripts.yuyao_ding.finalize_p1_content_review','scripts.pratiksha_kaushik'):
            self.assertIsNotNone(importlib.import_module(name))
        for folder in ('src','scripts','tests','config','audit','traces','eval'):
            for member in ('yuyao_ding','pratiksha_kaushik'):
                self.assertTrue((ROOT/folder/member/'README.md').is_file())
