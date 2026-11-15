"""CPU contracts only; these tests do not establish GPU/training performance."""
from copy import deepcopy
from dataclasses import asdict
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.p2_train_worker import invoke_trainer, adopt_logging_fix, PRE_LOGGING_FIX_SHA256

from src.p2_partner_baseline import (BaselineConfig, fetch_trainer, gpu_check,
    read, training_args, validate_generations, validate_prompts, validate_split, write)

ROOT=Path(__file__).resolve().parents[1]


class PartnerBaselineTest(unittest.TestCase):
    def prompts(self):return read(ROOT/'eval/p2_yuyao_baseline_prompts.json')['prompts']

    def test_fixed_config_and_prompt_coverage(self):
        BaselineConfig().validate()
        for cfg in (BaselineConfig(seed=1),BaselineConfig(train_steps=50),BaselineConfig(train_steps=601),BaselineConfig(rank=32)):
            with self.assertRaises(ValueError):cfg.validate()
        prompts=self.prompts();validate_prompts(prompts)
        for changed in (prompts[:-1],[*prompts[:-1],prompts[0]]):
            with self.assertRaises(ValueError):validate_prompts(changed)

    def test_no_test_leakage_or_silent_prompt_substitution(self):
        prompts=self.prompts()
        test=[dict(source_id=i,text=p['prompt']) for i,p in enumerate(prompts[:10])]
        train=[dict(source_id=100,text='training caption')]
        validate_split(train,test,prompts)
        for a,b in ((train,[*test,test[0]]),([dict(source_id=0,text='leaked')],test),(train,test[:-1])):
            with self.assertRaises(ValueError):validate_split(a,b,prompts)

    def test_image_identity_prompt_and_safety_grid(self):
        prompts=self.prompts()
        records=[dict(model=m,prompt_id=p['id'],seed=s,prompt=p['prompt'],category=p['category'],nsfw=False)
                 for m in ('base','yuyao_lora_r4') for p in prompts for s in (0,1)]
        validate_generations(records,prompts,complete=True)
        for altered in (records[:-1],[*records[:-1],records[0]]):
            with self.assertRaises(ValueError):validate_generations(altered,prompts,complete=True)
        for field,value in [('prompt','changed'),('nsfw',0),('model','teammate_final')]:
            changed=deepcopy(records);changed[0][field]=value
            with self.assertRaises(ValueError):validate_generations(changed,prompts,complete=True)

    def test_training_targets_only_accepted_train_directory_and_pinned_base(self):
        with TemporaryDirectory() as temp:
            run=Path(temp)
            write(run/'manifest.json',dict(config=asdict(BaselineConfig()),model_revisions=dict(base='a'*40)))
            args=training_args(run,base_path='/model/snapshot')
            self.assertEqual(args[args.index('--pretrained_model_name_or_path')+1],'/model/snapshot')
            self.assertEqual(Path(args[args.index('--train_data_dir')+1]),run/'data/train')
            self.assertNotIn('--dataset_name',args)
            self.assertNotIn('--push_to_hub',args)
            self.assertNotIn('--resume_from_checkpoint',args)
            (run/'adapter/checkpoint-100').mkdir(parents=True)
            args=training_args(run)
            self.assertEqual(args[args.index('--resume_from_checkpoint')+1],'latest')

    def test_downloaded_trainer_is_hash_checked_before_it_can_be_executed(self):
        with TemporaryDirectory() as temp:
            response=SimpleNamespace(read=lambda:b'changed external source')
            with patch('src.p2_partner_baseline.urllib.request.urlopen',return_value=response):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):fetch_trainer(temp)
            self.assertFalse((Path(temp)/'official/train_text_to_image_lora.py').exists())

    def test_source_normalization_is_limited_to_final_newlines(self):
        blob=b'print("synthetic source")\n'
        digest=hashlib.sha256(blob).hexdigest()
        with TemporaryDirectory() as temp:
            response=SimpleNamespace(read=lambda:blob+b'\n')
            with patch('src.p2_partner_baseline.urllib.request.urlopen',return_value=response), patch('src.p2_partner_baseline.TRAINER_SHA256',digest):
                path=fetch_trainer(temp)
            self.assertEqual(path.read_bytes(),blob)

    def test_hardware_contract_does_not_accept_cpu_4090_or_two_gpus(self):
        for available,count,name in [(False,0,'cpu'),(True,2,'Tesla T4'),(True,1,'RTX 4090')]:
            cuda=SimpleNamespace(is_available=lambda:available,device_count=lambda:count,
                get_device_properties=lambda _:SimpleNamespace(name=name,total_memory=16*1024**3))
            with patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda)}):
                with self.assertRaises(RuntimeError):gpu_check()
        cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,
            get_device_properties=lambda _:SimpleNamespace(name='Tesla T4',total_memory=15*1024**3))
        with patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda)}):
            self.assertEqual(gpu_check()['name'],'Tesla T4')

    def test_notebook_is_unexecuted_and_all_cells_parse(self):
        nb=read(ROOT/'notebooks/Yuyao_Ding/p2_baseline.ipynb')
        self.assertEqual(nb['cells'][0]['cell_type'],'code')
        first=''.join(nb['cells'][0]['source'])
        for text in ('Git commit:','Python:','PyTorch:','GPU:','SEED:'):
            self.assertIn(text,first)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])

    def test_actual_cli_parse_is_normalized_before_accelerator_construction(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic_trainer.py'
            path.write_text('''import argparse
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--report_to')
    p.add_argument('--max_train_steps',type=int)
    return p.parse_args()
def main():
    args=parse_args()
    # Model the API contract that rejected the user's CLI string.
    assert args.report_to is None, 'Unsupported logging capability'
    return args.max_train_steps
if __name__ == '__main__':
    raise AssertionError('Do not run the entry point before normalizing arguments.')
''')
            before=path.read_bytes()
            with patch('sys.argv',[str(path),'--report_to','none','--max_train_steps','600']):
                self.assertEqual(invoke_trainer(path),600)
            self.assertEqual(path.read_bytes(),before)

    def logging_failure_fixture(self, root):
        run=root/'run';(root/'scripts').mkdir();(root/'src').mkdir()
        (root/'scripts/p2_train_worker.py').write_text('new worker source')
        (root/'src/p2_partner_baseline.py').write_text('unchanged pipeline')
        parent_hash=hashlib.sha256((root/'src/p2_partner_baseline.py').read_bytes()).hexdigest()
        write(run/'manifest.json',dict(source_commit='original-source',source_hashes={
            'scripts/p2_train_worker.py':PRE_LOGGING_FIX_SHA256,'src/p2_partner_baseline.py':parent_hash}))
        from src.p2_partner_baseline import TRAINER_SHA256
        write(run/'training_stats.json',dict(status='failed',error_type='ValueError',peak_allocated_gib=0,
            peak_reserved_gib=0,trainer_sha256=TRAINER_SHA256,cumulative_wall_seconds=12))
        (run/'training.log').write_text('ValueError: Unsupported logging capability: none')
        (run/'prepared-data-sentinel').write_text('preserve this')
        return run

    def test_explicit_logging_recovery_keeps_failure_and_original_provenance(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);run=self.logging_failure_fixture(root)
            before_manifest=(run/'manifest.json').read_bytes()
            before_stats=(run/'training_stats.json').read_bytes()
            with patch('scripts.p2_train_worker.subprocess.check_output',return_value='new-source\n'):
                adopt_logging_fix(root,run)
                adopt_logging_fix(root,run)
            manifest=read(run/'manifest.json')
            self.assertEqual(manifest['source_commit'],'original-source')
            self.assertEqual(manifest['training_source_commit'],'new-source')
            self.assertEqual(len(manifest['training_compatibility_fixes']),1)
            history=manifest['training_compatibility_fixes'][0]
            self.assertEqual(history['previous_source_hashes']['scripts/p2_train_worker.py'],PRE_LOGGING_FIX_SHA256)
            self.assertEqual((run/'training_stats.json').read_bytes(),before_stats)
            self.assertEqual((run/'prepared-data-sentinel').read_text(),'preserve this')
            self.assertEqual(len(list((run/'training_attempts').glob('pre_logging_fix_*.json'))),2)
            self.assertTrue(any(read(p)['source_commit']=='original-source' for p in (run/'training_attempts').glob('*manifest*.json')))

    def test_logging_recovery_rejects_any_run_with_existing_training_checkpoints(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);run=self.logging_failure_fixture(root)
            (run/'adapter/checkpoint-100').mkdir(parents=True)
            original=(run/'manifest.json').read_bytes()
            with self.assertRaises(ValueError):adopt_logging_fix(root,run)
            self.assertEqual((run/'manifest.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
