"""Measure one-GPU training around the hash-pinned official Diffusers example."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.yuyao_ding.p2_partner_baseline import gpu_check,read,write,sha,TRAINER_SHA256

PRE_LOGGING_FIX_SHA256='29bed9c4f61c14cc4735ba55f0eb213caed1caadc346eb3ff06165f83e64722f'


def invoke_trainer(path):
    # Keep the upstream file byte-identical. Its CLI string 'none' must become
    # Python None before Accelerator(log_with=...) sees it.
    namespace=runpy.run_path(str(path),run_name='data266_official_lora')
    main_function=namespace['main']
    original_parse=main_function.__globals__['parse_args']
    def parse_without_external_tracking():
        args=original_parse()
        if isinstance(args.report_to,str) and args.report_to.lower()=='none':
            args.report_to=None
        return args
    main_function.__globals__['parse_args']=parse_without_external_tracking
    return main_function()


def adopt_logging_fix(root, run):
    """Explicit recovery for the known pre-training failure; preserve its audit."""
    root,run=Path(root),Path(run)
    path=run/'manifest.json';manifest=read(path)
    worker='scripts/yuyao_ding/p2_train_worker.py'
    current=sha(root/worker)
    if manifest['source_hashes'][worker]==current:
        print('Logging fix source already recorded; no source update needed.')
        return
    stats=read(run/'training_stats.json')
    if (manifest['source_hashes'][worker]!=PRE_LOGGING_FIX_SHA256
        or stats.get('status')!='failed' or stats.get('error_type')!='ValueError'
        or stats.get('peak_allocated_gib')!=0 or stats.get('peak_reserved_gib')!=0
        or 'Unsupported logging capability: none' not in (run/'training.log').read_text()
        or list((run/'adapter').glob('checkpoint-*'))
        or (run/'adapter/pytorch_lora_weights.safetensors').exists()
        or sha(root/'src/yuyao_ding/p2_partner_baseline.py')!=manifest['source_hashes']['src/yuyao_ding/p2_partner_baseline.py']):
        raise ValueError('This recovery is only for the recorded no-allocation logging failure. Do not rewrite other run provenance.')
    if stats.get('trainer_sha256')!=TRAINER_SHA256:
        raise ValueError('The upstream trainer differs from the reviewed source.')
    archive=run/'training_attempts';archive.mkdir(exist_ok=True)
    write(archive/f'pre_logging_fix_manifest_{sha(path)[:16]}.json',deepcopy(manifest))
    write(archive/f'pre_logging_fix_stats_{sha(run/"training_stats.json")[:16]}.json',stats)
    event=dict(reason='Convert report_to CLI string none into Python None before Accelerator initialization; no model/data/training-setting change.',previous_source_hashes=deepcopy(manifest['source_hashes']),training_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),worker_sha256=current,previous_training_started=False)
    manifest.setdefault('training_compatibility_fixes',[]).append(event)
    manifest['source_hashes'][worker]=current
    manifest['training_source_commit']=event['training_source_commit']
    write(path,manifest)
    print('Recorded logging-only source fix; original manifest and failed statistics preserved. Prepared data is unchanged.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--trainer',required=True,type=Path)
    p.add_argument('training_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if sha(args.trainer)!=TRAINER_SHA256:raise ValueError('Trainer hash mismatch.')
    if read(args.run/'manifest.json')['source_hashes']['scripts/yuyao_ding/p2_train_worker.py']!=sha(Path(__file__)):
        raise ValueError('Worker source changed. For the known zero-allocation logging failure, explicitly call adopt_logging_fix before retrying.')
    gpu=gpu_check()
    import torch
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    stats_path=args.run/'training_stats.json'
    previous=read(stats_path) if stats_path.exists() else {}
    if previous:
        write(args.run/'training_attempts'/f'{sha(stats_path)[:16]}.json',previous)
    started=time.perf_counter();stats=dict(status='failed',gpu=gpu['name'])
    sys.argv=[str(args.trainer),*(args.training_args[1:] if args.training_args[:1]==['--'] else args.training_args)]
    try:
        invoke_trainer(args.trainer)
        adapter=args.run/'adapter/pytorch_lora_weights.safetensors'
        from safetensors.torch import load_file
        weights=load_file(str(adapter),device='cpu')
        config=read(args.run/'manifest.json')['config']
        if not (args.run/f'adapter/checkpoint-{config["train_steps"]}').is_dir():
            raise RuntimeError('Final training-step checkpoint is missing.')
        stats.update(status='completed',adapter_sha256=sha(adapter),trainable_parameters=sum(v.numel() for v in weights.values()),rank=config['rank'],learning_rate=config['learning_rate'],requested_steps=config['train_steps'],micro_batch=config['micro_batch'],gradient_accumulation=config['gradient_accumulation'],effective_batch=config['micro_batch']*config['gradient_accumulation'])
    except BaseException as exc:
        stats['error_type']=type(exc).__name__
        raise
    finally:
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        stats.update(wall_seconds_this_invocation=seconds,cumulative_wall_seconds=previous.get('cumulative_wall_seconds',0)+seconds,peak_allocated_gib=max(previous.get('peak_allocated_gib',0),torch.cuda.max_memory_allocated()/1024**3),peak_reserved_gib=max(previous.get('peak_reserved_gib',0),torch.cuda.max_memory_reserved()/1024**3),trainer_sha256=sha(args.trainer),external_tracking='disabled: report_to converted to Python None')
        write(stats_path,stats)
        print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
