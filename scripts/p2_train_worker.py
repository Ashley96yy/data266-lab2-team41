"""Measure one-GPU training around the hash-pinned official Diffusers example."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.p2_partner_baseline import gpu_check,read,write,sha,TRAINER_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--trainer',required=True,type=Path)
    p.add_argument('training_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if sha(args.trainer)!=TRAINER_SHA256:raise ValueError('Trainer hash mismatch.')
    gpu=gpu_check()
    import torch
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    stats_path=args.run/'training_stats.json'
    previous=read(stats_path) if stats_path.exists() else {}
    started=time.perf_counter();stats=dict(status='failed',gpu=gpu['name'])
    sys.argv=[str(args.trainer),*(args.training_args[1:] if args.training_args[:1]==['--'] else args.training_args)]
    try:
        runpy.run_path(str(args.trainer),run_name='__main__')
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
        stats.update(wall_seconds_this_invocation=seconds,cumulative_wall_seconds=previous.get('cumulative_wall_seconds',0)+seconds,peak_allocated_gib=max(previous.get('peak_allocated_gib',0),torch.cuda.max_memory_allocated()/1024**3),peak_reserved_gib=max(previous.get('peak_reserved_gib',0),torch.cuda.max_memory_reserved()/1024**3),trainer_sha256=sha(args.trainer))
        write(stats_path,stats)
        print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
