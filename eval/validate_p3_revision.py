"""Fresh real-GPU diagnostics followed by the unchanged, unscored 15x2+10 cases."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
REVISIONS=dict(vision='66285546d2b821cf421d4f5eb2576359d3770cd3',
    base='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',lora='530fa3a05dbbf7bfb3d0c70d79318c38f1270308')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    parser.add_argument('--skip-evaluation',action='store_true')
    args=parser.parse_args()
    target=Path(args.output_root) if args.output_root else ROOT/'.runtime/part3-revisions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=target.resolve();target.mkdir(parents=True,exist_ok=False)
    manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),status='running',
        model_revisions=REVISIONS,seed=266,scored=False,interpretation='Development rerun after recorded implementation fixes; two independent human reviews pending.',stages={})
    try:manifest['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:manifest['source_commit']='unavailable'
    manifest['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('src/p3_*.py')}
    manifest['frozen_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('eval/p3*.json')}
    write(target/'manifest.json',manifest)
    start=time.perf_counter()
    try:
        import torch
        from PIL import Image,ImageDraw
        from src.p3_vision import load_qwen_vision,run_qwen_vision
        from src.p3_generation import DiffusionGenerator,GenerationSettings
        from src.p3_prompt import build_generation_prompt
        from src.p3_critique import VisionCritic
        from src.p3_orchestration import AgentOrchestrator
        from src.p3_evaluation import run_cases,load_cases
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this validation.')
        torch.manual_seed(266);torch.cuda.manual_seed_all(266);torch.cuda.reset_peak_memory_stats()
        manifest.update(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        source=target/'new_diagnostic.png'
        image=Image.new('RGB',(512,384),'#d6e0e8');draw=ImageDraw.Draw(image)
        draw.rectangle((0,295,512,384),fill='#8c7969')
        draw.rectangle((80,210,210,240),fill='#d88125');draw.rectangle((80,245,220,277),fill='#c26a1c')
        draw.ellipse((320,190,395,275),fill='#577d48');draw.rectangle((260,300,360,335),fill='white');draw.text((278,307),'NOVA',fill='black')
        image.save(source)
        cached=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots'/REVISIONS['vision']
        processor,model=load_qwen_vision(str(cached) if cached.is_dir() else 'Qwen/Qwen2.5-VL-3B-Instruct',revision=REVISIONS['vision'])
        vision=partial(run_qwen_vision,processor=processor,model=model)
        generator=DiffusionGenerator(base_revision=REVISIONS['base'],lora_revision=REVISIONS['lora'])
        critic=VisionCritic(processor,model)
        from src.p3_prompt_rewrite import QwenPromptEngineer
        engineer=QwenPromptEngineer(processor,model,critic.similarity.get_tokenizer)
        pipeline=AgentOrchestrator(vision,generator,critic,output_dir=target/'requests',prompt_engineer=engineer)
        observation=vision(source,['What main objects and colors are visible?','What readable text is visible?'])
        write(target/'diagnostic_vision.json',observation)
        settings=GenerationSettings()
        checks=[]
        for label,request,route in [
            ('base','Make this drawing a watercolor illustration.','style_change_img2img'),
            ('ink','Render this drawing as a monochrome ink illustration.','style_change_img2img'),
            ('domain','Make this drawing a Naruto-inspired anime illustration.','domain_style_generation'),
            ('enhance','Brighten this drawing while preserving the objects and text.','enhancement')]:
            draft=engineer(request,route,observation)
            generated=generator.generate(source,draft,target/f'diagnostic_{label}.png',settings)
            if not generated['safe']:raise RuntimeError(f'Diagnostic {label} was safety-blocked; preserve this attempt and inspect before evaluation.')
            if label in ('domain','enhance'):
                from peft.tuners.tuners_utils import BaseTunerLayer
                flags=[m.disable_adapters for m in generator.pipe.unet.modules() if isinstance(m,BaseTunerLayer)]
                assert flags and all(flag==(label=='enhance') for flag in flags)
                generated['adapter_layers_checked']=len(flags);generated['all_adapters_disabled']=all(flags)
                if label=='domain':
                    layers=[m for m in generator.pipe.unet.modules() if isinstance(m,BaseTunerLayer)]
                    assert all(math.isclose(m.scaling['team_domain'],m.lora_alpha['team_domain']/m.r['team_domain']*settings.lora_scale,rel_tol=1e-6) for m in layers)
                    generated['adapter_weight_verified']=True
            reviewed=critic.critique(generated['image_path'],request,{**observation,'image_path':str(source),'generation':generated})
            assert math.isfinite(reviewed['clip_cosine_similarity']) and -1<=reviewed['clip_cosine_similarity']<=1
            assert 'quality_review' in reviewed and 'safety_review' in reviewed
            if label=='enhance':
                assert generated['applied_diffusion_steps']==0 and Image.open(generated['image_path']).size==image.size
            checks.append(dict(label=label,generation=generated,critique=reviewed))
            write(target/'diagnostic_generation.json',checks)
            print('DIAGNOSTIC',label,reviewed['decision'],reviewed['clip_cosine_similarity'],flush=True)
        trace=pipeline.run('What color is the round object?',source)
        assert trace['status']=='answered' and len(trace['vision']['answers'])>=2 and not trace['attempts']
        manifest['stages']['independent_diagnostics']='completed'
        write(target/'manifest.json',manifest)
        if not args.skip_evaluation:
            for name,spec,enabled in [('with','eval/p3_scenarios.json',True),('without','eval/p3_scenarios.json',False),('red_team','eval/p3_red_team.json',True)]:
                print('FROZEN EVALUATION',name,'UNSCORED',flush=True)
                result=run_cases(pipeline,load_cases(spec),root=ROOT,critique_enabled=enabled,checkpoint_path=target/f'{name}.json')
                result['model_revisions']=REVISIONS;result['source_hashes']=manifest['source_hashes']
                result['reviewer_a_name']=result['reviewer_b_name']=None
                for row in result['records']:
                    row.update(reviewer_a_pass=None,reviewer_b_pass=None,adjudicated_pass=None,resolution_note='')
                    for attempt in row['trace']['attempts']:attempt['human_accept']=None
                write(target/f'{name}.json',result)
                manifest['stages'][name]=dict(records=len(result['records']),statuses=dict(Counter(r['trace']['status'] for r in result['records'])),execution_errors=sum(r['trace']['status']=='error' for r in result['records']),scored=False)
                write(target/'manifest.json',manifest)
            # Fair first-attempt comparison: critic feedback may change later attempts.
            paired=[json.loads((target/f'{mode}.json').read_text(encoding='utf-8')) for mode in ['with','without']]
            for a,b in zip(paired[0]['records'],paired[1]['records']):
                assert a['case_id']==b['case_id'] and a['trace']['request']==b['trace']['request']
                if a['trace']['attempts'] and b['trace']['attempts']:
                    x,y=a['trace']['attempts'][0]['generation'],b['trace']['attempts'][0]['generation']
                    for field in ['effective_prompt','negative_prompt','settings','backend','model_revision','lora_revision']:
                        assert x[field]==y[field],(a['case_id'],field)
            manifest['paired_first_generation_settings_equal']=True
            if any(stage.get('execution_errors',0) for stage in manifest['stages'].values() if isinstance(stage,dict)):
                raise RuntimeError('Evaluation contains execution errors; inspect traces before reporting completion.')
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if hasattr(exc,'model_attempts'):write(target/'format_failure.json',exc.model_attempts)
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        if 'torch' in locals() and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            completed = manifest['status'] == 'completed'
            manifest.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                allocator_budget_gib=14.0,
                actual_t4_verified=completed and 'T4' in device_name,
                actual_p100_verified=completed and 'P100' in device_name,
            )
        write(target/'manifest.json',manifest)
        print('VALIDATION RECORD:',target,manifest['status'],flush=True)


if __name__=='__main__':main()
