"""File-backed partner baseline: prepare, train one LoRA, generate and evaluate.

No inference runs at import time. Private artifacts stay under .runtime.
Training uses an inspected, commit/hash-pinned official Diffusers example.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
import sys
import urllib.request

BASE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
DATASET = 'lambdalabs/naruto-blip-captions'
CLIP = 'openai/clip-vit-base-patch16'
SAFETY = 'CompVis/stable-diffusion-safety-checker'
SAFETY_PROCESSOR = 'openai/clip-vit-base-patch32'
TRAINER_COMMIT = '2a1433913c7044f34fa4767dd84b28bba0bb0ecf'
TRAINER_URL = f'https://raw.githubusercontent.com/huggingface/diffusers/{TRAINER_COMMIT}/examples/text_to_image/train_text_to_image_lora.py'
# Hash of the inspected official source, with exactly one trailing LF.
TRAINER_SHA256 = '7d67f3ed756a3cb79183153349ebb2d141c999f898da964edb2ea4af5d72d6ab'


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 266
    resolution: int = 512
    rank: int = 4
    train_steps: int = 600
    micro_batch: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    inference_steps: int = 30
    guidance_scale: float = 7.5

    def validate(self):
        if self.seed != 266 or self.resolution != 512 or self.rank != 4:
            raise ValueError('This partner protocol fixes seed=266, resolution=512 and rank=4.')
        if self.train_steps < 100 or self.train_steps % 100:
            raise ValueError('Use at least 100 steps, in multiples of 100 for saved checkpoints.')
        if self.micro_batch != 1 or self.gradient_accumulation != 4:
            raise ValueError('This T4 baseline uses micro-batch 1 and accumulation 4.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prompts(prompts):
    if len(prompts) != 20 or [p['id'] for p in prompts] != list(range(20)):
        raise ValueError('Twenty unique ordered prompt IDs are required.')
    if Counter(p['category'] for p in prompts) != {'in_domain': 10, 'ood': 5, 'compositional': 5}:
        raise ValueError('Prompt coverage must be 10 in-domain, 5 OOD and 5 compositional.')
    if any(not isinstance(p['prompt'], str) or not p['prompt'].strip() for p in prompts):
        raise ValueError('Empty prompt.')


def validate_split(train, test, prompts):
    if len({r['source_id'] for r in train}) != len(train) or len({r['source_id'] for r in test}) != len(test):
        raise ValueError('Duplicate source IDs in split.')
    if {r['source_id'] for r in train} & {r['source_id'] for r in test}:
        raise ValueError('Training/test source overlap.')
    if not {p['prompt'] for p in prompts if p['category']=='in_domain'} <= {r['text'] for r in test}:
        raise ValueError('Frozen in-domain prompts are not test captions. Do not replace the prompts.')


def validate_generations(records, prompts, complete=False):
    expected={(model,p['id'],seed):p for model in ('base','yuyao_lora_r4') for p in prompts for seed in (0,1)}
    keys=[(r['model'],r['prompt_id'],r['seed']) for r in records]
    if len(keys)!=len(set(keys)) or any(key not in expected for key in keys):
        raise ValueError('Duplicate or unexpected generation identities.')
    if complete and set(keys)!=set(expected):
        raise ValueError('Require the complete paired 80-image grid.')
    for row,key in zip(records,keys):
        prompt=expected[key]
        if row['prompt']!=prompt['prompt'] or row['category']!=prompt['category'] or type(row['nsfw']) is not bool:
            raise ValueError('Generation prompt/category/safety mismatch.')


def gpu_check():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU before importing torch.')
    gpu = torch.cuda.get_device_properties(0)
    if not any(name in gpu.name.upper() for name in ('T4', 'P100')) or gpu.total_memory > 17 * 1024**3:
        raise RuntimeError('Use a course-approved single T4 or P100; this is not a CPU/Mac test.')
    return dict(name=gpu.name, total_gib=gpu.total_memory / 1024**3)


def start_run(root, output=None, config=None):
    import importlib.metadata as metadata
    from huggingface_hub import HfApi
    root = Path(root).resolve()
    cfg = config or BaselineConfig()
    cfg.validate()
    gpu = gpu_check()
    prompt_file = root / 'eval/yuyao_ding/p2_yuyao_baseline_prompts.json'
    validate_prompts(read(prompt_file)['prompts'])
    output = Path(output).resolve() if output else root / '.runtime/part2-yuyao' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # A restart may explicitly resume the same run, never silently load a teammate run.
    code_hashes = {p: sha(root / p) for p in ('src/yuyao_ding/p2_partner_baseline.py', 'scripts/yuyao_ding/p2_train_worker.py')}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = read(manifest_path)
        if old['config'] != asdict(cfg) or old['prompts_sha256'] != sha(prompt_file) or old['source_hashes'] != code_hashes:
            raise ValueError('Config, prompts or source changed; use a fresh run directory.')
        if any(metadata.version(p)!=version for p,version in old['packages'].items()):
            raise ValueError('Package versions differ from this run; restore its environment before resuming.')
        return output
    if output.exists() and any(output.iterdir()):
        raise ValueError('Existing nonempty output directory has no baseline manifest.')
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = dict(status='prepared', owner='Yuyao Ding', interpretation='Partner working baseline; not the team final comparison or an untouched holdout.', config=asdict(cfg), gpu=gpu, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(), source_hashes=code_hashes, prompts_sha256=sha(prompt_file), official_trainer_commit=TRAINER_COMMIT, official_trainer_sha256=TRAINER_SHA256, model_revisions={name: api.model_info(repo).sha for name, repo in [('base',BASE),('clip',CLIP),('safety',SAFETY),('safety_processor',SAFETY_PROCESSOR)]}, dataset_revision=api.dataset_info(DATASET).sha, packages={p: metadata.version(p) for p in ['torch','torchvision','diffusers','transformers','peft','accelerate','huggingface-hub','datasets','Pillow','torchmetrics','torch-fidelity','lpips','ImageHash']}, stages={})
    shutil.copy2(prompt_file, output / 'prompts.json')
    write(manifest_path, manifest)
    return output


def stage(run, name, value):
    manifest = read(Path(run) / 'manifest.json')
    manifest['stages'][name] = value
    write(Path(run) / 'manifest.json', manifest)


def release():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def prepare_data(run):
    import numpy as np
    import torch
    from datasets import load_dataset, Image as HFImage
    from huggingface_hub import HfApi
    import imagehash
    from PIL import Image, ImageOps
    from transformers import CLIPImageProcessor
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    run = Path(run)
    manifest = read(run / 'manifest.json')
    if (run / 'data_report.json').exists():
        rows = read(run / 'train.json') + read(run / 'test.json')
        if not all((run / r['file']).is_file() and sha(run / r['file']) == r['sha256'] for r in rows):
            raise ValueError('Cached training/test images are missing or changed.')
        if sha(run/'data/train/metadata.jsonl')!=read(run/'data_report.json')['train_metadata_sha256']:
            raise ValueError('Training caption metadata changed.')
        return read(run / 'data_report.json')
    data = load_dataset(DATASET, revision=manifest['dataset_revision'], split='train').cast_column('image', HFImage(decode=False))
    if len(data) < 500:
        raise ValueError('The course dataset must contain at least 500 images.')
    keep, removed = [], dict(corrupted=[], exact_duplicates=[], near_duplicates=[])
    md5s, phashes = set(), []
    for i, row in enumerate(data):
        try:
            source = row['image']
            blob = source.get('bytes') or Path(source['path']).read_bytes()
            Image.open(io.BytesIO(blob)).verify()
            image = Image.open(io.BytesIO(blob)).convert('RGB')
            if min(image.size) < 64 or not str(row['text']).strip():
                raise ValueError('Invalid image or caption.')
        except Exception:
            removed['corrupted'].append(i)
            continue
        digest = hashlib.md5(blob).hexdigest()
        phash = imagehash.phash(image)
        if digest in md5s:
            removed['exact_duplicates'].append(i)
        elif any(phash - old <= 2 for old in phashes):
            removed['near_duplicates'].append(i)
        else:
            md5s.add(digest); phashes.append(phash); keep.append(i)
    selected = data.add_column('source_id', list(range(len(data)))).select(keep)
    split = selected.train_test_split(test_size=0.1, seed=266)
    rows = {}
    for subset, entries in split.items():
        directory = run / 'data' / ('train_candidates' if subset == 'train' else 'test')
        directory.mkdir(parents=True, exist_ok=True)
        rows[subset] = []
        for row in entries:
            source = row['image']; blob = source.get('bytes') or Path(source['path']).read_bytes()
            image = Image.open(io.BytesIO(blob)).convert('RGB')
            image = ImageOps.fit(image, (512,512), method=Image.Resampling.BICUBIC)
            path = directory / f'{row["source_id"]:05d}.png'; image.save(path)
            rows[subset].append(dict(source_id=row['source_id'], file=str(path.relative_to(run)), text=row['text'], sha256=sha(path)))
    prompts = read(run / 'prompts.json')['prompts']
    validate_split(rows['train'],rows['test'],prompts)
    processor = CLIPImageProcessor.from_pretrained(SAFETY_PROCESSOR, revision=manifest['model_revisions']['safety_processor'])
    checker = StableDiffusionSafetyChecker.from_pretrained(SAFETY, revision=manifest['model_revisions']['safety']).to('cuda').eval()
    flags = []
    with torch.inference_mode():
        for offset in range(0, len(rows['train']), 8):
            images = [Image.open(run/r['file']).convert('RGB') for r in rows['train'][offset:offset+8]]
            pixel = processor(images=images, return_tensors='pt').pixel_values.to('cuda')
            _, batch = checker(images=np.stack([np.asarray(im,dtype=np.float32)/255 for im in images]), clip_input=pixel)
            flags.extend(bool(v) for v in batch)
            print(f'Training safety filter: {len(flags)}/{len(rows["train"])}', flush=True)
    del checker, processor; release()
    training = run / 'data/train'; training.mkdir(exist_ok=True)
    accepted = []
    for row, flagged in zip(rows['train'], flags):
        if not flagged:
            destination = training / Path(row['file']).name
            shutil.copy2(run / row['file'], destination)
            accepted.append({**row, 'file':str(destination.relative_to(run))})
    if len(accepted) < 2:
        raise ValueError('Too few accepted training images.')
    with (training/'metadata.jsonl').open('w',encoding='utf-8') as stream:
        for row in accepted:
            stream.write(json.dumps(dict(file_name=Path(row['file']).name, text=row['text']))+'\n')
    validate_split(accepted,rows['test'],prompts)
    write(run/'train.json', accepted); write(run/'test.json', rows['test'])
    info = HfApi().dataset_info(DATASET, revision=manifest['dataset_revision'])
    card = info.card_data.to_dict() if info.card_data else {}
    report = dict(dataset=DATASET, dataset_revision=manifest['dataset_revision'], original=len(data), kept_before_split=len(keep), removed_ids=removed, split_seed=266, train_before_safety=len(rows['train']), train_nsfw_flagged=sum(flags), train_after_safety=len(accepted), test=len(rows['test']), resolution=512, license=card.get('license') or 'Not stated in dataset card; copyrighted Naruto images, non-commercial research only.', train_metadata_sha256=sha(training/'metadata.jsonl'), train_ids_sha256=hashlib.sha256(json.dumps([r['source_id'] for r in accepted]).encode()).hexdigest(), test_ids_sha256=hashlib.sha256(json.dumps([r['source_id'] for r in rows['test']]).encode()).hexdigest())
    write(run/'data_report.json', report); stage(run,'data','completed')
    return report


def fetch_trainer(run):
    path = Path(run)/'official/train_text_to_image_lora.py'
    if path.exists():
        blob = path.read_bytes()
    else:
        blob = urllib.request.urlopen(TRAINER_URL, timeout=90).read()
    blob = blob.rstrip(b'\n') + b'\n'
    if hashlib.sha256(blob).hexdigest() != TRAINER_SHA256:
        raise ValueError('Official trainer source hash mismatch.')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(blob)
    return path


def training_args(run, base_path=None):
    run = Path(run); manifest=read(run/'manifest.json'); cfg=manifest['config']
    args = ['--pretrained_model_name_or_path', BASE, '--revision', manifest['model_revisions']['base'], '--train_data_dir', str(run/'data/train'), '--caption_column', 'text', '--resolution', '512', '--center_crop', '--rank', str(cfg['rank']), '--train_batch_size', str(cfg['micro_batch']), '--gradient_accumulation_steps', str(cfg['gradient_accumulation']), '--gradient_checkpointing', '--mixed_precision', 'fp16', '--max_train_steps', str(cfg['train_steps']), '--learning_rate', str(cfg['learning_rate']), '--lr_scheduler', 'cosine', '--lr_warmup_steps', str(min(60,cfg['train_steps']//10)), '--seed', '266', '--dataloader_num_workers', '0', '--checkpointing_steps', '100', '--checkpoints_total_limit', '2', '--report_to', 'none', '--output_dir', str(run/'adapter')]
    if base_path is not None:
        args[1]=str(base_path)
    if list((run/'adapter').glob('checkpoint-*')):
        args += ['--resume_from_checkpoint','latest']
    return args


def train_lora(root, run):
    import os
    root, run = Path(root), Path(run)
    if (run/'training_stats.json').exists() and read(run/'training_stats.json')['status']=='completed':
        stats=read(run/'training_stats.json')
        if sha(run/'adapter/pytorch_lora_weights.safetensors') != stats['adapter_sha256']:
            raise ValueError('Completed adapter changed.')
        print('Using this run\'s completed training record; no new training.')
        return stats
    if not (run/'data_report.json').exists():
        raise ValueError('Complete data preparation first.')
    if sha(run/'data/train/metadata.jsonl')!=read(run/'data_report.json')['train_metadata_sha256']:
        raise ValueError('Training caption metadata changed.')
    trainer=fetch_trainer(run)
    # The upstream noise-scheduler loader omits --revision, so pass a completely
    # pinned local snapshot rather than allowing that component to resolve main.
    from huggingface_hub import snapshot_download
    base_path=snapshot_download(BASE,revision=read(run/'manifest.json')['model_revisions']['base'],allow_patterns=['model_index.json','scheduler/*','tokenizer/*','text_encoder/config.json','text_encoder/model.safetensors','vae/config.json','vae/diffusion_pytorch_model.safetensors','unet/config.json','unet/diffusion_pytorch_model.safetensors'])
    command=[sys.executable,'-u',str(root/'scripts/yuyao_ding/p2_train_worker.py'),'--run',str(run),'--trainer',str(trainer),'--',*training_args(run,base_path)]
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUTF8':'1','TOKENIZERS_PARALLELISM':'false'}
    with (run/'training.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
    if code:
        stage(run,'training','failed; preserve training.log and checkpoints')
        raise RuntimeError('Training failed. Inspect training.log; rerun this same directory to resume saved checkpoints.')
    stats=read(run/'training_stats.json')
    if stats['status']!='completed':
        raise RuntimeError('Training did not complete.')
    stage(run,'training','completed')
    return stats


def generate(run):
    import time
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    run=Path(run);manifest=read(run/'manifest.json');cfg=manifest['config']
    stats=read(run/'training_stats.json')
    if stats['status']!='completed': raise ValueError('Train your own adapter first.')
    if sha(run/'adapter/pytorch_lora_weights.safetensors')!=stats['adapter_sha256']:
        raise ValueError('Trained adapter changed.')
    file=run/'generations.json';records=read(file) if file.exists() else []
    prompts=read(run/'prompts.json')['prompts']
    validate_generations(records,prompts)
    for row in records:
        if not (run/row['file']).is_file() or sha(run/row['file'])!=row['sha256']:
            raise ValueError('Saved generation changed.')
    if len(records)==80:
        return records
    pipe=StableDiffusionPipeline.from_pretrained(BASE,revision=manifest['model_revisions']['base'],torch_dtype=torch.float16).to('cuda')
    if pipe.safety_checker is None:
        raise ValueError('Output safety checking must stay enabled.')
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    for model in ['base','yuyao_lora_r4']:
        if model!='base':pipe.load_lora_weights(str(run/'adapter'))
        for prompt in prompts:
            for seed in [0,1]:
                if any((r['model'],r['prompt_id'],r['seed'])==(model,prompt['id'],seed) for r in records):continue
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
                result=pipe(prompt['prompt'],generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=cfg['inference_steps'],guidance_scale=cfg['guidance_scale'],height=512,width=512)
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                if result.nsfw_content_detected is None:raise ValueError('Missing output-safety result.')
                path=run/f'images/{model}_p{prompt["id"]:02d}_s{seed}.png';path.parent.mkdir(exist_ok=True);result.images[0].save(path)
                records.append(dict(model=model,prompt_id=prompt['id'],category=prompt['category'],prompt=prompt['prompt'],seed=seed,file=str(path.relative_to(run)),sha256=sha(path),seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,nsfw=bool(result.nsfw_content_detected[0])))
                write(file,records);print(f'Generated {len(records)}/80: {path.name}',flush=True)
    validate_generations(records,prompts,complete=True)
    del pipe;release();stage(run,'generation','completed')
    return records


def evaluate(run):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from torchmetrics.image.kid import KernelInceptionDistance
    import lpips
    run=Path(run);manifest=read(run/'manifest.json');records=read(run/'generations.json')
    validate_generations(records,read(run/'prompts.json')['prompts'],complete=True)
    def uint8(paths,size=None):
        images=[Image.open(run/p).convert('RGB') for p in paths]
        if size:images=[im.resize((size,size),Image.Resampling.BICUBIC) for im in images]
        return torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2)
    processor=CLIPProcessor.from_pretrained(CLIP,revision=manifest['model_revisions']['clip'])
    clip=CLIPModel.from_pretrained(CLIP,revision=manifest['model_revisions']['clip']).to('cuda').eval()
    def image_features(rows):
        features=[]
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                images=[Image.open(run/r['file']).convert('RGB') for r in rows[start:start+8]]
                inputs=processor(images=images,text=['']*len(images),padding=True,return_tensors='pt').to('cuda')
                features.append(F.normalize(clip(**inputs).image_embeds.float(),dim=-1).cpu())
        return torch.cat(features)
    with torch.inference_mode():
        for row in records:
            inputs=processor(text=[row['prompt']],images=Image.open(run/row['file']).convert('RGB'),padding=True,truncation=True,return_tensors='pt').to('cuda')
            out=clip(**inputs)
            row['clip_score']=float((100*F.cosine_similarity(out.image_embeds.float(),out.text_embeds.float())).clamp(min=0).item())
    training=read(run/'train.json');train_features=image_features(training)
    queries=[r for r in records if r['model']=='yuyao_lora_r4' and r['category']=='in_domain' and r['seed']==0]
    similarities=image_features(queries)@train_features.T
    memory=[]
    for row,values in zip(queries,similarities):
        score,index=values.max(dim=0)
        memory.append(dict(generated_file=row['file'],training_file=training[int(index)]['file'],cosine=float(score)))
    write(run/'memorization_neighbors.json',memory)
    del clip,processor,train_features,similarities;release()
    write(run/'generations.json',records)
    lp=lpips.LPIPS(net='alex').to('cuda').eval()
    diversity={}
    for model in ['base','yuyao_lora_r4']:
        distances=[]
        for prompt_id in range(20):
            pair=sorted([r for r in records if r['model']==model and r['prompt_id']==prompt_id],key=lambda r:r['seed'])
            assert [r['seed'] for r in pair]==[0,1]
            a,b=[uint8([r['file']]).float().to('cuda')/127.5-1 for r in pair]
            with torch.inference_mode():distances.append(float(lp(a,b).item()))
        diversity[model]=distances
    del lp;release()
    test=read(run/'test.json');results=[]
    for model in ['base','yuyao_lora_r4']:
        generated=[r for r in records if r['model']==model]
        fake=[r for r in generated if r['category']=='in_domain']
        kid=KernelInceptionDistance(subset_size=min(len(fake),len(test)),subsets=100).to('cuda')
        torch.manual_seed(266)
        for real_flag,items in [(True,test),(False,fake)]:
            for start in range(0,len(items),8):kid.update(uint8([r['file'] for r in items[start:start+8]],299).to('cuda'),real=real_flag)
        mu,sd=kid.compute();stats=read(run/'training_stats.json')
        results.append(dict(model=model,images=len(generated),kid_in_domain=float(mu),kid_std=float(sd),kid_generated_images=len(fake),kid_test_images=len(test),clip_score=mean(r['clip_score'] for r in generated),clip_by_category={c:mean(r['clip_score'] for r in generated if r['category']==c) for c in ['in_domain','ood','compositional']},lpips=mean(diversity[model]),nsfw_flagged=sum(r['nsfw'] for r in generated),median_seconds_per_image=median(r['seconds'] for r in generated),generation_gpu_hours=sum(r['seconds'] for r in generated)/3600,training_gpu_hours=0 if model=='base' else stats['cumulative_wall_seconds']/3600,training_peak_allocated_gib=None if model=='base' else stats['peak_allocated_gib'],generation_peak_allocated_gib=max(r['peak_allocated_gib'] for r in generated),human_preference=None))
        del kid;release()
    write(run/'results.json',results);write(run/'lpips_pairs.json',diversity)
    stage(run,'evaluation','completed')
    manifest=read(run/'manifest.json');manifest['status']='completed';manifest['human_evaluation']='Not performed for this separate partner baseline.';write(run/'manifest.json',manifest)
    return results
