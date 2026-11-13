"""Run Part 1 in a fresh kernel using THIS Python environment.

No installation or model/environment replacement is performed. Outputs are
isolated under .runtime/part1/<UTC timestamp>. Stop other GPU demos first.
Set QUESTIONS_PATH to a different course-format CSV for hidden-set rehearsal.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.runtime'/'notebook-runner'
WORK.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN=ROOT/'.runtime'/'part1'/stamp
RUN.mkdir(parents=True,exist_ok=False)
kernel_root=WORK/'jupyter'/'kernels'/'data266-part1'
kernel_root.mkdir(parents=True,exist_ok=True)
(kernel_root/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='DATA266 existing Windows environment',language='python')),encoding='utf-8')
os.environ.update(PYTHONUTF8='1',JUPYTER_PATH=str(WORK/'jupyter'),JUPYTER_RUNTIME_DIR=str(RUN/'jupyter-runtime'),IPYTHONDIR=str(RUN/'ipython'),
    DATA266_PART1_DIR=os.environ.get('DATA266_PART1_DIR',str(ROOT/'data/part1')),DATA266_PART1_INPUT_DIR=os.environ.get('DATA266_PART1_DIR',str(ROOT/'data/part1')),
    DATA266_OUTPUT_DIR=str(RUN),DATA266_RUN_FULL_PART1=os.environ.get('DATA266_RUN_FULL_PART1','1'),DATA266_RUN_P1_EVAL=os.environ.get('DATA266_RUN_P1_EVAL','1'),
    DATA266_RUN_VLM_SMOKE='0',DATA266_RETRIEVAL_DEVICE='cpu',HF_XET_CACHE=str(ROOT/'.runtime/xet-cache'),HF_HUB_CACHE=str(ROOT/'.runtime/part1-model-cache'))
os.chdir(ROOT)
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nbformat
from nbclient import NotebookClient
nb=nbformat.read(ROOT/'notebooks/Yuyao_Ding/p1_rag.ipynb',as_version=4)
out=RUN/'p1_rag.executed.ipynb'
manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),status='running',run_dir=str(RUN),
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    source_notebook_sha256=hashlib.sha256((ROOT/'notebooks/Yuyao_Ding/p1_rag.ipynb').read_bytes()).hexdigest(),
    source_evaluation_sha256=hashlib.sha256((ROOT/'src/p1_evaluation.py').read_bytes()).hexdigest(),
    source_evidence_sha256=hashlib.sha256((ROOT/'src/p1_evidence.py').read_bytes()).hexdigest(),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
    interpretation='Observed Windows RTX4090 execution; not a T4/P100 run or Kaggle submission.')
(WORK/'part1-active-run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
def save():
    nbformat.write(nb,out)
    (RUN/'execution_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
def start(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'[{datetime.now().isoformat(timespec="seconds")}] START cell {cell_index}: {cell.source.splitlines()[0][:100]}',flush=True)
def done(cell,cell_index,**kwargs):
    manifest['last_completed_cell']=cell_index
    save()
    if cell.cell_type=='code':
        tail=''.join(o.get('text','') for o in cell.get('outputs',[]) if o.get('output_type')=='stream')[-1200:]
        print(f'DONE cell {cell_index}\n{tail}',flush=True)
client=NotebookClient(nb,kernel_name='data266-part1',timeout=3600,
    resources={'metadata':{'path':str(ROOT)}},on_cell_start=start,on_cell_executed=done)
started=time.monotonic()
try:
    client.execute()
    manifest['status']='completed'
except BaseException as exc:
    manifest.update(status='failed',error=repr(exc))
    raise
finally:
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    save()
    print(json.dumps(manifest,indent=2),flush=True)
