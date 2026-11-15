# Part 2 partner working baseline: Yuyao Ding

Status: implementation prepared; no new GPU execution or baseline measurements
have been claimed. CPU checks are not T4/P100 validation.

Team Rules, page 1, require the partner who does not lead Parts 2/3 to implement
and run the complete pipeline at least as a working baseline. This notebook
provides the Part 2 execution path for Yuyao; completing it still requires Yuyao
to run it, inspect results, explain the code and record actual participation.
The assignment does not define a separate minimum step budget for this partner
baseline. The shorter budget below is a disclosed implementation choice, not a
claim that the course explicitly specifies 600 steps.

## Files and scope

- Notebook: [`notebooks/Yuyao_Ding/p2_baseline.ipynb`](../notebooks/Yuyao_Ding/p2_baseline.ipynb).
- Helpers: [`src/p2_partner_baseline.py`](../src/p2_partner_baseline.py).
- Measured GPU worker: [`scripts/p2_train_worker.py`](../scripts/p2_train_worker.py).
- Environment: [`config/requirements-part2-baseline.txt`](../config/requirements-part2-baseline.txt).
- Frozen prompts: [`p2_yuyao_baseline_prompts.json`](p2_yuyao_baseline_prompts.json).

The teammate's [executed full Part 2 notebook](../notebooks/Pratiksha_Kaushik/p2/p2_v4.ipynb)
and the rank-4/rank-32, 3,000-step models remain intact. This baseline reuses and
credits its dataset cleaning/split protocol, LoRA attention targets, fixed 20
prompts and sampler. It does not relabel the teammate's results as Yuyao's run.

## Protocol fixed before this training

| Item | Partner baseline |
| --- | --- |
| Data | Naruto BLIP Captions; byte/caption checks, MD5 and pHash deduplication |
| Split | 90/10, seed 266; hold out test images from training |
| Safety | SD checker on training data and every generated output |
| Resolution | 512×512, center crop |
| Model | SD 1.5, exact revision recorded at run start |
| Fine-tuning | One real rank-4 UNet LoRA; frozen text encoder and VAE |
| Budget | 600 optimizer steps; micro-batch 1, accumulation 4; effective batch 4 |
| Optimizer | AdamW, LR 1e-4; cosine schedule with 60 warmup steps |
| Memory | fp16 frozen weights, fp32 trainable adapter, gradient checkpointing |
| Generation | Base and this newly trained LoRA; 20 prompts × seeds 0/1 per model |
| Sampler | DPM-Solver++, 30 steps, CFG 7.5, 512×512 |
| Metrics | KID (100 subsets), CLIP ViT-B/16 score, AlexNet LPIPS, measured cost |
| Memorization diagnostic | Closest training image for 10 in-domain adapted outputs; inspect 3 closest pairs |
| Human ratings | Not synthesized; the team's 25-pair, 3-rater evaluation is separate |

All output metrics are computed in notebook cells. KID is noisy with only 20
generated in-domain samples. Returned safety-filtered images, including flagged
black placeholders, remain in the metric population; do not silently score a
different unfiltered image set. Nearest-neighbor similarity alone cannot establish
memorization or its absence. Write two failure examples after seeing actual images.

## Training implementation and attribution

Training calls the official [Diffusers LoRA example](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/examples/text_to_image/train_text_to_image_lora.py)
from the v0.42.1 release, commit `2a1433913c7044f34fa4767dd84b28bba0bb0ecf`.
Its Apache-2.0 notice is retained in the downloaded private copy. The inspected
source, normalized to one trailing LF, has SHA-256
`7d67f3ed756a3cb79183153349ebb2d141c999f898da964edb2ea4af5d72d6ab`.
Source mismatches fail before execution. Training receives a revision-pinned local
model snapshot so the example's scheduler also cannot resolve a different `main`.
See [the official LoRA guide](https://huggingface.co/docs/diffusers/training/lora)
and [CLIP API](https://huggingface.co/docs/transformers/model_doc/clip).

The worker maps the example's `--report_to none` CLI value to Python `None`
before constructing Accelerator, whose no-tracker default is `log_with=None`.
The pinned upstream file is not modified. A first user Colab attempt failed on
this string/API mismatch before any CUDA allocation. Its failed statistics and
log must remain recorded. The explicit `adopt_logging_fix(ROOT, RUN)` helper
allows only that known zero-allocation failure to adopt the new worker source,
archives its original manifest/statistics, and records the new training commit.
It does not change training data, prompts, model revisions, settings or the
original data-preparation source commit. Existing checkpoints or unrelated source
changes are rejected. This fix has CPU regression coverage; a successful GPU
training run is still pending.

The new implementation adds isolated outputs, revision/source/data hashes,
single-GPU checks, measured worker statistics, checkpoint restart, per-image
checkpointing, paired metrics and private evidence packaging. This is an
AI-assisted adaptation, not a claim to have authored the upstream trainer.

## Run on one T4/P100

1. Start a fresh Colab T4 session (or Kaggle with only CUDA device 0 exposed).
2. Clone the team repository in that runtime. If already cloned, update it without
   discarding local changes. For the current public repository a setup cell is:

   ```python
   from pathlib import Path
   import os, subprocess
   base = Path('/content') if Path('/content').is_dir() else Path('/kaggle/working')
   repo = base / 'data266-lab2-team41'
   env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
   command = ['git', 'pull', '--ff-only', 'origin', 'main'] if (repo/'.git').is_dir() else ['git', 'clone', 'https://github.com/Ashley96yy/data266-lab2-team41.git', str(repo)]
   result = subprocess.run(command, cwd=repo if (repo/'.git').is_dir() else base, env=env)
   if result.returncode:
       raise RuntimeError('Clone/update failed. If the repository is private, use the existing token/askpass clone cell; never paste a token into a URL.')
   os.chdir(repo)
   print('Repository:', repo)
   ```

3. Open `notebooks/Yuyao_Ding/p2_baseline.ipynb` in that same GPU runtime and run
   its cells in order. Opening the notebook alone does not create a Git checkout.
   The first cell prints the real run information. Setup preserves the working
   CUDA/PyTorch build. Never run this setup in the working Windows environment.
4. Keep the default 600-step protocol for this baseline. Use
   `DATA266_P2_OUTPUT_DIR` for your own new run on mounted persistent storage if
   session expiry is a concern; the default `.runtime/` is temporary on Colab.
5. Preserve the printed RUN path. If interrupted, use the same directory, source,
   packages and config. Training resumes the latest 100-step checkpoint; completed
   training is reused only after adapter-hash verification. Generated-image hashes
   are checked, and only missing image identities are generated.
6. Inspect the grids and complete your own observations and actual meeting record.
   Download the final private bundle and the executed notebook. A notebook opened
   in Colab does not automatically save its outputs back to the repository file.

## What to keep and submit

Save the executed notebook, actual config/version/commit information, cleaning
report, frozen prompts, training statistics, results table, generation metadata
and evidence bundle. The GPU worker measures wall time and peak VRAM in its own
process; the adapter's trainable parameter count is computed from saved tensors.
The final checkpoint must confirm that the requested step budget was reached.

Artifacts stay under `.runtime/` or the explicitly selected output path. The
download bundle excludes original training/test images and optimizer checkpoints.
It is not sufficient to resume an interrupted training session by itself: retain
the complete run directory for that. Weights and generated image files belong
on Hugging Face, with upload only when authorized. Do not commit datasets, weights,
credentials or raw image folders to GitHub.

Only after real execution should CONTRIBUTIONS.md state that this baseline ran.
Record the actual knowledge-transfer/design discussion (date, topic, recording
link). Both members must be able to explain the pipeline in the demo. Preparing
this notebook or using an existing model for one image is not execution evidence.
