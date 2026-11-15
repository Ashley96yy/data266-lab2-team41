# Yuyao Ding: completed Part 2 partner baseline

This is a real single Colab Tesla T4 execution of the separate working baseline,
run ID `20261115T034917Z` (UTC). It is not the team's two-run final comparison.
No inference, generation, training, or metric-model evaluation was rerun during
this import. Values below come from the supplied executed notebook and saved
artifacts. Data preparation used commit `ef31e7b`; training used the recorded
logging compatibility fix at `9e60390`. Do not retroactively assign the later
publication commit as the original inference commit.

## Recorded training

- LoRA rank: 4; saved trainable parameters: 797,184.
- Optimizer steps: 600; micro-batch 1, accumulation 4, effective batch 4; LR 0.0001.
- Successful worker invocation: 1186.69 seconds. Cumulative worker wall time: 1198.74 seconds, including the failed pre-training initialization.
- Peak PyTorch worker allocation: 3.6810 GiB; peak reserved: 4.1133 GiB. These are process allocator measurements, not whole-device NVML utilization.
- Cleaned dataset: 1221 original images; no corrupt/exact/perceptual duplicates removed; seed-266 90/10 split. Training safety filter flagged 240/1098, leaving **858 train / 123 test**. Train/test source IDs are disjoint.

## Saved base/adapter metrics

| Measurement | Base SD 1.5 | Yuyao rank-4, 600-step LoRA |
| --- | ---: | ---: |
| Generated images | 40 | 40 |
| KID, in-domain | 0.1337 | 0.0910 |
| KID subset standard deviation | 0.0122 | 0.0103 |
| Generated images used for KID | 20 | 20 |
| Held-out images used for KID | 123 | 123 |
| CLIP score, all prompts | 31.1084 | 30.7026 |
| LPIPS diversity | 0.8116 | 0.7928 |
| NSFW-flagged returned outputs | 1 | 0 |
| Median generation seconds/image | 4.8567 | 5.1072 |
| Cumulative training-worker wall-hours | 0 | 0.3330 |
| Peak train allocation (GiB) | N/A | 3.6810 |
| Peak generation allocation (GiB) | 3.1806 | 3.1821 |

Metrics describe returned safety-filtered images, including any flagged black
placeholder, not separately saved unfiltered generations. There are 20 in-domain
images/model for KID and 100 subsets; its uncertainty is substantial at this size.
CLIP uses ViT-B/16, LPIPS uses AlexNet. These automated values are not human style,
correctness or safety judgments. No personal-baseline human-preference score has
been filled in. The team's separate 25-pair/three-rater evaluation is still separate.

Lower KID here accompanies lower aggregate CLIP score and LPIPS diversity; it does
not establish universal improvement. The personal training budget, safety handling,
and resulting train size differ from the teammate's formal experiment. In particular,
this run has 858 accepted training images; the team report has 859. Preserve these
measured counts instead of rewriting them to agree. The cause of the one-image
filter difference has not been established. Do not pool this 600-step result into
an equal-budget comparison with the two 3,000-step team models.

## Failure and recovery remain recorded

The first invocation failed with `Unsupported logging capability: none` before
CUDA allocation. Its statistics are extracted without changing values from the
saved log into `initial_failed_attempt.json`. The worker then converted the CLI
string to Python `None`, preserving the pinned upstream trainer and prepared data.
`manifest.json` retains the initial source commit, old source hashes, fix event
and actual training source commit. The log records `checkpoint-600` and `600/600`;
`training_progress_excerpt.json` retains selected original lines and the full
log hash. The complete original log is in the private ZIP.

## Notebook and artifact provenance

The public [executed notebook](../../../../notebooks/Yuyao_Ding/p2_baseline.ipynb)
retains ten original code cells, execution counts and numerical outputs. The
uploaded original has twelve executed code cells with zero stored error outputs.
The clone bootstrap and one-off recovery cell are removed from the canonical
copy but retained in the private original. The ordinary training cell explicitly
re-reads the same completed record; the actual training ran in the original
recovery cell, and its log and adapter hashes were verified. This is not claimed
to be an uninterrupted Restart & Run All session.

Three embedded preview images, including nearest-training-image panels, are
omitted from the public notebook; the original figures remain in the private
executed notebook. Interactive progress/download widgets are removed. Generated
image files, adapter weights, raw train/test records, and original dataset images
are not uploaded here. The supplied archive excludes original train/test pixels
and optimizer checkpoints, so it is not a full interrupted-training resume bundle.

The adapter hash and tensor-header parameter count, all 80 PNG hashes/dimensions,
all paired generation identities, frozen prompt hash, source hashes, data split,
CLIP aggregates, LPIPS means, flags and timings were checked. KID was cross-checked
against saved output, not recomputed by fresh model inference. See
`integration_validation.json` and `provenance.json`. `generation_ledger.json`
contains image hashes and per-image numbers without image bytes or raw captions.
`memorization_scores.json` has ten nearest-training-image similarity values;
similarity alone is not proof of memorization or its absence.

## Still requires the team

Yuyao's own written visual interpretation and the real knowledge-transfer/design
meeting recording were not supplied. Do not invent either. Actual GPU execution
of this baseline is now recorded; these remaining interpretation/documentation
items do not require repeating training. Publish any supplemental weights/images
to an authorized Hugging Face location if they are needed for final sharing.
