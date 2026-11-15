# Part 2 partner baseline: result interpretation

**AI-assisted draft for Yuyao Ding to verify and revise into their own words.**
This analysis uses the saved Colab Tesla T4 run `20261115T034917Z` and actual
private output images. The visual observations below were made by Codex; they
are not independent human ratings or evidence that either team member has
completed their own interpretation. No training, generation or metric-model
inference was repeated. Derived differences were computed in a separately
executed CPU-only notebook cell from the saved numeric records; see
[analysis_computed.json](analysis_computed.json) and the
[executed notebook](../../../../../notebooks/Yuyao_Ding/p2_baseline.ipynb).

## What changed in this run

| Measurement | Base SD 1.5 | Rank-4 LoRA, 600 steps | Interpretation |
| --- | ---: | ---: | --- |
| In-domain KID ↓ | 0.1337 | 0.0910 | Observed relative reduction 31.94% |
| Aggregate CLIP score ↑ | 31.1084 | 30.7026 | Decrease of 0.4058 points |
| LPIPS seed-pair diversity ↑ | 0.8116 | 0.7928 | Decrease of 0.0189 |
| Median generation seconds/image ↓ | 4.8567 | 5.1072 | Increase of 0.2505 seconds (5.16%) |
| Returned-output safety flags | 1/40 | 0/40 | Automatic flags, not a human safety verdict |

The lower KID is consistent with the adapted in-domain images being closer to
the held-out image distribution in the metric's feature space. It does not by
itself establish recognizable Naruto style, better anatomy, or better prompt
following. KID uses only 20 generated in-domain images per model and 123 test
images; subset standard deviations are 0.0122 and 0.0103. These are subset
variability estimates, not confidence intervals for the improvement, and no
significance test or independent repeated training run was performed.

Prompt alignment and diversity show a tradeoff. Overall CLIP fell by 0.4058 points
and LPIPS fell by 0.0189, so this result does not support a claim of improvement
on every dimension. The CLIP category changes are -0.8661 in-domain, -1.1779 OOD,
and +1.2868 compositional. The compositional average increased, yet the concrete
count failure below remains. Category means do not establish that every prompt
improved. Lower LPIPS here means the two seeds produced less perceptually distinct
outputs on average; it is not a direct image-quality score.

## Concrete output observations

These are AI-assisted visual inspections of the private PNGs, keyed to the public
[image-hash/score ledger](generation_ledger.json). No image bytes are published.

1. **Prompt 15, seed 0 — subject count and color binding.** The prompt asks for
   two characters, one in red and one in blue. The adapted output shows one
   cropped figure in red with blue trim/belt, rather than two separately colored
   characters. Its CLIP score is 26.9594. The paired base output also shows a
   single red-clad figure and scores 26.8314. Both show a dark vertical band at
   the left. This is an existing composition limitation; this pair does not show
   that LoRA caused the failure or fixed it.
2. **Prompt 7, seed 0 — hat subtype is not clearly preserved.** The prompt asks
   for a man with a baseball cap. The adapted output has a rounded red cap,
   without a clearly visible baseball-cap visor. A generic hat is present, but
   the requested subtype is not convincingly demonstrated. Its CLIP score is
   27.0114. This is a cautious visual diagnosis, not a formal human pass/fail label.
3. **Counterexample, prompt 5, seed 0.** The adapted image visibly retains glasses
   and a white shirt, as requested, despite a low CLIP score of 27.4483. Therefore,
   selecting the lowest CLIP examples is useful for inspection but cannot replace
   visual review or turn those scores into failure labels automatically.

The examples support checking counts, attribute binding and specific object
subtypes in addition to domain appearance. They do not establish a population
failure rate, because these are selected diagnostic examples.

## Cost, safety and memorization limits

The successful 600-step training invocation took **19.78 minutes** and trained
**797,184 parameters**. Peak allocated training memory was **3.6810 GiB** (reserved
4.1133 GiB). Successful worker wall time was 0.3296 hours; the recorded cumulative
0.3330 hours includes the earlier 12.06-second logging-initialization failure,
which occurred before GPU allocation. These are worker timings and PyTorch
allocator readings, not whole-device utilization or energy measurements.

Base and adapted generation medians were 4.8567 and 5.1072 seconds/image under
this recorded protocol. The 5.16% observed increase is not an isolated estimate
of adapter overhead: order, warmup and runtime variation were not separated by
repeated randomized timing trials.

The enabled safety checker flagged one returned base image and no adapted images.
Metrics include returned safety-filtered outputs, including any black placeholder.
Zero flags among 40 images does not establish general safety, and checker flags
are not confirmed unsafe-content labels. Training screening retained 858 of 1098
training images; conservative screening may also remove benign anime images.

Ten nearest-training-image CLIP cosines range from **0.5259 to 0.7513**. Similarity
can reflect shared characters, style or composition and cannot alone establish
copying or its absence. Inspect the private nearest-neighbor panels before making
any stronger claim. Disjoint train/test source IDs establish split separation,
but do not guarantee the absence of visually related scenes across the split.

## Conclusion and comparison boundary

This run demonstrates execution of the complete partner baseline: preparation,
real adapter training, paired generation and recorded evaluation. It provides an
observed KID reduction alongside slightly lower aggregate alignment and diversity,
and the inspected outputs still have composition and attribute failures. Human
preference for this personal baseline remains unmeasured.

The 600-step partner baseline does not replace Pratiksha Kaushik's formal
3,000-step rank-4/rank-32 experiments or their chosen model. The accepted train
count here is 858 versus the teammate's reported 859; the cause of this one-image
screening difference has not been established. Different training budgets,
screening outcomes and safety handling prevent treating the personal run as an
equal-budget comparison with those formal models. The reused fixed prompts are
development prompts, not a newly independent holdout.

Yuyao should check these observations against the private images, revise the
interpretation in their own words and be ready to explain the pipeline. The
actual knowledge-transfer meeting date, topic and recording must come from the
team; this draft does not supply or certify that record.
