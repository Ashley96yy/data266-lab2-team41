# Yuyao Ding — workspace

DATA 266 Lab 2, Team 41. This folder holds Yuyao Ding's main Part 1 and Part 3
notebooks, the executed Part 2 partner baseline, and supporting analysis notebooks.
The [main README](../../README.md) is the starting point for the whole submission,
and [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md) records individual and shared work.
AI-assisted implementation, analysis and verification are disclosed in
[AI_USAGE.md](../../AI_USAGE.md).

| Part | Role | Main notebook | Supporting material |
| --- | --- | --- | --- |
| 1. Multimodal RAG | **Lead; team submission** | [p1_rag.ipynb](p1_rag.ipynb) | [Round comparison and completed review aggregation](p1_round2_comparison.ipynb) |
| 2. Diffusion fine-tuning | Partner working baseline, implemented and executed separately | [p2_baseline.ipynb](p2_baseline.ipynb) | [Measured baseline results](../../eval/part2/results/yuyao_t4_20261115T034917Z/) |
| 3. Multi-agent system | **Lead; team submission** | [p3_agents.ipynb](p3_agents.ipynb) | [T4 measured report](../../eval/PART3_T4_MEASURED_RESULTS.md), [human reviews](../../eval/part3/results/t4_20261114T060249Z/) |

Pratiksha Kaushik leads the team's formal Part 2 comparison and final model in
[p2_v4.ipynb](../Pratiksha_Kaushik/p2/p2_v4.ipynb). Her independent Part 1 and
Part 3 implementations are documented in [her workspace](../Pratiksha_Kaushik/README.md).
The partner baseline here does not replace her two formal training runs.

## Folder layout

```text
notebooks/Yuyao_Ding/
├── README.md                        # this file
├── p1_rag.ipynb                      # main Part 1 notebook, recorded Colab outputs
├── p1_round2_comparison.ipynb         # saved-round comparison + completed human metrics
├── p1_metrics.ipynb                  # earlier development-run metrics and review analysis
├── p2_baseline.ipynb                 # Yuyao's executed rank-4 partner baseline
├── p3_agents.ipynb                   # main Part 3 notebook, recorded Colab T4 outputs
└── history/
    ├── p1_round1_colab.executed.ipynb
    └── p1_round2_colab.original_executed.ipynb

eval/part1/results/                  # original round records and subsequent human reviews
eval/part2/results/                  # partner baseline metrics, provenance and analysis
eval/part3/results/                  # main Part 3 reviews, adjudication and metrics
traces/t4_20261114T060249Z/           # redacted main Part 3 request traces
```

The `eval/` and `traces/` paths are at the repository root, not inside this
notebook folder. Reusable implementations live in [`src/`](../../src/), and
runners, evidence-export tools and review forms live in [`scripts/`](../../scripts/).

## Part 1: main multimodal RAG system

The implementation covers the full read-only document-QA workflow:

- **Ingestion and indexing:** page-preserving PDF text extraction; numbered
  Figure/Table inventory and crops; CLIP visual retrieval; ChromaDB metadata
  carrying document identity, page, exact visual label and manifest trust.
- **Retrieval and answers:** dense text/visual retrieval as the baseline;
  dense + BM25, exact-label routing and structured evidence in the final system;
  Qwen-based grounded JSON answers with citation and format validation.
- **Conversation handling:** separate state per conversation, follow-up context,
  session-only user assumptions and refusal of permanent knowledge-base updates.
- **Safety:** exclusion of DOC3 Appendix C from evidence; source-trust and
  correction/revision handling; unsupported-answer and false-premise behavior.
- **Evaluation and evidence:** a frozen 25-case development golden set, paired
  dense-only/final evaluations, preserved first/second Colab runs, source-content
  review and independent two-member groundedness scoring.

### Recorded second-round results

These values come from the completed Colab Tesla T4 run and the subsequently
supplied reviews. The [comparison notebook](p1_round2_comparison.ipynb) computes
the completed table from saved files on CPU, without new inference.

| Measurement | Dense-only baseline | Final system |
| --- | ---: | ---: |
| Answer correctness | 7/25 (28%) | 19/25 (76%) |
| Retrieval Recall@5 (24 evidence-bearing questions) | 0.472 | 0.764 |
| Citation accuracy | 14/75 (18.67%) | 28/47 (59.57%) |
| Two-member groundedness, mean (1–5) | Not collected | 4.44 |
| Groundedness agreement within one point | Not collected | 25/25 (100%) |
| Median seconds/question | 9.1736 | 0.7958 |

Yuyao supplied the answer-correctness and per-citation source review. Yuyao and
Pratiksha independently supplied the final-system groundedness scores. These
measurements are distinct; a groundedness score is not converted into correctness.
The original review file, one user-confirmed baseline correction, and the
numerical-token-check limitation are recorded in the
[completed content-review results](../../eval/part1/results/round2_colab_20261114T032206Z/content_review_20261115/RESULTS.md).

The final latency includes structured-evidence/session-state answers that avoid
a VLM call; it is not model-only inference speed. The evaluations followed
recorded development fixes and are not an untouched holdout. Improved format
stability across rounds is separate from answer correctness and Kaggle results.
Kaggle submission has been completed, as confirmed by Yuyao; public leaderboard
results remain separate from the golden-set metrics and hidden-set grading.

`p1_metrics.ipynb` preserves an earlier development analysis. Use the second-round
comparison and completed review record above for the current reviewed table;
do not mix older provisional content judgments with these human scores.

## Part 2: executed partner baseline

Yuyao implemented and ran the end-to-end partner baseline on a single Colab Tesla
T4: data preparation, actual rank-4 UNet LoRA training, paired base/adapter image
generation, KID/CLIP/LPIPS evaluation, cost recording and safety diagnostics.

| Item | Recorded partner baseline |
| --- | --- |
| Training | 600 optimizer steps; rank 4; LR 1e-4; effective batch 4 |
| Trainable parameters | 797,184 |
| Accepted train / test images | 858 / 123 |
| Paired generation | 20 fixed prompts × seeds 0 and 1, for base and adapter: 80 images |
| Successful training invocation | 19.78 minutes |
| Peak training allocation | 3.6810 GiB |
| In-domain KID, base → adapter | 0.1337 → 0.0910 |
| Aggregate CLIP, base → adapter | 31.1084 → 30.7026 |
| LPIPS diversity, base → adapter | 0.8116 → 0.7928 |

Lower observed KID accompanies slightly lower aggregate alignment and diversity;
it does not establish improvement on every dimension. The
[result interpretation](../../eval/part2/results/yuyao_t4_20261115T034917Z/ANALYSIS.md)
includes selected composition/attribute failures and safety/memorization limits.
It is an AI-assisted draft for Yuyao to verify and revise in their own words.

This shorter partner run is separate from Pratiksha's two 3,000-step formal runs.
Its training budget, screening outcome and recorded train count differ, so it is
not an equal-budget third model in the team's formal comparison. The initial
pre-training logging failure and recovery remain recorded. Original private
images and weights are retained; the public notebook omits embedded previews
containing source images. See the [runbook](../../eval/PART2_YUYAO_BASELINE.md)
and [recorded provenance](../../eval/part2/results/yuyao_t4_20261115T034917Z/).

## Part 3: main multi-agent system

The implementation connects Planner, Vision Understanding, Prompt Engineering,
Image Generation and Critique through a shared typed state. It supports caption
and visual questions, SD1.5 image-to-image style changes, domain-style generation
using the team's Part 2 LoRA, and geometry-preserving photometric enhancement.

The Planner handles refusal, clarification and readable input errors. Image text
is treated as data rather than an instruction. Critique records CLIP similarity
and separate VLM quality/safety judgments, with at most two quality revisions.
Timed agent events are saved as JSON traces. The English Gradio interface shows
agent outputs; the local entry point is `python -m src.p3_app` from the repository
root with the appropriate GPU environment and private inputs available.

### Recorded T4 evaluation and human review

The unchanged suite contains 15 primary scenarios in each condition and 10
red-team cases. Actual inference used one Colab Tesla T4, source commit
`53c92a3`, seed 266, with zero execution errors. Both members supplied independent
reviews, followed by documented joint adjudication.

| Measurement | With quality loop | Without quality loop | Red team |
| --- | ---: | ---: | ---: |
| Cases | 15 | 15 | 10 |
| Human task success | 10/15 (66.67%) | 11/15 (73.33%) | N/A |
| Human attack success | N/A | N/A | 0/10 (0%) |
| Independent reviewer case agreement | 13/15 (86.67%) | 12/15 (80.00%) | 9/10 (90.00%) |
| Critic/human agreement | 6/9 (66.67%) | 5/9 (55.56%) | 1/1 (100%) |
| Median seconds/request | 17.2280 | 16.7367 | 0.0002 |

All nine paired generated images were byte-identical, and the critic accepted
every first attempt. No quality revision occurred in this recorded suite, so the
difference in human task scores is not evidence that the loop improved or harmed
the images. The zero observed attacks applies only to this small fixed suite.
Peak T4 allocation was 10.32 GiB; earlier local development/capacity checks remain
separate historical records, not substitutes for the actual T4 run.

## Results, runbooks and related links

| What | Where |
| --- | --- |
| Part 1 architecture | [PART1_ARCHITECTURE.md](../../eval/PART1_ARCHITECTURE.md) |
| Part 1 evaluation protocol | [PART1_EVALUATION_PROTOCOL.md](../../eval/PART1_EVALUATION_PROTOCOL.md) |
| Part 1 safety analysis | [PART1_SAFETY_ANALYSIS.md](../../eval/PART1_SAFETY_ANALYSIS.md) |
| Part 1 completed content review | [Second-round results](../../eval/part1/results/round2_colab_20261114T032206Z/content_review_20261115/) |
| Part 1 independent groundedness | [Original ratings and aggregation](../../eval/part1/results/round2_colab_20261114T032206Z/human_review_20261114/) |
| Part 2 partner baseline | [Results and provenance](../../eval/part2/results/yuyao_t4_20261115T034917Z/) |
| Team Part 2 final model | [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32) |
| Team Part 2 comparison model | [PratikshaK-1/naruto-sd15-lora-r4](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4) |
| Team Part 2 image dataset | [PratikshaK-1/naruto-sd15-p2-generations](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations) |
| Part 3 runbook | [PART3_IMPLEMENTATION.md](../../eval/PART3_IMPLEMENTATION.md) |
| Part 3 measured report | [PART3_T4_MEASURED_RESULTS.md](../../eval/PART3_T4_MEASURED_RESULTS.md) |
| Part 3 independent reviews and adjudication | [T4 review record](../../eval/part3/results/t4_20261114T060249Z/) |
| Part 3 published request traces | [Redacted JSON traces](../../traces/t4_20261114T060249Z/) |
| Part 3 system-card draft | [PART3_SYSTEM_CARD_DRAFT.md](../../eval/PART3_SYSTEM_CARD_DRAFT.md) |

Keep original course PDFs, private fixtures and portraits, generated image files,
weights, caches and full retrieved contexts outside GitHub. This README adds no
private assets.
The report, final release references and meeting recordings are team-level
submission items; this workspace README documents the recorded work rather than
claiming those deliverables are finished.
