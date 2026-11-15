# DATA 266 Lab 2 — Team 41

Shared implementation, results and documentation for DATA 266 Lab 2.

**Team:** Yuyao Ding and Pratiksha Kaushik. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for who did what.

## Submission at a glance

Graders use the tagged commit for each part.

| Part | Git tag | Main notebook | Headline result | Details |
| --- | --- | --- | --- | --- |
| 1. Multimodal RAG | `p1-v1.0` | [`notebooks/Yuyao_Ding/p1_rag.ipynb`](notebooks/Yuyao_Ding/p1_rag.ipynb) | Recall@5 0.764 (final) vs 0.472 (dense only); human groundedness 4.44/5 | [Part 1 results](#part-1--multimodal-rag) |
| 2. Diffusion fine-tuning | `p2-v1.0` | [`notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb`](notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb) | KID 0.0217 (LoRA r=32) vs 0.1337 (base SD 1.5) | [Part 2 README](notebooks/Pratiksha_Kaushik/p2/README.md) |
| 3. Multi-agent system | `p3-v1.0` | [`notebooks/Yuyao_Ding/p3_agents.ipynb`](notebooks/Yuyao_Ding/p3_agents.ipynb) | Task success 10/15 with loop, 11/15 without; 0/10 red-team attacks succeeded | [Part 3 report](eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md) |

All course GPU runs were done on a single Tesla T4 (Colab).

## Repository structure

```text
.
├── README.md                    # this file
├── CONTRIBUTIONS.md             # who did what
├── AI_USAGE.md                  # AI tool disclosure
├── requirements.txt             # pinned Part 1 dependencies
├── config/                      # optional environment records
├── notebooks/
│   ├── Yuyao_Ding/
│   │   ├── p1_rag.ipynb                  # Part 1 submission
│   │   ├── p1_round2_comparison.ipynb    # Part 1 round 1 vs round 2
│   │   ├── p2_baseline.ipynb             # Yuyao executed partner baseline
│   │   └── p3_agents.ipynb               # Part 3 submission
│   └── Pratiksha_Kaushik/
│       ├── p1/                  # Part 1 development notebooks (v1–v6, v15)
│       ├── p2/                  # Part 2 notebooks + README (p2_submission.ipynb is the submission)
│       └── p3/                  # independent Part 3 build (v1, v2) + README
├── eval/
│   ├── yuyao_ding/             # main Parts 1/3 + Yuyao Part 2 baseline
│   └── pratiksha_kaushik/      # Pratiksha's independent evaluation records
├── traces/
│   ├── yuyao_ding/             # main Part 3 T4 request traces
│   └── pratiksha_kaushik/      # independent Part 3 request traces
├── src/                        # Python packages: one namespace per member
│   ├── yuyao_ding/
│   └── pratiksha_kaushik/
├── scripts/                    # member runners and review tools
│   ├── yuyao_ding/
│   └── pratiksha_kaushik/
├── tests/                      # member test packages
│   ├── yuyao_ding/
│   └── pratiksha_kaushik/
├── config/                     # member environment records
│   ├── yuyao_ding/
│   └── pratiksha_kaushik/
└── audit/                      # member provenance and preserved history
    ├── yuyao_ding/
    └── pratiksha_kaushik/
```

Model weights and the full raw run archives are not in this repository. See [Artifacts](#artifacts).

The root README, dependency entry point, contribution/AI disclosures, and Python
package entry points are shared. Member folders describe file ownership; an empty
member code folder with a README does not claim additional work. Pratiksha's
current implementations remain in her notebooks. Live imports and commands use
member namespaces, for example `python -m src.yuyao_ding.p3_app`. Notebook folders
retain their existing capitalization; Python/member packages use lowercase names. Historical
paths/hashes in original records remain unchanged; see the
[member-directory mapping](audit/yuyao_ding/workspace_relocation.json).


## Setup

Course GPU work must run on one 16 GB GPU (Kaggle T4/P100 or Colab T4).

- **Part 1:** create a Python environment and install the pinned dependencies:
  ```bash
  python -m pip install -r requirements.txt
  jupyter lab
  ```
- **Parts 2 and 3:** each notebook installs its own pinned packages in its first code cell. Run them in Colab with **Runtime → Restart session and run all**.
- **Part 3 environment:** the earlier local development snapshot is kept at `config/yuyao_ding/requirements-part3.txt` for provenance only. See the [Part 3 runbook](eval/yuyao_ding/PART3_IMPLEMENTATION.md) before changing an existing GPU environment.

Each notebook begins with a reproducibility cell that prints the Git commit, Python version, PyTorch version, GPU and random seed.

---

## Part 1 — Multimodal RAG

**Run:** open [`notebooks/Yuyao_Ding/p1_rag.ipynb`](notebooks/Yuyao_Ding/p1_rag.ipynb) with the Part 1 inputs attached. Set `DATA266_RUN_FULL_PART1=1` to answer all questions, write `submission.csv` and run the provided validator.

**System:**
- **Ingestion:** text chunks keep their document ID and page. Every Figure and Table is cropped with its label and caption, and figures are made searchable with CLIP image embeddings. Everything is stored in ChromaDB with the metadata from `kb_manifest.json`.
- **Retrieval:** hybrid dense + BM25 search, with exact routing when a question names a Figure or Table.
- **Answers:** grounded JSON answers from a Qwen vision-language model, with a citation for every fact.
- **Conversations:** each conversation has its own memory. User-supplied values stay in that memory and never go into the vector database.

See the [architecture](eval/yuyao_ding/PART1_ARCHITECTURE.md) and [evaluation protocol](eval/yuyao_ding/PART1_EVALUATION_PROTOCOL.md).

### Results: round 2, Colab Tesla T4 (25-question golden set)

| Measurement | Baseline (dense only) | Final system |
| --- | ---: | ---: |
| Retrieval Recall@5 (24 questions with evidence) | 0.472 | **0.764** |
| Median seconds per question | 9.17 | **0.80** |
| Execution errors | 0 | 0 |
| Answer correctness (human source review) | 7/25 (28%) | **19/25 (76%)** |
| Citation accuracy (human source review) | 14/75 (18.67%) | **28/47 (59.57%)** |
| Human groundedness, mean of both members (1–5) | — | **4.44** |
| Both members within one point | — | **25/25 (100%)** |

- **Practice set:** 40 questions with 0 execution errors and 0 format fallbacks, median 5.13 s per question.
- **Memory:** peak GPU memory was 7.88 GiB allocated (9.01 GiB reserved) on a T4.
- **Timing:** the final system's median is low because many answers come straight from structured evidence or session memory without a model call. It is not a measure of model inference speed.
- **Earlier Windows run:** an earlier Windows RTX 4090 run, with an AI-assisted and provisional content review, scored 18/25 correct for the final system vs 6/25 for dense only. Citation accuracy was 0.49 vs 0.14. See [Part 1 measured results](eval/yuyao_ding/PART1_MEASURED_RESULTS.md). Those numbers are not the required human review and do not apply to the round-2 run.

Supporting material:
- [completed source-content review, result table and numerical-scoring boundary](eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/content_review_20261115/RESULTS.md)
- [round-2 results and human groundedness review](eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/)
- [round 1 vs round 2 comparison notebook](notebooks/Yuyao_Ding/p1_round2_comparison.ipynb)
- [round-2 protocol](eval/yuyao_ding/PART1_ROUND2.md)
- [safety analysis](eval/yuyao_ding/PART1_SAFETY_ANALYSIS.md)
- the golden set in `eval/yuyao_ding/p1_golden_v2.json`

The Colab rerun helper is [`scripts/yuyao_ding/p1_colab_schema_fix_and_rerun.py`](scripts/yuyao_ding/p1_colab_schema_fix_and_rerun.py). In the recorded run it was uploaded to `/content/`.

---

## Part 2 — Diffusion fine-tuning

**Run:** open [`notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb`](notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb) in Colab (T4) and run all. Full details are in the [Part 2 README](notebooks/Pratiksha_Kaushik/p2/README.md).

**Yuyao's partner baseline:** the separate [executed baseline notebook](notebooks/Yuyao_Ding/p2_baseline.ipynb)
completed data preparation, 600-step rank-4 LoRA training, 40 base plus 40 adapted
images and evaluation on a real single Colab Tesla T4. Its [measured record](eval/yuyao_ding/part2/results/yuyao_t4_20261115T034917Z/)
has KID 0.1337 → 0.0910, CLIP 31.1084 → 30.7026 and LPIPS 0.8116 → 0.7928.
Peak training allocation was 3.6810 GiB. The initial pre-training logging failure,
source transition and private original artifacts are preserved. This shorter
baseline is separate from Pratiksha's two formal runs and final model. The
[AI-assisted result analysis](eval/yuyao_ding/part2/results/yuyao_t4_20261115T034917Z/ANALYSIS.md)
explains metric tradeoffs and concrete output limitations; Yuyao's verification/
rewording and actual knowledge-transfer recording remain pending.
See the [runbook and attribution](eval/yuyao_ding/PART2_YUYAO_BASELINE.md).

**Setup:**
- **Data and models:** Stable Diffusion 1.5 fine-tuned on [Naruto BLIP Captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions).
- **Cleaning:** 1,221 images; 0 corrupted, 0 duplicates; 90/10 split with seed 266. The NSFW filter removed 239 training images, leaving 859 train and 123 test, all at 512×512.
- **Training:** two LoRA runs on the UNet attention layers with the same data and steps: rank 4 and rank 32, 3,000 steps, batch 4, LR 1e-4.
- **Generation:** 20 fixed prompts (10 in-domain, 5 out-of-domain, 5 compositional) × seeds 0 and 1 = 40 images per model.

### Results

| Metric | Base SD 1.5 | Run 1 (LoRA r=4) | Run 2 (LoRA r=32, final) |
| --- | ---: | ---: | ---: |
| KID vs test images (lower is better) | 0.1337 | 0.0389 | **0.0217** |
| CLIP score, all 20 prompts | **31.54** | 30.35 | 29.84 |
| LPIPS diversity between seeds | **0.808** | 0.776 | 0.749 |
| AI-judge (CLIP) win rate vs base, style / prompt | 0.50 / 0.50 | — | 0.92 / 0.20 |
| Human win rate vs base, style / prompt (25 blinded pairs, 3 raters) | 0.50 / 0.50 | — | **0.83 / 0.40** |
| GPU-hours (train) | — | 1.275 | 1.372 |
| Peak GPU memory (train) | — | 3.54 GB | 3.61 GB |

- **Final model:** Run 2. It has the best domain style at almost the same cost as Run 1.
- **Failure modes:** weaker prompt match on in-domain prompts, and less diversity between seeds.
- **Safety:** no sign of memorization. No generation is closer to the training set than unseen test images are.

**Hugging Face:**
- [final model (r=32)](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32)
- [comparison model (r=4)](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4)
- [all 120 generated images](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations)

---

## Part 3 — Multi-agent system

**Run:** open [`notebooks/Yuyao_Ding/p3_agents.ipynb`](notebooks/Yuyao_Ding/p3_agents.ipynb). The frozen scenarios and input verification steps are in [eval/yuyao_ding/PART3.md](eval/yuyao_ding/PART3.md). The [Part 3 runbook](eval/yuyao_ding/PART3_IMPLEMENTATION.md) covers the five stages, the local app, GPU validation and the review workflow.

### Results: Colab Tesla T4, source commit `53c92a3`

| Measurement | With quality loop | Without quality loop | Red team |
| --- | ---: | ---: | ---: |
| Cases | 15 | 15 | 10 |
| Execution errors | 0 | 0 | 0 |
| Human task success | **10/15 (66.7%)** | **11/15 (73.3%)** | N/A |
| Human attack success rate | N/A | N/A | **0/10 (0%)** |
| Independent reviewer agreement | 13/15 (86.7%) | 12/15 (80.0%) | 9/10 (90.0%) |
| Critic/human agreement | 6/9 (66.7%) | 5/9 (55.6%) | 1/1 (100%) |
| Median seconds per request | 17.23 | 16.74 | 0.0002 |

- **Memory:** peak GPU memory was 10.32 GiB allocated (11.04 GiB reserved).
- **Quality-loop result:** the critic accepted every first attempt, so the loop never revised an image. The paired images in the two modes are identical. The 10/15 vs 11/15 difference therefore comes from the reviewers' judgments, not from the quality loop.
- **Red-team scope:** the 0/10 attack-success result applies only to this small fixed test set.

Supporting material:
- [measured report](eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md)
- [human reviews and joint adjudication](eval/yuyao_ding/part3/results/t4_20261114T060249Z/)
- [completed CPU notebook result table](eval/yuyao_ding/part3/results/t4_20261114T060249Z/CPU_RESULTS.md)
- [system card](eval/yuyao_ding/PART3_SYSTEM_CARD.md)
- [redacted request traces](traces/yuyao_ding/t4_20261114T060249Z/)

**Independent second build:** Pratiksha Kaushik also built and ran a separate Part 3 system (Qwen2.5-VL-3B with SD 1.5, InstructPix2Pix and the Part 2 LoRA). It is kept for comparison, with per-question tables, in [`notebooks/Pratiksha_Kaushik/p3/`](notebooks/Pratiksha_Kaushik/p3/README.md).

---

## Artifacts

| What | Where |
| --- | --- |
| Part 2 LoRA adapters and model cards | [r=32 (final)](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32), [r=4](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4) on Hugging Face |
| Part 2 generated images (120) | [Hugging Face dataset](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations) |
| Yuyao Part 2 partner baseline weights (r=4, 600 steps) | [Hugging Face model](https://huggingface.co/ashley999999/data266-team41-yuyao-lora-r4-600steps) |
| Yuyao Part 2 paired generated images (80) | [Hugging Face dataset](https://huggingface.co/datasets/ashley999999/data266-team41-yuyao-part2-generations) |
| Part 3 actual T4 generated outputs (23; includes 4 diagnostics) | [Hugging Face dataset](https://huggingface.co/datasets/ashley999999/data266-team41-part3-t4-generations) |
| Part 2 run outputs (cleaned data, eval files, logs) | [Google Drive folder](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link) |
| Part 1 saved results | [`eval/yuyao_ding/part1/results/`](eval/yuyao_ding/part1/results/) |
| Part 3 saved results and redacted traces | [`eval/yuyao_ding/part3/results/`](eval/yuyao_ding/part3/results/), [`traces/`](traces/) |

The three Yuyao Hugging Face repositories were verified against all 117 prepared artifact files (103 PNGs and one adapter, plus metadata/cards); see the [publication verification record](audit/yuyao_ding/huggingface_publication_20261115.json). Original files remain in same-named subfolders. The root model card includes the required `subfolder` loading argument, and root dataset cards point to the nested images. The 23 Part 3 images are 19 assessed generation attempts plus 4 independent diagnostics; they are not 23 scored scenarios. Dataset Viewer indexing was still pending at verification.

Model weights, checkpoints and full raw run archives stay outside GitHub. Local inputs go in the ignored `data/` directory, and local generated artifacts go in the ignored `.runtime/` directory.

## Documentation

- [CONTRIBUTIONS.md](CONTRIBUTIONS.md): who did what, by part and task.
- [AI_USAGE.md](AI_USAGE.md): every use of AI tools and how it was checked.
- [eval/README.md](eval/README.md): guide to the evaluation files.

## Remaining before final submission

- [x] Part 1: human answer-correctness and citation-accuracy review, plus completed two-member groundedness.
- [x] Part 1: Kaggle submission completed, as confirmed by Yuyao.
- [x] Part 2: blinded human evaluation (25 pairs, 3 raters).
- [x] Part 2: Yuyao's partner baseline GPU execution and measured outputs saved.
- [x] Part 3: completed human metrics calculated and saved as CPU notebook output.
- [ ] Part 2: Yuyao verifies/rewords the AI-assisted baseline interpretation and adds the actual knowledge-transfer recording.
- [ ] Create or update the tags `p1-v1.0`, `p2-v1.0` and `p3-v1.0` on the final commits.
- [ ] Both members confirm [CONTRIBUTIONS.md](CONTRIBUTIONS.md) and complete [AI_USAGE.md](AI_USAGE.md).
- [ ] Final report.
