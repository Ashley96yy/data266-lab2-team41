# Contributions

DATA 266 Lab 2, Team 41.

## Summary

| Member | GitHub name(s) | Lead on | Also worked on |
| --- | --- | --- | --- |
| **Yuyao Ding** | `Ashley96yy`, `Yuyao Ding` | Part 1 (main RAG system), Part 3, repository setup | Part 3 independent review |
| **Pratiksha Kaushik** | `pratiksha kaushik` | Part 2 | Part 1 development notebooks, Part 3 independent review, team documentation |

Both members are responsible for the whole submission. The tables below show who led each piece of work.

**Commits on `main`** (as of 2026-11-14): Yuyao Ding 81, Pratiksha Kaushik 22. The count reflects working style, not effort. Part 2 was developed in Colab and committed as complete notebooks, while Parts 1 and 3 were committed in small steps.

---

## Part 1: Multimodal RAG

**Lead:** Yuyao Ding. **Main notebook:** [`notebooks/Yuyao_Ding/p1_rag.ipynb`](notebooks/Yuyao_Ding/p1_rag.ipynb).

| Task | Work | Member | Files |
| --- | --- | --- | --- |
| 1.1 Ingestion and indexing | PDF text extraction with page-preserving chunks; Figure and Table inventory and cropping; CLIP image embeddings; ChromaDB storage with `kb_manifest.json` metadata | Yuyao Ding | `src/yuyao_ding/p1_*.py`, main notebook |
| 1.2 Retrieval and answers | Dense + BM25 fusion; exact figure and table label routing; grounded JSON answers with a Qwen vision-language model; citation completion and answer repair | Yuyao Ding | `src/yuyao_ding/p1_evidence.py`, `src/yuyao_ding/p1_labels.py`, `src/yuyao_ding/p1_intent.py` |
| 1.3 Conversations | Separate memory per conversation, bounded context, user-value overrides kept out of the vector database | Yuyao Ding | main notebook, `tests/yuyao_ding/test_p1_*.py` |
| 1.4 Evaluation and safety | Golden set, baseline vs final evaluation, second Colab evaluation round, safety analysis, architecture documentation | Yuyao Ding | `eval/yuyao_ding/p1_golden_*`, `eval/yuyao_ding/PART1_*.md`, `src/yuyao_ding/p1_evaluation.py` |
| 1.5 Kaggle | Ordered question runner and submission output | Yuyao Ding | main notebook |
| Development notebooks | Independent Part 1 RAG notebook versions v1 to v6 and v15 | Pratiksha Kaushik | [`notebooks/Pratiksha_Kaushik/p1/`](notebooks/Pratiksha_Kaushik/p1/) |
| Human groundedness evaluation | Both members independently rated all 25 second-round final answers; original ratings and identity correction preserved | Both | [Recorded groundedness](eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/human_review_20261114/) |
| Answer correctness and citation support | Source-content review of 25 baseline and 25 final answers; one baseline label corrected only after explicit user confirmation | Yuyao Ding | [Recorded content review](eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/content_review_20261115/) |

## Part 2: Diffusion fine-tuning

**Lead:** Pratiksha Kaushik. **Main notebook:** [`notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb`](notebooks/Pratiksha_Kaushik/p2/p2_submission.ipynb). See the [Part 2 README](notebooks/Pratiksha_Kaushik/p2/README.md).

| Task | Work | Member |
| --- | --- | --- |
| 2.1 Data preparation | Corrupted and duplicate checks (MD5 and perceptual hash), 90/10 split with seed 266, 512×512 crop and resize, NSFW filter on training images, dataset description | Pratiksha Kaushik |
| 2.2 Fine-tuning | Run 1 (LoRA rank 4) and Run 2 (LoRA rank 32) with the same data and steps; parameters, LR, steps, batch, peak memory and time recorded; both adapters pushed to Hugging Face with model cards | Pratiksha Kaushik |
| 2.3 Image generation | 20 fixed prompts written before training; seeds 0 and 1 for base and both runs (120 images); side-by-side grids; image dataset on Hugging Face | Pratiksha Kaushik |
| 2.4 Evaluation and safety | KID, CLIP score, LPIPS diversity, GPU-hours and memory; CLIP-based AI judge; failure modes; final run choice; license, NSFW and memorization checks | Pratiksha Kaushik |
| Documentation | Part 2 README, Part 2 sections of the root README | Pratiksha Kaushik |
| Partner working baseline | AI-assisted end-to-end adaptation, actually executed on Colab Tesla T4: data preparation, 600-step rank-4 LoRA, 40+40 images, metrics and run provenance. User-provided notebook/logs/adapter/image hashes checked. AI-assisted result-analysis draft added; Yuyao verification/rewording and the actual knowledge-transfer recording remain pending | Yuyao Ding; [executed notebook](notebooks/Yuyao_Ding/p2_baseline.ipynb), [recorded results](eval/yuyao_ding/part2/results/yuyao_t4_20261115T034917Z/) |
| Human evaluation | Blinded 25 pairs, 3 or more raters | Both, plus an outside rater. **Pending** |

Hugging Face: [final model (rank 32)](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32), [comparison model (rank 4)](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4), [generated images](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations).

## Part 3: Multi-agent system

**Lead:** Yuyao Ding. **Main notebook:** [`notebooks/Yuyao_Ding/p3_agents.ipynb`](notebooks/Yuyao_Ding/p3_agents.ipynb).

| Work | Member | Files |
| --- | --- | --- |
| Scenarios, red-team cases, test inputs and fixture manifest | Yuyao Ding | `eval/yuyao_ding/p3_*.json`, `eval/yuyao_ding/PART3.md` |
| Planner, vision understanding, prompt engineering and rewriting agents | Yuyao Ding | `src/yuyao_ding/p3_planner.py`, `src/yuyao_ding/p3_vision.py`, `src/yuyao_ding/p3_prompt*.py` |
| Generation, critique, enhancement, orchestration, state and app | Yuyao Ding | `src/yuyao_ding/p3_*.py` |
| Tests, Colab T4 run, measured report, redacted traces | Yuyao Ding | `tests/test_p3_*.py`, `eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md`, `traces/` |
| Independent quality and safety review of all cases | Yuyao Ding and Pratiksha Kaushik, separately | `eval/yuyao_ding/part3/results/t4_20261114T060249Z/part3-member-*-review.json` |
| Joint adjudication of disagreements | Both | `eval/yuyao_ding/part3/results/t4_20261114T060249Z/part3-adjudication.json` |

## Repository and documentation

| Work | Member |
| --- | --- |
| Repository structure, member folders, configuration and environment records | Yuyao Ding |
| Root README (Parts 1 and 3 sections) | Yuyao Ding |
| Root README (Part 2 sections), this contributions file | Pratiksha Kaushik |
| Final report | Both. **Pending** |

## Use of AI tools

Both members used AI coding assistants. Each use, and how it was checked, is listed in [AI_USAGE.md](AI_USAGE.md).

## Confirmation

Each member confirms that their section above is accurate.

| Member | Confirmed |
| --- | --- |
| Yuyao Ding | ☐ |
| Pratiksha Kaushik | ☐ |
