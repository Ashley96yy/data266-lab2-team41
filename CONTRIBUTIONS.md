# Contributions

DATA 266 Lab 2, Team 41.

| Team member | GitHub | Main areas |
| --- | --- | --- |
| Yuyao Ding | `Ashley96yy` | Repository setup, Part 1 (main RAG system), Part 3 |
| Pratiksha Kaushik | `pratiksha kaushik` | Part 2, Part 1 (alternative notebook versions) |

This file is based on the commit history of this repository. Each member should check their own section before submission.

## Repository and shared work

| Work | Member |
| --- | --- |
| Project structure, member notebook folders, root README | Yuyao Ding |
| Part 2 sections of the root README | Pratiksha Kaushik |

## Part 1: Multimodal RAG

| Work | Member | Where |
| --- | --- | --- |
| PDF text extraction with page-preserving chunking | Yuyao Ding | `src/`, `notebooks/Yuyao_Ding/p1_rag.ipynb` |
| Figure and Table inventory and cropping | Yuyao Ding | same |
| CLIP image embeddings, ChromaDB storage with manifest metadata | Yuyao Ding | same |
| Hybrid retrieval (dense + BM25) and exact figure/table label routing | Yuyao Ding | same |
| Grounded JSON answers with Qwen vision-language model and citations | Yuyao Ding | same |
| Per-conversation memory, user-supplied values and overrides | Yuyao Ding | same |
| Golden evaluation set, baseline vs final evaluation, second Colab evaluation round | Yuyao Ding | `eval/`, `outputs/part1/`, `notebooks/Yuyao_Ding/` |
| Independent Part 1 RAG notebook versions (v1 to v6, v15) | Pratiksha Kaushik | `notebooks/Pratiksha_Kaushik/p1/` |

## Part 2: Diffusion fine-tuning

| Work | Member | Where |
| --- | --- | --- |
| Data preparation: cleaning, de-duplication, 90/10 split, 512×512 resize, NSFW filter | Pratiksha Kaushik | `notebooks/Pratiksha_Kaushik/p2/p2_v4.ipynb` |
| 20 fixed evaluation prompts and baseline (SD 1.5) generation and metrics | Pratiksha Kaushik | same |
| Run 1 (LoRA rank 4) and Run 2 (LoRA rank 32) training, 40 images per model | Pratiksha Kaushik | same |
| KID, CLIP score, LPIPS diversity, efficiency measurements, AI-judge evaluation | Pratiksha Kaushik | same |
| Failure modes, final run choice, safety and memorization checks | Pratiksha Kaushik | same |
| Model cards and Hugging Face upload of both adapters and all images | Pratiksha Kaushik | [Hugging Face](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32) |
| Part 2 README | Pratiksha Kaushik | `notebooks/Pratiksha_Kaushik/p2/README.md` |

## Part 3: Multi-agent system

| Work | Member | Where |
| --- | --- | --- |
| Scenarios, safety test cases and test-input fixtures | Yuyao Ding | `eval/` |
| Planner input gate, vision-understanding adapter, prompt-engineering agent | Yuyao Ding | `src/p3_*`, `notebooks/Yuyao_Ding/p3_agents.ipynb` |
| Generation, critique, orchestration, UI and evaluation | Yuyao Ding | same |
| Tests and Colab T4 run with measured report | Yuyao Ding | `tests/`, `eval/PART3_T4_MEASURED_RESULTS.md` |

## Work done by both members

| Work | Status |
| --- | --- |
| Part 1 human groundedness ratings (both members rate all 25 answers) | Pending |
| Part 2 blinded human evaluation (25 pairs, 3 or more raters) | Pending |
| Part 3 independent quality and safety reviews | Pending |
| Final report | Pending |
