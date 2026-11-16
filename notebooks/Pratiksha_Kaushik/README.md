# Pratiksha Kaushik

**DATA 266 · Lab 2 · Team 41** · San José State University

This folder holds my notebooks for all three parts of Lab 2. I led **Part 2 (diffusion fine-tuning)**, whose notebook is the team's Part 2 submission. For Parts 1 and 3 I built and ran my own complete pipelines alongside the team submissions.

For the whole submission, start at the [main README](../../README.md). For who did what, see [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md).

---

## At a glance

| Part | My role | Notebook | Headline result | Details |
| --- | --- | --- | --- | --- |
| **1 · Multimodal RAG** | Independent pipeline | [`p1_rag_v15.ipynb`](p1/p1_rag_v15.ipynb) | 14/25 correct vs 10/25 for the dense-only baseline | [Part 1 README](p1/README.md) |
| **2 · Diffusion fine-tuning** | **Lead · team submission** | [`p2_submission.ipynb`](p2/p2_submission.ipynb) | KID 0.0217 vs 0.1337 for base SD 1.5; humans prefer its style 83% of the time | [Part 2 README](p2/README.md) |
| **3 · Multi-agent system** | Independent pipeline | [`p3_agents_v2.ipynb`](p3/p3_agents_v2.ipynb) | 5 agents with a critique loop; 9 of 10 red-team attacks blocked | [Part 3 README](p3/README.md) |

The team submissions for Parts 1 and 3 are Yuyao Ding's [`p1_rag.ipynb`](../Yuyao_Ding/p1_rag.ipynb) and [`p3_agents.ipynb`](../Yuyao_Ding/p3_agents.ipynb).

---

## Part 2 · Fine-tuning Stable Diffusion 1.5 on Naruto images

Two LoRA runs (rank 4 and rank 32) on the UNet attention layers, trained with the same data and step budget and compared with the base model on 20 fixed prompts.

| Measurement | Base SD 1.5 | Final model (LoRA r=32) |
| --- | ---: | ---: |
| KID vs test images (lower is better) | 0.1337 | **0.0217** |
| CLIP score, 20 prompts | **31.54** | 29.84 |
| LPIPS diversity between seeds | **0.808** | 0.749 |
| Human win rate vs base: style / prompt match (3 raters, 25 blinded pairs) | 0.50 / 0.50 | **0.83** / 0.40 |
| Training cost | — | 1.37 GPU-hours, 3.61 GB peak on a Tesla T4 |

**Final model:** [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32)

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("PratikshaK-1/naruto-sd15-lora-r32")
image = pipe("a man with blue hair and a headband").images[0]
```

## Part 1 · Multimodal RAG (independent pipeline)

Hybrid dense + BM25 retrieval over the three PDFs, with figure and table extraction and Qwen2.5-VL-3B answers in the required JSON format, evaluated on my own 25-question golden set.

| Measurement | Dense only | Final |
| --- | ---: | ---: |
| Correct answers out of 25 | 10 | **14** |
| Recall@5 | — | 0.864 |
| Median seconds per question | 0.017 | 3.10 |

## Part 3 · Multi-agent system (independent pipeline)

Planner → Vision Understanding → Prompt Engineering → Image Generation → Critique, with up to 2 revisions. Built with Qwen2.5-VL-3B, SD 1.5 img2img, InstructPix2Pix and my Part 2 LoRA, plus a Gradio demo app.

| Measurement | No critique loop | With critique loop |
| --- | ---: | ---: |
| Planner routes the request correctly | 14/15 | 14/15 |
| Median seconds per request | **11.1** | 16.8 |
| Red-team attacks that got through | — | 1/10 |

---

## Repository layout

```text
notebooks/Pratiksha_Kaushik/
├── README.md                       # this page
├── p1/                             # Part 1 notebook versions (v1–v6, v15)
├── p2/
│   ├── p2_submission.ipynb         # Part 2 submission, executed end to end
│   └── development/                # earlier Part 2 notebooks
└── p3/                             # Part 3 notebooks (v1, v2)

eval/pratiksha_kaushik/part2/       # Part 2 prompts, results, human ratings, model card
eval/pratiksha_kaushik/part3/       # Part 3 scenarios, red-team cases, results, system card
traces/pratiksha_kaushik/part3/     # JSON trace of every Part 3 request
```

## Artifacts

| Artifact | Location |
| --- | --- |
| Part 2 final model (LoRA r=32) and model card | [Hugging Face](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32) |
| Part 2 comparison model (LoRA r=4) | [Hugging Face](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4) |
| Part 2 generated images (120, with prompt, seed and model) | [Hugging Face dataset](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations) |
| Part 2 evaluation files | [eval/pratiksha_kaushik/part2/](../../eval/pratiksha_kaushik/part2/) |
| Part 2 run outputs (cleaned data, adapters, images, logs) | [Google Drive](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link) |
| Part 1 data and outputs | [Google Drive](https://drive.google.com/drive/folders/1gzCpIAkHvJvOtVwFEoUw1UiM9SIOwvpc) |
| Part 3 evaluation files and traces | [eval/pratiksha_kaushik/part3/](../../eval/pratiksha_kaushik/part3/), [traces/pratiksha_kaushik/part3/](../../traces/pratiksha_kaushik/part3/) |

## Reproducibility

- **Hardware:** every notebook was run end to end on a single Tesla T4 in Google Colab.
- **How to run:** open a notebook in Colab with a GPU and choose **Runtime → Restart session and run all**. Each notebook installs its own pinned package versions in its first code cell.
- **Seeds:** Part 2 and Part 3 use seed 266; Part 2 images use seeds 0 and 1.
- **Models:** all models are open source and run locally (Stable Diffusion 1.5, InstructPix2Pix, Qwen2.5-VL-3B, CLIP). No hosted API is used.
- **Storage:** model weights and images live on Hugging Face and Google Drive, not in GitHub, as the course requires.
