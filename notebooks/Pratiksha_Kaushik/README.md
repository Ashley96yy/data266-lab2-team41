# Pratiksha Kaushik — workspace

DATA 266 Lab 2, Team 41. This folder holds Pratiksha Kaushik's notebooks for all three parts. The [main README](../../README.md) is the starting point for the whole submission, and [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md) shows who did what.

| Part | Folder | Role | Main notebook |
| --- | --- | --- | --- |
| 1. Multimodal RAG | [p1/](p1/README.md) | Development notebooks (the team submission is Yuyao Ding's [`p1_rag.ipynb`](../Yuyao_Ding/p1_rag.ipynb)) | [p1_rag_v15.ipynb](p1/p1_rag_v15.ipynb) |
| 2. Diffusion fine-tuning | [p2/](p2/README.md) | **Lead; team submission** | [p2_submission.ipynb](p2/p2_submission.ipynb) |
| 3. Multi-agent system | [p3/](p3/README.md) | Independent second build (the team submission is Yuyao Ding's [`p3_agents.ipynb`](../Yuyao_Ding/p3_agents.ipynb)) | [p3_agents_v2.ipynb](p3/p3_agents_v2.ipynb) |

## Folder layout

```text
notebooks/Pratiksha_Kaushik/
├── README.md                 # this file
├── p1/                       # Part 1 notebook versions v1–v6, v15 + README
├── p2/                       # Part 2 submission (p2_submission.ipynb) + README
│   └── development/          # earlier Part 2 notebooks
└── p3/                       # Part 3 build (p3_agents_v1, v2) + README

eval/pratiksha_kaushik/part2/            # Part 2 evaluation files (prompts, results, ratings, model card)
eval/pratiksha_kaushik/part3/            # Part 3 scenarios, red-team cases, results, judgments, system card, run log
traces/pratiksha_kaushik/part3/          # JSON trace of every Part 3 request
```

## Links

| What | Where |
| --- | --- |
| Part 1 data and outputs | [Google Drive folder](https://drive.google.com/drive/folders/1gzCpIAkHvJvOtVwFEoUw1UiM9SIOwvpc) |
| Part 2 final model (LoRA r=32) | [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32) |
| Part 2 comparison model (LoRA r=4) | [PratikshaK-1/naruto-sd15-lora-r4](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4) |
| Part 2 generated images (120) | [PratikshaK-1/naruto-sd15-p2-generations](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations) |
| Part 2 evaluation files (in repo) | [eval/pratiksha_kaushik/part2/](../../eval/pratiksha_kaushik/part2/) |
| Part 2 run outputs (data, adapters, images, logs) | [Google Drive folder](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link) |
| Part 3 evaluation files | [eval/pratiksha_kaushik/part3/](../../eval/pratiksha_kaushik/part3/) |
| Part 3 traces | [traces/pratiksha_kaushik/part3/](../../traces/pratiksha_kaushik/part3/) |

Images and model weights are kept on Hugging Face and Google Drive, not in GitHub, as the course rules require.
