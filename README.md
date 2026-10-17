# DATA 266 Lab 2 — Team 41

This repository contains the shared implementation and report materials for DATA 266 Lab 2.

## Repository structure

```text
.
├── part1_rag/
│   ├── src/
│   ├── data/
│   └── evaluations/
├── part2_diffusion/
│   ├── src/
│   ├── data/
│   └── evaluations/
├── part3_agents/
│   ├── src/
│   ├── data/
│   └── evaluations/
├── notebooks/
│   ├── p1_rag.ipynb
│   ├── p2_finetune.ipynb
│   └── p3_agents.ipynb
├── report/
├── CONTRIBUTIONS.md
├── AI_USAGE.md
└── requirements.txt
```

## Setup

Create and activate a Python environment, then install the dependencies once their versions have been finalized in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

## Running each part

- **Part 1 — RAG:** open `notebooks/p1_rag.ipynb`. Detailed data preparation, execution, and evaluation commands will be added with the implementation.
- **Part 2 — Diffusion fine-tuning:** open `notebooks/p2_finetune.ipynb`. Detailed training, inference, and evaluation commands will be added with the implementation.
- **Part 3 — Multi-agent system:** open `notebooks/p3_agents.ipynb`. Detailed setup, execution, and evaluation commands will be added with the implementation.

Each notebook begins with a reproducibility cell that reports the Git commit, Python version, PyTorch version, GPU information, and random seed.

## Artifact storage

Model weights, checkpoints, and generated images must be uploaded to Hugging Face rather than committed to GitHub. Add the final Hugging Face repository links here when they are available.

Small source files, evaluation summaries, configuration files, and report materials may be tracked in this repository when appropriate.

## Status

This initial commit establishes the agreed project structure. Part-specific instructions and pinned dependency versions will be added as the implementation develops.
