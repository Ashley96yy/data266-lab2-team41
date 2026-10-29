# DATA 266 Lab 2 — Team 41

This repository contains the shared implementation and report materials for DATA 266 Lab 2.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Yuyao_Ding/
│   │   ├── p1_rag.ipynb
│   │   └── p3_agents.ipynb
│   └── Pratiksha_Kaushik/
│       └── p2_finetune.ipynb
├── eval/
├── traces/
├── src/
├── CONTRIBUTIONS.md
├── AI_USAGE.md
└── .gitignore
```

- `eval/` stores the golden set, prompts, and evaluation scenarios.
- `traces/` stores the Part 3 JSON traces.
- `src/` is reserved for optional reusable helpers shared by the notebooks.
- The three notebooks under the member folders are the shared run and submission versions. The folder names organize files; both members contribute to all three parts. Record each person's work in `CONTRIBUTIONS.md`.

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

- **Part 1 — RAG:** open `notebooks/Yuyao_Ding/p1_rag.ipynb` with the Part 1 inputs attached. Set `DATA266_RUN_FULL_PART1=1` to run all questions, write `submission.csv`, and invoke the provided validator.
- **Part 2 — Diffusion fine-tuning:** open `notebooks/Pratiksha_Kaushik/p2_finetune.ipynb`. Detailed training, inference, and evaluation commands will be added with the implementation.
- **Part 3 — Multi-agent system:** open `notebooks/Yuyao_Ding/p3_agents.ipynb`. Detailed setup, execution, and evaluation commands will be added with the implementation.

Each notebook begins with a reproducibility cell that reports the Git commit, Python version, PyTorch version, GPU information, and random seed.

## Artifact storage

Model weights, checkpoints, and generated images must be uploaded to Hugging Face rather than committed to GitHub. Add the final Hugging Face repository links here when they are available.

Small source files, evaluation summaries, configuration files, and trace metadata may be tracked in this repository when appropriate. Local inputs may be placed in the ignored `data/` directory, and local generated artifacts are written under the ignored `.runtime/` directory.

## Status

The repository follows the required shared structure for all three parts. Part-specific instructions and pinned dependency versions will be completed as the implementation develops.
