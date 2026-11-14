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
- **Part 2 — Diffusion fine-tuning:** open `notebooks/Pratiksha_Kaushik/p2_v4.ipynb` in Colab and run all. See [notebooks/Pratiksha_Kaushik/README.md](notebooks/Pratiksha_Kaushik/README.md) for setup, results and Hugging Face links (final model: [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32)).
- **Part 3 — Multi-agent system:** open `notebooks/Yuyao_Ding/p3_agents.ipynb`. The frozen scenarios and input verification instructions are in [eval/PART3.md](eval/PART3.md); reusable Planner code is in `src/p3_planner.py`. Detailed execution commands will be added with the implementation.

Each notebook begins with a reproducibility cell that reports the Git commit, Python version, PyTorch version, GPU information, and random seed.

## Artifact storage

Model weights, checkpoints, and generated images must be uploaded to Hugging Face rather than committed to GitHub. Add the final Hugging Face repository links here when they are available.

Small source files, evaluation summaries, configuration files, and trace metadata may be tracked in this repository when appropriate. Local inputs may be placed in the ignored `data/` directory, and local generated artifacts are written under the ignored `.runtime/` directory.

## Status

The repository follows the required shared structure for all three parts. Part-specific instructions and pinned dependency versions will be completed as the implementation develops.

## Part 3 continuation

Generation, two-image critique, bounded orchestration, Gradio, and evaluation helpers are implemented. See [Windows runbook and validation limits](eval/PART3_IMPLEMENTATION.md). Install `requirements-part3.txt` in the existing CUDA environment. CPU tests and UI construction do not constitute the frozen model evaluation; real GPU results remain pending.

## Part 1 second Colab evaluation

The second real Tesla T4 run and first-run comparison are preserved under outputs/part1. See [round 2 protocol](eval/PART1_ROUND2.md) and [executed comparison notebook](notebooks/Yuyao_Ding/p1_round2_comparison.ipynb). The final cell of p1_rag.ipynb records the improved run; the original earlier outputs remain intact. For the recorded Colab layout, upload the root p1_colab_schema_fix_and_rerun.py helper to /content before that cell. Human metrics and final Git provenance are still pending.
