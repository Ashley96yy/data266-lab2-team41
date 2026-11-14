# DATA 266 Lab 2 — Team 41

This repository contains the shared implementation and report materials for DATA 266 Lab 2.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-part3.txt
├── config/
│   └── windows/                 # validated environment snapshots
├── notebooks/
│   ├── Yuyao_Ding/
│   │   ├── p1_rag.ipynb
│   │   └── p3_agents.ipynb
│   └── Pratiksha_Kaushik/
│       ├── p1/              # retained Part 1 development notebooks
│       └── p2_v4.ipynb
├── eval/
├── traces/
├── src/
├── scripts/                     # one-off runners and review tools
├── CONTRIBUTIONS.md
├── AI_USAGE.md
└── .gitignore
```

- `eval/` stores the golden set, prompts, and evaluation scenarios.
- `traces/` stores the Part 3 JSON traces.
- `src/` contains reusable helpers shared by the notebooks.
- `scripts/` contains executable helpers, including the preserved Part 1 Colab rerun script.
- `config/windows/` contains the Part 1 and Part 3 Windows environment snapshots and constraints. These are provenance records, not instructions to replace a working CUDA/PyTorch installation.
- The member folders contain the primary notebooks and retained development versions. The folder names organize files; both members contribute to all three parts. Record each person's work in `CONTRIBUTIONS.md`.

## Setup

For a fresh Part 1 environment, create and activate a Python environment, then install the pinned Part 1 dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Part 3 has separate dependencies in `requirements-part3.txt`. See its runbook before installing anything into an existing GPU environment. The historical Windows snapshots under `config/windows/` are not the default installation command.

Start Jupyter from the repository root:

```bash
jupyter lab
```

## Running each part

- **Part 1 — RAG:** open `notebooks/Yuyao_Ding/p1_rag.ipynb` with the Part 1 inputs attached. Set `DATA266_RUN_FULL_PART1=1` to run all questions, write `submission.csv`, and invoke the provided validator.
- **Part 2 — Diffusion fine-tuning:** open `notebooks/Pratiksha_Kaushik/p2_v4.ipynb` in Colab and run all. See [notebooks/Pratiksha_Kaushik/README.md](notebooks/Pratiksha_Kaushik/README.md) for setup, results and Hugging Face links (final model: [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32)).
- **Part 3 — Multi-agent system:** open `notebooks/Yuyao_Ding/p3_agents.ipynb`. The frozen scenarios and input verification instructions are in [eval/PART3.md](eval/PART3.md). See the [Part 3 runbook](eval/PART3_IMPLEMENTATION.md) for the five stages, local app, real GPU validation, and independent review workflow.

Each notebook begins with a reproducibility cell that reports the Git commit, Python version, PyTorch version, GPU information, and random seed.

## Artifact storage

Model weights, checkpoints, and generated images must be uploaded to Hugging Face rather than committed to GitHub. Add the final Hugging Face repository links here when they are available.

Small source files, evaluation summaries, configuration files, and trace metadata may be tracked in this repository when appropriate. Local inputs may be placed in the ignored `data/` directory, and local generated artifacts are written under the ignored `.runtime/` directory.

## Status

The repository keeps the three parts in the shared notebook structure. Part-specific runbooks and recorded environments are linked above. Human reviews, final report materials, and final competition submission remain separate completion items.

## Part 1 second Colab evaluation

The second real Tesla T4 run and first-run comparison are preserved under `outputs/part1/`. See the [round 2 protocol](eval/PART1_ROUND2.md) and [executed comparison notebook](notebooks/Yuyao_Ding/p1_round2_comparison.ipynb). The final cell of `p1_rag.ipynb` records the improved run; the original earlier outputs remain intact. The preserved helper is now at [`scripts/p1_colab_schema_fix_and_rerun.py`](scripts/p1_colab_schema_fix_and_rerun.py). In the recorded upload-to-Colab layout, that file was placed at `/content/p1_colab_schema_fix_and_rerun.py`, which is the path saved in the executed notebook. This directory cleanup does not require another GPU run. Human metrics and final Git provenance are still pending.

## Part 3 GPU development runs

The [executed Part 3 notebook](notebooks/Yuyao_Ding/p3_agents.ipynb) now records a real Colab Tesla T4 run from source commit `53c92a3`: the unchanged 15 scenarios in both modes and 10 red-team cases completed with zero program exceptions. Its [measured report](eval/PART3_T4_MEASURED_RESULTS.md) is calculated from the saved run records; the complete run archive remains private to the team. The earlier RTX 4090 [measured results](eval/PART3_LOCAL_RESULTS.md) and [Windows validation notes](eval/WINDOWS_GPU_VALIDATION.md) remain separate historical evidence. Operational completion and CLIP diagnostics are not human quality or safety scores. Both members' independent reviews remain pending. Keep the working Windows CUDA/PyTorch environment; do not reinstall the root requirements to start Part 3.
