# DATA 266 Lab 2 — Team 41

This repository contains the shared implementation and report materials for DATA 266 Lab 2.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config/                      # optional environment records
├── notebooks/
│   ├── Yuyao_Ding/
│   │   ├── p1_rag.ipynb
│   │   └── p3_agents.ipynb
│   └── Pratiksha_Kaushik/
│       ├── p1/              # retained Part 1 development notebooks
│       └── p2/              # Part 2 notebooks and README (p2_v4.ipynb is the submission)
├── eval/
│   └── part1/results/           # preserved Part 1 run results
├── traces/
├── src/
├── scripts/                     # one-off runners and review tools
├── CONTRIBUTIONS.md
├── AI_USAGE.md
└── .gitignore
```

- `eval/` stores the golden set, prompts, evaluation scenarios, and saved Part 1 results under `eval/part1/results/`.
- `traces/` contains redacted JSON traces for all 40 Part 3 T4 evaluation requests. The complete raw run archive, private fixtures, and generated images remain outside GitHub.
- `src/` contains reusable helpers shared by the notebooks.
- `scripts/` contains executable helpers, including the preserved Part 1 Colab rerun script.
- `config/` contains optional environment records from local development. The course GPU evidence is in the executed notebooks and measured reports.
- The member folders contain the primary notebooks and retained development versions. The folder names organize files; both members contribute to all three parts. Record each person's work in `CONTRIBUTIONS.md`.

## Setup

Course-required GPU work must run on one 16 GB GPU (Kaggle T4/P100 or Colab T4). The executed T4 runs and their measured results are linked below.

For a fresh Part 1 environment, create and activate a Python environment, then install the pinned Part 1 dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Part 3's actual Colab T4 package versions are printed in its executed notebook. The earlier local development snapshot is kept at `config/requirements-part3.txt` for provenance; it is not the default installation command for the T4 run. See the Part 3 runbook before changing an existing GPU environment.

Start Jupyter from the repository root:

```bash
jupyter lab
```

## Running each part

- **Part 1 — RAG:** open `notebooks/Yuyao_Ding/p1_rag.ipynb` with the Part 1 inputs attached. Set `DATA266_RUN_FULL_PART1=1` to run all questions, write `submission.csv`, and invoke the provided validator.
- **Part 2 — Diffusion fine-tuning:** open `notebooks/Pratiksha_Kaushik/p2/p2_v4.ipynb` in Colab and run all. See [notebooks/Pratiksha_Kaushik/p2/README.md](notebooks/Pratiksha_Kaushik/p2/README.md) for setup, results and Hugging Face links (final model: [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32)).
- **Part 3 — Multi-agent system:** open `notebooks/Yuyao_Ding/p3_agents.ipynb`. The frozen scenarios and input verification instructions are in [eval/PART3.md](eval/PART3.md). See the [Part 3 runbook](eval/PART3_IMPLEMENTATION.md) for the five stages, local app, real GPU validation, and independent review workflow.

Each notebook begins with a reproducibility cell that reports the Git commit, Python version, PyTorch version, GPU information, and random seed.

## Artifact storage

Model weights, checkpoints, and generated images must be uploaded to Hugging Face rather than committed to GitHub. Add the final Hugging Face repository links here when they are available.

Small source files, evaluation summaries, configuration files, and trace metadata may be tracked in this repository when appropriate. Local inputs may be placed in the ignored `data/` directory, and local generated artifacts are written under the ignored `.runtime/` directory.

## Status

The repository keeps the three parts in the shared notebook structure. Part-specific runbooks and recorded environments are linked above. Human reviews, final report materials, and final competition submission remain separate completion items.

## Part 1 second Colab evaluation

The second real Tesla T4 run and first-run comparison are preserved under [`eval/part1/results/`](eval/part1/results/). See the [round 2 protocol](eval/PART1_ROUND2.md) and [executed comparison notebook](notebooks/Yuyao_Ding/p1_round2_comparison.ipynb). The final cell of `p1_rag.ipynb` records the improved run; the original earlier outputs remain intact. The preserved helper is now at [`scripts/p1_colab_schema_fix_and_rerun.py`](scripts/p1_colab_schema_fix_and_rerun.py). In the recorded upload-to-Colab layout, that file was placed at `/content/p1_colab_schema_fix_and_rerun.py`, which is the path saved in the executed notebook. This directory cleanup does not require another GPU run. The [round-2 final-system human groundedness review](eval/part1/results/round2_colab_20261114T032206Z/human_review_20261114/) is complete: mean 4.44/5, with 25/25 reviewer pairs within one point. Human answer-correctness and citation-support review, final Git provenance and course submission remain pending.

## Part 3 Colab T4 run

The [executed Part 3 notebook](notebooks/Yuyao_Ding/p3_agents.ipynb) records a real Colab Tesla T4 run from source commit `53c92a3`: the unchanged 15 scenarios in both modes and 10 red-team cases completed with zero program exceptions. Its [measured report](eval/PART3_T4_MEASURED_RESULTS.md) is calculated from the saved run records; the complete run archive remains private to the team. Operational completion and CLIP diagnostics are not human quality or safety scores. The [supplied human reviews and joint resolutions](eval/part3/results/t4_20261114T060249Z/) are now recorded: task success is 10/15 with the loop and 11/15 without it, with 0/10 red-team attack successes. Paired generated images are identical, so these human score differences do not demonstrate a quality-loop effect. Final report materials and submission obligations remain pending.
