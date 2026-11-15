# Part 1: development notebooks (Pratiksha Kaushik)

Pratiksha Kaushik's own versions of the Part 1 multimodal RAG pipeline, kept to show how her work on Part 1 developed.

The team's Part 1 submission is [`notebooks/Yuyao_Ding/p1_rag.ipynb`](../../Yuyao_Ding/p1_rag.ipynb). Its results are in the [main README](../../../README.md#part-1--multimodal-rag).

| Notebook | Saved | Notes |
| --- | --- | --- |
| [p1_rag_v15.ipynb](p1_rag_v15.ipynb) | 2026-11-14 | Latest version |
| [p1_rag_v6.ipynb](p1_rag_v6.ipynb) | 2026-11-13 | |
| [p1_rag_v5.ipynb](p1_rag_v5.ipynb) | 2026-11-13 | |
| [p1_rag_v4.ipynb](p1_rag_v4.ipynb) | 2026-11-13 | |
| [p1_rag_v3.ipynb](p1_rag_v3.ipynb) | 2026-11-13 | |
| [p1_rag_v2.ipynb](p1_rag_v2.ipynb) | 2026-11-13 | First complete pipeline draft |
| [p1_rag_v1.ipynb](p1_rag_v1.ipynb) | 2026-11-13 | First version |

Versions 7 to 14 were Colab drafts and weren't kept.

## Results (v15, Colab Tesla T4)

All numbers below are printed in the outputs of `p1_rag_v15.ipynb`. The run used a 25-question golden set (G01–G25) written for this notebook, covering the six required categories: 5 fact/table, 6 figure, 3 two-source, 5 multi-turn in 2 conversations, 3 conflict/revision, 3 unanswerable or false-premise. It is a different set from the team's golden set, so these numbers can't be compared directly with the team results in the main README.

| Measurement | Baseline (dense only) | Final (hybrid + Qwen2.5-VL-3B) |
| --- | ---: | ---: |
| Correct answers out of 25 (golden set) | 10 | **14** |
| Retrieval Recall@5 (questions with a source) | not reported | 0.864 |
| Citation accuracy (automatic: cited sources found in the gold sources) | not reported | 0.617 |
| Median seconds per question (golden set) | 0.017 | 3.10 |
| Human groundedness (both members) | — | Pending |

Practice set (40 questions in `Questions.csv`):
- **Statuses:** 37 answered, 2 session_override, 1 false_premise.
- **Speed:** median 3.28 s per question.
- **Model use:** the Qwen generator's answer was used for 29 of 64 attempts. The rest fell back to extractive answers because the model returned "not found" (9) or an invalid citation (5).

Known problems in this version:
- **Format check fails:** the provided `validate_submission.py` rejects the practice `submission.csv` with "bad figure_references". The submission file isn't valid as written.
- **Seed:** the notebook uses seed 9002 instead of the course seed 266.
- **Development evidence:** the code was tuned against the same golden set, so the golden set isn't a holdout.

## Data and outputs

The inputs and run outputs for these notebooks are in the [Part 1 Google Drive folder](https://drive.google.com/drive/folders/1gzCpIAkHvJvOtVwFEoUw1UiM9SIOwvpc):

| Folder | Contents |
| --- | --- |
| `Data/` | Part 1 course inputs: the document PDFs, `Questions.csv`, `sample_submission.csv`, `validate_submission.py` |
| `outputs/` | Files written by the notebook runs |

The PDFs and other course data aren't committed to GitHub, as the course rules require.
