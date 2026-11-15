# Yuyao Ding — runners and review tools

| Tools | Purpose |
| --- | --- |
| `p1_colab_schema_fix_and_rerun.py`, `run_part1_windows.py` | Recorded Part 1 development runners; use runbooks before starting a new experiment |
| `p1_review_forms.py` | Independent final-answer groundedness forms |
| `p1_content_review_forms.py`, `p1_content_review_template.html`, `finalize_p1_content_review.py` | Correctness/citation-support forms and validated aggregation |
| `p2_train_worker.py` | Instrumented worker around the hash-pinned official trainer |
| `p3_review_forms.py`, `finalize_p3_reviews.py` | Independent Part 3 review, adjudication and aggregation |
| `export_p3_public_traces.py` | Privacy-reduced exports of the recorded T4 request traces |

The executable Part 3 fixture verifier and full GPU validator remain with the
[evaluation workspace](../../eval/yuyao_ding/README.md). These runners are separate
from [reusable model/pipeline helpers](../../src/yuyao_ding/README.md). Merely moving
them does not require repeating completed training or inference.
