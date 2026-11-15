# Part 1 round 2: notebook-computed development comparison

Analysis commit: `9caac5f34d2b843c5aaf5968eccd379ac2074c40`. Source GPU runs used one Tesla T4; analysis here is CPU/file-backed. Original inference Git commit was unavailable.

| Cohort | Round 1 fallbacks | Round 2 fallbacks | Fewer fallback percentage points |
| --- | ---: | ---: | ---: |
| practice | 4 | 0 | 10.0 |
| dense_only | 4 | 2 | 8.0 |
| final | 6 | 1 | 20.0 |

Within-round second-run required comparison:

| Metric | Dense-only baseline | Final system |
| --- | ---: | ---: |
| Questions | 25 | 25 |
| Recall@5 | 0.4722 | 0.7639 |
| Recall denominator | 24 | 24 |
| Median seconds/request | 9.1736 | 0.7958 |
| Answer correctness | Pending | Pending |
| Citation accuracy | Pending | Pending |
| Groundedness mean | Pending | Pending |
| Human agreement within one point | Pending | Pending |

This documents a measured format-stability improvement. It does not establish better answer correctness, citation accuracy or Kaggle score. Recall is unchanged between rounds. Final-system latency increased between rounds; fewer fallbacks may allow longer answers, but causal attribution is not established. Short final-system medians include deterministic structured routes, not just VLM inference.

Known content failure: P1V2-22 reverses the correction scope. DOC3:p6 corrects the plotting error for 2012; other values remain the same. Both members must independently rate groundedness, and content/citation judgments remain pending for both systems.

Fixed questions, rubrics, model revisions and seed were retained. These are development reruns after inspecting failures; do not claim an untouched holdout. The source repository is currently public; course privacy, final Git provenance, report, human review, final tag and Kaggle selection remain separate completion items.
