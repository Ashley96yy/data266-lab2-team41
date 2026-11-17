# Part 1: completed Kaggle competition run

Run ID: `20261117T003803178245Z` UTC; local date: November 16, 2026 PT.
The actual run used one Tesla T4 (`cuda:0`), CPU retrieval and seed 266.
Algorithm-base commit: `df1867f9f85078e738a6211e06c85b19e0c904db`.

[Executed submission notebook](../../../../../notebooks/Yuyao_Ding/p1_kaggle_submission.ipynb)
and [Kaggle page](https://www.kaggle.com/code/yuyaoding/part1-rag).
Yuyao confirmed that this CSV was submitted and selected as the final result.
This confirmation is user-supplied; no new leaderboard score or independent
inspection of the Kaggle selection is claimed.

## Recorded result

These values come from the notebook-generated audit and provenance JSON.

| Check or measurement | Recorded value |
| --- | ---: |
| Submission / diagnostic rows | 40 / 40 |
| Conversations | 35 |
| Completed competition rows with errors | 0 |
| All reported audit checks | passed |
| Official validator | `valid: 40 rows`, return code 0 |
| Answer repairs / safety fallbacks | 0 / 0 |
| Median seconds per question | 5.768 |
| Peak allocated / reserved GPU memory | 7.858869 / 7.931641 GiB |
| Elapsed seconds since setup | 1058.914 |
| Within 90 minutes | yes |
| Input / generated tokens | 96738 / 2532 |
| Rows with citations | 35/40 (87.5%) |
| Prohibited DOC3:p7 / Figure C-1 references | 0 / 0 |

Statuses: 30 `answered`, 3 `not_found`, 2 `conflict`, 1 `false_premise`,
4 `session_override`. Citation presence is not citation-support accuracy.
Elapsed time since setup includes interactive recovery/waiting; it is not a
claim of an uninterrupted cold run or model-only inference time.

## Failure and recovery

`generation_diagnostics.jsonl` retains one failed preflight and one successful
preflight before the final competition calls. The final 22 competition model
calls all completed and their token totals match the audit. Structured/session
answers do not require a model call. The failed preflight must not be counted as
an error in the later completed 40-row evaluation or removed from the record.

The runtime fixes pinned the Part 1 Transformers stack while preserving
Kaggle's PyTorch/CUDA, used CPU retrieval and efficient SDPA with explicit KV-head
expansion, guarded repeated model loading, and removed Mock argument recording
from the attention override. The previous Mock retained GPU tensor arguments
throughout generation; the replacement is an ordinary function returning the
same decision. CPU reference-lifetime tests reproduce that defect and check the
replacement without claiming GPU validation on the Mac.

## Files and verification

The eight imported files retain their original bytes; hashes are in
`artifact_hashes.json`. `local_verification.json` records the CPU consistency
checks, including the matching original question-file SHA-256 and row order.
The original ZIP and executed notebook remain unchanged in the ignored local
archive. The [import record](../../../../../audit/yuyao_ding/part1/kaggle_import_20261116.json)
describes their hashes, publication redactions and post-run bootstrap edits.

```bash
python scripts/yuyao_ding/verify_p1_kaggle_results.py \
  eval/yuyao_ding/part1/results/kaggle_20261117T003803178245Z
```

Add `--questions /private/path/Questions.csv` to also check the private input's
hash and identifiers. The verifier does not load models, repeat GPU inference,
run the course validator again, or score factual correctness. The official
validator output was generated on Kaggle and is preserved here.

The public notebook inlines exactly the uploaded live generation function.
Later bootstrap/name maintenance is disclosed separately and was not followed
by another GPU run. The earlier Colab Golden Set and human reviews remain
unchanged; this run does not replace those metrics or demonstrate a new score
improvement. Original course PDFs, input images, caches and full retrieval
contexts are not included in this directory.
