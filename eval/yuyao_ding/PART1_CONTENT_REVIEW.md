# Part 1: source-content scoring of the saved second Colab round

The two independent final-system groundedness ratings are already recorded under
`part1/results/round2_colab_20261114T032206Z/human_review_20261114/`.
Do not repeat them or infer binary correctness/support from their 1–5 values.

The missing measurements are answer correctness (25 baseline + 25 final answers)
and citation accuracy (75 baseline + 47 final emitted page citations). The local
form displays the frozen references, numeric targets, earlier conversation turns
and relevant original PDF pages. A named source-content reviewer supplies these
labels. No model runs, no judgments are prefilled, and original predictions and
independent groundedness ratings are never overwritten.

Generate the private form from the repository root using an environment with
PyMuPDF installed:

```bash
python scripts/yuyao_ding/p1_content_review_forms.py \
  --input-dir /absolute/path/to/the/three/course/PDFs \
  --output .runtime/part1-content-review-new
```

Open `content_review.html` in that output folder. Enter your actual name, judge
each complete answer and each emitted citation, then export
`part1-content-review.json`. Browser drafts stay local when supported; a draft
backup is not a completed submission. Keep original course page images private.

Correctness follows the frozen content rubric: required facts, units, numeric
targets within ±1%, and the correct status, without contradictory central claims.
Citation support is a separate decision for each emitted source, not overlap
with reference citations. Unsupported/extra citations reduce citation accuracy
even if the central answer is right. Absent citations do not automatically make
an answer wrong; assess the question and expected behavior.

On import, verify prediction, summary and specification hashes, question order,
run identities and unchanged displayed evidence with `validate_submission` in the
exporter. The run identity comes from `summary.json`; displayed records are a
projection of saved predictions, not byte-identical restored original run files.
Then compute the baseline/final metrics in notebook outputs, combining the
existing final groundedness ratings without changing them. Do not publish blank
forms as completed human results. A content assessment made by AI must be labeled
AI-assisted rather than passed through this human-review declaration.

## Completed review import (2026-11-15 PT)

[The supplied review, confirmed correction and computed table](part1/results/round2_colab_20261114T032206Z/content_review_20261115/)
now complete these measurements. Recompute with:

```bash
python scripts/yuyao_ding/finalize_p1_content_review.py \
  --review eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/content_review_20261115/part1-content-review.original.json \
  --corrections eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/content_review_20261115/corrections.json \
  --output .runtime/part1-content-recomputed
```

The supplied human source review assesses numerical meaning, units and direction
against the unchanged frozen reference/±1% rubric. The older token-presence helper
is retained as a diagnostic, not used to overturn a human-reviewed signed decrease:
P1V2-09 correctly expresses Germany's 1.6-point decline as a change of -1.6.
The frozen positive target token 1.6 therefore fails that helper's literal sign
match. No reference value, question, generated answer or inference rule changed.
Both the human result and this diagnostic discrepancy are disclosed in the audit.
