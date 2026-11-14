# Part 1 round 2: final-system human groundedness

These scores apply only to the 25 final-system answers from the completed second
Colab Tesla T4 run. They are a CPU aggregation of supplied reviews; no model was
loaded and no inference was rerun.

| Measurement | Result |
| --- | ---: |
| Questions per reviewer | 25 |
| Yuyao Ding mean | 4.44 / 5 |
| Pratiksha Kaushik mean | 4.44 / 5 |
| Mean across all 50 ratings | 4.44 / 5 |
| Pairs within one point (required agreement measure) | 25/25 (100%) |
| Exactly matching pairs (additional measure) | 23/25 (92%) |

The only score differences are P1V2-06 (4 versus 3) and P1V2-10 (2 versus 3).
Both are within one point. The two original ratings are retained; there is no
pass/fail adjudication step for this groundedness mean.

## Files and identity correction

- `part1-member-A-ratings.json` and `part1-member-B-ratings.json` are byte-identical
  copies of the supplied files.
- Both originals name Yuyao Ding and have member label A. The user explicitly
  confirmed that B was independently rated by Pratiksha Kaushik and its name was
  entered incorrectly. That correction is applied only to derived records.
- `provenance.json` records the correction authority and original file hashes.
- `ratings_and_metrics.json` retains each original score and the computed metrics,
  identifying the confirmed reviewers.

The review source was a projection of saved round-2 generated answers, linked to
the original run hash recorded in `summary.json`, not a restored full original
run file. All 25 projected answers and top-five rankings were checked against
the saved `golden_predictions.json`. Submitted files declare independent human
review; file validation does not independently establish the review process.

## Limits

Groundedness is a 1–5 support judgment, not answer correctness or citation
accuracy. Human correctness and per-citation support still need separate review.
No score is assigned to dense-only, round 1, or the earlier Windows run from these
files. Original experiment summaries and executed outputs remain intact.

## Recompute without GPU

From the repository root:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean
root = Path('eval/part1/results/round2_colab_20261114T032206Z/human_review_20261114')
a, b = [json.loads((root / f'part1-member-{m}-ratings.json').read_text()) for m in ('A', 'B')]
assert a['run_sha256'] == b['run_sha256']
assert [r['question_id'] for r in a['records']] == [r['question_id'] for r in b['records']]
pairs = [(x['groundedness'], y['groundedness']) for x, y in zip(a['records'], b['records'])]
assert len(pairs) == 25
print('Groundedness mean:', mean(v for pair in pairs for v in pair))
print('Agreement within one:', mean(abs(x-y) <= 1 for x, y in pairs))
print('Exact agreement:', mean(x == y for x, y in pairs))
PY
```
