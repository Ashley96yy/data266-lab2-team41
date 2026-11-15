# Part 3: supplied human reviews and adjudication

Source inference: completed Colab Tesla T4 run `20261114T060249Z`, commit
`53c92a3351d8c9bc9d4c33b4d42ab43df9057753`, seed 266. Aggregation uses saved
files on CPU; no models are loaded and no inference is rerun.

- `part3-member-A-review.json`: original submitted Yuyao Ding ratings.
- `part3-member-B-review.json`: original submitted Pratiksha Kaushik ratings.
- `part3-adjudication.json`: supplied joint decisions and reasons for all 10 disagreements.
- `with.reviewed.json`, `without.reviewed.json`, `red_team.reviewed.json`:
  privacy-reduced scoring inputs retaining original judgments and final decisions.
- `metrics.json`: computed counts, rates, timing, and paired-image hashes.
- `provenance.json`: input hashes and disclosure of the B file's member-label metadata.

The submitted files declare independent human review and joint adjudication.
The import validates their structure and provenance, not how the reviewers worked.
No human label or resolution reason was authored or changed by the import process.
Private images, raw traces, portrait inputs, and local paths are excluded.

| Measurement | With quality loop | Without quality loop | Red team |
| --- | ---: | ---: | ---: |
| Final rubric passes | 10/15 | 11/15 | 10/10 |
| Task success rate | 66.67% | 73.33% | N/A |
| Attack success rate | N/A | N/A | 0/10 (0%) |
| Original reviewer case agreement | 13/15 (86.67%) | 12/15 (80.00%) | 9/10 (90.00%) |
| Critic/human agreement | 6/9 (66.67%) | 5/9 (55.56%) | 1/1 (100%) |
| Median seconds/request | 17.2280 | 16.7367 | 0.0002 |

There were 6 case-level and 4 generated-attempt disagreements. All have supplied
joint resolutions. Original case agreement remains 34/40 (85%); adjudication
must not turn that into 100% reviewer agreement. Original image-acceptance
agreement is 15/19 (78.95%). Critic agreement compares all recorded critic
decisions against final human image decisions, including the diagnostic critic
in the no-quality-loop condition; it does not substitute whole-case pass/fail.

## Interpretation limits

All 9 paired generated images are byte-identical between conditions. Each critic
accepted the first generated attempt, so no quality-driven revision was exercised.
Final case judgments differ across modes for P3-S05, P3-S11 and P3-S12 despite
identical paired image bytes. These submitted labels are preserved, but their
score difference is not evidence of a causal benefit or harm from the quality loop.
Some whole-case and image-acceptance labels also differ (for example, the with-loop
P3-S05 case fails while its image is accepted); these are distinct supplied ratings,
not silently reconciled by code. Some supplied resolution notes are brief.

The 0/10 attack-success result applies only to the frozen small development suite,
not general safety. RT03's supplied case resolution says the embedded command was
not copied as an instruction. Failed artistic transformations must not automatically
be treated as successful prompt injection. Human scores do not remove known critic
false accepts or the development-set limitation.

## Recompute without GPU

From the repository root:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.p3_evaluation import summarize
root = Path('eval/yuyao_ding/part3/results/t4_20261114T060249Z')
for mode in ('with', 'without', 'red_team'):
    run = json.loads((root / f'{mode}.reviewed.json').read_text())
    print(mode, json.dumps(summarize(run), indent=2))
PY
```

The complete private import can be reproduced with `scripts/finalize_p3_reviews.py`:
pass `--run-dir`, `--member-a`, `--member-b`, `--adjudication`, and fresh
`--private-output` / `--public-output` directories. Original inference records and
original reviewer files are never overwritten.
