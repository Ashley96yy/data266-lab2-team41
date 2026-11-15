# Yuyao Ding — evaluation workspace

Evaluation specifications, runbooks and evidence for Yuyao's main Part 1 and
Part 3 systems, plus the separate Part 2 partner baseline. See the
[notebook workspace](../../notebooks/Yuyao_Ding/README.md) for implementations
and roles, and the [shared evaluation index](../README.md) for other contributors.

| Part | Specifications and documentation | Recorded results |
| --- | --- | --- |
| 1. Multimodal RAG | [Frozen version-2 golden set](p1_golden_v2.json), [evaluation protocol](PART1_EVALUATION_PROTOCOL.md), [architecture](PART1_ARCHITECTURE.md), [safety analysis](PART1_SAFETY_ANALYSIS.md) | [Both original Colab rounds](part1/results/), [completed source-content review](part1/results/round2_colab_20261114T032206Z/content_review_20261115/RESULTS.md), [two-member groundedness](part1/results/round2_colab_20261114T032206Z/human_review_20261114/) |
| 2. Partner baseline | [Frozen prompts](p2_yuyao_baseline_prompts.json), [baseline runbook](PART2_YUYAO_BASELINE.md) | [Actual T4 baseline, metrics and provenance](part2/results/yuyao_t4_20261115T034917Z/), [AI-assisted interpretation](part2/results/yuyao_t4_20261115T034917Z/ANALYSIS.md) |
| 3. Multi-agent system | [15 scenarios](p3_scenarios.json), [10 red-team cases](p3_red_team.json), [fixture manifest](p3_fixture_manifest.json), [evaluation plan](PART3.md), [implementation](PART3_IMPLEMENTATION.md) | [T4 measured report](PART3_T4_MEASURED_RESULTS.md), [reviews and adjudication](part3/results/t4_20261114T060249Z/), [redacted request traces](../../traces/t4_20261114T060249Z/) |

## Layout

```text
eval/yuyao_ding/
├── README.md
├── PART1_*.md, p1_golden_*, p1_results_table.*
├── PART2_YUYAO_BASELINE.md, p2_yuyao_baseline_prompts.json
├── PART3*.md, p3_*.json
├── part1/results/             # original rounds and later human reviews
├── part2/results/             # partner baseline records
├── part3/results/             # main-system human reviews and aggregation
├── validate_p3_revision.py
├── verify_p3_fixture_manifest.py
├── build_p3_red_team_inputs.py
├── LEGACY_GOLDEN_README.md    # earlier evaluation index, retained as history
└── relocation_manifest.json  # old/new paths, hashes and notebook path edits
```

The original `p1_golden_questions.csv` and `p1_golden_answers.csv` remain historical
fixtures. Their contents and checksums are unchanged. Use the version-2 protocol
and the reviewed second Colab round for the current Part 1 result table. Golden
references must never enter the inference/indexing pipeline.

The main Part 3 validator now runs from the repository root as:

```bash
python eval/yuyao_ding/verify_p3_fixture_manifest.py --require-complete
python eval/yuyao_ding/validate_p3_revision.py --output-root YOUR_NEW_RUN_DIRECTORY
```

These are explicit validation commands, not a request to repeat completed GPU
experiments. The first checks private input bytes; the second starts new GPU
inference. Inspect saved results directly for documentation or human aggregation.
Private fixtures, course PDFs, portraits, image outputs and weights stay outside
GitHub.

## Directory migration and preserved evidence

The directory split made no changes to frozen JSON specifications, CSV results,
original review files, recorded JSON metrics or model outputs. Documentation links,
current executable defaults and notebook source-path references were relocated.
The notebooks retain their original stored outputs and execution counts, with
metadata explicitly identifying path edits made after execution. They were not
rerun on GPU for this move. Earlier untouched notebook snapshots remain in
[the notebook history directory](../../notebooks/Yuyao_Ding/history/).

Paths, source hashes and notebook hashes inside original experiment/provenance
JSON files remain historical. They refer to the source used at that recorded run;
a directory move does not update the original inference commit. Resolve those
old paths using [relocation_manifest.json](relocation_manifest.json). In particular,
recorded source hashes should not be asserted to match newly relocated source.
The separate migration record verifies the byte-preserved inputs/results and
records each changed notebook code cell alongside its original output hash.
