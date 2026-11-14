# Part 1 golden evaluation set

`p1_golden_questions.csv` contains 25 held-out evaluation turns in 24
conversations. `p1_golden_answers.csv` contains the human-authored reference
answers, required facts, expected status, and evidence provenance.

The notebook must not import either golden file while building indexes,
retrieving evidence, generating answers, or selecting rules. Run the golden
evaluation only after a candidate implementation is fixed. Do not add
question IDs, exact question strings, or reference answers to the RAG logic.

Frozen SHA-256 checksums:

- `p1_golden_questions.csv`: `5d86263ab0125fbab8c724c8acbc876b82f27e2388acec4178f5a6d4881ab493`
- `p1_golden_answers.csv`: `77e78186179adbdf6a83230d8f8be4c19b5adf4bc27c5de8fce6d094d55b6b47`

Safety invariant: no golden answer cites `DOC3:p7` or `Figure C-1`.

## Saved Part 1 results

The two completed Colab evaluations are preserved in
[`part1/results/`](part1/results/):

- [`round1_colab_before_schema_fix/`](part1/results/round1_colab_before_schema_fix/)
- [`round2_colab_20261114T032206Z/`](part1/results/round2_colab_20261114T032206Z/)

These directories contain generated answers, diagnostics, summaries, and
provenance. The saved result files were moved without content changes. The
[comparison notebook](../notebooks/Yuyao_Ding/p1_round2_comparison.ipynb) reads
these paths; no new inference is needed to inspect the completed results.
Human answer, citation-support, and groundedness reviews remain pending.

## Frozen baseline record

The first evaluation of the fixed candidate was run at commit `c78d7f2`
before inspecting any golden answers. The 25-row run completed without
execution errors, and manual adjudication found 20 semantically correct
answers. All 25 rows respected the `DOC3:p7` and `Figure C-1` exclusions.

After this baseline, the questions were inspected to diagnose general system
weaknesses. Therefore, subsequent runs on these same 25 questions are
regression tests, not untouched held-out evaluations. The original question
and answer files and their checksums remain frozen.
