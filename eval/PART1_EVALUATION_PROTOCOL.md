# Part 1 completion protocol, version 2

The original `p1_golden_questions.csv`, `p1_golden_answers.csv` and their earlier
baseline record are historical and must remain unchanged. Version 2 is a new,
AI-assisted source-checked draft with the exact 5/6/3/5/3/3 coverage required by
Task 1.4, including five turns in at least two conversations. The team must
review authorship, originality and the reference answers. It is not described
as human-authored or as an untouched test of the entire development history.

Freeze the protocol and candidate before writing the version 2 cases, and
commit the cases before running them. Do not tune retrieval, prompts, rules,
references or the scoring rubric in response to the resulting scores. A
compatibility fix must be documented and the affected runs repeated under its
own source revision. Preserve failed attempts rather than relabeling them.

## Baseline and final system

Both use the same read-only corpus, Qwen version/decoding, conversation state,
JSON answer validation and Appendix C exclusion. The baseline uses only dense
BGE text retrieval and dense CLIP visual retrieval. It does not use BM25,
exact-label routing or the global structured-fact expansion. The final system
uses the existing hybrid/label/structured pipeline. This is an end-to-end
baseline comparison, not an isolated BM25 ablation.

Reference answers and rubrics are stripped from the immutable question objects
passed to inference. Corpus indexing must never import the golden references.
The public practice questions remain a separate, configurable `QUESTIONS_PATH`.
The final Kaggle-format run must not assume 40 rows or 35 conversations because
graders replace the question file.

## Measurements

- Correctness: content review plus correct status and all frozen numeric targets
  within 1% relative tolerance (zero uses only numerical precision tolerance).
  Record who reviewed content and whether the review was human or AI-assisted.
  Missing reviews remain null, never zero or a fabricated pass.
- Recall@5: interleave the original retrieval rankings T1,V1,T2,V2,... and
  deduplicate by evidence unit. Use only the first five units, before structured
  answer expansion. Text units are pages; visual units are exact labels. Each
  case lists required source groups with accepted alternatives; compute the
  fraction of groups retrieved, then the mean over evidence-bearing cases.
  Explicitly report that denominator and exclude no-evidence questions.
- Citation accuracy: review each emitted citation against the claim and actual
  source content. Matching a reference citation is not sufficient. Compute the
  supported-citation fraction over all emitted citations. Missing support
  judgments remain null. Preserve reviewer/method provenance.
- Human evaluation: both team members independently rate all 25 FINAL answers
  for groundedness from 1 to 5. Compute the average and fraction of pairs within
  one point. AI-assisted content review cannot fill either member's ratings.
- Efficiency: median end-to-end seconds per question after shared model/index
  setup, including all 25 questions and all retries. Record CPU/GPU placement,
  seed 266, package versions, resolved model revisions and source provenance.

## Runtime and remaining obligations

Reuse the existing Windows CUDA/PyTorch installation. Retrieval defaults to CPU
and Qwen to CUDA device 0, so the pipeline uses only one GPU. A run on an RTX
4090 must not be described as an actual 16 GB T4/P100 hardware run. Keep inputs
and runtime outputs private. Do not submit to Kaggle, change a published tag or
claim leaderboard selection without observing the respective action.

## Recorded amendment after the initial run

The initial complete candidate was retained. Source inspection of its outputs
revealed that plural figure lists did not take the required exact-label route.
A subsequent general routing correction was made without changing questions
or rubrics, alongside practice-driven arithmetic fixes. The revised paired
run is labeled a development-set evaluation. It must not be presented as an
independent holdout under the original no-feedback protocol. See the dated
compatibility log and commit history; the team must review this limitation.
