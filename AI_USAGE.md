# AI Usage Disclosure

## Final disclosure summary — 2026-11-16 PT

Yuyao used ChatGPT/Codex for coding, debugging, test preparation, documentation
and report formatting. Exact assistant model versions were not consistently
recorded. The detailed log below preserves the work and checks at each stage.

- **Part 1 human reviews:** Codex prepared review forms, checked the supplied
  ratings and calculated groundedness, correctness and citation-support results.
  The members supplied the ratings. Pratiksha's misfilled reviewer name was
  corrected after Yuyao confirmed her identity; one correctness label was changed
  only after Yuyao explicitly approved it. Original files were retained.
- **Part 3 human reviews:** Codex prepared review forms and combined the supplied
  independent member reviews and joint adjudication to calculate the results.
  AI did not fill in either member's ratings or decide disagreements for them.
- **Report and documentation:** Codex checked the teammate-written report against
  the assignment, suggested wording and table corrections, and converted the
  user-selected final document to IEEE-style PDF layouts, including a single-column
  version. It also helped with READMEs, the system card and `links.txt`. This
  assistance did not produce new experiments or human scores.
- **Current status:** Part 1 groundedness/content reviews and Part 3 member reviews
  and adjudication are recorded in `eval/yuyao_ding/part1/results/` and
  `eval/yuyao_ding/part3/results/`. Yuyao's Part 2 baseline ran on Colab T4, and
  Yuyao confirmed the final Part 1 Kaggle submission. Earlier “pending” statements
  below describe the status when those entries were written.
- **Teammate disclosure:** Pratiksha still needs to confirm the tools, tasks and
  checks for her own AI use. Yuyao's entries do not stand in for her disclosure.

## Part 1 Kaggle runtime and evidence import — 2026-11-16 PT

Codex prepared the Kaggle-only entry, dependency constraints, single-model load
guard, explicit generation settings and efficient-attention compatibility code.
An AI-generated `patch.object(..., return_value=False)` override retained GPU
tensor call arguments and caused repeated OOM failures. Codex diagnosed this
Python reference-lifetime defect and replaced the Mock with an ordinary function;
CPU weak-reference tests reproduce the old retention and verify the correction.
These CPU tests are distinct from the subsequent GPU execution.

Yuyao executed the corrected notebook on a real single Kaggle Tesla T4, supplied
the executed notebook and results ZIP, and confirmed the CSV was submitted and
selected as final. Codex checked original question-file/order hashes, all 40
submission and diagnostic rows, audit consistency, recorded official validation,
generation token totals and the exact uploaded live generation function. The
archive retains one earlier failed preflight before the successful probe and
competition calls. No score improvement or new human judgment is inferred.

The public notebook retains recorded numerical outputs while omitting course-text
previews and interactive widgets. The already executed live generation function
is inlined without changing its AST. Source-tag bootstrap and entry-name edits
made after execution are identified separately; they were not rerun on GPU.
Original notebooks/ZIPs remain private with recorded hashes. Existing Colab
experiments, human labels, teammate files, models and images were not replaced.
Affected files: `notebooks/Yuyao_Ding/p1_kaggle_submission.ipynb`,
`eval/yuyao_ding/part1/results/kaggle_20261117T003803178245Z/`, the Part 1 import
record, CPU verifier/regression tests and related README/release links.

## Historical usage log

These entries retain the status and file paths recorded at the time. See the
summary above and later entries for completed follow-up work.

| Date | Tool/model | Purpose | Human review and changes | Affected files |
| --- | --- | --- | --- | --- |
| 2026-11-13 | Codex; underlying model ID not exposed in this session | Recovered the private original Part 1 archive after the user signed in; implemented reproducible baseline/final evaluation, authored an AI-assisted 25-case draft from the course PDFs, added Windows cache/capacity compatibility changes and source-content review tooling | Automated tests and actual RTX4090 execution are recorded separately. Team authorship, originality, source-reference review and both independent human groundedness ratings are pending; no student review or approval is claimed | `src/yuyao_ding/p1_evaluation.py`, `tests/yuyao_ding/test_p1_evaluation.py`, Part 1 notebook, `eval/p1_golden_v2*`, Part 1 protocol/Windows documents and runner |
| 2026-11-14 | Codex; model ID not exposed | Part 1 second Colab development revision: CLIP return-type compatibility guidance, generic JSON container normalization, bounded repair instructions, captured-callback rebinding, source-hash/hardware provenance and first/second evaluation comparison | User executed the new script on Tesla T4. Raw runs and notebooks retained; CPU contracts and comparison execution recorded. Content/citation and both independent human ratings remain pending; format improvement is not claimed as answer correctness or Kaggle improvement | scripts/yuyao_ding/p1_colab_schema_fix_and_rerun.py, Part 1 notebooks, round1/round2 output summaries and protocol |
| 2026-11-13 | Codex; model ID not exposed | Part 3 development revision: typed state/events, effective prompt compilation, separate quality/safety critique, CPU CLIP similarity, photometric enhancement, regression tests and versioned real-GPU rerun | Existing private inputs and fixed cases retained; independent diagnostics and GPU traces recorded. Two-member quality/safety judgments and report authorship remain pending. No human labels fabricated | `src/yuyao_ding/p3_*`, `tests/test_p3_*`, `eval/yuyao_ding/validate_p3_revision.py`, Part 3 notebook and revision documents |
| 2026-11-13 | Codex; model ID not exposed | Follow-up Part 3 development fixes: bounded Qwen caption rewriting, approved-medium cues, singular-count compatibility, monochrome enforcement, per-revision Prompt events, blank local independent review forms and executed notebook packaging | Automated contracts and fresh real GPU execution are recorded. Earlier failed runs retained. No member reviews, authorship, rubric scores or approval claimed | Part 3 source/tests, notebook, review helper, runbook and AI-assisted system-card draft |
| 2026-11-14 | Codex; model ID not exposed | Fixed Part 3 approved body-part count preservation and T4 hardware-status reporting; checked the user-provided executed Colab notebook and private run archive; removed temporary setup cells and stale Windows demo instructions from the primary notebook | User executed the unchanged 15×2+10 cases on Tesla T4. The archived run and frozen hashes were checked, the report was recalculated from saved records with no numerical change, 41 Colab tests passed, and 42 local CPU contracts passed. Human judgments and final report authorship remain pending | `src/yuyao_ding/p3_prompt_rewrite.py`, `src/yuyao_ding/p3_reporting.py`, Part 3 tests/notebook, `eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md`, Part 3 documentation |

Additional team disclosures, including any Part 2 tool use, require confirmation from the responsible member before final submission.

## Part 2 partner baseline preparation — 2026-11-14

Codex prepared Yuyao Ding's separate working-baseline notebook and helper code,
adapting Pratiksha Kaushik's existing data/prompt/sampler protocol and the official
Apache-2.0 Diffusers v0.42.1 LoRA example. The official trainer is pinned by commit
and inspected source hash. The work includes one-GPU guards, dependency constraints
that preserve CUDA/PyTorch, private artifacts, checkpoint restart, paired metrics,
and run provenance. Existing team experiments and weights were not overwritten.
Only CPU contract/static checks were run here. The new baseline's real T4/P100
training, generation, measured results, Yuyao's own interpretation and meeting
recording are pending. No GPU result, human judgment or meeting record was created.
Affected files: `notebooks/Yuyao_Ding/p2_baseline.ipynb`,
`src/yuyao_ding/p2_partner_baseline.py`, `scripts/yuyao_ding/p2_train_worker.py`,
`tests/yuyao_ding/test_p2_partner_baseline.py`, `config/yuyao_ding/requirements-part2-baseline.txt`,
`eval/yuyao_ding/p2_yuyao_baseline_prompts.json`, `eval/yuyao_ding/PART2_YUYAO_BASELINE.md`.

Part 2 follow-up: the user's first Colab T4 training attempt failed in Accelerator
initialization because the CLI string `none` was treated as a tracker name; its
reported allocated/reserved GPU peaks were zero. Codex added argument
normalization to Python `None` in the worker, preserving the pinned trainer, and
a narrowly checked recovery that archives the failed attempt and records the
source transition without discarding prepared data. CPU regression tests cover
this boundary. No successful GPU training or new model result is claimed yet.


Part 2 executed evidence import (2026-11-14 PT; run ID `20261115T034917Z` UTC):
the user supplied the executed baseline notebook and a private 80-image/adapter
bundle after actual Colab T4 execution. Codex checked archive integrity, all image
hashes and dimensions, adapter hash/tensor-header parameter count, source/prompt
hashes, checkpoint-600 log evidence, split isolation and saved aggregate metrics.
Original files remain private and unchanged. The canonical notebook retains the
original code and numeric outputs, omitting the clone/recovery setup cells,
interactive widgets and three embedded visual previews. Metadata discloses the
interactive recovery and both source commits; no uninterrupted run is claimed.
No new training, generation, human rating or metric-model evaluation was performed.
The measured result record and contributions now reflect actual execution; Yuyao's
own interpretation, real meeting record and the team's human-preference evaluation
were not created by the import.


Part 2 result interpretation supplement (2026-11-14 PT): Codex inspected selected
private generated images and drafted the metric tradeoffs, count/hat-subtype
limitations and safety/memorization caveats in the baseline notebook and results
`ANALYSIS.md`. One separate CPU notebook cell computed differences from saved
numeric artifacts; the ten original T4 code cells and their outputs were retained.
No training, generation, metric-model inference or human scoring was repeated.
The draft is explicitly AI-assisted; Yuyao's verification and rewording, the real
meeting record and any independent human-preference evaluation remain separate.


Part 1 source-content review preparation (2026-11-15 PT): Codex added a local
HTML exporter for the saved second-round baseline/final predictions. It displays
frozen references and original private PDF pages, collects answer correctness and
per-citation support, and binds exports to the recorded evidence hashes. Synthetic
contract tests check identity and label validation. No human labels, new inference
or changes to the completed two-member groundedness scores were made.

Part 1 content-review integration (2026-11-15 PT): Codex validated the supplied
Yuyao Ding correctness and per-citation judgments against saved prediction,
specification and run hashes. Baseline P1V2-11 was corrected only after explicit
user confirmation; original bytes and the correction authority are preserved.
The older token-only numeric check is retained separately because final P1V2-09
expresses a decrease of 1.6 as -1.6. Human source judgments apply the unchanged
frozen numerical/content rubric; no inference rule or reference was changed.
Existing independent groundedness scores remain unchanged. File validation does
not establish how the human review was conducted. CPU notebook aggregation and
its source provenance are distinct from the original Colab T4 inference.


Evaluation-directory organization (2026-11-15 PT): Codex moved Yuyao's evaluation
assets under `eval/yuyao_ding/`, updated live paths/relative links and preserved
frozen JSON specifications, CSV/JSON results, original ratings and notebook
outputs. Notebook metadata and the relocation manifest disclose source-path
edits made after execution, with original/source/output hashes. No GPU experiment
or human scoring was repeated. Pratiksha's evaluation assets were not moved.


Member-workspace organization (2026-11-15 PT): Codex separated reusable code,
runners, tests, configuration, provenance and traces by member and normalized the
companion evaluation directory name. It updated current imports/entry points and
added historical-path resolution for archived frozen-file hash verification.
Original JSON/CSV records, environment snapshots, bundles and notebook outputs
remain unchanged; reversible source-reference edits and their hashes are recorded
in `audit/yuyao_ding/workspace_relocation.json`. Empty companion folders are
explicitly placeholders, not additional contributions. No model training,
inference, human rating or CUDA/PyTorch installation was performed.


Part 3 CPU result-table supplement (2026-11-15 PT): Codex appended and actually
executed one self-contained CPU aggregation cell in the main Part 3 notebook,
using the already supplied reviewed records and adjudication. It verifies review
file hashes, recomputes each mode with the existing summarizer, checks the stored
metrics, and exports a report table/CSV/JSON with provenance. Original labels,
raw result JSON and the original T4 cells/outputs remain unchanged. Append-only
snapshot verification lets earlier migration checks still validate the preserved
GPU notebook prefix. No model loading, training or GPU inference occurred. The
paired-image/no-revision limitation remains explicit; no improved quality-loop
result or general safety claim was introduced.


Hugging Face artifact publication (2026-11-15 PT): Codex prepared generation-only/adapter allowlists, model and dataset cards, metadata and SHA-256 manifests for the existing Yuyao Part 2 baseline and actual Part 3 T4 run. Yuyao uploaded the three prepared directories through the Hugging Face website. Codex verified all 117 source artifact files against remote file sizes and LFS SHA-256 or Git blob IDs, then added root cards through the website editor to document the uploaded nested layout and configure image-folder discovery. Public artifact links and pinned publication revisions are recorded in the repository. Original weights, image bytes, execution records, prompts and human labels remain unchanged. No model training, generation, human rating or GPU validation was repeated. Private input fixtures, the adult portrait, raw training/test pixels, credentials, caches and full raw traces were excluded. Dataset Viewer indexing was still pending at verification; no working viewer or new experimental improvement is claimed.


Part 3 system-card finalization (2026-11-15 PT): At Yuyao's request, Codex rewrote the existing draft in plain student-style English, finalized it as `eval/yuyao_ding/PART3_SYSTEM_CARD.md`, and updated live documentation links. Architecture and guardrails were checked against current source; model revisions, T4 provenance and numerical results were checked against the saved validator and human-review metrics. The card distinguishes the actual T4 run from earlier RTX 4090 development, explains that the critic made no revisions, and preserves the identical-image and differing-human-label limitations. It links the completed reviews and HF outputs without claiming that the combined IEEE report or other submission items are finished. No experiment, human judgment or authorship confirmation was created. Original draft history and frozen provenance manifests remain unchanged in Git history.
