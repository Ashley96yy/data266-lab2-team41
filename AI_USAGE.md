# AI Usage Disclosure

Document all uses of generative AI tools in this project according to the course requirements.

For each use, record:

- the tool and model, when known;
- the task for which it was used;
- how the output was reviewed, tested, or revised by the team; and
- the affected files or project sections.

## Usage log

| Date | Tool/model | Purpose | Human review and changes | Affected files |
| --- | --- | --- | --- | --- |
| 2026-11-13 | Codex; underlying model ID not exposed in this session | Recovered the private original Part 1 archive after the user signed in; implemented reproducible baseline/final evaluation, authored an AI-assisted 25-case draft from the course PDFs, added Windows cache/capacity compatibility changes and source-content review tooling | Automated tests and actual RTX4090 execution are recorded separately. Team authorship, originality, source-reference review and both independent human groundedness ratings are pending; no student review or approval is claimed | `src/p1_evaluation.py`, `tests/test_p1_evaluation.py`, Part 1 notebook, `eval/p1_golden_v2*`, Part 1 protocol/Windows documents and runner |
| 2026-11-14 | Codex; model ID not exposed | Part 1 second Colab development revision: CLIP return-type compatibility guidance, generic JSON container normalization, bounded repair instructions, captured-callback rebinding, source-hash/hardware provenance and first/second evaluation comparison | User executed the new script on Tesla T4. Raw runs and notebooks retained; CPU contracts and comparison execution recorded. Content/citation and both independent human ratings remain pending; format improvement is not claimed as answer correctness or Kaggle improvement | scripts/p1_colab_schema_fix_and_rerun.py, Part 1 notebooks, round1/round2 output summaries and protocol |
| 2026-11-13 | Codex; model ID not exposed | Part 3 development revision: typed state/events, effective prompt compilation, separate quality/safety critique, CPU CLIP similarity, photometric enhancement, regression tests and versioned real-GPU rerun | Existing private inputs and fixed cases retained; independent diagnostics and GPU traces recorded. Two-member quality/safety judgments and report authorship remain pending. No human labels fabricated | `src/p3_*`, `tests/test_p3_*`, `eval/yuyao_ding/validate_p3_revision.py`, Part 3 notebook and revision documents |
| 2026-11-13 | Codex; model ID not exposed | Follow-up Part 3 development fixes: bounded Qwen caption rewriting, approved-medium cues, singular-count compatibility, monochrome enforcement, per-revision Prompt events, blank local independent review forms and executed notebook packaging | Automated contracts and fresh real GPU execution are recorded. Earlier failed runs retained. No member reviews, authorship, rubric scores or approval claimed | Part 3 source/tests, notebook, review helper, runbook and AI-assisted system-card draft |
| 2026-11-14 | Codex; model ID not exposed | Fixed Part 3 approved body-part count preservation and T4 hardware-status reporting; checked the user-provided executed Colab notebook and private run archive; removed temporary setup cells and stale Windows demo instructions from the primary notebook | User executed the unchanged 15×2+10 cases on Tesla T4. The archived run and frozen hashes were checked, the report was recalculated from saved records with no numerical change, 41 Colab tests passed, and 42 local CPU contracts passed. Human judgments and final report authorship remain pending | `src/p3_prompt_rewrite.py`, `src/p3_reporting.py`, Part 3 tests/notebook, `eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md`, Part 3 documentation |

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
`src/p2_partner_baseline.py`, `scripts/p2_train_worker.py`,
`tests/test_p2_partner_baseline.py`, `config/requirements-part2-baseline.txt`,
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
