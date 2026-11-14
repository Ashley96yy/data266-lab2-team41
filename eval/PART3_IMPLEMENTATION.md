# Part 3 implementation and existing-environment runbook

Use the current working Windows environment. Do not reinstall the root
requirements or replace CUDA/PyTorch. The fixed private input bundle stays under
ignored data/part3; generated images, model caches and raw traces stay under
ignored .runtime. Do not publish the consented portrait.

## Five stages

- Planner validates input and chooses a supported route, clarification or refusal.
- Vision returns caption, objects, visible text and at least two question answers.
- Prompt Engineering uses sanitized Vision context and a bounded Qwen caption
  rewrite. Approved request colors, explicit counts and the actual CLIP text
  budget are validated. Medium cues come from the approved request/route only.
- Generation uses real SD1.5 img2img; domain style loads the pinned Part2 LoRA at
  the recorded weight and disables it for other routes. Enhancement uses Pillow
  photometric operations on the original geometry, with zero diffusion steps.
  Explicit monochrome conversion is recorded and safety-checked again.
- Critique records real CPU CLIP cosine similarity and separate VLM quality and
  safety reviews. Quality defects trigger concrete feedback, never an invented
  unsafe-content category. The prompt stage is reentered for at most two revisions.

A shared dataclass carries state through all stages. Timed inputs, outputs, raw
format failures, image paths and every bounded revision are written to JSON.
Each attempt starts from the original source. Safety remains enabled with and
without the quality loop. Unresolved quality outcomes remain revision_limit.

## Run and validate

From the repository root, with the existing .venv selected:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_p3*.py'
.\.venv\Scripts\python.exe eval/verify_p3_fixture_manifest.py --require-complete
.\.venv\Scripts\python.exe -m src.p3_app
```

The app listens on http://127.0.0.1:7860, with an English interface and no public
share link. Requests are serialized. Stop the app before the complete GPU notebook
run to avoid competing resident models. Open notebooks/Yuyao_Ding/p3_agents.ipynb
and Restart & Run All. Default execution creates new real GPU diagnostics and the
unchanged 15x2+10 suite, then computes every report number from saved records.
Replay is only enabled by an explicit DATA266_P3_REPLAY_DIR and is printed as replay.

The preferred standalone full validator is eval/validate_p3_revision.py. The older
src.p3_evaluation CLI uses the same Prompt Agent but runs one specified condition;
use new output paths and preserve earlier runs. Model revisions are pinned in the
validator and .runtime/model-revisions.json for local app startup.

## Independent human review

```powershell
.\.venv\Scripts\python.exe scripts/p3_review_forms.py export --run-dir RUN_DIR --output NEW_LOCAL_REVIEW_DIR
.\.venv\Scripts\python.exe scripts/p3_review_forms.py import --run-dir RUN_DIR --member-a MEMBER_A_JSON --member-b MEMBER_B_JSON --output NEW_REVIEWED_RUN_DIR
```

Both members independently fill the blank English forms. Critic predictions are
omitted. The importer validates raw run hashes, two different named reviewers,
case identities and generated-attempt identities. Disagreements stay blank in new
copies and require documented adjudication. Raw inference files remain unchanged.
Point DATA266_P3_REPLAY_DIR at the reviewed directory to compute final metrics in
notebook outputs. A completed execution or CLIP score is never a human pass label.

## Evidence boundaries

CPU tests use synthetic fixtures only for contracts, not experimental judgments.
Real GPU records include model revisions, source hashes, seed, prompt, scheduler
steps, effective img2img strength, applied steps, LoRA weight and safety results.
The unchanged cases were used during development; disclose this rather than
claiming an untouched holdout. Human metrics remain pending until real reviews.

A completed Colab Tesla T4 execution from source commit `53c92a3` is recorded in
`notebooks/Yuyao_Ding/p3_agents.ipynb` and `eval/PART3_T4_MEASURED_RESULTS.md`.
The 15 paired scenarios and 10 red-team cases had zero execution errors. The
earlier 14 GiB allocator budget on RTX 4090 remains separate historical evidence.
Both members' independent reviews, recordings, their own IEEE report/system card,
approved artifact hosting and the final tag remain required.
