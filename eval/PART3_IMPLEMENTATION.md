# Part 3 implementation and Windows runbook

## What has been implemented

The original Planner, Prompt Engineering, fixed scenarios, red-team cases and fixture manifest remain intact. Vision retains its existing interface, with optional revision pinning and revision metadata added.

- `p3_generation`: lazy SD 1.5 img2img, explicit Part 2 LoRA activation/deactivation, resolved Hugging Face revisions, seed/settings, and mandatory diffusion safety check. Unsafe or unchecked images are not saved. CLIP context overflow raises an error instead of silently truncating the request.
- `p3_critique`: Qwen compares source and result images with a fixed rubric; strict JSON accept/revise/reject validation. Model text never controls tools or routing.
- `p3_orchestration`: Planner → Vision → Prompt → Generation → Critique, initial attempt plus at most two revisions. Every attempt starts from the original image. Filtered concrete critique feedback is used for revision. Refusals, clarification, invalid inputs and model errors also produce JSON traces.
- `p3_app`: local Gradio app, serialized requests to avoid shared model/LoRA state races, result/status/trace/download outputs.
- `p3_evaluation`: manifest preflight, the unchanged case requests and questions, seed 266, execution records awaiting human review, strict completeness and two-reviewer adjudication checks, task success, attack successes, median latency and critic agreement with an explicit denominator.

## Install and test

Keep the existing Windows CUDA environment. From the repository root:

```powershell
python -m pip install -r requirements-part3.txt
python -m unittest discover -v
python eval/verify_p3_fixture_manifest.py --require-complete
python -m src.p3_app
```

The app listens on localhost and does not create a public share link. Test one **new smoke image/request** first, inspect the output and `.runtime/part3/<run_id>/trace.json`, and confirm the Part 2 LoRA loads before starting the scored suite. Model weights are downloaded on first use. Qwen and diffusion remain resident; actual VRAM use and Windows CUDA compatibility have not been measured here.

The 15/10 frozen inputs are local-only and must be extracted separately. Do not commit them or generated images. Preserve the portrait consent restrictions in `PART3.md`.

## Formal evaluation

Resolve and record fixed Hugging Face commit SHAs for Qwen, the base diffusion model, and the team LoRA. Use the same SHAs for both modes:

```powershell
python -m src.p3_evaluation --output .runtime/eval/with.json --vision-revision QWEN_SHA --base-revision BASE_SHA --lora-revision LORA_SHA
python -m src.p3_evaluation --output .runtime/eval/without.json --without-critique --vision-revision QWEN_SHA --base-revision BASE_SHA --lora-revision LORA_SHA
python -m src.p3_evaluation --spec eval/p3_red_team.json --output .runtime/eval/red_team.json --vision-revision QWEN_SHA --base-revision BASE_SHA --lora-revision LORA_SHA
```

Replace SHA placeholders with actual model commit IDs. Output safety review remains enabled in both modes; `--without-critique` disables quality-based revisions and acceptance gating. Thus this comparison measures the quality revision loop while preserving identical safety review. Record this interpretation in the report.

For every record, each member independently supplies `reviewer_a_pass` and `reviewer_b_pass`; add `adjudicated_pass` and a `resolution_note` for disagreements. Copy the frozen `pass_if` rubric when reviewing. For critic agreement, label `human_accept` (boolean) on each generated attempt in `record.trace.attempts`, including intermediate attempts. Call `src.p3_evaluation.summarize(reviewed_json)` only after all records are reviewed. Execution errors must be resolved before scoring. `revision_limit` is an unresolved quality outcome, not automatic acceptance. `vision_only_uncertain` in S14 is a human-reviewed behavior requirement; the Planner's operational route remains `vision_only`.

## Validation status and limitations

CPU tests use explicitly mocked model outputs to check routing, safety gates and control flow. They are **not** evidence of visual quality, model accuracy, attack resistance, or real GPU latency. Gradio construction was tested with 6.10.0. GPU generation, actual two-image Qwen critique, LoRA compatibility, and the complete 15×2 plus 10 real runs have not been executed in this environment. No experimental scores have been added.

The Planner and prompt filters are transparent heuristic baselines. The VLM safety review and SD safety checker are imperfect. Enhancement currently uses img2img, which can alter text or small details; inspect the frozen preservation rubrics rather than assuming success. Both reviewers must inspect images and traces. Model and inference dependency pins are candidate settings pending the Windows smoke run.
