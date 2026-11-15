# Yuyao Ding — implementation helpers

Reusable code used by [Yuyao's notebooks](../../notebooks/Yuyao_Ding/README.md).

- `p1_*.py`: structured evidence, intent/label handling and frozen-set evaluation.
- `p2_partner_baseline.py`: the separate end-to-end partner-baseline pipeline.
- `p3_*.py`: typed state, planning, vision, prompt rewriting, image generation,
  enhancement, critique, orchestration, reporting and the local Gradio app.
- `repository_paths.py`: resolve historical paths through the recorded directory
  migrations; original hashes and experiment records are not rewritten.

Use imports under `src.yuyao_ding`. Moving modules does not imply a new experiment
or change the original run's source commit. AI assistance and verification are
recorded in [AI_USAGE.md](../../AI_USAGE.md).
