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
| 2026-11-14 | Codex; model ID not exposed | Part 1 second Colab development revision: CLIP return-type compatibility guidance, generic JSON container normalization, bounded repair instructions, captured-callback rebinding, source-hash/hardware provenance and first/second evaluation comparison | User executed the new script on Tesla T4. Raw runs and notebooks retained; CPU contracts and comparison execution recorded. Content/citation and both independent human ratings remain pending; format improvement is not claimed as answer correctness or Kaggle improvement | p1_colab_schema_fix_and_rerun.py, Part 1 notebooks, round1/round2 output summaries and protocol |
| 2026-11-13 | Codex; model ID not exposed | Part 3 development revision: typed state/events, effective prompt compilation, separate quality/safety critique, CPU CLIP similarity, photometric enhancement, regression tests and versioned real-GPU rerun | Existing private inputs and fixed cases retained; independent diagnostics and GPU traces recorded. Two-member quality/safety judgments and report authorship remain pending. No human labels fabricated | `src/p3_*`, `tests/test_p3_*`, `eval/validate_p3_revision.py`, Part 3 notebook and revision documents |
| 2026-11-13 | Codex; model ID not exposed | Follow-up Part 3 development fixes: bounded Qwen caption rewriting, approved-medium cues, singular-count compatibility, monochrome enforcement, per-revision Prompt events, blank local independent review forms and executed notebook packaging | Automated contracts and fresh real GPU execution are recorded. Earlier failed runs retained. No member reviews, authorship, rubric scores or approval claimed | Part 3 source/tests, notebook, review helper, runbook and AI-assisted system-card draft |

Additional team disclosures, including any Part 2 tool use, require confirmation from the responsible member before final submission.
