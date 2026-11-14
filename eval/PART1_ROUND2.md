# Part 1 second Colab development evaluation

Round 1: original complete Colab evaluation before schema compatibility changes.
Round 2: improved real Tesla T4 run 20261114T032206Z. Questions, rubrics, seed and
model revisions stayed fixed. Changes were informed by prior failures; this is a
development rerun, not a new holdout or preregistered experiment.

Fixes: explicit array/schema instructions, format-only figure mapping conversion,
deduplication before unchanged strict evidence validation, timed/raw response
diagnostics, rebinding the captured generator callback, and measured GPU metadata.
Colab CLIP image/text outputs are compatible with Tensor and ModelOutput returns.
No CUDA/PyTorch environment was installed or replaced for this integration.

Two required distinctions: baseline vs final is the dense-only vs hybrid/structured
pipeline comparison within a round. First vs second evaluation shows development
format stability. Format fallbacks and latency are not answer correctness. No
human or citation-support scores have been invented. See the executed comparison
notebook and its notebook-generated ROUND2_IMPROVEMENT.md report for actual numbers.

The genuine executed p1_rag notebook is preserved; the canonical copy adds only
Markdown labels to distinguish rounds. Its final cell expects the provided root
helper at /content/p1_colab_schema_fix_and_rerun.py in the recorded Colab upload
layout. Original archives/full retrieved contexts/images are local-only under
.runtime/part1_colab. Published outputs omit those original inputs and contexts.

The saved Colab run had no Git checkout: retain null/unavailable rather than
retroactively assigning the later integration commit. Commit the actual source
before final reproducibility submission. Both-member groundedness review, content
and citation review, own-word report, final tag and Kaggle selection remain pending.
