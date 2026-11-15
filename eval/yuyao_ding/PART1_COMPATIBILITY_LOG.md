# Part 1 Windows compatibility log

- Attempt 20261113T212525Z, candidate f08a4bb: parsed all 51 pages and embedded 35 trusted visual objects. Stopped at cell 36 before any question inference. SentenceTransformer adapter lookup attempted the default read-only user cache despite cache_folder. Preserve the failed executed notebook and execution_manifest.json in the private runtime directory.
- Fix: resolve BGE to its pinned project-cache snapshot before calling SentenceTransformer. No retrieval, generation, questions or scoring changes. The notebook runner also keeps IPython history and Hub cache inside the project runtime. Repeat from a fresh kernel under the new commit.

- Attempt 20261113T212637Z: ingestion/index and conversation fixtures passed; public-practice GPU inference began. External nvidia-smi observed 16,658 MiB total process/device use, above the assignment capacity target. The operator interrupted it before golden-set inference. Partial files remain private. This is an interrupted capacity attempt, not a completed experiment.
- Capacity fix: bound the CUDA allocator to at most 14 GiB before Qwen loads; keep the same images, prompts, model, questions and scoring. Print measured peak allocated/reserved memory in the notebook after the run. This allocator-limited RTX4090 test still does not establish actual T4/P100 compatibility or runtime.

## Practice-driven correctness revision

- While candidate 1571e6c was running, reviewed only its public practice submission: Q029 accepted a false contraction premise; Q035/Q039 did not consistently preserve session_override status or comparison direction. Golden candidate answers and scores had not been inspected before designing the revision. Preserve candidate 1571e6c as the initial complete attempt; do not replace its files.
- General fix in p1_evidence.py: recognize directional output/economy/growth phrases; route cross-period comparisons to explicit higher/lower arithmetic; route cross-entity comparative phrases to existing difference arithmetic; reject a directional growth premise contradicted by one unambiguous trusted observation, without treating explicit counterfactuals as errors. Synthetic India/2040-2041 tests use different values and no practice/golden IDs. No reference or scoring changes.
- Repeat the full practice and both frozen evaluations under a new source commit. This is a practice-driven development comparison on an AI-assisted set, not a claim of a historically untouched holdout. Both initial and revised attempts remain available.

## Exact-label routing correction and evaluation disclosure

- Inspection of the initial completed golden answers revealed a functional exact-label bug: the parser accepted only singular fully repeated labels, so plural lists (Figures X and Y) could fall back to semantic retrieval; structured table lookup could also answer a question explicitly naming a chart. This inspection is development feedback, not a pristine held-out assessment.
- Preserve the initial complete 1571e6c run. The 6b59052 attempt 20261113T213631Z was interrupted during public practice before its golden phase so the two fixes can be evaluated together. No frozen questions, reference answers or scoring targets were changed.
- Fix: general label-list parser for Figures/Tables, shorthand labels, dash variants and corrected versions; named charts bypass table-only structured resolution. New tests use synthetic A-8/A-9/B-9 labels and India/2040 observations, and preserve Appendix C blocking.
- The rerun is explicitly a development-set evaluation after practice and routing fixes. Do not describe its metrics as an independent holdout or overwrite the original attempt. A team-reviewed fresh assessment remains advisable for final reporting.

## Completeness and source-intent revision after cbb6385 review

Retain the complete cbb6385 run, provisional AI judgments and bundle. Its measured draft exposed answer-completeness and intent failures, so this revision is explicitly development feedback, not holdout evaluation. Questions, references and numeric targets stay frozen.

General fixes: treat an average range as one comparison period; preserve same-period counterfactual comparisons and separately report original document values; restrict source-provenance shortcuts to actual provenance inquiries; infer document retrieval scope from explicit source requests rather than bare citations in untrusted instructions; pass facts to the VLM instead of terminating multi-part/explanatory requests with a single lookup; include corrected charts in revision-related requests. The common generation prompt now explicitly asks for every requested part and calculation. A local-snapshot model revision is recorded directly rather than relying on a config field that can be null. Synthetic tests use India/2040-2050 and unrelated numbers; no question IDs or frozen answers enter the inference code.

Repeat practice and both evaluation modes under a new committed source revision, preserving all prior attempts and their reported failures. Rebuild review forms for the new immutable run hash; never reuse human ratings for changed answers.

- Attempt 20261113T220726Z stopped in the exact-label routing fixture before any inference: the new revision expansion reordered two labels that were already explicitly named. Fix the expansion to add a missing corrected counterpart only; preserve explicit original/corrected comparison order. Retain this failed notebook and restart from a fresh kernel.
