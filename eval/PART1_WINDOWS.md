# Part 1: existing Windows environment

Use the already working project `.venv` (Python 3.11.9, PyTorch 2.8.0+cu128).
Do not reinstall the root requirements into this environment. The older Kaggle
snapshot pins pandas 2.2.3, while this working Windows environment has 3.1.1.
The Windows additions were planned against all existing pinned distributions,
then installed with `--no-deps`; none of the original package versions changed.
`pip check` passed. Keep the full validated snapshot for provenance, not as an
instruction to replace an existing CUDA environment.

## Inputs and execution

The seven original course files are private under ignored `data/part1/`.
Their archive/file hashes are in `data/part1/local_input_hashes.json`. Do not
commit PDFs, practice inputs, model weights, runtime databases or archives.
The input path can be overridden with `DATA266_PART1_DIR`. Graders can set
`QUESTIONS_PATH` to any course-format CSV; row counts are derived from that CSV.

Run from the repository root, after stopping the Gradio GPU demo:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_part1_windows.py
```

The runner starts a fresh kernel using exactly the Python that launched it.
It installs nothing. It saves cell outputs after every cell and a final
execution manifest under `.runtime/part1/<UTC timestamp>/`. Both full practice
inference and the frozen baseline/final evaluation are enabled by default.
For a development-only skip, set `DATA266_RUN_FULL_PART1=0` or
`DATA266_RUN_P1_EVAL=0`; skipped work must not be reported as completed.

Text and visual retrieval use CPU. Qwen2.5-VL-3B-Instruct uses CUDA device 0,
BF16 on supported hardware and FP16 otherwise, deterministic decoding, and
seed 266. The CUDA allocator is bounded to at most 14 GiB. The notebook reports
actual peak allocated/reserved memory. This is an allocator-limited RTX4090
test, not evidence of execution on a physical T4/P100 or a Kaggle submission.

The original historical golden CSVs are retained. The new 25-case version 2
has exact required category coverage and was committed before inference.
It is an AI-assisted draft requiring team authorship/originality/reference
review; do not claim it was written independently by the two students.
See `PART1_EVALUATION_PROTOCOL.md` and `PART1_COMPATIBILITY_LOG.md`.

## Reviews

The runtime `evaluation/*.review.json` files bind reviews to an immutable run
hash. Correctness and factual citation judgments must name their reviewer
and method. AI-assisted content review is disclosed separately from human
groundedness. Both students must independently rate all 25 final answers 1-5;
blank ratings stay missing. Use the exported member A/B forms and the review
import helper, then run the metrics notebook. Never manufacture a second rater.

## External completion items

The team must review the AI-assisted question draft, supply both independent
groundedness ratings, incorporate the measured table and safety analysis into
its own IEEE report, verify a final tagged Kaggle notebook on one 16 GB GPU,
validate and submit the output, and select the final competition submission.
Repository privacy, instructor/ISA access, meeting recordings and individual
contributions must be checked by the team. Local execution does not establish
these external actions.

## Regression checks

```powershell
$env:PYTHONUTF8='1'
$env:DATA266_PART1_INPUT_DIR=(Join-Path (Get-Location) 'data/part1')
.\.venv\Scripts\python.exe -m unittest tests.test_p1_evaluation tests.test_p1_evidence tests.test_p1_reasoning_regressions tests.test_p1_labels tests.test_p1_reviews tests.test_p1_intent -v
```

The real-PDF tests require the private inputs; synthetic fixtures are explicitly
marked and do not count as experimental results or human ratings.
