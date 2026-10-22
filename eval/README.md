# Part 1 golden evaluation set

`p1_golden_questions.csv` contains 25 held-out evaluation turns in 24
conversations. `p1_golden_answers.csv` contains the human-authored reference
answers, required facts, expected status, and evidence provenance.

The notebook must not import either golden file while building indexes,
retrieving evidence, generating answers, or selecting rules. Run the golden
evaluation only after a candidate implementation is fixed. Do not add
question IDs, exact question strings, or reference answers to the RAG logic.

Frozen SHA-256 checksums:

- `p1_golden_questions.csv`: `5d86263ab0125fbab8c724c8acbc876b82f27e2388acec4178f5a6d4881ab493`
- `p1_golden_answers.csv`: `77e78186179adbdf6a83230d8f8be4c19b5adf4bc27c5de8fce6d094d55b6b47`

Safety invariant: no golden answer cites `DOC3:p7` or `Figure C-1`.
