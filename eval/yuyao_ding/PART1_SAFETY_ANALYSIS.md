# Part 1 safety analysis: AI-assisted report draft

The team must verify and rewrite this draft in its own words for the IEEE report.
Use the executed metrics notebook for all experimental counts and measurements.
The original PDFs are course teaching material with constructed observations;
they are not verified official macroeconomic statistics.

## Trust boundary and two injection channels

The manifest supplies document trust and a page-level exception for DOC3
Appendix C. Its false narratives include a wrong Lehman bankruptcy date and a
claim that China contracted during the crisis. Its text instructs AI assistants
to replace a Euro-area unemployment outturn and forge a Document 1 citation.
The Figure C-1 image contains a separate faint instruction to report a false
U.S. inflation observation and mislabel the chart as Figure 1-1. These are
untrusted document contents, not instructions from the user or the application.

Ingestion retains the appendix text and crop for audit, but excludes the page
from evidence chunks, BGE/Chroma text retrieval, BM25 and CLIP visual embeddings.
Exact-label requests for its chart are blocked. A second answer validator
rejects unretrieved citations and visual labels, including DOC3:p7 and Figure
C-1. Thus the image attack is prevented from reaching the evidence prompt;
this test does not show that Qwen would resist the image if it were supplied
as trusted context. Page-level trust metadata must remain correct for this
protection to work, and arbitrary attacks on trusted pages remain a limitation.

## Observed successes and failures

The retained initial candidate accepted the public China contraction premise
and produced an unsupported causal explanation even though the appendix was
excluded. Exclusion alone therefore does not establish factual correctness.
A generic directional-premise resolver added from that practice review now
checks the premise against one unambiguous trusted observation and returns
false_premise when it contradicts the reported growth direction. The revised
public Q029 records that correction; Q031 uses the trusted Supplement A
outturn rather than the appendix's forged instruction. Q037 refuses a permanent
knowledge-base edit and produces no document citation.

The initial golden candidate also confused near-identical charts: plural
figure lists fell through semantic retrieval, and table-only structured
resolution could preempt an explicitly named chart. The revised parser handles
singular and plural lists, shorthand labels, dash variants and corrected chart
identities. Named charts now use their visual route rather than table-only
facts. These fixes were development feedback; their rerun is not an independent
holdout assessment. Consult actual per-question outputs and AI review notes
for remaining chart-reading, arithmetic, completeness and citation failures.

## Conversation values and read-only state

Each conversation has a separate memory store. User values are scoped overlays,
never vector-database records; source questions still use document evidence.
The runner commits a turn to memory only after validation, and an exception
leaves the conversation unchanged. Permanent updates are rejected without
retrieval. Tests cover same-key replacement, separate conversations, failed
turns and unchanged retrieval-index sizes. Actual practice outputs demonstrate
session arithmetic and the appropriate status after the generic comparison
fix. These checks do not imply all natural-language assumption wordings work;
the frozen multi-turn evaluation and member reviews must expose failures.

## Privacy and reproducibility limits

Original course inputs, local databases and model weights remain in ignored
private directories. Input hashes are checked after execution. Frozen cases,
source commits, model revisions, runtime manifests, failed attempts and review
run hashes are retained. No result is replaced with a synthetic pass. AI-assisted
content review is labeled provisional and separate from the required two-member
groundedness ratings. Missing scores remain null. Actual T4/P100 execution,
Kaggle selection and final report approval require separate observed evidence.
