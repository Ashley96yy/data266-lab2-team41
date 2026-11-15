# Part 1 architecture: report draft

```mermaid
flowchart LR
    PDFs[Private read-only PDFs + trust manifest] --> Text[Page text and overlapping chunks]
    PDFs --> Crops[Numbered figure and table crops]
    Text --> Filter[Exclude low-trust Appendix C]
    Crops --> Filter
    Filter --> Dense[BGE text and CLIP visual embeddings]
    Dense --> DB[Chroma with document/page/label/trust metadata]
    Filter --> BM25[BM25 text and caption search]
    Q[Question + conversation ID] --> Memory[Isolated session memory and scoped user values]
    Memory --> Retrieve[Hybrid text and visual retrieval]
    DB --> Retrieve
    BM25 --> Retrieve
    Q --> Labels[Explicit figure/table label router]
    Labels --> Retrieve
    Retrieve --> Evidence[Retrieved trusted evidence packet]
    PDFs --> Facts[Read-only structured source facts]
    Facts --> Resolve[Generic lookup/comparison/conflict resolution]
    Memory --> Resolve
    Resolve --> Evidence
    Evidence --> Answer[Deterministic supported answer or Qwen VLM]
    Answer --> Validate[JSON/status/word-limit/citation validation]
    Validate --> Commit[Commit validated turn to its own conversation]
    Validate --> CSV[Course-format output and audit]
```

Named-chart questions bypass table-only structured resolution. Corrected and
original chart labels retain distinct identities. The baseline replaces the
hybrid/explicit-label retrieval path with dense BGE and CLIP retrieval and
disables global structured-fact expansion; other model/decoding/validation
settings are shared. This is an end-to-end comparison rather than an isolated
BM25 ablation. Frozen references never enter the inference question objects.

All experiment measurements come from the executed inference and metrics
notebooks. Retrieval runs on CPU and generation on one CUDA device. Original
inputs, databases and checkpoints stay outside version control. See the protocol,
compatibility history, safety draft and measured report for observed limits.
