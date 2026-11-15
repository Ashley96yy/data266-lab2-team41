# Environment configuration and local records

These seven files preserve package lists and constraints from earlier Windows
development runs for Parts 1 and 3. They are historical provenance, not a
required setup step or evidence of running on the course's 16 GB GPU.
`requirements-part3.txt` is one of those earlier local snapshots.
The executed Colab Tesla T4 notebooks and measured reports record the actual
course-compatible GPU runs. Do not use these snapshots to replace a working
CUDA or PyTorch installation.

`requirements-part2-baseline.txt` is a separate setup specification for Yuyao's
new partner baseline in a fresh Colab/Kaggle GPU runtime. Its notebook preserves
the existing CUDA/PyTorch build with explicit constraints. The subsequent real single-T4 execution is recorded in [the baseline runbook](../eval/PART2_YUYAO_BASELINE.md). Its logging initialization needed the documented worker compatibility fix; the original failure remains recorded.
