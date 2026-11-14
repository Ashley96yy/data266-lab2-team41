# Part 3 Colab T4 run provenance

The team supplied an executed Colab notebook (`p3_agents (3).ipynb`, as-received
SHA-256 `0cf6aa56ef9103fe85e9d1326cbc1c3bfc58d10ba98960a20efb9c67f8ad1027`)
and its private run archive (`part3_t4_run_53c92a3_20261114T060249Z.zip`,
SHA-256 `bd70123ea99d06c48f3d8da6d959ec8bcc30da96ff02e7bdbe8c239d52ba99c1`).
The archive passed CRC, duplicate-name, path-traversal and symlink checks.
It is retained locally by the team and is **not** committed to GitHub.

The archived `manifest.json` reports a completed Tesla T4 run on source commit
`53c92a3351d8c9bc9d4c33b4d42ab43df9057753`, seed 266. The stored primary
records contain 15 cases with the quality loop and 15 without it; the red-team
records contain 10. All three modes have zero execution errors. The three frozen
specification hashes in the manifest match the committed files. The Notebook
records 41 passing CPU contract tests. Human outcomes remain unscored.

The primary Notebook was assembled from the original six code cells and their
exact matching executed outputs, removing temporary setup and troubleshooting
cells. Its environment cell was re-executed by the user after the final source
pull, so both that output and the GPU run manifest report `53c92a3`. No model
inference was rerun during this cleanup.

The original Colab-generated report contained a hardcoded statement that T4/P100
hardware was unverified, contradicting the completed T4 manifest. The reporting
function was corrected and run on the preserved JSON records using CPU only.
`eval/PART3_T4_MEASURED_RESULTS.md` differs from the original report only in that
hardware-status sentence and a provenance wording sentence; every numerical
result is unchanged. The corrected Markdown display in the primary Notebook is
marked as postprocessed. The private archive preserves the original report.

Generated images, full traces, private fixtures, and the consented portrait are
excluded from the repository. The observed execution and CLIP diagnostics do not
establish human task success, attack success, reviewer agreement or critic/human
agreement; those require the team's independent reviews and adjudication.
