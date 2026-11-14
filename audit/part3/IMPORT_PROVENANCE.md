# Part 3 ZIP import provenance

The source for this integration was the team-provided `Part3_Code_And_Executed_Notebook.zip` (SHA-256 `4a5a65142b0b4e3ddb9a8dd08d8c7f2f46bd0c4e7112e55813543af714161492`). The archive passed a CRC check, path and symlink checks, and all 45 file hashes in [part3_zip_file_hashes.json](part3_zip_file_hashes.json). That manifest describes the ZIP **as received**. The integration merges `README.md` and `AI_USAGE.md` with the newer Part 1 text and normalizes imported Windows text line endings to LF, so some final file hashes intentionally differ.

The frozen `eval/p3_scenarios.json`, `eval/p3_red_team.json`, and `eval/p3_fixture_manifest.json` were byte-identical to the existing `main` copies and were not replaced. The ZIP contains code, tests, documentation, and an executed notebook. It does not contain private fixtures, the consented portrait, model weights, generated images, or complete raw traces. These remain local to the team.

The executed notebook reports source commit `bffe8bbc48b19dea73d7f0dd2f609f8d018af75a` and a real RTX 4090 development run. The ZIP does not contain Windows Git objects, so the separately reported Windows snapshot `ffa9805` cannot be independently reconstructed from this archive. The as-received hashes provide an audit trail; the repository import preserves file content, with the line-ending and shared-document changes described above, rather than that unpushed Git commit identity.

The reported 15 paired scenarios plus 10 red-team cases had zero program exceptions. Flow statuses and CLIP similarity are diagnostics. Two independent member reviews, adjudicated task and attack outcomes, and actual T4/P100 execution remain pending. No GPU experiment was rerun during this import.
