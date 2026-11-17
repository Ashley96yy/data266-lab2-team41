# Yuyao Ding — provenance and preserved history

- [Part 1 history](part1/README.md): original Git bundles and Colab-round history.
- [Part 1 Kaggle import](part1/kaggle_import_20261116.json): original archive/notebook hashes, unchanged result-file hashes, the tested live generation function, public-output redactions and post-run bootstrap edits.
- [Part 3 import](part3/IMPORT_PROVENANCE.md) and [T4 archive record](part3/T4_RUN_PROVENANCE.md).
- [Member-directory migration](workspace_relocation.json): old/new file paths,
  original hashes, reversible notebook reference edits and preserved output hashes.
- [Earlier evaluation-directory split](../../eval/yuyao_ding/relocation_manifest.json).

The original JSON records, Git bundles and environment snapshots are retained
without changing their contents. Source-path edits after execution are identified
separately; old hashes must not be presented as current source hashes. No training,
GPU inference or human scoring was repeated for this organization work.
