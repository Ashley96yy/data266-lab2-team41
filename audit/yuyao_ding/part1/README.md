# Original Part 1 local Git history

Source branch: codex/part1-completion. Original head: 9fe36cb428b2ce9b0c8f0dbf813630c7626a802e.
This snapshot was synchronized with the GitHub connector because the Windows
credential store denied native Git authentication. The connector cannot set
original commit authors/dates, so the snapshot commit has a different SHA.
The accompanying Git bundle preserves the original commit objects, messages,
authors and timestamps without rewriting them. It requires base adf0e4110cacd74b50200b301c58665ae3f5dba0, which
is already in this repository. No private data directory, raw input PDFs or
model weights are included. This GitHub synchronization occurs after the
experiments; it does not prove that remote publication preceded evaluation.

Bundle SHA-256: aae5d026edd2dfe9a70c82495320033c21219f39a9807622fd6c7b8c2655c1c2

After cloning this repository, download the raw bundle and run:

```sh
git bundle verify part1-original-history.bundle
git fetch part1-original-history.bundle refs/heads/codex/part1-completion:refs/heads/audit/part1-original
git log audit/part1-original
```

The synced source tree before adding these audit files is exactly the tree
of original commit 9fe36cb428b2ce9b0c8f0dbf813630c7626a802e; upload verification checks that tree SHA.
