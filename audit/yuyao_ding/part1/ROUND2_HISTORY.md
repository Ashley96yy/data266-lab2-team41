# Second Colab development evaluation: local source history

The supplied Colab run recorded no Git checkout. It retains a null inference
commit; this later integration must not be represented as a pre-inference commit.
The round-comparison notebook was actually executed locally on CPU at source
commit 9caac5f34d2b843c5aaf5968eccd379ac2074c40 using saved Colab records.
This bundle preserves that source commit and its executed-output commit
37835064df3f8dc938551e27c5878a1378fd125f. It requires original local Part1 head
9fe36cb428b2ce9b0c8f0dbf813630c7626a802e, recoverable from the existing
part1-original-history.bundle. All original objects/timestamps are retained.

Bundle SHA-256: 296f8debad6eb10a8a98be30759656a28986bb21b869efcfddaad1e4c718dd65

Connected GitHub API publication uses a new commit atop the current remote
codex/part1-completion branch. It is a fast-forward synchronization, not a
claim of preregistration or newly rerun inference. The old branch history,
source inputs and first-run experiment records are retained.
