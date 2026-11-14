# Observations requiring human interpretation

The notebook completed real inference, but that does not establish rubric success.
In the latest run no quality revision was triggered. The two conditions therefore
do not demonstrate a quality benefit from the revision loop. Earlier archived runs
contain revisions and critic false accepts; the software bound is also tested with
synthetic control-flow fixtures, which are not experimental quality judgments.

Visual inspection found a clearer watercolor treatment after prompt changes, with
remaining geometry/preservation defects. The monochrome constraint now removes
color, while the requested ink-drawing treatment can remain weak and added markings
can appear. The critic can accept such defects. Both members must judge the complete
rubrics independently; these observations are AI-assisted development notes only.

A domain-style output was blocked by the unchanged SD safety checker in both modes.
The image was not accepted or saved as a safe result. The original request being
fictional does not establish whether the unseen generated content was safe, so this
record cannot be called a confirmed false positive or a rubric pass.

One caption request has an unusually long recorded wall-clock duration. The cause
is unverified. Its original duration is retained in the raw trace and per-case CSV;
no observation was removed. Treat the latency figures as a single observed desktop
run, not a robust performance benchmark. Red-team medians include inexpensive CPU
refusals and invalid-input handling, so they are not neural inference latency.

Unchanged inputs and specifications, execution completeness, equal first settings
and source hashes are recorded separately. Pending human metrics, actual 16GB GPU
verification and the team's report/publication obligations remain explicit.
