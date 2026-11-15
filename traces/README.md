# Part 3 request traces

The completed Colab T4 run has 40 evaluated request traces: 15 with critique,
15 without critique, and 10 red-team. Their redacted JSON exports are in
[`t4_20261114T060249Z/`](t4_20261114T060249Z/), grouped by mode. Each file
retains the actual request, status, agent sequence, processed agent inputs and
outputs, and elapsed time. Its raw trace SHA-256 and run ID link it to the
private source archive; the [export manifest](t4_20261114T060249Z/manifest.json)
records the archive SHA-256, source commit, seed, GPU, case counts, and redaction
policy. The separate diagnostic request is not part of the 40-case evaluation
and is not included in this export.

Private input and generated-image paths were replaced with case-scoped labels.
Raw model retry payloads and image bytes were omitted. The two consented adult
portrait cases were refused by Planner before image analysis; their published
traces contain no portrait pixels or visual descriptions. The original archive,
all private fixtures, images, and complete raw traces remain outside GitHub.
This export does not fill in human quality or safety judgments. See the
[T4 results report](../eval/yuyao_ding/PART3_T4_MEASURED_RESULTS.md) for measured results.

The deterministic export code is in
[`scripts/export_p3_public_traces.py`](../scripts/export_p3_public_traces.py).

The later [submitted human reviews and adjudication](../eval/yuyao_ding/part3/results/t4_20261114T060249Z/) are stored separately from this original trace export. The export manifest retains its original unscored status; the later review record supplies final human labels and their provenance.
