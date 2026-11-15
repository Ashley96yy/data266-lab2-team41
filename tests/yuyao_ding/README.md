# Yuyao Ding — tests

These CPU contract/regression tests cover the main Part 1/3 systems, the partner
baseline helper, review-file identity and aggregation, redacted traces and
preserved evidence across directory migrations. They use synthetic fixtures
where appropriate; their pass counts are not model accuracy or human judgments.

The app-construction check requires Gradio. Some optional dependency checks may
skip when their dependency is absent. No test should silently substitute a CPU
check for a recorded GPU result.
