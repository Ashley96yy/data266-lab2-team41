# Part 3 measured development rerun

Source commit: `53c92a3351d8c9bc9d4c33b4d42ab43df9057753`. Host: Tesla T4.

This is a development rerun after documented implementation fixes, using the unchanged frozen cases. Flow completion and CLIP similarity are not human pass/fail labels. The supplied independent reviews by Yuyao Ding and Pratiksha Kaushik and their joint adjudication have now been imported. The original GPU records are unchanged; human metrics below were aggregated from saved files without new inference. See [the review evidence and limitations](part3/results/t4_20261114T060249Z/).

| Measurement | With quality loop | Without quality loop | Red team |
| --- | ---: | ---: | ---: |
| Cases | 15 | 15 | 10 |
| Execution errors | 0 | 0 | 0 |
| Generated attempts | 9 | 9 | 1 |
| CLIP-scored attempts | 9 | 9 | 1 |
| Mean first-output CLIP cosine | 0.3497 | 0.3497 | 0.2972 |
| Mean final-output CLIP cosine | 0.3497 | 0.3497 | 0.2972 |
| Median seconds/request | 17.2280 | 16.7367 | 0.0002 |
| Human rubric passes | 10/15 | 11/15 | 10/10 |
| Human task success rate | 66.67% | 73.33% | N/A |
| Human attack successes | N/A | N/A | 0/10 |
| Human attack success rate | N/A | N/A | 0% |
| Independent reviewer agreement | 13/15 (86.67%) | 12/15 (80.00%) | 9/10 (90.00%) |
| Critic/human agreement | 6/9 (66.67%) | 5/9 (55.56%) | 1/1 (100%) |

Operational statuses, not human success rates:

- with: {"answered": 4, "completed": 9, "input_error": 1, "needs_clarification": 1}
- without: {"answered": 4, "completed": 9, "input_error": 1, "needs_clarification": 1}
- red_team: {"answered": 2, "completed": 1, "input_error": 2, "refused": 5}

Observed peak PyTorch allocation: 10.320 GiB; peak reserved: 11.041 GiB, under a 14.0 GiB allocator budget. Actual Tesla T4 hardware verified by this completed run. Total diagnostic and evaluation runtime: 582.51 seconds.

The no-quality-loop mode still runs the same output safety review and records the same diagnostic CLIP score. Its quality decision cannot trigger a revision. This is a revision-loop ablation, not removal of all safety checks. First-generation requests, prompts, negative prompts, model revisions, backends and settings were checked equal across the paired modes. Later prompts/photometric revision strengths may change within the fixed two-revision bound.

Enhancement is explicitly a geometry-preserving Pillow photometric backend with zero diffusion steps, followed by the existing SD safety checker and VLM review. Style and domain routes still use real SD img2img; domain routes use the pinned team LoRA. A low CLIP score alone never becomes an unsafe label, and a high score does not establish correct objects, text or safety. All 19 recorded generated attempts now have final human accept/nonaccept labels; critic agreement uses these attempt-level labels.

Every experimental number in this report is calculated from saved records. Human labels and all disagreement reasons come from the three submitted reviewer files; aggregation does not create human judgments. Inspect images and full traces before interpreting quality or safety, and retain any revision-limit or blocked outcomes.

Redacted per-request JSON traces for the 40 evaluated cases are published under [traces/t4_20261114T060249Z](../../traces/yuyao_ding/t4_20261114T060249Z/). The full raw traces and images remain in the team's private run archive.

The final human metrics are also saved as actual output of the appended CPU-only
cell in [the main notebook](../../notebooks/Yuyao_Ding/p3_agents.ipynb), with
[a computed CSV/JSON export and input hashes](part3/results/t4_20261114T060249Z/CPU_RESULTS.md).
This supplements the preserved pre-review GPU output; it does not repeat inference.

## Human-review interpretation limits

Six case-level and four image-level disagreements have supplied joint resolutions. Original case agreement remains 34/40 (85%), and original image-acceptance agreement is 15/19 (78.95%).

All nine corresponding generated images across the two conditions have identical SHA-256 hashes. All critics accepted their first attempts; the quality loop did not trigger revisions in this run. Final case labels differ for P3-S05, P3-S11 and P3-S12 despite identical paired images. Preserve the submitted judgments, but do not interpret the 10/15 versus 11/15 score difference as a causal quality-loop effect. Whole-case and image-acceptance labels sometimes differ, and some resolution notes are brief. The 0/10 attack-success result is restricted to this small frozen development suite.
