# Part 3: completed human metrics, calculated on CPU

This table is the saved output of the appended CPU-only cell in
[the main Part 3 notebook](../../../../../notebooks/Yuyao_Ding/p3_agents.ipynb).
It reads the supplied reviewed records and reuses the original T4 request timings.
No models, images or CUDA/PyTorch setup are loaded. All values match the previously
recorded human-metric JSON; original reviews and GPU outputs remain unchanged.

| Measurement | With quality loop (final) | Without quality loop (baseline) | Red team |
| --- | ---: | ---: | ---: |
| Cases | 15 | 15 | 10 |
| Rubric passes | 10/15 | 11/15 | 10/10 |
| Task success rate | 66.67% | 73.33% | N/A |
| Red-team attack success | N/A | N/A | 0/10 (0.00%) |
| Independent reviewer agreement | 13/15 (86.67%) | 12/15 (80.00%) | 9/10 (90.00%) |
| Critic/human agreement | 6/9 (66.67%) | 5/9 (55.56%) | 1/1 (100.00%) |
| Median seconds/request (recorded T4 requests) | 17.2280 | 16.7367 | 0.0002 |
| Human review complete | True | True | True |

The table supplies task success, critic agreement, red-team attack success,
independent reviewer agreement and efficiency for the final/baseline comparison.
Critic agreement uses generated-attempt decisions, not all caption/error cases.
Both conditions retain output safety; the baseline disables quality revisions.

The earlier unscored notebook table remains a historical pre-review snapshot.
Use this appended table for the report. All paired generated images have equal
recorded hashes and no quality revision occurred. The score difference across
conditions is a difference in supplied case judgments, not evidence of a causal
quality-loop effect. The red-team rate is restricted to this small fixed suite.

`cpu_result_table.json` records input hashes, recomputed metrics, original inference
identity and the separate CPU-analysis source. `cpu_result_table.csv` is an export
of the displayed table. All preexisting source/review files retain their bytes.
