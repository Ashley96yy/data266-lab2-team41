# Test workspaces

- [Yuyao Ding](yuyao_ding/README.md): contracts and regression tests for the main
  systems, review tools, private-data boundaries and directory migrations.
- [Pratiksha Kaushik](pratiksha_kaushik/README.md): location for future standalone tests.

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

A focused example is `python -m unittest tests.yuyao_ding.test_p1_evaluation -v`.
Package `__init__.py` files support discovery; they do not constitute test results.
Synthetic tests and CPU checks are not GPU experiments or human ratings.
