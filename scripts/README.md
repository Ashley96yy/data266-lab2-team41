# Runner and review-tool workspaces

- [Yuyao Ding](yuyao_ding/README.md): main-system runners, exports and human-review tools.
- [Pratiksha Kaushik](pratiksha_kaushik/README.md): location for future standalone tools;
  current notebook-based tools remain in her notebook workspace.

Run scripts from the repository root, for example:

```bash
python scripts/yuyao_ding/p1_content_review_forms.py --help
python scripts/yuyao_ding/finalize_p1_content_review.py --help
```

The root `__init__.py` supports package imports. Scripts which require private
inputs or GPU models still require those inputs and the appropriate environment.
