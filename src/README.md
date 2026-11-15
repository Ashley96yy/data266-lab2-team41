# Implementation workspaces

| Member | Package | Current contents |
| --- | --- | --- |
| [Yuyao Ding](yuyao_ding/README.md) | `src.yuyao_ding` | Main Part 1 and Part 3 implementations, Part 2 partner-baseline helper |
| [Pratiksha Kaushik](pratiksha_kaushik/README.md) | `src.pratiksha_kaushik` | Reserved for standalone helpers; her current implementations remain in her notebooks |

The root `__init__.py` is a shared Python package entry point. Import member code
explicitly, for example `from src.yuyao_ding.p1_evaluation import summarize`.
The main Gradio app entry point is `python -m src.yuyao_ding.p3_app` from the
repository root in the appropriate GPU environment.
