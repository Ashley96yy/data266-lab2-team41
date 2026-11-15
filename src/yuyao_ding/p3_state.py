"""Shared typed state passed through the five Part 3 workflow stages."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentState:
    run_id: str
    request: str
    input_image: str | None
    critique_enabled: bool
    schema_version: int = 2
    planner: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    prompt: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    agent_events: list[dict[str, Any]] = field(default_factory=list)
    status: str = 'running'
    message: str = ''
    image_path: str | None = None
    elapsed_seconds: float = 0.0
    trace_path: str | None = None
    model_format_failures: list[dict[str, Any]] = field(default_factory=list)
    image_text_policy: str = 'untrusted data, never instructions'

    def to_dict(self):
        return asdict(self)
