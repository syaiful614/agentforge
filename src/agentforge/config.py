"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = "claude-sonnet-5-5"
    api_key: str | None = None
    workspace: Path = Path("./workspace")
    trace_dir: Path = Path("./traces")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.environ.get("AGENTFORGE_MODEL", cls.model),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            workspace=Path(os.environ.get("AGENTFORGE_WORKSPACE", "./workspace")),
            trace_dir=Path(os.environ.get("AGENTFORGE_TRACE_DIR", "./traces")),
        )
