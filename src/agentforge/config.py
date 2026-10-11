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
    provider: str = "anthropic"  # anthropic | openai (any OpenAI-compatible endpoint)
    base_url: str | None = None
    compat_api_key: str | None = None  # key for the OpenAI-compatible provider
    timeout_s: float = 120.0  # per-request timeout for the OpenAI-compatible provider

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.environ.get("AGENTFORGE_MODEL", cls.model),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            workspace=Path(os.environ.get("AGENTFORGE_WORKSPACE", "./workspace")),
            trace_dir=Path(os.environ.get("AGENTFORGE_TRACE_DIR", "./traces")),
            provider=os.environ.get("AGENTFORGE_PROVIDER", "anthropic"),
            base_url=os.environ.get("AGENTFORGE_BASE_URL"),
            compat_api_key=os.environ.get("AGENTFORGE_API_KEY"),
            timeout_s=float(os.environ.get("AGENTFORGE_TIMEOUT", "120")),
        )

    def key_for(self, provider: str) -> str | None:
        return self.compat_api_key if provider == "openai" else self.api_key
