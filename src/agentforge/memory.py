"""Short-term (context-window) and long-term (persistent) memory.

ShortTermMemory keeps the message list and compacts it by summarising old turns.
The cut is always placed at a *real user turn*, so a `tool_use` block is never
separated from its `tool_result` (which would make the API reject the request).
LongTermMemory is a small SQLite-backed store; swap it for a vector DB in prod.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .llm import LLM, Usage


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap heuristic (~4 chars/token). Good enough to trigger compaction."""
    return len(json.dumps(messages, default=str, ensure_ascii=False)) // 4


def render_transcript(messages: list[dict[str, Any]], tool_chars: int = 500) -> str:
    lines: list[str] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"{m['role'].upper()}: {content}")
            continue
        for b in content:
            kind = b.get("type")
            if kind == "text":
                lines.append(f"{m['role'].upper()}: {b['text']}")
            elif kind == "tool_use":
                lines.append(f"TOOL_CALL {b['name']}({json.dumps(b['input'], default=str)})")
            elif kind == "tool_result":
                body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                lines.append(f"TOOL_RESULT: {body[:tool_chars]}")
    return "\n".join(lines)


SUMMARY_SYSTEM = (
    "You compress agent conversations. Produce a dense summary that preserves: the user's goal, "
    "decisions made, facts discovered (with exact numbers/paths/names), tool results that still "
    "matter, and open TODOs. Omit pleasantries and dead ends."
)


class ConversationMemory:
    def __init__(self, max_context_tokens: int = 60_000, keep_recent_tokens: int = 15_000) -> None:
        self.max_context_tokens = max_context_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.messages: list[dict[str, Any]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, blocks: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": blocks})

    def reset(self) -> None:
        self.messages.clear()

    def needs_compaction(self) -> bool:
        return estimate_tokens(self.messages) > self.max_context_tokens

    def _cut_points(self) -> list[int]:
        return [
            i
            for i, m in enumerate(self.messages)
            if i > 0 and m["role"] == "user" and isinstance(m["content"], str)
        ]

    def compact(self, llm: LLM) -> Usage | None:
        """Summarise the head of the conversation. Returns usage, or None if nothing to do."""
        cuts = self._cut_points()
        if not cuts:
            return None
        cut = next(
            (c for c in cuts if estimate_tokens(self.messages[c:]) <= self.keep_recent_tokens),
            cuts[-1],
        )
        head, tail = self.messages[:cut], self.messages[cut:]
        resp = llm.generate(
            system=SUMMARY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarise this conversation:\n\n{render_transcript(head)}",
                }
            ],
            max_tokens=1024,
        )
        first = dict(tail[0])
        first["content"] = (
            f"<summary_of_earlier_conversation>\n{resp.text}\n</summary_of_earlier_conversation>"
            f"\n\n{first['content']}"
        )
        self.messages = [first, *tail[1:]]
        return resp.usage


@dataclass
class MemoryRecord:
    id: int
    text: str
    tags: list[str]
    created_at: float
    score: float = 0.0


_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s) if len(w) > 2}


class LongTermMemory:
    """Persistent notes with keyword-overlap retrieval (thread-safe)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
                "tags TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.commit()

    def add(self, text: str, tags: Iterable[str] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (text, tags, created_at) VALUES (?, ?, ?)",
                (text, json.dumps(list(tags)), time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        q = _tokens(query)
        with self._lock:
            rows = self._conn.execute("SELECT id, text, tags, created_at FROM memories").fetchall()
        scored: list[MemoryRecord] = []
        for id_, text, tags_json, created in rows:
            tags = json.loads(tags_json)
            overlap = len(q & _tokens(text)) + 2 * len(q & {t.lower() for t in tags})
            if overlap:
                scored.append(MemoryRecord(id_, text, tags, created, float(overlap)))
        scored.sort(key=lambda r: (r.score, r.created_at), reverse=True)
        return scored[:k]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
