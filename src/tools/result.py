"""What running a tool produced, in the one shape everything downstream reads.

Three consumers, three needs, one object:

    the model      `for_model()` — a JSON string, bounded, with failures stated
                   rather than hidden
    the database   `src/chat/tool_calls.py` writes slug, ok, ms and a preview
    the browser    the activity feed shows what ran and how long it took

A tool that fails is a `ToolResult` with `ok=False`, never an exception. Every
caller here is mid-turn with a listener waiting, and the exception would only
travel as far as the caller's `except` and become this same object.
"""

from __future__ import annotations

import json
from typing import Any

# A tool result can be an entire inbox page. It goes back into the prompt, so it
# is capped: past a few thousand characters it stops adding anything a spoken
# answer can use and starts costing latency on every subsequent turn.
MAX_RESULT_CHARS = 4_000


class ToolResult:
    """What running one tool produced, in a shape both the model and the
    database can take."""

    __slots__ = ("slug", "ok", "data", "error", "ms")

    def __init__(
        self, slug: str, *, ok: bool, data: Any = None, error: str | None = None, ms: float = 0.0
    ) -> None:
        self.slug = slug
        self.ok = ok
        self.data = data
        self.error = error
        self.ms = ms

    def for_model(self) -> str:
        """The string that goes back as the `tool` message.

        A failure is reported rather than hidden. The model handling "that
        mailbox is not reachable" out loud is a better turn than it inventing
        an answer from a silence.
        """
        if not self.ok:
            return json.dumps({"error": self.error or "the tool failed"})

        try:
            rendered = json.dumps(self.data, default=str)
        except Exception:
            rendered = str(self.data)

        if len(rendered) > MAX_RESULT_CHARS:
            return rendered[:MAX_RESULT_CHARS] + "… (truncated)"
        return rendered

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"failed: {self.error}"
        return f"<ToolResult {self.slug} {state} {self.ms:.0f}ms>"


def toolkit_of(slug: str) -> str:
    """`GMAIL_SEND_EMAIL` → `gmail`, Composio's own convention.

    Duplicated in `src/chat/tool_calls.py` rather than imported from there: that
    module owns a Postgres table and this one runs on the microphone path, and
    the voice loop should not pull psycopg in behind a two-line string split.
    """
    head, _, _ = (slug or "").partition("_")
    return head.lower()
