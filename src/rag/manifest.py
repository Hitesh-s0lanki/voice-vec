"""What the last ingest built.

docs/07-evaluation.md's reporting checklist asks for the index to be stated
alongside every number — chunk count, strategies, model, dedup rate. The ingest
writes it here; `/health` and `/metrics` read it back, so a latency figure can
always be traced to the index that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

MANIFEST_PATH = Path("data/index-manifest.json")


@dataclass(slots=True)
class IndexManifest:
    collection: str
    model: str
    dim: int
    strategies: list[str]
    languages: list[str]
    rows: int
    passages: int
    chunks: int
    dedup_rate: float
    token_p99: int | None
    elapsed_seconds: float
    source: str
    built_at: str
    extras: dict = field(default_factory=dict)

    def write(self, path: Path = MANIFEST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    """The manifest, or None when nothing has been ingested on this machine.

    Cached on the file's mtime rather than read every call: `/ask` consults it
    twice per request (Gate 1's language check, and whether a language filter is
    worth applying), and a stat is cheaper than a parse. Keying on mtime means a
    fresh ingest is still picked up without a restart.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return _load(str(path), mtime)


def indexed_languages(fallback: list[str] | None = None) -> list[str]:
    manifest = read_manifest()
    languages = (manifest or {}).get("languages")
    if isinstance(languages, list) and languages:
        return [str(code) for code in languages]
    return fallback or []
