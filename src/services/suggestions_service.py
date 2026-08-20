"""Serve the verified suggestions written by scripts/suggestions.py."""

import json
import random
from functools import lru_cache
from pathlib import Path

from src.rag.manifest import read_manifest
from src.schemas.suggestions import Suggestion, SuggestionsResponse

SUGGESTIONS_PATH = Path("data/suggestions.json")


class SuggestionsService:
    def __init__(self, path: Path = SUGGESTIONS_PATH) -> None:
        self._path = path

    def _load(self) -> list[Suggestion]:
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            # Not generated yet. An empty list is correct — the UI hides the
            # row rather than inventing questions the index may not answer.
            return []

        return [Suggestion(**item) for item in raw.get("suggestions", [])]

    def sample(self, limit: int) -> SuggestionsResponse:
        pool = self._load()
        chosen = random.sample(pool, min(limit, len(pool)))
        return SuggestionsResponse(suggestions=chosen, corpus=self._describe())

    @staticmethod
    def _describe() -> str:
        manifest = read_manifest() or {}
        chunks = manifest.get("chunks")
        rows = manifest.get("rows")
        if not chunks:
            return "No index built yet — run scripts/ingest.py."
        return (
            f"{chunks:,} passages from {rows:,} MS MARCO rows, Hindi. "
            "It answers those questions and abstains on everything else."
        )


@lru_cache
def get_suggestions_service() -> SuggestionsService:
    return SuggestionsService()
