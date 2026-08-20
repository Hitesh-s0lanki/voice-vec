"""Cut a token stream into things worth saying out loud.

The reply arrives a few characters at a time and the synthesiser wants whole
clauses, so something has to decide when enough has arrived. Waiting for the
full reply is the obvious answer and the wrong one: it puts every token's
latency in front of the first sound. Cutting at every token is the other
extreme — the synthesiser needs context for prosody, and a fragment read alone
sounds like a fragment.

So: cut at the first sentence end, keep the first cut short because that is the
one the listener is waiting on, and let later ones run longer. Indic scripts
end sentences with a danda (।) rather than a period, and Sarvam's transcripts
mix both — the boundary set covers Latin, Devanagari, Urdu and CJK punctuation.
"""

from __future__ import annotations

# Ends a sentence anywhere in the world we transcribe.
_HARD = "।॥.!?…۔؟。！？"

# Ends a clause. Only used once a segment is already long enough to be worth
# speaking — cutting at every comma would chop the prosody to pieces.
_SOFT = ",;:—–､、，；："

# A period between digits is a decimal, not a sentence: "3.5 kg" must survive.
_DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩०१२३४५६७८९"


class Segmenter:
    """Feed it text as it arrives; take back whatever is ready to be spoken."""

    def __init__(
        self,
        *,
        first_chars: int = 90,
        chars: int = 220,
        max_chars: int = 320,
    ) -> None:
        self._first_chars = first_chars
        self._chars = chars
        self._max_chars = max_chars
        self._buffer = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        return self._buffer.strip()

    def feed(self, text: str) -> list[str]:
        """Add a token; return the segments that just became speakable."""
        self._buffer += text
        ready: list[str] = []

        while True:
            cut = self._find_cut()
            if cut is None:
                break
            segment, self._buffer = self._buffer[:cut], self._buffer[cut:]
            spoken = _clean(segment)
            if spoken:
                ready.append(spoken)
                self._emitted += 1

        return ready

    def flush(self) -> str | None:
        """Whatever is left when the reply ends. Empty tail returns None."""
        rest, self._buffer = _clean(self._buffer), ""
        if rest:
            self._emitted += 1
            return rest
        return None

    # ---- internals ------------------------------------------------------

    def _target(self) -> int:
        """How long this segment is allowed to get before a soft cut will do."""
        return self._first_chars if self._emitted == 0 else self._chars

    def _find_cut(self) -> int | None:
        """Index to split the buffer at, or None to keep waiting.

        Three ways a cut happens, in order of how much we like them: a sentence
        ended; the buffer grew past its target and a clause ended; the buffer
        grew past the hard ceiling and we take what we can get.
        """
        buffer = self._buffer
        if not buffer.strip():
            return None

        for index, char in enumerate(buffer):
            if char not in _HARD:
                continue
            if char == "." and _is_decimal(buffer, index):
                continue
            # A boundary is only a boundary once something follows it —
            # otherwise "3." might still turn out to be "3.5", and an ellipsis
            # would be cut into three separate segments.
            end = index + 1
            while end < len(buffer) and buffer[end] in _HARD + '"\'”’)]':
                end += 1
            if end >= len(buffer):
                return None
            if buffer[end].isspace() or _emitted_enough(buffer[:end]):
                return end

        target = self._target()
        if len(buffer) >= target:
            window = buffer[: self._max_chars]
            soft = max((window.rfind(char) for char in _SOFT), default=-1)
            if soft > target // 2:
                return soft + 1

        if len(buffer) >= self._max_chars:
            space = buffer.rfind(" ", 0, self._max_chars)
            return space + 1 if space > self._max_chars // 2 else self._max_chars

        return None


def _emitted_enough(text: str) -> bool:
    """A boundary with no trailing space still counts if there is real text."""
    return len(text.strip()) >= 12


def _is_decimal(text: str, index: int) -> bool:
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    return before in _DIGITS and after in _DIGITS


def _clean(segment: str) -> str:
    """Trim, and drop anything with nothing left to pronounce.

    Markdown leaks into replies no matter how firmly the prompt says not to,
    and a synthesiser reads `**` as a pause at best. Strip the marks that carry
    no sound; leave the words alone.
    """
    text = segment.strip()
    if not text:
        return ""

    for mark in ("**", "__", "```", "`", "#"):
        text = text.replace(mark, "")

    text = text.lstrip("*-• \t")
    text = text.strip()

    # Punctuation only — nothing to say.
    return text if any(char.isalnum() for char in text) else ""
