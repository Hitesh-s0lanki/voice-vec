"""Who heard it, who can say it back.

One language code travels the whole turn: Sarvam's STT detects it, the reply
model is told to answer in it, and it decides which synthesiser gets the text.
Codes are BCP-47-ish (`hi-IN`) because that is what Sarvam speaks; anything
else that arrives is normalised on the way in rather than special-cased later.
"""

from __future__ import annotations

# Everything Saaras can return. The value is the English name, which is what
# the reply model is actually told — "answer in Tamil" beats "answer in ta-IN".
LANGUAGES: dict[str, str] = {
    "as-IN": "Assamese",
    "bn-IN": "Bengali",
    "brx-IN": "Bodo",
    "doi-IN": "Dogri",
    "en-IN": "English",
    "gu-IN": "Gujarati",
    "hi-IN": "Hindi",
    "kn-IN": "Kannada",
    "kok-IN": "Konkani",
    "ks-IN": "Kashmiri",
    "mai-IN": "Maithili",
    "ml-IN": "Malayalam",
    "mni-IN": "Manipuri",
    "mr-IN": "Marathi",
    "ne-IN": "Nepali",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "sa-IN": "Sanskrit",
    "sat-IN": "Santali",
    "sd-IN": "Sindhi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ur-IN": "Urdu",
}

# Bulbul speaks eleven of them. The rest are transcribed and answered fine —
# they just get synthesised by OpenAI, which has the breadth Bulbul trades for
# its Indic quality.
SARVAM_TTS_LANGUAGES: frozenset[str] = frozenset(
    {
        "bn-IN",
        "en-IN",
        "gu-IN",
        "hi-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
    }
)

# Sarvam answers `od-IN`; ISO-639 and half the internet say `or`. Same language.
_ALIASES: dict[str, str] = {"or-IN": "od-IN", "or": "od-IN", "ory": "od-IN"}

# Whisper names the language instead of coding it — "tamil", not "ta". Same
# table, read backwards, so both providers land on the same key.
_BY_NAME: dict[str, str] = {name.lower(): code for code, name in LANGUAGES.items()}


def normalise(code: str | None) -> str | None:
    """`ta`, `ta-in`, `tam` → `ta-IN`. Unknown or absent → None.

    None is a real answer, not a failure: it means nobody has told us what was
    spoken, and every caller downstream has a defensible default for that.
    """
    if not code:
        return None

    raw = code.strip()
    if not raw or raw.lower() in {"unknown", "auto", "und"}:
        return None

    if raw in _ALIASES:
        return _ALIASES[raw]
    if raw.lower() in _BY_NAME:
        return _BY_NAME[raw.lower()]

    parts = raw.replace("_", "-").split("-")
    base = parts[0].lower()
    region = parts[1].upper() if len(parts) > 1 else None

    candidate = f"{base}-{region}" if region else None
    if candidate in LANGUAGES:
        return candidate
    if candidate in _ALIASES:
        return _ALIASES[candidate]

    # No region, or a region we do not index: `hi-US` is still Hindi.
    indian = f"{base}-IN"
    if indian in LANGUAGES:
        return indian
    if base in _ALIASES:
        return _ALIASES[base]

    # A language outside Sarvam's list — French, Japanese. Keep it: the reply
    # model and OpenAI's synthesiser both understand far more than this table.
    return raw if len(base) >= 2 else None


def display(code: str | None) -> str | None:
    """The English name of a language code, for prompts and for the UI."""
    normalised = normalise(code)
    if not normalised:
        return None
    return LANGUAGES.get(normalised, normalised)


def speaks(code: str | None) -> bool:
    """Can Bulbul say this one?"""
    return normalise(code) in SARVAM_TTS_LANGUAGES
