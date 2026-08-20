"""Sarvam's language codes ↔ the FLORES codes the corpus is tagged with.

Sarvam returns `hi-IN`; MSMARCO-XI tags rows `hin_Deva` (docs/01-dataset.md).
The detected language is a routing key: it picks which slice of the index to
search, as a payload filter, and Gate 1 abstains when we have no index for it.
"""

from __future__ import annotations

# Only the codes Sarvam can return, mapped to the FLORES tags in the dataset.
_SARVAM_TO_FLORES: dict[str, str] = {
    "as-IN": "asm_Beng",
    "bn-IN": "ben_Beng",
    "brx-IN": "brx_Deva",
    "doi-IN": "doi_Deva",
    "en-IN": "eng_Latn",
    "gu-IN": "guj_Gujr",
    "hi-IN": "hin_Deva",
    "kn-IN": "kan_Knda",
    "kok-IN": "gom_Deva",
    "ks-IN": "kas_Arab",
    "mai-IN": "mai_Deva",
    "ml-IN": "mal_Mlym",
    "mni-IN": "mni_Beng",
    "mr-IN": "mar_Deva",
    "ne-IN": "npi_Deva",
    "od-IN": "ory_Orya",
    "or-IN": "ory_Orya",
    "pa-IN": "pan_Guru",
    "sa-IN": "san_Deva",
    "sat-IN": "sat_Olck",
    "sd-IN": "snd_Arab",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "ur-IN": "urd_Arab",
}

_DISPLAY: dict[str, str] = {
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "eng_Latn": "English",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}


def to_flores(code: str | None) -> str | None:
    """`hi-IN` → `hin_Deva`. Unknown or absent codes return None."""
    if not code:
        return None
    return _SARVAM_TO_FLORES.get(code) or _SARVAM_TO_FLORES.get(code.lower())


def display_name(flores: str | None) -> str:
    if not flores:
        return "that language"
    return _DISPLAY.get(flores, flores)
