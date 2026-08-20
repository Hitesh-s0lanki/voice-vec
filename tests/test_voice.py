"""Unit tests for the parts of the voice path that fail silently.

The provider calls are not tested here — they are network calls, and a mock of
Sarvam only ever proves the mock works. What is tested is everything that turns
one thing into another with no network in between: where a reply gets cut for
speech, which language a code means, and the WAV header the streaming route
writes by hand.
"""

import struct

import pytest

from src.controllers.voice_controller import _wav_header
from src.core.config import Settings
from src.voice.languages import display, normalise, speaks
from src.voice.segment import Segmenter


def run(text: str, **kwargs) -> list[str]:
    """Feed a reply one character at a time — the worst case a stream offers."""
    segmenter = Segmenter(**kwargs)
    out: list[str] = []
    for char in text:
        out += segmenter.feed(char)
    tail = segmenter.flush()
    if tail:
        out.append(tail)
    return out


class TestSegmenter:
    def test_splits_on_danda(self):
        assert run("मैं ठीक हूँ। आप कैसे हैं?") == ["मैं ठीक हूँ।", "आप कैसे हैं?"]

    def test_splits_tamil_on_the_period(self):
        assert run("வணக்கம், நான் நல்லா இருக்கேன். நீங்க?") == [
            "வணக்கம், நான் நல்லா இருக்கேன்.",
            "நீங்க?",
        ]

    def test_decimals_do_not_split(self):
        # "3." spoken alone is a different number, and the rest of the
        # sentence would be synthesised as its own fragment.
        assert run("It weighs 3.5 kg exactly.") == ["It weighs 3.5 kg exactly."]

    def test_ellipsis_is_one_boundary(self):
        assert run("Well… that depends.") == ["Well…", "that depends."]

    def test_first_segment_is_cut_short(self):
        # The first segment is the one the listener is waiting on, so a long
        # opening clause is cut at a comma rather than held for the period.
        first = run(
            "The thing about the sea is that it covers most of the planet, "
            "and almost none of it has ever been seen by anyone at all.",
            first_chars=60,
            chars=220,
            max_chars=320,
        )[0]
        assert len(first) <= 90
        assert first.endswith(",")

    def test_a_run_on_sentence_still_gets_flushed(self):
        # No punctuation anywhere: without a ceiling this would hold the whole
        # reply back and the listener would hear nothing until it ended.
        text = "word " * 200
        assert len(run(text, max_chars=120)) > 1

    def test_markdown_marks_are_dropped(self):
        # A synthesiser reads `**` as a pause at best.
        assert run("**Bold** and `code`.") == ["Bold and code."]

    def test_punctuation_only_says_nothing(self):
        assert run("...") == []

    def test_nothing_is_lost(self):
        text = "पहला वाक्य। दूसरा वाक्य? तीसरा वाक्य!"
        assert "".join(run(text)).replace(" ", "") == text.replace(" ", "")


class TestLanguages:
    def test_bare_code_gets_a_region(self):
        assert normalise("ta") == "ta-IN"

    def test_odia_aliases_agree(self):
        assert normalise("or-IN") == normalise("od-IN") == "od-IN"

    def test_whisper_names_map_to_sarvam_codes(self):
        # Whisper answers "tamil" where Saaras answers "ta-IN".
        assert normalise("tamil") == "ta-IN"

    def test_unknown_is_none_not_a_guess(self):
        assert normalise("unknown") is None
        assert normalise(None) is None

    def test_a_language_off_the_list_survives(self):
        # French is not Sarvam's, but it is still the language to reply in.
        assert normalise("fr-FR") == "fr-FR"
        assert speaks("fr-FR") is False

    def test_display_is_a_name_a_model_can_use(self):
        assert display("hi-IN") == "Hindi"

    def test_bulbul_covers_eleven(self):
        assert speaks("hi-IN") and speaks("ta-IN") and speaks("en-IN")
        assert not speaks("as-IN")  # transcribed, but not spoken


class TestWavHeader:
    def test_it_is_a_riff_header(self):
        header = _wav_header(24_000)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert len(header) == 44

    def test_sizes_are_left_open(self):
        # The length is unknowable when the header goes out — players read
        # until the socket closes.
        header = _wav_header(24_000)
        assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
        assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF

    def test_the_rate_is_the_one_asked_for(self):
        rate, byte_rate = struct.unpack("<II", _wav_header(16_000)[24:32])
        assert rate == 16_000
        assert byte_rate == 32_000  # mono, 16-bit


class TestProviderResolution:
    def test_openai_wins_when_both_keys_are_set(self):
        llm = Settings(openai_api_key="o", sarvam_api_key="s").resolve_llm()
        assert llm.provider == "openai" and llm.api_key == "o"

    def test_sarvam_answers_on_its_own(self):
        # The whole point: a checkout with one key still holds a conversation.
        llm = Settings(openai_api_key="", sarvam_api_key="s").resolve_llm()
        assert llm.provider == "sarvam" and llm.ready

    def test_no_keys_is_not_ready(self):
        assert not Settings(openai_api_key="", sarvam_api_key="").resolve_llm().ready

    def test_an_explicit_base_url_picks_its_own_key(self):
        llm = Settings(
            openai_api_key="o",
            sarvam_api_key="s",
            llm_base_url="https://api.sarvam.ai/v1",
            llm_model="sarvam-105b",
        ).resolve_llm()
        assert llm.provider == "sarvam" and llm.api_key == "s"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
