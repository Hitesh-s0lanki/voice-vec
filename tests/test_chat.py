"""Unit tests for conversation storage — the parts with no network in them.

The SQL is not tested here for the same reason the provider calls are not
tested in `test_voice.py`: a mock of Postgres only proves the mock works, and
the statements are exercised for real by `scripts/migrate.py` and by the app
itself. What is tested is everything around them that can be wrong in silence —
the id shape a URL depends on, the title a sidebar row shows, and the queue
that decides whether a reply is written down *after* the question it answers.
"""

import asyncio

import pytest

from src.chat.store import (
    Owner,
    is_conversation_id,
    new_conversation_id,
    title_from,
)
from src.services.voice_service import VoiceSession, _Turn
from src.tools.result import ToolResult
from src.voice import llm


class TestConversationId:
    def test_round_trips_its_own_output(self):
        assert is_conversation_id(new_conversation_id())

    def test_is_prefixed_and_dash_free(self):
        generated = new_conversation_id()
        assert generated.startswith("conv_")
        assert "-" not in generated
        assert len(generated) == len("conv_") + 32

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "conv_",
            "9f3a",
            "conv_9f3a",  # too short
            "conv_" + "f" * 33,  # too long
            "conv_" + "F" * 32,  # uuid4().hex is lowercase
            "conv_" + "g" * 32,  # not hex
            "sess_" + "f" * 32,  # the other kind of id
            "conv_'; DROP TABLE conversations; --",
        ],
    )
    def test_rejects_anything_else(self, value):
        assert not is_conversation_id(value)


class TestTitle:
    def test_keeps_a_short_question_whole(self):
        assert title_from("What is the capital of Tamil Nadu?") == (
            "What is the capital of Tamil Nadu?"
        )

    def test_collapses_the_whitespace_a_transcript_arrives_with(self):
        assert title_from("  what   is\n this ") == "what is this"

    def test_cuts_a_long_one_on_a_word_boundary(self):
        title = title_from("the quick brown fox jumps over the lazy dog " * 4)
        assert title.endswith("…")
        assert len(title) <= 73
        assert not title[:-1].endswith(" ")
        # a word boundary, not mid-word
        assert title[:-1].split()[-1] in "the quick brown fox jumps over lazy dog".split()

    def test_falls_back_to_a_hard_cut_without_spaces(self):
        title = title_from("あ" * 200)
        assert len(title) == 73  # 72 + the ellipsis

    def test_keeps_devanagari_intact(self):
        assert title_from("नमस्ते, आज आप कैसे हैं?") == "नमस्ते, आज आप कैसे हैं?"


class TestOwner:
    def test_needs_one_of_the_two(self):
        assert not Owner().known
        assert Owner(session_id="sess_1").known
        assert Owner(user_id="user_1").known


class FakeChat:
    """A ChatStore that records instead of connecting."""

    def __init__(self, *, configured: bool = True) -> None:
        self.configured = configured
        self.calls: list[tuple] = []
        self.thread: list[dict[str, str]] = []
        self.row = None

    def create(self, owner, *, conversation_id=None, title=None, language=None):
        self.calls.append(("create", conversation_id, title, language))

    def append(self, conversation_id, *, role, text, **rest):
        self.calls.append(("append", conversation_id, role, text, rest.get("status")))

    def get(self, conversation_id, owner):
        return self.row

    def history(self, conversation_id, *, max_messages):
        return self.thread


class FakeTools:
    """A ToolCallStore that records instead of connecting."""

    def __init__(self, *, configured: bool = True) -> None:
        self.configured = configured
        self.rows: list[dict] = []

    def record(self, **fields):
        self.rows.append(fields)


class Row:
    def __init__(self, id: str, title: str | None = None, turns: int = 0) -> None:
        self.id, self.title, self.turns = id, title, turns


def session(
    chat: FakeChat, owner: Owner, events: list, tools: FakeTools | None = None
) -> VoiceSession:
    async def emit(event):
        events.append(event)

    async def send_audio(chunk):  # never called by these tests
        raise AssertionError("no audio in this test")

    return VoiceSession(
        emit=emit,
        send_audio=send_audio,
        owner=owner,
        chat=chat,
        tool_calls=tools or FakeTools(),
    )


def turn() -> _Turn:
    return _Turn(id="turn-1", started=0.0, language_code="hi-IN")


async def settle(voice: VoiceSession) -> None:
    """Let the writer task finish everything queued, then stop it."""
    if voice._writes is not None:
        await voice._writes.join()
    if voice._writer is not None:
        voice._writer.cancel()


class TestSessionPersistence:
    def test_first_take_opens_a_conversation_and_says_so(self):
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            await voice._open(turn(), "नमस्ते, आज आप कैसे हैं?")
            voice._save(turn(), "user", "नमस्ते, आज आप कैसे हैं?")
            await settle(voice)

        asyncio.run(run())

        announced = [event for event in events if event.type == "conversation"]
        assert len(announced) == 1
        assert announced[0].created is True
        assert is_conversation_id(announced[0].id)
        assert announced[0].id == voice.conversation_id
        assert announced[0].title == "नमस्ते, आज आप कैसे हैं?"

    def test_the_conversation_is_written_before_its_first_message(self):
        """The message carries a foreign key, and nothing awaited the insert."""
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            here = turn()
            await voice._open(here, "a question")
            voice._save(here, "user", "a question")
            voice._save(here, "assistant", "an answer", status="answered")
            await settle(voice)

        asyncio.run(run())

        assert [call[0] for call in chat.calls] == ["create", "append", "append"]
        assert chat.calls[1][2:4] == ("user", "a question")
        assert chat.calls[2][2:5] == ("assistant", "an answer", "answered")

    def test_a_second_take_stays_in_the_same_conversation(self):
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            await voice._open(turn(), "first")
            first = voice.conversation_id
            await voice._open(turn(), "second")
            await settle(voice)
            return first

        first = asyncio.run(run())

        assert voice.conversation_id == first
        assert len([event for event in events if event.type == "conversation"]) == 1
        assert [call[0] for call in chat.calls] == ["create"]

    def test_reset_starts_a_new_one(self):
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            await voice._open(turn(), "first")
            first = voice.conversation_id
            voice.reset()
            await voice._open(turn(), "second")
            await settle(voice)
            return first

        first = asyncio.run(run())

        assert voice.conversation_id != first
        assert [call[0] for call in chat.calls] == ["create", "create"]

    def test_an_anonymous_client_is_not_written_down(self):
        """No session id, no user id — the loop still runs, silently."""
        chat, events = FakeChat(), []
        voice = session(chat, Owner(), events)

        async def run():
            await voice._open(turn(), "a question")
            voice._save(turn(), "user", "a question")
            await settle(voice)

        asyncio.run(run())

        assert voice.persists is False
        assert voice.conversation_id is None
        assert chat.calls == []
        assert events == []

    def test_a_checkout_without_a_database_is_not_written_down(self):
        chat, events = FakeChat(configured=False), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            await voice._open(turn(), "a question")
            await settle(voice)

        asyncio.run(run())

        assert voice.persists is False
        assert chat.calls == []
        assert events == []

    def test_empty_replies_are_not_stored(self):
        """A turn cut before the first word leaves a question and no answer."""
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)

        async def run():
            here = turn()
            await voice._open(here, "a question")
            voice._save(here, "assistant", "   ", status="interrupted")
            await settle(voice)

        asyncio.run(run())

        assert [call[0] for call in chat.calls] == ["create"]

    def test_a_storage_failure_does_not_reach_the_turn(self):
        chat, events = FakeChat(), []
        voice = session(chat, Owner(session_id="sess_1"), events)
        chat.append = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("neon is down"))

        async def run():
            here = turn()
            await voice._open(here, "a question")
            voice._save(here, "user", "a question")
            voice._save(here, "assistant", "an answer")
            await settle(voice)

        asyncio.run(run())  # the exception is logged, not raised

        assert voice.conversation_id is not None


class TestBind:
    def test_picks_the_thread_back_up(self):
        chat, events = FakeChat(), []
        stored = "conv_" + "a" * 32
        chat.row = Row(stored, title="Earlier", turns=3)
        chat.thread = [
            {"role": "user", "content": "what did I ask?"},
            {"role": "assistant", "content": "this"},
        ]
        voice = session(chat, Owner(session_id="sess_1"), events)

        asyncio.run(voice.bind(stored))

        assert voice.conversation_id == stored
        assert voice.history == chat.thread
        assert events[0].type == "conversation"
        assert events[0].created is False
        assert events[0].title == "Earlier"
        assert events[0].turns == 3

    def test_a_conversation_that_is_not_yours_binds_to_nothing(self):
        chat, events = FakeChat(), []
        chat.row = None  # the store's ownership predicate found no row
        voice = session(chat, Owner(session_id="sess_1"), events)

        asyncio.run(voice.bind("conv_" + "b" * 32))

        assert voice.conversation_id is None
        assert events == []

    @pytest.mark.parametrize("value", [None, "", "not-an-id", "conv_short"])
    def test_a_malformed_id_never_reaches_the_database(self, value):
        chat, events = FakeChat(), []
        chat.row = Row("conv_" + "c" * 32)
        voice = session(chat, Owner(session_id="sess_1"), events)

        asyncio.run(voice.bind(value))

        assert voice.conversation_id is None
        assert events == []


class TestToolCalls:
    """What ran, said out loud and written down — and the two agreeing.

    The panel draws a turn being spoken from the socket and the same turn read
    back out of Postgres from `/conversations/{id}`. These are the assertions
    that keep those two the same card.
    """

    def call(self) -> llm.ToolCall:
        return llm.ToolCall(
            id="call_1", name="GMAIL_SEND_EMAIL", arguments={"to": "a@b.c"}
        )

    def test_a_finished_call_is_announced_whole(self):
        chat, tools, events = FakeChat(), FakeTools(), []
        voice = session(chat, Owner(user_id="user_1"), events, tools)
        voice.conversation_id = new_conversation_id()

        async def run():
            await voice._save_tool(
                turn(),
                self.call(),
                ToolResult("GMAIL_SEND_EMAIL", ok=True, data={"id": "m1"}, ms=42.0),
            )
            await settle(voice)

        asyncio.run(run())

        announced = [event for event in events if event.type == "tool"]
        assert len(announced) == 1
        said = announced[0]
        assert said.turn_id == "turn-1"
        assert said.slug == "GMAIL_SEND_EMAIL"
        assert said.toolkit == "gmail"
        assert said.ok is True
        # Both halves — the thread is unreadable with only the name.
        assert said.arguments == {"to": "a@b.c"}
        assert said.result == '{"id": "m1"}'
        assert said.result_bytes == len('{"id": "m1"}')
        assert said.latency_ms == 42.0

    def test_the_row_is_written_under_the_id_that_was_announced(self):
        """Otherwise a call already on screen arrives again on the next load."""
        chat, tools, events = FakeChat(), FakeTools(), []
        voice = session(chat, Owner(user_id="user_1"), events, tools)
        voice.conversation_id = new_conversation_id()

        async def run():
            await voice._save_tool(
                turn(), self.call(), ToolResult("GMAIL_SEND_EMAIL", ok=True, data={}, ms=1.0)
            )
            await settle(voice)

        asyncio.run(run())

        said = next(event for event in events if event.type == "tool")
        assert len(tools.rows) == 1
        assert tools.rows[0]["id"] == said.id
        assert tools.rows[0]["turn_id"] == said.turn_id
        assert tools.rows[0]["conversation_id"] == voice.conversation_id

    def test_a_failure_travels_as_a_failure(self):
        chat, tools, events = FakeChat(), FakeTools(), []
        voice = session(chat, Owner(user_id="user_1"), events, tools)
        voice.conversation_id = new_conversation_id()

        async def run():
            await voice._save_tool(
                turn(),
                self.call(),
                ToolResult("GMAIL_SEND_EMAIL", ok=False, error="timed out", ms=9000.0),
            )
            await settle(voice)

        asyncio.run(run())

        said = next(event for event in events if event.type == "tool")
        assert said.ok is False
        assert said.status == "failed"
        assert said.error == "timed out"
        # `for_model` renders a failure as {"error": …}; storing that would be
        # the error column said twice.
        assert said.result is None
        assert tools.rows[0]["result"] is None

    def test_it_is_announced_with_no_database_behind_it(self):
        """A session can still show what it just ran; it just cannot re-read it."""
        chat, tools, events = FakeChat(), FakeTools(configured=False), []
        voice = session(chat, Owner(user_id="user_1"), events, tools)

        async def run():
            await voice._save_tool(
                turn(), self.call(), ToolResult("GMAIL_SEND_EMAIL", ok=True, data={}, ms=1.0)
            )
            await settle(voice)

        asyncio.run(run())

        assert len([event for event in events if event.type == "tool"]) == 1
        assert tools.rows == []
