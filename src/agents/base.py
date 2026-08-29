"""What every agent in this package agrees to.

An "agent" here is a component that *decides something with a model* and is
invoked as a unit: it is handed a question and some measured context, it makes
one or more model calls, and it returns a typed answer. `ToolAgent` is the one
that decides nothing itself — it discovers and runs what somebody linked — and
it lives here anyway, because it is the surface the voice loop's own model acts
through and the promises below are exactly the ones it has to keep.

Three promises, and they are why this is a base class rather than a naming
convention:

**Never raise.** Every agent in this package runs inside a turn somebody is
listening through, or inside a rung of the ask ladder with a deadline on it. A
raised exception there is a dropped turn; a typed failure is an answer that can
be spoken ("I could not query that") or degraded to the rung below. So the
uniform shape is: a result object carrying its own error, or `None`. `_guard`
is the one place that conversion is written down.

**Say when you cannot run.** `ready` answers "would this agent do anything if I
called it right now" — no model key, no Composio credentials — so a caller can
skip the round trip instead of paying it to learn the same thing. It is
deliberately *not* a health check: it reads configuration, never the network.

**Report time in one unit.** Everything upstream — `Answer.ms`, `ToolResult.ms`,
the timings panel — is milliseconds measured with `perf_counter`. `_ms` is that
decision, made once.

    BaseAgent          settings, a logger named for the agent, `ready`, `_guard`
    └── ModelAgent     + a prompt file, a LangChain model, and two chains
        ├── SynthesisAgent, RelevanceGrader, QueryRewriter, AnswerGrader,
        │   RouterAgent          (src/agents/rag.py — the ask ladder's stages)
        └── DatasetAgent         (src/agents/dataset_agent.py — a tool loop)
    └── ToolAgent      (src/agents/tool_agent.py — needs no model of its own)

The model calls are LangChain: `prompt | model | parser`, with the prompt read
from `src/prompts/<name>.md` (see `src/agents/prompts.py`) and the model built by
`src/agents/model.py`. What LangChain buys here is the composition and the
parsers, not the abstraction — the promises above are still this package's, and
they are the reason `_text` and `_json` exist instead of callers invoking
chains directly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from typing import Any, Callable, ClassVar, TypeVar

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.agents import prompts
from src.agents.model import TolerantJson, chat_model
from src.core.config import Settings
from src.rag.llm import LlmUnavailable

T = TypeVar("T")


class BaseAgent(ABC):
    """Settings, a logger, a readiness flag, and the never-raises helper.

    Subclasses must set `name`, and `__init_subclass__` refuses the class if
    they do not. It is not decoration: it is what the logger is called, it is
    the name of the prompt file a `ModelAgent` reads, and it is what appears
    beside a swallowed failure in the log. An agent that inherited the base's
    name would log its degradations under somebody else's, which is the one
    line an operator has to go on when a stage quietly stops running.
    """

    #: Kebab-case, and unique in this package: it becomes `vec.agents.<name>`
    #: and `src/prompts/<name>.md`. Declared without a default on purpose — see
    #: `__init_subclass__`.
    name: ClassVar[str]

    #: Whether `ready` should consult the model configuration. False for an
    #: agent whose work is not a model call — `ToolAgent` overrides `ready`
    #: outright, because for it "configured" means Composio, not a model key.
    needs_model: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Intermediate classes in this package are contracts too — `ModelAgent`
        # names nothing because nothing instantiates it — so the requirement is
        # on classes that are not themselves abstract.
        if "__abstractbase__" not in vars(cls) and "name" not in vars(cls):
            raise TypeError(f"{cls.__name__} must set `name` — it names its logger")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.log = logging.getLogger(f"vec.agents.{self.name}")

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def ready(self) -> bool:
        """Would calling this agent do anything, given how it is configured?

        Configuration only. A model key that has been revoked reads as ready
        here and fails at the call, where the failure is already handled — the
        alternative is a network round trip in front of every stage, which is
        the cost this flag exists to avoid.
        """
        return self._settings.resolve_llm().ready if self.needs_model else True

    def _guard(self, what: str, call: Callable[[], T], *, default: Any = None) -> T | Any:
        """Run `call`, and turn anything it throws into `default`.

        The log line is at warning because a swallowed exception nobody can see
        is how a degraded rung becomes a mystery: the caller only knows the
        stage was unavailable, and this is the only place that still knows why.
        """
        try:
            return call()
        except Exception as error:
            self.log.warning("%s failed: %s: %s", what, type(error).__name__, error)
            return default

    @staticmethod
    def _ms(started: float) -> float:
        """Milliseconds since a `perf_counter()` reading."""
        return (time.perf_counter() - started) * 1000

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} ready={self.ready}>"


class ModelAgent(BaseAgent):
    """An agent whose work *is* a model call: a prompt file and two chains.

    The prompt is loaded in `__init__`, so an agent whose file is missing or
    malformed fails when it is constructed — at import of `AskService`, at
    startup — rather than by degrading a turn hours later. That is the one
    failure in this package that is a broken checkout rather than a provider
    having a bad minute, and it is treated differently on purpose.

    The two chain helpers differ in exactly one way, and it is the interesting
    one:

        _text       never raises; `None` means "no answer, for any reason"
        _json       the same, for a verdict — and it inherits `TolerantJson`'s
                    rule that an unparseable reply is `None`, never a default

    Defaulting a grader to `true` on a parse failure would turn every provider
    hiccup into an approval, silently, in the direction that emits answers
    rather than withholding them.

    Budgets are class-level properties rather than call arguments because they
    are a property of the *stage*, not of the question: a grader gets the
    grader's tokens and the grader's timeout on every call it will ever make,
    and a call site that could pass its own would eventually pass a different
    one.
    """

    __abstractbase__ = True

    #: Whether this agent's prompt file carries a `## Human` section — that is,
    #: whether it is a single-shot stage. `DatasetAgent` sets it False: it
    #: drives a tool loop and supplies its own user turn.
    single_shot: ClassVar[bool] = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.prompt = prompts.load(self.name)
        # Built here, not per call: `ChatPromptTemplate` parsing is cheap but
        # not free, and building it now is what makes a prompt file missing its
        # `## Human` section a startup failure rather than a failed turn.
        self.template = self.prompt.chat if self.single_shot else None

    # ---- the budget this stage runs on -----------------------------------

    @property
    def _max_tokens(self) -> int:
        return self._settings.grader_max_tokens

    @property
    def _temperature(self) -> float:
        return 0.0

    @property
    def _timeout_s(self) -> float:
        return self._settings.grader_timeout_s

    # ---- the model, and the two chains over it ---------------------------

    @property
    def model(self) -> ChatOpenAI:
        """This stage's client. Cached by budget in `src/agents/model.py`."""
        return chat_model(
            self._settings,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout_s=self._timeout_s,
        )

    def chain(self, parser: Runnable | None = None) -> Runnable:
        """`prompt | model | parser` — the composition, exposed for a caller
        that wants to stream it or bind tools to it."""
        if self.template is None:
            raise prompts.PromptError(f"{self.name} has no single-shot prompt to chain")
        return self.template | self.model | (parser or StrOutputParser())

    def _text(self, **values: object) -> str | None:
        answer = self._invoke(StrOutputParser(), values)
        return answer.strip() if isinstance(answer, str) and answer.strip() else None

    def _json(self, **values: object) -> dict[str, Any] | None:
        parsed = self._invoke(TolerantJson(), values)
        return parsed if isinstance(parsed, dict) else None

    def _invoke(self, parser: Runnable, values: dict[str, object]) -> Any:
        """One chain invocation, with every failure flattened to `None`.

        The readiness check comes first so an unconfigured deployment costs
        nothing at all — not a client, not a socket — on a path that runs this
        for every question. `LlmUnavailable` is then logged at info and
        everything else at warning, which is the difference between "this
        deployment has no key" — a configuration fact, true for every call —
        and "that call failed", which is news.
        """
        if not self.ready:
            self.log.info("unavailable: no model is configured")
            return None
        try:
            return self.chain(parser).invoke(values)
        except LlmUnavailable as error:
            self.log.info("unavailable: %s", error)
            return None
        except Exception as error:
            self.log.warning("call failed: %s: %s", type(error).__name__, error)
            return None
