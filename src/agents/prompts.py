"""The prompt files, read once and turned into LangChain templates.

Every agent's system message lives in `src/prompts/<name>.md` (see that
directory's README). This is the only thing that reads them.

**A missing prompt raises, it does not degrade.** Everywhere else in this
package a failure becomes `None` and the caller falls back — because everywhere
else the failure is a provider's, at runtime, with somebody waiting. A prompt
file that is absent or has no `## System` section is a broken checkout, and the
place to find that out is startup: `ModelAgent.__init__` loads its file, so an
agent whose prompt is missing cannot be constructed at all.

**Mustache, and always triple-braced.** `{{query}}` HTML-escapes what it
substitutes — a passage containing `<` or `&` would reach the model as `&lt;`,
silently changing the text it is grading — so every variable in every file is
written `{{{query}}}`. `_check` enforces that at load time rather than trusting
it, because the failure is invisible in the output and only shows up as a
slightly worse grade.

Mustache rather than f-string templating because these prompts are full of JSON
braces: `{"supported": true}` is a literal in three of the files, and f-string
templating would need every one of them doubled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

#: `src/agents/prompts.py` → `src/` → `src/prompts/`. Resolved from this file
#: rather than the working directory, so a server started from anywhere reads
#: the same files.
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

_HEADING = re.compile(r"^##\s+(system|human)\s*$", re.IGNORECASE | re.MULTILINE)

#: A variable written with two braces instead of three. Matches `{{ name }}`
#: but not `{{{ name }}}`, and not a JSON literal like `{"keep": []}`.
_ESCAPED = re.compile(r"(?<!\{)\{\{\s*[\w.]+\s*\}\}(?!\})")


class PromptError(RuntimeError):
    """A prompt file that cannot be used. Raised at construction, never at a turn."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One agent's prompt file, in the two shapes its agents need.

    `chat` is for a single-shot stage — prompt, model, parser, one answer.
    `system` is for the tool-loop agent, which supplies its own user turn and
    hands LangChain a rendered system string instead of a template.
    """

    name: str
    system_template: str
    human_template: str

    @property
    def chat(self) -> ChatPromptTemplate:
        if not self.human_template:
            raise PromptError(f"prompts/{self.name}.md has no `## Human` section")
        return ChatPromptTemplate.from_messages(
            [("system", self.system_template), ("human", self.human_template)],
            template_format="mustache",
        )

    def system(self, **values: object) -> str:
        """The system message with its variables filled in."""
        template = ChatPromptTemplate.from_messages(
            [("system", self.system_template)], template_format="mustache"
        )
        return str(template.format_messages(**values)[0].content)


@lru_cache(maxsize=32)
def load(name: str) -> Prompt:
    """`prompts/<name>.md`, parsed and cached. Raises if it is not usable."""
    path = PROMPTS / f"{name}.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PromptError(f"no prompt file at {path}") from error

    sections = _sections(text)
    if "system" not in sections:
        raise PromptError(f"{path} has no `## System` section")

    _check(path, sections)
    return Prompt(
        name=name,
        system_template=sections["system"],
        human_template=sections.get("human", ""),
    )


def _sections(text: str) -> dict[str, str]:
    """The `## System` / `## Human` bodies.

    Everything before the first heading is a note to whoever edits the file and
    is dropped here — which is the whole reason the format has headings at all.
    A prompt directory nobody can annotate ends up annotated in the prompt.
    """
    found: dict[str, str] = {}
    matches = list(_HEADING.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found[match.group(1).lower()] = text[match.end() : end].strip()
    return found


def _check(path: Path, sections: dict[str, str]) -> None:
    for section, body in sections.items():
        escaped = _ESCAPED.findall(body)
        if escaped:
            raise PromptError(
                f"{path} `## {section.title()}` interpolates {', '.join(escaped)} with two "
                "braces, which HTML-escapes the value. Use three: {{{name}}}"
            )
