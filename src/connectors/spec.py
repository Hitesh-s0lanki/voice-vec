"""What a connector is, described well enough that the UI can be generated.

Adding a connector should be a backend change and nothing else. That is the
whole reason this file exists: a `ConnectorSpec` carries enough about a service
— what it is called, what it needs, which of those are secret, and how to prove
the credentials work — that the panel can render a correct form for a connector
it has never heard of.

The alternative, a React component per service, means every new connector
touches two languages and two review cycles, and means the form and the
validation drift apart the first time somebody edits one of them.

Three kinds so far:

    tools    Composio — things the user can act through
    vector   Pinecone, Astra, pgvector — where their embeddings live
    dataset  a URL to parquet or CSV, pulled local and answered in SQL

`kind` is not decoration. It is what lets the retrieval path ask "which vector
store belongs to this user" without enumerating slugs, so a fourth vector
backend does not require editing the code that chooses between them — and it is
what keeps `dataset` out of that question entirely. A dataset is not searched
and has no backend under `src/rag/backends/`; it is queried through a tool
(docs/18-datasets.md), so the code that picks a store must never see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping

Kind = Literal["tools", "vector", "dataset"]


@dataclass(frozen=True, slots=True)
class Field:
    """One thing the user has to type.

    `secret` decides three separate behaviours and is the field that matters
    most to get right: the input renders as a password, the value is never sent
    back to the browser, and only its last four characters are kept in the
    clear for display. A field marked non-secret is stored readable, so
    anything that could be a credential must say so here.
    """

    name: str
    label: str
    secret: bool = False
    required: bool = True
    placeholder: str = ""
    help: str = ""


class ConnectorError(RuntimeError):
    """A connector's credentials do not work, phrased for the person who typed them.

    Carries a status because the reasons are genuinely different: 400 is "that
    key is wrong", 502 is "the service did not answer", and a panel that shows
    the same sentence for both sends people to re-check a key that was fine.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ConnectorSpec:
    slug: str
    name: str
    kind: Kind
    #: One line in the connectors panel, and it is a *line*: the row gives it
    #: about 250px at 0.7rem, which is roughly 45 characters. Longer text is
    #: truncated with an ellipsis rather than wrapped, so a summary written
    #: without that budget in mind gets chopped mid-word. The full explanation
    #: belongs on the fields' `help` text, which the detail view has room for.
    summary: str
    fields: tuple[Field, ...]
    # Raises `ConnectorError` when the credentials do not work. Called before
    # anything is written down: a credential that cannot answer one cheap call
    # is not worth encrypting and keeping, and finding out at connect time is
    # one clear message instead of every later action failing opaquely.
    #
    # May also *resolve* a value the user left blank, by returning the fields
    # to merge before sealing. That is not scope creep: the only way to know
    # which table on somebody's Postgres holds vectors is to look, and looking
    # is what this call already does. Returning `None` — what three of the four
    # do — means the submitted values stand as typed.
    verify: Callable[[Mapping[str, str]], Mapping[str, str] | None]
    docs_url: str = ""
    # Which field's last four characters identify this connection in the UI.
    # Defaults to the first secret field, which is right for all four today.
    hint_field: str = ""
    aliases: tuple[str, ...] = field(default=(), repr=False)

    @property
    def secret_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.secret)

    @property
    def public_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if not f.secret)

    @property
    def hint_source(self) -> str:
        if self.hint_field:
            return self.hint_field
        secrets = self.secret_fields
        return secrets[0] if secrets else ""

    def clean(self, submitted: Mapping[str, str]) -> dict[str, str]:
        """Take only the declared fields, and insist on the required ones.

        Unknown keys are dropped rather than stored. Otherwise a client could
        post arbitrary JSON into a credential blob this app later decrypts and
        hands to an SDK, which is a strange thing to allow for no benefit.
        """
        values: dict[str, str] = {}
        missing: list[str] = []

        for spec_field in self.fields:
            raw = str(submitted.get(spec_field.name, "") or "").strip()
            if not raw:
                if spec_field.required:
                    missing.append(spec_field.label)
                continue
            values[spec_field.name] = raw

        if missing:
            raise ConnectorError(f"{self.name} needs {', '.join(missing)}.")
        return values
