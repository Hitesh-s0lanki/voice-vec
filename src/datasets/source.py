"""What a dataset URL actually points at, resolved before anything is pulled.

A person pastes `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`. That is
not a file, it is a repository of 28 of them, and nothing downstream can pull,
measure or query "a repository". So this module is the one place that turns a
URL somebody typed into a concrete list of readable files with SQL identifiers
attached — and the one place that refuses a URL it cannot turn into that.

Three forms are accepted, because they are the three ways people actually
reference this data:

    .../datasets/org/name                          the whole repo
    .../datasets/org/name/tree/main/validation     one directory in it
    .../datasets/org/name/blob/main/val/hin.parquet  one file
    https://host/path/sales.parquet                a file anywhere

**Resolution is a read of the HF API, not a guess at a path.** The alternative —
constructing `resolve/main/<something>` from the repo name — produces a URL that
is well-formed and 404s, and the failure surfaces minutes later inside a pull
rather than immediately with a message about the URL that was typed.

**The table cap is announced, never silent.** A repo with more files than
`max_tables` is resolved to a prefix of them and `truncated` says so, which the
card then says out loud. A dataset that quietly describes half of itself is
worse than one that refuses: the agent routes on the half it was told about and
nothing anywhere reads as wrong.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

#: The file types DuckDB can read straight off a URL with no conversion step.
#: Anything else is rejected at add time rather than at pull time — a dataset
#: of `.arrow` shards should fail while the person who typed it is still
#: looking at the form.
FORMATS: dict[str, str] = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".jsonl": "json",
    ".ndjson": "json",
}

HF_HOST = "huggingface.co"
_HF_API = "https://huggingface.co/api/datasets/{repo}"
_HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

#: How long to wait on the HF API. This runs while somebody is watching a form,
#: so it is short and a timeout is a clear message rather than a spinner.
LIST_TIMEOUT_S = 15.0

_IDENT = re.compile(r"[^a-z0-9_]+")

#: Hugging Face's auto-converted parquet is written as shards —
#: `plain_text/train-00000-of-00001.parquet`, sometimes with a content hash
#: after it. The shard coordinates are storage detail, and carrying them into
#: the table name is not cosmetic: they are what a model has to type in every
#: query, and `SELECT ... FROM train_00000_of_00001_a09b74b3ef9c3b56` is a name
#: it gets wrong. Stripping them leaves `train`, which is what the split is
#: actually called everywhere else.
_SHARD = re.compile(r"^(?P<name>.+?)[-_]\d{4,}[-_]of[-_]\d{4,}(?:[-_][0-9a-f]{6,})?$")


class SourceError(ValueError):
    """A URL this app cannot turn into readable tables, phrased for whoever typed it."""


@dataclass(frozen=True, slots=True)
class Table:
    """One file, and the SQL name it will answer to.

    `name` is what the model writes in a query, so it is derived from the
    filename rather than assigned an index: `hinval` is something an agent can
    connect to "the Hindi validation split" from the card, and `t3` is not.
    """

    name: str
    url: str
    format: str
    #: The path as it appears in the repo, for the card. Not the URL — that
    #: carries a revision and would date the description.
    path: str = ""


@dataclass(frozen=True, slots=True)
class Source:
    """A resolved dataset: where it came from, and what can be read out of it."""

    slug: str
    kind: str  # "hf" | "file"
    #: What to show a human. Never carries a query string or a token.
    location: str
    url: str
    tables: tuple[Table, ...] = ()
    #: Files found beyond `max_tables`. Zero, or a number the card must state.
    truncated: int = 0
    revision: str = ""

    @property
    def empty(self) -> bool:
        return not self.tables


def normalise(url: str) -> str:
    """The URL in the one form the rest of this module understands.

    Two shorthands are accepted because they are what people actually paste:
    the `hf://` scheme DuckDB uses, and the bare `owner/name` a paper cites.
    """
    raw = (url or "").strip()
    if not raw:
        raise SourceError("Paste a dataset URL first.")

    if raw.startswith("hf://datasets/"):
        return "https://huggingface.co/datasets/" + raw[len("hf://datasets/") :]
    if "://" not in raw and re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        return f"https://huggingface.co/datasets/{raw}"
    return raw


def slug_for(url: str) -> str:
    """The dataset id a URL will be stored under, without touching the network.

    `resolve()` derives the same id, and both go through here so they cannot
    drift. That matters because the id is the join between two tables: a
    dataset attached through the Connectors panel is a row in
    `connector_accounts` keyed by slug and a row in `agent_datasets` keyed by
    id, and disconnecting has to find the second from the first. Re-resolving
    to learn it would put an HTTP call to Hugging Face inside a delete.
    """
    parsed = urlparse(normalise(url))
    parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]

    if parsed.netloc.lower().endswith(HF_HOST):
        if len(parts) < 2 or parts[0] != "datasets":
            raise SourceError(
                "That looks like a Hugging Face URL but not a dataset one — "
                "it should read huggingface.co/datasets/<owner>/<name>."
            )
        owned = len(parts) > 2 and parts[2] not in ("tree", "blob", "resolve")
        repo = f"{parts[1]}/{parts[2]}" if owned else parts[1]
        rest = parts[3:] if owned else parts[2:]
        prefix = "/".join(rest[2:]) if rest and rest[0] in ("tree", "blob", "resolve") else ""
        return _slug(repo if not prefix else f"{repo}-{prefix}")

    return _slug(posixpath.basename(parsed.path) or "data")


def resolve(url: str, *, max_tables: int = 12, timeout_s: float = LIST_TIMEOUT_S) -> Source:
    """A URL as typed → the files behind it. Raises `SourceError`, never returns empty."""
    raw = normalise(url)
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise SourceError("A dataset URL has to be http or https.")

    if parsed.netloc.lower().endswith(HF_HOST):
        return _huggingface(raw, parsed.path, max_tables=max_tables, timeout_s=timeout_s)
    return _direct(raw, parsed.path)


# ---- Hugging Face --------------------------------------------------------


def _huggingface(url: str, path: str, *, max_tables: int, timeout_s: float) -> Source:
    parts = [unquote(p) for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "datasets" or len(parts) < 2:
        raise SourceError(
            "That looks like a Hugging Face URL but not a dataset one — "
            "it should read huggingface.co/datasets/<owner>/<name>."
        )

    # The canonical datasets predate owners and are addressed by bare name —
    # `huggingface.co/datasets/imdb`. Requiring an owner rejected a URL the
    # site itself still serves.
    owned = len(parts) > 2 and parts[2] not in ("tree", "blob", "resolve")
    repo = f"{parts[1]}/{parts[2]}" if owned else parts[1]
    rest = parts[3:] if owned else parts[2:]
    revision, prefix = "main", ""

    # /tree/<rev>/<dir> and /blob/<rev>/<file> are the two forms the site's own
    # links use, and both mean "narrow this to a subpath".
    if rest and rest[0] in ("tree", "blob", "resolve"):
        revision = rest[1] if len(rest) > 1 else "main"
        prefix = "/".join(rest[2:])

    listing = _list_files(repo, timeout_s=timeout_s)
    if not listing:
        raise SourceError(
            f"{repo} has no files this app can read. It may be gated, private, "
            "or hold no parquet, CSV or JSONL."
        )

    if prefix:
        # A blob URL names one file exactly; a tree URL names a directory.
        candidates = [p for p in listing if p == prefix or p.startswith(prefix.rstrip("/") + "/")]
        if not candidates:
            raise SourceError(f"Nothing readable under {prefix} in {repo}.")
    else:
        candidates = listing

    kept, dropped = candidates[:max_tables], max(0, len(candidates) - max_tables)
    tables = _tables(
        [
            (
                p,
                _HF_FILE.format(repo=repo, revision=revision, path=p),
                FORMATS[_suffix(p)],
            )
            for p in kept
        ]
    )

    return Source(
        slug=_slug(repo if not prefix else f"{repo}-{prefix}"),
        kind="hf",
        location=repo if not prefix else f"{repo} · {prefix}",
        url=url,
        tables=tables,
        truncated=dropped,
        revision=revision,
    )


def _list_files(repo: str, *, timeout_s: float) -> list[str]:
    """Readable files in the repo, in repository order.

    Repository order rather than sorted, because it is the order the dataset's
    author laid the splits out in and a truncated prefix of it is more likely
    to be the interesting half than a prefix of the alphabet would be.
    """
    try:
        response = httpx.get(
            _HF_API.format(repo=repo), timeout=timeout_s, follow_redirects=True
        )
    except httpx.HTTPError as error:
        raise SourceError(f"Could not reach Hugging Face: {type(error).__name__}.") from error

    if response.status_code in (401, 403, 404):
        # Hugging Face answers 401 for a repository you cannot see, and it does
        # not distinguish "does not exist" from "exists and is private" — that
        # is the point of the response. So the message must not either: saying
        # "gated or private" for a typo sends somebody looking for an access
        # button that was never the problem.
        raise SourceError(
            f"Could not read {repo} on Hugging Face — check the name, or it may be "
            "gated or private. This app only reads public datasets."
        )
    if response.status_code >= 400:
        raise SourceError(f"Hugging Face answered {response.status_code} for {repo}.")

    try:
        siblings = response.json().get("siblings") or []
    except ValueError as error:
        raise SourceError("Hugging Face returned something that was not JSON.") from error

    return [
        name
        for entry in siblings
        if (name := str(entry.get("rfilename") or "")) and _suffix(name) in FORMATS
    ]


# ---- a plain URL ---------------------------------------------------------


def _direct(url: str, path: str) -> Source:
    suffix = _suffix(path)
    if suffix not in FORMATS:
        raise SourceError(
            "That URL does not end in a file this app can read. "
            f"Supported: {', '.join(sorted(FORMATS))}."
        )

    name = posixpath.basename(path) or "data"
    return Source(
        slug=_slug(name),
        kind="file",
        location=f"{urlparse(url).netloc}{path}",
        url=url,
        tables=_tables([(name, url, FORMATS[suffix])]),
    )


# ---- naming --------------------------------------------------------------


def _suffix(path: str) -> str:
    lowered = path.lower()
    # `.parquet.gz` and friends: the compression is DuckDB's problem, the
    # format is ours, so the meaningful suffix is the one before it.
    for compressed in (".gz", ".zst", ".bz2"):
        if lowered.endswith(compressed):
            lowered = lowered[: -len(compressed)]
    return posixpath.splitext(lowered)[1]


def _tables(entries: list[tuple[str, str, str]]) -> tuple[Table, ...]:
    """Files → tables, with names that are unique, legal unquoted, and *guessable*.

    Two passes, because the right name for a file depends on what the other
    files are called. `gsm8k` ships `main/test-00000-of-00001.parquet` and
    `socratic/test-00000-of-00001.parquet`: both want to be `test`, and the
    naive fix — a numeric suffix — gives `test` and `test_1`, which tells a
    model writing SQL nothing about which is which. The directory is right
    there and it is the name of the config, so a collision is resolved by
    qualifying with it: `main_test` and `socratic_test`.

    The numeric suffix survives as a last resort, for two files that genuinely
    collide on directory *and* stem.
    """
    stems = [(path, url, fmt, _stem(path)) for path, url, fmt in entries]
    clashes = {stem for stem in (s for _, _, _, s in stems) if [x[3] for x in stems].count(stem) > 1}

    seen: dict[str, int] = {}
    tables: list[Table] = []

    for path, url, fmt, stem in stems:
        base = stem
        if stem in clashes:
            parent = posixpath.basename(posixpath.dirname(path))
            if parent:
                base = _ident(f"{parent}_{stem}")

        count = seen.get(base, 0)
        seen[base] = count + 1
        tables.append(
            Table(name=base if not count else f"{base}_{count}", url=url, format=fmt, path=path)
        )

    return tuple(tables)


def _stem(path: str) -> str:
    """The filename as a table name, with shard coordinates removed."""
    raw = posixpath.splitext(posixpath.basename(path))[0]
    match = _SHARD.match(raw)
    return _ident(match.group("name") if match else raw)


def _ident(text: str) -> str:
    cleaned = _IDENT.sub("_", text.lower()).strip("_")
    if not cleaned:
        return "data"
    # A leading digit makes an identifier that has to be quoted everywhere it
    # appears, including in SQL a model writes from the card.
    return cleaned if not cleaned[0].isdigit() else f"t_{cleaned}"


def _slug(text: str) -> str:
    return _IDENT.sub("-", text.lower().replace("/", "-")).strip("-")[:64] or "dataset"
