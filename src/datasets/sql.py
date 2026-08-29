"""The gate every model-written statement passes before a database sees it.

Two independent defences, and the reason there are two is that each one fails
differently:

  **This module** decides that a statement is a single SELECT, by parsing it
  with the same parser that will execute it. It is a *readability* check, and
  a bug here is a bug of omission — a statement type nobody thought of.

  **`sandbox.py`** removes the filesystem and locks the configuration, so a
  statement that gets past this one still cannot read `/etc/passwd`, write a
  file, attach a database, or reach the network. It is a *capability* check,
  enforced by DuckDB rather than by this code being complete.

Neither is sufficient. A parser check alone trusts an allow-list to be
exhaustive; a sandbox alone would happily run `DELETE`, which is legal, local,
and destroys the sample. Together the interesting failure needs both to be
wrong at once.

**The parse is `duckdb.extract_statements`, not a regular expression.** A regex
over SQL loses to a comment, a string literal containing a semicolon, or a
dollar-quoted block, and every one of those has been somebody's injection.
DuckDB's own parser answers the only question worth asking — *what statements
did you actually read here* — and it answers it identically to the parser that
would run them.

**A statement type of SELECT is not the same as a SELECT.** DuckDB rewrites
`PRAGMA` into a table function, so `extract_statements` reports `PRAGMA
database_list` as a SELECT — measured, not assumed. `read_text('/etc/hosts')`
and `glob('/etc/*')` are honest SELECTs too. Both are why the leading-keyword
check below exists *and* why it is not trusted to be the defence: the filesystem
seal is what makes the second pair harmless, and this check only removes a
statement form that never answers a question anybody asked.

**The row cap is applied by reading, not by rewriting.** Appending `LIMIT` to
somebody else's SQL means understanding where the top-level query ends, which
is parsing it again, badly. `fetchmany` bounds what is materialised and leaves
the statement untouched, so what runs is what was reviewed.
"""

from __future__ import annotations

import re

import duckdb

#: The only statement type that may run. Deliberately one entry rather than a
#: set of "harmless" types: EXPLAIN and PRAGMA look read-only and are how you
#: enumerate a database you were not given, and neither answers a question
#: anybody asked in English.
_ALLOWED = "SELECT"

#: Statement types worth naming in the refusal, because a model that wrote one
#: usually just needs telling to phrase it as a query.
_ADVICE: dict[str, str] = {
    "INSERT": "This is a read-only view of the dataset.",
    "UPDATE": "This is a read-only view of the dataset.",
    "DELETE": "This is a read-only view of the dataset.",
    "CREATE": "Tables cannot be created here — query the ones in the schema.",
    "DROP": "This is a read-only view of the dataset.",
    "ATTACH": "No other database can be attached.",
    "COPY": "Results cannot be written to a file.",
    "LOAD": "Extensions cannot be loaded.",
    "PRAGMA": "Use the schema you were given rather than introspecting.",
    "EXPLAIN": "Return rows rather than a query plan.",
    "CALL": "Table functions that are not part of a SELECT are not available.",
    "SET": "Configuration cannot be changed.",
    "TRANSACTION": "Transactions are not available on a read-only view.",
}


#: What a query may begin with. Every real one begins with one of these, and
#: requiring it removes `PRAGMA` — which the parser calls a SELECT — without
#: relying on an exhaustive list of the things PRAGMA can do.
_LEADS = ("SELECT", "WITH", "FROM", "VALUES", "TABLE", "(")

_COMMENTS = re.compile(r"^(?:\s|--[^\n]*\n|/\*.*?\*/)+", re.DOTALL)


class Rejected(ValueError):
    """A statement that will not run, phrased so a model can correct itself.

    The message goes back into the repair round verbatim, so it says what was
    wrong rather than that something was — "this is a read-only view" produces
    a working second attempt, "invalid SQL" produces the same query again.
    """


def clean(sql: str) -> str:
    """Strip the wrapping a model puts around SQL when asked not to.

    Fenced blocks are what actually comes back — ```sql on one line, the query,
    ``` on another — often enough that handling it here beats a stricter prompt
    that fails on the one turn the model forgets.
    """
    text = (sql or "").strip()
    if text.startswith("```"):
        _, _, rest = text.partition("\n")
        text = rest.rpartition("```")[0] if "```" in rest else rest
    return text.strip().rstrip(";").strip()


def guard(sql: str) -> str:
    """The statement as it will run, or `Rejected` with a reason.

    Returns the cleaned text rather than a boolean so there is exactly one
    string that was checked and one string that runs. Handing back a verdict
    and letting the caller run its own copy is how the two drift apart.
    """
    text = clean(sql)
    if not text:
        raise Rejected("No SQL was produced.")

    try:
        statements = duckdb.extract_statements(text)
    except Exception as error:
        # A parse failure is a real answer and belongs in the repair round —
        # it is usually a hallucinated function or an unbalanced paren, and the
        # parser's own message names it.
        raise Rejected(f"That is not valid DuckDB SQL: {_message(error)}") from error

    if not statements:
        raise Rejected("No SQL was produced.")
    if len(statements) > 1:
        raise Rejected("Write one SELECT statement, not several.")

    kind = _type_name(statements[0])
    if kind != _ALLOWED:
        advice = _ADVICE.get(kind, "Only a single SELECT may run here.")
        raise Rejected(f"{kind} is not allowed. {advice}")

    if not _lead(text).startswith(_LEADS):
        # A SELECT by type that does not read like one — PRAGMA is the whole
        # reason this branch is here.
        raise Rejected("Start the statement with SELECT, WITH or FROM.")

    return text


def _lead(text: str) -> str:
    """The statement with leading whitespace and comments removed, upper-cased.

    Comments first, because `-- fetch the rows\nPRAGMA database_list` is a
    statement whose first characters are a comment and whose first *keyword*
    is the one being refused.
    """
    stripped = _COMMENTS.sub("", text, count=1)
    return stripped.upper()


def _type_name(statement: object) -> str:
    """`StatementType.SELECT` → `"SELECT"`, without depending on the enum's repr."""
    kind = getattr(statement, "type", None)
    name = getattr(kind, "name", None) or str(kind)
    return name.rsplit(".", 1)[-1].upper()


def _message(error: Exception) -> str:
    first = str(error).strip().splitlines()
    return (first[0] if first else type(error).__name__)[:200]
