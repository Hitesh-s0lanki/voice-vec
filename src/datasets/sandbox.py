"""Where model-written SQL runs: local, read-only, and with the world removed.

The shape of this module was decided by a measurement rather than a preference.
DuckDB can read `hf://` parquet directly, which is the obvious design — no
copy, always current. Two facts killed it:

  **It is not realtime.** A filtered read of one 55 GB repo's file measured
  6.3 s when it touched only the narrow columns, and 75 s the moment it touched
  the wide one. The voice path budgets 200 ms for everything
  (docs/04-latency.md).

  **It cannot be sandboxed.** `SET disabled_filesystems='LocalFileSystem'` is
  what makes running generated SQL safe, and httpfs needs the local filesystem
  for its own secret storage — turning the seal on breaks every remote read in
  the same connection. Measured, not assumed: with the seal on, `hf://` fails
  with a permission error and `https://` fails with the same one.

So the two halves are separated by a file. `materialise.py` runs unsealed and
writes a local DuckDB database; this runs sealed and never reaches the network.

**Open first, then seal.** The order is the whole trick, and reversing it
produces a sandbox that cannot open anything:

    duckdb.connect(path, read_only=True)   -- needs the filesystem
    SET disabled_filesystems='LocalFileSystem'
    SET lock_configuration=true            -- and now nothing can undo it

The file is opened *as* the database rather than `ATTACH`ed into an in-memory
one, which is not a stylistic choice: a `USE ds` on the parent connection does
not carry into `connection.cursor()`, so every generated query would have had
to qualify every table, against a schema card that shows them unqualified.
Opening the file directly makes its tables the default catalogue for the parent
and every cursor taken off it. (`USE` does still work post-lock on a cursor —
this avoids needing it at all.)

Verified against DuckDB 1.5.5, every one of these fails afterwards and the
tables still answer: `read_csv('/etc/hosts')`, `read_text(...)`,
`glob(...)`, `COPY … TO`, `ATTACH`, `INSTALL`, reading any `https://` URL,
`SET disabled_filesystems=''`, and every write into the attached database —
which is opened read-only as well as sealed, so `DELETE FROM t` is refused by
the catalogue even though `DELETE` never reaches here past `sql.guard`.

**A timeout is an interrupt, not a cancelled future.** Abandoning the thread
would leave DuckDB grinding through a cross join nobody is waiting for, on a
process that also serves audio. `interrupt()` stops the query inside the engine.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

log = logging.getLogger("vec.datasets.sandbox")

class SandboxUnavailable(RuntimeError):
    """The materialised file is missing or unreadable — the dataset needs rebuilding."""


@dataclass(frozen=True, slots=True)
class Result:
    """What one query produced, bounded on every axis a prompt is bounded on."""

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    #: True when the cap cut the result short, so an answer can say "at least".
    #: The difference between "12 rows" and "the first 12 of more" is the
    #: difference between a fact and a wrong fact.
    truncated: bool = False
    ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.rows)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def for_model(self, *, budget: int = 4000) -> str:
        """The result as the calling agent reads it back.

        JSON records rather than a rendered table: the consumer is a model that
        will quote a value into a sentence, and a column-aligned table invites
        it to quote the padding. Rows are dropped from the end rather than the
        text being cut, so what arrives is always parseable.
        """
        records = self.to_dicts()
        kept: list[dict[str, Any]] = []
        used = 0

        for record in records:
            rendered = json.dumps(record, default=str, ensure_ascii=False)
            if used + len(rendered) > budget and kept:
                break
            kept.append(record)
            used += len(rendered) + 1

        payload: dict[str, Any] = {
            "columns": list(self.columns),
            "rows": kept,
            "returned": len(kept),
        }
        if self.truncated or len(kept) < len(records):
            payload["truncated"] = True
            payload["note"] = (
                "More rows matched than are shown. Say so rather than reporting this as the total."
            )
        return json.dumps(payload, default=str, ensure_ascii=False)


class Sandbox:
    """Sealed connections to materialised datasets, one per file, reused.

    Reused because opening one costs an attach and three configuration
    statements, and a conversation asks several questions of the same dataset.
    Bounded because each holds a file handle and a memory limit, and the number
    of datasets across all signed-in users has no ceiling.
    """

    def __init__(self, *, max_open: int = 8, memory_limit: str = "512MB", threads: int = 2) -> None:
        self._max_open = max_open
        self._memory_limit = memory_limit
        self._threads = threads
        self._open: dict[str, duckdb.DuckDBPyConnection] = {}
        self._lock = threading.Lock()

    # ---- connections -----------------------------------------------------

    def connection(self, path: str | Path) -> duckdb.DuckDBPyConnection:
        resolved = str(Path(path).resolve())

        with self._lock:
            existing = self._open.get(resolved)
            if existing is not None:
                return existing

        # Built outside the lock: an attach reads a file and a slow disk should
        # not hold every other dataset's queries behind it.
        connection = self._seal(resolved)

        with self._lock:
            racer = self._open.get(resolved)
            if racer is not None:
                connection.close()
                return racer
            while len(self._open) >= self._max_open:
                _, evicted = self._open.popitem()
                _close(evicted)
            self._open[resolved] = connection
            return connection

    def _seal(self, path: str) -> duckdb.DuckDBPyConnection:
        """Open the file, then take the world away. Order matters — see above."""
        if not Path(path).exists():
            raise SandboxUnavailable(f"{Path(path).name} has not been built yet.")

        try:
            connection = duckdb.connect(path, read_only=True)
        except Exception as error:
            raise SandboxUnavailable(str(error)) from error

        try:
            connection.execute(f"SET memory_limit = '{self._memory_limit}'")
            connection.execute(f"SET threads = {int(self._threads)}")
            connection.execute("SET allow_community_extensions = false")
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            connection.execute("SET disabled_filesystems = 'LocalFileSystem'")
            connection.execute("SET lock_configuration = true")
        except Exception as error:
            _close(connection)
            raise SandboxUnavailable(str(error)) from error

        return connection

    def forget(self, path: str | Path) -> None:
        """Drop the connection to a file that is about to be rebuilt or deleted.

        Rebuilding under an open read-only handle leaves the sandbox serving
        the previous contents, and on Windows would refuse to replace the file
        at all.
        """
        resolved = str(Path(path).resolve())
        with self._lock:
            connection = self._open.pop(resolved, None)
        _close(connection)

    def close(self) -> None:
        with self._lock:
            connections = list(self._open.values())
            self._open.clear()
        for connection in connections:
            _close(connection)

    # ---- running ---------------------------------------------------------

    def run(
        self,
        path: str | Path,
        sql: str,
        *,
        max_rows: int = 200,
        timeout_s: float = 10.0,
    ) -> Result:
        """One guarded statement. Raises `duckdb.Error` for the repair round to read.

        `sql` must already have been through `sql.guard`. This does not re-check
        it: two places deciding what is allowed is two places to disagree, and
        the guard is the one that is tested.
        """
        connection = self.connection(path)
        # A cursor per query, because a session may run one while a background
        # refresh reads the same file, and a DuckDB connection is not safe to
        # use from two threads at once.
        cursor = connection.cursor()
        timer = threading.Timer(timeout_s, _interrupt, args=(cursor,))
        started = time.perf_counter()

        try:
            timer.start()
            cursor.execute(sql)
            columns = tuple(d[0] for d in (cursor.description or ()))
            # One more than the cap, so "there were more" is measured rather
            # than inferred from having filled the page exactly.
            fetched = cursor.fetchmany(max_rows + 1)
        finally:
            timer.cancel()
            cursor.close()

        ms = (time.perf_counter() - started) * 1000
        truncated = len(fetched) > max_rows

        return Result(
            columns=columns,
            rows=tuple(tuple(row) for row in fetched[:max_rows]),
            truncated=truncated,
            ms=ms,
        )


def _interrupt(cursor: duckdb.DuckDBPyConnection) -> None:
    try:
        cursor.interrupt()
    except Exception:  # already finished, already closed — both are fine
        pass


def _close(connection: duckdb.DuckDBPyConnection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except Exception as error:
        log.debug("closing a sandbox connection failed: %s", error)
