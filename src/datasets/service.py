"""The agent that understands a dataset, and the door other agents read it through.

Two responsibilities that look like one and have opposite constraints — the
same split `src/connectors/profile_service.py` makes, and for the same reason:

    add() / rebuild()   slow. Resolves a URL, pulls tens of megabytes, measures
                        every column, then asks a model what it is looking at.
                        Seconds to minutes. Runs off every request path.
    cards() / schema()  fast. Read inside a turn somebody is listening through.
                        Microseconds, from memory, or it is not usable.

They meet at a Postgres row and a file on local disk, which is what makes
"the agent understands the dataset in realtime" true: the expensive half runs
once when the dataset is attached, and every turn after that reads a string.

    add ──► resolve ──► claim(pending) ──► schedule()
                                              │  (worker thread, off the request)
                                              ▼
                            materialise ──► probe ──► narrate
                                              │
                             agent_datasets (user_id, dataset_id) + <id>.duckdb
                                              │
              ┌───────────────────────────────┼─────────────────────────┐
              ▼                               ▼                         ▼
       cards() → system prompt      schema() → the SQL writer     GET /datasets

**Adding may never break on the build.** A URL that resolves is claimed and
returns; everything after that — an unreachable host, a corrupt parquet, no
model, a dead Postgres — lands as a `failed` row carrying the reason, and the
user sees why rather than a spinner that never resolves.

**A profile is only served for a file that exists.** The row and the file are
two pieces of state and they can disagree: a deployment that redeployed with a
fresh disk has rows describing datasets it cannot query. So `queryable()`
checks the file, not the row, and a missing one schedules a rebuild rather than
being served as an understanding of nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from src.core.config import Settings, get_settings
from src.datasets import narrate
from src.datasets.materialise import build
from src.datasets.probe import observe
from src.datasets.profile import (
    STATUS_DEGRADED,
    STATUS_OK,
    DatasetProfile,
)
from src.datasets.sandbox import Sandbox
from src.datasets.source import Source, SourceError, resolve
from src.datasets.store import DatasetRow, DatasetStore, get_dataset_store

log = logging.getLogger("vec.datasets")


class DatasetError(ValueError):
    """Something the person who pasted the URL needs to read."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class DatasetService:
    def __init__(self, store: DatasetStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._sandbox = Sandbox(
            max_open=settings.dataset_sandbox_open,
            memory_limit=settings.dataset_sandbox_memory,
            threads=settings.dataset_sandbox_threads,
        )
        self._pool = ThreadPoolExecutor(
            max_workers=settings.dataset_build_workers, thread_name_prefix="dataset"
        )
        # (user_id, dataset_id) → (monotonic, row). The realtime read. Bounded
        # by eviction as well as TTL: the cost is memory per signed-in user and
        # there is no upper bound on those.
        self._cache: dict[tuple[str, str], tuple[float, DatasetRow]] = {}
        self._lock = threading.Lock()
        self._building: set[tuple[str, str]] = set()

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def configured(self) -> bool:
        return self._settings.datasets_enabled and self._store.configured

    @property
    def max_per_user(self) -> int:
        return self._settings.dataset_max_per_user

    # ---- attaching -------------------------------------------------------

    def add(self, user_id: str, url: str) -> DatasetRow:
        """Resolve the URL, claim a row, and build behind it.

        Resolution happens *synchronously* and everything after it does not.
        That line is where the errors worth showing somebody live: a typo, a
        gated repo, a URL that is not a dataset. Those are worth a red form
        field. "The third of fourteen files timed out" is not — it is a row
        that says so, next to thirteen that answer.
        """
        if not self.configured:
            raise DatasetError("Datasets are not enabled on this deployment.", 503)
        if not user_id:
            raise DatasetError("Sign in to attach a dataset.", 401)

        existing = self._store.list(user_id)
        try:
            source = resolve(
                url,
                max_tables=self._settings.dataset_max_tables,
                timeout_s=self._settings.dataset_http_timeout_s,
            )
        except SourceError as error:
            raise DatasetError(str(error)) from error

        # Checked after resolving, so re-adding a dataset that is already
        # attached replaces it rather than being refused for filling a quota it
        # is already counted in.
        if (
            len(existing) >= self._settings.dataset_max_per_user
            and source.slug not in {row.dataset_id for row in existing}
        ):
            raise DatasetError(
                f"You can attach {self._settings.dataset_max_per_user} datasets. "
                "Remove one first."
            )

        row = self._store.claim(
            user_id,
            source.slug,
            url=source.url,
            kind=source.kind,
            location=source.location,
        )
        self._forget(user_id, source.slug)
        self.schedule(user_id, source.slug, source)
        return row

    def remove(self, user_id: str, dataset_id: str) -> bool:
        """Delete the row, close the sandbox, and unlink the file.

        In that order. Closing after unlinking would leave a sealed connection
        open on a deleted inode, which on this platform keeps the bytes alive
        and on another refuses the unlink outright.

        Returns whether a *row* went, not whether a file did. A dataset removed
        while it is still building has no file yet and is removed all the same —
        which is how somebody cancels a URL they mistyped, and reporting that as
        a failure sends them to look for a dataset that is already gone.
        """
        path = self._store.delete(user_id, dataset_id)
        self._forget(user_id, dataset_id)
        if path:
            self._sandbox.forget(path)
            Path(path).unlink(missing_ok=True)
        return path is not None

    def schedule(self, user_id: str, dataset_id: str, source: Source | None = None) -> None:
        """Build this dataset soon, off whatever path asked for it.

        De-duplicated: a panel polling while a build runs gets one build. Without
        this, "rebuild when stale" turns every poll into another download.
        """
        if not self.configured or not user_id:
            return

        key = (user_id, dataset_id)
        with self._lock:
            if key in self._building:
                return
            self._building.add(key)

        try:
            self._pool.submit(self._run, user_id, dataset_id, source)
        except RuntimeError:
            # The pool is shut down — the process is going away.
            with self._lock:
                self._building.discard(key)

    def _run(self, user_id: str, dataset_id: str, source: Source | None) -> None:
        try:
            self.rebuild(user_id, dataset_id, source)
        except Exception as error:  # a worker thread may not die loudly
            log.warning("building %s for %s failed: %s", dataset_id, user_id, error)
            try:
                self._store.fail(user_id, dataset_id, str(error))
            except Exception:
                log.debug("could not record the failure either")
        finally:
            with self._lock:
                self._building.discard((user_id, dataset_id))

    def rebuild(self, user_id: str, dataset_id: str, source: Source | None = None) -> DatasetRow | None:
        """Pull, measure, narrate, store. The slow half, in full.

        Synchronous and safe to call directly — a test, a script and the
        explicit rebuild endpoint all want to wait for the answer.
        """
        row = self._store.get(user_id, dataset_id)
        if row is None:
            log.info("dataset %s was removed while it was being built", dataset_id)
            return None

        if source is None:
            source = resolve(
                row.url,
                max_tables=self._settings.dataset_max_tables,
                timeout_s=self._settings.dataset_http_timeout_s,
            )

        path = self.path_for(user_id, dataset_id)
        # A rebuild replaces the file the sandbox may be holding open, so the
        # handle is dropped first. `materialise` writes to staging and moves,
        # so a query already in flight finishes against the old contents.
        self._sandbox.forget(path)

        built = build(
            source,
            path,
            sample_rows=self._settings.dataset_sample_rows,
            column_budget_bytes=self._settings.dataset_column_budget_mb * 1024 * 1024,
            table_timeout_s=self._settings.dataset_table_timeout_s,
            http_timeout_s=self._settings.dataset_http_timeout_s,
        )

        observation = observe(source, built)
        understanding = None
        try:
            understanding = narrate.describe(observation, settings=self._settings)
        except Exception as error:
            # The measurement is the profile. A model that will not answer
            # costs the routing hints and nothing else.
            log.info("could not name %s: %s", dataset_id, error)

        profile = DatasetProfile(
            dataset_id=dataset_id, observation=observation, understanding=understanding
        )
        status = STATUS_DEGRADED if observation.notes else STATUS_OK

        stored = self._store.finish(
            user_id,
            dataset_id,
            status=status,
            profile=profile.to_json(),
            card=profile.card(budget=self._settings.dataset_card_chars),
            schema_card=profile.schema(budget=self._settings.dataset_schema_chars),
            local_path=str(path),
            rows=observation.rows,
            bytes_=built.bytes,
            error="",
        )
        if stored is not None:
            self._remember(user_id, dataset_id, stored)
        log.info(
            "built %s: %d rows across %d tables in %.1fs",
            dataset_id,
            observation.rows,
            len(observation.tables),
            built.ms / 1000,
        )
        return stored

    # ---- realtime reads --------------------------------------------------

    def get(self, user_id: str | None, dataset_id: str) -> DatasetRow | None:
        """One dataset's row, from memory where possible. Never raises.

        A stale row is served anyway and a rebuild is scheduled behind it: a
        slightly old description of a pinned dataset revision is worth far more
        to the turn in flight than a blank one.
        """
        if not user_id or not dataset_id or not self.configured:
            return None

        key = (user_id, dataset_id)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self._settings.dataset_cache_ttl_s:
            return cached[1]

        try:
            row = self._store.get(user_id, dataset_id)
        except Exception as error:
            log.debug("could not read dataset %s: %s", dataset_id, error)
            return None

        if row is None:
            return None
        if row.status == STATUS_OK and row.stale(
            timedelta(hours=self._settings.dataset_ttl_hours)
        ):
            self.schedule(user_id, dataset_id)

        self._remember(user_id, dataset_id, row)
        return row

    def list(self, user_id: str | None) -> list[DatasetRow]:
        if not user_id or not self.configured:
            return []
        try:
            return self._store.list(user_id)
        except Exception as error:
            log.debug("could not list datasets: %s", error)
            return []

    def queryable(self, user_id: str | None, dataset_id: str) -> DatasetRow | None:
        """A dataset that can actually answer a query right now, or None.

        The file is checked rather than the status, because they are separate
        state and a redeploy onto a fresh disk breaks the second without
        touching the first. A row whose file is gone schedules a rebuild — the
        URL is still in the row, so this heals itself rather than needing
        somebody to notice.
        """
        row = self.get(user_id, dataset_id)
        if row is None or row.status not in (STATUS_OK, STATUS_DEGRADED):
            return None
        if not row.local_path or not Path(row.local_path).exists():
            log.info("dataset %s has no local file — rebuilding", dataset_id)
            if user_id:
                self.schedule(user_id, dataset_id)
            return None
        return row

    def cards(self, user_id: str | None) -> str:
        """Every attached dataset, for the one block that goes in a system prompt.

        Only the ones that can answer. A dataset that is still building or has
        failed is a row in a panel, not a line in a prompt: an agent told about
        a dataset it cannot query will offer to query it.
        """
        if not user_id or not self.configured:
            return ""

        blocks: list[str] = []
        for row in self.list(user_id):
            if row.status in (STATUS_OK, STATUS_DEGRADED) and row.card:
                blocks.append(row.card)
        return "\n\n".join(blocks)

    def catalogue(self, user_id: str | None) -> list[tuple[str, str]]:
        """(dataset_id, one-line title) for everything queryable.

        What the tool schema is built from, so the model choosing a dataset
        sees ids that exist rather than inventing one from the conversation.
        """
        entries: list[tuple[str, str]] = []
        for row in self.list(user_id):
            if row.status not in (STATUS_OK, STATUS_DEGRADED):
                continue
            title = (row.card.splitlines() or [""])[0].strip() or row.location
            entries.append((row.dataset_id, title))
        return entries

    # ---- paths and caching -----------------------------------------------

    def path_for(self, user_id: str, dataset_id: str) -> str:
        """Where this user's copy of this dataset lives.

        Keyed by user as well as dataset: two people who attach the same public
        URL get a file each. Sharing one would be a cache that leaks whichever
        `dataset_sample_rows` and column budget were in force for whoever
        happened to add it first — and would tie one person's rebuild to
        another person's query.
        """
        safe_user = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)[:64]
        root = Path(self._settings.dataset_dir) / safe_user
        return str(root / f"{dataset_id}.duckdb")

    def _remember(self, user_id: str, dataset_id: str, row: DatasetRow) -> None:
        with self._lock:
            if len(self._cache) >= 512:
                self._cache.pop(next(iter(self._cache)), None)
            self._cache[(user_id, dataset_id)] = (time.monotonic(), row)

    def _forget(self, user_id: str, dataset_id: str) -> None:
        with self._lock:
            self._cache.pop((user_id, dataset_id), None)

    def close(self) -> None:
        self._sandbox.close()
        self._pool.shutdown(wait=False)


@lru_cache
def get_dataset_service() -> DatasetService:
    return DatasetService(get_dataset_store(), get_settings())
