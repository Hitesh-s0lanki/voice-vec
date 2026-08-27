"""Attaching connectors to an account: verify, seal, store, hand back out.

This is the one place credentials are in plaintext, and it is deliberately
small. Everything above it deals in `Attached` — a description with a hint on
it — and everything below deals in ciphertext. `credentials()` is the single
door between the two, and the only caller that should ever open it is something
about to build a client.

The order in `connect` is the part worth not rearranging:

    clean  →  verify  →  seal  →  store

Verifying before storing means a credential that cannot answer one cheap call
is never written down. Sealing before storing means the plaintext never reaches
a query. Doing it the other way round would be one `except` away from a column
full of keys that do not work, in the clear.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Mapping

from src.connectors.crypto import Sealed, SecretsUnavailable, get_sealed, hint
from src.connectors.registry import SPECS, get_spec
from src.connectors.spec import ConnectorError, ConnectorSpec
from src.connectors.store import ConnectorStore, get_connector_store

log = logging.getLogger("vec.connectors")


class Attached:
    """A connector as everything above this layer sees it: no secrets."""

    __slots__ = ("spec", "connected", "hints", "connected_at", "updated_at", "stale")

    def __init__(
        self,
        spec: ConnectorSpec,
        *,
        connected: bool = False,
        hints: dict[str, str] | None = None,
        connected_at=None,
        updated_at=None,
        stale: bool = False,
    ) -> None:
        self.spec = spec
        self.connected = connected
        self.hints = hints or {}
        self.connected_at = connected_at
        self.updated_at = updated_at
        # A row that will not decrypt: the master key was rotated. "Connected"
        # would be a lie and "not connected" loses the fact that there is a row
        # to replace, so it is its own flag.
        self.stale = stale


class ConnectorService:
    def __init__(self, store: ConnectorStore, sealed: Sealed) -> None:
        self._store = store
        self._sealed = sealed

    @property
    def configured(self) -> bool:
        """Whether this deployment can hold credentials at all.

        False means no `COMPOSIO_ENCRYPTION_KEY`, which is a server problem and
        not the signed-in user's — the panel says so rather than offering a
        form that would drop their secret on the floor.
        """
        return self._sealed.configured and self._store.configured

    # ---- reading --------------------------------------------------------

    def list(self, user_id: str) -> list[Attached]:
        """Every connector this app knows about, with the user's state on it.

        The full catalogue rather than only what is attached, because the panel
        renders both from one call and "what could I connect" is the question
        somebody with nothing connected is asking.
        """
        attached = {row.connector: row for row in self._rows(user_id)}
        return [self._describe(spec, attached.get(spec.slug)) for spec in SPECS]

    def get(self, user_id: str, slug: str) -> Attached | None:
        spec = get_spec(slug)
        if spec is None:
            return None
        return self._describe(spec, self._store.get(user_id, spec.slug))

    def _rows(self, user_id: str):
        if not self._store.configured:
            return []
        try:
            return self._store.list(user_id)
        except Exception as error:
            log.warning("could not list connectors for %s: %s", user_id, error)
            return []

    def _describe(self, spec: ConnectorSpec, row) -> Attached:
        if row is None:
            return Attached(spec)

        # Readability is what "connected" means. A row sealed under a rotated
        # master key cannot build a client, so reporting it as connected would
        # show a green row above a panel where every action fails.
        try:
            readable = self._sealed.open_map(row.credentials) is not None
        except SecretsUnavailable:
            readable = False

        return Attached(
            spec,
            connected=readable,
            hints=row.hints,
            connected_at=row.created_at,
            updated_at=row.updated_at,
            stale=not readable,
        )

    # ---- the one door to plaintext --------------------------------------

    def credentials(self, user_id: str, slug: str) -> dict[str, str] | None:
        """The decrypted credentials, for something about to build a client.

        None means every way this can fail to be usable — no such connector,
        not attached, or attached under a master key this process no longer
        has. They are one answer because they need one response: connect it.
        """
        spec = get_spec(slug)
        if spec is None or not user_id or not self.configured:
            return None

        row = self._store.get(user_id, spec.slug)
        if row is None:
            return None

        try:
            return self._sealed.open_map(row.credentials)
        except SecretsUnavailable as error:
            log.warning("cannot open credentials: %s", error)
            return None

    # ---- writing --------------------------------------------------------

    def connect(self, user_id: str, slug: str, submitted: Mapping[str, str]) -> Attached:
        """Attach a connector after proving its credentials work."""
        spec = get_spec(slug)
        if spec is None:
            raise ConnectorError(f"There is no {slug!r} connector.", status_code=404)
        if not self.configured:
            raise ConnectorError(
                "This server cannot store credentials — COMPOSIO_ENCRYPTION_KEY is unset.",
                status_code=503,
            )

        values = spec.clean(submitted)
        spec.verify(values)  # raises ConnectorError; nothing written yet

        try:
            sealed = self._sealed.seal_map(values)
        except SecretsUnavailable as error:
            raise ConnectorError(str(error), status_code=503) from error

        row = self._store.save(
            user_id, spec.slug, credentials=sealed, hints=self._hints(spec, values)
        )
        return self._describe(spec, row)

    def _hints(self, spec: ConnectorSpec, values: Mapping[str, str]) -> dict[str, str]:
        """The readable half of a credential set.

        Non-secret fields as typed — an index name and a keyspace are how a
        person recognises which connection this is — and the secret one reduced
        to four characters, which distinguishes two keys and reconstructs
        neither.
        """
        hints = {name: values[name] for name in spec.public_fields if name in values}
        source = spec.hint_source
        if source and source in values:
            hints[f"{source}_hint"] = hint(values[source])
        return hints

    def disconnect(self, user_id: str, slug: str) -> bool:
        spec = get_spec(slug)
        if spec is None:
            return False
        return self._store.delete(user_id, spec.slug)


@lru_cache
def get_connector_service() -> ConnectorService:
    return ConnectorService(get_connector_store(), get_sealed())
