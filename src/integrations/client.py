"""A Composio client per user, built from the key that user connected.

There is no server-side Composio account behind this app. Each signed-in person
connects their own — through the connector framework in `src/connectors/`,
which is where the key is verified, sealed and stored — so "the Composio
client" is not a singleton. It is a function of who is asking, and `for_user`
is the only way to get one.

That is the whole security property of this module. A caller cannot obtain a
client without naming a user, and the key that builds it is read from that
user's row and nowhere else. There is no ambient credential to fall back to and
no default to leak into somebody else's request.

Composio still holds the third-party tokens: this app never sees a Google
refresh token or a Slack bot token, only the Composio key that lets it ask for
consent URLs on that user's behalf. None of it is on the voice path, so a slow
call costs a rail panel and never a spoken turn.

Clients are cached because building one opens an HTTP client, and the panel
makes several calls per open. The cache is keyed by the *sealed* key as well as
the user id, so re-connecting with a different key does not keep serving the
old one — a stale client here would be an app still acting on a credential its
owner has revoked.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from functools import lru_cache

from composio import Composio

from src.connectors.service import ConnectorService, get_connector_service
from src.connectors.store import ConnectorStore, get_connector_store
from src.core.config import Settings, get_settings

log = logging.getLogger("vec.integrations")


class ComposioUnavailable(RuntimeError):
    """Composio is not answering, or the key it was handed does not work."""


class NotConnected(RuntimeError):
    """This user has not connected a Composio account yet.

    Distinct from `ComposioUnavailable` because the two need opposite answers:
    one is "try again", the other is "connect Composio first", and a panel that
    conflates them tells people to retry something that will never work.
    """


class ComposioClients:
    """Builds and caches one Composio SDK per connected user."""

    def __init__(
        self,
        connectors: ConnectorService,
        store: ConnectorStore,
        settings: Settings,
    ) -> None:
        self._connectors = connectors
        self._store = store
        self._settings = settings
        # user_id → (sealed key it was built from, sdk). Bounded and LRU: a
        # client is cheap to rebuild and holding one per user who ever signed
        # in is a slow leak nobody would notice until it mattered.
        self._cache: OrderedDict[str, tuple[str, Composio]] = OrderedDict()

    @property
    def configured(self) -> bool:
        """Whether this deployment can hold a Composio key at all.

        False means no `COMPOSIO_ENCRYPTION_KEY`, which is a server problem and
        not the signed-in user's — the panel says so rather than offering a
        form that would drop their secret on the floor.
        """
        return self._connectors.configured

    def build(self, api_key: str) -> Composio:
        """A client for a key that has not been stored yet.

        Used to *verify* a key at connect time, before anything is written. A
        key that cannot list a single toolkit is not worth encrypting and
        keeping.
        """
        try:
            return Composio(
                api_key=api_key,
                timeout=int(self._settings.composio_timeout_s),
            )
        except Exception as error:
            raise ComposioUnavailable(str(error)) from error

    def for_user(self, user_id: str) -> Composio:
        """This user's Composio, or an exception naming which problem it is.

        Raises `NotConnected` when there is no row, and also when there is one
        that will not decrypt — from the caller's point of view a key sealed
        under a rotated master key is exactly as usable as no key at all, and
        the fix is the same: connect Composio again.
        """
        if not user_id:
            raise NotConnected("Sign in to use Composio.")
        if not self.configured:
            raise ComposioUnavailable(
                "This server cannot store Composio keys — COMPOSIO_ENCRYPTION_KEY is unset."
            )

        account = self._store.get(user_id, "composio")
        if account is None:
            raise NotConnected("Connect your Composio account first.")

        cached = self._cache.get(user_id)
        if cached and cached[0] == account.credentials:
            self._cache.move_to_end(user_id)
            return cached[1]

        credentials = self._connectors.credentials(user_id, "composio")
        api_key = (credentials or {}).get("api_key")

        if not api_key:
            # Sealed under a master key this process no longer has. Drop the
            # cached client so a reconnect is picked up immediately.
            self._cache.pop(user_id, None)
            raise NotConnected("Your stored Composio key could not be read — reconnect it.")

        sdk = self.build(api_key)

        self._cache[user_id] = (account.credentials, sdk)
        self._cache.move_to_end(user_id)
        while len(self._cache) > self._settings.composio_client_cache:
            self._cache.popitem(last=False)

        return sdk

    def forget(self, user_id: str) -> None:
        """Drop a cached client — on disconnect, or on a key change.

        Called explicitly rather than relied upon: the sealed-key comparison in
        `for_user` already catches a changed key, but a *disconnected* user has
        no row to compare against, and a client left in this dict is a live
        credential sitting in memory for no reason.
        """
        self._cache.pop(user_id, None)

    @property
    def callback_url(self) -> str:
        return self._settings.composio_callback_url


@lru_cache
def get_clients() -> ComposioClients:
    return ComposioClients(get_connector_service(), get_connector_store(), get_settings())
