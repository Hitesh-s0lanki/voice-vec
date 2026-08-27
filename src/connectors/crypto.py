"""Encrypting a user's connector credentials before they reach Postgres.

Every connector this app knows about is reached with a credential the user
handed over: a Composio API key, a Pinecone key, an Astra token, a Postgres
DSN with a password in it. Each one is full access to something of theirs.
Storing any of them as typed would mean a database dump, a stray backup, or a
Neon console session is enough to take all of them — and none of those are the
same threat as "somebody broke into the API".

So it is encrypted here, with a master key that lives in the environment and
never in the database. The two have to be stolen separately.

Credentials are sealed as one JSON blob per connector rather than field by
field. A Pinecone key without its index name is not usable and not interesting,
so there is nothing gained by encrypting them apart — and one blob means one
place where the plaintext exists and one place to get the handling right.

Fernet rather than raw AES-GCM, and rather than anything hand-rolled. It is
AES-128-CBC with an HMAC over the ciphertext, it manages its own IV, and it
refuses to decrypt anything it did not produce. The nonce-reuse footgun that
makes AES-GCM a bad thing to wire up by hand is simply absent, and this is not
a hot path where the difference could matter — it runs once when somebody
connects and once per request they make afterwards.

`cryptography` is already a dependency: `pyjwt[crypto]` pulls it in for the
Clerk RS256 verification next door.

Rotating `COMPOSIO_ENCRYPTION_KEY` is not a migration and does not corrupt
anything. Every stored credential stops decrypting, `Sealed.open` returns None
rather than raising, and each user reconnects once. That is a deliberately
boring failure: the alternative — a server that cannot boot because one env var
changed — is worse than a panel that says "reconnect".

(The variable is still named for Composio because it is the key that was
already deployed under that name and renaming it would silently disconnect
everybody. It seals every connector.)
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import Settings, get_settings

log = logging.getLogger("vec.integrations")


class SecretsUnavailable(RuntimeError):
    """No COMPOSIO_ENCRYPTION_KEY, or one that is not a Fernet key."""


class Sealed:
    """Seals and opens one secret at a time. Holds no plaintext."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fernet: Fernet | None = None

    @property
    def configured(self) -> bool:
        """Whether there is anywhere safe to put a key.

        Checked by callers that must degrade rather than fail — the panel says
        "this deployment cannot store Composio keys" rather than offering a
        form that would drop the secret on the floor.
        """
        return bool(self._settings.composio_encryption_key)

    @property
    def _cipher(self) -> Fernet:
        if self._fernet is None:
            raw = self._settings.composio_encryption_key.strip()
            if not raw:
                raise SecretsUnavailable(
                    "COMPOSIO_ENCRYPTION_KEY is unset — generate one with "
                    "`Fernet.generate_key()` and add it to .env"
                )
            try:
                self._fernet = Fernet(raw.encode())
            except (ValueError, TypeError) as error:
                # A truncated paste, or somebody's idea of a passphrase. Say so
                # plainly: the alternative is every connect failing with a
                # padding error nobody can act on.
                raise SecretsUnavailable(
                    "COMPOSIO_ENCRYPTION_KEY is not a Fernet key — it must be "
                    "32 url-safe base64 bytes"
                ) from error
        return self._fernet

    def seal(self, secret: str) -> str:
        """Plaintext in, storable ciphertext out."""
        return self._cipher.encrypt(secret.encode()).decode()

    def seal_map(self, values: dict[str, str]) -> str:
        """A whole credential set, sealed as one blob.

        Sorted keys so the JSON is deterministic — not for security (Fernet's
        IV makes two seals of the same input differ anyway) but so a diff of
        two decrypted blobs is readable when something needs debugging.
        """
        return self.seal(json.dumps(values, sort_keys=True, separators=(",", ":")))

    def open_map(self, sealed: str) -> dict[str, str] | None:
        """The reverse, tolerating everything that can go wrong on the way.

        None covers all of it: a rotated master key, a column holding something
        that was never sealed, and — the case worth calling out — a blob that
        decrypts to valid text that is not a JSON object. All three mean the
        same thing to a caller, which is that this connector needs connecting
        again.
        """
        plain = self.open(sealed)
        if plain is None:
            return None
        try:
            values = json.loads(plain)
        except json.JSONDecodeError:
            log.warning("a stored credential decrypted to something that is not JSON")
            return None

        if not isinstance(values, dict):
            return None
        return {str(k): str(v) for k, v in values.items()}

    def open(self, sealed: str) -> str | None:
        """Ciphertext in, plaintext out — or None, which is not an error.

        None means this row cannot be read with the key this process holds:
        the master key was rotated, or the column holds something that was
        never sealed. Both are recoverable by the user reconnecting, and
        neither is worth raising through a request that can simply report
        "not connected".
        """
        if not sealed:
            return None
        try:
            return self._cipher.decrypt(sealed.encode()).decode()
        except InvalidToken:
            log.warning("a stored Composio key did not decrypt — was the master key rotated?")
            return None
        except SecretsUnavailable:
            raise


def hint(secret: str) -> str:
    """The last four characters, for showing which credential is connected.

    Enough to tell two keys apart in a UI and useless to anybody who has it.
    Deliberately not a hash: a hash of a short, guessable-format token is not
    the protection it looks like, and this is only ever a label.
    """
    clean = (secret or "").strip()
    return clean[-4:] if len(clean) >= 4 else ""


@lru_cache
def get_sealed() -> Sealed:
    return Sealed(get_settings())
