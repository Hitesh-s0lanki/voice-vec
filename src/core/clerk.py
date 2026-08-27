"""Proving that a caller is who they say they are.

The browser opens the voice socket against *this* server, not through Next, so
"who is speaking" cannot be a header somebody sets. It is a Clerk session
token, verified here against Clerk's own signing key, and the user id is the
`sub` claim of a signature that checked out. An unverified id is never trusted
for anything: fail verification and the caller falls back to being anonymous —
known only by the `sess_…` their browser minted.

Where the key comes from, in order:

  1. `CLERK_JWT_KEY` — the PEM public key from the Clerk dashboard. No network
     at all, which is the right answer for a deployment that would rather not
     depend on an outbound call at handshake time.
  2. `CLERK_PUBLISHABLE_KEY` — the frontend API host is base64'd into it, and
     the JWKS lives at a well-known path under that host. Nothing to copy, and
     `PyJWKClient` caches the fetched keys, so it costs one request per hour
     rather than one per connection.

Neither set means Clerk verification is off, and every caller is anonymous.
That is the state a checkout with no Clerk keys runs in, and the voice loop is
identical in it — conversations are simply owned by the browser rather than by
an account.
"""

from __future__ import annotations

import base64
import binascii
import logging
from functools import lru_cache

import jwt
from jwt import PyJWKClient

from src.core.config import Settings, get_settings

log = logging.getLogger("vec.clerk")

# Clerk signs session tokens with RS256. Pinning it here is not a preference:
# accepting whatever the token's own header asks for is how a verifier gets
# talked into `alg: none`.
_ALGORITHMS = ["RS256"]

# Clock skew allowance. Clerk's session tokens live ~60 seconds, so this is
# deliberately small — a generous tolerance here is most of a token's lifetime.
_LEEWAY_S = 5


def frontend_api(publishable_key: str) -> str:
    """The instance host, decoded out of the publishable key.

    `pk_test_<base64("host$")>`. It is called publishable because it is public —
    it ships to every browser — so reading it here reveals nothing, and it
    saves asking anyone to copy a second value into a second `.env`.
    """
    _, _, body = publishable_key.partition("_test_")
    if not body:
        _, _, body = publishable_key.partition("_live_")
    if not body:
        return ""

    try:
        decoded = base64.b64decode(body + "=" * (-len(body) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError):
        log.warning("CLERK_PUBLISHABLE_KEY is not decodable — token checks are off")
        return ""

    return decoded.rstrip("$")


class Verifier:
    """Turns a session token into a user id, or into nothing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: PyJWKClient | None = None

    @property
    def host(self) -> str:
        return frontend_api(self._settings.clerk_publishable_key)

    @property
    def enabled(self) -> bool:
        return bool(self._settings.clerk_jwt_key or self.host)

    def _jwks(self) -> PyJWKClient:
        """One client per process — it is the cache.

        Keys are held inside it and refetched only when a token arrives signed
        by a `kid` it has not seen, which is what makes a key rotation
        self-healing rather than an outage.
        """
        if self._client is None:
            self._client = PyJWKClient(
                f"https://{self.host}/.well-known/jwks.json",
                cache_keys=True,
                lifespan=self._settings.clerk_jwks_ttl_s,
            )
        return self._client

    def _key(self, token: str) -> str | None:
        if self._settings.clerk_jwt_key:
            key = self._settings.clerk_jwt_key.strip().replace("\\n", "\n")
            # A key pasted into an .env as one line, without the PEM armour.
            if not key.startswith("-----BEGIN"):
                key = "-----BEGIN PUBLIC KEY-----\n" + key + "\n-----END PUBLIC KEY-----"
            return key

        if not self.host:
            return None

        try:
            return self._jwks().get_signing_key_from_jwt(token).key
        except Exception as error:  # unreachable JWKS, unknown kid, junk token
            log.warning("no signing key for that token: %s", error)
            return None

    def user_id(self, token: str | None) -> str | None:
        """The signed-in user behind a token, or None for anyone else.

        None is not an error and is never reported as one. A signed-out
        visitor, an expired token, a token from another Clerk instance and a
        forged one all land here the same way, and all of them are served —
        as the anonymous browser they are.
        """
        if not token or not self.enabled:
            return None

        key = self._key(token)
        if key is None:
            return None

        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=_ALGORITHMS,
                leeway=_LEEWAY_S,
                # `iss` is checked only when the host is known. With a PEM key
                # configured and no publishable key there is nothing to compare
                # against, and the signature is already the binding claim.
                issuer=f"https://{self.host}" if self.host else None,
                options={"require": ["exp", "sub"], "verify_aud": False},
            )
        except jwt.InvalidTokenError as error:
            log.info("rejected a session token: %s", error)
            return None

        subject = claims.get("sub")
        return str(subject) if subject else None


@lru_cache
def get_verifier() -> Verifier:
    return Verifier(get_settings())
