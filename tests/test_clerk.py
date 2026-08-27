"""Unit tests for token verification — the gate in front of every account.

No network and no Clerk instance: the tests sign their own tokens with their
own key and hand the verifier the matching PEM through `CLERK_JWT_KEY`, which
is the same code path a deployment that pins the key takes. What is being
tested is the part that decides whether a `sub` is trustworthy, and the
failures worth having a test for are the ones that fail *open* — a token that
expired, one signed by somebody else, one from another instance, one asking to
be trusted without a signature at all.
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.core.clerk import Verifier, frontend_api
from src.core.config import Settings

ISSUER_HOST = "touching-panther-1346.clerk.accounts.dev"
# `pk_test_` plus base64 of "<host>$" — how Clerk packs the instance host in.
PUBLISHABLE = "pk_test_dG91Y2hpbmctcGFudGhlci0xMzQ2LmNsZXJrLmFjY291bnRzLmRldiQ"


@pytest.fixture(scope="module")
def keys() -> tuple[str, str]:
    """A throwaway RSA pair, as PEM."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private, public


def verifier(public: str, **overrides) -> Verifier:
    return Verifier(
        Settings(clerk_jwt_key=public, clerk_publishable_key=PUBLISHABLE, **overrides)
    )


def token(private: str, **claims) -> str:
    now = int(time.time())
    payload = {
        "sub": "user_2abcXYZ",
        "iss": f"https://{ISSUER_HOST}",
        "iat": now,
        "nbf": now - 1,
        "exp": now + 60,
        "sid": "sess_clerk_side",
        **claims,
    }
    return jwt.encode(payload, private, algorithm="RS256")


class TestFrontendApi:
    def test_decodes_the_host_out_of_a_test_key(self):
        assert frontend_api(PUBLISHABLE) == ISSUER_HOST

    @pytest.mark.parametrize("value", ["", "not-a-key", "pk_test_!!!!", "sk_test_abc"])
    def test_anything_unreadable_disables_verification(self, value):
        assert frontend_api(value) == ""


class TestVerifier:
    def test_accepts_a_token_it_can_check(self, keys):
        private, public = keys
        assert verifier(public).user_id(token(private)) == "user_2abcXYZ"

    def test_rejects_an_expired_one(self, keys):
        """Clerk's session tokens live about a minute; this is the common case."""
        private, public = keys
        now = int(time.time())
        stale = token(private, iat=now - 600, nbf=now - 600, exp=now - 300)

        assert verifier(public).user_id(stale) is None

    def test_rejects_one_signed_by_somebody_else(self, keys):
        _, public = keys
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = token(
            other.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode()
        )

        assert verifier(public).user_id(forged) is None

    def test_rejects_one_from_another_clerk_instance(self, keys):
        """Right key, wrong issuer — a token minted for a different app."""
        private, public = keys
        elsewhere = token(private, iss="https://someone-else.clerk.accounts.dev")

        assert verifier(public).user_id(elsewhere) is None

    def test_rejects_an_unsigned_token(self, keys):
        """`alg: none` is the classic way a verifier gets talked out of its job."""
        private, public = keys
        unsigned = jwt.encode(
            {"sub": "user_intruder", "iss": f"https://{ISSUER_HOST}", "exp": int(time.time()) + 60},
            None,
            algorithm="none",
        )

        assert verifier(public).user_id(unsigned) is None
        assert verifier(public).user_id(token(private)) == "user_2abcXYZ"  # still works

    def test_rejects_a_token_with_no_subject(self, keys):
        private, public = keys
        assert verifier(public).user_id(token(private, sub=None)) is None

    @pytest.mark.parametrize("value", [None, "", "garbage", "a.b.c"])
    def test_survives_anything_that_is_not_a_token(self, keys, value):
        _, public = keys
        assert verifier(public).user_id(value) is None

    def test_no_keys_configured_means_everyone_is_anonymous(self, keys):
        """A checkout with no Clerk keys still runs — nobody is signed in."""
        private, _ = keys
        off = Verifier(Settings(clerk_jwt_key="", clerk_publishable_key=""))

        assert off.enabled is False
        assert off.user_id(token(private)) is None
