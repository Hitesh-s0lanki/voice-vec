"""Unit tests for Composio's toolkit flow — the parts that fail silently.

Credentials, sealing and per-user client isolation are `test_connectors.py`;
this is only what Composio does once it is attached.

No Composio and no Postgres here, for the reason `test_chat.py` gives: a mock
of somebody else's API only proves the mock works. What is tested is the
decisions this app makes on its own, and every one of them is a decision that
would be wrong *quietly*:

  - the auth gate, because a route that serves an unverified caller looks
    exactly like a route that works;
  - `link()` versus `initiate()`, because picking the retired one fails only
    against a real Composio and only for managed OAuth;
  - slug normalisation, because it reaches a primary key and a URL path;
  - reconciliation's revoke rule, because a connection that went away upstream
    and stayed green here is the failure nobody notices until they trust it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.controllers.connectors_controller import require_user
from src.services.integration_service import IntegrationService, _slug


class FakeVerifier:
    """Stands in for `src.core.clerk.Verifier` — one token, one subject."""

    def __init__(self, good: str = "tok", subject: str = "user_abc") -> None:
        self._good = good
        self._subject = subject

    def user_id(self, token):
        return self._subject if token == self._good else None


class TestAuthGate:
    def test_accepts_a_verified_bearer_token(self):
        assert require_user(FakeVerifier(), "Bearer tok") == "user_abc"

    def test_is_case_insensitive_about_the_scheme(self):
        assert require_user(FakeVerifier(), "bearer tok") == "user_abc"

    @pytest.mark.parametrize(
        "header",
        [
            None,
            "",
            "tok",  # no scheme — a bare token is not a credential here
            "Bearer",
            "Bearer ",
            "Bearer forged",
            "Basic tok",  # right token, wrong scheme
            "Bearer tok extra",  # partition keeps everything after the space
        ],
    )
    def test_refuses_everything_else(self, header):
        with pytest.raises(HTTPException) as raised:
            require_user(FakeVerifier(), header)
        assert raised.value.status_code == 401

    def test_never_takes_a_user_id_from_the_caller(self):
        """The signature is the only thing that names an account.

        `require_user` takes exactly two arguments and neither is a user id, so
        there is nowhere for a client-supplied one to enter. This test exists
        to fail loudly if that ever gains a third.
        """
        import inspect

        parameters = set(inspect.signature(require_user).parameters)
        assert parameters == {"verifier", "authorization"}


class TestSlug:
    def test_lowercases_what_composio_lowercases(self):
        assert _slug("GitHub") == "github"

    def test_keeps_the_separators_slugs_actually_use(self):
        assert _slug("google_calendar") == "google_calendar"
        assert _slug("google-drive") == "google-drive"

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("", ""),
            (None, ""),
            ("  slack  ", "slack"),
            ("../../admin", "admin"),
            ("gmail; DROP TABLE integration_connections", "gmaildroptableintegration_connections"),
            ("a" * 200, "a" * 64),
        ],
    )
    def test_strips_anything_that_is_not_a_slug(self, value, expected):
        assert _slug(value) == expected


class TestStartPicksTheLiveEndpoint:
    """`initiate()` is retired for Composio-managed OAuth; `link()` is not.

    Composio blocked it for new organisations on 2026-05-08 and for everybody
    on 2026-07-03. Calling it on a managed auth config now raises instead of
    returning a consent URL — and the only signal is a 400 from a real
    Composio, which no local run would ever see.
    """

    def _service_and_sdk(self, calls):
        sdk = SimpleNamespace(
            connected_accounts=SimpleNamespace(
                link=lambda user_id, auth_config_id, callback_url=None: calls.append(
                    ("link", user_id, auth_config_id, callback_url)
                ),
                initiate=lambda user_id, auth_config_id, callback_url=None: calls.append(
                    ("initiate", user_id, auth_config_id, callback_url)
                ),
            )
        )
        clients = SimpleNamespace(
            configured=True, callback_url="http://localhost:3002/integration"
        )
        service = IntegrationService(clients, store=None, settings=None)
        return service, sdk

    def test_managed_auth_uses_link(self):
        calls = []
        service, sdk = self._service_and_sdk(calls)
        service._start(sdk, "gmail", "user_abc", "ac_1", managed=True)
        assert calls == [
            ("link", "user_abc", "ac_1", "http://localhost:3002/integration?toolkit=gmail")
        ]

    def test_custom_auth_still_uses_initiate(self):
        calls = []
        service, sdk = self._service_and_sdk(calls)
        service._start(sdk, "gmail", "user_abc", "ac_1", managed=False)
        assert calls == [
            ("initiate", "user_abc", "ac_1", "http://localhost:3002/integration?toolkit=gmail")
        ]


class TestPairs:
    """What reconciliation reads out of a Composio list response."""

    def _pairs(self, items):
        service = IntegrationService(clients=None, store=None, settings=None)
        return list(service._pairs(SimpleNamespace(items=items)))

    def test_reads_a_nested_toolkit_object(self):
        item = SimpleNamespace(
            id="ca_1", toolkit=SimpleNamespace(slug="GMAIL"), status="ACTIVE"
        )
        assert self._pairs([item]) == [("gmail", "ca_1", "ACTIVE")]

    def test_reads_a_bare_toolkit_slug(self):
        item = SimpleNamespace(id="ca_2", toolkit="slack", status="ACTIVE")
        assert self._pairs([item]) == [("slack", "ca_2", "ACTIVE")]

    def test_drops_an_item_naming_no_toolkit(self):
        """An empty slug would collide with every other one on the unique key."""
        item = SimpleNamespace(id="ca_3", toolkit=None, status="ACTIVE")
        assert self._pairs([item]) == []

    def test_drops_an_item_with_no_account_id(self):
        item = SimpleNamespace(id=None, toolkit="slack", status="ACTIVE")
        assert self._pairs([item]) == []

    def test_defaults_a_missing_status_rather_than_storing_none(self):
        item = SimpleNamespace(id="ca_4", toolkit="notion", status=None)
        assert self._pairs([item]) == [("notion", "ca_4", "INITIALIZING")]

    def test_survives_a_response_with_no_items(self):
        assert self._pairs(None) == []
