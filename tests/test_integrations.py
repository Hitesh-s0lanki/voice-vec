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
from src.agents.tool_agent import ToolAgent
from src.services.integration_service import (
    IntegrationError,
    IntegrationService,
    _slug,
    _tool_name,
)


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


def _agent(schemas: list[dict]) -> ToolAgent:
    """A ToolAgent with the Composio round trip taken out.

    `inventory` is the thing under test and it reads whatever `tools_for`
    returns, so the fake replaces exactly that and nothing else.
    """
    agent = ToolAgent.__new__(ToolAgent)
    agent.tools_for = lambda user_id: schemas  # type: ignore[method-assign]
    return agent


class TestToolInventory:
    """What the panel says the agent can do has to be what it *would* be
    handed — read back through the same call the turn makes."""

    def test_tools_are_grouped_by_their_toolkit(self):
        agent = _agent(
            [
                {"type": "function", "function": {"name": "GMAIL_SEND_EMAIL"}},
                {"type": "function", "function": {"name": "GMAIL_FETCH_EMAILS"}},
                {"type": "function", "function": {"name": "SLACK_POST_MESSAGE"}},
            ]
        )
        inventory = agent.inventory("u1")

        assert sorted(inventory) == ["gmail", "slack"]
        assert len(inventory["gmail"]) == 2

    def test_a_flat_schema_is_read_as_well_as_a_nested_one(self):
        """The gateway path builds the shape by hand; a panel is not worth a
        KeyError over which of the two it built."""
        inventory = _agent([{"name": "NOTION_CREATE_PAGE", "description": "d"}]).inventory("u1")

        assert inventory["notion"][0]["slug"] == "NOTION_CREATE_PAGE"

    @pytest.mark.parametrize("schema", [{}, {"function": {"name": ""}}, "not a dict"])
    def test_a_schema_with_no_name_is_skipped_not_raised(self, schema):
        assert _agent([schema]).inventory("u1") == {}

    def test_the_service_reports_the_total_and_whether_it_was_capped(self):
        service = IntegrationService(
            clients=None,
            store=None,
            settings=SimpleNamespace(tool_schema_limit=3),
            agent=_agent(
                [
                    {"function": {"name": "GMAIL_SEND_EMAIL", "description": "  Sends  mail. "}},
                    {"function": {"name": "GMAIL_FETCH_EMAILS"}},
                    {"function": {"name": "SLACK_POST_MESSAGE"}},
                ]
            ),
        )
        inventory = service.tools("u1")

        assert inventory.total == 3
        # Equal to the limit is the only sign there may be more behind it.
        assert inventory.limited is True
        assert [kit.toolkit for kit in inventory.toolkits] == ["gmail", "slack"]

        # Sorted by slug inside a toolkit, so the list does not reshuffle
        # between panel opens as Composio's own order changes.
        gmail = inventory.toolkits[0]
        assert [tool.slug for tool in gmail.tools] == [
            "GMAIL_FETCH_EMAILS",
            "GMAIL_SEND_EMAIL",
        ]
        assert gmail.tools[1].description == "Sends mail."

    def test_an_unreadable_inventory_empties_the_section_rather_than_the_panel(self):
        broken = ToolAgent.__new__(ToolAgent)

        def raise_it(user_id):
            raise RuntimeError("composio is down")

        broken.tools_for = raise_it  # type: ignore[method-assign]
        service = IntegrationService(
            clients=None, store=None, settings=SimpleNamespace(tool_schema_limit=40), agent=broken
        )

        assert service.tools("u1").total == 0


class TestGatewayReconcile:
    """The gateway has no "list everything" call, so what it is asked matters.

    Asking only about rows already on file made the panel a mirror of its own
    database: a toolkit linked in the Composio dashboard, or on the same key
    from anywhere else, never appeared — so the panel offered to link a service
    that was already linked, and the voice turn got no tools for it.
    """

    class Store:
        def __init__(self, rows=()):
            self.rows = [
                SimpleNamespace(toolkit=toolkit, status=status) for toolkit, status in rows
            ]
            self.reconciled = None
            self.sees_pending = None

        def list(self, user_id):
            return self.rows

        def reconcile(self, user_id, live, *, sees_pending=True):
            self.reconciled = list(live)
            self.sees_pending = sees_pending

    class Gateway:
        def __init__(self, answers):
            self.answers = answers
            self.asked = None

        def connections(self, toolkits):
            self.asked = list(toolkits)
            return [pair for pair in self.answers if pair[0] in self.asked]

    def _service(self, store, sampled=()):
        service = IntegrationService(clients=None, store=store, settings=SimpleNamespace())
        service.catalog = lambda user_id, **_: SimpleNamespace(  # type: ignore[method-assign]
            toolkits=[SimpleNamespace(slug=slug) for slug in sampled]
        )
        return service

    def test_asks_about_the_catalogue_sample_and_not_only_its_own_rows(self):
        store = self.Store([("gmail", "ACTIVE")])
        gateway = self.Gateway([("gmail", "gmail_a", "ACTIVE"), ("github", "github_b", "ACTIVE")])

        self._service(store, sampled=["gmail", "github", "slack"])._reconcile_gateway(
            "u1", gateway
        )

        assert gateway.asked == ["github", "gmail", "slack"]
        # GitHub was linked outside this app and is now on file.
        assert ("github", "github_b", "ACTIVE") in store.reconciled

    def test_an_accountless_answer_about_an_unknown_toolkit_is_not_a_connection(self):
        """Otherwise a search result becomes a row nobody asked for."""
        store = self.Store()
        gateway = self.Gateway([("slack", "", "FAILED")])

        self._service(store, sampled=["slack"])._reconcile_gateway("u1", gateway)

        assert store.reconciled == []

    def test_an_accountless_answer_about_a_known_toolkit_still_counts(self):
        store = self.Store([("slack", "ACTIVE")])
        gateway = self.Gateway([("slack", "", "FAILED")])

        self._service(store, sampled=[])._reconcile_gateway("u1", gateway)

        assert store.reconciled == [("slack", "", "FAILED")]

    def test_tells_the_store_it_cannot_see_a_consent_in_flight(self):
        """The gateway reports accounts, and consent in flight has none."""
        store = self.Store([("gmail", "INITIALIZING")])

        self._service(store)._reconcile_gateway("u1", self.Gateway([]))

        assert store.sees_pending is False

    def test_a_dead_catalogue_costs_discovery_and_not_the_reconcile(self):
        store = self.Store([("gmail", "ACTIVE")])
        gateway = self.Gateway([("gmail", "gmail_a", "ACTIVE")])

        service = IntegrationService(clients=None, store=store, settings=SimpleNamespace())

        def raise_it(user_id, **_):
            raise IntegrationError("Composio did not answer")

        service.catalog = raise_it  # type: ignore[method-assign]
        service._reconcile_gateway("u1", gateway)

        assert gateway.asked == ["gmail"]
        assert store.reconciled == [("gmail", "gmail_a", "ACTIVE")]


class TestToolNames:
    def test_the_toolkit_prefix_is_dropped_for_the_label(self):
        assert _tool_name("GMAIL_SEND_EMAIL", "gmail") == "Send email"

    def test_a_slug_that_does_not_carry_its_toolkit_keeps_its_words(self):
        assert _tool_name("SEARCH_DOCS", "notion") == "Search docs"
