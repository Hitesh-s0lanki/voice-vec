"""Unit tests for the Composio MCP gateway transport.

`test_integrations.py` covers the toolkit flow once Composio is attached. This
covers the second way it can be attached — a `ck_` gateway key, which speaks
JSON-RPC to one endpoint instead of REST to many.

No network. What is tested is the translation layer, because every mistake in
it is one that fails *quietly*:

  - key routing, because a `ck_` key sent to the SDK and an `ak_` key sent to
    the gateway both come back as "invalid API key" for a key that is fine;
  - result unwrapping, because Composio nests three envelopes and the innermost
    one is the only one that knows whether the *tool* worked — a failed send
    read at the wrong depth is reported to the model as a sent email;
  - toolkit scoping, because the gateway's search is semantic and will happily
    return a Slack tool for "use the gmail app", which would hand the model a
    tool its owner never authorised;
  - status casing, because the gateway says `active` where the store compares
    against `ACTIVE` exactly, and a connection that is live but never matches
    reads as permanently pending;
  - the consent URL, because the field has been renamed between gateway builds
    and a missing one must raise rather than hand the panel a blank redirect.
"""

from __future__ import annotations

import pytest

from src.integrations.mcp import (
    ComposioGateway,
    GatewayError,
    _find_url,
    _toolkit_slug,
    is_gateway_key,
)


class FakeGateway(ComposioGateway):
    """A gateway whose only difference is that it never opens a socket."""

    def __init__(self, payload=None, *, error: Exception | None = None) -> None:
        super().__init__("ck_test")
        self._payload = payload or {}
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, arguments: dict) -> dict:
        self.calls.append((tool, arguments))
        if self._error:
            raise self._error
        return self._payload


class TestKeyRouting:
    """The one decision made before any round trip."""

    @pytest.mark.parametrize("key", ["ck_abc", "  ck_abc  ", "ck_"])
    def test_recognises_a_gateway_key(self, key):
        assert is_gateway_key(key) is True

    @pytest.mark.parametrize("key", ["ak_abc", "", "pcsk_abc", "sk-abc", None])
    def test_everything_else_is_a_platform_key(self, key):
        assert is_gateway_key(key) is False


class TestExecuteUnwrapping:
    """Three envelopes deep, and only the innermost one is the verdict."""

    def _entry(self, entry):
        return FakeGateway({"data": {"results": [entry]}, "successful": True})

    def test_returns_the_innermost_data(self):
        gw = self._entry(
            {
                "tool_slug": "GMAIL_GET_PROFILE",
                "response": {"successful": True, "data": {"emailAddress": "a@b.c"}},
            }
        )
        ok, data, error = gw.execute("GMAIL_GET_PROFILE", {})
        assert ok is True
        assert data == {"emailAddress": "a@b.c"}
        assert error is None

    def test_a_tool_that_failed_inside_a_successful_call_is_a_failure(self):
        """The bug this unwrapping exists to prevent.

        The multi-executor runs up to fifty tools and reports its own success;
        a tool that refused rides inside it. Read at the outer depth this is a
        sent email.
        """
        gw = self._entry(
            {
                "tool_slug": "GMAIL_SEND_EMAIL",
                "response": {"successful": False, "error": "scope insufficient"},
            }
        )
        ok, data, error = gw.execute("GMAIL_SEND_EMAIL", {})
        assert ok is False
        assert data is None
        assert error == "scope insufficient"

    def test_reports_a_per_tool_error(self):
        gw = self._entry(
            {"tool_slug": "SLACK_SEND_MESSAGE", "error": "No active connection found"}
        )
        ok, _data, error = gw.execute("SLACK_SEND_MESSAGE", {})
        assert ok is False
        assert "No active connection" in error

    def test_a_gateway_error_becomes_a_result_rather_than_an_exception(self):
        """`ToolAgent.execute` is mid-turn with somebody listening."""
        gw = FakeGateway(error=GatewayError("gateway down"))
        ok, _data, error = gw.execute("GMAIL_GET_PROFILE", {})
        assert ok is False
        assert error == "gateway down"

    def test_survives_a_response_shape_it_has_never_seen(self):
        gw = self._entry({"tool_slug": "X_DO", "output": [1, 2, 3]})
        ok, data, _error = gw.execute("X_DO", {})
        assert ok is True
        assert data == [1, 2, 3]


class TestConnections:
    """Shaped to feed `IntegrationStore.reconcile` unchanged."""

    def test_upper_cases_status_because_the_store_compares_exactly(self):
        gw = FakeGateway(
            {
                "data": {
                    "results": {
                        "gmail": {
                            "status": "active",
                            "accounts": [{"id": "gmail_x", "status": "active"}],
                        }
                    }
                }
            }
        )
        assert gw.connections(["gmail"]) == [("gmail", "gmail_x", "ACTIVE")]

    def test_drops_a_toolkit_with_no_account_because_that_is_not_a_connection(self):
        """The gateway answers `initiated` for every toolkit nobody has linked.

        Relaying that filed a "waiting for consent" row against any service the
        caller merely asked about — and the panel polls every three seconds for
        as long as one of those exists. A consent genuinely in flight is
        protected by `reconcile(sees_pending=False)` instead, which is the only
        place that can tell the two apart.
        """
        gw = FakeGateway(
            {"data": {"results": {"slack": {"status": "initiated", "accounts": []}}}}
        )
        assert gw.connections(["slack"]) == []

    def test_keeps_a_dead_toolkit_status_because_that_is_news(self):
        """FAILED with no account is a connection that broke, not one never made."""
        gw = FakeGateway(
            {"data": {"results": {"slack": {"status": "failed", "accounts": []}}}}
        )
        assert gw.connections(["slack"]) == [("slack", "", "FAILED")]

    def test_asks_nothing_when_there_is_nothing_to_ask_about(self):
        gw = FakeGateway({})
        assert gw.connections([]) == []
        assert gw.calls == []

    def test_tolerates_a_shape_it_does_not_recognise(self):
        assert FakeGateway({"data": {"results": "nonsense"}}).connections(["gmail"]) == []


class TestToolScoping:
    """The gateway's search is semantic; the toolkit filter is not optional."""

    def _searching(self, primary, related=()):
        class Gateway(FakeGateway):
            def call(inner, tool, arguments):
                inner.calls.append((tool, arguments))
                if tool == "COMPOSIO_SEARCH_TOOLS":
                    return {
                        "data": {
                            "results": [
                                {
                                    "primary_tool_slugs": list(primary),
                                    "related_tool_slugs": list(related),
                                }
                            ]
                        }
                    }
                return {
                    "data": {
                        "tool_schemas": {
                            slug: {"tool_slug": slug, "description": "d", "input_schema": {}}
                            for slug in arguments["tool_slugs"]
                        }
                    }
                }

        return Gateway()

    def test_drops_tools_from_a_toolkit_the_user_never_linked(self):
        gw = self._searching(["GMAIL_SEND_EMAIL", "SLACK_SEND_MESSAGE"])
        names = [schema["function"]["name"] for schema in gw.tools_for(["gmail"])]
        assert names == ["GMAIL_SEND_EMAIL"]

    def test_takes_related_tools_too_so_the_toolset_does_not_swing(self):
        gw = self._searching(["GMAIL_SEND_EMAIL"], ["GMAIL_FETCH_EMAILS"])
        names = [schema["function"]["name"] for schema in gw.tools_for(["gmail"])]
        assert names == ["GMAIL_SEND_EMAIL", "GMAIL_FETCH_EMAILS"]

    def test_asks_for_nothing_when_no_toolkit_is_linked(self):
        gw = self._searching(["GMAIL_SEND_EMAIL"])
        assert gw.tools_for([]) == []
        assert gw.calls == []


class TestSchemas:
    def test_builds_the_openai_shape_the_llm_payload_expects(self):
        gw = FakeGateway(
            {
                "data": {
                    "tool_schemas": {
                        "GMAIL_SEND_EMAIL": {
                            "tool_slug": "GMAIL_SEND_EMAIL",
                            "description": "Sends an email",
                            "input_schema": {"type": "object", "properties": {"to": {}}},
                        }
                    }
                }
            }
        )
        assert gw.schemas(["GMAIL_SEND_EMAIL"]) == [
            {
                "type": "function",
                "function": {
                    "name": "GMAIL_SEND_EMAIL",
                    "description": "Sends an email",
                    "parameters": {"type": "object", "properties": {"to": {}}},
                },
            }
        ]

    def test_always_has_parameters_even_when_composio_omits_them(self):
        """A function with no `parameters` key is rejected by the chat API."""
        gw = FakeGateway({"data": {"tool_schemas": {"X_DO": {"tool_slug": "X_DO"}}}})
        assert gw.schemas(["X_DO"])[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestCatalogueSearch:
    def test_ranks_by_how_often_a_toolkit_was_returned(self):
        gw = FakeGateway(
            {
                "data": {
                    "results": [
                        {"toolkits": ["slack", "gmail"]},
                        {"toolkits": ["gmail"]},
                    ]
                }
            }
        )
        assert [entry["slug"] for entry in gw.search_toolkits("email")] == ["gmail", "slack"]

    def test_falls_back_to_the_prefix_of_a_tool_slug(self):
        gw = FakeGateway({"data": {"results": [{"primary_tool_slugs": ["NOTION_ADD_PAGE"]}]}})
        assert [entry["slug"] for entry in gw.search_toolkits("notes")] == ["notion"]

    def test_an_empty_query_still_asks_something(self):
        """An unsearched panel opens with recognisable services, not nothing."""
        gw = FakeGateway({"data": {"results": []}})
        gw.search_toolkits("")
        _tool, arguments = gw.calls[0]
        assert len(arguments["queries"]) > 1


class TestConsentUrl:
    @pytest.mark.parametrize(
        "payload",
        [
            {"redirect_url": "https://consent"},
            {"auth_url": "https://consent"},
            {"url": "https://consent"},
            {"results": {"gmail": {"connection": {"redirect_url": "https://consent"}}}},
            {"items": [{"link": "https://consent"}]},
        ],
    )
    def test_finds_the_url_wherever_this_build_put_it(self, payload):
        assert _find_url(payload) == "https://consent"

    def test_ignores_a_non_https_value(self):
        assert _find_url({"url": "javascript:alert(1)"}) == ""

    def test_add_connection_raises_rather_than_returning_a_blank_redirect(self):
        gw = FakeGateway({"data": {"results": {"gmail": {"status": "initiated"}}}})
        with pytest.raises(GatewayError, match="consent URL"):
            gw.add_connection("gmail")

    def test_returns_the_url_and_the_status(self):
        gw = FakeGateway(
            {
                "data": {
                    "results": {"gmail": {"status": "initiated"}},
                    "redirect_url": "https://consent",
                }
            }
        )
        assert gw.add_connection("gmail") == ("https://consent", "INITIATED")


class TestToolkitSlug:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Gmail", "gmail"),
            ({"slug": "SLACK"}, "slack"),
            ({"name": "Notion"}, "notion"),
            ({}, ""),
            (None, ""),
            (7, ""),
        ],
    )
    def test_reads_a_slug_out_of_whatever_wrapped_it(self, value, expected):
        assert _toolkit_slug(value) == expected


class TestVerifyRouting:
    def test_a_gateway_key_never_reaches_the_sdk(self, monkeypatch):
        """The whole point: a `ck_` key must not be graded by the REST API."""
        from src.connectors import registry

        seen: dict[str, str] = {}

        class Stub:
            def __init__(self, api_key, **_):
                seen["key"] = api_key

            def verify(self):
                seen["verified"] = "yes"

        monkeypatch.setattr("src.integrations.mcp.ComposioGateway", Stub)
        registry.verify_composio({"api_key": "ck_abc"})
        assert seen == {"key": "ck_abc", "verified": "yes"}

    def test_a_rejected_gateway_key_names_the_gateway(self, monkeypatch):
        from src.connectors import registry
        from src.connectors.spec import ConnectorError

        class Stub:
            def __init__(self, *_a, **_k):
                pass

            def verify(self):
                raise GatewayError("nope")

        monkeypatch.setattr("src.integrations.mcp.ComposioGateway", Stub)
        with pytest.raises(ConnectorError, match="gateway"):
            registry.verify_composio({"api_key": "ck_abc"})
