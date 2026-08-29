"""Unit tests for the connector framework — the parts that fail silently.

Four connectors share one path here (credentials in, sealed, stored, handed
back to build a client), so a mistake in it is a mistake in all four. What is
tested is the decisions this app makes on its own, and every one is quiet when
wrong:

  - sealing, because plaintext in that column looks identical from every
    direction except a database dump;
  - the field schema, because it is what the browser renders a form from and
    what the server trusts;
  - per-user client isolation, because serving one person's credential to
    another looks exactly like it working;
  - backend resolution, because falling back to the wrong store answers
    somebody's question from somebody else's corpus.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.connectors.crypto import Sealed, SecretsUnavailable, hint
from src.connectors.registry import SPECS, get_spec, vector_slugs
from src.connectors.spec import ConnectorError, ConnectorSpec, Field


class FakeSettings:
    def __init__(self, key: str) -> None:
        self.composio_encryption_key = key


class TestSealing:
    """A user's Composio key, on its way to a column.

    The property under test is not "encryption works" — `cryptography` owns
    that — but that this app never writes a readable key and never *raises*
    over an unreadable one. Both failures are quiet: plaintext in a column
    looks fine until somebody dumps the database, and an exception on a rotated
    master key takes out a panel that could simply have said "reconnect".
    """

    KEY = "6xJ0lRr5nJ6z1Yq2c8-vXQ_9dMoPfKgWtEaHuLbNsCk="  # a valid Fernet key
    OTHER = "Zk3mVn8pQr2sTu5wXy7zAbCdEfGhIjKlMnOpQrStUvY="

    def _sealed(self, key=None):
        return Sealed(FakeSettings(key or self.KEY))

    def test_round_trips_a_key(self):
        sealed = self._sealed()
        assert sealed.open(sealed.seal("ak_live_secret")) == "ak_live_secret"

    def test_round_trips_a_whole_credential_set(self):
        """Pinecone and Astra need several fields, sealed as one blob."""
        sealed = self._sealed()
        values = {"api_key": "pcsk_secret", "index": "vec-chunks", "namespace": ""}
        assert sealed.open_map(sealed.seal_map(values)) == values

    def test_a_blob_that_is_not_json_returns_none(self):
        """Decrypting to valid text that is not an object is still unusable."""
        sealed = self._sealed()
        assert sealed.open_map(sealed.seal("just a string")) is None

    def test_ciphertext_does_not_contain_the_key(self):
        """The whole point. A grep of the column must not find the secret."""
        assert "ak_live_secret" not in self._sealed().seal("ak_live_secret")

    def test_two_seals_of_one_key_differ(self):
        """Fernet carries its own IV, so equal keys must not look equal.

        Otherwise the column leaks which users share a key by inspection.
        """
        sealed = self._sealed()
        assert sealed.seal("ak_same") != sealed.seal("ak_same")

    def test_a_rotated_master_key_returns_none_rather_than_raising(self):
        stored = self._sealed().seal("ak_live_secret")
        assert self._sealed(self.OTHER).open(stored) is None

    def test_garbage_in_the_column_returns_none(self):
        assert self._sealed().open("not-a-fernet-token") is None

    def test_empty_returns_none(self):
        assert self._sealed().open("") is None

    def test_configured_is_false_without_a_master_key(self):
        assert not Sealed(FakeSettings("")).configured

    def test_a_missing_master_key_names_itself(self):
        with pytest.raises(SecretsUnavailable, match="COMPOSIO_ENCRYPTION_KEY"):
            Sealed(FakeSettings("")).seal("ak_live_secret")

    def test_a_malformed_master_key_names_itself(self):
        """A truncated paste must not surface as a padding error at connect."""
        with pytest.raises(SecretsUnavailable, match="not a Fernet key"):
            Sealed(FakeSettings("passphrase")).seal("ak_live_secret")


class TestHint:
    def test_is_the_last_four_characters(self):
        assert hint("ak_live_abcd1234") == "1234"

    @pytest.mark.parametrize("value", ["", None, "abc"])
    def test_is_empty_when_there_is_too_little_to_show(self, value):
        assert hint(value) == ""

    def test_never_returns_most_of_a_short_key(self):
        """Four characters of a four-character key is the whole key.

        It cannot happen — `connect_composio` rejects anything under 16 — but
        the label must not become the secret if that bound ever moves.
        """
        assert len(hint("wxyz")) == 4


class FakeStore:
    """Just the lookup, holding sealed rows the way Postgres would."""

    def __init__(self, rows: dict) -> None:
        self._rows = rows
        self.configured = True

    def get(self, user_id, connector):
        return self._rows.get(user_id)

    def list(self, user_id):
        row = self._rows.get(user_id)
        return [row] if row else []


class TestClientIsolation:
    """One Composio per user, and no way to get one without naming a user.

    This is the property the whole rewrite exists for. The old shape had a
    process-wide client built from a server key; if a caller can obtain a
    client without a user id, or can be handed one built from somebody else's
    key, every other guarantee in this module is decoration.
    """

    KEY = TestSealing.KEY

    def _clients(self, rows):
        from src.connectors.service import ConnectorService
        from src.integrations.client import ComposioClients

        sealed = Sealed(FakeSettings(self.KEY))
        settings = SimpleNamespace(
            composio_timeout_s=15.0,
            composio_client_cache=8,
            composio_callback_url="http://localhost:3002/integration",
        )
        store = FakeStore(rows)
        clients = ComposioClients(ConnectorService(store, sealed), store, settings)
        # Don't build a real SDK — record which key it would have been built
        # from. That string is exactly what must never cross users.
        clients.build = lambda api_key: SimpleNamespace(built_from=api_key)
        return clients, sealed

    def _row(self, sealed, key):
        return SimpleNamespace(
            connector="composio",
            credentials=sealed.seal_map({"api_key": key}),
            hints={"api_key_hint": key[-4:]},
        )

    def test_each_user_gets_a_client_built_from_their_own_key(self):
        clients, sealed = self._clients({})
        rows = {
            "user_a": self._row(sealed, "ak_aaaa_aaaaaaaa"),
            "user_b": self._row(sealed, "ak_bbbb_bbbbbbbb"),
        }
        clients._store = FakeStore(rows)
        clients._connectors._store = clients._store

        assert clients.for_user("user_a").built_from == "ak_aaaa_aaaaaaaa"
        assert clients.for_user("user_b").built_from == "ak_bbbb_bbbbbbbb"

    def test_a_cached_client_is_not_served_to_another_user(self):
        clients, sealed = self._clients({})
        clients._store = FakeStore(
            {
                "user_a": self._row(sealed, "ak_aaaa_aaaaaaaa"),
                "user_b": self._row(sealed, "ak_bbbb_bbbbbbbb"),
            }
        )
        clients._connectors._store = clients._store

        first = clients.for_user("user_a")
        assert clients.for_user("user_b") is not first
        assert clients.for_user("user_a") is first  # still cached, still theirs

    def test_changing_a_key_invalidates_the_cached_client(self):
        """A rotated key must stop being used immediately.

        Otherwise this app keeps acting on a credential its owner revoked, for
        as long as the process lives.
        """
        rows = {"user_a": None}
        clients, sealed = self._clients(rows)
        store = FakeStore(rows)
        clients._store = store

        rows["user_a"] = self._row(sealed, "ak_old_oldoldold")
        assert clients.for_user("user_a").built_from == "ak_old_oldoldold"

        rows["user_a"] = self._row(sealed, "ak_new_newnewnew")
        assert clients.for_user("user_a").built_from == "ak_new_newnewnew"

    def test_a_user_with_no_row_gets_not_connected(self):
        from src.integrations.client import NotConnected

        clients, _ = self._clients({})
        with pytest.raises(NotConnected):
            clients.for_user("user_nobody")

    def test_no_user_id_gets_not_connected_rather_than_anything(self):
        """There is no ambient credential to fall back to, and no default."""
        from src.integrations.client import NotConnected

        clients, sealed = self._clients({})
        clients._store = FakeStore({"user_a": self._row(sealed, "ak_aaaa_aaaaaaaa")})
        clients._connectors._store = clients._store

        for missing in ("", None):
            with pytest.raises(NotConnected):
                clients.for_user(missing)

    def test_a_row_sealed_under_a_rotated_master_key_is_not_connected(self):
        """Unreadable and absent are the same thing to a caller: reconnect."""
        from src.integrations.client import NotConnected

        clients, _ = self._clients({})
        stranger = Sealed(FakeSettings(TestSealing.OTHER))
        clients._store = FakeStore({"user_a": self._row(stranger, "ak_aaaa_aaaaaaaa")})
        clients._connectors._store = clients._store

        with pytest.raises(NotConnected, match="reconnect"):
            clients.for_user("user_a")

    def test_forget_drops_the_cached_client(self):
        clients, sealed = self._clients({})
        clients._store = FakeStore({"user_a": self._row(sealed, "ak_aaaa_aaaaaaaa")})
        clients._connectors._store = clients._store

        first = clients.for_user("user_a")
        clients.forget("user_a")
        assert clients.for_user("user_a") is not first

    def test_the_cache_is_bounded(self):
        """A client per user who ever signed in is a slow leak of live keys."""
        clients, sealed = self._clients({})
        clients._store = FakeStore(
            {f"user_{n}": self._row(sealed, f"ak_{n:04d}_padding") for n in range(20)}
        )
        clients._connectors._store = clients._store

        for n in range(20):
            clients.for_user(f"user_{n}")

        assert len(clients._cache) <= 8


class TestRegistry:
    """The five connectors, and the promises the panel renders them on."""

    def test_every_spec_declares_at_least_one_field(self):
        for spec in SPECS:
            assert spec.fields, f"{spec.slug} has nothing to fill in"

    def test_every_spec_has_something_to_label_a_connection_with(self):
        """The panel needs one field that tells two connections apart.

        For a connector holding a credential that is the secret's last four
        characters — the only part of a key that may be shown. For `dataset`,
        which holds no credential at all, it is the URL itself, which is public
        by definition and is the only thing that identifies which dataset is
        attached.

        The invariant that matters is not "there is a secret" — that stopped
        being true when a connector arrived whose credential is a public
        address. It is that a spec never hints from a *credential field that
        was not marked secret*, which is the mistake this test was written to
        catch and still catches.
        """
        for spec in SPECS:
            assert spec.hint_source, f"{spec.slug} has nothing to label a row with"
            if spec.secret_fields:
                assert spec.hint_source in spec.secret_fields, spec.slug
            else:
                assert spec.hint_source in spec.public_fields, spec.slug

    def test_field_names_are_unique_within_a_connector(self):
        for spec in SPECS:
            names = [f.name for f in spec.fields]
            assert len(names) == len(set(names)), f"{spec.slug} repeats a field"

    def test_slugs_are_unique(self):
        slugs = [spec.slug for spec in SPECS]
        assert len(slugs) == len(set(slugs))

    # Spelled out rather than inferred. A heuristic gets `keyspace` wrong —
    # it contains "key" and is a namespace name — and the cost of being wrong
    # here is somebody's credential in the readable `hints` column and on the
    # wire back to the browser, with nothing else looking amiss.
    SECRETS = {
        "composio": {"api_key"},
        "pinecone": {"api_key"},
        "astra": {"token"},
        "pgvector": {"dsn"},
        # Deliberately empty, and the empty set is the assertion. A dataset is
        # a public URL: there is nothing to encrypt, and marking the URL secret
        # would hide from the panel the one field that says which dataset is
        # attached. Listed here rather than skipped so that adding a token
        # field later — for gated datasets — fails this test until somebody
        # decides it is a secret.
        "dataset": set(),
    }

    def test_exactly_the_credential_fields_are_marked_secret(self):
        for spec in SPECS:
            assert spec.slug in self.SECRETS, f"{spec.slug} is not pinned here"
            assert set(spec.secret_fields) == self.SECRETS[spec.slug], spec.slug

    def test_a_dsn_is_secret_because_it_carries_a_password(self):
        """The one that is easy to mistake for a plain connection detail."""
        assert "dsn" in get_spec("pgvector").secret_fields

    def test_names_and_endpoints_are_not_secret(self):
        """They are how a person recognises which connection this is."""
        assert "keyspace" in get_spec("astra").public_fields
        assert "endpoint" in get_spec("astra").public_fields
        assert "index" in get_spec("pinecone").public_fields
        assert "url" in get_spec("dataset").public_fields

    def test_a_dataset_is_not_a_vector_store(self):
        """`vector_slugs()` decides which connector answers a search.

        A dataset is queried through a tool and has no backend under
        `src/rag/backends/`, so it appearing in that list would make
        `BackendResolver` try to build one and fall back to the deployment
        store on every request — silently, since resolution never raises.
        """
        from src.connectors.registry import vector_slugs

        assert get_spec("dataset").kind == "dataset"
        assert "dataset" not in vector_slugs()

    def test_vector_slugs_are_derived_not_hard_coded(self):
        assert set(vector_slugs()) == {
            spec.slug for spec in SPECS if spec.kind == "vector"
        }

    def test_aliases_resolve_to_their_connector(self):
        assert get_spec("datastax").slug == "astra"
        assert get_spec("postgres").slug == "pgvector"

    @pytest.mark.parametrize("value", ["", None, "nope", "  "])
    def test_unknown_slugs_are_none_rather_than_raising(self, value):
        assert get_spec(value) is None

    def test_lookup_is_case_and_space_insensitive(self):
        assert get_spec("  PineCone ").slug == "pinecone"


class TestPgvectorTable:
    """What a connected Postgres has to look like to be worth searching.

    This used to require *this app's exact schema*, which made "connect your
    own Postgres" mean "connect a copy of ours" — a working pgvector table
    holding `id` and `chunk_text` was turned away at the form for lacking
    columns its owner had never heard of.

    The bar is now readability, not sameness: something to search and something
    to read back. Everything else this app's schema carries is a *capability*,
    and a table without one loses that channel rather than being rejected —
    which `src/rag/columns.py` makes safe, because the query is built from the
    column names rather than hardcoding them.

    The failure still guarded is the quiet one: a table that connects green and
    then abstains on every question. That is now only ever the dimension, which
    no amount of mapping can paper over.
    """

    #: The schema this app used to ingest into, kept as the "familiar table"
    #: fixture: it is still the shape `ColumnMap()`'s defaults describe, and
    #: it exercises every capability at once — including the `english` column
    #: the map reads `parallel_text` off rather than asserting.
    OURS = {
        "chunk_key": "text",
        "strategy": "text",
        "language": "text",
        "query_type": "text",
        "text": "text",
        "english": "text",
        "embedding": "vector(384)",
        "embedding_en": "vector(384)",
        "meta": "jsonb",
        "tsv": "tsvector",
        "tsv_en": "tsvector",
        "created_at": "timestamp with time zone",
    }

    THEIRS = {
        "id": "integer",
        "book_id": "text",
        "chunk_text": "text",
        "page": "integer",
        "embedding": "vector(384)",
    }

    def _check(self, columns, dim=384):
        """Discover the map, then check it — the order `verify_pgvector` uses."""
        from src.connectors.registry import _check_pgvector_table, discover_columns

        mapped = discover_columns(columns)
        _check_pgvector_table("chunks", columns, dim, mapped)
        return mapped

    def test_the_apps_own_schema_is_accepted(self):
        self._check(self.OURS)

    def test_a_table_without_tsvectors_is_still_accepted(self):
        """An older migration has no `tsv`; rung 2 degrades to dense-only.

        Required here would mean rejecting a table dense retrieval can read
        perfectly well, to protect a channel that already knows how to lose.
        """
        self._check({k: v for k, v in self.OURS.items() if not k.startswith("tsv")})

    def test_a_missing_table_says_so(self):
        with pytest.raises(ConnectorError) as error:
            self._check({})
        assert "no table called" in str(error.value)

    def test_somebody_elses_schema_is_accepted_and_mapped(self):
        """The change the whole column map exists for.

        Nothing here is named the way this app names it, and all of it is
        searchable — so it connects, and the map records what plays each role.
        """
        mapped = self._check(self.THEIRS)

        assert mapped["text"] == "chunk_text"
        assert mapped["embedding"] == "embedding"
        assert mapped["id"] == "id"

    def test_the_capabilities_it_lacks_are_lost_not_faked(self):
        """No `strategy` means the predicate is dropped, not defaulted to one
        that matches everything — a dropped filter is honest, a default one is
        a narrowing the caller thinks it applied."""
        from src.rag.columns import ColumnMap

        cmap = ColumnMap.from_mapping(self._check(self.THEIRS))

        assert cmap.searchable
        assert not (cmap.filters or cmap.lexical or cmap.parallel_text)

    def test_the_apps_own_schema_still_maps_to_everything(self):
        from src.rag.columns import ColumnMap

        cmap = ColumnMap.from_mapping(self._check(self.OURS))
        assert cmap.filters and cmap.lexical and cmap.parallel_text

    def test_a_table_with_no_text_cannot_be_read_from(self):
        """Searchable and unquotable. An answer cut from it has no source."""
        with pytest.raises(ConnectorError) as error:
            self._check({"id": "integer", "embedding": "vector(384)"})
        assert "read answers out of" in str(error.value)

    def test_a_table_with_no_vector_cannot_be_searched(self):
        with pytest.raises(ConnectorError) as error:
            self._check({"id": "integer", "body": "text"})
        assert "vector column to search" in str(error.value)

    def test_the_width_is_reported_for_the_caller_to_reconcile(self):
        """No longer an outright rejection. `reconcile_dimension` decides, and
        it can accept the store by embedding queries with *its* model."""
        from src.connectors.registry import _check_pgvector_table, discover_columns

        columns = self.OURS | {"embedding": "vector(768)", "embedding_en": "vector(768)"}
        assert _check_pgvector_table("t", columns, 384, discover_columns(columns)) == 768

    def test_halfvec_is_a_vector_too(self):
        """A table that halved its index storage is not one to refuse."""
        from src.connectors.registry import _check_pgvector_table, discover_columns

        columns = {"id": "integer", "text": "text", "embedding": "halfvec(384)"}
        assert _check_pgvector_table("t", columns, 384, discover_columns(columns)) == 384

    def test_a_column_that_is_not_a_vector_at_all(self):
        with pytest.raises(ConnectorError) as error:
            self._check({"id": "integer", "text": "text", "embedding": "text"})
        assert "vector column to search" in str(error.value)


class TestClean:
    """What a submitted form is reduced to before anything is stored."""

    SPEC = ConnectorSpec(
        slug="t",
        name="Test",
        kind="vector",
        summary="",
        fields=(
            Field("api_key", "API key", secret=True),
            Field("index", "Index"),
            Field("namespace", "Namespace", required=False),
        ),
        verify=lambda credentials: None,
    )

    def test_keeps_the_declared_fields(self):
        assert self.SPEC.clean(
            {"api_key": "k", "index": "i", "namespace": "n"}
        ) == {"api_key": "k", "index": "i", "namespace": "n"}

    def test_drops_undeclared_keys(self):
        """Otherwise a client posts arbitrary JSON into a credential blob."""
        cleaned = self.SPEC.clean({"api_key": "k", "index": "i", "evil": "x"})
        assert "evil" not in cleaned

    def test_trims_whitespace_around_a_pasted_secret(self):
        assert self.SPEC.clean({"api_key": "  k\n", "index": "i"})["api_key"] == "k"

    def test_omits_an_empty_optional_rather_than_storing_blank(self):
        assert "namespace" not in self.SPEC.clean({"api_key": "k", "index": "i"})

    @pytest.mark.parametrize("missing", [{}, {"api_key": "k"}, {"index": "i"}])
    def test_names_what_is_missing(self, missing):
        with pytest.raises(ConnectorError):
            self.SPEC.clean(missing)

    def test_whitespace_only_counts_as_missing(self):
        with pytest.raises(ConnectorError):
            self.SPEC.clean({"api_key": "   ", "index": "i"})


class TestResolution:
    """Which vector store answers for whom — and when the answer is nothing.

    There is no deployment corpus behind this app. Every one of these used to
    assert a fallback to it; what they assert now is `None`, which is what the
    ask pipeline turns into "I don't have a source to search yet".
    """

    def _resolver(self, connected):
        from src.rag.backends.resolve import BackendResolver

        rows = [SimpleNamespace(connector=slug, credentials=f"sealed-{slug}") for slug in connected]
        store = SimpleNamespace(list=lambda user_id: rows, configured=True)
        connectors = SimpleNamespace(
            credentials=lambda user_id, slug: {"marker": slug},
            configured=True,
        )
        return BackendResolver(
            connectors, store, SimpleNamespace(vector_backend_cache=4)
        )

    def _with_builders(self, resolver, ready=True):
        import src.rag.backends.resolve as mod

        original = dict(mod._BUILDERS)
        mod._BUILDERS.update(
            {slug: (lambda s: (lambda creds: SimpleNamespace(
                name=s, ready=lambda: ready, close=lambda: None
            )))(slug)
             for slug in original}
        )
        return original

    def test_anonymous_has_nothing_to_search(self):
        """Never another user's store, and never a shared one — there is none."""
        from src.rag.backends.resolve import get_resolver

        assert get_resolver().for_user(None) is None

    def test_nothing_connected_has_nothing_to_search(self):
        resolver = self._resolver([])
        assert resolver.for_user("user_a") is None

    def test_a_connected_store_answers(self):
        import src.rag.backends.resolve as mod

        resolver = self._resolver(["pinecone"])
        original = self._with_builders(resolver)
        try:
            assert resolver.for_user("user_a").name == "pinecone"
        finally:
            mod._BUILDERS.clear()
            mod._BUILDERS.update(original)

    def test_preference_decides_when_several_are_connected(self):
        """Same user, same store, every request — not whichever row came first."""
        import src.rag.backends.resolve as mod

        resolver = self._resolver(["pgvector", "astra", "pinecone"])
        original = self._with_builders(resolver)
        try:
            assert resolver.for_user("user_a").name == "pinecone"
        finally:
            mod._BUILDERS.clear()
            mod._BUILDERS.update(original)

    def test_a_store_that_cannot_answer_is_not_a_store(self):
        """Built is not the same as searchable — an emptied index is neither."""
        import src.rag.backends.resolve as mod

        resolver = self._resolver(["pinecone"])
        original = self._with_builders(resolver, ready=False)
        try:
            assert resolver.for_user("user_a") is None
            # And the miss is cached, so the probe costs one round trip per
            # connect rather than one per question.
            assert resolver._cache["user_a"][2] is None
            assert resolver.for_user("user_a") is None
        finally:
            mod._BUILDERS.clear()
            mod._BUILDERS.update(original)

    def test_a_broken_connector_returns_none_rather_than_raising(self):
        """A connector that cannot be built must abstain, not 500."""
        resolver = self._resolver(["pinecone"])
        resolver._connectors = SimpleNamespace(
            credentials=lambda user_id, slug: (_ for _ in ()).throw(RuntimeError("boom")),
            configured=True,
        )
        assert resolver.for_user("user_a") is None

    def test_unreadable_credentials_return_none_rather_than_raising(self):
        resolver = self._resolver(["pinecone"])
        resolver._connectors = SimpleNamespace(
            credentials=lambda user_id, slug: None, configured=True
        )
        assert resolver.for_user("user_a") is None

    def test_a_tools_connector_never_becomes_a_vector_backend(self):
        """Composio is attached the same way and is not a place to search."""
        resolver = self._resolver(["composio"])
        assert resolver.for_user("user_a") is None
