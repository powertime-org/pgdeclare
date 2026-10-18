"""Publications, rewrite rules and event triggers.

The unit tests drive the differ and the rendered SQL. The integration tests
run the whole pipeline (plan database, introspection, diff, apply, convergence
check) against a real server and need TEST_DATABASE_URL. Event triggers need
a superuser; those tests skip without one.
"""

import os
import uuid
from pathlib import Path

import psycopg2
import pytest

from pgdeclare import (
    PgDeclareConfig,
    apply_declared_schema,
    build_desired_schema,
    compute_diff,
    introspect_schema,
)
from pgdeclare.changes import (
    AddPublicationObjects,
    AlterEventTriggerEnabled,
    AlterPublicationOptions,
    AlterRuleEnabled,
    CreateEventTrigger,
    CreatePublication,
    CreateRule,
    DropAndCreateFunction,
    DropAndCreatePublication,
    DropEventTrigger,
    DropPublication,
    DropPublicationObjects,
    DropRule,
    DropTable,
    ReplacePublicationTable,
)
from pgdeclare.models import (
    EventTrigger,
    Function,
    Publication,
    PublicationTable,
    Rule,
    Schema,
    Table,
)
from pgdeclare.schema_parser import _event_triggers_last
from pgdeclare.statement_parser import parse_sql_content


# =============================================================================
# Unit tests
# =============================================================================


def _table(name: str, schema: str = "public") -> Table:
    return Table(schema=schema, name=name)


def _pub(name: str = "feed", *tables: PublicationTable, **kwargs) -> Publication:
    return Publication(
        name=name, tables={t.qualified_name: t for t in tables}, **kwargs
    )


def _member(name: str, schema: str = "public", **kwargs) -> PublicationTable:
    return PublicationTable(schema=schema, name=name, **kwargs)


def _schema(*pubs: Publication, tables: tuple[Table, ...] = (), **kwargs) -> Schema:
    return Schema(
        publications={p.name: p for p in pubs},
        tables={t.qualified_name: t for t in tables},
        **kwargs,
    )


class TestPublicationDiff:
    def test_equal_publications_produce_no_changes(self):
        pub = _pub("feed", _member("a", columns=("x",), row_filter="(x > 1)"))
        assert compute_diff(_schema(pub), _schema(pub)) == []

    def test_create(self):
        pub = _pub("feed", _member("a"), _member("b", columns=("x", "y")))
        changes = compute_diff(_schema(pub), Schema())

        assert [type(c) for c in changes] == [CreatePublication]
        assert changes[0].to_sql() == (
            'CREATE PUBLICATION "feed" FOR TABLE "public"."a", "public"."b" '
            '("x", "y") WITH (publish = \'insert, update, delete, truncate\', '
            "publish_via_partition_root = false);"
        )

    def test_create_with_schemas_filter_and_options(self):
        pub = _pub(
            "feed",
            _member("a", row_filter="(id > 0)"),
            schemas=("audit",),
            publish=("insert",),
            via_root=True,
            generated_columns="stored",
        )
        sql = CreatePublication(publication=pub).to_sql()

        assert sql == (
            'CREATE PUBLICATION "feed" FOR TABLE "public"."a" WHERE ((id > 0)), '
            "TABLES IN SCHEMA \"audit\" WITH (publish = 'insert', "
            "publish_via_partition_root = true, publish_generated_columns = stored);"
        )

    def test_empty_and_all_tables(self):
        assert (
            CreatePublication(publication=_pub("e"))
            .to_sql()
            .startswith('CREATE PUBLICATION "e" WITH (')
        )
        assert 'CREATE PUBLICATION "e" FOR ALL TABLES WITH (' in (
            CreatePublication(publication=_pub("e", all_tables=True)).to_sql()
        )

    def test_membership_converges_with_drop_replace_add(self):
        actual = _pub("feed", _member("a"), _member("b"), _member("c"))
        desired = _pub("feed", _member("a"), _member("b", columns=("x",)), _member("d"))
        changes = compute_diff(_schema(desired), _schema(actual))

        assert [type(c) for c in changes] == [
            DropPublicationObjects,
            ReplacePublicationTable,
            AddPublicationObjects,
        ]
        drop, replace, add = changes
        assert isinstance(drop, DropPublicationObjects)
        assert [t.name for t in drop.tables] == ["c"]
        assert 'DROP TABLE "public"."c"' in drop.to_sql()
        assert "to_regclass" in drop.to_sql()  # only while still a member
        assert isinstance(replace, ReplacePublicationTable)
        assert 'ADD TABLE "public"."b" ("x")' in replace.to_sql()
        assert add.to_sql() == 'ALTER PUBLICATION "feed" ADD TABLE "public"."d";'

    def test_schema_membership(self):
        actual = _pub("feed", schemas=("a", "b"))
        desired = _pub("feed", schemas=("b", "c"))
        changes = compute_diff(_schema(desired), _schema(actual))

        assert [c.to_sql() for c in changes] == [
            'ALTER PUBLICATION "feed" DROP TABLES IN SCHEMA "a";',
            'ALTER PUBLICATION "feed" ADD TABLES IN SCHEMA "c";',
        ]

    def test_options_are_set_in_place(self):
        changes = compute_diff(
            _schema(_pub("feed", publish=("insert", "update"), via_root=True)),
            _schema(_pub("feed")),
        )

        assert [type(c) for c in changes] == [AlterPublicationOptions]
        assert changes[0].to_sql() == (
            "ALTER PUBLICATION \"feed\" SET (publish = 'insert, update', "
            "publish_via_partition_root = true);"
        )

    def test_switching_all_tables_recreates(self):
        changes = compute_diff(
            _schema(_pub("feed", all_tables=True)), _schema(_pub("feed", _member("a")))
        )

        assert [type(c) for c in changes] == [DropAndCreatePublication]
        assert changes[0].to_sql().startswith('DROP PUBLICATION IF EXISTS "feed";\n')

    def test_undeclared_publication_on_managed_tables_is_dropped(self):
        actual = _schema(_pub("old", _member("a")), tables=(_table("a"),))
        changes = compute_diff(Schema(tables=actual.tables), actual)

        assert [type(c) for c in changes] == [DropPublication]
        assert changes[0].to_sql() == 'DROP PUBLICATION IF EXISTS "old";'

    @pytest.mark.parametrize(
        "pub",
        [
            _pub("all", all_tables=True),
            _pub("empty"),
            _pub("outside", _member("a"), _member("x", schema="other")),
            _pub("ignored", _member("_hidden")),
            _pub("schema_outside", schemas=("other",)),
        ],
        ids=["all-tables", "empty", "foreign-table", "unmanaged-table", "schema"],
    )
    def test_undeclared_publication_reaching_outside_is_kept(self, pub):
        actual = _schema(pub, tables=(_table("a"),))
        assert compute_diff(Schema(tables=actual.tables), actual) == []

    def test_member_of_a_dropped_table_is_guarded(self):
        actual = _schema(
            _pub("feed", _member("a"), _member("b")),
            tables=(
                _table("a"),
                _table("b"),
            ),
        )
        desired = _schema(_pub("feed", _member("a")), tables=(_table("a"),))
        changes = compute_diff(desired, actual)

        assert {type(c) for c in changes} == {DropTable, DropPublicationObjects}

    def test_names_are_quoted(self):
        pub = _pub(
            'f"; DROP',
            _member('t"x', schema='s"y', columns=('c"z',)),
            schemas=('q"w',),
        )
        sql = CreatePublication(publication=pub).to_sql()
        assert '"f""; DROP"' in sql
        assert '"s""y"."t""x" ("c""z")' in sql
        assert 'TABLES IN SCHEMA "q""w"' in sql

        drop = DropPublicationObjects(
            name="it's", tables=[_member("o'k")], schemas=[]
        ).to_sql()
        # The inner statement and the lookups are literals, quotes doubled.
        assert "p.pubname = 'it''s'" in drop
        assert "to_regclass('\"public\".\"o''k\"')" in drop


def _rule(**kwargs) -> Rule:
    base = {
        "schema": "public",
        "table": "t",
        "name": "r",
        "definition": "CREATE RULE r AS\n    ON DELETE TO public.t DO INSTEAD NOTHING;",
    }
    return Rule(**{**base, **kwargs})


def _rules(*rules: Rule, tables: tuple[Table, ...] = ()) -> Schema:
    return Schema(
        rules={r.qualified_key: r for r in rules},
        tables={t.qualified_name: t for t in tables},
    )


class TestRuleDiff:
    def test_equal_rules_produce_no_changes(self):
        assert compute_diff(_rules(_rule()), _rules(_rule())) == []

    def test_create_uses_or_replace(self):
        changes = compute_diff(_rules(_rule()), Schema())

        assert [type(c) for c in changes] == [CreateRule]
        assert changes[0].to_sql() == (
            "CREATE OR REPLACE RULE r AS\n    ON DELETE TO public.t DO INSTEAD NOTHING;"
        )

    def test_disabled_rule_states_its_state(self):
        sql = CreateRule(rule=_rule(enabled="D")).to_sql()
        assert sql.endswith('\nALTER TABLE "public"."t" DISABLE RULE "r";')

    def test_changed_definition_replaces_and_restates(self):
        changed = _rule(
            definition="CREATE RULE r AS\n    ON UPDATE TO public.t DO INSTEAD NOTHING;"
        )
        changes = compute_diff(_rules(changed), _rules(_rule(enabled="A")))

        assert [type(c) for c in changes] == [CreateRule]
        assert changes[0].to_sql().endswith('ENABLE RULE "r";')

    def test_state_only_change(self):
        changes = compute_diff(_rules(_rule(enabled="R")), _rules(_rule()))

        assert [type(c) for c in changes] == [AlterRuleEnabled]
        assert changes[0].to_sql() == (
            'ALTER TABLE "public"."t" ENABLE REPLICA RULE "r";'
        )

    def test_drop(self):
        changes = compute_diff(_rules(), _rules(_rule()))

        assert [c.to_sql() for c in changes] == [
            'DROP RULE IF EXISTS "r" ON "public"."t";'
        ]

    def test_rule_goes_with_its_dropped_table(self):
        changes = compute_diff(Schema(), _rules(_rule(), tables=(_table("t"),)))

        assert [type(c) for c in changes] == [DropTable]
        assert not any(isinstance(c, DropRule) for c in changes)


def _evt(**kwargs) -> EventTrigger:
    base = {
        "name": "audit",
        "event": "ddl_command_end",
        "function_schema": "public",
        "function_name": "on_ddl",
    }
    return EventTrigger(**{**base, **kwargs})


def _evts(*triggers: EventTrigger, **kwargs) -> Schema:
    return Schema(event_triggers={t.name: t for t in triggers}, **kwargs)


class TestEventTriggerDiff:
    def test_equal_produce_no_changes(self):
        evt = _evt(tags=("CREATE TABLE",))
        assert compute_diff(_evts(evt), _evts(evt)) == []

    def test_create(self):
        changes = compute_diff(
            _evts(_evt(tags=("ALTER TABLE", "CREATE TABLE"), enabled="A")), Schema()
        )

        assert [type(c) for c in changes] == [CreateEventTrigger]
        assert changes[0].to_sql() == (
            'CREATE EVENT TRIGGER "audit" ON "ddl_command_end" '
            "WHEN TAG IN ('ALTER TABLE', 'CREATE TABLE') "
            'EXECUTE FUNCTION "public"."on_ddl"();\n'
            'ALTER EVENT TRIGGER "audit" ENABLE ALWAYS;'
        )

    @pytest.mark.parametrize(
        "change",
        [
            {"event": "sql_drop"},
            {"tags": ("DROP TABLE",)},
            {"function_name": "other"},
        ],
    )
    def test_definition_change_recreates(self, change):
        changes = compute_diff(_evts(_evt(**change)), _evts(_evt()))

        assert [type(c) for c in changes] == [CreateEventTrigger]
        assert (
            changes[0]
            .to_sql()
            .startswith('DROP EVENT TRIGGER IF EXISTS "audit";\nCREATE EVENT TRIGGER')
        )

    def test_enabled_change_alters(self):
        changes = compute_diff(_evts(_evt(enabled="D")), _evts(_evt()))

        assert [type(c) for c in changes] == [AlterEventTriggerEnabled]
        assert changes[0].to_sql() == 'ALTER EVENT TRIGGER "audit" DISABLE;'

    def test_drop_only_when_the_function_is_managed(self):
        actual = _evts(_evt(), _evt(name="theirs", function_schema="ops"))
        changes = compute_diff(Schema(), actual)

        assert [type(c) for c in changes] == [DropEventTrigger]
        assert changes[0].to_sql() == 'DROP EVENT TRIGGER IF EXISTS "audit";'

    def test_recreated_function_recreates_its_event_trigger(self):
        # A signature change drops the function with CASCADE, which takes the
        # event trigger along; the diff has to build it again.
        old = Function(
            schema="public", name="on_ddl", args="x integer", full_definition="old"
        )
        new = Function(schema="public", name="on_ddl", args="", full_definition="new")
        changes = compute_diff(
            _evts(_evt(), functions={new.signature: new}),
            _evts(_evt(), functions={old.signature: old}),
        )

        assert [type(c) for c in changes] == [DropAndCreateFunction, CreateEventTrigger]
        assert isinstance(changes[1], CreateEventTrigger) and changes[1].replace


def test_event_triggers_build_last_in_the_plan():
    statements = parse_sql_content(
        "CREATE EVENT TRIGGER e ON ddl_command_end EXECUTE FUNCTION f();\n"
        "-- note\nALTER EVENT TRIGGER e DISABLE;\n"
        "CREATE TABLE t (id int);\n"
    )
    ordered = _event_triggers_last(statements)
    assert "CREATE TABLE" in ordered[0].raw_sql
    assert [s.raw_sql for s in ordered[1:]] == [s.raw_sql for s in statements[:2]]


# =============================================================================
# Integration tests
# =============================================================================


@pytest.fixture
def server_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


def _with_db(url: str, name: str) -> str:
    return url.rsplit("/", 1)[0] + f"/{name}"


def _admin(url: str, sql: str) -> None:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


def _rows(url: str, sql: str) -> list[tuple]:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


def _server_version(url: str) -> int:
    conn = psycopg2.connect(url)
    try:
        return conn.server_version
    finally:
        conn.close()


@pytest.fixture
def databases(server_url):
    """A target database and a plan database, both empty, both dropped after."""
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_pub_target_{suffix}", f"pgdeclare_pub_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def superuser(server_url):
    if not _rows(
        server_url, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
    )[0][0]:
        pytest.skip("event triggers need a superuser")


QUIET = PgDeclareConfig(report=lambda _line: None)


def _apply(sql_dir: Path, sql: str, target_url: str, plan_db: str, config=QUIET):
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "schema.sql").write_text(sql)
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)


def _pending(sql_dir: Path, target_url: str, plan_db: str, config=QUIET):
    desired = build_desired_schema(sql_dir, target_url, plan_db, config=config)
    actual = introspect_schema(target_url, config=config)
    return compute_diff(desired, actual, schemas=list(config.schemas), config=config)


def _publications(url: str) -> list[tuple]:
    return _rows(
        url,
        "SELECT pubname, puballtables, pubinsert, pubupdate, pubdelete, "
        "pubtruncate, pubviaroot FROM pg_publication ORDER BY pubname",
    )


def _members(url: str, pub: str) -> list[str]:
    return [
        r[0]
        for r in _rows(
            url,
            "SELECT c.relname FROM pg_publication_rel pr "
            "JOIN pg_publication p ON p.oid = pr.prpubid "
            "JOIN pg_class c ON c.oid = pr.prrelid "
            f"WHERE p.pubname = '{pub}' ORDER BY c.relname",
        )
    ]


TABLES = """
CREATE TABLE note (id int PRIMARY KEY, body text, secret text);
CREATE TABLE tag (id int PRIMARY KEY, label text);
CREATE TABLE log (id int PRIMARY KEY, line text);
"""


class TestPublicationRoundTrip:
    def test_create_change_and_drop(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        v1 = TABLES + (
            "CREATE PUBLICATION feed FOR TABLE note, tag WITH (publish = 'insert, update');\n"
            "CREATE PUBLICATION empty_pub;\n"
        )
        _apply(sql_dir, v1, target_url, plan_db)

        assert _publications(target_url) == [
            ("empty_pub", False, True, True, True, True, False),
            ("feed", False, True, True, False, False, False),
        ]
        assert _members(target_url, "feed") == ["note", "tag"]
        assert _pending(sql_dir, target_url, plan_db) == []

        v2 = TABLES + (
            "CREATE PUBLICATION feed FOR TABLE tag, log "
            "WITH (publish = 'insert', publish_via_partition_root = true);\n"
        )
        _apply(sql_dir, v2, target_url, plan_db)
        # empty_pub is no longer declared but is empty, so it cannot be told
        # apart from someone else's: kept.
        assert _publications(target_url) == [
            ("empty_pub", False, True, True, True, True, False),
            ("feed", False, True, False, False, False, True),
        ]
        assert _members(target_url, "feed") == ["log", "tag"]
        assert _pending(sql_dir, target_url, plan_db) == []

        _apply(sql_dir, TABLES, target_url, plan_db)
        assert [p[0] for p in _publications(target_url)] == ["empty_pub"]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_dropping_a_published_table(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(
            sql_dir,
            TABLES + "CREATE PUBLICATION feed FOR TABLE note, tag;",
            target_url,
            plan_db,
        )
        without_tag = TABLES.replace(
            "CREATE TABLE tag (id int PRIMARY KEY, label text);", ""
        )
        _apply(
            sql_dir,
            without_tag + "CREATE PUBLICATION feed FOR TABLE note;",
            target_url,
            plan_db,
        )
        assert _members(target_url, "feed") == ["note"]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_all_tables_switch(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(
            sql_dir,
            TABLES + "CREATE PUBLICATION feed FOR TABLE note;",
            target_url,
            plan_db,
        )
        _apply(
            sql_dir,
            TABLES + "CREATE PUBLICATION feed FOR ALL TABLES;",
            target_url,
            plan_db,
        )
        assert _publications(target_url)[0][:2] == ("feed", True)
        assert _pending(sql_dir, target_url, plan_db) == []
        _apply(
            sql_dir,
            TABLES + "CREATE PUBLICATION feed FOR TABLE log;",
            target_url,
            plan_db,
        )
        assert _publications(target_url)[0][:2] == ("feed", False)
        assert _members(target_url, "feed") == ["log"]

    def test_safety_rule_on_undeclared_publications(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _admin(
            target_url,
            "CREATE SCHEMA other; CREATE TABLE other.ext (id int);"
            + TABLES
            + "CREATE TABLE _hidden (id int);"
            "CREATE PUBLICATION mine FOR TABLE note;"
            "CREATE PUBLICATION mixed FOR TABLE note, other.ext;"
            "CREATE PUBLICATION hidden FOR TABLE _hidden;"
            "CREATE PUBLICATION everything FOR ALL TABLES;",
        )
        _apply(sql_dir, TABLES, target_url, plan_db)

        assert [p[0] for p in _publications(target_url)] == [
            "everything",
            "hidden",
            "mixed",
        ]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_column_lists_row_filters_and_schemas(self, tmp_path, databases):
        target_url, plan_db = databases
        if _server_version(target_url) < 150000:
            pytest.skip("column lists, row filters and TABLES IN SCHEMA need 15+")
        sql_dir = tmp_path / "sql"
        base = TABLES
        # The library leaves creating managed schemas to the caller.
        _admin(target_url, "CREATE SCHEMA audit")
        config = PgDeclareConfig(
            schemas=("public", "audit"),
            plan_recreate_schemas=("public", "audit"),
            report=lambda _l: None,
        )
        v1 = base + (
            "CREATE TABLE audit.event (id int PRIMARY KEY);\n"
            "CREATE PUBLICATION feed FOR TABLE note (id, body) WHERE (id > 10), tag;\n"
            "CREATE PUBLICATION by_schema FOR TABLES IN SCHEMA audit;\n"
        )
        _apply(sql_dir, v1, target_url, plan_db, config)
        assert _rows(
            target_url,
            "SELECT attnames, rowfilter FROM pg_publication_tables "
            "WHERE pubname = 'feed' AND tablename = 'note'",
        ) == [(["id", "body"], "(id > 10)")]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        v2 = base + (
            "CREATE TABLE audit.event (id int PRIMARY KEY);\n"
            "CREATE PUBLICATION feed FOR TABLE note (id) WHERE (id > 20), tag;\n"
            "CREATE PUBLICATION by_schema FOR TABLE log, TABLES IN SCHEMA audit;\n"
        )
        _apply(sql_dir, v2, target_url, plan_db, config)
        assert _rows(
            target_url,
            "SELECT attnames, rowfilter FROM pg_publication_tables "
            "WHERE pubname = 'feed' AND tablename = 'note'",
        ) == [(["id"], "(id > 20)")]
        assert _members(target_url, "by_schema") == ["log"]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # An undeclared publication over a managed schema is dropped.
        _apply(
            sql_dir,
            base + "CREATE TABLE audit.event (id int PRIMARY KEY);",
            target_url,
            plan_db,
            config,
        )
        assert _publications(target_url) == []

    def test_awkward_names(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        sql = (
            'CREATE TABLE "My ""Tab""; x" (id int);\n'
            'CREATE PUBLICATION "Pub ""1""; y" FOR TABLE "My ""Tab""; x";\n'
        )
        _apply(sql_dir, sql, target_url, plan_db)
        assert _pending(sql_dir, target_url, plan_db) == []
        _apply(
            sql_dir,
            'CREATE TABLE "My ""Tab""; x" (id int);',
            target_url,
            plan_db,
        )
        assert _publications(target_url) == []

    def test_reused_plan_database_forgets_removed_publications(
        self, tmp_path, databases
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(
            sql_dir,
            TABLES + "CREATE PUBLICATION feed FOR TABLE note;",
            target_url,
            plan_db,
        )
        _apply(sql_dir, TABLES, target_url, plan_db)
        assert _publications(target_url) == []


RULE_SQL = """
CREATE TABLE note (id int PRIMARY KEY, body text);
CREATE TABLE note_log (id int, body text);
CREATE VIEW note_view AS SELECT id, body FROM note;
"""


class TestRuleRoundTrip:
    def _rules(self, url: str) -> list[tuple]:
        return _rows(
            url,
            "SELECT c.relname, r.rulename, r.ev_type, r.ev_enabled FROM pg_rewrite r "
            "JOIN pg_class c ON c.oid = r.ev_class "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' ORDER BY 1, 2",
        )

    def test_create_change_and_drop(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        v1 = RULE_SQL + (
            "CREATE RULE keep_notes AS ON DELETE TO note DO INSTEAD NOTHING;\n"
            "CREATE RULE log_notes AS ON INSERT TO note "
            "DO ALSO INSERT INTO note_log VALUES (NEW.id, NEW.body);\n"
        )
        _apply(sql_dir, v1, target_url, plan_db)
        assert self._rules(target_url) == [
            ("note", "keep_notes", "4", "O"),
            ("note", "log_notes", "3", "O"),
            ("note_view", "_RETURN", "1", "O"),
        ]
        assert introspect_schema(target_url).rules.keys() == {
            "public.note.keep_notes",
            "public.note.log_notes",
        }
        assert _pending(sql_dir, target_url, plan_db) == []

        v2 = RULE_SQL + (
            "CREATE RULE keep_notes AS ON UPDATE TO note DO INSTEAD NOTHING;\n"
            "ALTER TABLE note DISABLE RULE keep_notes;\n"
        )
        _apply(sql_dir, v2, target_url, plan_db)
        assert self._rules(target_url)[:1] == [("note", "keep_notes", "2", "D")]
        assert len(self._rules(target_url)) == 2
        assert _pending(sql_dir, target_url, plan_db) == []

        _apply(sql_dir, RULE_SQL, target_url, plan_db)
        assert self._rules(target_url) == [("note_view", "_RETURN", "1", "O")]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_rule_on_a_dropped_table(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(
            sql_dir,
            RULE_SQL + "CREATE RULE r AS ON DELETE TO note_log DO INSTEAD NOTHING;",
            target_url,
            plan_db,
        )
        _apply(
            sql_dir,
            RULE_SQL.replace("CREATE TABLE note_log (id int, body text);", ""),
            target_url,
            plan_db,
        )
        assert _pending(sql_dir, target_url, plan_db) == []


EVT_FUNCTIONS = """
CREATE TABLE ddl_log (tag text);
CREATE FUNCTION log_ddl() RETURNS event_trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO public.ddl_log VALUES (tg_tag); END $$;
CREATE FUNCTION log_drop() RETURNS event_trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO public.ddl_log VALUES ('drop:' || tg_tag); END $$;
"""


class TestEventTriggerRoundTrip:
    def _triggers(self, url: str) -> list[tuple]:
        return _rows(
            url,
            "SELECT evtname, evtevent, evtfoid::regproc::text, evttags, evtenabled "
            "FROM pg_event_trigger ORDER BY evtname",
        )

    def test_create_change_and_drop(self, tmp_path, databases, superuser):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        v1 = EVT_FUNCTIONS + (
            "CREATE EVENT TRIGGER audit ON ddl_command_end "
            "WHEN TAG IN ('CREATE TABLE', 'ALTER TABLE') EXECUTE FUNCTION log_ddl();\n"
        )
        _apply(sql_dir, v1, target_url, plan_db)
        assert self._triggers(target_url) == [
            (
                "audit",
                "ddl_command_end",
                "log_ddl",
                ["ALTER TABLE", "CREATE TABLE"],
                "O",
            )
        ]
        assert _pending(sql_dir, target_url, plan_db) == []
        _admin(target_url, "CREATE TABLE fires (id int)")
        assert _rows(target_url, "SELECT tag FROM ddl_log") == [("CREATE TABLE",)]

        v2 = EVT_FUNCTIONS + (
            "CREATE EVENT TRIGGER audit ON sql_drop EXECUTE FUNCTION log_drop();\n"
            "ALTER EVENT TRIGGER audit ENABLE ALWAYS;\n"
        )
        _apply(sql_dir, v2, target_url, plan_db)
        assert self._triggers(target_url) == [
            ("audit", "sql_drop", "log_drop", None, "A")
        ]
        assert _pending(sql_dir, target_url, plan_db) == []

        v3 = v2.replace("ENABLE ALWAYS", "DISABLE")
        _apply(sql_dir, v3, target_url, plan_db)
        assert self._triggers(target_url)[0][4] == "D"
        assert _pending(sql_dir, target_url, plan_db) == []

        _apply(sql_dir, EVT_FUNCTIONS, target_url, plan_db)
        assert self._triggers(target_url) == []
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_undeclared_with_unmanaged_function_is_kept(
        self, tmp_path, databases, superuser
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _admin(
            target_url,
            "CREATE SCHEMA ops;"
            "CREATE FUNCTION ops.f() RETURNS event_trigger LANGUAGE plpgsql "
            "AS $$ BEGIN END $$;"
            "CREATE EVENT TRIGGER theirs ON ddl_command_start EXECUTE FUNCTION ops.f();"
            "CREATE FUNCTION public.g() RETURNS event_trigger LANGUAGE plpgsql "
            "AS $$ BEGIN END $$;"
            "CREATE EVENT TRIGGER stale ON ddl_command_start EXECUTE FUNCTION public.g();",
        )
        _apply(
            sql_dir,
            "CREATE FUNCTION g() RETURNS event_trigger LANGUAGE plpgsql AS $$ BEGIN END $$;",
            target_url,
            plan_db,
        )
        assert [t[0] for t in self._triggers(target_url)] == ["theirs"]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_awkward_names(self, tmp_path, databases, superuser):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        sql = (
            'CREATE FUNCTION "f ""1""; x"() RETURNS event_trigger LANGUAGE plpgsql '
            "AS $$ BEGIN END $$;\n"
            'CREATE EVENT TRIGGER "E ""t""; y" ON ddl_command_end '
            'WHEN TAG IN (\'CREATE TABLE\') EXECUTE FUNCTION "f ""1""; x"();\n'
        )
        _apply(sql_dir, sql, target_url, plan_db)
        assert _pending(sql_dir, target_url, plan_db) == []
        assert [t[0] for t in self._triggers(target_url)] == ['E "t"; y']


def test_dropping_a_column_in_a_column_list(tmp_path, databases):
    target_url, plan_db = databases
    if _server_version(target_url) < 150000:
        pytest.skip("column lists need PostgreSQL 15+")
    sql_dir = tmp_path / "sql"
    table = "CREATE TABLE note (id int PRIMARY KEY, body text, extra text);\n"
    _apply(
        sql_dir,
        table + "CREATE PUBLICATION feed FOR TABLE note (id, body, extra);",
        target_url,
        plan_db,
    )
    _admin(target_url, "INSERT INTO note VALUES (1, 'a', NULL)")
    _apply(
        sql_dir,
        table.replace(", extra text", "")
        + "CREATE PUBLICATION feed FOR TABLE note (id, body);",
        target_url,
        plan_db,
    )
    assert _rows(
        target_url, "SELECT attnames FROM pg_publication_tables WHERE pubname = 'feed'"
    ) == [(["id", "body"],)]
    assert _pending(sql_dir, target_url, plan_db) == []
