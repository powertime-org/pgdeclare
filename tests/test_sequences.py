"""Standalone sequences: CREATE SEQUENCE, serial columns, OWNED BY, grants.

The unit tests drive the differ and the rendered SQL. The integration tests
run the whole pipeline (plan database, introspection, diff, apply, convergence
check) against a real server and need TEST_DATABASE_URL.
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
    AlterSequence,
    CreateSequence,
    CreateTable,
    DropDefault,
    DropSequence,
    DropTable,
    GrantPrivilege,
    RevokePrivilege,
    SetSequenceOwner,
)
from pgdeclare.models import Column, Grant, Schema, Sequence, Table


# =============================================================================
# Unit tests
# =============================================================================


def _seq(**kwargs) -> Sequence:
    return Sequence(**{"schema": "public", "name": "order_no", **kwargs})


def _schema(*sequences: Sequence, tables: tuple[Table, ...] = ()) -> Schema:
    return Schema(
        sequences={s.qualified_name: s for s in sequences},
        tables={t.qualified_name: t for t in tables},
    )


def _table_using(seq_name: str) -> Table:
    return Table(
        schema="public",
        name="orders",
        columns={
            "n": Column(
                name="n",
                type="integer",
                nullable=False,
                default=f"nextval('{seq_name}'::regclass)",
            )
        },
    )


class TestSequenceDiff:
    def test_equal_sequences_produce_no_changes(self):
        seq = _seq(start=100, owned_by=("public", "orders", "n"))
        assert (
            compute_diff(_schema(seq), _schema(_seq(start=100, owned_by=seq.owned_by)))
            == []
        )

    def test_create_comes_before_the_table_and_owned_by_after(self):
        seq = _seq(owned_by=("public", "orders", "n"))
        changes = compute_diff(
            _schema(seq, tables=(_table_using("order_no"),)), Schema()
        )

        kinds = [type(c) for c in changes]
        assert kinds == [CreateSequence, CreateTable, SetSequenceOwner]

    def test_option_change_is_an_alter_never_a_restart(self):
        changes = compute_diff(
            _schema(_seq(start=500, increment=3, cycle=True)), _schema(_seq())
        )

        assert [type(c) for c in changes] == [AlterSequence]
        sql = changes[0].to_sql()
        assert "START WITH 500" in sql
        assert "INCREMENT BY 3" in sql
        assert " CYCLE" in sql and "NO CYCLE" not in sql
        assert "RESTART" not in sql

    def test_ownership_change_releases_before_tables_and_sets_after(self):
        old = _seq(owned_by=("public", "orders", "old"))
        new = _seq(owned_by=("public", "orders", "n"))
        changes = compute_diff(_schema(new), _schema(old))

        assert [type(c) for c in changes] == [SetSequenceOwner, SetSequenceOwner]
        assert changes[0].to_sql().endswith("OWNED BY NONE;")
        assert changes[1].to_sql().endswith('OWNED BY "public"."orders"."n";')

    def test_undeclared_sequence_drops_after_the_default_that_used_it(self):
        actual = _schema(_seq(), tables=(_table_using("order_no"),))
        desired_table = _table_using("order_no")
        desired_table.columns["n"].default = None
        changes = compute_diff(_schema(tables=(desired_table,)), actual)

        assert [type(c) for c in changes] == [DropDefault, DropSequence]
        assert changes[1].to_sql() == 'DROP SEQUENCE IF EXISTS "public"."order_no";'

    def test_drop_has_no_cascade(self):
        changes = compute_diff(
            Schema(), _schema(_seq(), tables=(_table_using("order_no"),))
        )

        assert [type(c) for c in changes] == [DropTable, DropSequence]
        assert "CASCADE" not in changes[1].to_sql()


class TestSequenceSql:
    def test_create_spells_out_every_option(self):
        sql = CreateSequence(
            sequence=_seq(
                data_type="integer",
                increment=-1,
                min_value=-100,
                max_value=-1,
                start=-1,
                cache=20,
            )
        ).to_sql()

        assert sql == (
            'CREATE SEQUENCE "public"."order_no" AS integer INCREMENT BY -1 '
            "MINVALUE -100 MAXVALUE -1 START WITH -1 CACHE 20 NO CYCLE;"
        )

    def test_names_are_quoted(self):
        seq = Sequence(
            schema='odd"schema',
            name='seq"; DROP TABLE x; --',
            owned_by=('odd"schema', 't"t', 'c"c'),
        )

        assert (
            CreateSequence(sequence=seq)
            .to_sql()
            .startswith(
                'CREATE SEQUENCE "odd""schema"."seq""; DROP TABLE x; --" AS bigint'
            )
        )
        assert SetSequenceOwner(sequence=seq).to_sql() == (
            'ALTER SEQUENCE "odd""schema"."seq""; DROP TABLE x; --" '
            'OWNED BY "odd""schema"."t""t"."c""c";'
        )
        assert DropSequence(schema=seq.schema, sequence_name=seq.name).to_sql() == (
            'DROP SEQUENCE IF EXISTS "odd""schema"."seq""; DROP TABLE x; --";'
        )

    def test_unexpected_type_is_refused(self):
        with pytest.raises(ValueError, match="unexpected type"):
            CreateSequence(sequence=_seq(data_type="bigint; DROP TABLE x")).to_sql()

    def test_grant_and_revoke_name_the_sequence(self):
        grant = Grant(
            object_type="SEQUENCE",
            object_name="order_no",
            schema="public",
            privilege="USAGE",
            grantee="app",
        )

        assert (
            GrantPrivilege(grant=grant).to_sql()
            == 'GRANT USAGE ON SEQUENCE "public"."order_no" TO app;'
        )
        revoke = RevokePrivilege(grant=grant).to_sql()
        assert 'REVOKE USAGE ON SEQUENCE "public"."order_no" FROM app' in revoke
        assert "undefined_table" in revoke


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


@pytest.fixture
def databases(server_url):
    """A target database and a plan database, both empty, both dropped after."""
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_seq_target_{suffix}", f"pgdeclare_seq_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


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


def _options(url: str, name: str) -> tuple:
    return _rows(
        url,
        "SELECT format_type(seqtypid, NULL), seqincrement, seqmin, seqmax, "
        "seqstart, seqcache, seqcycle FROM pg_sequence "
        f"WHERE seqrelid = 'public.{name}'::regclass",
    )[0]


def _owner(url: str, name: str):
    return _rows(
        url,
        "SELECT t.relname, a.attname FROM pg_depend d "
        "JOIN pg_class t ON t.oid = d.refobjid "
        "JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid "
        f"WHERE d.objid = 'public.{name}'::regclass AND d.deptype = 'a'",
    )


def _exists(url: str, name: str) -> bool:
    return _rows(url, f"SELECT to_regclass('public.{name}') IS NOT NULL")[0][0]


ORDERS_V1 = """
CREATE SEQUENCE IF NOT EXISTS order_no AS integer START 1000 INCREMENT 5 CACHE 10;

CREATE TABLE IF NOT EXISTS orders (
    n     integer NOT NULL DEFAULT nextval('order_no'),
    label text
);

ALTER SEQUENCE order_no OWNED BY orders.n;

CREATE SEQUENCE IF NOT EXISTS ticket_no;

CREATE TABLE IF NOT EXISTS note (
    id   serial PRIMARY KEY,
    big  bigserial,
    body text
);
"""


class TestSequenceRoundTrip:
    def test_declared_sequences_create_and_converge(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"

        _apply(sql_dir, ORDERS_V1, target_url, plan_db)

        assert _options(target_url, "order_no") == (
            "integer",
            5,
            1,
            2147483647,
            1000,
            10,
            False,
        )
        assert _owner(target_url, "order_no") == [("orders", "n")]
        assert _owner(target_url, "ticket_no") == []
        assert _owner(target_url, "note_id_seq") == [("note", "id")]
        assert _owner(target_url, "note_big_seq") == [("note", "big")]
        _admin(target_url, "INSERT INTO orders (label) VALUES ('a'), ('b')")
        assert _rows(target_url, "SELECT n FROM orders ORDER BY n") == [
            (1000,),
            (1005,),
        ]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_option_change_keeps_the_current_value(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ORDERS_V1, target_url, plan_db)
        _admin(target_url, "INSERT INTO orders (label) VALUES ('a'), ('b'), ('c')")
        # With CACHE 10 the session reserved ten values, so last_value is the
        # end of that block, not the last value handed out.
        [(last_value,)] = _rows(target_url, "SELECT last_value FROM order_no")
        assert last_value >= 1010

        changed = ORDERS_V1.replace(
            "AS integer START 1000 INCREMENT 5 CACHE 10",
            "AS bigint START 1 INCREMENT 7 MAXVALUE 999999 CACHE 1 CYCLE",
        )
        _apply(sql_dir, changed, target_url, plan_db)

        assert _options(target_url, "order_no") == (
            "bigint",
            7,
            1,
            999999,
            1,
            1,
            True,
        )
        # The next value follows on with the new increment instead of
        # restarting at the new START.
        assert _rows(target_url, "SELECT last_value FROM order_no") == [(last_value,)]
        assert _rows(target_url, "SELECT nextval('order_no')") == [(last_value + 7,)]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_serial_columns_made_outside_the_engine_show_no_diff(
        self, tmp_path, databases
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        (sql_dir / "schemas").mkdir(parents=True)
        (sql_dir / "schemas" / "schema.sql").write_text(ORDERS_V1)
        # The same schema, built by hand rather than by the engine.
        _admin(target_url, ORDERS_V1)
        _admin(target_url, "SELECT nextval('note_id_seq'), nextval('note_id_seq')")

        assert _pending(sql_dir, target_url, plan_db) == []

    def test_identity_sequences_stay_out_of_the_model(self, databases):
        target_url, _plan_db = databases
        _admin(target_url, "CREATE TABLE t (id bigint GENERATED ALWAYS AS IDENTITY)")

        assert introspect_schema(target_url).sequences == {}

    def test_owned_by_moves_and_is_released(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ORDERS_V1, target_url, plan_db)

        # Ownership moves to a column created in the same run, and the old
        # owning column is dropped: the sequence must survive both.
        moved = ORDERS_V1.replace(
            "n     integer NOT NULL DEFAULT nextval('order_no'),",
            "m     integer NOT NULL DEFAULT nextval('order_no'),",
        ).replace("OWNED BY orders.n", "OWNED BY orders.m")
        _apply(sql_dir, moved, target_url, plan_db)
        assert _owner(target_url, "order_no") == [("orders", "m")]

        released = moved.replace("ALTER SEQUENCE order_no OWNED BY orders.m;", "")
        _apply(sql_dir, released, target_url, plan_db)
        assert _owner(target_url, "order_no") == []
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_undeclared_sequence_is_dropped(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ORDERS_V1, target_url, plan_db)

        # ticket_no goes; order_no goes together with the default using it.
        without = ORDERS_V1.replace(
            "CREATE SEQUENCE IF NOT EXISTS ticket_no;", ""
        ).replace(
            "CREATE SEQUENCE IF NOT EXISTS order_no AS integer START 1000 INCREMENT 5 CACHE 10;",
            "",
        )
        without = without.replace(
            "n     integer NOT NULL DEFAULT nextval('order_no'),",
            "n     integer NOT NULL,",
        ).replace("ALTER SEQUENCE order_no OWNED BY orders.n;", "")
        _apply(sql_dir, without, target_url, plan_db)

        assert not _exists(target_url, "ticket_no")
        assert not _exists(target_url, "order_no")
        assert _exists(target_url, "note_id_seq")
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_drop_is_refused_while_a_default_still_uses_it(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, "CREATE SEQUENCE shared_no;", target_url, plan_db)
        # A table the engine ignores (leading underscore) still draws from it.
        _admin(
            target_url,
            "CREATE TABLE _legacy (n bigint DEFAULT nextval('shared_no'))",
        )

        with pytest.raises(Exception):
            _apply(sql_dir, "CREATE TABLE keep (id int);", target_url, plan_db)
        assert _exists(target_url, "shared_no")

    def test_awkward_names_round_trip(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        sql = """
        CREATE SEQUENCE "Odd ""Seq""; x" START 5;
        CREATE TABLE "My Table" (v bigint DEFAULT nextval('"Odd ""Seq""; x"'));
        ALTER SEQUENCE "Odd ""Seq""; x" OWNED BY "My Table".v;
        """
        _apply(sql_dir, sql, target_url, plan_db)
        assert _pending(sql_dir, target_url, plan_db) == []

        _apply(sql_dir, 'CREATE TABLE "My Table" (v bigint);', target_url, plan_db)
        assert _rows(
            target_url, "SELECT count(*) FROM pg_class WHERE relkind = 'S'"
        ) == [(0,)]


class TestSequenceGrants:
    @pytest.fixture
    def role(self, server_url):
        name = f"pgdeclare_seq_role_{uuid.uuid4().hex[:8]}"
        _admin(server_url, f'CREATE ROLE "{name}"')
        try:
            yield name
        finally:
            # Requested before `databases`, so its grants are already gone
            # with the dropped databases by now.
            _admin(server_url, f'DROP ROLE IF EXISTS "{name}"')

    def test_grants_on_sequences_are_managed(self, tmp_path, role, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = PgDeclareConfig(managed_roles=(role,), report=lambda _line: None)
        base = "CREATE SEQUENCE order_no;\nCREATE TABLE note (id serial);\n"
        granted = (
            base
            + f'GRANT USAGE, SELECT ON SEQUENCE order_no TO "{role}";\n'
            + f'GRANT USAGE ON SEQUENCE note_id_seq TO "{role}";\n'
        )

        _apply(sql_dir, granted, target_url, plan_db, config)

        def privileges(name: str) -> list[tuple]:
            return _rows(
                target_url,
                f"SELECT has_sequence_privilege('{role}', 'public.{name}', 'USAGE'), "
                f"has_sequence_privilege('{role}', 'public.{name}', 'SELECT'), "
                f"has_sequence_privilege('{role}', 'public.{name}', 'UPDATE')",
            )

        assert privileges("order_no") == [(True, True, False)]
        assert privileges("note_id_seq") == [(True, False, False)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(sql_dir, base, target_url, plan_db, config)
        assert privileges("order_no") == [(False, False, False)]
        assert privileges("note_id_seq") == [(False, False, False)]
        assert _pending(sql_dir, target_url, plan_db, config) == []
