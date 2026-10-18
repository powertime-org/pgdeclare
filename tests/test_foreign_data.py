"""Foreign servers, user mappings, foreign tables, extensions and owners.

The unit tests drive the differ and the rendered SQL, including the masking of
user-mapping options. The integration tests run the whole pipeline against a
real server (postgres_fdw and hstore ship with the postgres images) and need
TEST_DATABASE_URL.
"""

import json
import logging
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
    AlterExtensionUpdate,
    AlterForeignServer,
    AlterForeignTable,
    AlterOwner,
    AlterUserMapping,
    CreateExtension,
    CreateForeignServer,
    CreateForeignTable,
    CreateUserMapping,
    CreateView,
    DropAndCreateForeignTable,
    DropAndCreateView,
    DropExtension,
    DropForeignTable,
    DropUserMapping,
    GrantPrivilege,
)
from pgdeclare.cli import main
from pgdeclare.differ import undeclared_extensions
from pgdeclare.executor import execute_migration
from pgdeclare.models import (
    Column,
    Extension,
    ForeignServer,
    ForeignTable,
    Grant,
    ObjectOwner,
    Role,
    Schema,
    UserMapping,
    View,
)
from pgdeclare.schema_parser import (
    extension_drop_directives,
    parse_extension_statement,
)

SECRET = "s3cr3t-'pw"


# =============================================================================
# Unit tests: user mappings and masking
# =============================================================================


def _mapping(**options: str) -> UserMapping:
    return UserMapping(server="remote", user="app", options=dict(options))


class TestUserMappingMasking:
    def test_create_executes_the_password_and_displays_it_masked(self):
        change = CreateUserMapping(mapping=_mapping(user="bob", password=SECRET))

        assert change.to_sql() == (
            'CREATE USER MAPPING FOR "app" SERVER "remote" '
            "OPTIONS (\"user\" 'bob', \"password\" 's3cr3t-''pw');"
        )
        shown = change.to_display_sql()
        assert shown == (
            'CREATE USER MAPPING FOR "app" SERVER "remote" '
            "OPTIONS (\"user\" 'bob', \"password\" '***');"
        )
        assert "s3cr3t" not in shown + change.description

    def test_alter_masks_and_the_description_names_options_only(self):
        change = AlterUserMapping(
            mapping=_mapping(user="bob", password=SECRET, sslkey="/k"),
            current_options={"user": "al", "password": "old", "dropme": "x"},
        )

        assert "SET \"password\" 's3cr3t-''pw'" in change.to_sql()
        shown = change.to_display_sql()
        assert shown == (
            'ALTER USER MAPPING FOR "app" SERVER "remote" OPTIONS ('
            "SET \"user\" 'bob', SET \"password\" '***', ADD \"sslkey\" '***', "
            'DROP "dropme");'
        )
        assert "s3cr3t" not in shown and "old" not in shown
        assert change.description == (
            "Alter options (dropme, password, sslkey, user) of user mapping "
            "for app on server remote"
        )

    def test_public_mapping_is_written_for_public(self):
        change = CreateUserMapping(
            mapping=UserMapping(server='odd"srv', user="public", options={})
        )
        assert change.to_sql() == 'CREATE USER MAPPING FOR PUBLIC SERVER "odd""srv";'
        drop = DropUserMapping(mapping=UserMapping(server="s", user='x"y'))
        assert drop.to_sql() == 'DROP USER MAPPING IF EXISTS FOR "x""y" SERVER "s";'

    def test_dry_run_returns_and_logs_masked_sql(self, caplog):
        change = CreateUserMapping(mapping=_mapping(password=SECRET))
        with caplog.at_level(logging.INFO, logger="pgdeclare"):
            ok, executed = execute_migration("unused", [change], dry_run=True)

        assert ok
        assert executed == [change.to_display_sql()]
        assert "s3cr3t" not in caplog.text


class TestUserMappingDiff:
    def _schemas(self, desired: UserMapping, actual: UserMapping | None):
        server = ForeignServer(name="remote", fdw="postgres_fdw")
        d = Schema(
            foreign_servers={"remote": server},
            user_mappings={desired.key: desired},
            roles={"app": Role(name="app")},
        )
        a = Schema(
            foreign_servers={"remote": server},
            user_mappings={} if actual is None else {actual.key: actual},
            roles={"app": Role(name="app")},
        )
        return d, a

    def test_an_undeclared_password_set_by_hand_is_left_alone(self):
        d, a = self._schemas(_mapping(user="bob"), _mapping(user="bob", password="x"))
        assert compute_diff(d, a) == []

    def test_a_declared_password_is_converged(self):
        d, a = self._schemas(
            _mapping(user="bob", password=SECRET), _mapping(user="bob", password="x")
        )
        changes = compute_diff(d, a)
        assert [type(c) for c in changes] == [AlterUserMapping]
        assert "s3cr3t" not in changes[0].to_display_sql()

    def test_an_undeclared_public_option_is_dropped(self):
        d, a = self._schemas(_mapping(), _mapping(user="bob"))
        changes = compute_diff(d, a)
        assert changes[0].to_sql().endswith('OPTIONS (DROP "user");')

    def test_unreadable_options_are_not_compared(self):
        d, a = self._schemas(
            _mapping(user="bob"), UserMapping(server="remote", user="app", options=None)
        )
        assert compute_diff(d, a) == []

    def test_only_a_managed_roles_mapping_on_a_declared_server_is_dropped(self):
        server = ForeignServer(name="remote", fdw="postgres_fdw")
        managed = UserMapping(server="remote", user="app")
        stranger = UserMapping(server="remote", user="someone")
        elsewhere = UserMapping(server="other", user="app")
        desired = Schema(foreign_servers={"remote": server}, roles={"app": Role("app")})
        actual = Schema(
            foreign_servers={"remote": server},
            user_mappings={m.key: m for m in (managed, stranger, elsewhere)},
            roles={"app": Role("app")},
        )

        changes = compute_diff(desired, actual)
        assert [type(c) for c in changes] == [DropUserMapping]
        assert changes[0].object_name == "app@remote"


# =============================================================================
# Unit tests: servers and foreign tables
# =============================================================================


class TestForeignServerDiff:
    def test_create_spells_out_type_version_and_options(self):
        srv = ForeignServer(
            name="remote",
            fdw="postgres_fdw",
            type="pg",
            version="18",
            options={"host": "db'1", "port": "5432"},
        )
        changes = compute_diff(Schema(foreign_servers={"remote": srv}), Schema())
        assert [type(c) for c in changes] == [CreateForeignServer]
        assert changes[0].to_sql() == (
            "CREATE SERVER \"remote\" TYPE 'pg' VERSION '18' FOREIGN DATA WRAPPER "
            "\"postgres_fdw\" OPTIONS (\"host\" 'db''1', \"port\" '5432');"
        )

    def test_options_and_version_alter_in_place(self):
        want = ForeignServer(
            name="remote", fdw="postgres_fdw", version="2", options={"host": "b"}
        )
        have = ForeignServer(
            name="remote",
            fdw="postgres_fdw",
            version="1",
            options={"host": "a", "port": "1"},
        )
        changes = compute_diff(
            Schema(foreign_servers={"remote": want}),
            Schema(foreign_servers={"remote": have}),
        )
        assert [type(c) for c in changes] == [AlterForeignServer]
        assert changes[0].to_sql() == (
            "ALTER SERVER \"remote\" VERSION '2' "
            'OPTIONS (SET "host" \'b\', DROP "port");'
        )

    def test_undeclared_servers_are_never_dropped(self):
        srv = ForeignServer(name="remote", fdw="postgres_fdw")
        assert compute_diff(Schema(), Schema(foreign_servers={"remote": srv})) == []

    def test_a_new_wrapper_is_refused(self):
        with pytest.raises(ValueError, match="foreign data wrapper"):
            compute_diff(
                Schema(foreign_servers={"s": ForeignServer(name="s", fdw="b")}),
                Schema(foreign_servers={"s": ForeignServer(name="s", fdw="a")}),
            )


def _ft(**kwargs) -> ForeignTable:
    base = {
        "schema": "public",
        "name": "remote_note",
        "server": "remote",
        "columns": {
            "id": Column(name="id", type="integer", nullable=False),
            "body": Column(name="body", type="text", nullable=True),
        },
        "options": {"table_name": "note"},
    }
    base.update(kwargs)
    return ForeignTable(**base)


def _ft_schema(*tables: ForeignTable, grants=()) -> Schema:
    return Schema(
        foreign_tables={t.qualified_name: t for t in tables}, grants=list(grants)
    )


class TestForeignTableDiff:
    def test_create(self):
        table = _ft(column_options={"body": {"column_name": "text"}})
        changes = compute_diff(_ft_schema(table), Schema())
        assert [type(c) for c in changes] == [CreateForeignTable]
        assert changes[0].to_sql() == (
            'CREATE FOREIGN TABLE "public"."remote_note" (\n'
            '    "id" integer NOT NULL,\n'
            '    "body" text OPTIONS ("column_name" \'text\')\n'
            ') SERVER "remote" OPTIONS ("table_name" \'note\');'
        )

    def test_columns_and_options_alter_in_one_statement(self):
        want = _ft(
            columns={
                "id": Column(name="id", type="bigint", nullable=True, default="0"),
                "title": Column(name="title", type="text", nullable=True),
            },
            options={"table_name": "note2"},
            column_options={"id": {"column_name": "nid"}},
        )
        changes = compute_diff(_ft_schema(want), _ft_schema(_ft()))
        assert [type(c) for c in changes] == [AlterForeignTable]
        assert changes[0].to_sql() == (
            'ALTER FOREIGN TABLE "public"."remote_note" DROP COLUMN "body", '
            'ADD COLUMN "title" text, ALTER COLUMN "id" TYPE bigint, '
            'ALTER COLUMN "id" DROP NOT NULL, ALTER COLUMN "id" SET DEFAULT 0, '
            'ALTER COLUMN "id" OPTIONS (ADD "column_name" \'nid\'), '
            "OPTIONS (SET \"table_name\" 'note2');"
        )

    def test_a_new_server_recreates_and_replays_grants(self):
        grant = Grant(
            object_type="TABLE",
            object_name="remote_note",
            schema="public",
            privilege="SELECT",
            grantee="app",
        )
        changes = compute_diff(
            _ft_schema(_ft(server="other"), grants=[grant]),
            _ft_schema(_ft(), grants=[grant]),
        )
        assert [type(c) for c in changes] == [DropAndCreateForeignTable]
        sql = changes[0].to_sql()
        assert sql.startswith('DO $$ BEGIN DROP FOREIGN TABLE IF EXISTS "public"')
        assert 'SERVER "other"' in sql
        assert 'GRANT SELECT ON "public"."remote_note" TO app' in sql

    def test_undeclared_drops_without_cascade_or_directive(self):
        changes = compute_diff(Schema(), _ft_schema(_ft(name='x"; --')))
        assert [type(c) for c in changes] == [DropForeignTable]
        assert changes[0].to_sql() == 'DROP FOREIGN TABLE IF EXISTS "public"."x""; --";'
        assert not changes[0].requires_drop_directive


# =============================================================================
# Unit tests: extensions
# =============================================================================


class TestExtensionDiff:
    def test_create_with_schema_version_and_cascade(self):
        ext = Extension(name="hstore", schema="ext", version="1.7", cascade=True)
        changes = compute_diff(Schema(extensions={"hstore": ext}), Schema())
        assert [type(c) for c in changes] == [CreateExtension]
        assert changes[0].to_sql() == (
            'CREATE EXTENSION IF NOT EXISTS "hstore" WITH SCHEMA "ext" '
            "VERSION '1.7' CASCADE;"
        )

    def test_a_stated_version_updates_and_an_unstated_one_is_ignored(self):
        live = Schema(
            extensions={"hstore": Extension("hstore", schema="public", version="1.7")}
        )
        assert (
            compute_diff(Schema(extensions={"hstore": Extension("hstore")}), live) == []
        )

        changes = compute_diff(
            Schema(extensions={"hstore": Extension("hstore", version="1.8")}), live
        )
        assert [type(c) for c in changes] == [AlterExtensionUpdate]
        assert changes[0].to_sql() == "ALTER EXTENSION \"hstore\" UPDATE TO '1.8';"

    def test_undeclared_is_noted_never_dropped(self):
        live = Schema(
            extensions={
                "hstore": Extension("hstore", "public", "1.8"),
                "plpgsql": Extension("plpgsql", "pg_catalog", "1.0"),
            }
        )
        assert compute_diff(Schema(), live) == []
        assert undeclared_extensions(Schema(), live) == ["hstore"]

    def test_a_drop_directive_drops_without_cascade(self):
        live = Schema(extensions={"hstore": Extension("hstore", "public", "1.8")})
        desired = Schema(extension_drops={"hstore"})
        changes = compute_diff(desired, live)
        assert [type(c) for c in changes] == [DropExtension]
        assert changes[0].to_sql() == 'DROP EXTENSION IF EXISTS "hstore";'
        assert undeclared_extensions(desired, live) == []

    def test_statement_and_directive_parsing(self):
        ext = parse_extension_statement(
            "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\" WITH SCHEMA ext VERSION '1.1' CASCADE;"
        )
        assert ext == Extension("uuid-ossp", schema="ext", version="1.1", cascade=True)
        assert parse_extension_statement("CREATE EXTENSION citext;") == Extension(
            "citext"
        )
        assert extension_drop_directives(
            ["note.old", "extension:hstore", "Extension:citext", "extension:"]
        ) == {"hstore", "citext"}


# =============================================================================
# Unit tests: owners
# =============================================================================


class TestOwnerDiff:
    def test_only_stated_owners_are_written(self):
        want = ObjectOwner(kind="TABLE", schema="public", name="note", owner="app")
        have = ObjectOwner(kind="TABLE", schema="public", name="note", owner="admin")
        other = ObjectOwner(kind="VIEW", schema="public", name="v", owner="admin")
        changes = compute_diff(
            Schema(owners={want.key: want}),
            Schema(owners={have.key: have, other.key: other}),
        )
        assert [type(c) for c in changes] == [AlterOwner]
        assert changes[0].to_sql() == 'ALTER TABLE "public"."note" OWNER TO "app";'

    def test_a_recreated_object_gets_its_owner_back(self):
        view = View(schema="public", name="v", definition=" SELECT 2;")
        owner = ObjectOwner(kind="VIEW", schema="public", name="v", owner="app")
        changes = compute_diff(
            Schema(views={"public.v": view}, owners={owner.key: owner}),
            Schema(
                views={"public.v": View("public", "v", " SELECT 1;")},
                owners={owner.key: owner},
            ),
        )
        assert [type(c) for c in changes] == [DropAndCreateView, AlterOwner]

    def test_an_owners_implicit_privileges_are_not_grants(self):
        owner = ObjectOwner(kind="TABLE", schema="public", name="note", owner="app")
        implicit = Grant(
            object_type="TABLE",
            object_name="note",
            schema="public",
            privilege="SELECT",
            grantee="app",
        )
        changes = compute_diff(
            Schema(owners={owner.key: owner}, grants=[implicit]),
            Schema(owners={owner.key: owner}),
        )
        assert changes == []

    def test_only_the_exact_owned_object_loses_its_owner_grants(self):
        """A same-named routine or sequence, or another overload, keeps its diff."""
        table = ObjectOwner(kind="TABLE", schema="public", name="note", owner="app")
        func = ObjectOwner(
            kind="FUNCTION", schema="public", name="f", args="integer", owner="app"
        )

        def grant(object_type: str, name: str, privilege: str, args=None) -> Grant:
            return Grant(
                object_type=object_type,
                object_name=name,
                schema="public",
                privilege=privilege,
                grantee="app",
                args=args,
            )

        owned = [
            grant("TABLE", "note", "SELECT"),
            grant("FUNCTION", "f", "EXECUTE", args="integer"),
        ]
        others = [
            grant("FUNCTION", "note", "EXECUTE", args="integer"),
            grant("SEQUENCE", "note", "USAGE"),
            grant("FUNCTION", "f", "EXECUTE", args="text"),
        ]
        owners = {table.key: table, func.key: func}
        changes = compute_diff(
            Schema(owners=owners, grants=owned + others), Schema(owners=owners)
        )
        assert all(isinstance(c, GrantPrivilege) for c in changes)
        assert sorted(c.to_sql() for c in changes) == [
            'GRANT EXECUTE ON FUNCTION "public"."f"(text) TO app;',
            'GRANT EXECUTE ON FUNCTION "public"."note"(integer) TO app;',
            'GRANT USAGE ON SEQUENCE "public"."note" TO app;',
        ]

        # And a grant on one of them that is no longer declared is revoked.
        changes = compute_diff(
            Schema(owners=owners, grants=owned), Schema(owners=owners, grants=others)
        )
        assert sorted(type(c).__name__ for c in changes) == ["RevokePrivilege"] * 3

    def test_routine_owner_kinds(self):
        sql = {
            kind: AlterOwner(
                owner=ObjectOwner(kind=kind, schema="s", name="r", args=args, owner="o")
            ).to_sql()
            for kind, args in (
                ("PROCEDURE", "integer"),
                ("AGGREGATE", ""),
                ("COLLATION", ""),
            )
        }
        assert sql == {
            "PROCEDURE": 'ALTER PROCEDURE "s"."r"(integer) OWNER TO "o";',
            "AGGREGATE": 'ALTER AGGREGATE "s"."r"(*) OWNER TO "o";',
            "COLLATION": 'ALTER COLLATION "s"."r" OWNER TO "o";',
        }

    def test_owner_sql_quotes_every_name(self):
        assert (
            AlterOwner(
                owner=ObjectOwner(
                    kind="FUNCTION", schema='s"s', name="f", args="integer", owner='r"r'
                )
            ).to_sql()
            == 'ALTER FUNCTION "s""s"."f"(integer) OWNER TO "r""r";'
        )
        assert (
            AlterOwner(
                owner=ObjectOwner(kind="SCHEMA", schema="", name="app", owner="o")
            ).to_sql()
            == 'ALTER SCHEMA "app" OWNER TO "o";'
        )
        with pytest.raises(ValueError):
            ObjectOwner(kind="TABLE; --", schema="s", name="n", owner="o").target_sql

    def test_created_object_owner_follows_the_create(self):
        view = View(schema="public", name="v", definition=" SELECT 1;")
        owner = ObjectOwner(kind="VIEW", schema="public", name="v", owner="app")
        changes = compute_diff(
            Schema(views={"public.v": view}, owners={owner.key: owner}), Schema()
        )
        assert [type(c) for c in changes] == [CreateView, AlterOwner]


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
    target, plan = f"pgdeclare_fdw_target_{suffix}", f"pgdeclare_fdw_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def role(server_url):
    """A cluster-wide role for one test, dropped afterwards."""
    name = f"pgdeclare_fdw_role_{uuid.uuid4().hex[:8]}"
    _admin(server_url, f'CREATE ROLE "{name}"')
    try:
        yield name
    finally:
        _admin(server_url, f'DROP ROLE IF EXISTS "{name}"')


def _config(tmp_path: Path, **kwargs) -> PgDeclareConfig:
    # The library leaves extensions in the plan database to the caller.
    bootstrap = tmp_path / "bootstrap.sql"
    bootstrap.write_text("CREATE EXTENSION IF NOT EXISTS postgres_fdw;\n")
    return PgDeclareConfig(
        report=lambda _line: None, bootstrap_sql_files=(bootstrap,), **kwargs
    )


def _write(sql_dir: Path, sql: str) -> None:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "schema.sql").write_text(sql)


def _apply(sql_dir: Path, sql: str, target_url: str, plan_db: str, config) -> None:
    _write(sql_dir, sql)
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)


def _pending(sql_dir: Path, target_url: str, plan_db: str, config):
    desired = build_desired_schema(sql_dir, target_url, plan_db, config=config)
    actual = introspect_schema(target_url, config=config)
    return compute_diff(desired, actual, schemas=list(config.schemas), config=config)


FDW_V1 = """
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

CREATE SERVER remote FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'remote.invalid', dbname 'app', port '5432');

CREATE USER MAPPING FOR PUBLIC SERVER remote
    OPTIONS (user 'reader', password 's3cr3t-''pw');

CREATE USER MAPPING FOR {role} SERVER remote OPTIONS (user 'writer');

CREATE FOREIGN TABLE remote_note (
    id   integer NOT NULL,
    body text OPTIONS (column_name 'content'),
    old  text
) SERVER remote OPTIONS (schema_name 'public', table_name 'note');

GRANT SELECT ON remote_note TO {role};
"""

FDW_V2 = """
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

CREATE SERVER remote FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'remote2.invalid', dbname 'app');

CREATE SERVER archive FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'archive.invalid');

CREATE USER MAPPING FOR PUBLIC SERVER remote
    OPTIONS (user 'reader', password 'n3w-s3cr3t');

CREATE FOREIGN TABLE remote_note (
    id    bigint,
    body  text OPTIONS (column_name 'body'),
    title text DEFAULT 'untitled'
) SERVER remote OPTIONS (table_name 'note2');

CREATE FOREIGN TABLE archived_note (id integer) SERVER remote;
"""


class TestForeignDataRoundTrip:
    def test_servers_mappings_and_tables_converge(self, tmp_path, role, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(tmp_path, managed_roles=(role,))

        _write(sql_dir, FDW_V1.format(role=f'"{role}"'))
        pending = _pending(sql_dir, target_url, plan_db, config)
        shown = "\n".join(c.to_display_sql() + c.description for c in pending)
        assert "s3cr3t" not in shown
        assert "\"password\" '***'" in shown

        apply_declared_schema(target_url, sql_dir, plan_db, config=config)

        assert _rows(
            target_url, "SELECT srvname, srvoptions FROM pg_foreign_server"
        ) == [("remote", ["host=remote.invalid", "dbname=app", "port=5432"])]
        assert sorted(
            _rows(target_url, "SELECT usename, umoptions FROM pg_user_mappings")
        ) == [
            (role, ["user=writer"]),
            ("public", ["user=reader", "password=s3cr3t-'pw"]),
        ]
        assert _rows(
            target_url,
            "SELECT ftoptions FROM pg_foreign_table "
            "WHERE ftrelid = 'public.remote_note'::regclass",
        ) == [(["schema_name=public", "table_name=note"],)]
        assert _rows(
            target_url,
            f"SELECT has_table_privilege('{role}', 'public.remote_note', 'SELECT')",
        ) == [(True,)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # A password set by hand on an option the declaration omits survives.
        _admin(
            target_url,
            f'ALTER USER MAPPING FOR "{role}" SERVER remote '
            "OPTIONS (ADD password 'by-hand')",
        )
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(sql_dir, FDW_V2, target_url, plan_db, config)

        assert sorted(_rows(target_url, "SELECT srvname FROM pg_foreign_server")) == [
            ("archive",),
            ("remote",),
        ]
        assert _rows(
            target_url,
            "SELECT srvoptions FROM pg_foreign_server WHERE srvname = 'remote'",
        ) == [(["host=remote2.invalid", "dbname=app"],)]
        # The managed role's mapping is no longer declared: dropped. PUBLIC's
        # password is converged.
        assert _rows(target_url, "SELECT usename, umoptions FROM pg_user_mappings") == [
            ("public", ["user=reader", "password=n3w-s3cr3t"])
        ]
        cols = _rows(
            target_url,
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull, "
            "a.attfdwoptions FROM pg_attribute a "
            "WHERE a.attrelid = 'public.remote_note'::regclass AND a.attnum > 0 "
            "AND NOT a.attisdropped ORDER BY a.attnum",
        )
        assert cols == [
            ("id", "bigint", False, None),
            ("body", "text", False, ["column_name=body"]),
            ("title", "text", False, None),
        ]
        assert _rows(
            target_url,
            "SELECT ftoptions FROM pg_foreign_table "
            "WHERE ftrelid = 'public.remote_note'::regclass",
        ) == [(["table_name=note2"],)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # Moving a table to another server recreates it; dropping the
        # declaration drops it. The archive server stays: servers are never
        # dropped.
        _apply(
            sql_dir,
            FDW_V2.replace(
                "CREATE FOREIGN TABLE archived_note (id integer) SERVER remote;",
                "CREATE FOREIGN TABLE archived_note (id integer) SERVER archive;",
            ).replace("CREATE SERVER archive", "CREATE SERVER archive2")
            + "\nCREATE SERVER archive FOREIGN DATA WRAPPER postgres_fdw "
            "OPTIONS (host 'archive.invalid');\n",
            target_url,
            plan_db,
            config,
        )
        assert _rows(
            target_url,
            "SELECT s.srvname FROM pg_foreign_table ft "
            "JOIN pg_foreign_server s ON s.oid = ft.ftserver "
            "WHERE ft.ftrelid = 'public.archived_note'::regclass",
        ) == [("archive",)]

        _apply(
            sql_dir,
            FDW_V2.replace(
                "CREATE FOREIGN TABLE archived_note (id integer) SERVER remote;", ""
            ),
            target_url,
            plan_db,
            config,
        )
        assert _rows(
            target_url, "SELECT to_regclass('public.archived_note') IS NULL"
        ) == [(True,)]
        assert ("archive2",) in _rows(
            target_url, "SELECT srvname FROM pg_foreign_server"
        )
        assert _pending(sql_dir, target_url, plan_db, config) == []

    def test_cli_diff_masks_the_password(self, tmp_path, databases, capsys):
        target_url, _plan_db = databases
        sql_dir = tmp_path / "sql"
        _write(
            sql_dir,
            FDW_V1.replace(
                "CREATE USER MAPPING FOR {role} SERVER remote OPTIONS (user 'writer');",
                "",
            ).replace("GRANT SELECT ON remote_note TO {role};", ""),
        )
        base = ["--database-url", target_url, "--sql-dir", str(sql_dir)]

        assert main(["diff", *base]) == 0
        out = capsys.readouterr()
        assert "s3cr3t" not in out.out + out.err
        assert "Create user mapping for public on server remote" in out.out

        assert main(["diff", "--json", *base]) == 0
        out = capsys.readouterr()
        assert "s3cr3t" not in out.out + out.err
        payload = json.loads(out.out)
        assert "CreateExtension" in [c["kind"] for c in payload["changes"]]

        assert main(["apply", "--dry-run", *base]) == 0
        out = capsys.readouterr()
        assert "s3cr3t" not in out.out + out.err

        assert main(["apply", *base]) == 0
        out = capsys.readouterr()
        assert "s3cr3t" not in out.out + out.err
        assert _rows(target_url, "SELECT umoptions FROM pg_user_mappings") == [
            (["user=reader", "password=s3cr3t-'pw"],)
        ]
        assert main(["diff", "--exit-code", *base]) == 0


class TestExtensionRoundTrip:
    def test_create_update_note_and_drop(self, tmp_path, databases, capsys):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = PgDeclareConfig(report=lambda _line: None)

        def version() -> list[tuple]:
            return _rows(
                target_url,
                "SELECT e.extversion, n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace "
                "WHERE e.extname = 'hstore'",
            )

        _apply(
            sql_dir,
            "CREATE EXTENSION hstore WITH SCHEMA public VERSION '1.7';\n"
            "CREATE TABLE kv (id int PRIMARY KEY);\n",
            target_url,
            plan_db,
            config,
        )
        assert version() == [("1.7", "public")]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(
            sql_dir,
            "CREATE EXTENSION hstore VERSION '1.8';\nCREATE TABLE kv (id int PRIMARY KEY);\n",
            target_url,
            plan_db,
            config,
        )
        assert version() == [("1.8", "public")]

        # Undeclared: left in place, and said so.
        _write(sql_dir, "CREATE TABLE kv (id int PRIMARY KEY);\n")
        assert _pending(sql_dir, target_url, plan_db, config) == []
        assert (
            main(["diff", "--database-url", target_url, "--sql-dir", str(sql_dir)]) == 0
        )
        assert "Note: extension hstore is installed but not declared" in (
            capsys.readouterr().out
        )
        apply_declared_schema(target_url, sql_dir, plan_db, config=config)
        assert version() == [("1.8", "public")]

        # Consent drops it.
        _apply(
            sql_dir,
            "CREATE TABLE kv (id int PRIMARY KEY);\n"
            "-- @migrate-before:2026-10-18_drop_hstore\n"
            "-- @drop extension:hstore\n"
            "SELECT 1;\n"
            "-- @end-migrate\n",
            target_url,
            plan_db,
            config,
        )
        assert version() == []
        assert _pending(sql_dir, target_url, plan_db, config) == []


OWNED = """
ALTER SCHEMA public OWNER TO {role};

CREATE TYPE mood AS ENUM ('calm', 'loud');
ALTER TYPE mood OWNER TO {role};

CREATE DOMAIN short_text AS text CHECK (length(VALUE) < 10);
ALTER DOMAIN short_text OWNER TO {role};

CREATE SEQUENCE ticket_no;
ALTER SEQUENCE ticket_no OWNER TO {role};

CREATE TABLE note (id serial PRIMARY KEY, mood mood, label short_text);
ALTER TABLE note OWNER TO {role};

CREATE TABLE unowned (id int);

CREATE VIEW note_view AS SELECT id FROM note;
ALTER VIEW note_view OWNER TO {role};

CREATE MATERIALIZED VIEW note_count AS SELECT count(*) AS n FROM note;
ALTER MATERIALIZED VIEW note_count OWNER TO {role};

CREATE FUNCTION answer(x integer) RETURNS integer LANGUAGE sql AS $$ SELECT x $$;
ALTER FUNCTION answer(integer) OWNER TO {role};

CREATE PROCEDURE tidy(n integer) LANGUAGE sql AS $$ SELECT n $$;
ALTER PROCEDURE tidy(integer) OWNER TO {role};

CREATE AGGREGATE total(integer) (SFUNC = int4pl, STYPE = integer);
ALTER AGGREGATE total(integer) OWNER TO {role};

CREATE TYPE span AS RANGE (subtype = integer);
ALTER TYPE span OWNER TO {role};

CREATE COLLATION plain_c (locale = 'C');
ALTER COLLATION plain_c OWNER TO {role};
"""

OWNERS_SQL = """
SELECT c.relname, pg_get_userbyid(c.relowner) FROM pg_class c
WHERE c.relnamespace = 'public'::regnamespace
  AND c.relname IN ('note', 'note_id_seq', 'ticket_no', 'unowned', 'note_view',
                    'note_count')
UNION ALL SELECT proname, pg_get_userbyid(proowner) FROM pg_proc
WHERE proname IN ('answer', 'tidy', 'total')
UNION ALL SELECT typname, pg_get_userbyid(typowner) FROM pg_type
WHERE typname IN ('mood', 'short_text', 'span')
UNION ALL SELECT collname, pg_get_userbyid(collowner) FROM pg_collation
WHERE collname = 'plain_c'
UNION ALL SELECT 'schema', pg_get_userbyid(nspowner) FROM pg_namespace
WHERE nspname = 'public'
ORDER BY 1
"""


class TestOwnershipRoundTrip:
    def test_stated_owners_converge_only_when_enabled(
        self, tmp_path, role, databases, server_url
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        me = _rows(server_url, "SELECT current_user")[0][0]
        sql = OWNED.format(role=f'"{role}"')

        # Off (the default): nothing about owners is planned.
        off = PgDeclareConfig(report=lambda _line: None, managed_roles=(role,))
        _admin(target_url, "ALTER SCHEMA public OWNER TO CURRENT_USER")
        _apply(sql_dir, sql, target_url, plan_db, off)
        assert {owner for _name, owner in _rows(target_url, OWNERS_SQL)} == {me}
        assert _pending(sql_dir, target_url, plan_db, off) == []

        on = PgDeclareConfig(
            report=lambda _line: None, managed_roles=(role,), manage_ownership=True
        )
        pending = _pending(sql_dir, target_url, plan_db, on)
        assert {type(c) for c in pending} == {AlterOwner}
        apply_declared_schema(target_url, sql_dir, plan_db, config=on)

        owners = dict(_rows(target_url, OWNERS_SQL))
        assert owners.pop("unowned") == me
        assert set(owners.values()) == {role}, owners
        assert _pending(sql_dir, target_url, plan_db, on) == []

        # A recreated view comes back owned by the stated role.
        _apply(
            sql_dir,
            sql.replace("SELECT id FROM note", "SELECT id, mood FROM note"),
            target_url,
            plan_db,
            on,
        )
        assert dict(_rows(target_url, OWNERS_SQL))["note_view"] == role

        # Drift by hand is put back.
        _admin(target_url, "ALTER TABLE note OWNER TO CURRENT_USER")
        assert [c.description for c in _pending(sql_dir, target_url, plan_db, on)] == [
            f"Set owner of table public.note to {role}"
        ]
