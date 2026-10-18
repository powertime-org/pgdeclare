"""Default privileges: ALTER DEFAULT PRIVILEGES for managed grantees.

The unit tests drive the differ, the rendered SQL and the statement parser.
The integration tests run the whole pipeline (plan database, introspection,
diff, apply, convergence check) against a real server and need
TEST_DATABASE_URL.
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
    CreateTable,
    GrantDefaultPrivilege,
    GrantPrivilege,
    RevokeDefaultPrivilege,
)
from pgdeclare.models import DefaultPrivilege, Grant, Schema, Table
from pgdeclare.statement_parser import ALTER_DEFAULT_PRIVILEGES, parse_sql_content


# =============================================================================
# Unit tests
# =============================================================================


def _entry(**kwargs) -> DefaultPrivilege:
    return DefaultPrivilege(
        **{
            "target_role": "owner",
            "schema": None,
            "object_type": "TABLES",
            "privilege": "SELECT",
            "grantee": "reader",
            **kwargs,
        }
    )


class TestDefaultPrivilegeDiff:
    def test_equal_entries_produce_no_changes(self):
        assert (
            compute_diff(
                Schema(default_privileges=[_entry()]),
                Schema(default_privileges=[_entry()]),
            )
            == []
        )

    def test_revokes_come_before_creates_and_grants_after_object_grants(self):
        table = Table(schema="public", name="note")
        grant = Grant(
            object_type="TABLE",
            object_name="note",
            schema="public",
            privilege="SELECT",
            grantee="reader",
        )
        desired = Schema(
            tables={table.qualified_name: table},
            grants=[grant],
            default_privileges=[_entry(privilege="INSERT")],
        )
        actual = Schema(default_privileges=[_entry()])

        kinds = [type(c) for c in compute_diff(desired, actual)]

        assert kinds == [
            RevokeDefaultPrivilege,
            CreateTable,
            GrantPrivilege,
            GrantDefaultPrivilege,
        ]

    def test_grant_option_change_revokes_then_grants(self):
        changes = compute_diff(
            Schema(default_privileges=[_entry(grantable=True)]),
            Schema(default_privileges=[_entry()]),
        )

        assert [type(c) for c in changes] == [
            RevokeDefaultPrivilege,
            GrantDefaultPrivilege,
        ]
        assert changes[1].to_sql().endswith(" WITH GRANT OPTION;")

    def test_scope_and_target_role_are_part_of_the_key(self):
        entries = [
            _entry(),
            _entry(schema="app"),
            _entry(target_role="other"),
            _entry(object_type="SEQUENCES"),
        ]
        assert len({e.key for e in entries}) == len(entries)


class TestDefaultPrivilegeSql:
    def test_global_grant_names_the_target_role(self):
        assert GrantDefaultPrivilege(entry=_entry()).to_sql() == (
            "ALTER DEFAULT PRIVILEGES FOR ROLE owner GRANT SELECT ON TABLES TO reader;"
        )

    def test_schema_revoke(self):
        sql = RevokeDefaultPrivilege(
            entry=_entry(schema="app", object_type="FUNCTIONS", privilege="EXECUTE")
        ).to_sql()
        assert sql == (
            'ALTER DEFAULT PRIVILEGES FOR ROLE owner IN SCHEMA "app" '
            "REVOKE EXECUTE ON FUNCTIONS FROM reader;"
        )

    def test_names_are_quoted(self):
        entry = _entry(
            target_role='Own"er', schema='a"; DROP TABLE x; --', grantee="Read er"
        )
        for change in (
            GrantDefaultPrivilege(entry=entry),
            RevokeDefaultPrivilege(entry=entry),
        ):
            sql = change.to_sql()
            assert 'FOR ROLE "Own""er"' in sql
            assert 'IN SCHEMA "a""; DROP TABLE x; --"' in sql
            assert '"Read er";' in sql


class TestDefaultPrivilegeParsing:
    def test_statement_is_recognised(self):
        [stmt] = parse_sql_content(
            "-- defaults\nalter  default\n  privileges in schema app "
            "grant select on tables to reader;"
        )
        assert stmt.statement_type == ALTER_DEFAULT_PRIVILEGES

    def test_other_alters_stay_generic(self):
        [stmt] = parse_sql_content("ALTER TABLE note ADD COLUMN default_x int;")
        assert stmt.statement_type != ALTER_DEFAULT_PRIVILEGES


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
def roles(server_url):
    """Two fresh roles: `reader` and `writer`. Requested before `databases`, so
    the default ACLs that name them are gone with the databases by teardown."""
    suffix = uuid.uuid4().hex[:8]
    names = (f"pgdeclare_dp_reader_{suffix}", f"pgdeclare_dp_writer_{suffix}")
    for name in names:
        _admin(server_url, f'CREATE ROLE "{name}"')
    try:
        yield names
    finally:
        for name in names:
            _admin(server_url, f'DROP ROLE IF EXISTS "{name}"')


@pytest.fixture
def databases(server_url):
    """A target database and a plan database, both empty, both dropped after.

    The plan database persists between the applies of one test, as one passed
    with --plan-database does.
    """
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_dp_target_{suffix}", f"pgdeclare_dp_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _config(*managed: str, schemas: tuple[str, ...] = ("public",)) -> PgDeclareConfig:
    return PgDeclareConfig(
        schemas=schemas,
        plan_recreate_schemas=schemas,
        managed_roles=managed,
        report=lambda _line: None,
    )


def _apply(sql_dir: Path, sql: str, target_url: str, plan_db: str, config):
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "schema.sql").write_text(sql)
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)


def _pending(sql_dir: Path, target_url: str, plan_db: str, config):
    desired = build_desired_schema(sql_dir, target_url, plan_db, config=config)
    actual = introspect_schema(target_url, config=config)
    return compute_diff(desired, actual, schemas=list(config.schemas), config=config)


def _defaults(url: str, grantee: str) -> list[tuple]:
    """(target role, schema, object type, privilege, grantable) for `grantee`."""
    return _rows(
        url,
        "SELECT r.rolname, n.nspname, d.defaclobjtype::text, a.privilege_type, "
        "a.is_grantable FROM pg_default_acl d "
        "JOIN pg_roles r ON r.oid = d.defaclrole "
        "LEFT JOIN pg_namespace n ON n.oid = d.defaclnamespace "
        "CROSS JOIN LATERAL aclexplode(d.defaclacl) a "
        f"WHERE a.grantee = '{grantee}'::regrole ORDER BY 1, 2, 3, 4",
    )


def _table_privileges(url: str, role: str, table: str) -> tuple:
    return _rows(
        url,
        f"SELECT has_table_privilege('{role}', '{table}', 'SELECT'), "
        f"has_table_privilege('{role}', '{table}', 'INSERT')",
    )[0]


def _current_user(url: str) -> str:
    return _rows(url, "SELECT current_user")[0][0]


class TestDefaultPrivilegeRoundTrip:
    def test_declare_change_and_remove(self, tmp_path, roles, databases):
        reader, _writer = roles
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(reader)
        me = _current_user(target_url)
        # The table is declared above the default privileges: the plan runs
        # them first, so it gets the grant as a later table would.
        base = "CREATE TABLE note (id int);\n"

        _apply(
            sql_dir,
            base + f'ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO "{reader}";\n',
            target_url,
            plan_db,
            config,
        )
        assert _defaults(target_url, reader) == [(me, None, "r", "SELECT", False)]
        assert _table_privileges(target_url, reader, "note") == (True, False)
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # A table created afterwards, outside the engine, really gets it.
        # (`_` tables are not managed, so it is no drift either.)
        _admin(target_url, "CREATE TABLE _later (id int)")
        assert _table_privileges(target_url, reader, "_later") == (True, False)

        # Change: another privilege, with grant option.
        _apply(
            sql_dir,
            base
            + "ALTER DEFAULT PRIVILEGES GRANT INSERT ON TABLES "
            + f'TO "{reader}" WITH GRANT OPTION;\n',
            target_url,
            plan_db,
            config,
        )
        assert _defaults(target_url, reader) == [(me, None, "r", "INSERT", True)]
        assert _table_privileges(target_url, reader, "note") == (False, True)
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # Remove: the default is revoked, and the persistent plan database
        # forgets it too. A table added in the same run is created after the
        # revoke, so it does not pick the old default up.
        _apply(
            sql_dir, base + "CREATE TABLE memo (id int);\n", target_url, plan_db, config
        )
        assert _defaults(target_url, reader) == []
        assert _table_privileges(target_url, reader, "note") == (False, False)
        assert _table_privileges(target_url, reader, "memo") == (False, False)
        assert _pending(sql_dir, target_url, plan_db, config) == []
        _admin(target_url, "CREATE TABLE _after_removal (id int)")
        assert _table_privileges(target_url, reader, "_after_removal") == (
            False,
            False,
        )

    def test_for_role_in_schema(self, tmp_path, roles, databases):
        reader, writer = roles
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(reader, schemas=("public", "app"))
        _admin(target_url, "CREATE SCHEMA app")
        _admin(target_url, f'GRANT USAGE, CREATE ON SCHEMA app TO "{writer}"')
        declared = (
            "CREATE SCHEMA IF NOT EXISTS app;\n"
            f'ALTER DEFAULT PRIVILEGES FOR ROLE "{writer}" IN SCHEMA app '
            f'GRANT SELECT, INSERT ON TABLES TO "{reader}";\n'
        )

        _apply(sql_dir, declared, target_url, plan_db, config)
        assert _defaults(target_url, reader) == [
            (writer, "app", "r", "INSERT", False),
            (writer, "app", "r", "SELECT", False),
        ]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        conn = psycopg2.connect(target_url)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f'SET ROLE "{writer}"')
                cur.execute("CREATE TABLE app._made_by_writer (id int)")
        finally:
            conn.close()
        assert _table_privileges(target_url, reader, "app._made_by_writer") == (
            True,
            True,
        )

        _apply(
            sql_dir, "CREATE SCHEMA IF NOT EXISTS app;\n", target_url, plan_db, config
        )
        assert _defaults(target_url, reader) == []
        assert _pending(sql_dir, target_url, plan_db, config) == []

    def test_every_object_type(self, tmp_path, roles, databases):
        reader, _writer = roles
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(reader)
        declared = "".join(
            f'ALTER DEFAULT PRIVILEGES GRANT {privilege} ON {kind} TO "{reader}";\n'
            for privilege, kind in (
                ("UPDATE", "TABLES"),
                ("USAGE", "SEQUENCES"),
                ("EXECUTE", "ROUTINES"),
                ("USAGE", "TYPES"),
                ("CREATE", "SCHEMAS"),
            )
        )

        _apply(sql_dir, declared, target_url, plan_db, config)
        assert sorted((row[2], row[3]) for row in _defaults(target_url, reader)) == [
            ("S", "USAGE"),
            ("T", "USAGE"),
            ("f", "EXECUTE"),
            ("n", "CREATE"),
            ("r", "UPDATE"),
        ]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(sql_dir, "", target_url, plan_db, config)
        assert _defaults(target_url, reader) == []
        assert _pending(sql_dir, target_url, plan_db, config) == []

    def test_managed_target_role_keeps_its_owner_defaults(
        self, tmp_path, roles, databases
    ):
        # The global row for `writer` holds writer's own built-in privileges
        # too. They are not declarations, so removing the grant to `reader`
        # must leave the row equal to the built-in default, which PostgreSQL
        # then deletes.
        reader, writer = roles
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(reader, writer)

        _apply(
            sql_dir,
            f'ALTER DEFAULT PRIVILEGES FOR ROLE "{writer}" '
            f'GRANT SELECT ON TABLES TO "{reader}";\n',
            target_url,
            plan_db,
            config,
        )
        assert _defaults(target_url, reader) == [(writer, None, "r", "SELECT", False)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(sql_dir, "", target_url, plan_db, config)
        assert _rows(
            target_url,
            f"SELECT count(*) FROM pg_default_acl WHERE defaclrole = '{writer}'::regrole",
        ) == [(0,)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

    def test_unmanaged_grantees_are_left_alone(self, tmp_path, roles, databases):
        reader, writer = roles
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = _config(reader)
        _admin(
            target_url,
            f'ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO "{writer}"',
        )

        assert _pending(sql_dir, target_url, plan_db, config) == []
        _apply(sql_dir, "", target_url, plan_db, config)
        assert len(_defaults(target_url, writer)) == 1
