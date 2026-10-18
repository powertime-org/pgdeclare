"""Hostile names, declared passwords, and schema privileges.

The migrator usually runs with far more privilege than the roles whose objects
it reads from the live catalog. A name is data, never SQL; a declared password
is executed, never shown; and a schema the engine creates is not opened to
everyone unless asked.
"""

import json
import uuid

import psycopg2
import pytest

from pgdeclare import PgDeclareConfig, compute_diff, ensure_schemas, execute_migration
from pgdeclare.changes import (
    AddEnumValue,
    AddIdentity,
    AlterRole,
    CreateRole,
    CreateTable,
    DropColumn,
    DropTable,
    GrantPrivilege,
)
from pgdeclare.cli import main
from pgdeclare.introspection import introspect_schema
from pgdeclare.models import Column, Constraint, Grant, Role, Schema, Table
from pgdeclare.quoting import do_block, quote_ident, quote_literal, role_ident

HOSTILE = 'x"; DROP TABLE canary; --'


# =============================================================================
# Quoting
# =============================================================================


def test_quote_ident_doubles_embedded_quotes():
    assert quote_ident("plain") == '"plain"'
    assert quote_ident(HOSTILE) == '"x""; DROP TABLE canary; --"'


def test_quote_literal_doubles_embedded_quotes():
    assert quote_literal("calm") == "'calm'"
    assert quote_literal("it's") == "'it''s'"


def test_role_ident_keeps_ordinary_roles_bare():
    assert role_ident("app_reader") == "app_reader"
    assert role_ident("public") == "public"
    assert role_ident("Mixed") == '"Mixed"'
    assert role_ident(HOSTILE) == '"x""; DROP TABLE canary; --"'


def test_do_block_picks_a_tag_the_body_cannot_close():
    assert do_block("BEGIN NULL; END") == "DO $$ BEGIN NULL; END $$;"
    wrapped = do_block('BEGIN DROP TABLE "a$$b"; END')
    assert wrapped.startswith("DO $pgdeclare1$ ") and wrapped.endswith(" $pgdeclare1$;")


def test_ordinary_names_render_exactly_as_before():
    assert (
        DropTable(schema="public", table_name="note").to_sql()
        == 'DROP TABLE IF EXISTS "public"."note" CASCADE;'
    )
    assert (
        AddEnumValue(schema="public", enum_name="mood", value="loud", after="calm")
        .to_sql()
        .endswith("ADD VALUE 'loud' AFTER 'calm';")
    )
    grant = Grant(
        object_type="TABLE",
        object_name="note",
        schema="public",
        privilege="SELECT",
        grantee="app_reader",
    )
    assert (
        GrantPrivilege(grant=grant).to_sql()
        == 'GRANT SELECT ON "public"."note" TO app_reader;'
    )


def test_hostile_names_stay_inside_their_quotes():
    sql = DropColumn(schema="public", table=HOSTILE, column_name=HOSTILE).to_sql()
    assert sql == (
        'ALTER TABLE "public"."x""; DROP TABLE canary; --" '
        'DROP COLUMN "x""; DROP TABLE canary; --" CASCADE;'
    )
    identity = AddIdentity(
        schema="public", table="t", column="it's", identity_type="d"
    ).to_sql()
    assert "'it''s'" in identity


def test_inline_constraint_names_are_quoted():
    table = Table(
        schema="public",
        name="t",
        columns={"id": Column(name="id", type="integer", nullable=False)},
        constraints=[Constraint(name=HOSTILE, type="p", definition="PRIMARY KEY (id)")],
    )
    sql = CreateTable(table=table).to_sql()
    assert 'CONSTRAINT "x""; DROP TABLE canary; --" PRIMARY KEY (id)' in sql


# =============================================================================
# Passwords
# =============================================================================


def test_role_changes_mask_the_password_for_display():
    role = Role(name="app_login", can_login=True, password="s3cret-'x")
    for change in (CreateRole(role=role), AlterRole(role=role)):
        assert "PASSWORD 's3cret-''x'" in change.to_sql()
        shown = change.to_display_sql()
        assert "s3cret" not in shown
        assert "PASSWORD '***'" in shown


def test_cli_json_shape_masks_the_password():
    from pgdeclare.cli import change_to_dict

    role = Role(name="app_login", can_login=True, password="s3cret")
    data = change_to_dict(CreateRole(role=role))
    assert "s3cret" not in json.dumps(data)
    assert "PASSWORD '***'" in data["sql"]


def test_other_changes_display_their_sql():
    change = DropTable(schema="public", table_name="note")
    assert change.to_display_sql() == change.to_sql()


# =============================================================================
# Against a database
# =============================================================================


def _admin(url: str, sql: str) -> None:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


def _rows(url: str, sql: str, params=None) -> list[tuple]:
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


@pytest.fixture
def database(integration_db_url):
    name = f"pgdeclare_sec_{uuid.uuid4().hex[:8]}"
    _admin(integration_db_url, f'CREATE DATABASE "{name}"')
    url = integration_db_url.rsplit("/", 1)[0] + f"/{name}"
    try:
        yield url
    finally:
        _admin(integration_db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        _admin(
            integration_db_url,
            f'DROP DATABASE IF EXISTS "{name}_pgdeclare_plan" WITH (FORCE)',
        )


@pytest.mark.integration
def test_a_hostile_live_object_is_dropped_as_a_name(database):
    """An undeclared table named like an injection is dropped, nothing else."""
    _admin(database, "CREATE TABLE canary (id int)")
    _admin(database, f"CREATE TABLE {quote_ident(HOSTILE)} (id int)")
    _admin(database, f"ALTER TABLE canary ADD COLUMN {quote_ident(HOSTILE)} int")

    desired = Schema(
        tables={"public.canary": introspect_schema(database).tables["public.canary"]}
    )
    del desired.tables["public.canary"].columns[HOSTILE]

    changes = compute_diff(desired, introspect_schema(database))
    assert sorted(type(c).__name__ for c in changes) == ["DropColumn", "DropTable"]
    success, _ = execute_migration(database, changes)
    assert success is True

    assert _rows(database, "SELECT to_regclass('public.canary')::text") == [("canary",)]
    assert _rows(
        database, "SELECT count(*) FROM pg_class WHERE relname = %s", (HOSTILE,)
    ) == [(0,)]
    assert _rows(
        database,
        "SELECT count(*) FROM information_schema.columns WHERE column_name = %s",
        (HOSTILE,),
    ) == [(0,)]


@pytest.mark.integration
def test_diff_json_never_contains_a_declared_password(tmp_path, database, capsys):
    role = f"pgdeclare_login_{uuid.uuid4().hex[:8]}"
    secret = f"never-print-{uuid.uuid4().hex}"
    schemas = tmp_path / "sql" / "schemas"
    schemas.mkdir(parents=True)
    (schemas / "roles.sql").write_text(
        f"CREATE ROLE {role} WITH LOGIN PASSWORD '{secret}';\n"
        "CREATE TABLE note (id int);\n"
    )
    base = ["--database-url", database, "--sql-dir", str(tmp_path / "sql")]
    try:
        assert main(["diff", "--json", "--role", role, *base]) == 0
        out = capsys.readouterr()
        payload = json.loads(out.out)
        # Roles are cluster-wide, so building the plan on the same server may
        # already have created this one; the password must not show either way.
        assert any(c["kind"] == "CreateTable" for c in payload["changes"])
        assert secret not in out.out and secret not in out.err

        assert main(["diff", "--role", role, *base]) == 0
        assert main(["apply", "--dry-run", "--role", role, *base]) == 0
        out = capsys.readouterr()
        assert secret not in out.out and secret not in out.err
    finally:
        _admin(database, f'DROP ROLE IF EXISTS "{role}"')


@pytest.mark.integration
def test_created_schemas_are_not_opened_to_public_by_default(database):
    ensure_schemas(database, ("closed",))
    ensure_schemas(
        database, ("opened",), PgDeclareConfig(grant_public_on_created_schemas=True)
    )
    acl = dict(
        _rows(
            database,
            "SELECT nspname, coalesce(nspacl::text, '') FROM pg_namespace "
            "WHERE nspname IN ('closed', 'opened')",
        )
    )
    assert "=UC/" not in acl["closed"]
    assert "=UC/" in acl["opened"]
