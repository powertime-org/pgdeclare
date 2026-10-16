"""The engine on its own: a plain project, the default config, a real database.

Most of this suite drives the engine's parts directly. This file proves the defaults work for a project that has none of it: one
`public` schema, no managed roles, no bootstrap SQL, the ledger in
`pgdeclare.applied_migrations`. Requires TEST_DATABASE_URL.
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
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()


@pytest.fixture
def databases(server_url):
    """A target database and a plan database, both empty, both dropped after."""
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_target_{suffix}", f"pgdeclare_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


V1 = """
CREATE TABLE IF NOT EXISTS note (
    id integer PRIMARY KEY,
    body text NOT NULL
);

-- @migrate-after:2026-01-01_seed_note
INSERT INTO note (id, body) VALUES (1, 'hello') ON CONFLICT DO NOTHING;
-- @end-migrate

DO $$
BEGIN
  CREATE TEMP TABLE IF NOT EXISTS do_block_ran (x int);
END $$;
"""

V2 = """
CREATE TABLE IF NOT EXISTS note (
    id integer PRIMARY KEY,
    body text NOT NULL,
    shout text
);

-- @migrate-after:2026-01-01_seed_note
INSERT INTO note (id, body) VALUES (1, 'hello') ON CONFLICT DO NOTHING;
-- @end-migrate

-- @migrate-after:2026-01-02_backfill_shout
UPDATE note SET shout = upper(body) WHERE shout IS NULL;
-- @end-migrate
"""


def _write(sql_dir: Path, content: str) -> None:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "note.sql").write_text(content)


def test_default_config_migrates_a_plain_project(tmp_path, databases):
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"
    lines: list[str] = []
    config = PgDeclareConfig(report=lines.append)

    # Fresh database: the schema is created, the inline block is recorded as
    # satisfied (not executed), the DO block runs, and the run converges.
    _write(sql_dir, V1)
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)

    assert _rows(target_url, "SELECT count(*) FROM note") == [(0,)]
    assert _rows(
        target_url, "SELECT id FROM pgdeclare.applied_migrations ORDER BY id"
    ) == [("2026-01-01_seed_note",)]
    assert "Executing 1 DO blocks..." in lines
    assert "✓ Schema converged (a second diff finds no changes)" in lines

    # Second version: a new column and a backfill that needs it.
    _write(sql_dir, V2)
    lines.clear()
    _admin(target_url, "INSERT INTO note (id, body) VALUES (2, 'two')")
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)

    assert _rows(target_url, "SELECT id, shout FROM note ORDER BY id") == [(2, "TWO")]
    assert _rows(
        target_url, "SELECT id FROM pgdeclare.applied_migrations ORDER BY id"
    ) == [("2026-01-01_seed_note",), ("2026-01-02_backfill_shout",)]
    assert "✓ Schema changes applied successfully" in lines

    # And the desired schema now matches what is there.
    desired = build_desired_schema(sql_dir, target_url, plan_db, config=config)
    actual = introspect_schema(target_url, config=config)
    assert compute_diff(desired, actual, schemas=list(config.schemas)) == []


def test_a_custom_ledger_and_layout_are_honoured(tmp_path, databases):
    target_url, plan_db = databases
    sql_dir = tmp_path / "db"
    (sql_dir / "ddl").mkdir(parents=True)
    (sql_dir / "ddl" / "note.sql").write_text(V1)
    config = PgDeclareConfig(
        schemas_subdir="ddl",
        ledger_table="ops.inline_migrations",
        report=lambda _line: None,
    )

    apply_declared_schema(target_url, sql_dir, plan_db, config=config)

    assert _rows(target_url, "SELECT id FROM ops.inline_migrations") == [
        ("2026-01-01_seed_note",)
    ]
    assert _rows(
        target_url, "SELECT to_regclass('pgdeclare.applied_migrations')::text"
    ) == [(None,)]


def test_not_null_is_a_column_property_not_a_constraint(databases):
    """PostgreSQL 18 stores NOT NULL as a pg_constraint row; the engine must not.

    Nullability is modelled on the column. Introspecting the NOT NULL rows as
    constraints too would manage it twice, and a desired schema built without
    them would plan to drop them.
    """
    target_url, _plan_db = databases
    _admin(target_url, "CREATE TABLE t (id int NOT NULL, label text NOT NULL)")

    table = introspect_schema(target_url).tables["public.t"]

    assert [c.type for c in table.constraints] == []
    assert table.columns["id"].nullable is False
    assert table.columns["label"].nullable is False


def test_extension_objects_are_not_introspected(databases):
    """What an extension owns is the extension's, never the schema files'."""
    target_url, _plan_db = databases
    _admin(target_url, "CREATE EXTENSION IF NOT EXISTS citext")
    _admin(target_url, "CREATE TABLE t (name citext)")

    schema = introspect_schema(target_url)

    assert "public.t" in schema.tables
    assert not [f for f in schema.functions.values() if f.name.startswith("citext")]
