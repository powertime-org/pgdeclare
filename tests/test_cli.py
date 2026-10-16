"""The `pgdeclare` command line, end to end against a real server where needed."""

import json
import tomllib
import uuid
from pathlib import Path

import psycopg2
import pytest

import pgdeclare
from pgdeclare import PgDeclareConfig
from pgdeclare.changes import AddColumn
from pgdeclare.cli import (
    EXIT_CHANGES,
    EXIT_ERROR,
    EXIT_OK,
    change_to_dict,
    declared_extensions,
    main,
)
from pgdeclare.models import Column

ROOT = Path(__file__).parent.parent


# =============================================================================
# Without a database
# =============================================================================


def test_version_matches_pyproject(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert capsys.readouterr().out.strip() == f"pgdeclare {project['version']}"
    assert pgdeclare.__version__ == project["version"]


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2  # argparse usage error


def test_no_database_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main(["diff", "--sql-dir", str(tmp_path)]) == EXIT_ERROR
    assert "no database" in capsys.readouterr().err


def test_a_missing_sql_dir_is_an_error(tmp_path, capsys):
    missing = tmp_path / "nope"
    code = main(
        ["diff", "--database-url", "postgresql://x/y", "--sql-dir", str(missing)]
    )
    assert code == EXIT_ERROR
    assert "SQL directory not found" in capsys.readouterr().err


def test_a_key_value_dsn_is_refused(tmp_path, capsys):
    code = main(["diff", "--database-url", "dbname=x", "--sql-dir", str(tmp_path)])
    assert code == EXIT_ERROR
    assert "postgresql:// URL" in capsys.readouterr().err


def test_change_to_dict_carries_the_sql():
    change = AddColumn(
        schema="public", table="t", column=Column(name="c", type="text", nullable=True)
    )
    data = change_to_dict(change)
    assert data["kind"] == "AddColumn"
    assert data["category"] == "online"
    assert data["object_name"] == "public.t.c"
    assert data["sql"] == change.to_sql()


def test_declared_extensions_are_made_idempotent(tmp_path):
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "a.sql").write_text(
        "CREATE EXTENSION citext;\n"
        "create extension if not exists pgcrypto;\n"
        "CREATE TABLE t (id int);\n"
    )
    assert declared_extensions(tmp_path, PgDeclareConfig()) == [
        "CREATE EXTENSION IF NOT EXISTS citext;",
        "create extension if not exists pgcrypto;",
    ]


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


def _rows(url: str, sql: str) -> list[tuple]:
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


@pytest.fixture
def target(integration_db_url):
    """An empty database for one test; dropped afterwards."""
    name = f"pgdeclare_cli_{uuid.uuid4().hex[:8]}"
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


def _write(sql_dir: Path, content: str, name: str = "main.sql") -> Path:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / name).write_text(content)
    return sql_dir


V1 = """
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE IF NOT EXISTS note (
    id integer PRIMARY KEY,
    author citext NOT NULL,
    body text NOT NULL
);

-- @migrate-after:2026-01-01_seed
INSERT INTO note (id, author, body) VALUES (1, 'ada', 'hello');
-- @end-migrate
"""

V2 = (
    V1.replace("body text NOT NULL", "body text NOT NULL,\n    title text")
    + """
-- @migrate-after:2026-01-02_titles
UPDATE note SET title = upper(body) WHERE title IS NULL;
-- @end-migrate
"""
)


@pytest.mark.integration
def test_diff_apply_diff_cycle(tmp_path, target, capsys):
    sql = str(_write(tmp_path / "sql", V1))
    base = ["--database-url", target, "--sql-dir", sql]

    # A fresh database: changes pending, exit 2 with --exit-code, 0 without.
    assert main(["diff", *base]) == EXIT_OK
    assert main(["diff", "--exit-code", *base]) == EXIT_CHANGES
    out = capsys.readouterr().out
    assert "Create table public.note" in out
    # Extension-owned functions are the extension's, never planned.
    assert "citext_eq" not in out
    assert "(fresh database: recorded, not run)" in out

    assert main(["apply", *base]) == EXIT_OK
    assert "Schema converged" in capsys.readouterr().out
    assert _rows(target, "SELECT id FROM pgdeclare.applied_migrations") == [
        ("2026-01-01_seed",)
    ]
    # The plan database the CLI created is gone again.
    plan = target.rsplit("/", 1)[1] + "_pgdeclare_plan"
    assert _rows(target, f"SELECT 1 FROM pg_database WHERE datname = '{plan}'") == []

    assert main(["diff", "--exit-code", "--json", *base]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["changes"] == [] and payload["pending_migrations"] == []

    # A new column and a backfill that needs it.
    _write(tmp_path / "sql", V2)
    _admin(target, "INSERT INTO note (id, author, body) VALUES (2, 'bo', 'two')")
    assert main(["diff", "--exit-code", "--json", *base]) == EXIT_CHANGES
    payload = json.loads(capsys.readouterr().out)
    assert [c["kind"] for c in payload["changes"]] == ["AddColumn"]
    assert payload["changes"][0]["sql"].startswith('ALTER TABLE "public"."note"')
    assert [b["id"] for b in payload["pending_migrations"]] == ["2026-01-02_titles"]
    assert payload["fresh_database"] is False

    # Dry run: nothing written.
    assert main(["apply", "--dry-run", *base]) == EXIT_OK
    assert _rows(
        target,
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name = 'note' AND column_name = 'title'",
    ) == [(0,)]

    assert main(["apply", *base]) == EXIT_OK
    assert _rows(target, "SELECT id, title FROM note ORDER BY id") == [(2, "TWO")]
    assert main(["diff", "--exit-code", *base]) == EXIT_OK


@pytest.mark.integration
def test_a_drop_needs_a_directive(tmp_path, target, capsys):
    sql_dir = tmp_path / "sql"
    base = ["--database-url", target, "--sql-dir", str(sql_dir)]
    _write(sql_dir, "CREATE TABLE note (id int PRIMARY KEY, old text);")
    assert main(["apply", *base]) == EXIT_OK
    _admin(target, "INSERT INTO note VALUES (1, 'keep me')")

    _write(sql_dir, "CREATE TABLE note (id int PRIMARY KEY);")
    assert main(["apply", *base]) == EXIT_ERROR
    assert "@drop" in capsys.readouterr().err
    assert _rows(target, "SELECT old FROM note") == [("keep me",)]

    _write(
        sql_dir,
        "CREATE TABLE note (id int PRIMARY KEY);\n"
        "-- @migrate-before:2026-02-01_drop_old\n"
        "-- @drop note.old\n"
        "SELECT 1;\n"
        "-- @end-migrate\n",
    )
    assert main(["apply", *base]) == EXIT_OK
    assert _rows(
        target,
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_name = 'note' AND column_name = 'old'",
    ) == [(0,)]


@pytest.mark.integration
def test_schema_and_ledger_flags(tmp_path, target):
    sql_dir = _write(
        tmp_path / "sql",
        "CREATE TABLE app.item (id int PRIMARY KEY);\n"
        "-- @migrate-after:2026-03-01_seed\n"
        "INSERT INTO app.item VALUES (1);\n"
        "-- @end-migrate\n",
    )
    args = [
        "apply",
        "--database-url",
        target,
        "--sql-dir",
        str(sql_dir),
        "--schema",
        "app",
        "--ledger-table",
        "ops.ledger",
    ]
    assert main(args) == EXIT_OK
    assert _rows(target, "SELECT id FROM ops.ledger") == [("2026-03-01_seed",)]
    assert _rows(target, "SELECT to_regclass('app.item')::text") == [("app.item",)]
    assert _rows(
        target, "SELECT to_regclass('pgdeclare.applied_migrations')::text"
    ) == [(None,)]


@pytest.mark.integration
def test_database_url_from_the_environment(tmp_path, target, monkeypatch, capsys):
    sql_dir = _write(tmp_path / "sql", "CREATE TABLE t (id int);")
    monkeypatch.setenv("DATABASE_URL", target)
    assert main(["diff", "--exit-code", "--sql-dir", str(sql_dir)]) == EXIT_CHANGES
    assert "Create table public.t" in capsys.readouterr().out


@pytest.mark.integration
def test_a_broken_schema_file_is_an_error(tmp_path, target, capsys):
    sql_dir = _write(tmp_path / "sql", "CREATE TABLE t (id int REFERENCES nowhere);")
    code = main(["diff", "--database-url", target, "--sql-dir", str(sql_dir)])
    assert code == EXIT_ERROR
    assert "pgdeclare: error" in capsys.readouterr().err


@pytest.mark.integration
def test_dropping_a_table_with_data_takes_an_unqualified_directive(tmp_path, target):
    sql_dir = tmp_path / "sql"
    base = ["apply", "--database-url", target, "--sql-dir", str(sql_dir)]
    _write(sql_dir, "CREATE TABLE keep (id int);\nCREATE TABLE old (id int);")
    assert main(base) == EXIT_OK
    _admin(target, "INSERT INTO old VALUES (1)")

    _write(sql_dir, "CREATE TABLE keep (id int);")
    assert main(base) == EXIT_ERROR

    _write(
        sql_dir,
        "CREATE TABLE keep (id int);\n"
        "-- @migrate-before:2026-02-02_drop_old\n"
        "-- @drop old\n"
        "SELECT 1;\n"
        "-- @end-migrate\n",
    )
    assert main(base) == EXIT_OK
    assert _rows(target, "SELECT to_regclass('public.old')::text") == [(None,)]
