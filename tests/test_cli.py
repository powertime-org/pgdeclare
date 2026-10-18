"""The `pgdeclare` command line, end to end against a real server where needed."""

import json
import tomllib
import uuid
from pathlib import Path
from urllib.parse import urlsplit

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
        for (plan,) in _plan_databases(integration_db_url, name):
            _admin(integration_db_url, f'DROP DATABASE IF EXISTS "{plan}" WITH (FORCE)')


def _plan_databases(url: str, target_name: str) -> list[tuple]:
    return _rows(
        url,
        "SELECT datname FROM pg_database "
        f"WHERE datname LIKE '{target_name}\\_pgdeclare\\_plan%'",
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
    assert _plan_databases(target, target.rsplit("/", 1)[1]) == []

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


@pytest.mark.integration
def test_the_target_is_never_the_plan_database(tmp_path, target, capsys):
    sql = str(_write(tmp_path / "sql", V1))
    _admin(target, "CREATE TABLE keep (v int)")
    _admin(target, "INSERT INTO keep VALUES (1)")
    name = target.rsplit("/", 1)[1]
    code = main(
        ["apply", "--database-url", target, "--sql-dir", sql, "--plan-database", name]
    )
    assert code == EXIT_ERROR
    assert "cannot be the target database" in capsys.readouterr().err
    assert _rows(target, "SELECT v FROM keep") == [(1,)]


@pytest.mark.integration
def test_a_plan_database_owned_by_someone_else_is_refused(tmp_path, target, capsys):
    sql = str(_write(tmp_path / "sql", V1))
    suffix = uuid.uuid4().hex[:8]
    role, plan = f"pgdeclare_other_{suffix}", f"pgdeclare_foreign_{suffix}"
    _admin(target, f'CREATE ROLE "{role}" NOLOGIN')
    try:
        _admin(target, f'CREATE DATABASE "{plan}" OWNER "{role}"')
        code = main(
            [
                "diff",
                "--database-url",
                target,
                "--sql-dir",
                sql,
                "--plan-database",
                plan,
            ]
        )
        assert code == EXIT_ERROR
        assert "not owned by the current user" in capsys.readouterr().err
        assert _rows(target, f"SELECT 1 FROM pg_database WHERE datname = '{plan}'")
    finally:
        _admin(target, f'DROP DATABASE IF EXISTS "{plan}" WITH (FORCE)')
        _admin(target, f'DROP ROLE IF EXISTS "{role}"')


@pytest.mark.integration
def test_an_owned_plan_database_is_reused_and_kept(tmp_path, target):
    sql = str(_write(tmp_path / "sql", V1))
    plan = f"pgdeclare_mine_{uuid.uuid4().hex[:8]}"
    _admin(target, f'CREATE DATABASE "{plan}"')
    try:
        code = main(
            [
                "diff",
                "--database-url",
                target,
                "--sql-dir",
                sql,
                "--plan-database",
                plan,
            ]
        )
        assert code == EXIT_OK
        assert _rows(target, f"SELECT 1 FROM pg_database WHERE datname = '{plan}'")
    finally:
        _admin(target, f'DROP DATABASE IF EXISTS "{plan}" WITH (FORCE)')


@pytest.mark.integration
def test_an_undeclared_managed_role_is_a_clear_error(tmp_path, target, capsys):
    sql_dir = tmp_path / "sql"
    role = f"pgdeclare_reader_{uuid.uuid4().hex[:8]}"
    base = ["--database-url", target, "--sql-dir", str(sql_dir), "--role", role]
    _write(sql_dir, "CREATE TABLE note (id int PRIMARY KEY);")
    assert main(["diff", *base]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert f"managed role(s) not found: {role}" in err
    assert "CREATE ROLE" in err

    _write(sql_dir, f'CREATE ROLE "{role}" NOLOGIN;', name="00_roles.sql")
    try:
        assert main(["apply", *base]) == EXIT_OK
    finally:
        _admin(target, f'DROP ROLE IF EXISTS "{role}"')


@pytest.mark.integration
def test_a_dry_run_rename_is_not_a_blocked_drop(tmp_path, target, capsys):
    sql_dir = tmp_path / "sql"
    base = ["--database-url", target, "--sql-dir", str(sql_dir)]
    _write(sql_dir, "CREATE TABLE note (id int PRIMARY KEY, author text);")
    assert main(["apply", *base]) == EXIT_OK
    _admin(target, "INSERT INTO note VALUES (1, 'ada')")

    _write(
        sql_dir,
        "CREATE TABLE note (id int PRIMARY KEY, writer text);\n"
        "-- @migrate-before:2026-02-01_rename_author\n"
        "ALTER TABLE note RENAME COLUMN author TO writer;\n"
        "-- @end-migrate\n",
    )
    # The dry run cannot see the rename, so the drop is a warning, not an error.
    assert main(["apply", "--dry-run", *base]) == EXIT_OK
    assert _rows(target, "SELECT author FROM note") == [("ada",)]
    assert main(["apply", *base]) == EXIT_OK
    assert _rows(target, "SELECT writer FROM note") == [("ada",)]

    # With nothing pending, a real data-loss drop still fails the dry run.
    _write(sql_dir, "CREATE TABLE note (id int PRIMARY KEY);")
    assert main(["apply", "--dry-run", *base]) == EXIT_ERROR


@pytest.mark.integration
def test_a_role_without_createdb_is_told_what_to_do(tmp_path, target, capsys):
    sql = str(_write(tmp_path / "sql", "CREATE TABLE note (id int PRIMARY KEY);"))
    role = f"pgdeclare_nocreatedb_{uuid.uuid4().hex[:8]}"
    _admin(target, f"CREATE ROLE \"{role}\" LOGIN PASSWORD 'pw'")
    try:
        parsed = urlsplit(target)
        url = parsed._replace(
            netloc=f"{role}:pw@{parsed.hostname}:{parsed.port or 5432}"
        ).geturl()
        assert main(["diff", "--database-url", url, "--sql-dir", sql]) == EXIT_ERROR
        err = capsys.readouterr().err
        assert "lacks CREATEDB" in err and "--plan-database" in err
    finally:
        _admin(target, f'DROP ROLE IF EXISTS "{role}"')


COMBINED = """
CREATE DOMAIN email AS text CONSTRAINT email_has_at CHECK (VALUE LIKE '%@%');
CREATE SEQUENCE ticket_no AS integer START WITH 1000;

CREATE TABLE IF NOT EXISTS person (
    id     serial PRIMARY KEY,
    email  email NOT NULL,
    ticket integer NOT NULL DEFAULT nextval('ticket_no')
);
COMMENT ON TABLE person IS 'People';
COMMENT ON COLUMN person.email IS 'Contact address';

CREATE VIEW person_emails AS SELECT id, email FROM person;
COMMENT ON VIEW person_emails IS 'Emails only';

CREATE MATERIALIZED VIEW person_count AS SELECT count(*) AS n FROM person_emails;
CREATE INDEX IF NOT EXISTS person_count_n_idx ON person_count (n);
"""


@pytest.mark.integration
def test_new_object_kinds_converge_together(tmp_path, target):
    sql_dir = tmp_path / "sql"
    base = ["--database-url", target, "--sql-dir", str(sql_dir)]
    _write(sql_dir, COMBINED)
    assert main(["apply", *base]) == EXIT_OK
    assert main(["diff", "--exit-code", *base]) == EXIT_OK
    _admin(target, "INSERT INTO person (email) VALUES ('a@b.c')")

    # Change the view under the matview, a comment, and the sequence options.
    changed = (
        COMBINED.replace(
            "SELECT id, email FROM person;", "SELECT id, email, ticket FROM person;"
        )
        .replace("'Emails only'", "'Emails and tickets'")
        .replace("START WITH 1000", "START WITH 1000 CACHE 5")
    )
    _write(sql_dir, changed)
    assert main(["apply", "--offline", *base]) == EXIT_OK
    assert main(["diff", "--exit-code", *base]) == EXIT_OK
    assert _rows(target, "SELECT id, ticket FROM person") == [(1, 1000)]
    assert _rows(
        target, "SELECT obj_description('public.person_emails'::regclass)"
    ) == [("Emails and tickets",)]


ROLLUP = """
CREATE ROLE {role} NOLOGIN;

CREATE COLLATION plain_c (locale = 'C');
CREATE TYPE span AS RANGE (subtype = integer);

CREATE TABLE events (
    id   bigint NOT NULL,
    at   date NOT NULL,
    tag  text COLLATE plain_c,
    PRIMARY KEY (id, at)
) PARTITION BY RANGE (at);
CREATE TABLE events_2026 PARTITION OF events
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE TABLE note (id int PRIMARY KEY, body text);
CREATE PUBLICATION note_feed FOR TABLE note;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {role};

CREATE PROCEDURE log_note(n int) LANGUAGE sql AS $$ SELECT n $$;
CREATE AGGREGATE total(int) (SFUNC = int4pl, STYPE = int, INITCOND = '0');

CREATE EXTENSION IF NOT EXISTS postgres_fdw;
CREATE SERVER elsewhere FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host 'elsewhere.invalid', dbname 'app');
CREATE FOREIGN TABLE remote_note (id int, body text) SERVER elsewhere
    OPTIONS (table_name 'note');
"""


@pytest.mark.integration
def test_rollup_object_kinds_converge_together(tmp_path, target):
    """The 0.1.1 kinds in one schema: apply, re-diff empty, change, re-diff empty."""
    sql_dir = tmp_path / "sql"
    role = f"pgdeclare_rollup_{uuid.uuid4().hex[:8]}"
    base = ["--database-url", target, "--sql-dir", str(sql_dir), "--role", role]
    sql = ROLLUP.format(role=f'"{role}"')
    try:
        _write(sql_dir, sql)
        assert main(["apply", *base]) == EXIT_OK
        assert main(["diff", "--exit-code", *base]) == EXIT_OK

        # A partition made at runtime (pg_partman, say) is left alone, and it
        # carries the parent's column collation like the declared one.
        _admin(
            target,
            "CREATE TABLE events_2030 PARTITION OF events "
            "FOR VALUES FROM ('2030-01-01') TO ('2031-01-01')",
        )
        assert main(["diff", "--exit-code", *base]) == EXIT_OK
        assert _rows(
            target,
            "SELECT c.relname, co.collname FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_collation co ON co.oid = a.attcollation "
            "WHERE a.attname = 'tag' AND c.relname LIKE 'events%' ORDER BY 1",
        ) == [
            ("events", "plain_c"),
            ("events_2026", "plain_c"),
            ("events_2030", "plain_c"),
        ]
        assert _rows(
            target, f"SELECT has_table_privilege('{role}', 'public.note', 'SELECT')"
        ) == [(True,)]

        changed = (
            sql.replace("$$ SELECT n $$", "$$ SELECT n + 1 $$")
            .replace("INITCOND = '0'", "INITCOND = '10'")
            .replace("GRANT SELECT ON TABLES", "GRANT SELECT, INSERT ON TABLES")
            .replace("subtype = integer", "subtype = bigint")
            .replace("FOR TABLE note;", "FOR TABLE note, events;")
            .replace(
                "(id int, body text) SERVER", "(id int, body text, at date) SERVER"
            )
            + "CREATE TABLE events_2027 PARTITION OF events\n"
            "    FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');\n"
        )
        _write(sql_dir, changed)
        assert main(["apply", *base]) == EXIT_OK
        assert main(["diff", "--exit-code", *base]) == EXIT_OK

        assert _rows(
            target,
            "SELECT agginitval FROM pg_aggregate "
            "WHERE aggfnoid = 'public.total(int)'::regprocedure",
        ) == [("10",)]
        assert _rows(
            target,
            "SELECT format_type(rngsubtype, NULL) FROM pg_range "
            "WHERE rngtypid = 'public.span'::regtype",
        ) == [("bigint",)]
        assert _rows(
            target,
            "SELECT tablename FROM pg_publication_tables "
            "WHERE pubname = 'note_feed' ORDER BY 1",
        ) == [("events_2026",), ("events_2027",), ("events_2030",), ("note",)]
        assert _rows(
            target,
            "SELECT relname FROM pg_class "
            "WHERE relname LIKE 'events_20%' AND relkind = 'r' ORDER BY 1",
        ) == [("events_2026",), ("events_2027",), ("events_2030",)]
        assert _rows(
            target,
            "SELECT attname FROM pg_attribute "
            "WHERE attrelid = 'public.remote_note'::regclass AND attnum > 0 "
            "ORDER BY attnum",
        ) == [("id",), ("body",), ("at",)]
        assert _rows(
            target,
            f"SELECT has_table_privilege('{role}', 'public.events_2027', 'INSERT')",
        ) == [(True,)]
    finally:
        _admin(target, f'DROP OWNED BY "{role}"; DROP ROLE IF EXISTS "{role}"')
