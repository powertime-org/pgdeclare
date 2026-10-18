"""`--prove`: rehearse the real run on a throwaway clone of the target."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import psycopg2
import pytest

from pgdeclare import (
    PgDeclareConfig,
    ProveFailedError,
    TableRows,
    apply_declared_schema,
    prove_declared_schema,
)
from pgdeclare.changes import AlterRole, GrantRoleMembership
from pgdeclare.cli import EXIT_CHANGES, EXIT_ERROR, EXIT_OK, main
from pgdeclare.migrate import is_cluster_change
from pgdeclare.models import Role, RoleMembership
from pgdeclare.prove import (
    LedgerEntry,
    ProveResult,
    clone_name,
    compare_row_counts,
    report_prove_result,
)

# =============================================================================
# Unit
# =============================================================================


def test_clone_names_are_random_and_fit_an_identifier():
    a, b = clone_name("app"), clone_name("app")
    assert a != b
    assert a.startswith("app_pgdeclare_prove_")
    long = clone_name("x" * 63)
    assert len(long.encode()) <= 63 and "_pgdeclare_prove_" in long
    # Multi-byte names are cut on a character boundary.
    assert len(clone_name("é" * 40).encode()) <= 63


def test_row_counts_match_tables_by_oid():
    before = {
        1: ("public.kept", 3),
        2: ("public.renamed", 2),
        3: ("public.shrunk", 5),
        4: ("public.gone", 1),
        5: ("public.dropped", 4),
        6: ("public.grown", 1),
    }
    after = {
        1: ("public.kept", 3),
        2: ("public.new_name", 2),
        3: ("public.shrunk", 4),
        6: ("public.grown", 2),
    }
    rows = {r.table: r for r in compare_row_counts(before, after, {"dropped"})}
    assert "public.kept" not in rows
    assert rows["public.renamed"].renamed_to == "public.new_name"
    assert not rows["public.renamed"].lost_rows
    assert rows["public.shrunk"].lost_rows and not rows["public.shrunk"].covered_by_drop
    assert rows["public.gone"].after is None and rows["public.gone"].lost_rows
    assert rows["public.dropped"].covered_by_drop
    assert not rows["public.grown"].lost_rows


def test_a_qualified_drop_directive_covers_a_table():
    rows = compare_row_counts({1: ("app.t", 2)}, {}, {"app.t"})
    assert rows == [TableRows("app.t", 2, None, covered_by_drop=True)]


def test_role_changes_are_cluster_changes():
    role = Role(name="reader")
    assert is_cluster_change(AlterRole(role=role))
    assert is_cluster_change(
        GrantRoleMembership(membership=RoleMembership(role="a", member="b"))
    )


def test_the_report_names_blocks_ddl_and_rows():
    lines: list[str] = []
    result = ProveResult(
        database="app",
        clone="app_pgdeclare_prove_0",
        executed_sql=["ALTER TABLE t\n  ADD COLUMN c int"],
        ledger_entries=[
            LedgerEntry("2026-01-01_a", "before", True),
            LedgerEntry("2026-01-02_b", "after", False),
        ],
        row_changes=[TableRows("public.t", 2, None, covered_by_drop=True)],
    )
    report_prove_result(result, lines.append, lambda _style, text: text)
    text = "\n".join(lines)
    assert "@migrate-before:2026-01-01_a would run" in text
    assert "@migrate-after:2026-01-02_b would be recorded, not run" in text
    assert "    ADD COLUMN c int" in text
    assert "public.t: 2 row(s), table dropped (covered by @drop)" in text


def test_a_key_value_dsn_gets_its_database_replaced():
    # Returned unchanged, it would point the plan build (or the clone) at the
    # target database.
    from psycopg2.extensions import parse_dsn

    from pgdeclare.schema_parser import _replace_db_in_url

    swapped = _replace_db_in_url("host=db user=app dbname=app", "app_plan")
    assert parse_dsn(swapped) == {"host": "db", "user": "app", "dbname": "app_plan"}
    assert _replace_db_in_url("postgresql://h/app?sslmode=require", "p") == (
        "postgresql://h/p?sslmode=require"
    )


def test_prove_needs_a_dry_run(tmp_path):
    with pytest.raises(ValueError, match="dry_run"):
        apply_declared_schema("postgresql:///x", tmp_path, "plan", prove=True)


def test_apply_prove_without_dry_run_is_refused(tmp_path, capsys):
    (tmp_path / "schemas").mkdir()
    code = main(
        [
            "apply",
            "--prove",
            "--database-url",
            "postgresql:///x",
            "--sql-dir",
            str(tmp_path),
        ]
    )
    assert code == EXIT_ERROR
    assert "use it with --dry-run" in capsys.readouterr().err


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


def _leftovers(url: str, name: str) -> list[tuple]:
    return _rows(
        url,
        f"SELECT datname FROM pg_database WHERE datname LIKE '{name}\\_pgdeclare\\_%'",
    )


@pytest.fixture
def target(integration_db_url):
    """An empty database for one test; it and any scratch databases are dropped."""
    name = f"pgdeclare_prove_{uuid.uuid4().hex[:8]}"
    _admin(integration_db_url, f'CREATE DATABASE "{name}"')
    url = integration_db_url.rsplit("/", 1)[0] + f"/{name}"
    try:
        yield url
    finally:
        _admin(integration_db_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        for (db,) in _leftovers(integration_db_url, name):
            _admin(integration_db_url, f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')


def _name(url: str) -> str:
    return url.rsplit("/", 1)[1]


def _write(sql_dir: Path, content: str) -> Path:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "main.sql").write_text(content)
    return sql_dir


def _ledger_exists(url: str) -> bool:
    found = _rows(url, "SELECT to_regclass('pgdeclare.applied_migrations')::text")
    return found != [(None,)]


def _seeded(tmp_path: Path, target: str) -> tuple[Path, list[str]]:
    """A deployed `note` table with two rows."""
    sql_dir = _write(
        tmp_path / "sql", "CREATE TABLE note (id int PRIMARY KEY, author text);"
    )
    base = ["--database-url", target, "--sql-dir", str(sql_dir)]
    assert main(["apply", *base]) == EXIT_OK
    _admin(target, "INSERT INTO note VALUES (1, 'ada'), (2, 'bo')")
    return sql_dir, base


RENAME = """
CREATE TABLE note (id int PRIMARY KEY, writer text, title text);

-- @migrate-before:2026-02-01_rename_author
ALTER TABLE note RENAME COLUMN author TO writer;
-- @end-migrate

-- @migrate-after:2026-02-02_titles
UPDATE note SET title = upper(writer);
-- @end-migrate
"""


@pytest.mark.integration
def test_a_rename_proves_clean_and_the_target_is_untouched(tmp_path, target, capsys):
    sql_dir, base = _seeded(tmp_path, target)
    assert not _ledger_exists(target)
    _write(sql_dir, RENAME)

    assert main(["apply", "--dry-run", "--prove", *base]) == EXIT_OK
    out = capsys.readouterr().out
    assert "@migrate-before:2026-02-01_rename_author would run" in out
    assert "@migrate-after:2026-02-02_titles would run" in out
    # The rename ran first, so the only DDL is the new column, not drop plus add.
    assert '"title"' in out and "DROP COLUMN" not in out
    assert "Proved" in out

    # The target is exactly as it was, and the clone is gone.
    assert _rows(target, "SELECT id, author FROM note ORDER BY id") == [
        (1, "ada"),
        (2, "bo"),
    ]
    assert not _ledger_exists(target)
    assert _leftovers(target, _name(target)) == []

    # diff --prove: the usual diff on stdout, the rehearsal as JSON too.
    assert main(["diff", "--prove", "--exit-code", "--json", *base]) == EXIT_CHANGES
    payload = json.loads(capsys.readouterr().out)
    assert [e["id"] for e in payload["prove"]["ledger_entries"]] == [
        "2026-02-01_rename_author",
        "2026-02-02_titles",
    ]
    assert payload["prove"]["row_changes"] == []
    assert _leftovers(target, _name(target)) == []

    # The real run then does what the rehearsal said.
    assert main(["apply", *base]) == EXIT_OK
    assert _rows(target, "SELECT writer, title FROM note ORDER BY id") == [
        ("ada", "ADA"),
        ("bo", "BO"),
    ]


@pytest.mark.integration
def test_rows_deleted_without_drop_fail_the_proof(tmp_path, target, capsys):
    sql_dir, base = _seeded(tmp_path, target)
    _write(
        sql_dir,
        "CREATE TABLE note (id int PRIMARY KEY, author text);\n"
        "-- @migrate-after:2026-03-01_prune\n"
        "DELETE FROM note WHERE id = 2;\n"
        "-- @end-migrate\n",
    )
    assert main(["apply", "--dry-run", "--prove", *base]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "table public.note would lose 1 of 2 row(s) without a @drop" in err
    assert _rows(target, "SELECT count(*) FROM note") == [(2,)]
    assert _leftovers(target, _name(target)) == []

    # Consent through @drop makes the loss expected.
    _write(
        sql_dir,
        "CREATE TABLE note (id int PRIMARY KEY, author text);\n"
        "-- @migrate-after:2026-03-01_prune\n"
        "-- @drop note\n"
        "DELETE FROM note WHERE id = 2;\n"
        "-- @end-migrate\n",
    )
    assert main(["apply", "--dry-run", "--prove", *base]) == EXIT_OK
    assert "(covered by @drop)" in capsys.readouterr().out


NOT_CONVERGING = """
CREATE TABLE note (id int PRIMARY KEY, author text);
-- @migrate-after:2026-04-01_sneaky_default
ALTER TABLE note ALTER COLUMN author SET DEFAULT 'anon';
-- @end-migrate
"""


@pytest.mark.integration
def test_a_run_that_does_not_converge_fails_the_proof(tmp_path, target, capsys):
    sql_dir, base = _seeded(tmp_path, target)
    _write(sql_dir, NOT_CONVERGING)

    assert main(["apply", "--dry-run", "--prove", *base]) == EXIT_ERROR
    assert "did not converge" in capsys.readouterr().err
    assert _leftovers(target, _name(target)) == []

    # --force lets a real run continue past non-convergence; a proof never does.
    assert main(["apply", "--dry-run", "--prove", "--force", *base]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "prove failed: the schema did not converge" in err
    assert _leftovers(target, _name(target)) == []
    assert not _ledger_exists(target)


@pytest.mark.integration
def test_a_failing_block_fails_the_proof_and_drops_the_clone(tmp_path, target, capsys):
    sql_dir, base = _seeded(tmp_path, target)
    _write(
        sql_dir,
        "CREATE TABLE note (id int PRIMARY KEY, author text);\n"
        "-- @migrate-before:2026-05-01_broken\n"
        "UPDATE nowhere SET x = 1;\n"
        "-- @end-migrate\n",
    )
    assert main(["diff", "--prove", *base]) == EXIT_ERROR
    assert "the real run would fail" in capsys.readouterr().err
    assert _leftovers(target, _name(target)) == []


@pytest.mark.integration
def test_another_session_on_the_target_needs_exclusive_access(tmp_path, target, capsys):
    sql_dir, base = _seeded(tmp_path, target)
    other = psycopg2.connect(target)
    try:
        assert main(["apply", "--dry-run", "--prove", *base]) == EXIT_ERROR
    finally:
        other.close()
    err = capsys.readouterr().err
    assert "other session(s) are connected" in err
    assert "exclusive access" in err and "staging copy" in err
    assert _leftovers(target, _name(target)) == []


@pytest.mark.integration
def test_a_fresh_database_records_blocks_without_running_them(tmp_path, target):
    sql_dir = _write(tmp_path / "sql", RENAME)
    lines: list[str] = []
    cfg = PgDeclareConfig(report=lines.append)
    plan = f"{_name(target)}_pgdeclare_plan_t"
    _admin(target, f'CREATE DATABASE "{plan}"')
    result = prove_declared_schema(target, sql_dir, plan, config=cfg)
    assert [(e.id, e.executed) for e in result.ledger_entries] == [
        ("2026-02-01_rename_author", False),
        ("2026-02-02_titles", False),
    ]
    assert any("CREATE TABLE" in s for s in result.executed_sql)
    assert _rows(target, "SELECT to_regclass('public.note')::text") == [(None,)]
    assert not _ledger_exists(target)
    assert [d for (d,) in _leftovers(target, _name(target))] == [plan]

    # The library raises with the reasons and the partial result.
    _admin(target, "CREATE TABLE note (id int PRIMARY KEY, author text)")
    _admin(target, "INSERT INTO note VALUES (1, 'ada')")
    sql_dir = _write(tmp_path / "sql", "CREATE TABLE other (id int);")
    with pytest.raises(ProveFailedError) as caught:
        prove_declared_schema(target, sql_dir, plan, config=cfg)
    assert "@drop directive required" in str(caught.value)


@pytest.mark.integration
def test_a_role_without_createdb_is_told_what_prove_needs(tmp_path, target, capsys):
    sql_dir, _base = _seeded(tmp_path, target)
    role = f"pgdeclare_noclone_{uuid.uuid4().hex[:8]}"
    plan = f"{_name(target)}_pgdeclare_plan_mine"
    _admin(target, f"CREATE ROLE \"{role}\" LOGIN PASSWORD 'pw'")
    try:
        _admin(target, f'CREATE DATABASE "{plan}" OWNER "{role}"')
        # PostgreSQL 14 gives public to the bootstrap superuser.
        _admin(
            target.rsplit("/", 1)[0] + f"/{plan}",
            f'ALTER SCHEMA public OWNER TO "{role}"',
        )
        _admin(target, f'GRANT ALL ON SCHEMA public TO "{role}"')
        _admin(target, f'GRANT ALL ON ALL TABLES IN SCHEMA public TO "{role}"')
        parsed = urlsplit(target)
        url = parsed._replace(
            netloc=f"{role}:pw@{parsed.hostname}:{parsed.port or 5432}"
        ).geturl()
        code = main(
            [
                "diff",
                "--prove",
                "--database-url",
                url,
                "--sql-dir",
                str(sql_dir),
                "--plan-database",
                plan,
            ]
        )
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "Proving needs CREATEDB" in err
    finally:
        _admin(target, f'DROP DATABASE IF EXISTS "{plan}" WITH (FORCE)')
        _admin(target, f'DROP OWNED BY "{role}"')
        _admin(target, f'DROP ROLE IF EXISTS "{role}"')
