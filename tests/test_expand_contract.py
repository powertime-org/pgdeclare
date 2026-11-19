"""Routines the running application may still call: expand/contract.

A blue-green deploy migrates while the previous version of the application
still serves. Dropping a routine it calls, or replacing its signature, breaks
every one of those calls until the new version takes over. The differ marks
such a drop as a contract change, which an `expand_only` run holds back for a
run after the old version has stopped; and a signature change whose old and
new overloads cannot coexist (a call could match both) is offline.

The unit tests drive the differ; the integration tests run the pipeline
against a real server and need TEST_DATABASE_URL.
"""

import os
import uuid
from pathlib import Path

import psycopg2
import pytest

from pgdeclare import (
    ExpandRefusedError,
    PgDeclareConfig,
    apply_declared_schema,
    breaking_blocks,
    breaking_statements,
    compute_diff,
)
from pgdeclare.changes import (
    CreateFunction,
    DropAndCreateFunction,
    DropFunction,
    DropView,
    RevokePrivilege,
)
from pgdeclare.ledger import MigrationBlock
from pgdeclare.migrate import is_held_back
from pgdeclare.models import ChangeCategory, Function, Grant, Schema, View


def _fn(
    name: str,
    args: str,
    arg_types: str,
    min_args: int,
    max_args: int | None,
    body: str = "1",
) -> Function:
    return Function(
        schema="public",
        name=name,
        args=args,
        full_definition=f"CREATE FUNCTION {name}({args}) {body}",
        definition_hash=f"{name}-{args}-{body}",
        arg_types=arg_types,
        min_call_args=min_args,
        max_call_args=max_args,
    )


def _schema(*functions: Function, **kw) -> Schema:
    return Schema(functions={f.signature: f for f in functions}, **kw)


class TestSignatureChanges:
    def test_appended_default_parameters_are_one_online_statement(self):
        # A third parameter with a DEFAULT: the two forms cannot coexist (a
        # call with two arguments matches both), but swapped in one statement
        # every call of the old form resolves to the new one.
        old = _fn("account", "a uuid, s uuid", "uuid, uuid", 2, 2)
        new = _fn("account", "a uuid, s uuid, ip text", "uuid, uuid, text", 2, 3)
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert changes[0].category == ChangeCategory.ONLINE
        swap = changes[0]
        assert isinstance(swap, DropAndCreateFunction)
        assert swap.old_args == "a uuid, s uuid"
        assert "online" in changes[0].description
        assert not is_held_back(changes[0])

    def test_appended_defaults_keep_earlier_defaults_optional(self):
        old = _fn("f", "a int, b int", "integer, integer", 1, 2)
        new = _fn("f", "a int, b int, c int", "integer, integer, integer", 1, 3)
        changes = compute_diff(_schema(new), _schema(old))
        assert changes[0].category == ChangeCategory.ONLINE

    @pytest.mark.parametrize(
        "old,new",
        [
            # An old default dropped: a one-argument call no longer resolves.
            (
                _fn("f", "a int, b int", "integer, integer", 1, 2),
                _fn("f", "a int, b int, c int", "integer, integer, integer", 2, 3),
            ),
            # An old parameter renamed: a call naming it fails.
            (
                _fn("f", "a int", "integer", 1, 1),
                _fn("f", "x int, b int", "integer, integer", 1, 2),
            ),
            # Same count, new types: competes, and resolves to neither.
            (
                _fn("f", "a int", "integer", 1, 1),
                _fn("f", "a text, b int", "text, integer", 1, 2),
            ),
            # A new return type: the old caller reads a different answer.
            (
                _fn("f", "a int", "integer", 1, 1, body="RETURNS int AS '1'"),
                _fn(
                    "f",
                    "a int, b int",
                    "integer, integer",
                    1,
                    2,
                    body="RETURNS text AS '1'",
                ),
            ),
        ],
    )
    def test_other_competing_signatures_stay_offline(self, old, new):
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert changes[0].category == ChangeCategory.OFFLINE

    def test_a_signature_no_old_call_reaches_is_created_beside_the_old(self):
        old = _fn("total", "a int", "integer", 1, 1)
        new = _fn("total", "a int, b int", "integer, integer", 2, 2)
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [CreateFunction, DropFunction]
        assert all(c.category == ChangeCategory.ONLINE for c in changes)
        assert not changes[0].contract
        assert changes[1].contract and is_held_back(changes[1])
        assert changes[1].object_name == "public.total(a int)"

    def test_a_variadic_overload_overlaps_every_longer_call(self):
        old = _fn("total", "a int, b int, c int", "integer, integer, integer", 3, 3)
        new = _fn("total", "VARIADIC a int[]", "integer[]", 1, None)
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert changes[0].category == ChangeCategory.OFFLINE

    def test_renamed_parameters_are_one_offline_statement(self):
        # A call in named notation (`total(a => 1)`) fails once `a` is gone.
        old = _fn("total", "a int", "integer", 1, 1)
        new = _fn("total", "amount int", "integer", 1, 1)
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert changes[0].category == ChangeCategory.OFFLINE
        assert "naming the old parameters" in changes[0].description
        assert not changes[0].contract

    def test_an_unknown_call_shape_counts_as_overlapping(self):
        old = Function("public", "f", "a int", "CREATE f 1", "f1")
        new = Function("public", "f", "a int, b int", "CREATE f 2", "f2")
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert changes[0].category == ChangeCategory.OFFLINE


class TestDroppedRoutines:
    def test_an_undeclared_routine_is_a_contract_drop(self):
        gone = _fn("gone", "a int", "integer", 1, 1)
        changes = compute_diff(Schema(), _schema(gone))
        assert [type(c) for c in changes] == [DropFunction]
        assert changes[0].category == ChangeCategory.ONLINE
        assert is_held_back(changes[0])

    def test_the_revoke_on_a_dropped_routine_waits_with_it(self):
        gone = _fn("gone", "a int", "integer", 1, 1)
        grant = Grant("FUNCTION", "gone", "public", "EXECUTE", "app", args="a int")
        changes = compute_diff(Schema(), _schema(gone, grants=[grant]))
        revokes = [c for c in changes if isinstance(c, RevokePrivilege)]
        assert len(revokes) == 1 and is_held_back(revokes[0])

    def test_a_routine_another_drop_needs_goes_now_and_offline(self):
        # DROP VIEW runs without CASCADE, so it would fail on a function
        # returning its row type that an expand run had held back.
        view = View(schema="public", name="report", definition="SELECT 1 AS x")
        gone = _fn("gone", "", "", 0, 0, body="RETURNS SETOF report")
        kept = _fn("kept", "", "", 0, 0)
        changes = compute_diff(
            _schema(kept), _schema(gone, kept, views={"public.report": view})
        )
        assert any(isinstance(c, DropView) for c in changes)
        drop = next(c for c in changes if isinstance(c, DropFunction))
        assert drop.category == ChangeCategory.OFFLINE
        assert not drop.contract
        assert "report" in drop.description


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


def _run(url: str, sql: str) -> list[tuple]:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall() if cur.description else []
    finally:
        conn.close()


@pytest.fixture
def databases(server_url):
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_ec_target_{suffix}", f"pgdeclare_ec_plan_{suffix}"
    for name in (target, plan):
        _run(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _run(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


QUIET = PgDeclareConfig(report=lambda _line: None)


def _declare(sql_dir: Path, sql: str) -> None:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "schema.sql").write_text(sql)


V1 = """
CREATE FUNCTION account(a uuid, s uuid) RETURNS int
    LANGUAGE sql AS 'SELECT 1';
CREATE FUNCTION total(a int) RETURNS int LANGUAGE sql AS 'SELECT a';
CREATE FUNCTION gone() RETURNS int LANGUAGE sql AS 'SELECT 3';
"""

OLD_CALLS = "SELECT account(gen_random_uuid(), gen_random_uuid()), total(1), gone()"


class TestDeployWhileTheOldVersionServes:
    def test_renamed_parameters_are_refused_online(self, tmp_path, databases):
        target, plan = databases
        sql_dir = tmp_path / "sql"
        _declare(sql_dir, V1)
        apply_declared_schema(target, sql_dir, plan, config=QUIET)

        _declare(
            sql_dir,
            V1.replace(
                "total(a int) RETURNS int LANGUAGE sql AS 'SELECT a'",
                "total(amount int) RETURNS int LANGUAGE sql AS 'SELECT amount'",
            ),
        )
        with pytest.raises(Exception, match="Migration execution failed"):
            apply_declared_schema(target, sql_dir, plan, expand_only=True, config=QUIET)
        # The serving version's named call still resolves.
        assert _run(target, "SELECT total(a => 4)") == [(4,)]

        apply_declared_schema(target, sql_dir, plan, offline=True, config=QUIET)
        assert _run(target, "SELECT total(amount => 4)") == [(4,)]

    def test_a_new_default_parameter_swaps_online(self, tmp_path, databases):
        target, plan = databases
        sql_dir = tmp_path / "sql"
        _declare(sql_dir, V1)
        apply_declared_schema(target, sql_dir, plan, config=QUIET)

        _declare(
            sql_dir,
            V1.replace(
                "account(a uuid, s uuid)",
                "account(a uuid, s uuid, ip text DEFAULT NULL)",
            ),
        )
        # One statement drops account(uuid, uuid) and creates the new form,
        # so no moment has neither (the 2026-11-19 incident, a hand DROP
        # committed before the CREATE) or both ("is not unique").
        apply_declared_schema(target, sql_dir, plan, expand_only=True, config=QUIET)
        # The serving version's calls, by position and by name, still resolve.
        assert _run(target, OLD_CALLS) == [(1, 1, 3)]
        assert _run(
            target, "SELECT account(a => gen_random_uuid(), s => gen_random_uuid())"
        ) == [(1,)]
        assert _run(
            target, "SELECT account(gen_random_uuid(), gen_random_uuid(), 'x')"
        ) == [(1,)]
        apply_declared_schema(
            target, sql_dir, plan, expect_converged=True, config=QUIET
        )

    def test_expand_keeps_old_calls_working_and_contract_drops(
        self, tmp_path, databases
    ):
        target, plan = databases
        sql_dir = tmp_path / "sql"
        _declare(sql_dir, V1)
        apply_declared_schema(target, sql_dir, plan, config=QUIET)

        v2 = V1.replace(
            "total(a int) RETURNS int LANGUAGE sql AS 'SELECT a'",
            "total(a int, b int) RETURNS int LANGUAGE sql AS 'SELECT a + b'",
        ).replace("CREATE FUNCTION gone() RETURNS int LANGUAGE sql AS 'SELECT 3';", "")
        _declare(sql_dir, v2)

        # Expand: the new signature exists, and so does everything the old
        # version calls; the run converges up to the held-back drops.
        apply_declared_schema(target, sql_dir, plan, expand_only=True, config=QUIET)
        assert _run(target, OLD_CALLS) == [(1, 1, 3)]
        assert _run(target, "SELECT total(1, 2)") == [(3,)]

        # Contract, once the old version has stopped.
        apply_declared_schema(target, sql_dir, plan, config=QUIET)
        assert _run(
            target,
            "SELECT proname || '(' || pg_get_function_identity_arguments(oid) || ')' "
            "FROM pg_proc WHERE pronamespace = 'public'::regnamespace ORDER BY 1",
        ) == [("account(a uuid, s uuid)",), ("total(a integer, b integer)",)]
        # And a second run, expand or not, finds nothing.
        apply_declared_schema(
            target, sql_dir, plan, expect_converged=True, config=QUIET
        )


# =============================================================================
# Inline migrations
# =============================================================================


class TestBreakingStatements:
    @pytest.mark.parametrize(
        "sql",
        [
            # The block that broke production on 2026-11-19.
            "DROP FUNCTION IF EXISTS sec_get_account_for_session(UUID, UUID);",
            "DROP TABLE old_things;",
            "DROP VIEW v;",
            "DROP INDEX i CASCADE;",
            "ALTER TABLE t RENAME COLUMN a TO b;",
            "ALTER TABLE t RENAME TO u;",
            "ALTER FUNCTION f(int) RENAME TO g;",
            "ALTER FUNCTION f(int) SET SCHEMA archive;",
            "ALTER TABLE t DROP COLUMN c;",
            "ALTER TABLE t ALTER COLUMN c TYPE bigint;",
            "ALTER TABLE t ALTER COLUMN c SET NOT NULL;",
            "DO $$ BEGIN EXECUTE 'DROP FUNCTION g()'; END $$;",
            "DO $$ BEGIN ALTER TABLE t RENAME COLUMN a TO b; END $$;",
            "THIS IS NOT SQL;",
        ],
    )
    def test_breaks(self, sql):
        assert breaking_statements(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE t SET a = 1 WHERE a IS NULL;",
            "INSERT INTO t (a) SELECT 1 ON CONFLICT DO NOTHING;",
            "DROP INDEX IF EXISTS i;",
            "DROP TRIGGER IF EXISTS tr ON t;",
            "ALTER INDEX i RENAME TO j;",
            "ALTER TABLE t RENAME CONSTRAINT c TO d;",
            "ALTER TABLE t ALTER COLUMN c DROP NOT NULL;",
            "ALTER TABLE t ADD COLUMN c int;",
            "DO $$ BEGIN UPDATE t SET a = 1; END $$;",
            "-- DROP FUNCTION f(); in a comment\nSELECT 1;",
        ],
    )
    def test_does_not_break(self, sql):
        assert breaking_statements(sql) == []

    def test_names_the_statement_in_a_block(self):
        sql = "UPDATE t SET a = 1;\n-- why\nDROP FUNCTION f(uuid);\nSELECT 'é';"
        assert breaking_statements(sql) == ["DROP FUNCTION f(uuid)"]

    def test_only_pending_blocks_count(self):
        def block(id_: str, sql: str) -> MigrationBlock:
            return MigrationBlock(id_, Path("x.sql"), sql, "h")

        blocks = [
            block("old", "DROP FUNCTION f();"),
            block("new", "DROP FUNCTION g();"),
            block("backfill", "UPDATE t SET a = 1;"),
        ]
        found = breaking_blocks(blocks, {"old": ("when", "h")})
        assert [b.block.id for b in found] == ["new"]
        assert found[0].describe() == "@migrate-before:new: DROP FUNCTION g()"


INCIDENT_V1 = """
CREATE TABLE session (id uuid);
CREATE FUNCTION account(a uuid, s uuid) RETURNS int LANGUAGE sql AS 'SELECT 1';
"""

# What #15547 wrote: a block dropping the old signature by hand.
INCIDENT_V2 = """
-- @migrate-before:2026-11-18_account_ip
DROP FUNCTION IF EXISTS account(UUID, UUID);
-- @end-migrate
CREATE TABLE session (id uuid);
CREATE FUNCTION account(a uuid, s uuid, ip text DEFAULT NULL) RETURNS int
    LANGUAGE sql AS 'SELECT 1';
"""


class TestExpandRefusesBreakingBlocks:
    def test_a_pending_block_dropping_a_routine_is_refused(self, tmp_path, databases):
        target, plan = databases
        sql_dir = tmp_path / "sql"
        _declare(sql_dir, INCIDENT_V1)
        apply_declared_schema(target, sql_dir, plan, config=QUIET)

        _declare(sql_dir, INCIDENT_V2)
        with pytest.raises(ExpandRefusedError, match="2026-11-18_account_ip"):
            apply_declared_schema(target, sql_dir, plan, expand_only=True, config=QUIET)
        # Nothing ran: the serving version's call still resolves.
        old_call = "SELECT account(gen_random_uuid(), gen_random_uuid())"
        assert _run(target, old_call) == [(1,)]

        apply_declared_schema(target, sql_dir, plan, offline=True, config=QUIET)
        # Once applied, the block is in the ledger and no longer holds anything.
        apply_declared_schema(
            target, sql_dir, plan, expand_only=True, expect_converged=True, config=QUIET
        )

    def test_a_fresh_database_runs_no_block_and_is_not_refused(
        self, tmp_path, databases
    ):
        target, plan = databases
        sql_dir = tmp_path / "sql"
        _declare(sql_dir, INCIDENT_V2)
        apply_declared_schema(target, sql_dir, plan, expand_only=True, config=QUIET)
