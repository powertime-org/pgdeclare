"""Procedures, aggregates and window functions.

They share pg_proc with functions and ride the same Function model and
changes, told apart by `kind` (pg_proc.prokind). The unit tests drive the
differ and the rendered SQL; the integration tests run the whole pipeline
against a real server and need TEST_DATABASE_URL.
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
    CreateFunction,
    DropAndCreateFunction,
    DropFunction,
    GrantPrivilege,
    ReplaceFunction,
    RevokePrivilege,
    RevokePublicExecute,
)
from pgdeclare.models import Comment, Function, Grant, Schema
from pgdeclare.statement_parser import _parse_statement_metadata


# =============================================================================
# Unit tests
# =============================================================================


def _fn(name: str, args: str = "", kind: str = "f", body: str = "1", **kw) -> Function:
    return Function(
        schema="public",
        name=name,
        args=args,
        full_definition=f"CREATE {name} {kind} {body}",
        definition_hash=f"{name}-{kind}-{body}",
        kind=kind,
        **kw,
    )


def _schema(*functions: Function) -> Schema:
    return Schema(functions={f.signature: f for f in functions})


class TestRoutineSql:
    def test_drop_names_the_kind(self):
        assert DropFunction(name="p", args="IN a integer", kind="p").to_sql() == (
            'DROP PROCEDURE IF EXISTS "public"."p"(IN a integer) CASCADE;'
        )
        assert DropFunction(name="agg", args="integer", kind="a").to_sql() == (
            'DROP AGGREGATE IF EXISTS "public"."agg"(integer) CASCADE;'
        )

    def test_zero_argument_aggregate_is_written_with_a_star(self):
        assert DropFunction(name="cnt", kind="a").to_sql() == (
            'DROP AGGREGATE IF EXISTS "public"."cnt"(*) CASCADE;'
        )
        comment = Comment("AGGREGATE", "public", "cnt", "x")
        assert comment.target_sql == 'AGGREGATE "public"."cnt"(*)'

    def test_recreate_drops_the_old_kind(self):
        change = DropAndCreateFunction(function=_fn("r", kind="p"), old_kind="f")
        sql = change.to_sql()
        assert 'DROP FUNCTION IF EXISTS "public"."r"() CASCADE' in sql
        assert "Drop and recreate procedure" in change.description

    def test_grants_on_procedures_say_procedure(self):
        grant = Grant("PROCEDURE", "p", "public", "EXECUTE", "app")
        assert GrantPrivilege(grant=grant).to_sql() == (
            'GRANT EXECUTE ON PROCEDURE "public"."p" TO app;'
        )
        revoke = RevokePrivilege(grant=grant).to_sql()
        assert 'REVOKE EXECUTE ON PROCEDURE "public"."p" FROM app' in revoke
        assert "wrong_object_type" in revoke

    def test_public_execute_revoke_on_a_procedure(self):
        change = RevokePublicExecute(name="p", args="", keyword="PROCEDURE")
        assert change.to_sql() == (
            'REVOKE EXECUTE ON PROCEDURE "public"."p"() FROM public;'
        )

    def test_comment_keys_keep_the_kind(self):
        proc = Comment("PROCEDURE", "public", "p", "x", args="a integer")
        assert proc.key == "PROCEDURE:public.p(a integer)"
        assert proc.target_sql == 'PROCEDURE "public"."p"(a integer)'
        target = _schema(_fn("p", "a integer", kind="p"))
        assert target.has_comment_target(proc)
        assert not target.has_comment_target(
            Comment("FUNCTION", "public", "p", "x", args="a integer")
        )


class TestRoutineDiff:
    def test_procedure_body_change_is_a_replace(self):
        changes = compute_diff(
            _schema(_fn("p", kind="p", body="2")), _schema(_fn("p", kind="p"))
        )
        assert [type(c) for c in changes] == [ReplaceFunction]
        assert changes[0].description == "Replace procedure public.p()"

    def test_kind_change_is_a_drop_and_recreate(self):
        changes = compute_diff(_schema(_fn("r", kind="p")), _schema(_fn("r")))
        assert [type(c) for c in changes] == [DropAndCreateFunction]
        assert isinstance(changes[0], DropAndCreateFunction)
        assert changes[0].old_kind == "f"

    def test_aggregate_result_change_is_a_drop_and_recreate(self):
        old = _fn("agg", "integer", kind="a", result_type="n:integer")
        new = _fn("agg", "integer", kind="a", body="2", result_type="n:bigint")
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [DropAndCreateFunction]

    def test_aggregate_option_change_is_a_replace(self):
        old = _fn("agg", "integer", kind="a", result_type="n:integer")
        new = _fn("agg", "integer", kind="a", body="2", result_type="n:integer")
        changes = compute_diff(_schema(new), _schema(old))
        assert [type(c) for c in changes] == [ReplaceFunction]

    def test_aggregates_are_built_after_functions(self):
        agg = _fn(
            "agg",
            "integer",
            kind="a",
            support_functions=("public.zz_step(integer, integer)",),
        )
        step = _fn("zz_step", "integer, integer")
        changes = compute_diff(_schema(agg, step), Schema())
        assert [c.object_name for c in changes] == [
            "public.zz_step(integer, integer)",
            "public.agg(integer)",
        ]
        assert all(isinstance(c, CreateFunction) for c in changes)

    def test_aggregate_is_recreated_with_a_dropped_support_function(self):
        uses = ("public.step(a integer, b integer)",)
        agg = _fn("agg", "integer", kind="a", support_functions=uses)
        old_step = _fn("step", "a integer, b integer")
        new_step = _fn("step", "x integer, y integer")
        changes = compute_diff(_schema(agg, new_step), _schema(agg, old_step))
        assert [type(c) for c in changes] == [
            DropAndCreateFunction,
            DropAndCreateFunction,
        ]
        assert [c.object_name for c in changes] == [
            "public.step(x integer, y integer)",
            "public.agg(integer)",
        ]

    def test_undeclared_routines_drop_with_their_kind(self):
        changes = compute_diff(
            Schema(), _schema(_fn("p", kind="p"), _fn("agg", kind="a"))
        )
        assert sorted(c.to_sql() for c in changes) == [
            'DROP AGGREGATE IF EXISTS "public"."agg"(*) CASCADE;',
            'DROP PROCEDURE IF EXISTS "public"."p"() CASCADE;',
        ]


class TestRoutineStatements:
    def test_procedure_signature_keeps_out_parameters(self):
        stmt = _parse_statement_metadata(
            "CREATE PROCEDURE p(a int, OUT b int) LANGUAGE sql AS $$ SELECT 1 $$",
            None,
            1,
        )
        assert stmt.statement_type == "CREATE PROCEDURE"
        assert stmt.duplicate_key == "PROCEDURE:public.p(int4, int4)"

    def test_aggregate_signature(self):
        stmt = _parse_statement_metadata(
            "CREATE OR REPLACE AGGREGATE s.pct(float8 ORDER BY anyelement) "
            "(SFUNC = f, STYPE = internal)",
            None,
            1,
        )
        assert stmt.statement_type == "CREATE AGGREGATE"
        assert stmt.duplicate_key == "AGGREGATE:s.pct(float8, anyelement)"
        assert stmt.is_idempotent


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
    target, plan = f"pgdeclare_rt_target_{suffix}", f"pgdeclare_rt_plan_{suffix}"
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


def _routines(url: str) -> list[tuple]:
    return _rows(
        url,
        "SELECT proname, prokind, pg_get_function_identity_arguments(oid) "
        "FROM pg_proc WHERE pronamespace = 'public'::regnamespace ORDER BY 1, 3",
    )


def _comments(url: str) -> list[tuple]:
    return _rows(
        url,
        "SELECT p.proname, d.description FROM pg_description d "
        "JOIN pg_proc p ON d.classoid = 'pg_proc'::regclass AND d.objoid = p.oid "
        "WHERE p.pronamespace = 'public'::regnamespace ORDER BY 1",
    )


ROUTINES_V1 = '''
CREATE FUNCTION step(a int, b int) RETURNS int
    LANGUAGE sql IMMUTABLE AS 'SELECT a + b';

CREATE AGGREGATE total(int) (SFUNC = step, STYPE = int, INITCOND = '0');

CREATE AGGREGATE tally(*) (SFUNC = int8inc, STYPE = int8, INITCOND = '0');

CREATE AGGREGATE pct(float8 ORDER BY anyelement) (
    SFUNC = ordered_set_transition,
    STYPE = internal,
    FINALFUNC = percentile_disc_final,
    FINALFUNC_EXTRA
);

CREATE AGGREGATE mean(float8) (
    SFUNC = float8_accum, STYPE = float8[], FINALFUNC = float8_avg,
    COMBINEFUNC = float8_combine, INITCOND = '{0,0,0}', PARALLEL = SAFE,
    MSFUNC = float8_accum, MINVFUNC = float8_accum, MSTYPE = float8[],
    MFINALFUNC = float8_avg, MINITCOND = '{0,0,0}'
);

CREATE AGGREGATE biggest(int) (SFUNC = int4larger, STYPE = int, SORTOP = >);

CREATE AGGREGATE "Odd""Agg"(text) (
    SFUNC = textcat, STYPE = text, INITCOND = 'it''s'
);

CREATE TABLE ledger (amount int);

CREATE PROCEDURE book(IN amount int, INOUT booked int)
    LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO ledger VALUES (amount);
    booked := amount;
END
$$;

CREATE PROCEDURE "odd ""proc"""() LANGUAGE sql AS $$ SELECT 1 $$;

COMMENT ON PROCEDURE book(int, int) IS 'books an amount';
COMMENT ON AGGREGATE tally(*) IS 'counts rows';
COMMENT ON AGGREGATE pct(float8 ORDER BY anyelement) IS 'a percentile';
'''


class TestRoutineRoundTrip:
    def test_declared_routines_create_and_converge(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ROUTINES_V1, target_url, plan_db)

        assert _routines(target_url) == [
            ('Odd"Agg', "a", "text"),
            ("biggest", "a", "integer"),
            ("book", "p", "IN amount integer, INOUT booked integer"),
            ("mean", "a", "double precision"),
            ('odd "proc"', "p", ""),
            ("pct", "a", "double precision ORDER BY anyelement"),
            ("step", "f", "a integer, b integer"),
            ("tally", "a", ""),
            ("total", "a", "integer"),
        ]
        assert _comments(target_url) == [
            ("book", "books an amount"),
            ("pct", "a percentile"),
            ("tally", "counts rows"),
        ]
        assert _rows(
            target_url,
            "SELECT total(x), tally(*), biggest(x), mean(x), "
            'pct(0.5) WITHIN GROUP (ORDER BY x), "Odd""Agg"(\'!\') '
            "FROM generate_series(1, 4) x",
        ) == [(10, 4, 4, 2.5, 2, "it's!!!!")]
        _admin(target_url, "CALL book(7, NULL)")
        assert _rows(target_url, "SELECT amount FROM ledger") == [(7,)]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_changes_replace_or_recreate_and_converge(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ROUTINES_V1, target_url, plan_db)
        [(book_oid,)] = _rows(target_url, "SELECT 'book'::regproc::oid")

        # Same signatures: CREATE OR REPLACE, which keeps the oid.
        replaced = ROUTINES_V1.replace(
            "booked := amount;", "booked := amount * 2;"
        ).replace("STYPE = int, INITCOND = '0');", "STYPE = int, INITCOND = '100');")
        assert sorted(
            type(c).__name__ for c in _plan(sql_dir, replaced, target_url, plan_db)
        ) == [
            "ReplaceFunction",
            "ReplaceFunction",
        ]
        _apply(sql_dir, replaced, target_url, plan_db)
        assert _rows(target_url, "SELECT 'book'::regproc::oid") == [(book_oid,)]
        assert _rows(target_url, "SELECT total(x) FROM generate_series(1, 4) x") == [
            (110,)
        ]
        assert _pending(sql_dir, target_url, plan_db) == []

        # A new result type, a new procedure signature, a function that
        # becomes a procedure, and a support function whose parameters are
        # renamed (its DROP ... CASCADE takes `total` with it).
        recreated = (
            replaced.replace(
                "CREATE AGGREGATE tally(*) (SFUNC = int8inc, STYPE = int8, INITCOND = '0');",
                "CREATE AGGREGATE tally(*) (SFUNC = int4inc, STYPE = int4, INITCOND = '0');",
            )
            .replace(
                "IN amount int, INOUT booked int",
                "IN amount bigint, INOUT booked bigint",
            )
            .replace(
                "COMMENT ON PROCEDURE book(int, int)",
                "COMMENT ON PROCEDURE book(bigint, bigint)",
            )
            .replace(
                'CREATE PROCEDURE "odd ""proc"""() LANGUAGE sql AS $$ SELECT 1 $$;',
                'CREATE FUNCTION "odd ""proc"""() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$;',
            )
            .replace("step(a int, b int)", "step(x int, y int)")
            .replace("'SELECT a + b'", "'SELECT x + y'")
        )
        _apply(sql_dir, recreated, target_url, plan_db)
        routines = _routines(target_url)
        assert ("book", "p", "IN amount bigint, INOUT booked bigint") in routines
        assert ('odd "proc"', "f", "") in routines
        assert ("total", "a", "integer") in routines
        assert _rows(
            target_url,
            "SELECT format_type(prorettype, NULL) FROM pg_proc WHERE proname = 'tally'",
        ) == [("integer",)]
        # Comments come back on the recreated objects.
        assert _comments(target_url) == [
            ("book", "books an amount"),
            ("pct", "a percentile"),
            ("tally", "counts rows"),
        ]
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_undeclared_routines_are_dropped(self, tmp_path, databases):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        _apply(sql_dir, ROUTINES_V1, target_url, plan_db)

        _apply(sql_dir, "CREATE TABLE ledger (amount int);", target_url, plan_db)
        assert _routines(target_url) == []
        assert _pending(sql_dir, target_url, plan_db) == []

    def test_window_function_round_trips(self, tmp_path, databases):
        target_url, plan_db = databases
        if not _rows(
            target_url, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
        )[0][0]:
            pytest.skip("LANGUAGE internal needs a superuser")
        sql_dir = tmp_path / "sql"
        window = (
            "CREATE FUNCTION row_no() RETURNS bigint LANGUAGE internal "
            "WINDOW IMMUTABLE AS 'window_row_number';"
        )
        _apply(sql_dir, window, target_url, plan_db)
        assert _routines(target_url) == [("row_no", "w", "")]
        assert _pending(sql_dir, target_url, plan_db) == []

        # A window function cannot be replaced by a plain one.
        plain = "CREATE FUNCTION row_no() RETURNS bigint LANGUAGE sql AS 'SELECT 1::bigint';"
        _apply(sql_dir, plain, target_url, plan_db)
        assert _routines(target_url) == [("row_no", "f", "")]
        assert _pending(sql_dir, target_url, plan_db) == []


def _plan(sql_dir: Path, sql: str, target_url: str, plan_db: str):
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "schema.sql").write_text(sql)
    return _pending(sql_dir, target_url, plan_db)


class TestRoutineGrants:
    @pytest.fixture
    def role(self, server_url):
        name = f"pgdeclare_rt_role_{uuid.uuid4().hex[:8]}"
        _admin(server_url, f'CREATE ROLE "{name}"')
        try:
            yield name
        finally:
            # Requested before `databases`, so its grants are already gone
            # with the dropped databases by now.
            _admin(server_url, f'DROP ROLE IF EXISTS "{name}"')

    def test_grants_on_procedures_and_aggregates_are_managed(
        self, tmp_path, role, databases
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        config = PgDeclareConfig(managed_roles=(role,), report=lambda _line: None)
        base = (
            "CREATE PROCEDURE work() LANGUAGE sql AS $$ SELECT 1 $$;\n"
            "CREATE AGGREGATE tally(*) (SFUNC = int8inc, STYPE = int8, INITCOND = '0');\n"
        )
        granted = (
            base
            + f'GRANT EXECUTE ON PROCEDURE work TO "{role}";\n'
            + f'GRANT EXECUTE ON FUNCTION tally TO "{role}";\n'
        )

        def privileges() -> list[tuple]:
            # The role's own ACL entries; PUBLIC keeps its default EXECUTE.
            return _rows(
                target_url,
                "SELECT p.proname FROM pg_proc p "
                "CROSS JOIN LATERAL aclexplode(p.proacl) a "
                f"WHERE a.grantee = '{role}'::regrole "
                "AND a.privilege_type = 'EXECUTE' "
                "AND p.pronamespace = 'public'::regnamespace ORDER BY 1",
            )

        _apply(sql_dir, granted, target_url, plan_db, config)
        assert privileges() == [("tally",), ("work",)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        # A new result type recreates the aggregate: the grant is replayed.
        changed = granted.replace(
            "CREATE AGGREGATE tally(*) (SFUNC = int8inc, STYPE = int8, INITCOND = '0');",
            "CREATE AGGREGATE tally(*) (SFUNC = int4inc, STYPE = int4, INITCOND = '0');",
        )
        _apply(sql_dir, changed, target_url, plan_db, config)
        assert privileges() == [("tally",), ("work",)]
        assert _pending(sql_dir, target_url, plan_db, config) == []

        _apply(sql_dir, base, target_url, plan_db, config)
        assert privileges() == []
        assert _pending(sql_dir, target_url, plan_db, config) == []

    def test_locked_security_definer_procedure_loses_public_execute(
        self, tmp_path, databases
    ):
        target_url, plan_db = databases
        sql_dir = tmp_path / "sql"
        (sql_dir / "schemas").mkdir(parents=True)
        (sql_dir / "schemas" / "schema.sql").write_text(
            "CREATE PROCEDURE guarded() LANGUAGE sql SECURITY DEFINER "
            "AS $$ SELECT 1 $$;"
        )
        apply_declared_schema(
            target_url, sql_dir, plan_db, config=QUIET, locked_functions={"guarded"}
        )

        assert _rows(
            target_url,
            "SELECT has_function_privilege('public', 'guarded()', 'EXECUTE')",
        ) == [(False,)]
        desired = build_desired_schema(sql_dir, target_url, plan_db, config=QUIET)
        actual = introspect_schema(target_url, config=QUIET)
        assert compute_diff(desired, actual, locked_functions={"guarded"}) == []

        # Drift: PUBLIC EXECUTE granted by hand is revoked again.
        _admin(target_url, "GRANT EXECUTE ON PROCEDURE guarded TO PUBLIC")
        actual = introspect_schema(target_url, config=QUIET)
        drift = compute_diff(desired, actual, locked_functions={"guarded"})
        assert [c.to_sql() for c in drift] == [
            'REVOKE EXECUTE ON PROCEDURE "public"."guarded"() FROM public;'
        ]
