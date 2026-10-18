"""Domain types (CREATE DOMAIN): model, SQL, diff, and round trips.

The unit tests need nothing. The round trips run the whole engine against a
real server (TEST_DATABASE_URL): declare, apply, and check that a second diff
is empty, for every kind of domain change.
"""

import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    AddDomainConstraint,
    AlterDomainDropDefault,
    AlterDomainDropNotNull,
    AlterDomainSetDefault,
    AlterDomainSetNotNull,
    CreateDomain,
    DropAndCreateDomain,
    DropAndCreateDomainConstraint,
    DropColumn,
    DropDomain,
    DropDomainConstraint,
    ValidateDomainConstraint,
)
from pgdeclare.executor import execute_migration
from pgdeclare.models import ChangeCategory, Column, Constraint, Domain, Schema, Table
from pgdeclare.statement_parser import parse_sql_file


def _check(name: str, definition: str) -> Constraint:
    return Constraint(name=name, type="c", definition=definition)


def _domain(**changes: Any) -> Domain:
    return replace(Domain(schema="public", name="pos", base_type="integer"), **changes)


def _schema(*domains: Domain, tables: dict[str, Table] | None = None) -> Schema:
    return Schema(
        domains={d.qualified_name: d for d in domains}, tables=dict(tables or {})
    )


# =============================================================================
# Model and SQL
# =============================================================================


class TestDomainModel:
    def test_equality_ignores_dependents_and_constraint_order(self):
        a = _domain(
            constraints=[_check("a", "CHECK ((VALUE > 0))"), _check("b", "CHECK (x)")],
            dependents=["column v of table t"],
        )
        b = _domain(
            constraints=[_check("b", "CHECK (x)"), _check("a", "CHECK ((VALUE > 0))")]
        )
        assert a == b

    def test_every_property_takes_part_in_equality(self):
        base = _domain()
        assert base != _domain(base_type="bigint")
        assert base != _domain(nullable=False)
        assert base != _domain(default="1")
        assert base != _domain(collation='"pg_catalog"."C"')
        assert base != _domain(constraints=[_check("c", "CHECK ((VALUE > 0))")])


class TestDomainSql:
    def test_create_domain_carries_every_clause(self):
        domain = _domain(
            base_type="text",
            collation='"pg_catalog"."C"',
            default="'x'::text",
            nullable=False,
            constraints=[
                _check("short", "CHECK ((length(VALUE) < 10))"),
                _check("has_x", "CHECK ((VALUE ~ 'x'::text)) NOT VALID"),
            ],
        )
        assert CreateDomain(domain=domain).to_sql() == (
            'CREATE DOMAIN "public"."pos" AS text COLLATE "pg_catalog"."C" '
            "DEFAULT 'x'::text NOT NULL "
            "CONSTRAINT \"has_x\" CHECK ((VALUE ~ 'x'::text)) "
            'CONSTRAINT "short" CHECK ((length(VALUE) < 10));'
        )

    def test_names_are_quoted_with_embedded_quotes_doubled(self):
        evil = 'x"; DROP TABLE t; --'
        domain = _domain(
            schema='s"', name=evil, constraints=[_check(evil, "CHECK (true)")]
        )
        sql = CreateDomain(domain=domain).to_sql()
        assert sql.startswith('CREATE DOMAIN "s"""."x""; DROP TABLE t; --" AS integer')
        assert 'CONSTRAINT "x""; DROP TABLE t; --" CHECK (true)' in sql
        assert DropDomain(schema='s"', name=evil).to_sql() == (
            'DROP DOMAIN IF EXISTS "s"""."x""; DROP TABLE t; --";'
        )

    def test_constraint_changes_add_not_valid(self):
        c = _check("pos_check", "CHECK ((VALUE > 0))")
        add = AddDomainConstraint(schema="public", domain="pos", constraint=c)
        assert add.to_sql() == (
            'ALTER DOMAIN "public"."pos" ADD CONSTRAINT "pos_check" '
            "CHECK ((VALUE > 0)) NOT VALID;"
        )
        recreate = DropAndCreateDomainConstraint(
            schema="public", domain="pos", constraint=c
        ).to_sql()
        assert recreate.startswith("DO $$ BEGIN ALTER DOMAIN")
        assert 'DROP CONSTRAINT IF EXISTS "pos_check"' in recreate
        assert "CHECK ((VALUE > 0)) NOT VALID" in recreate

    def test_categories(self):
        assert AlterDomainSetNotNull(name="pos").category == ChangeCategory.OFFLINE
        for change in (
            CreateDomain(domain=_domain()),
            DropDomain(name="pos"),
            AlterDomainDropNotNull(name="pos"),
            AlterDomainSetDefault(name="pos", default_value="1"),
            AlterDomainDropDefault(name="pos"),
            AddDomainConstraint(domain="pos", constraint=_check("c", "CHECK (true)")),
            DropDomainConstraint(domain="pos", constraint_name="c"),
            ValidateDomainConstraint(domain="pos", constraint_name="c"),
        ):
            assert change.category == ChangeCategory.ONLINE, change


class TestCreateDomainParsing:
    def test_create_domain_is_keyed_in_the_type_namespace(self, tmp_path):
        f = tmp_path / "d.sql"
        f.write_text("CREATE DOMAIN app.pos AS integer CHECK (VALUE > 0);\n")
        (stmt,) = parse_sql_file(f)
        assert stmt.statement_type == "CREATE DOMAIN"
        assert stmt.duplicate_key == "TYPE:app.pos"


# =============================================================================
# Diff
# =============================================================================


class TestDiffDomains:
    def test_missing_domain_is_created(self):
        changes = compute_diff(_schema(_domain()), Schema())
        assert [type(c) for c in changes] == [CreateDomain]

    def test_identical_domains_diff_empty(self):
        d = _domain(default="1", constraints=[_check("c", "CHECK ((VALUE > 0))")])
        assert compute_diff(_schema(d), _schema(replace(d))) == []

    def test_schema_qualified_base_type_is_not_drift(self):
        desired = _domain(base_type="app.mood", schema="app")
        actual = _domain(base_type="mood", schema="app")
        assert compute_diff(_schema(desired), _schema(actual), schemas=["app"]) == []

    def test_default_and_not_null_changes(self):
        actual = _domain(default="1")
        changes = compute_diff(
            _schema(_domain(default="2", nullable=False)), _schema(actual)
        )
        assert [type(c) for c in changes] == [
            AlterDomainSetDefault,
            AlterDomainSetNotNull,
        ]
        changes = compute_diff(
            _schema(_domain()), _schema(_domain(default="1", nullable=False))
        )
        assert [type(c) for c in changes] == [
            AlterDomainDropDefault,
            AlterDomainDropNotNull,
        ]

    def test_check_constraints_are_added_dropped_and_replaced_then_validated(self):
        actual = _domain(
            constraints=[
                _check("gone", "CHECK ((VALUE < 100))"),
                _check("tight", "CHECK ((VALUE > 0))"),
            ]
        )
        desired = _domain(
            constraints=[
                _check("new", "CHECK ((VALUE <> 7))"),
                _check("tight", "CHECK ((VALUE > 1))"),
            ]
        )
        changes = compute_diff(_schema(desired), _schema(actual))
        assert [(type(c), c.object_name) for c in changes] == [
            (DropDomainConstraint, "public.pos.gone"),
            (AddDomainConstraint, "public.pos.new"),
            (DropAndCreateDomainConstraint, "public.pos.tight"),
            (ValidateDomainConstraint, "public.pos.new"),
            (ValidateDomainConstraint, "public.pos.tight"),
        ]

    def test_base_type_change_without_dependents_recreates(self):
        changes = compute_diff(_schema(_domain(base_type="bigint")), _schema(_domain()))
        assert [type(c) for c in changes] == [DropAndCreateDomain]
        assert changes[0].to_sql() == (
            'DROP DOMAIN "public"."pos";\nCREATE DOMAIN "public"."pos" AS bigint;'
        )

    def test_base_type_change_with_dependents_is_refused(self):
        actual = _domain(dependents=["column v of table t"])
        with pytest.raises(ValueError, match="column v of table t"):
            compute_diff(_schema(_domain(base_type="bigint")), _schema(actual))

    def test_collation_change_is_a_recreate(self):
        changes = compute_diff(
            _schema(_domain(base_type="text", collation='"pg_catalog"."C"')),
            _schema(_domain(base_type="text")),
        )
        assert [type(c) for c in changes] == [DropAndCreateDomain]

    def test_undeclared_domain_is_dropped_after_the_column_using_it(self):
        table = Table(
            schema="public",
            name="t",
            columns={
                "id": Column(name="id", type="integer", nullable=False),
                "v": Column(name="v", type="pos", nullable=True),
            },
        )
        desired_table = Table(
            schema="public", name="t", columns={"id": table.columns["id"]}
        )
        changes = compute_diff(
            _schema(tables={"public.t": desired_table}),
            _schema(
                _domain(dependents=["column v of table t"]),
                tables={"public.t": table},
            ),
        )
        kinds = [type(c) for c in changes]
        assert kinds == [DropColumn, DropDomain]


# =============================================================================
# Round trips against a real server
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
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()


@pytest.fixture
def databases(server_url):
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_dom_target_{suffix}", f"pgdeclare_dom_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


class _Project:
    """A one-file schema tree, applied and re-diffed against a target."""

    def __init__(self, sql_dir: Path, target_url: str, plan_db: str):
        self.sql_dir = sql_dir
        self.target_url = target_url
        self.plan_db = plan_db
        self.config = PgDeclareConfig(report=lambda _line: None)

    def write(self, content: str) -> None:
        schemas = self.sql_dir / "schemas"
        schemas.mkdir(parents=True, exist_ok=True)
        (schemas / "schema.sql").write_text(content)

    def apply(self, content: str, **kwargs) -> None:
        self.write(content)
        apply_declared_schema(
            self.target_url, self.sql_dir, self.plan_db, config=self.config, **kwargs
        )

    def pending(self) -> list:
        desired = build_desired_schema(
            self.sql_dir, self.target_url, self.plan_db, config=self.config
        )
        actual = introspect_schema(self.target_url, config=self.config)
        return compute_diff(desired, actual, schemas=list(self.config.schemas))

    def apply_converged(self, content: str, **kwargs) -> None:
        self.apply(content, **kwargs)
        assert self.pending() == []


@pytest.fixture
def project(tmp_path, databases) -> _Project:
    target_url, plan_db = databases
    return _Project(tmp_path / "sql", target_url, plan_db)


BASE = """
CREATE TYPE mood AS ENUM ('calm', 'loud');
CREATE DOMAIN posint AS integer NOT NULL DEFAULT 1 CHECK (VALUE > 0);
CREATE DOMAIN email AS text COLLATE "C"
    CONSTRAINT email_has_at CHECK (VALUE ~ '@');
CREATE DOMAIN "Odd Name" AS mood DEFAULT 'calm';
CREATE TABLE IF NOT EXISTS note (
    id     integer PRIMARY KEY,
    n      posint,
    emails email[],
    m      "Odd Name"
);
"""


def test_domains_round_trip(project):
    """Declare, apply, re-diff empty; columns introspect with the domain type."""
    project.apply_converged(BASE)

    note = introspect_schema(project.target_url).tables["public.note"]
    assert note.columns["n"].type == "posint"
    assert note.columns["emails"].type == "email[]"
    assert note.columns["m"].type == '"Odd Name"'

    domains = introspect_schema(project.target_url).domains
    assert set(domains) == {"public.posint", "public.email", "public.Odd Name"}
    posint = domains["public.posint"]
    assert (posint.base_type, posint.nullable, posint.default) == (
        "integer",
        False,
        "1",
    )
    assert [c.definition for c in posint.constraints] == ["CHECK ((VALUE > 0))"]
    assert domains["public.email"].collation == 'pg_catalog."C"'
    assert "column n of table note" in posint.dependents

    # The domain is enforced on the live table.
    with pytest.raises(psycopg2.errors.CheckViolation):
        _admin(project.target_url, "INSERT INTO note (id, n) VALUES (1, 0)")


def test_default_not_null_and_checks_converge(project):
    project.apply_converged(BASE)
    _admin(project.target_url, "INSERT INTO note (id, n) VALUES (1, 5), (2, 50)")

    # Default changed, NOT NULL dropped, CHECK tightened, a CHECK added.
    v2 = BASE.replace(
        "CREATE DOMAIN posint AS integer NOT NULL DEFAULT 1 CHECK (VALUE > 0);",
        "CREATE DOMAIN posint AS integer DEFAULT 2 CHECK (VALUE > 1)\n"
        "    CONSTRAINT posint_small CHECK (VALUE < 100);",
    )
    project.write(v2)
    pending = {type(c) for c in project.pending()}
    assert pending == {
        AlterDomainSetDefault,
        AlterDomainDropNotNull,
        DropAndCreateDomainConstraint,
        AddDomainConstraint,
        ValidateDomainConstraint,
    }
    project.apply_converged(v2)
    posint = introspect_schema(project.target_url).domains["public.posint"]
    assert posint.default == "2"
    assert posint.nullable is True
    assert sorted(c.definition for c in posint.constraints) == [
        "CHECK ((VALUE < 100))",
        "CHECK ((VALUE > 1))",
    ]

    # Default and CHECKs dropped.
    v3 = v2.replace(
        "CREATE DOMAIN posint AS integer DEFAULT 2 CHECK (VALUE > 1)\n"
        "    CONSTRAINT posint_small CHECK (VALUE < 100);",
        "CREATE DOMAIN posint AS integer;",
    )
    project.apply_converged(v3)
    posint = introspect_schema(project.target_url).domains["public.posint"]
    assert (posint.default, posint.constraints) == (None, [])

    # SET NOT NULL validates every column using the domain: offline only.
    v4 = v3.replace(
        "CREATE DOMAIN posint AS integer;", "CREATE DOMAIN posint AS integer NOT NULL;"
    )
    with pytest.raises(Exception):
        project.apply(v4)
    assert introspect_schema(project.target_url).domains["public.posint"].nullable
    project.apply_converged(v4, offline=True)
    assert (
        introspect_schema(project.target_url).domains["public.posint"].nullable is False
    )


def test_a_check_existing_rows_violate_fails_the_run(project):
    project.apply_converged(BASE)
    _admin(project.target_url, "INSERT INTO note (id, n) VALUES (1, 5)")
    with pytest.raises(Exception):
        project.apply(BASE.replace("CHECK (VALUE > 0)", "CHECK (VALUE > 10)"))
    # Nothing was half-applied: the old constraint is still there and valid.
    posint = introspect_schema(project.target_url).domains["public.posint"]
    assert [c.definition for c in posint.constraints] == ["CHECK ((VALUE > 0))"]


def test_base_type_change(project):
    unused = BASE + "CREATE DOMAIN code AS varchar(4);\n"
    project.apply_converged(unused)

    # Nothing uses `code`: drop and recreate.
    project.apply_converged(unused.replace("varchar(4)", "varchar(8)"))
    code = introspect_schema(project.target_url).domains["public.code"]
    assert code.base_type == "character varying(8)"

    # `posint` is used by note.n: refused, naming the column.
    with pytest.raises(ValueError, match="column n of table note"):
        project.apply(
            unused.replace("varchar(4)", "varchar(8)").replace(
                "posint AS integer", "posint AS bigint"
            )
        )


def test_undeclared_domain_is_dropped_with_the_column_using_it(project):
    project.apply_converged(BASE)

    without = BASE.replace(
        "CREATE DOMAIN posint AS integer NOT NULL DEFAULT 1 CHECK (VALUE > 0);\n", ""
    ).replace("    n      posint,\n", "")
    project.apply_converged(without)

    assert "public.posint" not in introspect_schema(project.target_url).domains
    assert _rows(project.target_url, "SELECT to_regtype('public.posint')::text") == [
        (None,)
    ]


def test_column_retyped_off_a_dropped_domain(project):
    project.apply_converged(BASE)
    _admin(project.target_url, "INSERT INTO note (id, n) VALUES (1, 5)")

    retyped = BASE.replace(
        "CREATE DOMAIN posint AS integer NOT NULL DEFAULT 1 CHECK (VALUE > 0);\n", ""
    ).replace("n      posint,", "n      integer,")
    project.apply_converged(retyped)
    assert _rows(project.target_url, "SELECT n FROM note") == [(5,)]
    assert "public.posint" not in introspect_schema(project.target_url).domains


def test_drop_waits_for_its_dependent_change(server_url, databases):
    """The executor retries a DropDomain that still has a dependent, so the
    order of the changes in the plan does not matter."""
    target_url, _plan = databases
    _admin(target_url, "CREATE DOMAIN d AS int; CREATE TABLE t (id int, v d)")
    changes = [
        DropDomain(schema="public", name="d"),
        DropColumn(schema="public", table="t", column_name="v"),
    ]
    success, _ = execute_migration(target_url, changes)
    assert success is True
    assert _rows(target_url, "SELECT to_regtype('public.d')::text") == [(None,)]
