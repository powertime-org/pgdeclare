"""Range types (CREATE TYPE ... AS RANGE) and collations (CREATE COLLATION).

The unit tests need nothing. The round trips run the whole engine against a
real server (TEST_DATABASE_URL): declare, apply, and check that a second diff
is empty. ICU collations are tested only where the server was built with ICU;
libc collations work everywhere.
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
    AlterColumnType,
    CreateCollation,
    CreateDomain,
    CreateFunction,
    CreateRangeType,
    CreateShellType,
    CreateTable,
    DropAndCreateCollation,
    DropAndCreateRangeType,
    DropCollation,
    DropColumn,
    DropFunction,
    DropRangeType,
    SetComment,
)
from pgdeclare.executor import execute_migration
from pgdeclare.models import (
    Collation,
    Column,
    Comment,
    Domain,
    Function,
    RangeType,
    Schema,
    Table,
)
from pgdeclare.statement_parser import parse_sql_file

EVIL = 'x"; DROP TABLE t; --'


def _range(**changes: Any) -> RangeType:
    return replace(
        RangeType(
            schema="public",
            name="floatrange",
            subtype="double precision",
            multirange_type_name="public.floatmultirange",
        ),
        **changes,
    )


def _collation(**changes: Any) -> Collation:
    return replace(
        Collation(schema="public", name="c1", lc_collate="C", lc_ctype="C"), **changes
    )


def _schema(*objects: Any, **fields: Any) -> Schema:
    schema = Schema(**fields)
    for obj in objects:
        if isinstance(obj, RangeType):
            schema.ranges[obj.qualified_name] = obj
        elif isinstance(obj, Collation):
            schema.collations[obj.qualified_name] = obj
        elif isinstance(obj, Domain):
            schema.domains[obj.qualified_name] = obj
        elif isinstance(obj, Table):
            schema.tables[obj.qualified_name] = obj
        elif isinstance(obj, Function):
            schema.functions[obj.signature] = obj
    return schema


# =============================================================================
# Model and SQL
# =============================================================================


class TestModels:
    def test_range_equality_ignores_dependents_and_canonical_signature(self):
        a = _range(dependents=["column r of table t"], canonical_signature="x")
        assert a == _range()

    def test_every_range_option_takes_part_in_equality(self):
        base = _range()
        assert base != _range(subtype="real")
        assert base != _range(subtype_opclass="public.my_ops")
        assert base != _range(collation='pg_catalog."C"')
        assert base != _range(canonical="public.canon")
        assert base != _range(subtype_diff="public.diff")
        assert base != _range(multirange_type_name="public.other")

    def test_every_collation_option_takes_part_in_equality(self):
        base = _collation()
        assert base == _collation(dependents=["column a of table t"])
        assert base != _collation(lc_collate="C.UTF-8")
        assert base != _collation(lc_ctype="C.UTF-8")
        assert base != _collation(provider="icu", locale="de-DE")
        assert base != _collation(deterministic=False)
        assert base != _collation(rules="&a < b")

    def test_column_collation_takes_part_in_equality(self):
        col = Column(name="a", type="text", nullable=True)
        assert col != replace(col, collation='pg_catalog."C"')


class TestRangeSql:
    def test_create_range_carries_every_option(self):
        r = _range(
            subtype="text",
            subtype_opclass="public.my_ops",
            collation='pg_catalog."C"',
            canonical="public.canon",
            subtype_diff="public.diff",
            multirange_type_name='public."Multi"',
        )
        assert CreateRangeType(range_type=r).to_sql() == (
            'CREATE TYPE "public"."floatrange" AS RANGE (SUBTYPE = text, '
            "SUBTYPE_OPCLASS = public.my_ops, "
            'COLLATION = pg_catalog."C", CANONICAL = public.canon, '
            'SUBTYPE_DIFF = public.diff, MULTIRANGE_TYPE_NAME = public."Multi");'
        )

    def test_names_are_quoted_with_embedded_quotes_doubled(self):
        r = _range(schema='s"', name=EVIL)
        assert (
            CreateRangeType(range_type=r)
            .to_sql()
            .startswith('CREATE TYPE "s"""."x""; DROP TABLE t; --" AS RANGE')
        )
        assert DropRangeType(schema='s"', name=EVIL).to_sql() == (
            'DROP TYPE IF EXISTS "s"""."x""; DROP TABLE t; --";'
        )
        assert CreateShellType(schema='s"', name=EVIL).to_sql() == (
            'CREATE TYPE "s"""."x""; DROP TABLE t; --";'
        )
        assert (
            DropAndCreateRangeType(range_type=r)
            .to_sql()
            .startswith('DROP TYPE "s"""."x""; DROP TABLE t; --";\nCREATE TYPE')
        )

    def test_drop_with_canonical_checks_dependents_then_cascades(self):
        sql = DropRangeType(schema="public", name="it's", has_canonical=True).to_sql()
        assert sql.startswith("DO $$ ")
        assert "to_regtype('\"public\".\"it''s\"')" in sql
        assert "dependent_objects_still_exist" in sql
        assert sql.endswith('DROP TYPE "public"."it\'s" CASCADE; END $$;')


class TestCollationSql:
    def test_libc(self):
        c = _collation(lc_collate="C", lc_ctype="C.UTF-8")
        assert CreateCollation(collation=c).to_sql() == (
            'CREATE COLLATION "public"."c1" '
            "(provider = libc, lc_collate = 'C', lc_ctype = 'C.UTF-8');"
        )

    def test_icu_nondeterministic_with_rules(self):
        c = _collation(
            provider="icu",
            locale="de-DE",
            lc_collate=None,
            lc_ctype=None,
            deterministic=False,
            rules="&a < b",
        )
        assert CreateCollation(collation=c).to_sql() == (
            'CREATE COLLATION "public"."c1" (provider = icu, '
            "locale = 'de-DE', rules = '&a < b', deterministic = false);"
        )

    def test_names_and_literals_are_quoted(self):
        c = _collation(schema='s"', name=EVIL, lc_collate="it's", lc_ctype="C")
        sql = CreateCollation(collation=c).to_sql()
        assert sql.startswith('CREATE COLLATION "s"""."x""; DROP TABLE t; --" (')
        assert "lc_collate = 'it''s'" in sql
        assert DropCollation(schema='s"', name=EVIL).to_sql() == (
            'DROP COLLATION IF EXISTS "s"""."x""; DROP TABLE t; --";'
        )
        assert (
            DropAndCreateCollation(collation=c)
            .to_sql()
            .startswith(
                'DROP COLLATION "s"""."x""; DROP TABLE t; --";\nCREATE COLLATION'
            )
        )

    def test_column_collation_in_create_add_and_alter(self):
        col = Column(name="a", type="text", nullable=True, collation="public.c1")
        table = Table(schema="public", name="t", columns={"a": col})
        assert '"a" text COLLATE public.c1' in CreateTable(table=table).to_sql()
        alter = AlterColumnType(
            schema="public",
            table="t",
            column="a",
            old_type="text",
            new_type="text",
            collation="public.c1",
        )
        assert alter.to_sql() == (
            'ALTER TABLE "public"."t" ALTER COLUMN "a" TYPE text '
            'COLLATE public.c1 USING "a"::text;'
        )
        assert "collation" in alter.description


class TestParsing:
    def test_range_is_keyed_in_the_type_namespace_and_a_shell_is_not(self, tmp_path):
        f = tmp_path / "r.sql"
        f.write_text(
            "CREATE TYPE app.r;\nCREATE TYPE app.r AS RANGE (subtype = int4);\n"
        )
        shell, rng = parse_sql_file(f)
        assert shell.duplicate_key is None
        assert rng.duplicate_key == "TYPE:app.r"

    def test_collation_is_keyed(self, tmp_path):
        f = tmp_path / "c.sql"
        f.write_text(
            "CREATE COLLATION \"De\" (provider = libc, locale = 'C');\n"
            'CREATE COLLATION IF NOT EXISTS app.c FROM "C";\n'
        )
        first, second = parse_sql_file(f)
        assert first.statement_type == "CREATE COLLATION"
        assert first.duplicate_key == "COLLATION:public.De"
        assert second.duplicate_key == "COLLATION:app.c"
        assert second.is_idempotent


# =============================================================================
# Diff
# =============================================================================


class TestDiffRanges:
    def test_missing_range_is_created(self):
        changes = compute_diff(_schema(_range()), Schema())
        assert [type(c) for c in changes] == [CreateRangeType]

    def test_canonical_range_gets_a_shell_type_first(self):
        changes = compute_diff(_schema(_range(canonical="public.canon")), Schema())
        assert [type(c) for c in changes] == [CreateShellType, CreateRangeType]

    def test_identical_and_schema_qualified_subtype_diff_empty(self):
        assert compute_diff(_schema(_range()), _schema(_range())) == []
        desired = _range(schema="app", subtype="app.mood")
        actual = _range(schema="app", subtype="mood")
        assert compute_diff(_schema(desired), _schema(actual), schemas=["app"]) == []

    def test_change_without_dependents_recreates(self):
        changes = compute_diff(
            _schema(_range(subtype_diff="public.d")), _schema(_range())
        )
        assert [type(c) for c in changes] == [DropAndCreateRangeType]

    def test_change_with_dependents_is_refused(self):
        actual = _range(dependents=["column r of table t"])
        with pytest.raises(ValueError, match="column r of table t"):
            compute_diff(_schema(_range(subtype="real")), _schema(actual))

    def test_recreate_to_a_canonical_range_is_refused(self):
        with pytest.raises(ValueError, match="canonical"):
            compute_diff(_schema(_range(canonical="public.c")), _schema(_range()))

    def test_undeclared_range_is_dropped_after_tables_and_functions(self):
        table = Table(
            schema="public",
            name="t",
            columns={"r": Column(name="r", type="floatrange", nullable=True)},
        )
        actual = _schema(_range(), replace(table))
        table.columns = {}
        changes = compute_diff(_schema(table), actual)
        assert [type(c) for c in changes] == [DropColumn, DropRangeType]

    def test_canonical_function_drop_is_left_to_the_range_drop(self):
        canon = Function(
            schema="public", name="canon", args="floatrange", full_definition=""
        )
        other = Function(schema="public", name="other", args="", full_definition="")
        actual = _schema(
            _range(
                canonical="public.canon",
                canonical_signature="public.canon(floatrange)",
            ),
            canon,
            other,
        )
        changes = compute_diff(Schema(), actual)
        assert [type(c) for c in changes] == [DropFunction, DropRangeType]
        drop_function, drop_range = changes
        assert isinstance(drop_function, DropFunction)
        assert drop_function.name == "other"
        assert isinstance(drop_range, DropRangeType) and drop_range.has_canonical


class TestDiffCollations:
    def test_created_before_domains_and_dropped_last(self):
        domain = Domain(
            schema="public", name="d", base_type="text", collation="public.c1"
        )
        changes = compute_diff(_schema(_collation(), domain), Schema())
        assert [type(c) for c in changes] == [CreateCollation, CreateDomain]

        changes = compute_diff(
            Schema(), _schema(_collation(), _range(collation="public.c1"))
        )
        assert [type(c) for c in changes] == [DropRangeType, DropCollation]

    def test_change_without_dependents_recreates(self):
        changes = compute_diff(
            _schema(_collation(lc_ctype="C.UTF-8")), _schema(_collation())
        )
        assert [type(c) for c in changes] == [DropAndCreateCollation]

    def test_change_with_dependents_is_refused(self):
        actual = _collation(dependents=["column a of table t"])
        with pytest.raises(ValueError, match="column a of table t"):
            compute_diff(_schema(_collation(lc_ctype="C.UTF-8")), _schema(actual))

    def test_column_collation_changes(self):
        def table(collation: str | None) -> Table:
            col = Column(name="a", type="text", nullable=True, collation=collation)
            return Table(schema="public", name="t", columns={"a": col})

        (set_it,) = compute_diff(_schema(table("public.c1")), _schema(table(None)))
        assert isinstance(set_it, AlterColumnType)
        assert set_it.collation == "public.c1"
        (unset,) = compute_diff(_schema(table(None)), _schema(table("public.c1")))
        assert isinstance(unset, AlterColumnType)
        assert unset.collation == 'pg_catalog."default"'
        assert compute_diff(_schema(table("x")), _schema(table("x"))) == []

    def test_comment_is_rewritten_after_a_recreate(self):
        comment = Comment(object_type="COLLATION", schema="public", name="c1", text="t")
        desired = _schema(_collation(lc_ctype="C.UTF-8"))
        desired.comments[comment.key] = comment
        actual = _schema(_collation())
        actual.comments[comment.key] = replace(comment)
        changes = compute_diff(desired, actual)
        assert [type(c) for c in changes] == [DropAndCreateCollation, SetComment]
        assert changes[1].to_sql() == 'COMMENT ON COLLATION "public"."c1" IS \'t\';'


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
    target, plan = f"pgdeclare_rng_target_{suffix}", f"pgdeclare_rng_plan_{suffix}"
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

    def live(self) -> Schema:
        return introspect_schema(self.target_url)


@pytest.fixture
def project(tmp_path, databases) -> _Project:
    target_url, plan_db = databases
    return _Project(tmp_path / "sql", target_url, plan_db)


def _require_icu(url: str) -> None:
    if not _rows(url, "SELECT 1 FROM pg_collation WHERE collprovider = 'i' LIMIT 1"):
        pytest.skip("server built without ICU")


def _require_superuser(url: str) -> None:
    if not _rows(
        url, "SELECT 1 FROM pg_roles WHERE rolname = current_user AND rolsuper"
    ):
        pytest.skip("a LANGUAGE internal canonical function needs a superuser")


RANGES = """
CREATE COLLATION "Bytes" (provider = libc, locale = 'C');
CREATE FUNCTION float_diff(double precision, double precision)
    RETURNS double precision LANGUAGE sql IMMUTABLE AS 'SELECT $1 - $2';
CREATE TYPE floatrange AS RANGE (subtype = float8, subtype_diff = float_diff);
CREATE TYPE "Text Range" AS RANGE (
    subtype = text, collation = "Bytes", multirange_type_name = "Text Multi"
);
CREATE TYPE intr AS RANGE (subtype = int4, subtype_opclass = int4_ops);
CREATE TYPE spare AS RANGE (subtype = int8);
CREATE TABLE IF NOT EXISTS booking (
    id      integer PRIMARY KEY,
    during  floatrange,
    spans   floatmultirange,
    words   "Text Range"[],
    hits    intr,
    EXCLUDE USING gist (during WITH &&)
);
COMMENT ON TYPE floatrange IS 'a float range';
"""


def test_ranges_round_trip(project):
    """Declare, apply, re-diff empty; columns introspect with the range types."""
    project.apply_converged(RANGES)

    live = project.live()
    booking = live.tables["public.booking"]
    assert booking.columns["during"].type == "floatrange"
    assert booking.columns["spans"].type == "floatmultirange"
    assert booking.columns["words"].type == '"Text Range"[]'
    assert booking.columns["hits"].type == "intr"

    assert set(live.ranges) == {
        "public.floatrange",
        "public.Text Range",
        "public.intr",
        "public.spare",
    }
    floatrange = live.ranges["public.floatrange"]
    assert floatrange.subtype == "double precision"
    assert floatrange.subtype_diff == "public.float_diff"
    assert floatrange.multirange_type_name == "public.floatmultirange"
    assert "column during of table booking" in floatrange.dependents
    assert "column spans of table booking" in floatrange.dependents
    text_range = live.ranges["public.Text Range"]
    assert text_range.collation == 'public."Bytes"'
    assert text_range.multirange_type_name == 'public."Text Multi"'
    # The default operator class, spelled out, is not drift.
    assert live.ranges["public.intr"].subtype_opclass is None
    # The constructors belong to the type, not the schema files.
    assert not any(f.startswith("public.floatrange(") for f in live.functions)
    assert live.comments["TYPE:public.floatrange"].text == "a float range"

    _admin(
        project.target_url,
        "INSERT INTO booking (id, during, spans) "
        "VALUES (1, floatrange(1, 2), floatmultirange(floatrange(1, 2)))",
    )
    with pytest.raises(psycopg2.errors.ExclusionViolation):
        _admin(
            project.target_url,
            "INSERT INTO booking (id, during) VALUES (2, floatrange(1.5, 3))",
        )


def test_range_change_recreates_only_while_unused(project):
    project.apply_converged(RANGES)

    # Nothing uses `spare`: dropped and recreated, its comment written again.
    v2 = RANGES.replace("subtype = int8", "subtype = int4") + (
        "COMMENT ON TYPE spare IS 'spare';\n"
    )
    project.write(v2)
    assert {type(c) for c in project.pending()} == {DropAndCreateRangeType, SetComment}
    project.apply_converged(v2)
    assert project.live().ranges["public.spare"].subtype == "integer"

    # `floatrange` is used by booking: refused, naming the column.
    with pytest.raises(ValueError, match="column during of table booking"):
        project.apply(v2.replace(", subtype_diff = float_diff", ""))


def test_undeclared_range_is_dropped_with_the_columns_using_it(project):
    project.apply_converged(RANGES)
    without = (
        RANGES.replace(
            "CREATE TYPE floatrange AS RANGE (subtype = float8, subtype_diff = float_diff);\n",
            "",
        )
        .replace("    during  floatrange,\n", "")
        .replace("    spans   floatmultirange,\n", "")
        .replace(",\n    EXCLUDE USING gist (during WITH &&)", "")
        .replace("COMMENT ON TYPE floatrange IS 'a float range';\n", "")
    )
    project.apply_converged(without)
    assert "public.floatrange" not in project.live().ranges
    assert _rows(project.target_url, "SELECT to_regtype('floatmultirange')::text") == [
        (None,)
    ]


def test_range_drop_waits_for_its_dependent_change(server_url, databases):
    target_url, _plan = databases
    _admin(
        target_url,
        "CREATE TYPE r AS RANGE (subtype = int4); CREATE TABLE t (id int, v r_multirange)",
    )
    changes = [
        DropRangeType(schema="public", name="r"),
        DropColumn(schema="public", table="t", column_name="v"),
    ]
    success, _ = execute_migration(target_url, changes)
    assert success is True
    assert _rows(target_url, "SELECT to_regtype('public.r')::text") == [(None,)]


CANONICAL = """
CREATE TYPE dayr;
CREATE FUNCTION dayr_canonical(dayr) RETURNS dayr
    AS 'int4range_canonical' LANGUAGE internal IMMUTABLE STRICT;
CREATE TYPE dayr AS RANGE (subtype = int4, canonical = dayr_canonical);
CREATE TABLE IF NOT EXISTS shift (id integer PRIMARY KEY, days dayr);
"""


def test_canonical_range_lifecycle(project):
    _require_superuser(project.target_url)
    project.write(CANONICAL)
    pending = [type(c) for c in project.pending()]
    assert pending.index(CreateShellType) < pending.index(CreateFunction)
    project.apply_converged(CANONICAL)

    dayr = project.live().ranges["public.dayr"]
    assert dayr.canonical == "public.dayr_canonical"
    _admin(project.target_url, "INSERT INTO shift VALUES (1, dayr(1, 5, '[]'))")
    assert _rows(project.target_url, "SELECT days::text FROM shift") == [("[1,6)",)]

    # In use by shift.days and by its own canonical function: refused.
    with pytest.raises(ValueError, match="column days of table shift"):
        project.apply(
            CANONICAL.replace(
                "canonical = dayr_canonical",
                "canonical = dayr_canonical, subtype_diff = int_diff",
            )
            + "CREATE FUNCTION int_diff(integer, integer) RETURNS double precision\n"
            "    LANGUAGE sql IMMUTABLE AS 'SELECT ($1 - $2)::float8';\n"
        )

    # Undeclared together with its column: the range takes its function along.
    _admin(project.target_url, "DELETE FROM shift")
    project.apply_converged(
        "CREATE TABLE IF NOT EXISTS shift (id integer PRIMARY KEY);\n"
    )
    live = project.live()
    assert live.ranges == {}
    assert not any(f.startswith("public.dayr_canonical") for f in live.functions)


COLLATIONS = """
CREATE COLLATION bytes (provider = libc, lc_collate = 'C', lc_ctype = 'C');
CREATE COLLATION spare (provider = libc, locale = 'C');
CREATE DOMAIN code AS text COLLATE public.bytes;
CREATE TABLE IF NOT EXISTS item (
    id     integer PRIMARY KEY,
    name   text COLLATE public.bytes,
    alias  text COLLATE bytes,
    plain  text COLLATE "C",
    code   code
);
COMMENT ON COLLATION bytes IS 'byte order';
"""


def test_collations_round_trip(project):
    project.apply_converged(COLLATIONS)

    live = project.live()
    bytes_ = live.collations["public.bytes"]
    assert (bytes_.provider, bytes_.lc_collate, bytes_.lc_ctype) == ("libc", "C", "C")
    assert bytes_.deterministic is True
    assert "column name of table item" in bytes_.dependents
    assert "type code" in bytes_.dependents
    item = live.tables["public.item"]
    assert item.columns["name"].collation == "public.bytes"
    assert item.columns["alias"].collation == "public.bytes"
    assert item.columns["plain"].collation == 'pg_catalog."C"'
    # A column typed with a domain inherits its collation: nothing to declare.
    assert item.columns["code"].collation is None
    assert live.domains["public.code"].collation == "public.bytes"
    assert live.comments["COLLATION:public.bytes"].text == "byte order"

    # The comment changes, then goes.
    project.apply_converged(COLLATIONS.replace("'byte order'", "'bytes'"))
    assert project.live().comments["COLLATION:public.bytes"].text == "bytes"
    project.apply_converged(
        COLLATIONS.replace("COMMENT ON COLLATION bytes IS 'byte order';\n", "")
    )
    assert "COLLATION:public.bytes" not in project.live().comments


def test_collation_change_recreates_only_while_unused(project):
    project.apply_converged(COLLATIONS)

    v2 = COLLATIONS.replace("locale = 'C'", "locale = 'POSIX'")
    project.write(v2)
    assert [type(c) for c in project.pending()] == [DropAndCreateCollation]
    project.apply_converged(v2)
    assert project.live().collations["public.spare"].lc_collate == "POSIX"

    with pytest.raises(ValueError, match="column name of table item"):
        project.apply(v2.replace("lc_ctype = 'C'", "lc_ctype = 'POSIX'"))


def test_column_collation_converges(project):
    project.apply_converged(COLLATIONS)
    _admin(project.target_url, "INSERT INTO item (id, plain) VALUES (1, 'b'), (2, 'a')")

    moved = COLLATIONS.replace('plain  text COLLATE "C"', "plain  text COLLATE spare")
    project.write(moved)
    assert [type(c) for c in project.pending()] == [AlterColumnType]
    project.apply_converged(moved)
    assert project.live().tables["public.item"].columns["plain"].collation == (
        "public.spare"
    )

    project.apply_converged(moved.replace("plain  text COLLATE spare", "plain  text"))
    assert project.live().tables["public.item"].columns["plain"].collation is None
    assert _rows(project.target_url, "SELECT plain FROM item ORDER BY id") == [
        ("b",),
        ("a",),
    ]


def test_undeclared_collation_is_dropped_after_what_uses_it(project):
    project.apply_converged(COLLATIONS)
    without = (
        COLLATIONS.replace(
            "CREATE COLLATION bytes (provider = libc, lc_collate = 'C', lc_ctype = 'C');\n",
            "",
        )
        .replace("CREATE DOMAIN code AS text COLLATE public.bytes;\n", "")
        .replace("    name   text COLLATE public.bytes,\n", "")
        .replace("    alias  text COLLATE bytes,\n", "")
        .replace("    code   code\n", "    code   text\n")
        .replace("COMMENT ON COLLATION bytes IS 'byte order';\n", "")
    )
    project.apply_converged(without)
    live = project.live()
    assert set(live.collations) == {"public.spare"}
    assert live.domains == {}


def test_icu_collation(project):
    _require_icu(project.target_url)
    icu = """
CREATE COLLATION ci (provider = icu, locale = 'und-u-ks-level2', deterministic = false);
CREATE TABLE IF NOT EXISTS person (id integer PRIMARY KEY, name text COLLATE ci);
"""
    project.apply_converged(icu)
    ci = project.live().collations["public.ci"]
    assert (ci.provider, ci.deterministic, ci.lc_collate) == ("icu", False, None)
    assert ci.locale
    _admin(project.target_url, "INSERT INTO person VALUES (1, 'Ann')")
    assert _rows(project.target_url, "SELECT id FROM person WHERE name = 'ANN'") == [
        (1,)
    ]
