"""Declarative partitioning: diffing, SQL, and full round trips on a real database.

The unit tests need nothing. The integration tests drive the whole pipeline
(`apply_declared_schema`) against a target and a plan database, then check
that a second diff is empty; they need TEST_DATABASE_URL.
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
    execute_migration,
    introspect_schema,
    partition_notes,
)
from pgdeclare.changes import (
    AddColumn,
    AddConstraint,
    AttachPartition,
    CreateIndex,
    CreatePartitionedIndex,
    CreateTable,
    DetachPartition,
    DropDefault,
    DropIndex,
    DropTable,
    RevokePrivilege,
    SetDefault,
    SetNotNull,
)
from pgdeclare.differ import split_index_operations
from pgdeclare.models import (
    ChangeCategory,
    Column,
    Constraint,
    Grant,
    Index,
    Schema,
    Table,
)


# =============================================================================
# Unit tests
# =============================================================================


def _cols(**overrides: Column) -> dict[str, Column]:
    cols = {
        "id": Column(name="id", type="bigint", nullable=False),
        "ts": Column(name="ts", type="date", nullable=False),
        "v": Column(name="v", type="integer", nullable=True, default="1"),
    }
    cols.update(overrides)
    return cols


def _parent(**kw) -> Table:
    kw.setdefault("columns", _cols())
    return Table(schema="public", name="events", partition_key="RANGE (ts)", **kw)


def _part(name: str, bound: str, **kw) -> Table:
    kw.setdefault("columns", _cols())
    return Table(
        schema="public",
        name=name,
        partition_of=("public", "events"),
        partition_bound=bound,
        **kw,
    )


JAN = "FOR VALUES FROM ('2026-01-01') TO ('2026-02-01')"
FEB = "FOR VALUES FROM ('2026-02-01') TO ('2026-03-01')"


def _schema(*tables: Table, **kw) -> Schema:
    return Schema(tables={t.qualified_name: t for t in tables}, **kw)


class TestPartitionSql:
    def test_partitioned_parent(self):
        sql = CreateTable(table=_parent()).to_sql()
        assert sql.startswith('CREATE TABLE "public"."events" (\n')
        assert sql.endswith("\n) PARTITION BY RANGE (ts);")

    def test_partition(self):
        assert CreateTable(table=_part("events_jan", JAN)).to_sql() == (
            f'CREATE TABLE "public"."events_jan" PARTITION OF "public"."events" {JAN};'
        )

    def test_default_partition_with_own_check(self):
        part = _part(
            "events_rest",
            "DEFAULT",
            constraints=[Constraint(name="c", type="c", definition="CHECK (v > 0)")],
        )
        assert CreateTable(table=part).to_sql() == (
            'CREATE TABLE "public"."events_rest" PARTITION OF "public"."events" (\n'
            '    CONSTRAINT "c" CHECK (v > 0)\n) DEFAULT;'
        )

    def test_sub_partitioned_partition(self):
        part = _part("events_jan", JAN, partition_key="LIST (v)")
        assert (
            CreateTable(table=part).to_sql().endswith(f"{JAN} PARTITION BY LIST (v);")
        )

    def test_names_are_quoted(self):
        part = Table(
            schema='we"ird',
            name='p"; DROP TABLE x; --',
            partition_of=('o"ther', 'pa"rent'),
            partition_bound="DEFAULT",
        )
        assert CreateTable(table=part).to_sql() == (
            'CREATE TABLE "we""ird"."p""; DROP TABLE x; --" PARTITION OF '
            '"o""ther"."pa""rent" DEFAULT;'
        )
        attach = AttachPartition(
            schema='we"ird',
            table='p"',
            parent=('o"ther', 'pa"rent'),
            bound="DEFAULT",
            detach_from=('o"ther', 'pa"rent'),
            old_bound=JAN,
        )
        assert attach.to_sql() == (
            'ALTER TABLE "o""ther"."pa""rent" DETACH PARTITION "we""ird"."p""";\n'
            'ALTER TABLE "o""ther"."pa""rent" ATTACH PARTITION "we""ird"."p""" '
            "DEFAULT;"
        )

    def test_inline_constraint_names_are_quoted(self):
        table = Table(
            schema="public",
            name="t",
            columns={"a": Column(name="a", type="int", nullable=False)},
            constraints=[
                Constraint(name='t"pk', type="p", definition="PRIMARY KEY (a)")
            ],
        )
        assert (
            '    CONSTRAINT "t""pk" PRIMARY KEY (a)'
            in CreateTable(table=table).to_sql()
        )


class TestPartitionDiff:
    def test_declared_partitions_are_created(self):
        desired = _schema(_parent(), _part("events_jan", JAN))
        changes = compute_diff(desired, Schema())
        assert [type(c) for c in changes] == [CreateTable, CreateTable]
        assert all(c.category == ChangeCategory.ONLINE for c in changes)

    def test_in_sync_is_empty(self):
        schema = _schema(_parent(), _part("events_jan", JAN))
        assert compute_diff(schema, schema) == []

    def test_undeclared_partition_is_left_alone_and_noted(self):
        desired = _schema(_parent(), _part("events_jan", JAN))
        runtime = _part("events_feb", FEB)
        actual = _schema(
            _parent(),
            _part("events_jan", JAN),
            runtime,
            indexes={
                "public.events_feb_extra": Index(
                    name="events_feb_extra",
                    table="events_feb",
                    schema="public",
                    definition="CREATE INDEX events_feb_extra ON public.events_feb "
                    "USING btree (v)",
                )
            },
            grants=[
                Grant(
                    object_type="TABLE",
                    object_name="events_feb",
                    schema="public",
                    privilege="SELECT",
                    grantee="reader",
                )
            ],
        )
        assert compute_diff(desired, actual) == []
        assert partition_notes(desired, actual) == [
            f"Partition public.events_feb of public.events ({FEB}) is not "
            "declared; left in place."
        ]

    def test_partitions_of_a_dropped_parent_go_with_it(self):
        actual = _schema(_parent(), _part("events_jan", JAN))
        changes = compute_diff(Schema(), actual)
        assert [type(c) for c in changes] == [DropTable]
        assert changes[0].object_name == "public.events"
        assert partition_notes(Schema(), actual) == []

    def test_new_column_is_added_to_the_parent_only(self):
        wide = _cols(w=Column(name="w", type="text", nullable=True))
        desired = _schema(_parent(columns=wide), _part("events_jan", JAN, columns=wide))
        actual = _schema(_parent(), _part("events_jan", JAN))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, AddColumn)
        assert change.table == "events"

    def test_parent_default_change_does_not_repeat_on_partitions(self):
        five = _cols(v=Column(name="v", type="integer", nullable=True, default="5"))
        desired = _schema(_parent(columns=five), _part("events_jan", JAN, columns=five))
        actual = _schema(_parent(), _part("events_jan", JAN))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, SetDefault)
        assert change.table == "events"

    def test_partition_override_is_rewritten_after_a_parent_change(self):
        """The parent's SET DEFAULT recurses and overwrites the partition's own."""
        five = _cols(v=Column(name="v", type="integer", nullable=True, default="5"))
        nine = _cols(v=Column(name="v", type="integer", nullable=True, default="9"))
        desired = _schema(_parent(columns=five), _part("events_jan", JAN, columns=nine))
        actual = _schema(_parent(), _part("events_jan", JAN, columns=nine))
        changes = compute_diff(desired, actual)
        assert [(type(c), getattr(c, "table")) for c in changes] == [
            (SetDefault, "events"),
            (SetDefault, "events_jan"),
        ]

    def test_partition_override_on_create(self):
        own = _cols(
            v=Column(name="v", type="integer", nullable=False),
        )
        desired = _schema(_parent(), _part("events_jan", JAN, columns=own))
        actual = _schema(_parent())
        changes = compute_diff(desired, actual)
        assert [type(c) for c in changes] == [CreateTable, SetNotNull, DropDefault]
        # A NOT NULL on a table created in the same run scans nothing.
        assert changes[1].category == ChangeCategory.ONLINE

    def test_bound_change_detaches_and_attaches_offline(self):
        desired = _schema(_parent(), _part("events_jan", FEB))
        actual = _schema(_parent(), _part("events_jan", JAN))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, AttachPartition)
        assert change.category == ChangeCategory.OFFLINE
        assert change.to_sql() == (
            'ALTER TABLE "public"."events" DETACH PARTITION "public"."events_jan";\n'
            'ALTER TABLE "public"."events" ATTACH PARTITION "public"."events_jan" '
            f"{FEB};"
        )

    def test_plain_table_declared_as_partition_is_attached(self):
        desired = _schema(_parent(), _part("events_jan", JAN))
        actual = _schema(_parent(), Table(schema="public", name="events_jan"))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, AttachPartition)
        assert change.detach_from is None
        assert change.to_sql().startswith('ALTER TABLE "public"."events" ATTACH')

    def test_partition_declared_as_plain_table_is_detached(self):
        plain = Table(schema="public", name="events_jan", columns=_cols())
        desired = _schema(_parent(), plain)
        actual = _schema(_parent(), _part("events_jan", JAN))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, DetachPartition)
        assert change.category == ChangeCategory.ONLINE

    def test_new_partition_beside_a_default_partition_is_offline(self):
        desired = _schema(
            _parent(), _part("events_rest", "DEFAULT"), _part("events_jan", JAN)
        )
        actual = _schema(_parent(), _part("events_rest", "DEFAULT"))
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, CreateTable)
        assert change.category == ChangeCategory.OFFLINE
        assert "default partition" in change.description

    def test_partition_key_change_is_refused(self):
        desired = _schema(
            Table(schema="public", name="events", partition_key="LIST (v)")
        )
        actual = _schema(_parent())
        with pytest.raises(ValueError, match="partition key.*@migrate-before"):
            compute_diff(desired, actual)

    def test_regular_to_partitioned_is_refused(self):
        desired = _schema(_parent())
        actual = _schema(Table(schema="public", name="events", columns=_cols()))
        with pytest.raises(ValueError, match="into a partitioned.*@migrate-before"):
            compute_diff(desired, actual)

    def test_partitioned_to_regular_is_refused(self):
        desired = _schema(Table(schema="public", name="events", columns=_cols()))
        actual = _schema(_parent())
        with pytest.raises(ValueError, match="into a plain table.*@migrate-before"):
            compute_diff(desired, actual)

    def test_foreign_key_on_partitioned_table_is_validated_offline(self):
        fk = Constraint(
            name="fk", type="f", definition="FOREIGN KEY (id) REFERENCES r(id)"
        )
        desired = _schema(_parent(constraints=[fk]))
        actual = _schema(_parent())
        (change,) = compute_diff(desired, actual)
        assert isinstance(change, AddConstraint)
        assert change.category == ChangeCategory.OFFLINE
        assert "NOT VALID" not in change.to_sql()


def _pindex(name: str = "events_v_idx", definition: str = "") -> Index:
    return Index(
        name=name,
        table="events",
        schema="public",
        definition=definition
        or f"CREATE INDEX {name} ON ONLY public.events USING btree (v)",
        partitioned=True,
    )


class TestPartitionedIndexDiff:
    def test_created_per_partition_in_the_index_phase(self):
        desired = Schema(indexes={"public.events_v_idx": _pindex()})
        (change,) = compute_diff(desired, Schema())
        assert isinstance(change, CreatePartitionedIndex)
        assert change.category == ChangeCategory.ONLINE
        ddl, index_ops = split_index_operations([change])
        assert ddl == [] and index_ops == [change]
        assert "ATTACH PARTITION" in change.to_sql()

    def test_dropped_without_concurrently(self):
        (change,) = compute_diff(
            Schema(), Schema(indexes={"public.events_v_idx": _pindex()})
        )
        assert isinstance(change, DropIndex)
        assert change.to_sql() == 'DROP INDEX IF EXISTS "public"."events_v_idx";'

    def test_invalid_index_is_finished(self):
        index = _pindex()
        broken = _pindex()
        broken.is_valid = False
        (change,) = compute_diff(
            Schema(indexes={"public.events_v_idx": index}),
            Schema(indexes={"public.events_v_idx": broken}),
        )
        assert isinstance(change, CreatePartitionedIndex)

    def test_regular_index_unchanged(self):
        plain = Index(
            name="t_a", table="t", schema="public", definition="CREATE INDEX t_a ..."
        )
        (change,) = compute_diff(Schema(indexes={"public.t_a": plain}), Schema())
        assert type(change) is CreateIndex

    def test_grants_on_undeclared_partition_are_not_revoked(self):
        grant = Grant(
            object_type="TABLE",
            object_name="events_feb",
            schema="public",
            privilege="SELECT",
            grantee="reader",
        )
        actual = _schema(_parent(), _part("events_feb", FEB), grants=[grant])
        changes = compute_diff(_schema(_parent()), actual)
        assert not any(isinstance(c, RevokePrivilege) for c in changes)


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


def _rows(url: str, sql: str, params=None) -> list[tuple]:
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@pytest.fixture
def env(server_url, tmp_path):
    """Target and plan databases, a managed role, and an SQL dir; all removed after."""
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_pt_target_{suffix}", f"pgdeclare_pt_plan_{suffix}"
    role = f"pt_reader_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _Env(_with_db(server_url, target), plan, role, tmp_path / "sql")
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        _admin(server_url, f'DROP ROLE IF EXISTS "{role}"')


class _Env:
    def __init__(self, url: str, plan: str, role: str, sql_dir: Path):
        self.url, self.plan, self.role, self.sql_dir = url, plan, role, sql_dir
        self.reported: list[str] = []
        self.config = PgDeclareConfig(
            managed_roles=(role,), report=self.reported.append
        )

    def declare(self, sql: str) -> None:
        schemas = self.sql_dir / "schemas"
        schemas.mkdir(parents=True, exist_ok=True)
        role = f'CREATE ROLE "{self.role}" NOLOGIN;\n'
        role += f'GRANT USAGE ON SCHEMA public TO "{self.role}";\n'
        (schemas / "schema.sql").write_text(role + sql.replace("ROLE", self.role))

    def apply(self, offline: bool = False) -> None:
        apply_declared_schema(
            self.url, self.sql_dir, self.plan, offline=offline, config=self.config
        )

    def schemas(self) -> tuple[Schema, Schema]:
        desired = build_desired_schema(
            self.sql_dir, self.url, self.plan, config=self.config
        )
        return desired, introspect_schema(self.url, config=self.config)

    def residual(self) -> list:
        desired, actual = self.schemas()
        return compute_diff(desired, actual, schemas=["public"], config=self.config)

    def partitions(self, parent: str) -> list[tuple[str, str]]:
        return _rows(
            self.url,
            """
            SELECT c.relname, pg_get_expr(c.relpartbound, c.oid)
            FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
            WHERE i.inhparent = %s::regclass ORDER BY 1
            """,
            (parent,),
        )

    def index_state(self, index: str) -> tuple[bool, list[str]]:
        """(valid, tables of the partition indexes attached to it)."""
        valid = _rows(
            self.url,
            "SELECT indisvalid FROM pg_index WHERE indexrelid = %s::regclass",
            (index,),
        )[0][0]
        attached = _rows(
            self.url,
            """
            SELECT x.indrelid::regclass::text FROM pg_inherits i
            JOIN pg_index x ON x.indexrelid = i.inhrelid
            WHERE i.inhparent = %s::regclass ORDER BY 1
            """,
            (index,),
        )
        return valid, [r[0] for r in attached]


BASE = """
CREATE TABLE ref (id bigint PRIMARY KEY);

CREATE TABLE events (
    id bigint NOT NULL,
    ref_id bigint REFERENCES ref (id),
    ts date NOT NULL,
    kind text NOT NULL DEFAULT 'a',
    v integer DEFAULT 1 CHECK (v > 0),
    PRIMARY KEY (id, ts, kind)
) PARTITION BY RANGE (ts);
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
CREATE POLICY events_all ON events USING (true);
GRANT SELECT ON events TO "ROLE";
CREATE TRIGGER events_noop BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION suppress_redundant_updates_trigger();

CREATE TABLE events_2026_01 PARTITION OF events
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE events_2026_02 PARTITION OF events
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01')
    PARTITION BY LIST (kind);
CREATE TABLE events_2026_02_a PARTITION OF events_2026_02 FOR VALUES IN ('a');
CREATE TABLE events_2026_02_rest PARTITION OF events_2026_02 DEFAULT;
CREATE TABLE events_rest PARTITION OF events (
    CONSTRAINT events_rest_v CHECK (v < 1000)
) DEFAULT;
"""

INDEX = "CREATE INDEX events_kind_idx ON events (kind, v);\n"


@pytest.mark.integration
class TestPartitionRoundTrip:
    def test_declare_apply_rediff_is_empty(self, env):
        env.declare(BASE + INDEX)
        env.apply()

        assert env.residual() == []
        assert env.partitions("events") == [
            ("events_2026_01", "FOR VALUES FROM ('2026-01-01') TO ('2026-02-01')"),
            ("events_2026_02", "FOR VALUES FROM ('2026-02-01') TO ('2026-03-01')"),
            ("events_rest", "DEFAULT"),
        ]
        valid, attached = env.index_state("events_kind_idx")
        assert valid and attached == ["events_2026_01", "events_2026_02", "events_rest"]

        _admin(
            env.url,
            "INSERT INTO ref VALUES (1);"
            "INSERT INTO events (id, ref_id, ts, kind) VALUES"
            " (1, 1, '2026-01-05', 'a'), (2, 1, '2026-02-05', 'a'),"
            " (3, 1, '2026-02-06', 'b'), (4, 1, '2027-01-01', 'a')",
        )
        assert _rows(
            env.url,
            "SELECT tableoid::regclass::text, id FROM events ORDER BY id",
        ) == [
            ("events_2026_01", 1),
            ("events_2026_02_a", 2),
            ("events_2026_02_rest", 3),
            ("events_rest", 4),
        ]
        assert env.residual() == []

    def test_add_a_partition(self, env):
        no_default = BASE.split("CREATE TABLE events_rest")[0]
        env.declare(no_default + INDEX)
        env.apply()
        _admin(env.url, "INSERT INTO events (id, ts) VALUES (1, '2026-01-02')")

        march = (
            "CREATE TABLE events_2026_03 PARTITION OF events"
            " FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');\n"
        )
        env.declare(no_default + march + INDEX)
        # No default partition to scan: online.
        env.apply()

        assert (
            "events_2026_03",
            "FOR VALUES FROM ('2026-03-01') TO ('2026-04-01')",
        ) in env.partitions("events")
        assert "events_2026_03" in env.index_state("events_kind_idx")[1]
        assert env.residual() == []

    def test_runtime_partition_is_left_alone_and_noted(self, env):
        env.declare(BASE + INDEX)
        env.apply()
        # What pg_partman or a cron job does: a partition the files never name,
        # with an index and a grant of its own.
        _admin(
            env.url,
            "ALTER TABLE events DETACH PARTITION events_rest;"
            "CREATE TABLE events_2026_03 PARTITION OF events"
            " FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');"
            "ALTER TABLE events ATTACH PARTITION events_rest DEFAULT;"
            "CREATE INDEX events_2026_03_id ON events_2026_03 (id);"
            f'GRANT SELECT ON events_2026_03 TO "{env.role}";'
            "INSERT INTO events (id, ts) VALUES (7, '2026-03-03');",
        )

        desired, actual = env.schemas()
        assert compute_diff(desired, actual, schemas=["public"]) == []
        assert partition_notes(desired, actual) == [
            "Partition public.events_2026_03 of public.events (FOR VALUES FROM "
            "('2026-03-01') TO ('2026-04-01')) is not declared; left in place."
        ]

        env.apply()
        assert any("events_2026_03" in line for line in env.reported)
        assert _rows(env.url, "SELECT id FROM events_2026_03") == [(7,)]
        assert _rows(
            env.url, "SELECT 1 FROM pg_indexes WHERE indexname = 'events_2026_03_id'"
        ) == [(1,)]

    def test_parent_index_built_per_partition_and_attached(self, env):
        env.declare(BASE)
        env.apply()
        _admin(
            env.url,
            "INSERT INTO events (id, ts, kind) VALUES"
            " (1, '2026-01-02', 'a'), (2, '2026-02-02', 'z'), (3, '2030-01-01', 'a');"
            # An undeclared partition gets the index too.
            "ALTER TABLE events DETACH PARTITION events_rest;"
            "CREATE TABLE events_2026_03 PARTITION OF events"
            " FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');"
            "ALTER TABLE events ATTACH PARTITION events_rest DEFAULT;",
        )

        env.declare(BASE + INDEX)
        desired, actual = env.schemas()
        changes = compute_diff(desired, actual, schemas=["public"])
        assert [type(c) for c in changes] == [CreatePartitionedIndex]
        ok, executed = execute_migration(env.url, changes)
        assert ok
        # No CREATE INDEX on the parent that builds every partition at once.
        assert executed[0].startswith("CREATE INDEX events_kind_idx ON ONLY ")
        builds = [e for e in executed if "CONCURRENTLY" in e]
        assert len(builds) == 5  # one per leaf partition
        # Each leaf, plus the sub-partitioned partition's own index.
        assert sum("ATTACH PARTITION" in e for e in executed) == 6

        valid, attached = env.index_state("events_kind_idx")
        assert valid
        assert attached == [
            "events_2026_01",
            "events_2026_02",
            "events_2026_03",
            "events_rest",
        ]
        sub_index = _rows(
            env.url,
            """
            SELECT x.indexrelid::regclass::text FROM pg_inherits i
            JOIN pg_index x ON x.indexrelid = i.inhrelid
            WHERE i.inhparent = 'events_kind_idx'::regclass
              AND x.indrelid = 'events_2026_02'::regclass
            """,
        )[0][0]
        assert env.index_state(sub_index) == (
            True,
            ["events_2026_02_a", "events_2026_02_rest"],
        )
        assert env.residual() == []

        # Dropped again, partitions' indexes with it.
        env.declare(BASE)
        env.apply()
        assert (
            _rows(
                env.url,
                "SELECT count(*) FROM pg_indexes WHERE indexdef LIKE '%(kind, v)%'",
            )[0][0]
            == 0
        )

    def test_half_built_index_is_finished(self, env):
        env.declare(BASE)
        env.apply()
        # A run that died after creating the parent's index and one partition's.
        _admin(
            env.url,
            "CREATE INDEX events_kind_idx ON ONLY events (kind, v);"
            "CREATE INDEX jan_kind ON events_2026_01 (kind, v);"
            "ALTER INDEX events_kind_idx ATTACH PARTITION jan_kind;",
        )
        assert env.index_state("events_kind_idx") == (False, ["events_2026_01"])

        env.declare(BASE + INDEX)
        env.apply()
        valid, attached = env.index_state("events_kind_idx")
        assert valid and len(attached) == 3
        assert env.residual() == []

    def test_bound_change_is_offline(self, env):
        env.declare(BASE)
        env.apply()
        _admin(env.url, "INSERT INTO events (id, ts) VALUES (1, '2026-01-02')")

        env.declare(BASE.replace("TO ('2026-02-01');", "TO ('2026-01-15');"))
        with pytest.raises(Exception):
            env.apply()
        env.apply(offline=True)

        assert (
            "events_2026_01",
            "FOR VALUES FROM ('2026-01-01') TO ('2026-01-15')",
        ) in env.partitions("events")
        assert _rows(env.url, "SELECT id FROM events_2026_01") == [(1,)]
        assert env.residual() == []

    def test_partition_key_change_is_refused(self, env):
        keyed = (
            "CREATE TABLE k (a int, b int) PARTITION BY RANGE (a);\n"
            "CREATE TABLE k1 PARTITION OF k FOR VALUES FROM (0) TO (10);\n"
        )
        env.declare(keyed)
        env.apply()
        env.declare(keyed.replace("RANGE (a)", "RANGE (b)"))
        with pytest.raises(ValueError, match="partition key.*@migrate-before"):
            env.apply()

    def test_regular_to_partitioned_is_refused(self, env):
        env.declare("CREATE TABLE t (id int NOT NULL, ts date NOT NULL);")
        env.apply()
        env.declare(
            "CREATE TABLE t (id int NOT NULL, ts date NOT NULL)"
            " PARTITION BY RANGE (ts);"
        )
        with pytest.raises(ValueError, match="into a partitioned.*@migrate-before"):
            env.apply()

    def test_partition_override_survives_a_parent_default_change(self, env):
        own = BASE.replace(
            "CREATE TABLE events_2026_01 PARTITION OF events\n",
            "CREATE TABLE events_2026_01 PARTITION OF events"
            " (kind WITH OPTIONS DEFAULT 'jan')\n",
        )
        env.declare(own)
        env.apply()
        assert env.residual() == []

        env.declare(own.replace("DEFAULT 'a'", "DEFAULT 'b'"))
        env.apply()
        defaults = dict(
            _rows(
                env.url,
                """
                SELECT c.relname, pg_get_expr(d.adbin, d.adrelid)
                FROM pg_attrdef d
                JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                JOIN pg_class c ON c.oid = d.adrelid
                WHERE a.attname = 'kind'
                """,
            )
        )
        assert defaults["events"] == "'b'::text"
        assert defaults["events_2026_01"] == "'jan'::text"
        assert defaults["events_rest"] == "'b'::text"
        assert env.residual() == []
