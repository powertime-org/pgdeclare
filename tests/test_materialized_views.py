"""Materialized views: diffing, SQL, and full round trips on a real database.

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
    introspect_schema,
)
from pgdeclare.changes import (
    AlterMaterializedViewOptions,
    CreateIndex,
    CreateMaterializedView,
    DropAndCreateMaterializedView,
    DropAndCreateView,
    DropIndex,
    DropMaterializedView,
    RevokePrivilege,
)
from pgdeclare.differ import _relation_dependent_order
from pgdeclare.models import (
    ChangeCategory,
    Grant,
    Index,
    MaterializedView,
    Schema,
    View,
)
from pgdeclare.statement_parser import parse_sql_content


# =============================================================================
# Unit tests
# =============================================================================


def _mv(definition: str = " SELECT a\n   FROM t;", **kw) -> MaterializedView:
    return MaterializedView(schema="public", name="m", definition=definition, **kw)


def _grant(name: str = "m") -> Grant:
    return Grant(
        object_type="TABLE",
        object_name=name,
        schema="public",
        privilege="SELECT",
        grantee="reader",
    )


def _index(name: str = "m_a_idx", table: str = "m") -> Index:
    return Index(
        name=name,
        table=table,
        schema="public",
        definition=f"CREATE INDEX {name} ON public.{table} USING btree (a)",
    )


class TestMaterializedViewDiff:
    def test_new_matview_is_created_as_declared(self):
        desired = Schema(
            materialized_views={
                "public.m": _mv(with_data=False, reloptions=["FILLFACTOR = 70"])
            }
        )

        (change,) = compute_diff(desired, Schema())

        assert isinstance(change, CreateMaterializedView)
        assert change.category == ChangeCategory.ONLINE
        assert change.to_sql() == (
            'CREATE MATERIALIZED VIEW "public"."m" WITH (fillfactor=70) AS '
            "SELECT a\n   FROM t WITH NO DATA;"
        )

    def test_names_are_quoted(self):
        mv = MaterializedView(
            schema='we"ird', name='m"; DROP TABLE x; --', definition="SELECT 1;"
        )
        sql = CreateMaterializedView(matview=mv).to_sql()
        assert sql.startswith(
            'CREATE MATERIALIZED VIEW "we""ird"."m""; DROP TABLE x; --" AS'
        )
        assert sql.endswith("SELECT 1 WITH DATA;")

    def test_population_state_is_not_drift(self):
        """A refreshed (or emptied) matview has not drifted from its declaration."""
        desired = Schema(materialized_views={"public.m": _mv(with_data=False)})
        actual = Schema(materialized_views={"public.m": _mv(with_data=True)})
        assert compute_diff(desired, actual) == []

    def test_definition_change_recreates_offline_with_grants_and_indexes(self):
        grant, index = _grant(), _index()
        new_index = _index("m_b_idx")
        desired = Schema(
            materialized_views={"public.m": _mv(" SELECT a, b\n   FROM t;")},
            grants=[grant],
            indexes={"public.m_a_idx": index, "public.m_b_idx": new_index},
        )
        actual = Schema(
            materialized_views={"public.m": _mv()},
            grants=[grant],
            indexes={"public.m_a_idx": index, "public.m_old_idx": _index("m_old_idx")},
        )

        changes = compute_diff(desired, actual)

        (recreate,) = changes
        assert isinstance(recreate, DropAndCreateMaterializedView)
        assert recreate.category == ChangeCategory.OFFLINE
        # The index changes are folded into the recreate, not left standalone.
        assert not [c for c in changes if isinstance(c, (CreateIndex, DropIndex))]
        assert [i.name for i in recreate.indexes] == ["m_a_idx", "m_b_idx"]
        sql = recreate.to_sql()
        drop = sql.index('DROP MATERIALIZED VIEW IF EXISTS "public"."m"')
        create = sql.index("CREATE MATERIALIZED VIEW")
        assert drop < create < sql.index("TO reader") < sql.index("m_a_idx")
        assert "CONCURRENTLY" not in sql
        assert "rebuilds its data" in recreate.description
        assert "replaying 1 grant" in recreate.description
        assert "rebuilding 2 indexes" in recreate.description

    def test_storage_options_change_in_place(self):
        desired = Schema(
            materialized_views={"public.m": _mv(reloptions=["fillfactor=70"])}
        )
        actual = Schema(
            materialized_views={
                "public.m": _mv(
                    reloptions=["fillfactor=90", "autovacuum_enabled=false"]
                )
            }
        )

        (change,) = compute_diff(desired, actual)

        assert isinstance(change, AlterMaterializedViewOptions)
        assert change.category == ChangeCategory.ONLINE
        assert change.to_sql() == (
            'ALTER MATERIALIZED VIEW "public"."m" RESET (autovacuum_enabled); '
            'ALTER MATERIALIZED VIEW "public"."m" SET (fillfactor=70);'
        )

    def test_undeclared_matview_is_dropped_without_revoking_first(self):
        actual = Schema(materialized_views={"public.m": _mv()}, grants=[_grant()])

        changes = compute_diff(Schema(), actual)

        assert [type(c) for c in changes] == [DropMaterializedView]
        assert changes[0].to_sql() == 'DROP MATERIALIZED VIEW IF EXISTS "public"."m";'
        assert not [c for c in changes if isinstance(c, RevokePrivilege)]

    def test_view_recreate_carries_a_dependent_matview(self):
        """A view that a matview reads from cannot be dropped on its own."""
        base_old = View(schema="public", name="v", definition=" SELECT a FROM t;")
        base_new = View(schema="public", name="v", definition=" SELECT a, b FROM t;")
        dependent = MaterializedView(
            schema="public", name="m", definition=" SELECT a FROM v;"
        )
        desired = Schema(
            views={"public.v": base_new},
            materialized_views={"public.m": dependent},
            indexes={"public.m_a_idx": _index()},
            grants=[_grant()],
        )
        actual = Schema(
            views={"public.v": base_old},
            materialized_views={"public.m": dependent},
            indexes={"public.m_a_idx": _index()},
            grants=[_grant()],
            relation_dependents={"public.v": ["public.m"]},
        )

        (recreate,) = compute_diff(desired, actual)

        assert isinstance(recreate, DropAndCreateView)
        # Rebuilding the matview's data makes the view recreate offline too.
        assert recreate.category == ChangeCategory.OFFLINE
        sql = recreate.to_sql()
        order = [
            sql.index('DROP MATERIALIZED VIEW IF EXISTS "public"."m"'),
            sql.index('DROP VIEW IF EXISTS "public"."v"'),
            sql.index('CREATE VIEW "public"."v"'),
            sql.index('CREATE MATERIALIZED VIEW "public"."m"'),
            sql.index("TO reader"),
            sql.index("CREATE INDEX m_a_idx"),
        ]
        assert order == sorted(order)
        assert "recreating dependent m" in recreate.description

    def test_matview_recreate_folds_a_changed_dependent_view(self):
        """A dependent that also changed is rebuilt once, from its declaration."""
        top_old = View(schema="public", name="top", definition=" SELECT a FROM m;")
        top_new = View(schema="public", name="top", definition=" SELECT a, b FROM m;")
        desired = Schema(
            views={"public.top": top_new},
            materialized_views={"public.m": _mv(" SELECT a, b\n   FROM t;")},
        )
        actual = Schema(
            views={"public.top": top_old},
            materialized_views={"public.m": _mv()},
            relation_dependents={"public.m": ["public.top"]},
        )

        (recreate,) = compute_diff(desired, actual)

        assert isinstance(recreate, DropAndCreateMaterializedView)
        assert [d.relation for d in recreate.dependents] == [top_new]

    def test_undeclared_dependent_is_dropped_not_rebuilt(self):
        top = View(schema="public", name="top", definition=" SELECT a FROM m;")
        desired = Schema(materialized_views={"public.m": _mv(" SELECT a, b FROM t;")})
        actual = Schema(
            views={"public.top": top},
            materialized_views={"public.m": _mv()},
            relation_dependents={"public.m": ["public.top"]},
        )

        changes = compute_diff(desired, actual)

        (recreate,) = [
            c for c in changes if isinstance(c, DropAndCreateMaterializedView)
        ]
        sql = recreate.to_sql()
        assert 'DROP VIEW IF EXISTS "public"."top"' in sql
        assert 'CREATE VIEW "public"."top"' not in sql
        assert "dropping dependent top" in recreate.description

    def test_dependents_come_in_creation_order(self):
        # v2 reads from m and from v1; v1 reads from m.
        order = _relation_dependent_order(
            "m", {"m": ["v2", "v1"], "v1": ["v2"], "v2": []}
        )
        assert order == ["v1", "v2"]


def test_statement_parser_tracks_materialized_views():
    (stmt,) = parse_sql_content(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS app.totals AS SELECT 1 WITH NO DATA;"
    )
    assert stmt.statement_type == "CREATE MATERIALIZED VIEW"
    assert stmt.object_type == "MATERIALIZED VIEW"
    assert stmt.object_name == "app.totals"
    assert stmt.duplicate_key == "MATERIALIZED VIEW:app.totals"
    assert stmt.is_idempotent is True


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
    target, plan = f"pgdeclare_mv_target_{suffix}", f"pgdeclare_mv_plan_{suffix}"
    role = f"mv_reader_{suffix}"
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
        self.config = PgDeclareConfig(managed_roles=(role,), report=lambda _l: None)

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

    def residual(self) -> list:
        desired = build_desired_schema(
            self.sql_dir, self.url, self.plan, config=self.config
        )
        actual = introspect_schema(self.url, config=self.config)
        return compute_diff(desired, actual, schemas=["public"], config=self.config)

    def can_select(self, relation: str) -> bool:
        return _rows(
            self.url,
            "SELECT has_table_privilege(%s, %s, 'SELECT')",
            (self.role, relation),
        )[0][0]

    def populated(self, relation: str) -> bool:
        return _rows(
            self.url,
            "SELECT relispopulated FROM pg_class WHERE oid = %s::regclass",
            (relation,),
        )[0][0]

    def indexes(self, relation: str) -> list[str]:
        return [
            r[0]
            for r in _rows(
                self.url,
                "SELECT indexname FROM pg_indexes WHERE tablename = %s ORDER BY 1",
                (relation,),
            )
        ]


BASE = """
CREATE TABLE IF NOT EXISTS t (id int PRIMARY KEY, a int NOT NULL, b int);
"""

V1 = (
    BASE
    + """
CREATE MATERIALIZED VIEW totals WITH (fillfactor = 70) AS
    SELECT a, count(*) AS n FROM t GROUP BY a WITH DATA;
CREATE UNIQUE INDEX totals_a_idx ON totals (a);
GRANT SELECT ON totals TO "ROLE";

CREATE MATERIALIZED VIEW lazy AS SELECT id FROM t WITH NO DATA;
"""
)


@pytest.mark.integration
class TestMaterializedViewRoundTrip:
    def test_declare_apply_rediff_is_empty(self, env):
        env.declare(V1)
        env.apply()

        assert env.populated("totals") is True
        assert env.populated("lazy") is False
        assert env.indexes("totals") == ["totals_a_idx"]
        assert env.can_select("totals") is True
        assert _rows(
            env.url, "SELECT reloptions FROM pg_class WHERE relname = 'totals'"
        ) == [(["fillfactor=70"],)]
        assert env.residual() == []

        # A refresh is runtime state, not drift.
        _admin(env.url, "REFRESH MATERIALIZED VIEW lazy")
        assert env.residual() == []

    def test_definition_change_recreates_and_keeps_index_and_grants(self, env):
        env.declare(V1)
        env.apply()
        _admin(env.url, "INSERT INTO t VALUES (1, 10, 5), (2, 10, 7), (3, 20, 1)")

        env.declare(V1.replace("count(*) AS n", "count(*) AS n, sum(b) AS total"))
        # Rebuilding a materialized view's data is offline.
        with pytest.raises(Exception):
            env.apply()
        env.apply(offline=True)

        assert _rows(env.url, "SELECT a, n, total FROM totals ORDER BY a") == [
            (10, 2, 12),
            (20, 1, 1),
        ]
        assert env.indexes("totals") == ["totals_a_idx"]
        assert env.can_select("totals") is True
        assert env.residual() == []

    def test_storage_option_change_is_online(self, env):
        env.declare(V1)
        env.apply()

        env.declare(V1.replace("fillfactor = 70", "fillfactor = 50"))
        env.apply()

        assert _rows(
            env.url, "SELECT reloptions FROM pg_class WHERE relname = 'totals'"
        ) == [(["fillfactor=50"],)]
        assert env.residual() == []

    def test_undeclared_matview_is_dropped(self, env):
        env.declare(V1)
        env.apply()

        env.declare(BASE)
        env.apply()

        assert _rows(env.url, "SELECT matviewname FROM pg_matviews") == []
        assert env.residual() == []


LAYERED = (
    BASE
    + """
CREATE VIEW base_v AS SELECT id, a FROM t;
CREATE MATERIALIZED VIEW mid AS SELECT id, a FROM base_v WITH DATA;
CREATE INDEX mid_a_idx ON mid (a);
GRANT SELECT ON mid TO "ROLE";
CREATE VIEW top_v AS SELECT id, a FROM mid;
GRANT SELECT ON top_v TO "ROLE";
"""
)


@pytest.mark.integration
class TestMaterializedViewDependencies:
    """Views on matviews and matviews on views, created, changed and dropped."""

    def test_layered_build_converges(self, env):
        env.declare(LAYERED)
        env.apply()

        assert env.residual() == []
        assert env.can_select("top_v") is True

    def test_changing_the_view_under_a_matview(self, env):
        env.declare(LAYERED)
        env.apply()
        _admin(env.url, "INSERT INTO t VALUES (1, 10, 5)")

        mid = _rows(env.url, "SELECT 'mid'::regclass::oid")

        # An appended column is replaced in place (online): the matview reading
        # from the view is left as it is, data included, as after any change to
        # what it reads. Refreshing it is the application's call.
        env.declare(LAYERED.replace("SELECT id, a FROM t", "SELECT id, a, b FROM t"))
        env.apply()

        assert _rows(env.url, "SELECT 'mid'::regclass::oid") == mid
        assert _rows(env.url, "SELECT id, a FROM top_v") == []
        _admin(env.url, "REFRESH MATERIALIZED VIEW mid")
        assert _rows(env.url, "SELECT id, a FROM top_v") == [(1, 10)]
        assert env.indexes("mid") == ["mid_a_idx"]
        assert env.can_select("mid") is True
        assert env.can_select("top_v") is True
        assert env.residual() == []

    def test_recreating_the_view_under_a_matview(self, env):
        env.declare(LAYERED)
        env.apply()
        _admin(env.url, "INSERT INTO t VALUES (1, 10, 5)")

        # A retyped column cannot be replaced in place: the view is recreated,
        # which rebuilds the matview reading from it (offline).
        env.declare(
            LAYERED.replace("SELECT id, a FROM t", "SELECT id, a::bigint AS a FROM t")
        )
        (change,) = env.residual()
        assert isinstance(change, DropAndCreateView)
        assert change.category == ChangeCategory.OFFLINE
        env.apply(offline=True)

        assert _rows(env.url, "SELECT id, a FROM top_v") == [(1, 10)]
        assert env.indexes("mid") == ["mid_a_idx"]
        assert env.can_select("mid") is True
        assert env.can_select("top_v") is True
        assert env.residual() == []

    def test_changing_the_matview_under_a_view(self, env):
        env.declare(LAYERED)
        env.apply()
        _admin(env.url, "INSERT INTO t VALUES (1, 10, 5), (2, 30, 1)")

        env.declare(
            LAYERED.replace(
                "SELECT id, a FROM base_v WITH DATA",
                "SELECT id, a FROM base_v WHERE a > 20 WITH DATA",
            )
        )
        env.apply(offline=True)

        assert _rows(env.url, "SELECT id, a FROM top_v") == [(2, 30)]
        assert env.indexes("mid") == ["mid_a_idx"]
        assert env.can_select("mid") is True
        assert env.can_select("top_v") is True
        assert env.residual() == []

    def test_dropping_the_whole_stack(self, env):
        env.declare(LAYERED)
        env.apply()

        env.declare(BASE)
        env.apply()

        assert _rows(env.url, "SELECT matviewname FROM pg_matviews") == []
        assert (
            _rows(env.url, "SELECT viewname FROM pg_views WHERE schemaname = 'public'")
            == []
        )
        assert env.residual() == []

    def test_matview_over_a_function(self, env):
        sql = (
            BASE
            + """
CREATE FUNCTION doubled(x int) RETURNS int LANGUAGE sql IMMUTABLE AS $$ SELECT x * 2 $$;
CREATE MATERIALIZED VIEW twice AS SELECT id, doubled(a) AS a2 FROM t WITH DATA;
"""
        )
        env.declare(sql)
        env.apply()
        _admin(env.url, "INSERT INTO t VALUES (1, 4, 0)")
        _admin(env.url, "REFRESH MATERIALIZED VIEW twice")

        assert _rows(env.url, "SELECT a2 FROM twice") == [(8,)]
        assert env.residual() == []


def test_a_drop_blocked_by_dependents_is_retried():
    """DROP of a relation something still reads from waits for that one first."""
    from pgdeclare.retry_executor import ERROR_CODE_CATEGORIES, ErrorCategory

    assert ERROR_CODE_CATEGORIES["2BP01"] == ErrorCategory.DEPENDENCY


def test_no_revoke_on_a_dropped_view():
    """The DROP takes the ACL; a REVOKE after it would be stuck forever."""
    view = View(schema="public", name="v", definition=" SELECT 1;")
    actual = Schema(views={"public.v": view}, grants=[_grant("v")])

    changes = compute_diff(Schema(), actual)

    assert not [c for c in changes if isinstance(c, RevokePrivilege)]
