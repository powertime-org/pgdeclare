"""Views: dropping a granted view, and a body change under dependent views.

The integration tests drive the whole pipeline (`apply_declared_schema`)
against a target and a plan database, then check that a second diff is empty;
they need TEST_DATABASE_URL.
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
from pgdeclare.changes import DropAndCreateView, ReplaceView
from pgdeclare.differ import _view_replaceable
from pgdeclare.models import View

from conftest import require_server_version


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
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_v_target_{suffix}", f"pgdeclare_v_plan_{suffix}"
    role = f"v_reader_{suffix}"
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

    def plan_changes(self) -> list:
        desired = build_desired_schema(
            self.sql_dir, self.url, self.plan, config=self.config
        )
        actual = introspect_schema(self.url, config=self.config)
        return compute_diff(desired, actual, schemas=["public"], config=self.config)

    def apply(self) -> None:
        apply_declared_schema(self.url, self.sql_dir, self.plan, config=self.config)

    def can_select(self, relation: str) -> bool:
        return _rows(
            self.url,
            "SELECT has_table_privilege(%s, %s, 'SELECT')",
            (self.role, relation),
        )[0][0]

    def oid(self, relation: str) -> int:
        return _rows(self.url, "SELECT %s::regclass::oid", (relation,))[0][0]


BASE = """
CREATE TABLE IF NOT EXISTS people (id int PRIMARY KEY, name text, active bool);
"""

PEOPLE_SET = """
CREATE VIEW people_set AS SELECT id, name FROM people WHERE active;
GRANT SELECT ON people_set TO "ROLE";
"""

DEPENDENTS = """
CREATE VIEW people_adder AS SELECT id FROM people_set;
GRANT SELECT ON people_adder TO "ROLE";
CREATE VIEW people_picker AS SELECT a.id, s.name
    FROM people_adder AS a JOIN people_set AS s USING (id);
GRANT SELECT ON people_picker TO "ROLE";
"""


@pytest.mark.integration
class TestViewLifecycle:
    def test_dropping_a_granted_view_plans_no_revoke(self, env):
        env.declare(BASE + PEOPLE_SET)
        env.apply()
        assert env.can_select("people_set")

        env.declare(BASE)
        assert not any("REVOKE" in c.to_sql() for c in env.plan_changes())
        env.apply()

        assert (
            _rows(env.url, "SELECT viewname FROM pg_views WHERE schemaname = 'public'")
            == []
        )
        assert env.plan_changes() == []

    def test_body_change_under_dependents_replaces_in_place(self, env):
        env.declare(BASE + PEOPLE_SET + DEPENDENTS)
        env.apply()
        _admin(env.url, "INSERT INTO people VALUES (1, 'a', true), (2, 'b', false)")
        before = {
            v: env.oid(v) for v in ("people_set", "people_adder", "people_picker")
        }

        env.declare(
            BASE + PEOPLE_SET.replace("WHERE active", "WHERE NOT active") + DEPENDENTS
        )
        (change,) = env.plan_changes()
        assert "CREATE OR REPLACE VIEW" in change.to_sql()
        assert "DROP" not in change.to_sql()
        env.apply()

        # Replaced in place: no relation was dropped, so nothing lost its ACL.
        assert {v: env.oid(v) for v in before} == before
        assert _rows(env.url, "SELECT id, name FROM people_picker") == [(2, "b")]
        for v in before:
            assert env.can_select(v)
        assert env.plan_changes() == []

    def test_appended_column_and_options_replace_in_place(self, env):
        require_server_version(env.url, 150000, "security_invoker views")
        env.declare(BASE + PEOPLE_SET + DEPENDENTS)
        env.apply()
        before = env.oid("people_set")

        env.declare(
            BASE
            + PEOPLE_SET.replace(
                "CREATE VIEW people_set AS SELECT id, name",
                "CREATE VIEW people_set WITH (security_invoker = true)"
                " AS SELECT id, name, active",
            )
            + DEPENDENTS
        )
        (change,) = env.plan_changes()
        assert isinstance(change, ReplaceView)
        env.apply()

        assert env.oid("people_set") == before
        assert _rows(
            env.url, "SELECT reloptions FROM pg_class WHERE relname = 'people_set'"
        ) == [(["security_invoker=true"],)]
        assert env.plan_changes() == []

        # A body change that drops the option clears it in the same statement.
        env.declare(
            BASE
            + PEOPLE_SET.replace("SELECT id, name", "SELECT id, name, active").replace(
                "WHERE active", "WHERE NOT active"
            )
            + DEPENDENTS
        )
        (change,) = env.plan_changes()
        assert isinstance(change, ReplaceView)
        env.apply()
        assert env.oid("people_set") == before
        assert _rows(
            env.url, "SELECT reloptions FROM pg_class WHERE relname = 'people_set'"
        ) == [(None,)]
        assert env.plan_changes() == []

    def test_retyped_column_still_recreates_with_dependents(self, env):
        env.declare(BASE + PEOPLE_SET + DEPENDENTS)
        env.apply()

        env.declare(
            BASE
            + PEOPLE_SET.replace(
                "SELECT id, name", "SELECT id, name::varchar(10) AS name"
            )
            + DEPENDENTS
        )
        (change,) = env.plan_changes()
        assert isinstance(change, DropAndCreateView)
        env.apply()

        for v in ("people_set", "people_adder", "people_picker"):
            assert env.can_select(v)
        assert env.plan_changes() == []


class TestViewReplaceable:
    def _v(self, *columns: str) -> View:
        return View(schema="public", name="v", definition="", columns=list(columns))

    def test_same_or_appended_columns_qualify(self):
        assert _view_replaceable(self._v("a int"), self._v("a int"))
        assert _view_replaceable(self._v("a int"), self._v("a int", "b text"))

    def test_renamed_retyped_reordered_or_removed_do_not(self):
        current = self._v("a int", "b text")
        assert not _view_replaceable(current, self._v("x int", "b text"))
        assert not _view_replaceable(current, self._v("a bigint", "b text"))
        assert not _view_replaceable(current, self._v("b text", "a int"))
        assert not _view_replaceable(current, self._v("a int"))
        assert not _view_replaceable(current, self._v("a int", 'b text COLLATE "C"'))

    def test_unknown_columns_do_not(self):
        assert not _view_replaceable(self._v(), self._v("a int"))
        assert not _view_replaceable(self._v("a int"), self._v())
