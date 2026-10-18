"""COMMENT ON: declared, changed, removed, and kept across recreates.

The unit tests drive the differ and the rendered SQL. The integration tests run
the whole pipeline on a real database (they need TEST_DATABASE_URL), so every
apply also passes the engine's own convergence check.
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
    DropAndCreateView,
    SetComment,
)
from pgdeclare.differ import split_index_operations
from pgdeclare.models import (
    Column,
    Comment,
    Constraint,
    Function,
    Schema,
    Table,
    View,
)


# =============================================================================
# Unit tests
# =============================================================================


def _table_schema(*comments: Comment) -> Schema:
    table = Table(
        schema="public",
        name="note",
        columns={"body": Column(name="body", type="text", nullable=False)},
        constraints=[Constraint(name="note_pkey", type="p", definition="PRIMARY KEY")],
    )
    schema = Schema(tables={"public.note": table})
    schema.comments = {c.key: c for c in comments}
    return schema


def _comment_changes(changes) -> list[SetComment]:
    return [c for c in changes if isinstance(c, SetComment)]


class TestCommentModel:
    @pytest.mark.parametrize(
        ("comment", "target"),
        [
            (Comment("TABLE", "public", "note", "t"), '"public"."note"'),
            (Comment("VIEW", "app", "v", "t"), '"app"."v"'),
            (Comment("INDEX", "public", "note_idx", "t"), '"public"."note_idx"'),
            (Comment("TYPE", "public", "mood", "t"), '"public"."mood"'),
            (
                Comment("COLUMN", "public", "note", "t", member="body"),
                '"public"."note"."body"',
            ),
            (
                Comment("CONSTRAINT", "public", "note", "t", member="note_pkey"),
                '"note_pkey" ON "public"."note"',
            ),
            (
                Comment("TRIGGER", "public", "note", "t", member="note_trg"),
                '"note_trg" ON "public"."note"',
            ),
            (
                Comment("POLICY", "public", "note", "t", member="note_pol"),
                '"note_pol" ON "public"."note"',
            ),
            (
                Comment("FUNCTION", "public", "f", "t", args="a integer, b text"),
                '"public"."f"(a integer, b text)',
            ),
        ],
    )
    def test_target_sql(self, comment, target):
        assert comment.target_sql == f"{comment.object_type} {target}"

    def test_keys_distinguish_members_and_overloads(self):
        keys = {
            Comment("COLUMN", "public", "note", "", member="a").key,
            Comment("COLUMN", "public", "note", "", member="b").key,
            Comment("FUNCTION", "public", "f", "", args="integer").key,
            Comment("FUNCTION", "public", "f", "", args="text").key,
            Comment("TABLE", "public", "note", "").key,
        }
        assert len(keys) == 5

    def test_names_and_text_are_quoted(self):
        """Names are data: a quote in a name or the text cannot end the token."""
        comment = Comment(
            "COLUMN", 'we"ird', 'ta"ble', "it's'; DROP TABLE x; --", member='c"ol'
        )
        sql = SetComment(comment=comment).to_sql()
        assert sql == (
            'COMMENT ON COLUMN "we""ird"."ta""ble"."c""ol" '
            "IS 'it''s''; DROP TABLE x; --';"
        )

    def test_remove_renders_is_null(self):
        comment = Comment("TABLE", "public", "note", "old text")
        change = SetComment(comment=comment, remove=True)
        assert change.to_sql() == 'COMMENT ON TABLE "public"."note" IS NULL;'
        assert change.description == "Remove comment on table public.note"


class TestCommentDiff:
    def test_declared_comment_is_created(self):
        comment = Comment("TABLE", "public", "note", "Notes.")
        changes = compute_diff(_table_schema(comment), _table_schema())
        assert [(c.comment, c.remove) for c in _comment_changes(changes)] == [
            (comment, False)
        ]

    def test_same_text_is_no_change(self):
        comment = Comment("COLUMN", "public", "note", "Body.", member="body")
        assert compute_diff(_table_schema(comment), _table_schema(comment)) == []

    def test_changed_text_is_updated(self):
        old = Comment("CONSTRAINT", "public", "note", "old", member="note_pkey")
        new = Comment("CONSTRAINT", "public", "note", "new", member="note_pkey")
        [change] = compute_diff(_table_schema(new), _table_schema(old))
        assert isinstance(change, SetComment)
        assert not change.remove
        assert "IS 'new'" in change.to_sql()

    def test_undeclared_comment_is_removed(self):
        old = Comment("TABLE", "public", "note", "Notes.")
        [change] = compute_diff(_table_schema(), _table_schema(old))
        assert isinstance(change, SetComment) and change.remove

    def test_comment_on_a_dropped_object_is_left_to_the_drop(self):
        """Dropping the column drops its comment; IS NULL on it would fail."""
        actual = _table_schema(
            Comment("COLUMN", "public", "note", "Gone.", member="legacy")
        )
        actual.tables["public.note"].columns["legacy"] = Column(
            name="legacy", type="text", nullable=True
        )
        changes = compute_diff(_table_schema(), actual)
        assert _comment_changes(changes) == []

    def test_recreated_view_gets_its_comments_back(self):
        """A view recreate drops its comments; the snapshot predates the drop."""
        view_comment = Comment("VIEW", "public", "v", "A view.")
        column_comment = Comment("COLUMN", "public", "v", "A column.", member="a")
        desired = Schema(views={"public.v": View("public", "v", " SELECT 2 AS a;")})
        actual = Schema(views={"public.v": View("public", "v", " SELECT 1 AS a;")})
        for schema in (desired, actual):
            schema.comments = {c.key: c for c in (view_comment, column_comment)}

        changes = compute_diff(desired, actual)

        assert any(isinstance(c, DropAndCreateView) for c in changes)
        assert {c.comment.key for c in _comment_changes(changes)} == {
            view_comment.key,
            column_comment.key,
        }

    def test_recreated_function_gets_its_comment_back(self):
        def func(returns: str) -> Function:
            definition = (
                f"CREATE OR REPLACE FUNCTION public.f() RETURNS {returns} "
                "LANGUAGE sql AS $$ SELECT 1 $$"
            )
            return Function("public", "f", "", definition, definition_hash=returns)

        comment = Comment("FUNCTION", "public", "f", "Answer.", args="")
        desired = Schema(functions={"public.f()": func("bigint")})
        actual = Schema(functions={"public.f()": func("integer")})
        desired.comments = actual.comments = {comment.key: comment}

        changes = compute_diff(desired, actual)

        assert [c.comment for c in _comment_changes(changes)] == [comment]

    def test_removed_comment_on_recreated_view_is_not_nulled(self):
        old = Comment("VIEW", "public", "v", "A view.")
        desired = Schema(views={"public.v": View("public", "v", " SELECT 2 AS a;")})
        actual = Schema(views={"public.v": View("public", "v", " SELECT 1 AS a;")})
        actual.comments = {old.key: old}
        assert _comment_changes(compute_diff(desired, actual)) == []

    def test_comments_stay_out_of_the_ddl_and_index_phases(self):
        comment = Comment("TABLE", "public", "note", "Notes.")
        changes = compute_diff(_table_schema(comment), Schema())
        ddl, index = split_index_operations(
            [c for c in changes if not isinstance(c, SetComment)]
        )
        assert not any(isinstance(c, SetComment) for c in ddl + index)
        # The comment comes after the CREATE TABLE in the plan as well.
        assert isinstance(changes[-1], SetComment)


# =============================================================================
# Integration tests: the full pipeline
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


@pytest.fixture
def databases(server_url):
    """A target database and a plan database, both empty, both dropped after."""
    suffix = uuid.uuid4().hex[:8]
    target, plan = f"pgdeclare_target_{suffix}", f"pgdeclare_plan_{suffix}"
    for name in (target, plan):
        _admin(server_url, f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(server_url, target), plan
    finally:
        for name in (target, plan):
            _admin(server_url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _write(sql_dir: Path, content: str) -> None:
    schemas = sql_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "all.sql").write_text(content)


def _comments(url: str) -> dict[str, str]:
    return {k: c.text for k, c in introspect_schema(url).comments.items()}


def _apply(sql_dir: Path, target_url: str, plan_db: str, content: str) -> None:
    """Apply `content`, then prove a fresh diff against it is empty."""
    config = PgDeclareConfig(report=lambda _line: None)
    _write(sql_dir, content)
    apply_declared_schema(target_url, sql_dir, plan_db, config=config)
    desired = build_desired_schema(sql_dir, target_url, plan_db, config=config)
    actual = introspect_schema(target_url, config=config)
    assert compute_diff(desired, actual, schemas=["public"]) == []


OBJECTS = """
CREATE TYPE mood AS ENUM ('calm', 'loud');
CREATE TYPE pair AS (a integer, b integer);

CREATE TABLE note (
    id integer PRIMARY KEY,
    body text NOT NULL,
    mood mood NOT NULL DEFAULT 'calm',
    CONSTRAINT body_not_empty CHECK (body <> '')
);
CREATE INDEX note_body_idx ON note ({index_expr});
ALTER TABLE note ENABLE ROW LEVEL SECURITY;
CREATE POLICY note_read ON note FOR SELECT USING ({policy_expr});

CREATE FUNCTION answer() RETURNS {answer_type} LANGUAGE sql AS $$ SELECT 42 $$;
CREATE FUNCTION touch() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RETURN NEW; END $$;
CREATE TRIGGER note_touch BEFORE {trigger_event} ON note
    FOR EACH ROW EXECUTE FUNCTION touch();

CREATE VIEW note_view AS SELECT {view_columns} FROM note;
"""

SHAPE_V1 = {
    "index_expr": "body",
    "policy_expr": "true",
    "answer_type": "integer",
    "trigger_event": "INSERT",
    "view_columns": "id, body",
}

# Every recreate path at once: index, policy, function return type, trigger,
# and a view whose column list changes (DROP + CREATE, not CREATE OR REPLACE).
SHAPE_V2 = {
    "index_expr": "lower(body)",
    "policy_expr": "id > 0",
    "answer_type": "bigint",
    "trigger_event": "UPDATE",
    "view_columns": "body, id",
}

COMMENTS_V1 = """
COMMENT ON TABLE note IS 'Notes people write.';
COMMENT ON COLUMN note.body IS 'The text; it''s never empty.';
COMMENT ON CONSTRAINT body_not_empty ON note IS 'No blank notes.';
COMMENT ON INDEX note_body_idx IS 'Search by body.';
COMMENT ON POLICY note_read ON note IS 'Everyone reads.';
COMMENT ON FUNCTION answer() IS 'The answer.';
COMMENT ON TRIGGER note_touch ON note IS 'Touches rows.';
COMMENT ON VIEW note_view IS 'Notes, viewed.';
COMMENT ON COLUMN note_view.body IS 'Body, viewed.';
COMMENT ON TYPE mood IS 'How a note feels.';
COMMENT ON TYPE pair IS 'Two integers.';
"""

EXPECTED_V1 = {
    "TABLE:public.note": "Notes people write.",
    "COLUMN:public.note.body": "The text; it's never empty.",
    "CONSTRAINT:public.note.body_not_empty": "No blank notes.",
    "INDEX:public.note_body_idx": "Search by body.",
    "POLICY:public.note.note_read": "Everyone reads.",
    "FUNCTION:public.answer()": "The answer.",
    "TRIGGER:public.note.note_touch": "Touches rows.",
    "VIEW:public.note_view": "Notes, viewed.",
    "COLUMN:public.note_view.body": "Body, viewed.",
    "TYPE:public.mood": "How a note feels.",
    "TYPE:public.pair": "Two integers.",
}


def test_comments_are_created_changed_and_removed(tmp_path, databases):
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"
    objects = OBJECTS.format(**SHAPE_V1)

    # Declared on a fresh database.
    _apply(sql_dir, target_url, plan_db, objects + COMMENTS_V1)
    assert _comments(target_url) == EXPECTED_V1

    # Text changed on some, dropped from others.
    changed = (
        COMMENTS_V1.replace("Notes people write.", "Notes, revised.")
        .replace("'The answer.'", "'Still the answer.'")
        .replace("COMMENT ON TYPE pair IS 'Two integers.';\n", "")
        .replace("COMMENT ON INDEX note_body_idx IS 'Search by body.';\n", "")
    )
    _apply(sql_dir, target_url, plan_db, objects + changed)
    expected = dict(EXPECTED_V1)
    expected["TABLE:public.note"] = "Notes, revised."
    expected["FUNCTION:public.answer()"] = "Still the answer."
    del expected["TYPE:public.pair"]
    del expected["INDEX:public.note_body_idx"]
    assert _comments(target_url) == expected

    # Every declaration removed.
    _apply(sql_dir, target_url, plan_db, objects)
    assert _comments(target_url) == {}


def test_recreated_objects_keep_their_comments(tmp_path, databases):
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"

    _apply(sql_dir, target_url, plan_db, OBJECTS.format(**SHAPE_V1) + COMMENTS_V1)
    assert _comments(target_url) == EXPECTED_V1

    # Same comments, but the index, policy, function, trigger and view are all
    # dropped and recreated. Each comment must be written back.
    _apply(sql_dir, target_url, plan_db, OBJECTS.format(**SHAPE_V2) + COMMENTS_V1)
    assert _comments(target_url) == EXPECTED_V1
    assert introspect_schema(target_url).functions["public.answer()"].full_definition


def test_live_only_comment_is_removed(tmp_path, databases):
    """A comment added by hand is drift: the schema files are the truth."""
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"
    objects = OBJECTS.format(**SHAPE_V1)

    _apply(sql_dir, target_url, plan_db, objects)
    _admin(target_url, "COMMENT ON TABLE note IS 'added by hand'")
    assert _comments(target_url) == {"TABLE:public.note": "added by hand"}

    _apply(sql_dir, target_url, plan_db, objects)
    assert _comments(target_url) == {}


def test_hostile_names_and_text_round_trip(tmp_path, databases):
    """Names are data: quotes and semicolons in names and text stay inert."""
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"
    schema = """
CREATE TABLE "we""ird; x" ("c""ol" text);
COMMENT ON TABLE "we""ird; x" IS 'it''s''; DROP TABLE "we""ird; x"; --';
COMMENT ON COLUMN "we""ird; x"."c""ol" IS $$dollar $ and \\ backslash$$;
"""
    _apply(sql_dir, target_url, plan_db, schema)
    assert _comments(target_url) == {
        'TABLE:public.we"ird; x': 'it\'s\'; DROP TABLE "we""ird; x"; --',
        'COLUMN:public.we"ird; x.c"ol': "dollar $ and \\ backslash",
    }

    _apply(sql_dir, target_url, plan_db, schema.replace("dollar", "euro"))
    assert _comments(target_url)['COLUMN:public.we"ird; x.c"ol'] == (
        "euro $ and \\ backslash"
    )


def test_comment_on_a_new_column_of_an_existing_table(tmp_path, databases):
    target_url, plan_db = databases
    sql_dir = tmp_path / "sql"
    _apply(sql_dir, target_url, plan_db, "CREATE TABLE t (id int);")
    _apply(
        sql_dir,
        target_url,
        plan_db,
        "CREATE TABLE t (id int, label text);\n"
        "COMMENT ON COLUMN t.label IS 'Shown to people.';",
    )
    assert _comments(target_url) == {"COLUMN:public.t.label": "Shown to people."}
