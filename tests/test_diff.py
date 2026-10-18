"""
Tests for schema diff engine.

Covers:
- Object model (dataclasses)
- Introspection (pg_catalog queries)
- Change detection (diffing)
- SQL generation
- @drop enforcement
- @migrate-before tracking
"""

from typing import Any


import pytest

from pgdeclare.models import (
    ChangeCategory,
    Column,
    CompositeType,
    Constraint,
    EnumType,
    Function,
    Grant,
    Index,
    Policy,
    Role,
    RoleMembership,
    Schema,
    Table,
    Trigger,
    View,
)
from pgdeclare.changes import (
    AddColumn,
    AddConstraint,
    AddEnumValue,
    AlterColumnDropExpression,
    AlterColumnSetExpression,
    AlterColumnType,
    AlterRole,
    AlterViewOptions,
    CreateCompositeType,
    CreateEnum,
    CreateFunction,
    CreateIndex,
    CreatePolicy,
    CreateRole,
    CreateSchema,
    CreateTable,
    CreateTrigger,
    CreateView,
    DropColumn,
    DropConstraint,
    DropDefault,
    DropEnumValue,
    DropIndex,
    DropNotNull,
    DropPolicy,
    DropTable,
    DropTrigger,
    DropView,
    DropAndCreateConstraint,
    DropAndCreateFunction,
    DropAndCreatePolicy,
    DropAndCreateView,
    EnableRLS,
    DisableRLS,
    GrantPrivilege,
    GrantRoleMembership,
    ReplaceFunction,
    RevokePrivilege,
    RevokePublicExecute,
    SetDefault,
    SetNotNull,
    ValidateConstraint,
)
from pgdeclare.differ import (
    compute_diff,
    categorize_changes,
    split_index_operations,
)
from pgdeclare.drop_directive import (
    validate_drops_ci,
)
from pgdeclare import parse_migration_blocks


class TestModels:
    """Unit tests for object model dataclasses."""

    def test_column_equality(self):
        """Columns with same properties are equal."""
        col1 = Column(
            name="id", type="uuid", nullable=False, default="gen_random_uuid()"
        )
        col2 = Column(
            name="id", type="uuid", nullable=False, default="gen_random_uuid()"
        )
        assert col1 == col2

    def test_column_inequality_type(self):
        """Columns with different types are not equal."""
        col1 = Column(name="id", type="uuid", nullable=False)
        col2 = Column(name="id", type="text", nullable=False)
        assert col1 != col2

    def test_table_qualified_name(self):
        """Table qualified name includes schema."""
        table = Table(schema="public", name="account")
        assert table.qualified_name == "public.account"

    def test_function_signature(self):
        """Function signature includes schema, name, and args."""
        func = Function(
            schema="app_generated",
            name="list_accounts",
            args="p_cursor text, p_limit integer",
            full_definition="CREATE FUNCTION...",
        )
        assert (
            func.signature
            == "app_generated.list_accounts(p_cursor text, p_limit integer)"
        )

    def test_policy_qualified_key(self):
        """Policy key includes schema, table, and name."""
        policy = Policy(
            name="account_select",
            table="account",
            schema="public",
            command="SELECT",
        )
        assert policy.qualified_key == "public.account.account_select"

    def test_grant_key_with_column(self):
        """Grant key includes column for column-level grants."""
        grant = Grant(
            object_type="COLUMN",
            object_name="account",
            schema="public",
            privilege="SELECT",
            grantee="app_user",
            column="email",
        )
        assert "email" in grant.key

    def test_schema_get_table(self):
        """Schema.get_table retrieves by schema.name."""
        table = Table(schema="public", name="account")
        schema = Schema(tables={"public.account": table})
        assert schema.get_table("public", "account") == table
        assert schema.get_table("public", "nonexistent") is None


class TestChanges:
    """Unit tests for Change classes."""

    def test_create_table_sql(self):
        """CreateTable generates valid CREATE TABLE SQL."""
        table = Table(
            schema="public",
            name="test_table",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=True),
            },
        )
        change = CreateTable(table=table)
        sql = change.to_sql()

        assert 'CREATE TABLE "public"."test_table"' in sql
        assert '"id" uuid NOT NULL DEFAULT gen_random_uuid()' in sql
        assert '"name" text' in sql

    def test_create_table_excludes_fk_constraints(self):
        """CreateTable excludes FK constraints from inline SQL for circular dependency handling.

        FK constraints are added as separate AddConstraint changes by the differ,
        allowing retry-until-stable to resolve circular dependencies like:
        post -> comment -> post
        """
        table = Table(
            schema="public",
            name="post",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "last_comment_id": Column(
                    name="last_comment_id", type="uuid", nullable=True
                ),
            },
            constraints=[
                # Primary key - should be included inline
                Constraint(
                    name="post_pkey",
                    type="p",  # Primary key
                    definition="PRIMARY KEY (id)",
                ),
                # FK to comment - should be EXCLUDED (circular dep)
                Constraint(
                    name="post_last_comment_fkey",
                    type="f",  # Foreign key
                    definition="FOREIGN KEY (last_comment_id) REFERENCES comment(id)",
                ),
            ],
        )
        change = CreateTable(table=table)
        sql = change.to_sql()

        # PK constraint should be included inline
        assert 'CONSTRAINT "post_pkey" PRIMARY KEY (id)' in sql

        # FK constraint should NOT be included inline
        assert "post_last_comment_fkey" not in sql
        assert "REFERENCES comment" not in sql

    def test_create_table_defers_unique_constraints(self):
        """CreateTable must not inline UNIQUE constraints.

        PostgreSQL's transformIndexConstraints() deduplicates the inline index
        constraint list by column set, keeping the primary key and silently
        discarding the rest. `enrollment` declares a UNIQUE on the same pair as its
        primary key, so an inlined CREATE TABLE produced a database one
        constraint short of the declaration, with no error and no notice. The
        differ emits UNIQUE (and EXCLUDE) as separate AddConstraint changes;
        ALTER TABLE ADD CONSTRAINT does no such deduplication.
        """
        table = Table(
            schema="public",
            name="enrollment",
            columns={
                "course_id": Column(name="course_id", type="uuid", nullable=False),
                "student_id": Column(name="student_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="enrollment_pkey",
                    type="p",
                    definition="PRIMARY KEY (course_id, student_id)",
                ),
                Constraint(
                    name="enrollment_course_id_student_id_key",
                    type="u",
                    definition="UNIQUE (course_id, student_id)",
                ),
            ],
        )
        sql = CreateTable(table=table).to_sql()

        assert 'CONSTRAINT "enrollment_pkey" PRIMARY KEY (course_id, student_id)' in sql
        assert "enrollment_course_id_student_id_key" not in sql

    def test_differ_adds_unique_constraints_for_new_tables(self):
        """The UNIQUE that CreateTable defers must be emitted as AddConstraint."""
        table = Table(
            schema="public",
            name="enrollment",
            columns={
                "course_id": Column(name="course_id", type="uuid", nullable=False),
                "student_id": Column(name="student_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="enrollment_pkey",
                    type="p",
                    definition="PRIMARY KEY (course_id, student_id)",
                ),
                Constraint(
                    name="enrollment_course_id_student_id_key",
                    type="u",
                    definition="UNIQUE (course_id, student_id)",
                ),
            ],
        )
        changes = compute_diff(Schema(tables={"public.enrollment": table}), Schema())

        added = {c.constraint.name for c in changes if isinstance(c, AddConstraint)}
        assert "enrollment_course_id_student_id_key" in added
        # The primary key stays inline; it is never deduplicated away.
        assert "enrollment_pkey" not in added

    def test_add_column_sql(self):
        """AddColumn generates valid ALTER TABLE ADD COLUMN SQL."""
        col = Column(
            name="created_at", type="timestamptz", nullable=False, default="NOW()"
        )
        change = AddColumn(schema="public", table="account", column=col)
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account" ADD COLUMN "created_at"' in sql
        assert "timestamptz NOT NULL DEFAULT NOW()" in sql

    def test_drop_column_requires_drop(self):
        """DropColumn with data requires @drop directive."""
        change = DropColumn(
            schema="public",
            table="account",
            column_name="legacy_field",
            has_data=True,
        )
        assert change.requires_drop_directive is True
        assert change.category == ChangeCategory.BLOCKED

    def test_set_not_null_is_offline(self):
        """SetNotNull is an offline operation."""
        change = SetNotNull(schema="public", table="account", column="email")
        assert change.category == ChangeCategory.OFFLINE

    def test_create_index_concurrent(self):
        """CreateIndex with concurrent=True generates CONCURRENTLY SQL."""
        index = Index(
            name="idx_account_email",
            table="account",
            schema="public",
            definition="CREATE INDEX idx_account_email ON public.account(email)",
        )
        change = CreateIndex(index=index, concurrent=True)
        sql = change.to_sql()

        assert "CREATE INDEX CONCURRENTLY" in sql

    def test_create_function_is_online(self):
        """CreateFunction is an online operation."""
        func = Function(
            schema="app_generated",
            name="list_accounts",
            args="",
            full_definition="CREATE OR REPLACE FUNCTION app_generated.list_accounts() RETURNS TABLE...",
        )
        change = CreateFunction(function=func)
        assert change.category == ChangeCategory.ONLINE


class TestDiffer:
    """Unit tests for compute_diff and helpers."""

    def test_diff_add_table(self):
        """Detect table addition."""
        desired = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
                "public.member": Table(schema="public", name="customer"),
            }
        )
        actual = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
            }
        )

        changes = compute_diff(desired, actual)

        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 1
        assert create_tables[0].table.name == "customer"

    def test_diff_circular_fk_generates_separate_add_constraint(self):
        """Circular FK dependencies generate separate AddConstraint changes.

        When creating tables with circular FK references (like post <-> comment),
        the differ should generate:
        1. CreateTable without FK constraints
        2. Separate AddConstraint for each FK

        This allows retry-until-stable to execute in the correct order:
        1. Create both tables (no FK yet)
        2. Add FK constraints (tables now exist)

        This test replicates the CI failure case where post and comment
        tables have mutual FK references.
        """
        # Two tables with circular FK references
        post = Table(
            schema="public",
            name="post",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "last_comment_id": Column(
                    name="last_comment_id", type="uuid", nullable=True
                ),
            },
            constraints=[
                Constraint(name="post_pkey", type="p", definition="PRIMARY KEY (id)"),
                Constraint(
                    name="post_last_comment_fkey",
                    type="f",  # FK to comment
                    definition="FOREIGN KEY (last_comment_id) REFERENCES comment(id)",
                ),
            ],
        )
        comment = Table(
            schema="public",
            name="comment",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "post_id": Column(name="post_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="comment_pkey", type="p", definition="PRIMARY KEY (id)"
                ),
                Constraint(
                    name="comment_post_fkey",
                    type="f",  # FK to post
                    definition="FOREIGN KEY (post_id) REFERENCES post(id)",
                ),
            ],
        )

        desired = Schema(
            tables={
                "public.post": post,
                "public.comment": comment,
            }
        )
        actual = Schema()  # Empty database

        changes = compute_diff(desired, actual)

        # Should have 2 CreateTable + 2 AddConstraint (for FKs)
        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        add_constraints = [c for c in changes if isinstance(c, AddConstraint)]

        assert len(create_tables) == 2
        assert len(add_constraints) == 2

        # CreateTable SQL should NOT include FK constraints
        for ct in create_tables:
            sql = ct.to_sql()
            assert "REFERENCES" not in sql, f"FK should not be inline: {sql}"

        # AddConstraint should include the FK constraints
        fk_names = {ac.constraint.name for ac in add_constraints}
        assert "post_last_comment_fkey" in fk_names
        assert "comment_post_fkey" in fk_names

    def test_diff_drop_table(self):
        """Detect table removal."""
        desired = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
            }
        )
        actual = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
                "public.legacy": Table(schema="public", name="legacy"),
            }
        )

        changes = compute_diff(desired, actual)

        drop_tables = [c for c in changes if isinstance(c, DropTable)]
        assert len(drop_tables) == 1
        assert drop_tables[0].table_name == "legacy"

    def test_diff_add_column(self):
        """Detect column addition."""
        desired = Schema(
            tables={
                "public.account": Table(
                    schema="public",
                    name="account",
                    columns={
                        "id": Column(name="id", type="uuid", nullable=False),
                        "email": Column(name="email", type="text", nullable=False),
                    },
                ),
            }
        )
        actual = Schema(
            tables={
                "public.account": Table(
                    schema="public",
                    name="account",
                    columns={
                        "id": Column(name="id", type="uuid", nullable=False),
                    },
                ),
            }
        )

        changes = compute_diff(desired, actual)

        add_columns = [c for c in changes if isinstance(c, AddColumn)]
        assert len(add_columns) == 1
        assert add_columns[0].column.name == "email"

    def test_diff_function_change(self):
        """Detect function modification."""
        desired = Schema(
            functions={
                "public.my_func()": Function(
                    schema="public",
                    name="my_func",
                    args="",
                    full_definition="CREATE FUNCTION... v2",
                    definition_hash="new_hash",
                ),
            }
        )
        actual = Schema(
            functions={
                "public.my_func()": Function(
                    schema="public",
                    name="my_func",
                    args="",
                    full_definition="CREATE FUNCTION... v1",
                    definition_hash="old_hash",
                ),
            }
        )

        changes = compute_diff(desired, actual)

        replace_funcs = [c for c in changes if isinstance(c, ReplaceFunction)]
        assert len(replace_funcs) == 1

    def test_categorize_changes(self):
        """Group changes by category."""
        changes = [
            AddColumn(
                schema="public",
                table="t",
                column=Column(name="c", type="text", nullable=True),
            ),
            SetNotNull(schema="public", table="t", column="c"),
            DropColumn(schema="public", table="t", column_name="old", has_data=True),
        ]

        categorized = categorize_changes(changes)

        assert len(categorized[ChangeCategory.ONLINE]) == 1
        assert len(categorized[ChangeCategory.OFFLINE]) == 1
        assert len(categorized[ChangeCategory.BLOCKED]) == 1

    def test_split_index_operations(self):
        """Separate concurrent index ops from DDL."""
        index = Index(
            name="idx", table="t", schema="public", definition="CREATE INDEX..."
        )
        changes = [
            AddColumn(
                schema="public",
                table="t",
                column=Column(name="c", type="text", nullable=True),
            ),
            CreateIndex(index=index, concurrent=True),
            DropIndex(schema="public", index_name="old_idx", concurrent=True),
        ]

        ddl, index_ops = split_index_operations(changes)

        assert len(ddl) == 1
        assert len(index_ops) == 2

    def test_compute_diff_schedules_validate_constraint(self):
        """AddConstraint with NOT VALID schedules ValidateConstraint."""
        from pgdeclare.models import Constraint
        from pgdeclare.changes import (
            AddConstraint,
            ValidateConstraint,
        )

        # Create tables with FK constraint
        parent = Table(schema="public", name="parent")
        child = Table(
            schema="public",
            name="child",
            constraints=[
                Constraint(
                    name="fk_parent",
                    type="f",  # Foreign key
                    definition="FOREIGN KEY (parent_id) REFERENCES parent(id)",
                )
            ],
        )

        desired = Schema(
            tables={
                "public.parent": parent,
                "public.child": child,
            }
        )
        actual = Schema(
            tables={
                "public.parent": parent,
                "public.child": Table(schema="public", name="child"),  # No constraint
            }
        )

        changes = compute_diff(desired, actual)

        # Should have AddConstraint and ValidateConstraint
        add_constraints = [c for c in changes if isinstance(c, AddConstraint)]
        validate_constraints = [c for c in changes if isinstance(c, ValidateConstraint)]

        assert len(add_constraints) == 1
        assert len(validate_constraints) == 1
        assert validate_constraints[0].constraint_name == "fk_parent"

    def test_compute_diff_validates_recreated_check_constraint(self):
        """A tightened CHECK is re-added NOT VALID, so it must be validated too.

        Without the follow-up VALIDATE the constraint stays NOT VALID,
        pg_get_constraintdef keeps reporting the suffix, and the next diff
        re-emits the same drop/recreate forever.
        """
        from pgdeclare.models import Constraint
        from pgdeclare.changes import (
            DropAndCreateConstraint,
            ValidateConstraint,
        )

        def induction(upper_bound: int) -> Table:
            return Table(
                schema="public",
                name="induction",
                constraints=[
                    Constraint(
                        name="induction_offer_expiration_days_range",
                        type="c",
                        definition=(
                            "CHECK ((offer_expiration_days >= 1) "
                            f"AND (offer_expiration_days <= {upper_bound}))"
                        ),
                    )
                ],
            )

        desired = Schema(tables={"public.induction": induction(90)})
        actual = Schema(tables={"public.induction": induction(365)})

        changes = compute_diff(desired, actual)

        recreates = [c for c in changes if isinstance(c, DropAndCreateConstraint)]
        validates = [c for c in changes if isinstance(c, ValidateConstraint)]

        assert len(recreates) == 1
        assert len(validates) == 1
        assert validates[0].constraint_name == "induction_offer_expiration_days_range"

    def test_recreated_view_carries_its_grants(self):
        """A view recreate owns its ACL replay, not a separate GRANT.

        Before the fix the differ appended a standalone GrantPrivilege after the
        recreate. The executor runs changes retry-until-stable, so a recreate
        deferred behind a dependency built later in the same run landed AFTER
        that GRANT, and the DROP threw the ACL away. Carrying the grants inside
        the recreate makes the ordering irrelevant.
        """
        grant = Grant(
            object_type="TABLE",
            object_name="headlines",
            schema="public",
            privilege="SELECT",
            grantee="app_user",
        )
        desired = Schema(
            views={
                "public.headlines": View(
                    schema="public",
                    name="headlines",
                    definition="SELECT id, title, subtitle FROM post",
                )
            },
            grants=[grant],
        )
        actual = Schema(
            views={
                "public.headlines": View(
                    schema="public",
                    name="headlines",
                    definition="SELECT id, title FROM post",
                )
            },
            grants=[grant],
        )

        changes = compute_diff(desired, actual)

        recreates = [c for c in changes if isinstance(c, DropAndCreateView)]
        assert len(recreates) == 1
        assert [g.key for g in recreates[0].grants] == [grant.key]
        # The replay is inside the recreate's own statement.
        assert "TO app_user" in recreates[0].to_sql()

    def test_recreated_function_carries_its_grants(self):
        """Same for a function recreate: DROP wipes EXECUTE grants too."""
        grant = Grant(
            object_type="FUNCTION",
            object_name="count_reviews",
            schema="public",
            privilege="EXECUTE",
            grantee="app_user",
        )
        func_old = Function(
            schema="public",
            name="count_reviews",
            args="",
            full_definition=(
                "CREATE FUNCTION count_reviews() RETURNS integer AS $$ "
                "SELECT 1 $$ LANGUAGE sql"
            ),
            definition_hash="old",
        )
        func_new = Function(
            schema="public",
            name="count_reviews",
            args="",
            full_definition=(
                "CREATE FUNCTION count_reviews() RETURNS bigint AS $$ "
                "SELECT 1::bigint $$ LANGUAGE sql"
            ),
            definition_hash="new",
        )
        desired = Schema(functions={"public.count_reviews()": func_new}, grants=[grant])
        actual = Schema(functions={"public.count_reviews()": func_old}, grants=[grant])

        changes = compute_diff(desired, actual)

        recreates = [c for c in changes if isinstance(c, DropAndCreateFunction)]
        assert len(recreates) == 1
        assert [g.key for g in recreates[0].grants] == [grant.key]
        assert "TO app_user" in recreates[0].to_sql()

    def test_view_security_invoker_drift_is_detected_and_repaired(self):
        """A view whose only drift is a lost security_invoker converges.

        Every database built before this fix is in exactly this state: the
        definition matches the declaration, but the differ never wrote the
        `WITH (security_invoker = TRUE)`, so `pg_class.reloptions` is empty and
        the view runs with owner rights. The owner is a BYPASSRLS superuser, so
        RLS on the base tables never applies to anything read through the view.
        The repair is an in-place ALTER, not a recreate: dependent views stay put.
        """
        definition = "SELECT id, content FROM comment"
        desired = Schema(
            views={
                "public.post_comment": View(
                    schema="public",
                    name="post_comment",
                    definition=definition,
                    reloptions=["security_invoker=true"],
                )
            }
        )
        actual = Schema(
            views={
                "public.post_comment": View(
                    schema="public",
                    name="post_comment",
                    definition=definition,
                )
            }
        )

        changes = compute_diff(desired, actual)

        alters = [c for c in changes if isinstance(c, AlterViewOptions)]
        assert len(alters) == 1
        assert not [c for c in changes if isinstance(c, DropAndCreateView)]
        assert (
            alters[0].to_sql()
            == 'ALTER VIEW "public"."post_comment" SET (security_invoker=true);'
        )

    def test_view_options_matching_produces_no_change(self):
        """No drift, no churn. Formatting differences do not count as drift."""
        definition = "SELECT id FROM comment"
        desired = Schema(
            views={
                "public.v": View(
                    schema="public",
                    name="v",
                    definition=definition,
                    # As written in the schema files: WITH (security_invoker = TRUE)
                    reloptions=["security_invoker = TRUE"],
                )
            }
        )
        actual = Schema(
            views={
                "public.v": View(
                    schema="public",
                    name="v",
                    definition=definition,
                    # As reported by pg_class.reloptions
                    reloptions=["security_invoker=true"],
                )
            }
        )

        assert compute_diff(desired, actual) == []

    def test_view_options_removed_from_declaration_are_reset(self):
        """Dropping the declaration RESETs the option rather than leaving it on."""
        definition = "SELECT id FROM comment"
        desired = Schema(
            views={"public.v": View(schema="public", name="v", definition=definition)}
        )
        actual = Schema(
            views={
                "public.v": View(
                    schema="public",
                    name="v",
                    definition=definition,
                    reloptions=["security_invoker=true"],
                )
            }
        )

        changes = compute_diff(desired, actual)

        alters = [c for c in changes if isinstance(c, AlterViewOptions)]
        assert len(alters) == 1
        assert alters[0].to_sql() == 'ALTER VIEW "public"."v" RESET (security_invoker);'

    def test_recreated_view_carries_its_options(self):
        """A definition change keeps the security mode in the same statement.

        Same reasoning as the grant replay above: the retry-until-stable executor
        can defer a recreate, so a trailing standalone ALTER VIEW would land on
        the pre-drop view and be dropped with it.
        """
        desired = Schema(
            views={
                "public.take": View(
                    schema="public",
                    name="take",
                    definition="SELECT id, amount FROM customer",
                    reloptions=["security_invoker=true"],
                )
            }
        )
        actual = Schema(
            views={
                "public.take": View(
                    schema="public",
                    name="take",
                    definition="SELECT id FROM customer",
                    reloptions=["security_invoker=true"],
                )
            }
        )

        changes = compute_diff(desired, actual)

        recreates = [c for c in changes if isinstance(c, DropAndCreateView)]
        assert len(recreates) == 1
        sql = recreates[0].to_sql()
        assert 'CREATE VIEW "public"."take" WITH (security_invoker=true) AS' in sql
        assert not [c for c in changes if isinstance(c, AlterViewOptions)]

    def test_new_view_is_created_with_its_options(self):
        """A fresh install honours the declaration on the very first CREATE."""
        desired = Schema(
            views={
                "public.take": View(
                    schema="public",
                    name="take",
                    definition="SELECT id FROM customer",
                    reloptions=["security_invoker=true"],
                )
            }
        )

        changes = compute_diff(desired, Schema())

        creates = [c for c in changes if isinstance(c, CreateView)]
        assert len(creates) == 1
        assert creates[0].to_sql() == (
            'CREATE OR REPLACE VIEW "public"."take" WITH (security_invoker=true) '
            "AS SELECT id FROM customer"
        )

    def test_normalize_default_preserves_structure(self):
        """Default normalization preserves type casts (case-insensitive)."""
        from pgdeclare.differ import _normalize_default

        # Simple cast - should preserve structure
        assert _normalize_default("'hello'::TEXT") == "'hello'::text"

        # Complex expression - should NOT be truncated
        complex_default = "(now() + '7 days'::INTERVAL)"
        normalized = _normalize_default(complex_default)
        assert normalized is not None
        assert "interval" in normalized.lower()
        assert "now()" in normalized.lower()

        # NULL should stay None
        assert _normalize_default(None) is None


class TestDropDirective:
    """Tests for @drop directive parsing and enforcement."""

    def test_drop_directives_within_migration_blocks(self, tmp_path):
        """@drop directives within migration blocks are collected."""
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- @migrate-before:2025-12-08_test
-- @drop customer.legacy_status
-- @drop old_table
UPDATE customer SET legacy_status = NULL;
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)
        directives: set[str] = set()
        for block in blocks:
            directives.update(block.drop_directives)

        assert "customer.legacy_status" in directives
        assert "old_table" in directives

    def test_drop_directives_outside_blocks_ignored(self, tmp_path):
        """@drop directives outside migration blocks are NOT collected (scope enforcement)."""
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- This @drop is OUTSIDE a migration block and should be ignored
-- @drop rogue_directive

-- @migrate-before:2025-12-08_test
-- @drop valid_directive
UPDATE customer SET status = NULL;
-- @end-migrate

-- Another stray @drop that should be ignored
-- @drop another_rogue
""")

        blocks = parse_migration_blocks(tmp_path)
        directives: set[str] = set()
        for block in blocks:
            directives.update(block.drop_directives)

        # Only the directive within the block should be collected
        assert "valid_directive" in directives
        # Directives outside blocks should NOT be collected
        assert "rogue_directive" not in directives
        assert "another_rogue" not in directives

    def test_validate_drops_ci_detects_missing_directive(self):
        """CI validation catches drops without @drop."""
        source = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
            }
        )
        main_branch = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
                "public.legacy": Table(schema="public", name="legacy"),
            }
        )

        errors = validate_drops_ci(source, main_branch, set())

        assert len(errors) == 1
        assert "legacy" in errors[0]
        assert "@drop" in errors[0]

    def test_validate_drops_ci_passes_with_directive(self):
        """CI validation passes when @drop directive exists."""
        source = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
            }
        )
        main_branch = Schema(
            tables={
                "public.account": Table(schema="public", name="account"),
                "public.legacy": Table(schema="public", name="legacy"),
            }
        )

        errors = validate_drops_ci(source, main_branch, {"legacy", "public.legacy"})

        assert len(errors) == 0


class TestMigrations:
    """Tests for @migrate-before inline migrations."""

    def test_parse_migration_blocks(self, tmp_path):
        """Parse @migrate-before blocks from files."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "02_test.sql"
        sql_file.write_text("""
ALTER TABLE customer ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

-- @migrate-before:2025-12-08_backfill_archived
UPDATE customer SET archived_at = NOW() WHERE status = 'legacy' AND archived_at IS NULL;
-- @end-migrate

-- @migrate-before:2025-12-09_another
-- @drop customer.old_column
ALTER TABLE customer DROP COLUMN IF EXISTS old_column;
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)

        assert len(blocks) == 2
        assert blocks[0].id == "2025-12-08_backfill_archived"
        assert "UPDATE customer" in blocks[0].sql
        assert blocks[1].id == "2025-12-09_another"
        assert "customer.old_column" in blocks[1].drop_directives

    def test_parse_migration_blocks_recursive(self, tmp_path):
        """Parse @migrate-before blocks from files in subdirectories (regression test for recursive discovery)."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        # Create a file in the root schemas directory
        root_file = schemas_dir / "01_types.sql"
        root_file.write_text("""
-- @migrate-before:2025-12-01_root_migration
UPDATE settings SET version = 1;
-- @end-migrate
""")

        # Create a subdirectory with a migration block (like projects/core.sql)
        projects_dir = schemas_dir / "projects"
        projects_dir.mkdir()
        projects_file = projects_dir / "core.sql"
        projects_file.write_text("""
-- @migrate-before:2025-12-02_projects_migration
ALTER TYPE project_type RENAME VALUE 'old' TO 'new';
-- @end-migrate
""")

        # Create another subdirectory with a migration block
        teams_dir = schemas_dir / "teams"
        teams_dir.mkdir()
        teams_file = teams_dir / "core.sql"
        teams_file.write_text("""
-- @migrate-before:2025-12-03_teams_migration
UPDATE team SET status = 'active';
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)

        # Should find all 3 migration blocks from root and subdirectories
        assert len(blocks) == 3
        migration_ids = {b.id for b in blocks}
        assert "2025-12-01_root_migration" in migration_ids
        assert "2025-12-02_projects_migration" in migration_ids
        assert "2025-12-03_teams_migration" in migration_ids

    def test_migration_hash_changes_on_sql_change(self, tmp_path):
        """Migration hash changes when SQL content changes."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- @migrate-before:test_migration
UPDATE customer SET status = 'active';
-- @end-migrate
""")

        blocks1 = parse_migration_blocks(tmp_path)
        hash1 = blocks1[0].sql_hash

        # Change the SQL
        sql_file.write_text("""
-- @migrate-before:test_migration
UPDATE customer SET status = 'inactive';
-- @end-migrate
""")

        blocks2 = parse_migration_blocks(tmp_path)
        hash2 = blocks2[0].sql_hash

        assert hash1 != hash2

    def test_parse_migrate_before_phase(self, tmp_path):
        """@migrate-before blocks have phase='before'."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- @migrate-before:2026-01-01_rename_column
ALTER TABLE customer RENAME COLUMN old_name TO new_name;
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)

        assert len(blocks) == 1
        assert blocks[0].id == "2026-01-01_rename_column"
        assert blocks[0].phase == "before"

    def test_parse_migrate_after_phase(self, tmp_path):
        """@migrate-after blocks have phase='after'."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- @migrate-after:2026-03-08_backfill_new_col
UPDATE my_table SET new_col = 0 WHERE new_col IS NULL;
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)

        assert len(blocks) == 1
        assert blocks[0].id == "2026-03-08_backfill_new_col"
        assert blocks[0].phase == "after"

    def test_parse_mixed_phases(self, tmp_path):
        """Mixed @migrate-before and @migrate-after blocks are parsed with correct phases."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()

        sql_file = schemas_dir / "01_test.sql"
        sql_file.write_text("""
-- @migrate-before:2026-01-01_pre_rename
ALTER TABLE customer RENAME COLUMN old_name TO new_name;
-- @end-migrate

-- @migrate-after:2026-01-01_post_backfill
UPDATE customer SET new_col = default_value();
-- @end-migrate
""")

        blocks = parse_migration_blocks(tmp_path)

        assert len(blocks) == 2
        before_blocks = [b for b in blocks if b.phase == "before"]
        after_blocks = [b for b in blocks if b.phase == "after"]
        assert len(before_blocks) == 1
        assert len(after_blocks) == 1
        assert before_blocks[0].id == "2026-01-01_pre_rename"
        assert after_blocks[0].id == "2026-01-01_post_backfill"


# Integration tests (require database connection)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring a live database."""

    @pytest.fixture
    def connection_string(self):
        """Get connection string from environment or skip."""
        import os

        conn = os.environ.get("TEST_DATABASE_URL")
        if not conn:
            pytest.skip("TEST_DATABASE_URL not set")
        return conn

    def test_introspect_empty_database(self, connection_string):
        """Introspect an empty database."""
        from pgdeclare import introspect_schema

        schema = introspect_schema(connection_string, schemas=["public"])

        # Empty database should have no user tables
        # (may have system objects)
        assert isinstance(schema, Schema)

    def test_check_table_has_data(self, connection_string):
        """Check if table has data."""
        import psycopg2
        from pgdeclare.drop_directive import (
            check_table_has_data,
        )

        # Create a test table
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS _test_data_check")
                cur.execute("CREATE TABLE _test_data_check (id serial)")
            conn.commit()

        # Empty table
        assert (
            check_table_has_data(connection_string, "public", "_test_data_check")
            is False
        )

        # Add data
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO _test_data_check DEFAULT VALUES")
            conn.commit()

        # Table with data
        assert (
            check_table_has_data(connection_string, "public", "_test_data_check")
            is True
        )

        # Cleanup
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS _test_data_check")
            conn.commit()


# =============================================================================
# Extended Change Type Tests
# =============================================================================


class TestColumnChanges:
    """Tests for column modification change types."""

    def test_alter_column_type_with_using_clause(self):
        """AlterColumnType generates USING clause for type conversion."""
        change = AlterColumnType(
            schema="public",
            table="account",
            column="status",
            old_type="text",
            new_type="integer",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account"' in sql
        assert 'ALTER COLUMN "status" TYPE integer' in sql
        assert 'USING "status"::integer' in sql

    def test_alter_column_type_category(self):
        """AlterColumnType is TRY_FIRST category (let Postgres try)."""
        change = AlterColumnType(
            schema="public",
            table="t",
            column="c",
            old_type="text",
            new_type="varchar(100)",
        )
        assert change.category == ChangeCategory.TRY_FIRST

    def test_set_default_sql(self):
        """SetDefault generates correct ALTER TABLE SQL."""
        change = SetDefault(
            schema="public",
            table="account",
            column="created_at",
            default_value="NOW()",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account"' in sql
        assert 'ALTER COLUMN "created_at" SET DEFAULT NOW()' in sql

    def test_drop_default_sql(self):
        """DropDefault generates correct ALTER TABLE SQL."""
        change = DropDefault(
            schema="public",
            table="account",
            column="updated_at",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account"' in sql
        assert 'ALTER COLUMN "updated_at" DROP DEFAULT' in sql

    def test_drop_not_null_sql(self):
        """DropNotNull generates correct ALTER TABLE SQL."""
        change = DropNotNull(
            schema="public",
            table="account",
            column="nickname",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account"' in sql
        assert 'ALTER COLUMN "nickname" DROP NOT NULL' in sql
        assert change.category == ChangeCategory.ONLINE


class TestConstraintChanges:
    """Tests for constraint change types."""

    def test_add_constraint_fk_not_valid(self):
        """AddConstraint with FK uses NOT VALID for online addition."""
        constraint = Constraint(
            name="fk_parent",
            type="f",  # Foreign key
            definition="FOREIGN KEY (parent_id) REFERENCES parent(id)",
        )
        change = AddConstraint(
            schema="public",
            table="child",
            constraint=constraint,
            not_valid=True,
        )
        sql = change.to_sql()

        assert 'ADD CONSTRAINT "fk_parent"' in sql
        assert "NOT VALID" in sql

    def test_add_constraint_check_not_valid(self):
        """AddConstraint with CHECK uses NOT VALID for online addition."""
        constraint = Constraint(
            name="chk_positive",
            type="c",  # Check
            definition="CHECK (amount > 0)",
        )
        change = AddConstraint(
            schema="public",
            table="transaction",
            constraint=constraint,
            not_valid=True,
        )
        sql = change.to_sql()

        assert 'ADD CONSTRAINT "chk_positive"' in sql
        assert "CHECK (amount > 0)" in sql
        assert "NOT VALID" in sql

    def test_add_constraint_pk_no_not_valid(self):
        """AddConstraint with PK does NOT use NOT VALID."""
        constraint = Constraint(
            name="pk_id",
            type="p",  # Primary key
            definition="PRIMARY KEY (id)",
        )
        change = AddConstraint(
            schema="public",
            table="account",
            constraint=constraint,
            not_valid=True,  # Should be ignored for PK
        )
        sql = change.to_sql()

        assert "PRIMARY KEY" in sql
        assert "NOT VALID" not in sql

    def test_validate_constraint_sql(self):
        """ValidateConstraint generates correct SQL."""
        change = ValidateConstraint(
            schema="public",
            table="child",
            constraint_name="fk_parent",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."child"' in sql
        assert 'VALIDATE CONSTRAINT "fk_parent"' in sql
        assert change.category == ChangeCategory.ONLINE

    def test_drop_constraint_if_exists(self):
        """DropConstraint uses IF EXISTS for safety."""
        change = DropConstraint(
            schema="public",
            table="account",
            constraint_name="old_constraint",
        )
        sql = change.to_sql()

        assert 'DROP CONSTRAINT IF EXISTS "old_constraint"' in sql

    def test_drop_and_create_constraint(self):
        """DropAndCreateConstraint generates DROP + ADD SQL."""
        constraint = Constraint(
            name="fk_updated",
            type="f",
            definition="FOREIGN KEY (ref_id) REFERENCES new_ref(id)",
        )
        change = DropAndCreateConstraint(
            schema="public",
            table="child",
            constraint=constraint,
        )
        sql = change.to_sql()

        assert "DROP CONSTRAINT IF EXISTS" in sql
        assert "ADD CONSTRAINT" in sql
        assert "NOT VALID" in sql  # FK constraint should have NOT VALID


class TestPolicyChanges:
    """Tests for RLS policy change types."""

    def test_create_policy_with_using_and_check(self):
        """CreatePolicy generates USING and WITH CHECK clauses."""
        policy = Policy(
            name="account_isolation",
            table="account",
            schema="public",
            command="ALL",
            using_expr="tenant_id = current_setting('app.tenant_id')::uuid",
            check_expr="tenant_id = current_setting('app.tenant_id')::uuid",
            roles=["app_user"],
        )
        change = CreatePolicy(policy=policy)
        sql = change.to_sql()

        assert 'CREATE POLICY "account_isolation"' in sql
        assert 'ON "public"."account"' in sql
        assert "TO app_user" in sql
        assert "USING (tenant_id = current_setting" in sql
        assert "WITH CHECK (tenant_id = current_setting" in sql

    def test_create_policy_for_select(self):
        """CreatePolicy with FOR SELECT generates correct clause."""
        policy = Policy(
            name="read_policy",
            table="account",
            schema="public",
            command="SELECT",
            using_expr="true",
        )
        change = CreatePolicy(policy=policy)
        sql = change.to_sql()

        assert "FOR SELECT" in sql
        assert "USING (true)" in sql
        # SELECT-only policies don't have WITH CHECK
        assert "WITH CHECK" not in sql

    def test_enable_rls_sql(self):
        """EnableRLS generates correct ALTER TABLE SQL."""
        change = EnableRLS(
            schema="public",
            table="account",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account" ENABLE ROW LEVEL SECURITY' in sql
        assert change.category == ChangeCategory.ONLINE

    def test_disable_rls_sql(self):
        """DisableRLS generates correct ALTER TABLE SQL."""
        change = DisableRLS(
            schema="public",
            table="account",
        )
        sql = change.to_sql()

        assert 'ALTER TABLE "public"."account" DISABLE ROW LEVEL SECURITY' in sql

    def test_drop_policy_if_exists(self):
        """DropPolicy uses IF EXISTS for safety."""
        change = DropPolicy(
            schema="public",
            table="account",
            policy_name="old_policy",
        )
        sql = change.to_sql()

        assert 'DROP POLICY IF EXISTS "old_policy"' in sql
        assert 'ON "public"."account"' in sql

    def test_drop_and_create_policy(self):
        """DropAndCreatePolicy generates DROP + CREATE SQL."""
        policy = Policy(
            name="updated_policy",
            table="account",
            schema="public",
            command="SELECT",
            using_expr="active = true",
        )
        change = DropAndCreatePolicy(policy=policy)
        sql = change.to_sql()

        assert "DROP POLICY IF EXISTS" in sql
        assert "CREATE POLICY" in sql
        assert "updated_policy" in sql

    def test_create_policy_default_is_permissive(self):
        """CreatePolicy without permissive=False omits AS RESTRICTIVE."""
        policy = Policy(
            name="default_policy",
            table="account",
            schema="public",
            command="SELECT",
            using_expr="true",
        )
        sql = CreatePolicy(policy=policy).to_sql()
        assert "AS RESTRICTIVE" not in sql

    def test_create_policy_restrictive_emits_modifier(self):
        """CreatePolicy with permissive=False emits AS RESTRICTIVE before FOR clause."""
        policy = Policy(
            name="review_no_write_policy",
            table="review",
            schema="public",
            command="ALL",
            using_expr="false",
            check_expr="false",
            permissive=False,
        )
        sql = CreatePolicy(policy=policy).to_sql()
        assert "AS RESTRICTIVE" in sql
        # Modifier must come before FOR / TO / USING / WITH CHECK
        as_idx = sql.index("AS RESTRICTIVE")
        # command=ALL is omitted, but USING comes after AS RESTRICTIVE
        assert as_idx < sql.index("USING")

    def test_drop_and_create_policy_restrictive_emits_modifier(self):
        """DropAndCreatePolicy preserves AS RESTRICTIVE on the recreate side."""
        policy = Policy(
            name="review_no_write_policy",
            table="review",
            schema="public",
            command="ALL",
            using_expr="false",
            check_expr="false",
            permissive=False,
        )
        sql = DropAndCreatePolicy(policy=policy).to_sql()
        assert "DROP POLICY IF EXISTS" in sql
        assert "AS RESTRICTIVE" in sql

    def test_policy_equality_distinguishes_permissive_from_restrictive(self):
        """Two policies that differ only in permissive flag must NOT compare equal."""
        # Annotated because it is splatted into Policy(**common): a bare
        # dict[str, str] would be matched against every field, including
        # the list-typed ones.
        common: dict[str, Any] = dict(
            name="review_no_write_policy",
            table="review",
            schema="public",
            command="ALL",
            using_expr="false",
            check_expr="false",
        )
        permissive = Policy(**common, permissive=True)
        restrictive = Policy(**common, permissive=False)
        assert permissive != restrictive

    def test_compute_diff_emits_drop_and_create_when_permissive_flips(self):
        """End-to-end round trip: a permissive->restrictive transition shows up
        as a DropAndCreatePolicy whose SQL emits AS RESTRICTIVE.

        This exercises the path a real migration relies on:
        introspection (read polpermissive) -> Policy.__eq__ (detect mismatch) ->
        differ._diff_policies (choose DropAndCreatePolicy) -> to_sql (emit
        AS RESTRICTIVE).
        """
        # Annotated because it is splatted into Policy(**common): a bare
        # dict[str, str] would be matched against every field, including
        # the list-typed ones.
        common: dict[str, Any] = dict(
            name="review_no_write_policy",
            table="review",
            schema="public",
            command="ALL",
            using_expr="false",
            check_expr="false",
        )
        desired = Schema(
            tables={"public.review": Table(schema="public", name="review")},
            policies={
                "public.review.review_no_write_policy": Policy(
                    **common, permissive=False
                )
            },
        )
        actual = Schema(
            tables={"public.review": Table(schema="public", name="review")},
            policies={
                "public.review.review_no_write_policy": Policy(
                    **common, permissive=True
                )
            },
        )

        changes = compute_diff(desired, actual)

        policy_changes = [c for c in changes if isinstance(c, DropAndCreatePolicy)]
        assert len(policy_changes) == 1
        sql = policy_changes[0].to_sql()
        assert "AS RESTRICTIVE" in sql
        assert "review_no_write_policy" in sql


class TestTriggerChanges:
    """Tests for trigger change types."""

    def test_create_trigger_or_replace(self):
        """CreateTrigger uses OR REPLACE for idempotent creation."""
        trigger = Trigger(
            name="audit_trigger",
            table="account",
            schema="public",
            definition="CREATE TRIGGER audit_trigger AFTER INSERT ON public.account FOR EACH ROW EXECUTE FUNCTION audit_log()",
        )
        change = CreateTrigger(trigger=trigger)
        sql = change.to_sql()

        assert "CREATE OR REPLACE TRIGGER" in sql
        assert "audit_trigger" in sql
        assert change.category == ChangeCategory.ONLINE

    def test_create_constraint_trigger_drops_first(self):
        """CreateTrigger with CONSTRAINT TRIGGER drops before creating."""
        trigger = Trigger(
            name="check_trigger",
            table="transaction",
            schema="public",
            definition="CREATE CONSTRAINT TRIGGER check_trigger AFTER INSERT ON public.transaction DEFERRABLE FOR EACH ROW EXECUTE FUNCTION check_balance()",
        )
        change = CreateTrigger(trigger=trigger)
        sql = change.to_sql()

        # Constraint triggers don't support OR REPLACE, so we DROP + CREATE
        assert "DROP TRIGGER IF EXISTS" in sql
        assert "CREATE CONSTRAINT TRIGGER" in sql

    def test_drop_trigger_if_exists(self):
        """DropTrigger uses IF EXISTS for safety."""
        change = DropTrigger(
            schema="public",
            table="account",
            trigger_name="old_trigger",
        )
        sql = change.to_sql()

        assert 'DROP TRIGGER IF EXISTS "old_trigger"' in sql
        assert 'ON "public"."account"' in sql


class TestEnumChanges:
    """Tests for enum type change types."""

    def test_create_enum_sql(self):
        """CreateEnum generates correct CREATE TYPE AS ENUM SQL."""
        enum = EnumType(
            schema="public",
            name="status",
            values=["active", "inactive", "pending"],
        )
        change = CreateEnum(enum=enum)
        sql = change.to_sql()

        assert 'CREATE TYPE "public"."status" AS ENUM' in sql
        assert "'active'" in sql
        assert "'inactive'" in sql
        assert "'pending'" in sql
        assert change.category == ChangeCategory.ONLINE

    def test_add_enum_value_with_after(self):
        """AddEnumValue with AFTER clause positions value correctly."""
        change = AddEnumValue(
            schema="public",
            enum_name="status",
            value="suspended",
            after="active",
        )
        sql = change.to_sql()

        assert 'ALTER TYPE "public"."status" ADD VALUE' in sql
        assert "'suspended'" in sql
        assert "AFTER 'active'" in sql

    def test_add_enum_value_without_after(self):
        """AddEnumValue without AFTER appends to end."""
        change = AddEnumValue(
            schema="public",
            enum_name="status",
            value="deleted",
        )
        sql = change.to_sql()

        assert "ADD VALUE 'deleted'" in sql
        assert "AFTER" not in sql

    def test_drop_enum_value_is_offline(self):
        """DropEnumValue is OFFLINE (requires type recreation)."""
        change = DropEnumValue(
            schema="public",
            enum_name="status",
            value="obsolete",
        )

        assert change.category == ChangeCategory.OFFLINE
        # SQL should indicate this is a special case
        sql = change.to_sql()
        assert "Cannot directly drop" in sql or "--" in sql


class TestViewChanges:
    """Tests for view change types."""

    def test_create_view_or_replace(self):
        """CreateView uses OR REPLACE for idempotent creation."""
        view = View(
            schema="public",
            name="active_accounts",
            definition="SELECT * FROM account WHERE active = true",
        )
        change = CreateView(view=view)
        sql = change.to_sql()

        assert 'CREATE OR REPLACE VIEW "public"."active_accounts"' in sql
        assert "SELECT * FROM account WHERE active = true" in sql
        assert change.category == ChangeCategory.ONLINE

    def test_drop_view_if_exists(self):
        """DropView uses IF EXISTS for safety."""
        change = DropView(
            schema="public",
            view_name="old_view",
        )
        sql = change.to_sql()

        assert 'DROP VIEW IF EXISTS "public"."old_view"' in sql

    def test_drop_and_create_view(self):
        """DropAndCreateView generates DROP + CREATE SQL."""
        view = View(
            schema="public",
            name="updated_view",
            definition="SELECT id, name FROM account",
        )
        change = DropAndCreateView(view=view)
        sql = change.to_sql()

        assert "DROP VIEW IF EXISTS" in sql
        assert 'CREATE VIEW "public"."updated_view"' in sql
        # Note: DropAndCreateView uses CREATE VIEW (not OR REPLACE)

    def test_drop_and_create_view_replays_grants_atomically(self):
        """The recreate carries its grants inside its own DO block.

        DROP wipes the view's ACL. A standalone GRANT elsewhere in the plan is
        not enough, because the retry-until-stable executor can run the recreate
        after it. Object and ACL have to be one statement.
        """
        view = View(
            schema="public",
            name="headlines",
            definition="SELECT id, title FROM post",
        )
        grants = [
            Grant(
                object_type="TABLE",
                object_name="headlines",
                schema="public",
                privilege="SELECT",
                grantee="app_user",
            ),
            Grant(
                object_type="TABLE",
                object_name="headlines",
                schema="public",
                privilege="SELECT",
                grantee="notification_worker",
            ),
        ]
        change = DropAndCreateView(view=view, grants=grants)
        sql = change.to_sql()

        # One DO block: drop, create, then both grants.
        assert sql.startswith("DO $$ BEGIN ")
        assert sql.rstrip().endswith("END $$;")
        drop_at = sql.index("DROP VIEW IF EXISTS")
        create_at = sql.index("CREATE VIEW")
        api_at = sql.index("TO app_user")
        worker_at = sql.index("TO notification_worker")
        assert drop_at < create_at < api_at
        assert create_at < worker_at
        assert 'EXECUTE \'GRANT SELECT ON "public"."headlines" TO app_user\';' in sql
        assert change.description.endswith("(replaying 2 grants)")

    def test_drop_and_create_view_without_grants_is_unchanged(self):
        """No grants means no trailing EXECUTE and no description suffix."""
        view = View(
            schema="public",
            name="plain",
            definition="SELECT 1 AS one",
        )
        change = DropAndCreateView(view=view)

        assert "GRANT" not in change.to_sql()
        assert change.description == "Drop and recreate view plain"

    def test_drop_and_create_view_applies_options_before_grants(self):
        """The WITH clause is on the CREATE, inside the same DO block."""
        view = View(
            schema="public",
            name="post_comment",
            definition="SELECT id FROM comment",
            reloptions=["security_invoker = TRUE"],
        )
        change = DropAndCreateView(
            view=view,
            grants=[
                Grant(
                    object_type="TABLE",
                    object_name="post_comment",
                    schema="public",
                    privilege="SELECT",
                    grantee="app_user",
                )
            ],
        )
        sql = change.to_sql()

        assert sql.startswith("DO $$ BEGIN ")
        assert (
            'CREATE VIEW "public"."post_comment" '
            "WITH (security_invoker=true) AS SELECT id FROM comment" in sql
        )
        # Options ride on the create, so they land before the ACL replay and
        # cannot be separated from the object by any execution order.
        assert sql.index("security_invoker=true") < sql.index("TO app_user")
        assert "(security_invoker=true)" in change.description

    def test_alter_view_options_sets_declared_options(self):
        """The in-place backfill for a view that only lost its security mode."""
        change = AlterViewOptions(
            view=View(
                schema="public",
                name="teams_with_budgets",
                definition="SELECT 1",
                reloptions=["security_invoker=true"],
            )
        )

        assert change.to_sql() == (
            'ALTER VIEW "public"."teams_with_budgets" SET (security_invoker=true);'
        )
        assert change.object_type == "VIEW"
        assert change.object_name == "public.teams_with_budgets"

    def test_alter_view_options_resets_undeclared_options(self):
        """Options present on the object but no longer declared are RESET."""
        change = AlterViewOptions(
            view=View(
                schema="public",
                name="v",
                definition="SELECT 1",
                reloptions=["security_invoker=true"],
            ),
            current_reloptions=["check_option=cascaded", "security_invoker=true"],
        )
        sql = change.to_sql()

        assert 'ALTER VIEW "public"."v" RESET (check_option);' in sql
        assert 'ALTER VIEW "public"."v" SET (security_invoker=true);' in sql
        # RESET first: SET must have the last word on an option it also declares.
        assert sql.index("RESET") < sql.index("SET")


class TestGrantChanges:
    """Tests for grant/revoke change types."""

    def test_grant_table_privilege(self):
        """GrantPrivilege generates correct GRANT SQL for tables."""
        grant = Grant(
            object_type="TABLE",
            object_name="account",
            schema="public",
            privilege="SELECT",
            grantee="app_user",
        )
        change = GrantPrivilege(grant=grant)
        sql = change.to_sql()

        assert "GRANT SELECT" in sql
        assert 'ON "public"."account"' in sql
        assert "TO app_user" in sql

    def test_grant_column_privilege(self):
        """GrantPrivilege generates correct GRANT SQL for columns."""
        grant = Grant(
            object_type="COLUMN",
            object_name="account",
            schema="public",
            privilege="UPDATE",
            grantee="app_user",
            column="email",
        )
        change = GrantPrivilege(grant=grant)
        sql = change.to_sql()

        assert 'GRANT UPDATE ("email")' in sql
        assert 'ON "public"."account"' in sql

    def test_grant_function_privilege(self):
        """GrantPrivilege generates correct GRANT SQL for functions."""
        grant = Grant(
            object_type="FUNCTION",
            object_name="process_order(uuid)",
            schema="public",
            privilege="EXECUTE",
            grantee="app_role",
        )
        change = GrantPrivilege(grant=grant)
        sql = change.to_sql()

        assert "GRANT EXECUTE" in sql
        assert "ON FUNCTION" in sql
        assert 'process_order(uuid)"' in sql

    def test_revoke_privilege(self):
        """RevokePrivilege generates correct REVOKE SQL."""
        grant = Grant(
            object_type="TABLE",
            object_name="account",
            schema="public",
            privilege="DELETE",
            grantee="app_user",
        )
        change = RevokePrivilege(grant=grant)
        sql = change.to_sql()

        assert "REVOKE DELETE" in sql
        assert "FROM app_user" in sql

    def test_table_revoke_precedes_column_grants(self):
        """Table-wide INSERT narrowed to per-column INSERT revokes first.

        PostgreSQL's table-level REVOKE also removes the column-level grants of
        the same privilege, so emitting the column GRANTs before the table
        REVOKE would wipe them and the run would never converge. The executor
        runs changes in list order.
        """

        def g(column=None):
            return Grant(
                object_type="COLUMN" if column else "TABLE",
                object_name="signup",
                schema="public",
                privilege="INSERT",
                grantee="app_user",
                column=column,
            )

        actual = Schema(grants=[g()])
        desired = Schema(grants=[g("team_id"), g("note"), g("as_owner")])

        changes = [
            c
            for c in compute_diff(desired, actual)
            if isinstance(c, (GrantPrivilege, RevokePrivilege))
        ]

        assert [type(c) for c in changes] == [
            RevokePrivilege,
            GrantPrivilege,
            GrantPrivilege,
            GrantPrivilege,
        ]
        assert changes[0].grant.column is None


class TestFunctionPublicExecute:
    """Tests for SECDEF PUBLIC EXECUTE lockdown via the differ."""

    @staticmethod
    def _secdef(name, args="", definition_hash="h", public_execute=True, defn=None):
        return Function(
            schema="public",
            name=name,
            args=args,
            full_definition=defn or f"CREATE FUNCTION {name}() ... SECURITY DEFINER",
            definition_hash=definition_hash,
            security_definer=True,
            public_execute=public_execute,
        )

    def test_drift_emits_standalone_revoke(self):
        """Locked function that exists on both sides with PUBLIC EXECUTE drift."""
        func = self._secdef("sec_thing")
        desired = Schema(functions={func.signature: func})
        actual = Schema(functions={func.signature: self._secdef("sec_thing")})

        changes = compute_diff(desired, actual, locked_functions={"sec_thing"})

        revokes = [c for c in changes if isinstance(c, RevokePublicExecute)]
        assert len(revokes) == 1
        assert revokes[0].name == "sec_thing"
        assert "REVOKE EXECUTE" in revokes[0].to_sql()
        assert "FROM public" in revokes[0].to_sql()

    def test_not_locked_no_revoke(self):
        """A SECDEF function not in the locked set is untouched."""
        func = self._secdef("sec_api_thing")
        desired = Schema(functions={func.signature: func})
        actual = Schema(functions={func.signature: self._secdef("sec_api_thing")})

        changes = compute_diff(desired, actual, locked_functions={"sec_other"})

        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]

    def test_non_secdef_not_revoked(self):
        """A non-SECDEF function is never revoked even if name is locked."""
        func = Function(
            schema="public",
            name="helper",
            args="",
            full_definition="CREATE FUNCTION helper() ...",
            definition_hash="h",
            security_definer=False,
            public_execute=True,
        )
        desired = Schema(functions={func.signature: func})
        actual = Schema(functions={func.signature: func})

        changes = compute_diff(desired, actual, locked_functions={"helper"})

        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]

    def test_converged_no_revoke(self):
        """Already-locked function (no PUBLIC EXECUTE, no recreate) -> clean diff."""
        desired_func = self._secdef("sec_thing", public_execute=True)
        actual_func = self._secdef("sec_thing", public_execute=False)
        desired = Schema(functions={desired_func.signature: desired_func})
        actual = Schema(functions={actual_func.signature: actual_func})

        changes = compute_diff(desired, actual, locked_functions={"sec_thing"})

        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]

    def test_fresh_create_sets_atomic_flag(self):
        """A locked function being created flags the CreateFunction, no standalone."""
        func = self._secdef("sec_thing")
        desired = Schema(functions={func.signature: func})
        actual = Schema(functions={})

        changes = compute_diff(desired, actual, locked_functions={"sec_thing"})

        creates = [c for c in changes if isinstance(c, CreateFunction)]
        assert len(creates) == 1
        assert creates[0].revoke_public_execute is True
        # Atomic path: no standalone change.
        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]
        # SQL appends the REVOKE.
        sql = creates[0].to_sql()
        assert "REVOKE EXECUTE" in sql and "FROM public" in sql

    def test_recreate_sets_atomic_flag(self):
        """A locked function recreated (return-type change) flags DropAndCreateFunction."""
        desired_func = self._secdef(
            "sec_thing",
            definition_hash="new",
            defn="CREATE FUNCTION sec_thing() RETURNS integer LANGUAGE sql "
            "SECURITY DEFINER AS $$ SELECT 1 $$",
        )
        actual_func = self._secdef(
            "sec_thing",
            definition_hash="old",
            defn="CREATE FUNCTION sec_thing() RETURNS text LANGUAGE sql "
            "SECURITY DEFINER AS $$ SELECT 1 $$",
        )
        desired = Schema(functions={desired_func.signature: desired_func})
        actual = Schema(functions={actual_func.signature: actual_func})

        changes = compute_diff(desired, actual, locked_functions={"sec_thing"})

        dac = [c for c in changes if isinstance(c, DropAndCreateFunction)]
        assert len(dac) == 1
        assert dac[0].revoke_public_execute is True
        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]
        # REVOKE is inside the DO block, after the recreate.
        sql = dac[0].to_sql()
        assert "REVOKE EXECUTE" in sql and "FROM public" in sql

    def test_no_locked_set_is_noop(self):
        """Back-compat: no locked_functions -> no PUBLIC-revoke behavior."""
        func = self._secdef("sec_thing")
        desired = Schema(functions={func.signature: func})
        actual = Schema(functions={func.signature: self._secdef("sec_thing")})

        changes = compute_diff(desired, actual)  # locked_functions defaults to None

        assert not [c for c in changes if isinstance(c, RevokePublicExecute)]


class TestRoleChanges:
    """Tests for role change types."""

    def test_create_role_with_options(self):
        """CreateRole generates correct CREATE ROLE SQL with options."""
        role = Role(
            name="app_user",
            can_login=True,
            create_db=False,
        )
        change = CreateRole(role=role)
        sql = change.to_sql()

        assert 'CREATE ROLE "app_user"' in sql
        assert "WITH" in sql
        assert "LOGIN" in sql

    def test_create_role_minimal(self):
        """CreateRole with no options generates minimal SQL."""
        role = Role(name="simple_role")
        change = CreateRole(role=role)
        sql = change.to_sql()

        assert 'CREATE ROLE "simple_role"' in sql

    def test_alter_role_attributes(self):
        """AlterRole generates correct ALTER ROLE SQL."""
        role = Role(
            name="app_user",
            can_login=True,
            connection_limit=10,
        )
        change = AlterRole(role=role)
        sql = change.to_sql()

        assert 'ALTER ROLE "app_user"' in sql
        assert "CONNECTION LIMIT 10" in sql

    def test_grant_role_membership(self):
        """GrantRoleMembership generates correct GRANT SQL."""
        membership = RoleMembership(
            role="admin",
            member="app_user",
            admin_option=False,
        )
        change = GrantRoleMembership(membership=membership)
        sql = change.to_sql()

        assert 'GRANT "admin" TO "app_user"' in sql
        assert "WITH ADMIN OPTION" not in sql

    def test_grant_role_membership_with_admin(self):
        """GrantRoleMembership with admin option generates correct SQL."""
        membership = RoleMembership(
            role="admin",
            member="superadmin",
            admin_option=True,
        )
        change = GrantRoleMembership(membership=membership)
        sql = change.to_sql()

        assert "WITH ADMIN OPTION" in sql

    def test_create_schema_if_not_exists(self):
        """CreateSchema uses IF NOT EXISTS for idempotent creation."""
        change = CreateSchema(schema_name="api")
        sql = change.to_sql()

        assert 'CREATE SCHEMA IF NOT EXISTS "api"' in sql
        assert change.category == ChangeCategory.ONLINE


class TestChangeDescriptions:
    """Tests that change descriptions are set correctly."""

    def test_create_table_description(self):
        """CreateTable has descriptive description."""
        table = Table(schema="public", name="account")
        change = CreateTable(table=table)
        assert "Create table" in change.description
        assert "public.account" in change.description

    def test_add_column_description(self):
        """AddColumn has descriptive description."""
        col = Column(name="email", type="text", nullable=False)
        change = AddColumn(schema="public", table="account", column=col)
        assert "Add column" in change.description
        assert "email" in change.description

    def test_drop_constraint_description(self):
        """DropConstraint has descriptive description."""
        change = DropConstraint(
            schema="public",
            table="account",
            constraint_name="fk_old",
        )
        assert "Drop constraint" in change.description
        assert "fk_old" in change.description

    def test_enable_rls_description(self):
        """EnableRLS has descriptive description."""
        change = EnableRLS(schema="public", table="account")
        assert "Enable RLS" in change.description
        assert "account" in change.description


class TestObjectNames:
    """Tests that object names are computed correctly."""

    def test_column_change_object_name(self):
        """Column changes include full path in object_name."""
        change = SetNotNull(schema="public", table="account", column="email")
        assert change.object_name == "public.account.email"

    def test_constraint_change_object_name(self):
        """Constraint changes include full path in object_name."""
        change = ValidateConstraint(
            schema="public",
            table="child",
            constraint_name="fk_parent",
        )
        assert change.object_name == "public.child.fk_parent"

    def test_policy_change_object_name(self):
        """Policy changes use qualified_key as object_name."""
        policy = Policy(
            name="my_policy",
            table="account",
            schema="public",
            command="SELECT",
        )
        change = CreatePolicy(policy=policy)
        assert change.object_name == "public.account.my_policy"

    def test_function_change_object_name(self):
        """Function changes use signature as object_name."""
        func = Function(
            schema="public",
            name="process",
            args="uuid, text",
            full_definition="CREATE OR REPLACE FUNCTION...",
        )
        change = CreateFunction(function=func)
        assert change.object_name == "public.process(uuid, text)"


# =============================================================================
# Composite Type Tests
# =============================================================================


class TestCompositeTypeModel:
    """Tests for CompositeType model."""

    def test_composite_type_qualified_name(self):
        """CompositeType qualified name includes schema."""
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        assert comp.qualified_name == "public.sort_spec"

    def test_composite_type_equality(self):
        """CompositeTypes with same properties are equal."""
        comp1 = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        comp2 = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        assert comp1 == comp2

    def test_composite_type_inequality_columns(self):
        """CompositeTypes with different columns are not equal."""
        comp1 = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        comp2 = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "text")],  # Different type
        )
        assert comp1 != comp2


class TestCreateCompositeType:
    """Tests for CreateCompositeType change class."""

    def test_create_composite_type_sql(self):
        """CreateCompositeType generates valid CREATE TYPE AS SQL."""
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        change = CreateCompositeType(composite_type=comp)
        sql = change.to_sql()

        assert 'CREATE TYPE "public"."sort_spec" AS' in sql
        assert '"column_name" text' in sql
        assert '"direction" sort_direction' in sql
        assert change.category == ChangeCategory.ONLINE

    def test_create_composite_type_description(self):
        """CreateCompositeType has descriptive description."""
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text")],
        )
        change = CreateCompositeType(composite_type=comp)
        assert "Create composite type" in change.description
        assert "sort_spec" in change.description

    def test_create_composite_type_object_name(self):
        """CreateCompositeType uses qualified_name as object_name."""
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[],
        )
        change = CreateCompositeType(composite_type=comp)
        assert change.object_name == "public.sort_spec"


class TestCompositeTypeDiffer:
    """Tests for composite type diffing."""

    def test_diff_composite_type_detects_missing(self):
        """Differ generates CreateCompositeType for missing types."""
        desired = Schema(
            composite_types={
                "public.sort_spec": CompositeType(
                    schema="public",
                    name="sort_spec",
                    columns=[
                        ("column_name", "text"),
                        ("direction", "sort_direction"),
                    ],
                )
            }
        )
        actual = Schema()  # Empty

        changes = compute_diff(desired, actual)

        create_types = [c for c in changes if isinstance(c, CreateCompositeType)]
        assert len(create_types) == 1
        assert create_types[0].composite_type.name == "sort_spec"

    def test_diff_composite_type_ignores_existing(self):
        """Differ does not generate changes for existing composite types."""
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        desired = Schema(composite_types={"public.sort_spec": comp})
        actual = Schema(composite_types={"public.sort_spec": comp})

        changes = compute_diff(desired, actual)

        create_types = [c for c in changes if isinstance(c, CreateCompositeType)]
        assert len(create_types) == 0

    def test_diff_multiple_composite_types(self):
        """Differ generates CreateCompositeType for all missing types."""
        desired = Schema(
            composite_types={
                "public.sort_spec": CompositeType(
                    schema="public",
                    name="sort_spec",
                    columns=[("column_name", "text")],
                ),
                "public.validation_result": CompositeType(
                    schema="public",
                    name="validation_result",
                    columns=[("valid", "boolean"), ("reason", "text")],
                ),
            }
        )
        actual = Schema()  # Empty

        changes = compute_diff(desired, actual)

        create_types = [c for c in changes if isinstance(c, CreateCompositeType)]
        assert len(create_types) == 2
        type_names = {c.composite_type.name for c in create_types}
        assert type_names == {"sort_spec", "validation_result"}

    def test_composite_types_created_before_functions(self):
        """Composite types appear before functions in diff output.

        This test verifies the fix for CI failures where functions using
        composite types (like sort_spec[]) failed because the types didn't exist.
        Enums must come first (composite types may reference them), then
        composite types, then functions.
        """
        enum = EnumType(
            schema="public",
            name="sort_direction",
            values=["asc", "desc"],
        )
        comp = CompositeType(
            schema="public",
            name="sort_spec",
            columns=[("column_name", "text"), ("direction", "sort_direction")],
        )
        func = Function(
            schema="public",
            name="list_items",
            args="p_sort sort_spec[]",
            full_definition="CREATE OR REPLACE FUNCTION list_items(p_sort sort_spec[]) RETURNS TABLE...",
            definition_hash="abc123",
        )

        desired = Schema(
            enums={"public.sort_direction": enum},
            composite_types={"public.sort_spec": comp},
            functions={"public.list_items(p_sort sort_spec[])": func},
        )
        actual = Schema()  # Empty

        changes = compute_diff(desired, actual)

        # Find indices of each change type
        enum_indices = [i for i, c in enumerate(changes) if isinstance(c, CreateEnum)]
        composite_indices = [
            i for i, c in enumerate(changes) if isinstance(c, CreateCompositeType)
        ]
        func_indices = [
            i for i, c in enumerate(changes) if isinstance(c, CreateFunction)
        ]

        # Verify ordering: enums < composite types < functions
        assert len(enum_indices) == 1
        assert len(composite_indices) == 1
        assert len(func_indices) == 1

        # Enums must come before composite types (sort_spec uses sort_direction)
        assert enum_indices[0] < composite_indices[0], (
            "Enums must be created before composite types"
        )
        # Composite types must come before functions (function uses sort_spec[])
        assert composite_indices[0] < func_indices[0], (
            "Composite types must be created before functions"
        )


class TestGeneratedColumnsSupport:
    """STORED generated columns (GENERATED ALWAYS AS (...) STORED)."""

    def _table(self, columns: dict) -> Schema:
        return Schema(
            tables={
                "public.t": Table(schema="public", name="t", columns=columns),
            }
        )

    def test_column_equality_includes_generated(self):
        """The generated field participates in Column equality."""
        a = Column(name="m", type="text", nullable=True, generated="f(x)")
        b = Column(name="m", type="text", nullable=True, generated="f(x)")
        c = Column(name="m", type="text", nullable=True, generated="g(x)")
        plain = Column(name="m", type="text", nullable=True)
        assert a == b
        assert a != c
        assert a != plain

    def test_add_generated_column_emits_stored_clause(self):
        """Adding a generated column emits STORED and never a DEFAULT."""
        desired = self._table(
            {
                "id": Column(name="id", type="uuid", nullable=False),
                "type": Column(name="type", type="text", nullable=False),
                "m": Column(
                    name="m", type="text", nullable=True, generated="mech_of(type)"
                ),
            }
        )
        actual = self._table(
            {
                "id": Column(name="id", type="uuid", nullable=False),
                "type": Column(name="type", type="text", nullable=False),
            }
        )
        changes = compute_diff(desired, actual)
        adds = [c for c in changes if isinstance(c, AddColumn)]
        assert len(adds) == 1
        sql = adds[0].to_sql()
        assert "GENERATED ALWAYS AS (mech_of(type)) STORED" in sql
        assert "DEFAULT" not in sql
        # Rewrites a populated table under ACCESS EXCLUSIVE -> must be offline.
        assert adds[0].category == ChangeCategory.OFFLINE

    def test_create_table_with_generated_column(self):
        """CreateTable renders the STORED clause and no DEFAULT."""
        table = Table(
            schema="public",
            name="t",
            columns={
                "type": Column(name="type", type="text", nullable=False),
                "m": Column(
                    name="m", type="text", nullable=False, generated="mech_of(type)"
                ),
            },
        )
        sql = CreateTable(table=table).to_sql()
        assert "GENERATED ALWAYS AS (mech_of(type)) STORED" in sql
        assert "DEFAULT" not in sql
        # NOT NULL is still allowed on a generated column.
        assert '"m" text NOT NULL GENERATED ALWAYS AS' in sql

    def test_changed_expression_uses_set_expression(self):
        """generated->generated expr change emits SET EXPRESSION, nothing else."""
        desired = self._table(
            {"m": Column(name="m", type="text", nullable=True, generated="v2(x)")}
        )
        actual = self._table(
            {"m": Column(name="m", type="text", nullable=True, generated="v1(x)")}
        )
        changes = compute_diff(desired, actual)
        set_expr = [c for c in changes if isinstance(c, AlterColumnSetExpression)]
        assert len(set_expr) == 1
        assert set_expr[0].to_sql() == (
            'ALTER TABLE "public"."t" ALTER COLUMN "m" SET EXPRESSION AS (v2(x));'
        )
        assert set_expr[0].category == ChangeCategory.OFFLINE
        # No drop/recreate, no spurious type/default change.
        assert not [c for c in changes if isinstance(c, DropColumn)]
        assert not [c for c in changes if isinstance(c, AlterColumnType)]
        assert not [c for c in changes if isinstance(c, SetDefault)]

    def test_generated_to_plain_uses_drop_expression(self):
        """generated->plain emits DROP EXPRESSION, then applies the new DEFAULT."""
        desired = self._table(
            {"m": Column(name="m", type="text", nullable=True, default="'x'")}
        )
        actual = self._table(
            {"m": Column(name="m", type="text", nullable=True, generated="v1(x)")}
        )
        changes = compute_diff(desired, actual)
        drop_expr = [c for c in changes if isinstance(c, AlterColumnDropExpression)]
        assert len(drop_expr) == 1
        assert drop_expr[0].to_sql() == (
            'ALTER TABLE "public"."t" ALTER COLUMN "m" DROP EXPRESSION;'
        )
        # The desired plain column has a DEFAULT -> it gets set after dropping.
        assert [c for c in changes if isinstance(c, SetDefault)]
        assert not [c for c in changes if isinstance(c, DropColumn)]

    def test_plain_to_generated_fails_fast(self):
        """plain->generated raises rather than emitting unsafe DDL."""
        desired = self._table(
            {"m": Column(name="m", type="text", nullable=True, generated="v1(x)")}
        )
        actual = self._table({"m": Column(name="m", type="text", nullable=True)})
        with pytest.raises(ValueError, match="stored generated column"):
            compute_diff(desired, actual)

    def test_unchanged_generated_column_no_diff(self):
        """Same generation expression on both sides -> empty diff."""
        cols = {"m": Column(name="m", type="text", nullable=True, generated="v1(x)")}
        changes = compute_diff(self._table(cols), self._table(dict(cols)))
        assert changes == []

    def test_removing_generated_column_is_normal_drop(self):
        """Dropping a generated column entirely uses the normal DropColumn path."""
        desired = self._table({"id": Column(name="id", type="uuid", nullable=False)})
        actual = self._table(
            {
                "id": Column(name="id", type="uuid", nullable=False),
                "m": Column(name="m", type="text", nullable=True, generated="v1(x)"),
            }
        )
        changes = compute_diff(desired, actual)
        drops = [c for c in changes if isinstance(c, DropColumn)]
        assert len(drops) == 1
        assert drops[0].column_name == "m"
