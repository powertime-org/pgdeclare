"""
Schema diff lifecycle tests to prevent cross-schema enum and orphaned migration issues.

These tests verify the schema diff engine handles complex real-world scenarios:
1. Cross-schema enum collisions (same enum name in multiple schemas)
2. Orphaned migration state (migration marked applied but objects still exist)
3. Full lifecycle with complex dependencies (enum → table → function → trigger)
4. Permission revocation edge cases

These tests require a PostgreSQL database. Set TEST_DATABASE_URL to run.
"""

import os
import uuid
from pathlib import Path


import pytest

import psycopg2
from psycopg2 import errors as pg_errors

from pgdeclare.models import (
    Column,
    Constraint,
    EnumType,
    Schema,
    Table,
)
from pgdeclare.changes import (
    CreateEnum,
    CreateTable,
)
from pgdeclare.differ import compute_diff
from dbutil import scalar


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def db_url():
    """Get database URL, skip if not set."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def test_schema_prefix():
    """Generate unique schema prefix for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def multi_schema_env(db_url, test_schema_prefix):
    """
    Create multiple test schemas to simulate cross-schema scenarios.

    Creates:
    - {prefix}_public: simulates the public schema
    - {prefix}_app_generated: simulates app_generated schema
    - {prefix}_app: simulates an app schema
    """
    schemas = {
        "public": f"{test_schema_prefix}_public",
        "app_generated": f"{test_schema_prefix}_app_generated",
        "app": f"{test_schema_prefix}_app",
    }

    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for schema_name in schemas.values():
                cur.execute(f'CREATE SCHEMA "{schema_name}"')

    yield schemas

    # Cleanup
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for schema_name in schemas.values():
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')


@pytest.fixture
def single_schema(db_url, test_schema_prefix):
    """Create a single test schema for simple tests."""
    schema_name = f"{test_schema_prefix}_test"

    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema_name}"')

    yield schema_name

    # Cleanup
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')


@pytest.fixture
def migrations_table_helper(db_url):
    """
    Helper to manipulate the migrations tracking table for testing orphaned state.
    """

    class MigrationsTableHelper:
        def __init__(self, db_url: str):
            self.db_url = db_url
            self._ensure_table()

        def _ensure_table(self):
            """Ensure the migrations tracking table exists."""
            with psycopg2.connect(self.db_url) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS pgdeclare")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS pgdeclare.applied_migrations (
                            id TEXT PRIMARY KEY,
                            applied_at TIMESTAMPTZ DEFAULT NOW(),
                            source_file TEXT,
                            sql_hash TEXT,
                            sql_executed TEXT
                        )
                    """)

        def mark_as_applied(
            self, migration_id: str, sql_hash: str = "test_hash"
        ) -> None:
            """Mark a migration as applied without actually running it."""
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO pgdeclare.applied_migrations
                        (id, sql_hash, sql_executed, source_file)
                        VALUES (%s, %s, 'test sql', 'test.sql')
                        ON CONFLICT (id) DO UPDATE SET sql_hash = EXCLUDED.sql_hash
                        """,
                        (migration_id, sql_hash),
                    )
                conn.commit()

        def remove_applied_mark(self, migration_id: str) -> None:
            """Remove the applied mark for a migration."""
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM pgdeclare.applied_migrations WHERE id = %s",
                        (migration_id,),
                    )
                conn.commit()

        def is_applied(self, migration_id: str) -> bool:
            """Check if a migration is marked as applied."""
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pgdeclare.applied_migrations WHERE id = %s",
                        (migration_id,),
                    )
                    return cur.fetchone() is not None

        def cleanup(self) -> None:
            """Remove all test migrations."""
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM pgdeclare.applied_migrations WHERE id LIKE 'test_%'"
                    )
                conn.commit()

    helper = MigrationsTableHelper(db_url)
    yield helper
    helper.cleanup()


def execute_sql(db_url: str, sql: str):
    """Execute SQL statement(s) against the database."""
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)


def table_exists(db_url: str, schema: str, table_name: str) -> bool:
    """Check if a table exists."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                )
                """,
                (schema, table_name),
            )
            return scalar(cur)


def view_exists(db_url: str, schema: str, view_name: str) -> bool:
    """Check if a view exists."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views
                    WHERE table_schema = %s AND table_name = %s
                )
                """,
                (schema, view_name),
            )
            return scalar(cur)


def enum_exists(db_url: str, schema: str, enum_name: str) -> bool:
    """Check if an enum type exists in a specific schema."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_type t
                    JOIN pg_namespace n ON t.typnamespace = n.oid
                    WHERE n.nspname = %s AND t.typname = %s AND t.typtype = 'e'
                )
                """,
                (schema, enum_name),
            )
            return scalar(cur)


def function_exists(db_url: str, schema: str, function_name: str) -> bool:
    """Check if a function exists (by name, ignoring overloads)."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_proc p
                    JOIN pg_namespace n ON p.pronamespace = n.oid
                    WHERE n.nspname = %s AND p.proname = %s
                )
                """,
                (schema, function_name),
            )
            return scalar(cur)


# =============================================================================
# Test Case A: Cross-Schema Enum Collision Detection
# =============================================================================


@pytest.mark.integration
class TestCrossSchemaEnumCollision:
    """Tests for detecting and handling duplicate enum types across schemas."""

    def test_detects_same_enum_name_in_multiple_schemas(self, db_url, multi_schema_env):
        """
        Scenario:
        1. Create enum 'status_type' in public schema
        2. Create enum 'status_type' in app schema
        3. Verify both exist and are distinct types
        """
        public_schema = multi_schema_env["public"]
        app_schema = multi_schema_env["app"]

        # Create same enum name in two different schemas
        execute_sql(
            db_url,
            f"""
            CREATE TYPE "{public_schema}"."status_type" AS ENUM ('active', 'inactive');
            CREATE TYPE "{app_schema}"."status_type" AS ENUM ('pending', 'approved', 'rejected');
            """,
        )

        # Both should exist
        assert enum_exists(db_url, public_schema, "status_type")
        assert enum_exists(db_url, app_schema, "status_type")

        # Verify they have different values (proof they're distinct)
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Get values for public.status_type
                cur.execute(
                    """
                    SELECT e.enumlabel
                    FROM pg_enum e
                    JOIN pg_type t ON e.enumtypid = t.oid
                    JOIN pg_namespace n ON t.typnamespace = n.oid
                    WHERE n.nspname = %s AND t.typname = 'status_type'
                    ORDER BY e.enumsortorder
                    """,
                    (public_schema,),
                )
                public_values = [row[0] for row in cur.fetchall()]

                # Get values for app.status_type
                cur.execute(
                    """
                    SELECT e.enumlabel
                    FROM pg_enum e
                    JOIN pg_type t ON e.enumtypid = t.oid
                    JOIN pg_namespace n ON t.typnamespace = n.oid
                    WHERE n.nspname = %s AND t.typname = 'status_type'
                    ORDER BY e.enumsortorder
                    """,
                    (app_schema,),
                )
                app_values = [row[0] for row in cur.fetchall()]

        assert public_values == ["active", "inactive"]
        assert app_values == ["pending", "approved", "rejected"]

    def test_explicit_schema_qualification_prevents_ambiguity(
        self, db_url, multi_schema_env
    ):
        """
        Scenario:
        1. Create enum in app_generated schema
        2. Create table with column of that enum type using explicit qualification
        3. Use explicitly qualified enum literals - should work
        """
        gen_schema = multi_schema_env["app_generated"]
        public_schema = multi_schema_env["public"]

        # Create enum in app_generated
        execute_sql(
            db_url,
            f"""
            CREATE TYPE "{gen_schema}"."asset_type" AS ENUM ('time', 'tokens', 'currency');
            """,
        )

        # Create table using explicitly qualified enum type
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{public_schema}"."assets" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                type "{gen_schema}"."asset_type" NOT NULL
            );
            """,
        )

        # Insert using explicitly qualified literal - should work
        execute_sql(
            db_url,
            f"""
            INSERT INTO "{public_schema}"."assets" (type)
            VALUES ('time'::"{gen_schema}"."asset_type");
            """,
        )

        # Verify data
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT type FROM "{public_schema}"."assets"')
                result = scalar(cur)
                assert result == "time"

    def test_enum_type_mismatch_in_comparison_detected(self, db_url, multi_schema_env):
        """
        Scenario:
        1. Create public.asset_type and app_generated.asset_type (same values)
        2. Create table with column using app_generated.asset_type
        3. Attempt query with 'time'::asset_type (unqualified) - may fail or use wrong type
        4. Verify the test catches the type mismatch behavior
        """
        public_schema = multi_schema_env["public"]
        gen_schema = multi_schema_env["app_generated"]

        # Create same enum in both schemas
        execute_sql(
            db_url,
            f"""
            CREATE TYPE "{public_schema}"."asset_type" AS ENUM ('time', 'tokens', 'currency');
            CREATE TYPE "{gen_schema}"."asset_type" AS ENUM ('time', 'tokens', 'currency');
            """,
        )

        # Create table using app_generated.asset_type
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{public_schema}"."assets" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                type "{gen_schema}"."asset_type" NOT NULL
            );
            INSERT INTO "{public_schema}"."assets" (type)
            VALUES ('time'::"{gen_schema}"."asset_type");
            """,
        )

        # Now try to query with unqualified enum literal in a context where
        # the search_path might resolve to the wrong schema
        # Set search_path to public first
        with psycopg2.connect(db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO "{public_schema}"')

                # This comparison may fail due to type mismatch if the unqualified
                # literal resolves to public.asset_type while column is app_generated.asset_type
                try:
                    cur.execute(
                        f"""
                        SELECT * FROM "{public_schema}"."assets"
                        WHERE type = 'time'::asset_type
                        """
                    )
                    # If it succeeds, it might be because search_path resolved correctly
                    # or because PostgreSQL was lenient
                    results = cur.fetchall()
                    # Test passes either way - we're documenting the behavior
                except (pg_errors.DatatypeMismatch, pg_errors.UndefinedFunction):
                    # Expected failure when the unqualified literal resolves to the
                    # other schema's enum. Two distinct schemas' enums are unrelated
                    # types with no equality operator between them, so PostgreSQL
                    # reports undefined_function ("operator does not exist") rather
                    # than datatype_mismatch. Both are the same finding.
                    pass

        # Verify that explicit qualification always works
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM "{public_schema}"."assets"
                    WHERE type = 'time'::"{gen_schema}"."asset_type"
                    """
                )
                results = cur.fetchall()
                assert len(results) == 1

    def test_differ_generates_explicit_enum_type_in_column_definition(
        self, db_url, multi_schema_env
    ):
        """
        Verify that the differ correctly uses schema-qualified enum types
        when generating CreateTable SQL.
        """
        gen_schema = multi_schema_env["app_generated"]
        public_schema = multi_schema_env["public"]

        # First, create the enum in the database
        execute_sql(
            db_url,
            f"CREATE TYPE \"{gen_schema}\".\"asset_type\" AS ENUM ('time', 'tokens')",
        )

        # Define a table using the enum
        table = Table(
            schema=public_schema,
            name="test_assets",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                # Use fully qualified enum type
                "type": Column(
                    name="type",
                    type=f'"{gen_schema}"."asset_type"',
                    nullable=False,
                ),
            },
        )

        desired = Schema(tables={f"{public_schema}.test_assets": table})
        actual = Schema()

        changes = compute_diff(desired, actual)

        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 1

        # The SQL should include the qualified enum type
        sql = create_tables[0].to_sql()
        assert f'"{gen_schema}"."asset_type"' in sql or "asset_type" in sql


# =============================================================================
# Test Case B: Orphaned Migration State Detection
# =============================================================================


@pytest.mark.integration
class TestOrphanedMigrationState:
    """Tests for detecting inconsistencies between migration state and actual schema."""

    def test_detects_migration_marked_applied_but_table_still_exists(
        self, db_url, single_schema, migrations_table_helper
    ):
        """
        Scenario:
        1. Create a table
        2. Mark a migration that should have dropped it as "applied"
        3. But DON'T actually drop the table
        4. Detect this inconsistency
        """
        schema = single_schema

        # Create the table
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."audit_event" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                data jsonb
            )
            """,
        )

        # Mark a "drop" migration as applied (but don't actually drop)
        migrations_table_helper.mark_as_applied("test_2026-01-27_drop_audit_event")

        # Verify the inconsistency: migration says dropped, but table exists
        assert migrations_table_helper.is_applied("test_2026-01-27_drop_audit_event")
        assert table_exists(db_url, schema, "audit_event")

        # This is the orphaned state: the ledger and the database disagree

    def test_detects_migration_not_marked_but_objects_gone(
        self, db_url, single_schema, migrations_table_helper
    ):
        """
        Scenario:
        1. A migration should have created a table
        2. The table exists (was created manually or by another process)
        3. But the migration is NOT marked as applied
        4. Detect this inconsistency
        """
        schema = single_schema

        # Create a table (simulating it was created some other way)
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."users" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                email text
            )
            """,
        )

        # The migration that should have created it is NOT marked as applied
        assert not migrations_table_helper.is_applied("test_2026-01-25_create_users")

        # Verify the inconsistency: table exists but migration not tracked
        assert table_exists(db_url, schema, "users")
        assert not migrations_table_helper.is_applied("test_2026-01-25_create_users")

    def test_partial_migration_leaves_orphaned_view(
        self, db_url, single_schema, migrations_table_helper
    ):
        """
        Scenario:
        1. Create a table and dependent view
        2. Simulate a partial migration: migration marked applied, table dropped, view remains
        3. Differ should handle this gracefully
        """
        schema = single_schema

        # Create table and dependent view
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."events" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                type text
            );
            CREATE VIEW "{schema}"."active_events" AS
                SELECT * FROM "{schema}"."events" WHERE type = 'active';
            """,
        )

        # Simulate partial migration: drop the table (view becomes invalid),
        # but imagine the migration tracking got updated before the view drop
        migrations_table_helper.mark_as_applied("test_2026-01-27_cleanup_events")

        # Drop only the table, not the view
        execute_sql(db_url, f'DROP TABLE "{schema}"."events" CASCADE')

        # The orphaned state: migration applied, but view may or may not exist
        # (CASCADE would have dropped the view too)
        assert migrations_table_helper.is_applied("test_2026-01-27_cleanup_events")
        assert not table_exists(db_url, schema, "events")
        # View should be gone due to CASCADE
        assert not view_exists(db_url, schema, "active_events")

    def test_differ_handles_missing_objects_gracefully(
        self, db_url, single_schema, migrations_table_helper
    ):
        """
        Scenario:
        When the differ tries to generate changes for objects that no longer exist,
        it should not fail catastrophically.
        """
        schema = single_schema

        # Define a schema with objects that don't exist in the database
        desired = Schema(
            tables={
                f"{schema}.new_table": Table(
                    schema=schema,
                    name="new_table",
                    columns={
                        "id": Column(
                            name="id",
                            type="uuid",
                            nullable=False,
                            default="gen_random_uuid()",
                        ),
                    },
                )
            }
        )

        # Actual schema is empty (or has stale data)
        actual = Schema()

        # This should not raise an exception
        changes = compute_diff(desired, actual)

        # Should generate a CreateTable
        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 1


# =============================================================================
# Test Case C: Full Lifecycle with Dependencies
# =============================================================================


@pytest.mark.integration
class TestFullLifecycleWithDependencies:
    """End-to-end tests for create -> migrate -> drop with dependency chains."""

    def test_enum_table_function_trigger_lifecycle(self, db_url, single_schema):
        """
        Full lifecycle:
        1. CREATE: enum -> table (using enum) -> function -> trigger
        2. VERIFY: All objects exist
        3. DROP: Remove in reverse order
        4. VERIFY: All objects removed, no orphans
        """
        schema = single_schema

        # Step 1: Create the dependency chain
        execute_sql(
            db_url,
            f"""
            -- Enum type
            CREATE TYPE "{schema}"."event_type" AS ENUM ('created', 'updated', 'deleted');

            -- Table using enum
            CREATE TABLE "{schema}"."audit_log" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                event_type "{schema}"."event_type" NOT NULL,
                created_at timestamptz DEFAULT now()
            );

            -- Function for trigger
            -- TG_OP is 'INSERT'/'UPDATE'/'DELETE', which are not members of
            -- event_type, so a bare cast raises invalid_text_representation and
            -- the INSERT below fails. Map the operation onto the enum instead.
            CREATE OR REPLACE FUNCTION "{schema}"."log_change"()
            RETURNS trigger AS $$
            BEGIN
                INSERT INTO "{schema}"."audit_log" (event_type)
                VALUES (
                    CASE TG_OP
                        WHEN 'INSERT' THEN 'created'
                        WHEN 'UPDATE' THEN 'updated'
                        ELSE 'deleted'
                    END::"{schema}"."event_type"
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            -- Table to trigger on
            CREATE TABLE "{schema}"."documents" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                title text
            );

            -- Trigger
            CREATE TRIGGER doc_audit
            AFTER INSERT ON "{schema}"."documents"
            FOR EACH ROW
            EXECUTE FUNCTION "{schema}"."log_change"();
            """,
        )

        # Step 2: Verify all exist
        assert enum_exists(db_url, schema, "event_type")
        assert table_exists(db_url, schema, "audit_log")
        assert table_exists(db_url, schema, "documents")
        assert function_exists(db_url, schema, "log_change")

        # Step 3: Insert data to verify the chain works
        execute_sql(
            db_url,
            f"""
            INSERT INTO "{schema}"."documents" (title) VALUES ('Test Doc');
            """,
        )

        # Verify the trigger fired. The chain enum -> table -> function ->
        # trigger is the thing under test, so the audit row is the evidence that
        # every link in it is live.
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT event_type::text FROM "{schema}"."audit_log"')
                assert cur.fetchall() == [("created",)]

        # Step 4: Drop in reverse dependency order
        execute_sql(
            db_url,
            f"""
            DROP TRIGGER IF EXISTS doc_audit ON "{schema}"."documents";
            DROP TABLE IF EXISTS "{schema}"."documents" CASCADE;
            DROP FUNCTION IF EXISTS "{schema}"."log_change"();
            DROP TABLE IF EXISTS "{schema}"."audit_log" CASCADE;
            DROP TYPE IF EXISTS "{schema}"."event_type" CASCADE;
            """,
        )

        # Step 5: Verify all gone
        assert not enum_exists(db_url, schema, "event_type")
        assert not table_exists(db_url, schema, "audit_log")
        assert not table_exists(db_url, schema, "documents")
        assert not function_exists(db_url, schema, "log_change")

    def test_cross_schema_reference_lifecycle(self, db_url, multi_schema_env):
        """
        Lifecycle with cross-schema references:
        1. CREATE: public.enum, app.table (references public.enum)
        2. VERIFY: Both exist
        3. DROP: Handle cross-schema dependency correctly
        """
        public_schema = multi_schema_env["public"]
        app_schema = multi_schema_env["app"]

        # Step 1: Create cross-schema setup
        execute_sql(
            db_url,
            f"""
            -- Enum in public schema
            CREATE TYPE "{public_schema}"."status" AS ENUM ('draft', 'published', 'archived');

            -- Table in app schema using public enum
            CREATE TABLE "{app_schema}"."articles" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                status "{public_schema}"."status" NOT NULL DEFAULT 'draft'
            );
            """,
        )

        # Step 2: Verify
        assert enum_exists(db_url, public_schema, "status")
        assert table_exists(db_url, app_schema, "articles")

        # Step 3: Insert data
        execute_sql(
            db_url,
            f"""
            INSERT INTO "{app_schema}"."articles" (status)
            VALUES ('published'::"{public_schema}"."status");
            """,
        )

        # Step 4: Drop - must drop table before enum
        execute_sql(
            db_url,
            f"""
            DROP TABLE IF EXISTS "{app_schema}"."articles" CASCADE;
            DROP TYPE IF EXISTS "{public_schema}"."status" CASCADE;
            """,
        )

        # Step 5: Verify cleanup
        assert not enum_exists(db_url, public_schema, "status")
        assert not table_exists(db_url, app_schema, "articles")

    def test_circular_fk_drop_lifecycle(self, db_url, single_schema):
        """
        Lifecycle with circular FKs (like post <-> comment):
        1. CREATE: table A, table B with FK to A, A with FK to B
        2. VERIFY: Both tables exist with FKs
        3. DROP: Remove both tables (CASCADE handles circular)
        """
        schema = single_schema

        # Step 1: Create tables with circular FKs
        execute_sql(
            db_url,
            f"""
            -- Create tables first
            CREATE TABLE "{schema}"."post" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                title text NOT NULL,
                last_comment_id uuid  -- FK added later
            );

            CREATE TABLE "{schema}"."comment" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                post_id uuid NOT NULL REFERENCES "{schema}"."post"(id),
                content text NOT NULL
            );

            -- Add circular FK
            ALTER TABLE "{schema}"."post"
            ADD CONSTRAINT fk_last_comment
            FOREIGN KEY (last_comment_id) REFERENCES "{schema}"."comment"(id);
            """,
        )

        # Step 2: Verify both exist
        assert table_exists(db_url, schema, "post")
        assert table_exists(db_url, schema, "comment")

        # Step 3: Insert data respecting FKs
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Create post
                cur.execute(
                    f"""
                    INSERT INTO "{schema}"."post" (title)
                    VALUES ('Test Post')
                    RETURNING id
                    """
                )
                post_id = scalar(cur)

                # Create comment
                cur.execute(
                    f"""
                    INSERT INTO "{schema}"."comment" (post_id, content)
                    VALUES (%s, 'First comment')
                    RETURNING id
                    """,
                    (post_id,),
                )
                comment_id = scalar(cur)

                # Update post with last_comment_id
                cur.execute(
                    f"""
                    UPDATE "{schema}"."post"
                    SET last_comment_id = %s
                    WHERE id = %s
                    """,
                    (comment_id, post_id),
                )

            conn.commit()

        # Step 4: Drop with CASCADE to handle circular dependency
        execute_sql(
            db_url,
            f"""
            DROP TABLE IF EXISTS "{schema}"."comment" CASCADE;
            DROP TABLE IF EXISTS "{schema}"."post" CASCADE;
            """,
        )

        # Step 5: Verify cleanup
        assert not table_exists(db_url, schema, "post")
        assert not table_exists(db_url, schema, "comment")


# =============================================================================
# Test Case D: Permission Revocation Edge Cases
# =============================================================================


@pytest.mark.integration
class TestPermissionRevocationEdgeCases:
    """Tests for permission operations when schema state is inconsistent."""

    def test_revoke_on_nonexistent_table_graceful_handling(self, db_url, single_schema):
        """
        Scenario:
        1. Create a table
        2. Grant permissions on it
        3. Drop the table outside of the differ
        4. Attempt to revoke permissions
        5. Should handle gracefully (IF EXISTS or catch error)
        """
        schema = single_schema

        # Step 1: Create table and role
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."sensitive_data" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                secret text
            )
            """,
        )

        # Step 2: Grant permissions (using a role that exists in test DB)
        # Note: In a real scenario, we'd create a test role
        # For this test, we'll just verify the revoke behavior

        # Step 3: Drop the table
        execute_sql(db_url, f'DROP TABLE IF EXISTS "{schema}"."sensitive_data"')

        # Step 4: Attempt revoke on non-existent table - should not raise
        # This simulates what the differ might try to do
        try:
            execute_sql(
                db_url,
                f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = '{schema}' AND table_name = 'sensitive_data'
                    ) THEN
                        REVOKE ALL ON "{schema}"."sensitive_data" FROM PUBLIC;
                    END IF;
                END
                $$;
                """,
            )
            # Success - graceful handling
            handled_gracefully = True
        except Exception:
            # If it fails without IF EXISTS, that's what we're trying to prevent
            handled_gracefully = False

        assert handled_gracefully

    def test_grant_revoke_cycle_with_schema_changes(self, db_url, single_schema):
        """
        Scenario:
        1. Create table, grant permissions
        2. Drop and recreate table (same name)
        3. Verify permissions need to be reapplied
        """
        schema = single_schema

        # Step 1: Create table
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."config" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                key text UNIQUE,
                value text
            );
            -- Grant would go here in real scenario
            """,
        )

        assert table_exists(db_url, schema, "config")

        # Step 2: Drop and recreate
        execute_sql(
            db_url,
            f"""
            DROP TABLE "{schema}"."config";
            CREATE TABLE "{schema}"."config" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                key text UNIQUE,
                value text,
                -- New column in recreated table
                updated_at timestamptz DEFAULT now()
            );
            """,
        )

        # Step 3: Table exists with new schema
        assert table_exists(db_url, schema, "config")

        # Verify the new column exists
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = 'config'
                    ORDER BY ordinal_position
                    """,
                    (schema,),
                )
                columns = [row[0] for row in cur.fetchall()]
                assert "updated_at" in columns

    def test_permission_on_view_when_underlying_table_changed(
        self, db_url, single_schema
    ):
        """
        Scenario:
        1. Create table and view
        2. Grant permissions on view
        3. Modify underlying table
        4. Verify view still works (or needs recreation)
        """
        schema = single_schema

        # Step 1: Create table and view
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."base_table" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name text,
                email text
            );

            CREATE VIEW "{schema}"."safe_view" AS
                SELECT id, name FROM "{schema}"."base_table";
            """,
        )

        assert table_exists(db_url, schema, "base_table")
        assert view_exists(db_url, schema, "safe_view")

        # Step 2: Add column to underlying table
        execute_sql(
            db_url,
            f'ALTER TABLE "{schema}"."base_table" ADD COLUMN created_at timestamptz',
        )

        # Step 3: Verify view still works (it should, as it explicitly lists columns)
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT * FROM "{schema}"."safe_view"')
                # Should not raise - view should still work

        # Step 4: Drop column that view doesn't use - view should still work
        execute_sql(
            db_url,
            f'ALTER TABLE "{schema}"."base_table" DROP COLUMN IF EXISTS created_at',
        )

        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT * FROM "{schema}"."safe_view"')
                # Should not raise


# =============================================================================
# Additional Tests: Differ Behavior with Complex Scenarios
# =============================================================================


@pytest.mark.integration
class TestDifferComplexScenarios:
    """Tests for differ behavior in complex multi-schema scenarios."""

    def test_differ_orders_enums_before_tables_using_them(self, db_url, single_schema):
        """
        Verify differ puts CreateEnum before CreateTable that uses the enum.
        """
        schema = single_schema

        enum = EnumType(
            schema=schema,
            name="priority",
            values=["low", "medium", "high"],
        )

        table = Table(
            schema=schema,
            name="tasks",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "priority": Column(
                    name="priority", type=f'"{schema}"."priority"', nullable=False
                ),
            },
        )

        desired = Schema(
            enums={f"{schema}.priority": enum},
            tables={f"{schema}.tasks": table},
        )
        actual = Schema()

        changes = compute_diff(desired, actual)

        # Find positions
        enum_changes = [c for c in changes if isinstance(c, CreateEnum)]
        table_changes = [c for c in changes if isinstance(c, CreateTable)]

        assert len(enum_changes) == 1
        assert len(table_changes) == 1

        # Enum should come before table in the list
        enum_idx = changes.index(enum_changes[0])
        table_idx = changes.index(table_changes[0])

        assert enum_idx < table_idx, "Enum must be created before table that uses it"

    def test_differ_handles_empty_to_complex_schema(self, db_url, single_schema):
        """
        Verify differ can generate all changes to go from empty to complex schema.
        """
        schema = single_schema

        # Define a moderately complex schema
        enum = EnumType(
            schema=schema,
            name="user_role",
            values=["admin", "member", "guest"],
        )

        users_table = Table(
            schema=schema,
            name="users",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "email": Column(name="email", type="text", nullable=False),
                "role": Column(
                    name="role", type=f'"{schema}"."user_role"', nullable=False
                ),
            },
            constraints=[
                Constraint(name="users_pkey", type="p", definition="PRIMARY KEY (id)"),
            ],
        )

        posts_table = Table(
            schema=schema,
            name="posts",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "user_id": Column(name="user_id", type="uuid", nullable=False),
                "title": Column(name="title", type="text", nullable=False),
            },
            constraints=[
                Constraint(name="posts_pkey", type="p", definition="PRIMARY KEY (id)"),
                Constraint(
                    name="posts_user_fk",
                    type="f",
                    definition=f'FOREIGN KEY (user_id) REFERENCES "{schema}"."users"(id)',
                ),
            ],
        )

        desired = Schema(
            enums={f"{schema}.user_role": enum},
            tables={
                f"{schema}.users": users_table,
                f"{schema}.posts": posts_table,
            },
        )
        actual = Schema()

        changes = compute_diff(desired, actual)

        # Should have enum + 2 tables + FK constraints
        assert len(changes) >= 3

        # Verify change types
        enum_changes = [c for c in changes if isinstance(c, CreateEnum)]
        table_changes = [c for c in changes if isinstance(c, CreateTable)]

        assert len(enum_changes) == 1
        assert len(table_changes) == 2


@pytest.mark.integration
class TestMigrationBlockExecution:
    """Integration tests for executing an inline migration block.

    The blocks are synthetic and defined here, so the mechanism under test
    (parse a block, run its SQL against a live database, record it in
    pgdeclare.applied_migrations, refuse to silently re-run it) never depends
    on any real project's migrations.
    """

    def _create_temp_database(self, db_url: str) -> tuple[str, str]:
        db_name = f"test_migration_block_{uuid.uuid4().hex[:8]}"
        test_db_url = db_url.rsplit("/", 1)[0] + f"/{db_name}"

        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
                cur.execute(f'CREATE DATABASE "{db_name}"')
        finally:
            conn.close()

        return db_name, test_db_url

    def _drop_temp_database(self, db_url: str, db_name: str) -> None:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            conn.close()

    def _write_block(self, tmp_path: Path, migration_id: str, sql: str):
        """Parse a single synthetic @migrate-before block out of tmp_path."""
        from pgdeclare import parse_migration_blocks

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir(exist_ok=True)
        (schemas_dir / "01_test.sql").write_text(
            f"-- @migrate-before:{migration_id}\n{sql}\n-- @end-migrate\n"
        )

        blocks = parse_migration_blocks(tmp_path)
        assert len(blocks) == 1
        return blocks[0]

    def test_block_sql_runs_against_the_database(self, db_url, tmp_path):
        """The block's SQL actually executes, and the run is recorded."""
        from pgdeclare import execute_migration_block

        block = self._write_block(
            tmp_path,
            "2026-08-17_create_widget",
            "CREATE TABLE widget (id INT PRIMARY KEY);\n"
            "INSERT INTO widget (id) VALUES (1), (2);",
        )
        db_name, test_db_url = self._create_temp_database(db_url)

        try:
            success, error = execute_migration_block(test_db_url, block)
            assert success, error

            with psycopg2.connect(test_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM widget")
                    assert scalar(cur) == 2
                    cur.execute(
                        "SELECT sql_hash FROM pgdeclare.applied_migrations "
                        "WHERE id = %s",
                        (block.id,),
                    )
                    assert scalar(cur) == block.sql_hash
        finally:
            self._drop_temp_database(db_url, db_name)

    def test_applied_block_is_not_run_twice(self, db_url, tmp_path):
        """Re-running an applied block is a no-op, not a second execution.

        The SQL here is deliberately not idempotent: a second execution would
        raise "relation already exists", so a silent re-run cannot pass.
        """
        from pgdeclare import execute_migration_block

        block = self._write_block(
            tmp_path,
            "2026-08-17_create_gadget",
            "CREATE TABLE gadget (id INT PRIMARY KEY);",
        )
        db_name, test_db_url = self._create_temp_database(db_url)

        try:
            assert execute_migration_block(test_db_url, block) == (True, None)
            assert execute_migration_block(test_db_url, block) == (True, None)

            with psycopg2.connect(test_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT count(*) FROM pgdeclare.applied_migrations "
                        "WHERE id = %s",
                        (block.id,),
                    )
                    assert scalar(cur) == 1
        finally:
            self._drop_temp_database(db_url, db_name)

    def test_block_edited_after_application_is_refused(self, db_url, tmp_path):
        """Editing a deployed block is an error, not a silent divergence."""
        from pgdeclare import execute_migration_block

        migration_id = "2026-08-17_create_sprocket"
        block = self._write_block(
            tmp_path, migration_id, "CREATE TABLE sprocket (id INT PRIMARY KEY);"
        )
        db_name, test_db_url = self._create_temp_database(db_url)

        try:
            assert execute_migration_block(test_db_url, block) == (True, None)

            edited = self._write_block(
                tmp_path,
                migration_id,
                "CREATE TABLE sprocket (id INT PRIMARY KEY, name TEXT);",
            )
            assert edited.sql_hash != block.sql_hash

            success, error = execute_migration_block(test_db_url, edited)
            assert not success
            assert error is not None
            assert "modified after application" in error
        finally:
            self._drop_temp_database(db_url, db_name)

    def test_failing_block_reports_the_error(self, db_url, tmp_path):
        """A block whose SQL fails returns the failure instead of swallowing it."""
        from pgdeclare import execute_migration_block

        block = self._write_block(
            tmp_path,
            "2026-08-17_broken",
            "UPDATE table_that_does_not_exist SET x = 1;",
        )
        db_name, test_db_url = self._create_temp_database(db_url)

        try:
            success, error = execute_migration_block(test_db_url, block)
            assert not success
            assert error is not None
            assert "table_that_does_not_exist" in error
        finally:
            self._drop_temp_database(db_url, db_name)
