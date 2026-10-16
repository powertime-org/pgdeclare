"""
Integration tests for schema diff - Full migration cycle verification.

Tests that verify the differ correctly migrates from one schema state to another
by:
1. Setting up a "before" schema in the database
2. Defining a "target" desired schema
3. Running the differ to compute changes
4. Applying those changes
5. Introspecting the database and verifying it matches the target

These tests require a PostgreSQL database. Set TEST_DATABASE_URL to run.
"""

import os
import uuid
from contextlib import contextmanager


import pytest

import psycopg2

from pgdeclare.models import (
    Column,
    Constraint,
    Function,
    Grant,
    Index,
    Schema,
    Table,
    View,
)
from pgdeclare.changes import (
    AddColumn,
    AddConstraint,
    AlterColumnSetExpression,
    CreateFunction,
    CreateIndex,
    CreateTable,
    DropAndCreateFunction,
    RevokePublicExecute,
    ValidateConstraint,
)
from pgdeclare.differ import compute_diff
from pgdeclare.introspection import _query_columns, _query_indexes
from conftest import require_server_version
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
def test_schema_name():
    """Generate unique schema name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def clean_schema(db_url, test_schema_name):
    """Create a clean test schema and clean up after."""
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{test_schema_name}"')

    yield test_schema_name

    # Cleanup
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{test_schema_name}" CASCADE')


@contextmanager
def autocommit_connection(db_url: str):
    """Yield a connection that is genuinely outside a transaction block.

    `with psycopg2.connect(...) as conn: conn.autocommit = True` does NOT do
    this. Since psycopg2 2.9 the connection context manager opens a transaction
    on entry, so setting autocommit afterwards flips the client-side flag while
    the server is already in a transaction block. Statements that refuse to run
    there (CREATE DATABASE, DROP DATABASE, CREATE INDEX CONCURRENTLY) then fail
    with ActiveSqlTransaction.
    """
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def execute_sql(db_url: str, sql: str):
    """Execute SQL statement(s) against the database."""
    with psycopg2.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)


def introspect_table(db_url: str, schema: str, table_name: str) -> dict:
    """Introspect a single table's structure."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Get columns
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """,
                (schema, table_name),
            )
            columns = {
                row[0]: {
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3],
                }
                for row in cur.fetchall()
            }

            # Get constraints
            cur.execute(
                """
                SELECT tc.constraint_name, tc.constraint_type
                FROM information_schema.table_constraints tc
                WHERE tc.table_schema = %s AND tc.table_name = %s
            """,
                (schema, table_name),
            )
            constraints = {row[0]: row[1] for row in cur.fetchall()}

            # Get indexes
            cur.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = %s AND tablename = %s
            """,
                (schema, table_name),
            )
            indexes = [row[0] for row in cur.fetchall()]

            return {
                "columns": columns,
                "constraints": constraints,
                "indexes": indexes,
            }


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


def column_exists(db_url: str, schema: str, table_name: str, column_name: str) -> bool:
    """Check if a column exists."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = %s
                )
            """,
                (schema, table_name, column_name),
            )
            return scalar(cur)


def index_exists(db_url: str, schema: str, index_name: str) -> bool:
    """Check if an index exists."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = %s AND indexname = %s
                )
            """,
                (schema, index_name),
            )
            return scalar(cur)


def constraint_exists(
    db_url: str, schema: str, table_name: str, constraint_name: str
) -> bool:
    """Check if a constraint exists."""
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE table_schema = %s AND table_name = %s AND constraint_name = %s
                )
            """,
                (schema, table_name, constraint_name),
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
# Full Migration Cycle Tests
# =============================================================================


@pytest.mark.integration
class TestFullMigrationCycle:
    """Tests that verify complete migration cycles."""

    def test_create_table_and_introspect(self, db_url, clean_schema):
        """Create a table and verify it exists via introspection."""
        schema = clean_schema

        # Define desired schema with one table
        desired_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
                "email": Column(name="email", type="text", nullable=True),
            },
        )
        desired = Schema(tables={f"{schema}.account": desired_table})

        # Actual schema is empty
        actual = Schema()

        # Compute diff
        changes = compute_diff(desired, actual)

        # Should have one CreateTable change
        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 1

        # Apply the change
        sql = create_tables[0].to_sql()
        execute_sql(db_url, sql)

        # Verify table exists
        assert table_exists(db_url, schema, "account")

        # Verify columns
        info = introspect_table(db_url, schema, "account")
        assert "id" in info["columns"]
        assert "name" in info["columns"]
        assert "email" in info["columns"]

    def test_add_column_migration(self, db_url, clean_schema):
        """Add a column to an existing table."""
        schema = clean_schema

        # Create initial table
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."account" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name text NOT NULL
            )
        """,
        )

        # Define desired schema with additional column
        desired_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
                "email": Column(name="email", type="text", nullable=True),
            },
        )
        desired = Schema(tables={f"{schema}.account": desired_table})

        # Actual schema (what's in DB now)
        actual_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
            },
        )
        actual = Schema(tables={f"{schema}.account": actual_table})

        # Compute diff
        changes = compute_diff(desired, actual)

        # Should have one AddColumn change
        add_columns = [c for c in changes if isinstance(c, AddColumn)]
        assert len(add_columns) == 1
        assert add_columns[0].column.name == "email"

        # Apply the change
        sql = add_columns[0].to_sql()
        execute_sql(db_url, sql)

        # Verify column was added
        assert column_exists(db_url, schema, "account", "email")

    def test_fk_constraint_with_not_valid_and_validate(self, db_url, clean_schema):
        """Add FK constraint with NOT VALID, then validate."""
        schema = clean_schema

        # Create tables
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."parent" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid()
            );
            CREATE TABLE "{schema}"."child" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                parent_id uuid NOT NULL
            );
        """,
        )

        # Define constraint
        fk_constraint = Constraint(
            name="fk_parent",
            type="f",
            definition=f"FOREIGN KEY (parent_id) REFERENCES {schema}.parent(id)",
        )

        # Create AddConstraint change
        add_fk = AddConstraint(
            schema=schema,
            table="child",
            constraint=fk_constraint,
            not_valid=True,
        )

        # Apply NOT VALID constraint
        execute_sql(db_url, add_fk.to_sql())

        # Constraint should exist but not be validated
        assert constraint_exists(db_url, schema, "child", "fk_parent")

        # Now validate
        validate = ValidateConstraint(
            schema=schema,
            table="child",
            constraint_name="fk_parent",
        )
        execute_sql(db_url, validate.to_sql())

        # Constraint should still exist (now validated)
        assert constraint_exists(db_url, schema, "child", "fk_parent")

    def test_concurrent_index_creation(self, db_url, clean_schema):
        """Create index CONCURRENTLY."""
        schema = clean_schema

        # Create table
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."account" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                email text NOT NULL
            )
        """,
        )

        # Create index
        idx = Index(
            name="idx_account_email",
            table="account",
            schema=schema,
            definition=f'CREATE INDEX idx_account_email ON "{schema}"."account"(email)',
        )
        create_idx = CreateIndex(index=idx, concurrent=True)

        # Apply - note: CONCURRENTLY cannot run in a transaction, so this needs
        # a connection that is genuinely outside one.
        with autocommit_connection(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(create_idx.to_sql())

        # Verify index exists
        assert index_exists(db_url, schema, "idx_account_email")

    def test_function_replacement(self, db_url, clean_schema):
        """Replace a function with updated definition."""
        schema = clean_schema

        # Create initial function
        execute_sql(
            db_url,
            f"""
            CREATE OR REPLACE FUNCTION "{schema}"."get_greeting"()
            RETURNS text AS $$
            BEGIN
                RETURN 'Hello';
            END;
            $$ LANGUAGE plpgsql;
        """,
        )

        # Verify initial function exists
        assert function_exists(db_url, schema, "get_greeting")

        # Replace with new version
        new_func = Function(
            schema=schema,
            name="get_greeting",
            args="",
            full_definition=f"""
                CREATE OR REPLACE FUNCTION "{schema}"."get_greeting"()
                RETURNS text AS $$
                BEGIN
                    RETURN 'Hello, World!';
                END;
                $$ LANGUAGE plpgsql;
            """,
        )
        create_func = CreateFunction(function=new_func)
        execute_sql(db_url, create_func.to_sql())

        # Verify function still exists and returns new value
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT "{schema}"."get_greeting"()')
                result = scalar(cur)
                assert result == "Hello, World!"


@pytest.mark.integration
class TestPublicExecuteRevoke:
    """The locked-SECDEF PUBLIC-EXECUTE revoke executes correctly."""

    @staticmethod
    def _public_exec(db_url, schema, name):
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.proacl IS NULL OR EXISTS(
                        SELECT 1 FROM aclexplode(p.proacl) a
                        WHERE a.grantee = 0 AND a.privilege_type = 'EXECUTE')
                    FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = %s AND p.proname = %s
                    """,
                    (schema, name),
                )
                return scalar(cur)

    @staticmethod
    def _secdef_def(schema, name):
        # Use a named dollar-quote ($func$) like pg_get_functiondef emits, so it
        # does not collide with the DO $$ wrapper in DropAndCreateFunction.to_sql.
        return (
            f'CREATE FUNCTION "{schema}"."{name}"() RETURNS integer '
            f"LANGUAGE sql SECURITY DEFINER AS $func$ SELECT 1 $func$"
        )

    def test_create_function_atomic_revoke(self, db_url, clean_schema):
        """CreateFunction with the flag revokes PUBLIC EXECUTE atomically."""
        schema = clean_schema
        func = Function(
            schema=schema,
            name="sec_locked",
            args="",
            full_definition=self._secdef_def(schema, "sec_locked"),
            security_definer=True,
        )
        execute_sql(
            db_url, CreateFunction(function=func, revoke_public_execute=True).to_sql()
        )
        assert self._public_exec(db_url, schema, "sec_locked") is False

    def test_drop_and_create_atomic_revoke(self, db_url, clean_schema):
        """DropAndCreateFunction with the flag revokes after the recreate (clobber-safe)."""
        schema = clean_schema
        execute_sql(db_url, self._secdef_def(schema, "sec_recreated"))
        # Default ACL grants PUBLIC EXECUTE.
        assert self._public_exec(db_url, schema, "sec_recreated") is True
        func = Function(
            schema=schema,
            name="sec_recreated",
            args="",
            full_definition=self._secdef_def(schema, "sec_recreated"),
            security_definer=True,
        )
        execute_sql(
            db_url,
            DropAndCreateFunction(function=func, revoke_public_execute=True).to_sql(),
        )
        assert self._public_exec(db_url, schema, "sec_recreated") is False

    def test_standalone_revoke_public_execute(self, db_url, clean_schema):
        """RevokePublicExecute (drift path) revokes PUBLIC EXECUTE on an existing fn."""
        schema = clean_schema
        execute_sql(db_url, self._secdef_def(schema, "sec_drift"))
        assert self._public_exec(db_url, schema, "sec_drift") is True
        execute_sql(
            db_url,
            RevokePublicExecute(schema=schema, name="sec_drift", args="").to_sql(),
        )
        assert self._public_exec(db_url, schema, "sec_drift") is False


@pytest.mark.integration
class TestMigrationScenarios:
    """Tests for specific migration scenarios."""

    def test_add_multiple_columns(self, db_url, clean_schema):
        """Add multiple columns in a single migration."""
        schema = clean_schema

        # Create initial table
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."account" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid()
            )
        """,
        )

        # Define desired schema with multiple new columns
        desired_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=True),
                "email": Column(name="email", type="text", nullable=True),
                "created_at": Column(
                    name="created_at", type="timestamp with time zone", nullable=True
                ),
            },
        )
        desired = Schema(tables={f"{schema}.account": desired_table})

        actual_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
            },
        )
        actual = Schema(tables={f"{schema}.account": actual_table})

        # Compute diff
        changes = compute_diff(desired, actual)

        # Should have three AddColumn changes
        add_columns = [c for c in changes if isinstance(c, AddColumn)]
        assert len(add_columns) == 3

        # Apply all changes
        for change in add_columns:
            execute_sql(db_url, change.to_sql())

        # Verify all columns exist
        info = introspect_table(db_url, schema, "account")
        assert "name" in info["columns"]
        assert "email" in info["columns"]
        assert "created_at" in info["columns"]

    def test_create_table_with_fk_dependencies(self, db_url, clean_schema):
        """Tables with FK dependencies are handled by retry-until-stable execution.

        The executor discovers the correct order by retrying failed statements.
        This test verifies both tables can be created regardless of initial order.
        """
        schema = clean_schema

        # Define parent table
        parent = Table(
            schema=schema,
            name="parent",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
            },
            constraints=[
                Constraint(name="parent_pkey", type="p", definition="PRIMARY KEY (id)"),
            ],
        )

        # Define child table with FK to parent
        child = Table(
            schema=schema,
            name="child",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "parent_id": Column(name="parent_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(name="child_pkey", type="p", definition="PRIMARY KEY (id)"),
                Constraint(
                    name="fk_parent",
                    type="f",
                    definition=f'FOREIGN KEY (parent_id) REFERENCES "{schema}"."parent"(id)',
                ),
            ],
        )

        desired = Schema(
            tables={
                f"{schema}.parent": parent,
                f"{schema}.child": child,
            }
        )
        actual = Schema()

        # Compute diff
        changes = compute_diff(desired, actual)

        # There should be CreateTable changes for both
        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 2

        # For this test, manually execute in correct order since we're
        # not using the full executor pipeline
        # First, create parent (no dependencies)
        for c in create_tables:
            if c.table.name == "parent":
                execute_sql(db_url, c.to_sql())
        # Then create child (depends on parent)
        for c in create_tables:
            if c.table.name == "child":
                execute_sql(db_url, c.to_sql())

        # Both tables should exist
        assert table_exists(db_url, schema, "parent")
        assert table_exists(db_url, schema, "child")

    def test_multiple_table_creation(self, db_url, clean_schema):
        """Create multiple independent tables."""
        schema = clean_schema

        # Define multiple tables
        tables = {}
        for name in ["users", "products", "orders"]:
            tables[f"{schema}.{name}"] = Table(
                schema=schema,
                name=name,
                columns={
                    "id": Column(
                        name="id",
                        type="uuid",
                        nullable=False,
                        default="gen_random_uuid()",
                    ),
                    "name": Column(name="name", type="text", nullable=False),
                },
            )

        desired = Schema(tables=tables)
        actual = Schema()

        # Compute diff
        changes = compute_diff(desired, actual)

        # Should have three CreateTable changes
        create_tables = [c for c in changes if isinstance(c, CreateTable)]
        assert len(create_tables) == 3

        # Apply all changes
        for change in create_tables:
            execute_sql(db_url, change.to_sql())

        # Verify all tables exist
        for name in ["users", "products", "orders"]:
            assert table_exists(db_url, schema, name)


@pytest.mark.integration
class TestRolePasswordAndRetryIntegration:
    """Integration tests for role password extraction and retry executor handling."""

    def test_role_already_exists_handled_gracefully(self, db_url, clean_schema):
        """
        Verify retry executor handles "role already exists" errors gracefully.

        This is the key fix for CI validation: when building schema from git,
        roles persist across plan database drops. The retry executor should
        treat "role already exists" as a skip rather than a failure.
        """
        from pgdeclare.statement_parser import ParsedStatement
        from pgdeclare.retry_executor import execute_with_retry

        schema = clean_schema
        test_role_name = f"test_role_{schema}"  # Unique role name for test isolation

        # First, create the role (simulating it already exists from previous run)
        execute_sql(db_url, f'CREATE ROLE "{test_role_name}" WITH LOGIN;')

        try:
            # Now try to execute statements that include CREATE ROLE for same role
            # This simulates what happens in CI when the role already exists
            statements = [
                ParsedStatement(
                    raw_sql=f"CREATE ROLE \"{test_role_name}\" WITH LOGIN PASSWORD 'testpass';",
                    statement_type="CREATE ROLE",
                    object_type="ROLE",
                    object_name=test_role_name,
                    source_file="test.sql",
                    line_number=1,
                ),
                ParsedStatement(
                    raw_sql=f'CREATE TABLE "{schema}"."test_table" (id uuid PRIMARY KEY);',
                    statement_type="CREATE TABLE",
                    object_type="TABLE",
                    object_name=f"{schema}.test_table",
                    source_file="test.sql",
                    line_number=2,
                ),
            ]

            result = execute_with_retry(db_url, statements)

            # Should succeed - role error treated as "already exists, skip"
            assert result.success is True
            # Table should be created
            assert table_exists(db_url, schema, "test_table")

        finally:
            # Cleanup: drop the test role
            execute_sql(db_url, f'DROP ROLE IF EXISTS "{test_role_name}";')

    def test_role_password_extraction_and_injection(self, db_url, clean_schema):
        """
        Verify password extraction from SQL and injection into schema.

        This tests the fix for password dropping: passwords parsed from
        declarative SQL should be injected into introspected roles.
        """
        from pgdeclare.statement_parser import (
            ParsedStatement,
            parse_role_from_sql,
        )
        from pgdeclare.schema_parser import (
            _extract_role_passwords,
            _inject_role_passwords,
        )
        from pgdeclare.models import Schema, Role

        # Test password extraction from SQL
        sql = "CREATE ROLE app_user WITH LOGIN PASSWORD 'secret_pass';"
        role_info = parse_role_from_sql(sql)

        assert role_info is not None
        assert role_info["name"] == "app_user"
        assert role_info["password"] == "secret_pass"
        assert role_info["login"] is True

        # Test extraction from parsed statements
        statements = [
            ParsedStatement(
                raw_sql="CREATE ROLE app_user WITH LOGIN PASSWORD 'app_user';",
                statement_type="CREATE ROLE",
                object_type="ROLE",
                object_name="app_user",
                source_file="00_roles.sql",
                line_number=1,
            ),
            ParsedStatement(
                raw_sql="CREATE TABLE test (id int);",
                statement_type="CREATE TABLE",
                object_type="TABLE",
                object_name="public.test",
                source_file="01_tables.sql",
                line_number=1,
            ),
        ]

        passwords = _extract_role_passwords(statements)
        assert passwords == {"app_user": "app_user"}

        # Test injection into schema
        schema_obj = Schema(roles={"app_user": Role(name="app_user", can_login=True)})
        assert schema_obj.roles["app_user"].password is None

        _inject_role_passwords(schema_obj, passwords)
        assert schema_obj.roles["app_user"].password == "app_user"


@pytest.mark.integration
class TestRetryExecutorIntegration:
    """Integration tests for retry executor with real database."""

    def test_retry_resolves_forward_dependencies(self, db_url, clean_schema):
        """Retry executor resolves forward dependencies."""
        from pgdeclare.statement_parser import ParsedStatement
        from pgdeclare.retry_executor import execute_with_retry

        schema = clean_schema

        # Create statements in wrong order (child before parent)
        # The retry executor should handle this
        statements = [
            # Child table referencing parent (will fail first)
            ParsedStatement(
                raw_sql=f"""
                    CREATE TABLE "{schema}"."child" (
                        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                        parent_id uuid NOT NULL REFERENCES "{schema}"."parent"(id)
                    );
                """,
                statement_type="CREATE TABLE",
                object_type="TABLE",
                object_name=f"{schema}.child",
                source_file="test.sql",
                line_number=1,
            ),
            # Parent table
            ParsedStatement(
                raw_sql=f"""
                    CREATE TABLE "{schema}"."parent" (
                        id uuid PRIMARY KEY DEFAULT gen_random_uuid()
                    );
                """,
                statement_type="CREATE TABLE",
                object_type="TABLE",
                object_name=f"{schema}.parent",
                source_file="test.sql",
                line_number=2,
            ),
        ]

        result = execute_with_retry(db_url, statements, max_iterations=10)

        # Should succeed after retry
        assert result.success is True
        assert result.executed_count == 2

        # Both tables should exist
        assert table_exists(db_url, schema, "parent")
        assert table_exists(db_url, schema, "child")

    def test_syntax_error_reported_clearly(self, db_url, clean_schema):
        """Syntax errors are reported clearly in result."""
        from pgdeclare.statement_parser import ParsedStatement
        from pgdeclare.retry_executor import (
            execute_with_retry,
            ErrorCategory,
        )

        schema = clean_schema

        statements = [
            ParsedStatement(
                raw_sql=f'CRAETE TABLE "{schema}"."typo" (id int);',  # Typo
                statement_type="CREATE TABLE",
                object_type="TABLE",
                object_name=f"{schema}.typo",
                source_file="test.sql",
                line_number=1,
            ),
        ]

        result = execute_with_retry(db_url, statements)

        # Should fail with syntax error
        assert result.success is False
        assert len(result.failed_statements) == 1
        assert result.failed_statements[0].category == ErrorCategory.SYNTAX


@pytest.mark.integration
class TestSchemaVerification:
    """Tests that verify schema state after migrations."""

    def test_migration_produces_equivalent_schema(self, db_url, clean_schema):
        """
        Verify that applying diff changes produces the target schema.

        This is the key "migration verification" test - we:
        1. Start with a known "before" state
        2. Define a target state
        3. Compute and apply changes
        4. Verify the database matches the target
        """
        schema = clean_schema

        # === Step 1: Create "before" state ===
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."account" (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name text NOT NULL
            );
        """,
        )

        # === Step 2: Define target state ===
        target_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
                "email": Column(name="email", type="text", nullable=True),
                "created_at": Column(
                    name="created_at",
                    type="timestamp with time zone",
                    nullable=True,
                    default="now()",
                ),
            },
        )

        # === Step 3: Compute diff from current to target ===
        current_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
            },
        )

        desired_schema = Schema(tables={f"{schema}.account": target_table})
        actual_schema = Schema(tables={f"{schema}.account": current_table})

        changes = compute_diff(desired_schema, actual_schema)

        # === Step 4: Apply changes ===
        for change in changes:
            execute_sql(db_url, change.to_sql())

        # === Step 5: Verify database matches target ===
        info = introspect_table(db_url, schema, "account")

        # All target columns should exist
        assert "id" in info["columns"]
        assert "name" in info["columns"]
        assert "email" in info["columns"]
        assert "created_at" in info["columns"]

        # No extra columns
        assert len(info["columns"]) == 4

    def test_idempotent_migration(self, db_url, clean_schema):
        """
        Running the same migration twice should be safe.

        After the first migration, the diff should be empty.
        """
        schema = clean_schema

        # Define target
        target_table = Table(
            schema=schema,
            name="account",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
            },
        )
        target_schema = Schema(tables={f"{schema}.account": target_table})

        # First migration: from empty to target
        changes = compute_diff(target_schema, Schema())
        for change in changes:
            execute_sql(db_url, change.to_sql())

        # Second migration: target to target (should be no-op)
        changes_second = compute_diff(target_schema, target_schema)

        # No changes needed
        assert len(changes_second) == 0


@pytest.mark.integration
class TestRetryUntilStableDiffExecution:
    """
    Integration tests for retry-until-stable diff execution.

    These tests verify that execute_migration() correctly discovers
    execution order for changes with complex dependencies.
    """

    def test_function_depends_on_table(self, db_url, clean_schema):
        """
        Function referencing a table is deferred until table exists.

        Scenario: CreateFunction depends on CreateTable
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import (
            CreateTable,
            CreateFunction,
        )

        schema = clean_schema

        # Create a table and a function that queries it
        table = Table(
            schema=schema,
            name="accounts",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "balance": Column(name="balance", type="numeric", nullable=False),
            },
        )

        func = Function(
            schema=schema,
            name="get_total_balance",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."get_total_balance"()
                RETURNS numeric AS $$
                BEGIN
                    RETURN (SELECT COALESCE(SUM(balance), 0) FROM "{schema}"."accounts");
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Put function FIRST (wrong order) - executor should handle this
        changes = [
            CreateFunction(function=func),
            CreateTable(table=table),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert len(executed_sql) == 2
        assert table_exists(db_url, schema, "accounts")
        assert function_exists(db_url, schema, "get_total_balance")

    def test_trigger_depends_on_function(self, db_url, clean_schema):
        """
        Trigger referencing a function is deferred until function exists.

        Scenario: CreateTrigger depends on CreateFunction depends on CreateTable
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import (
            CreateTable,
            CreateFunction,
            CreateTrigger,
        )
        from pgdeclare.models import Trigger

        schema = clean_schema

        # Table for the trigger
        table = Table(
            schema=schema,
            name="audit_log",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "action": Column(name="action", type="text", nullable=True),
            },
        )

        # Trigger function
        func = Function(
            schema=schema,
            name="audit_trigger_func",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."audit_trigger_func"()
                RETURNS trigger AS $$
                BEGIN
                    NEW.action := 'modified';
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Trigger that calls the function
        trigger = Trigger(
            name="audit_trigger",
            table="audit_log",
            schema=schema,
            definition=f'''
                CREATE TRIGGER audit_trigger
                BEFORE INSERT ON "{schema}"."audit_log"
                FOR EACH ROW
                EXECUTE FUNCTION "{schema}"."audit_trigger_func"()
            ''',
        )

        # Put in worst possible order: trigger, function, table
        changes = [
            CreateTrigger(trigger=trigger),
            CreateFunction(function=func),
            CreateTable(table=table),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert len(executed_sql) == 3
        assert table_exists(db_url, schema, "audit_log")
        assert function_exists(db_url, schema, "audit_trigger_func")

    def test_multiple_fk_chain(self, db_url, clean_schema):
        """
        Chain of FK dependencies: C -> B -> A.

        Tables must be created in order A, B, C regardless of input order.
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import CreateTable

        schema = clean_schema

        # Table A (no deps)
        table_a = Table(
            schema=schema,
            name="table_a",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
            },
            constraints=[
                Constraint(
                    name="table_a_pkey", type="p", definition="PRIMARY KEY (id)"
                ),
            ],
        )

        # Table B (depends on A)
        table_b = Table(
            schema=schema,
            name="table_b",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "a_id": Column(name="a_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="table_b_pkey", type="p", definition="PRIMARY KEY (id)"
                ),
                Constraint(
                    name="fk_a",
                    type="f",
                    definition=f'FOREIGN KEY (a_id) REFERENCES "{schema}"."table_a"(id)',
                ),
            ],
        )

        # Table C (depends on B)
        table_c = Table(
            schema=schema,
            name="table_c",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "b_id": Column(name="b_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="table_c_pkey", type="p", definition="PRIMARY KEY (id)"
                ),
                Constraint(
                    name="fk_b",
                    type="f",
                    definition=f'FOREIGN KEY (b_id) REFERENCES "{schema}"."table_b"(id)',
                ),
            ],
        )

        # Worst possible order: C, B, A
        changes = [
            CreateTable(table=table_c),
            CreateTable(table=table_b),
            CreateTable(table=table_a),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert len(executed_sql) == 3
        assert table_exists(db_url, schema, "table_a")
        assert table_exists(db_url, schema, "table_b")
        assert table_exists(db_url, schema, "table_c")

    def test_function_used_as_column_default(self, db_url, clean_schema):
        """
        Function used as column default must be created before table.

        This was a known edge case with priority-based ordering.
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import (
            CreateTable,
            CreateFunction,
        )

        schema = clean_schema

        # Function that generates a default value
        func = Function(
            schema=schema,
            name="generate_code",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."generate_code"()
                RETURNS text AS $$
                BEGIN
                    RETURN 'CODE-' || floor(random() * 10000)::text;
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Table using function as default
        table = Table(
            schema=schema,
            name="items",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "code": Column(
                    name="code",
                    type="text",
                    nullable=False,
                    default=f'"{schema}"."generate_code"()',
                ),
            },
        )

        # Table first (wrong order) - executor should handle this
        changes = [
            CreateTable(table=table),
            CreateFunction(function=func),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert len(executed_sql) == 2
        assert function_exists(db_url, schema, "generate_code")
        assert table_exists(db_url, schema, "items")

        # Verify the default actually works
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'INSERT INTO "{schema}"."items" (id) VALUES (gen_random_uuid()) RETURNING code'
                )
                code = scalar(cur)
                assert code.startswith("CODE-")

    def test_policy_depends_on_function(self, db_url, clean_schema):
        """
        RLS policy using a function is deferred until function exists.
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import (
            CreateTable,
            CreateFunction,
            CreatePolicy,
            EnableRLS,
        )
        from pgdeclare.models import Policy

        schema = clean_schema

        # Table for RLS
        table = Table(
            schema=schema,
            name="documents",
            columns={
                "id": Column(name="id", type="uuid", nullable=False),
                "owner_id": Column(name="owner_id", type="uuid", nullable=False),
            },
        )

        # Function used in policy
        func = Function(
            schema=schema,
            name="current_user_id",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."current_user_id"()
                RETURNS uuid AS $$
                BEGIN
                    RETURN '00000000-0000-0000-0000-000000000001'::uuid;
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Policy using the function
        policy = Policy(
            name="owner_policy",
            table="documents",
            schema=schema,
            command="SELECT",
            using_expr=f'owner_id = "{schema}"."current_user_id"()',
        )

        # Policy first (wrong order)
        changes = [
            CreatePolicy(policy=policy),
            EnableRLS(schema=schema, table="documents"),
            CreateFunction(function=func),
            CreateTable(table=table),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert table_exists(db_url, schema, "documents")
        assert function_exists(db_url, schema, "current_user_id")

    def test_stuck_detection_missing_dependency(self, db_url, clean_schema):
        """
        Executor detects when changes are stuck due to missing external dependency.

        Note: We use a SQL function (not plpgsql) because SQL functions validate
        table references at creation time, while plpgsql defers validation to runtime.
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import CreateFunction

        schema = clean_schema

        # SQL function referencing non-existent table (validated at creation time)
        func = Function(
            schema=schema,
            name="broken_func",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."broken_func"()
                RETURNS bigint
                LANGUAGE sql
                AS $$ SELECT count(*) FROM "{schema}"."nonexistent_table" $$;
            ''',
        )

        changes = [CreateFunction(function=func)]

        success, executed_sql = execute_migration(db_url, list(changes))

        # Should fail because nonexistent_table doesn't exist and won't be created
        assert success is False
        assert len(executed_sql) == 0

    def test_mixed_dependencies_complex_scenario(self, db_url, clean_schema):
        """
        Complex scenario with multiple interrelated dependencies.

        Tables: users, posts (FK to users), comments (FK to posts)
        Functions: get_user_post_count (refs users, posts)
        Trigger: update_timestamp on posts
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.changes import (
            CreateTable,
            CreateFunction,
            CreateTrigger,
        )
        from pgdeclare.models import Trigger

        schema = clean_schema

        # Users table
        users = Table(
            schema=schema,
            name="users",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "name": Column(name="name", type="text", nullable=False),
            },
            constraints=[
                Constraint(name="users_pkey", type="p", definition="PRIMARY KEY (id)"),
            ],
        )

        # Posts table (FK to users)
        posts = Table(
            schema=schema,
            name="posts",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "user_id": Column(name="user_id", type="uuid", nullable=False),
                "title": Column(name="title", type="text", nullable=False),
                "updated_at": Column(
                    name="updated_at", type="timestamptz", nullable=True
                ),
            },
            constraints=[
                Constraint(name="posts_pkey", type="p", definition="PRIMARY KEY (id)"),
                Constraint(
                    name="fk_user",
                    type="f",
                    definition=f'FOREIGN KEY (user_id) REFERENCES "{schema}"."users"(id)',
                ),
            ],
        )

        # Comments table (FK to posts)
        comments = Table(
            schema=schema,
            name="comments",
            columns={
                "id": Column(
                    name="id", type="uuid", nullable=False, default="gen_random_uuid()"
                ),
                "post_id": Column(name="post_id", type="uuid", nullable=False),
                "body": Column(name="body", type="text", nullable=False),
            },
            constraints=[
                Constraint(
                    name="comments_pkey", type="p", definition="PRIMARY KEY (id)"
                ),
                Constraint(
                    name="fk_post",
                    type="f",
                    definition=f'FOREIGN KEY (post_id) REFERENCES "{schema}"."posts"(id)',
                ),
            ],
        )

        # Function that queries users and posts
        func = Function(
            schema=schema,
            name="get_user_post_count",
            args="p_user_id uuid",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."get_user_post_count"(p_user_id uuid)
                RETURNS bigint AS $$
                BEGIN
                    RETURN (
                        SELECT count(*)
                        FROM "{schema}"."posts" p
                        WHERE p.user_id = p_user_id
                    );
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Trigger function for updating timestamp
        trigger_func = Function(
            schema=schema,
            name="update_timestamp",
            args="",
            full_definition=f'''
                CREATE OR REPLACE FUNCTION "{schema}"."update_timestamp"()
                RETURNS trigger AS $$
                BEGIN
                    NEW.updated_at := now();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            ''',
        )

        # Trigger on posts
        trigger = Trigger(
            name="posts_update_timestamp",
            table="posts",
            schema=schema,
            definition=f'''
                CREATE TRIGGER posts_update_timestamp
                BEFORE UPDATE ON "{schema}"."posts"
                FOR EACH ROW
                EXECUTE FUNCTION "{schema}"."update_timestamp"()
            ''',
        )

        # Worst possible order
        changes = [
            CreateTrigger(trigger=trigger),
            CreateFunction(function=func),
            CreateTable(table=comments),
            CreateFunction(function=trigger_func),
            CreateTable(table=posts),
            CreateTable(table=users),
        ]

        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        assert len(executed_sql) == 6
        assert table_exists(db_url, schema, "users")
        assert table_exists(db_url, schema, "posts")
        assert table_exists(db_url, schema, "comments")
        assert function_exists(db_url, schema, "get_user_post_count")
        assert function_exists(db_url, schema, "update_timestamp")


@pytest.mark.integration
class TestGeneratedColumnsIntegration:
    """End-to-end coverage for STORED generated columns."""

    def _make_table(self, db_url, schema):
        """Create an IMMUTABLE helper + a table with a STORED generated column."""
        execute_sql(
            db_url,
            f"""
            CREATE FUNCTION "{schema}".double_it(integer) RETURNS integer
                LANGUAGE sql IMMUTABLE AS 'SELECT $1 * 2';
            CREATE TABLE "{schema}"."t" (
                n integer NOT NULL,
                n2 integer GENERATED ALWAYS AS ("{schema}".double_it(n)) STORED
            );
            INSERT INTO "{schema}"."t" (n) VALUES (21);
            """,
        )

    def _columns(self, db_url, schema, table):
        with psycopg2.connect(db_url) as conn:
            return _query_columns(conn, schema, table)

    def test_introspect_generated_column_round_trips(self, db_url, clean_schema):
        """A STORED generated column introspects as `generated`, not `default`,
        and re-diffing the introspected schema against itself yields no change."""
        schema = clean_schema
        self._make_table(db_url, schema)

        cols = self._columns(db_url, schema, "t")
        assert cols["n2"].generated is not None
        assert "double_it" in cols["n2"].generated
        # The generation expression must NOT leak into `default`.
        assert cols["n2"].default is None

        actual = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=cols)}
        )
        desired = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=dict(cols))}
        )
        # No spurious DEFAULT-mangling churn.
        assert compute_diff(desired, actual) == []

    def test_changed_expression_applies_via_set_expression(self, db_url, clean_schema):
        """Changing the generation expression emits SET EXPRESSION (no
        drop/CASCADE), applies cleanly, and recomputes existing rows."""
        require_server_version(db_url, 170000, "ALTER COLUMN ... SET EXPRESSION")
        schema = clean_schema
        self._make_table(db_url, schema)
        execute_sql(
            db_url,
            f"""CREATE FUNCTION "{schema}".triple_it(integer) RETURNS integer
                LANGUAGE sql IMMUTABLE AS 'SELECT $1 * 3';""",
        )

        cols = self._columns(db_url, schema, "t")
        actual = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=cols)}
        )

        new_cols = dict(cols)
        new_cols["n2"] = Column(
            name="n2",
            type=cols["n2"].type,
            nullable=cols["n2"].nullable,
            generated=f'"{schema}".triple_it(n)',
        )
        desired = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=new_cols)}
        )

        changes = compute_diff(desired, actual)
        set_expr = [c for c in changes if isinstance(c, AlterColumnSetExpression)]
        assert len(set_expr) == 1
        execute_sql(db_url, set_expr[0].to_sql())

        # Existing row recomputed: 21 -> 63 (was 42 under double_it).
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT n2 FROM "{schema}"."t"')
                assert scalar(cur) == 63

        # Re-introspect: the column is still generated, now on triple_it.
        recols = self._columns(db_url, schema, "t")
        assert recols["n2"].generated is not None
        assert "triple_it" in recols["n2"].generated
        assert recols["n2"].default is None

    def test_plain_to_generated_fails_fast(self, db_url, clean_schema):
        """Converting an existing plain column to generated raises, rather than
        emitting an unsafe drop+add."""
        schema = clean_schema
        execute_sql(
            db_url,
            f"""
            CREATE FUNCTION "{schema}".double_it(integer) RETURNS integer
                LANGUAGE sql IMMUTABLE AS 'SELECT $1 * 2';
            CREATE TABLE "{schema}"."t" (n integer NOT NULL, n2 integer);
            """,
        )
        cols = self._columns(db_url, schema, "t")
        actual = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=cols)}
        )
        new_cols = dict(cols)
        new_cols["n2"] = Column(
            name="n2",
            type=cols["n2"].type,
            nullable=cols["n2"].nullable,
            generated=f'"{schema}".double_it(n)',
        )
        desired = Schema(
            tables={f"{schema}.t": Table(schema=schema, name="t", columns=new_cols)}
        )
        with pytest.raises(ValueError, match="stored generated column"):
            compute_diff(desired, actual)


@pytest.mark.integration
class TestRecreatedViewKeepsGrants:
    """A view recreate must not lose its ACL for a converge cycle.

    The differ lists views before grants, but the executor runs changes
    retry-until-stable: a recreate whose new definition depends on an object
    built later in the same run is deferred to a later iteration and lands
    AFTER the standalone GRANT. The DROP then wipes the ACL, the run still
    reports success, and the view is owner-only ("permission denied for view")
    until the next converge. The grants therefore ride inside the recreate's
    own DO block.
    """

    @staticmethod
    def _relacl(db_url, schema, name):
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.relacl::text
                    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s
                    """,
                    (schema, name),
                )
                return scalar(cur)

    @staticmethod
    def _can_select(db_url, role, schema, name):
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT has_table_privilege(%s, %s, 'SELECT')",
                    (role, f'"{schema}"."{name}"'),
                )
                return scalar(cur)

    def test_deferred_view_recreate_keeps_its_grants(self, db_url, clean_schema):
        """A recreate deferred behind a new dependency keeps its grants."""
        from pgdeclare.executor import execute_migration

        schema = clean_schema
        role = f"reader_{uuid.uuid4().hex[:8]}"
        execute_sql(db_url, f'CREATE ROLE "{role}"')
        try:
            execute_sql(
                db_url,
                f"""
                CREATE TABLE "{schema}"."t_base" (id integer, a integer);
                CREATE VIEW "{schema}"."v_granted" AS
                    SELECT id, a FROM "{schema}"."t_base";
                GRANT USAGE ON SCHEMA "{schema}" TO "{role}";
                GRANT SELECT ON "{schema}"."v_granted" TO "{role}";
                """,
            )
            assert self._can_select(db_url, role, schema, "v_granted") is True

            grant = Grant(
                object_type="TABLE",
                object_name="v_granted",
                schema=schema,
                privilege="SELECT",
                grantee=role,
            )
            old = View(
                schema=schema,
                name="v_granted",
                definition=f'SELECT id, a FROM "{schema}"."t_base"',
            )
            # The new definition reads from a view this same run creates, so
            # PostgreSQL rejects the recreate on the first pass and the executor
            # defers it behind the GRANT.
            new = View(
                schema=schema,
                name="v_granted",
                definition=f'SELECT id, a FROM "{schema}"."v_new"',
            )
            dependency = View(
                schema=schema,
                name="v_new",
                definition=f'SELECT id, a FROM "{schema}"."t_base"',
            )
            actual = Schema(views={f"{schema}.v_granted": old}, grants=[grant])
            desired = Schema(
                views={
                    f"{schema}.v_granted": new,
                    f"{schema}.v_new": dependency,
                },
                grants=[grant],
            )

            changes = compute_diff(desired, actual)
            success, executed_sql = execute_migration(db_url, list(changes))

            assert success is True
            # The recreate really did run last, which is the ordering that used
            # to lose the ACL.
            assert "DROP VIEW IF EXISTS" in executed_sql[-1]
            assert self._can_select(db_url, role, schema, "v_granted") is True
            assert role in (self._relacl(db_url, schema, "v_granted") or "")
        finally:
            execute_sql(db_url, f'DROP OWNED BY "{role}"')
            execute_sql(db_url, f'DROP ROLE "{role}"')


@pytest.mark.integration
class TestViewSecurityInvoker:
    """A declared `WITH (security_invoker = TRUE)` must actually apply.

    A differ that modelled a view as schema/name/definition only would drop the
    option on every trip through the plan database, and `View.__eq__` could not
    see the drift. When views are owned by a BYPASSRLS role, owner-rights views
    mean RLS on the base tables never applies to anything read through a view:
    it fails open, silently.
    """

    @pytest.fixture(autouse=True)
    def _needs_security_invoker(self, db_url):
        require_server_version(db_url, 150000, "security_invoker views")

    @staticmethod
    def _reloptions(db_url, schema, name):
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.reloptions
                    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s AND c.relkind = 'v'
                    """,
                    (schema, name),
                )
                return scalar(cur) or []

    def test_existing_view_is_backfilled_in_one_run(self, db_url, clean_schema):
        """The backfill path: definition already matches, only the option drifted.

        This is the state of every database built before this fix. There is no
        definition change, so a drop-and-recreate never fires and the drift would
        persist forever. The differ has to notice it and repair it in place.
        """
        from pgdeclare.executor import execute_migration

        schema = clean_schema
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."t_base" (id integer, owner_id integer);
            CREATE VIEW "{schema}"."v_scoped" AS
                SELECT id, owner_id FROM "{schema}"."t_base";
            """,
        )
        assert self._reloptions(db_url, schema, "v_scoped") == []

        definition = f'SELECT id, owner_id FROM "{schema}"."t_base"'
        actual = Schema(
            views={
                f"{schema}.v_scoped": View(
                    schema=schema, name="v_scoped", definition=definition
                )
            }
        )
        desired = Schema(
            views={
                f"{schema}.v_scoped": View(
                    schema=schema,
                    name="v_scoped",
                    definition=definition,
                    reloptions=["security_invoker=true"],
                )
            }
        )

        changes = compute_diff(desired, actual)
        assert len(changes) == 1
        success, _ = execute_migration(db_url, list(changes))

        assert success is True
        assert self._reloptions(db_url, schema, "v_scoped") == ["security_invoker=true"]

        # And it converges: a second diff against the repaired state is clean.
        assert compute_diff(desired, desired) == []

    def test_deferred_view_recreate_keeps_its_options(self, db_url, clean_schema):
        """A recreate deferred behind a new dependency keeps its security mode.

        Same failure shape as the grants on a recreated view: the executor runs
        changes retry-until-stable, so a trailing standalone
        `ALTER VIEW ... SET (...)` would land on the pre-drop view and be thrown
        away by the DROP. The option rides inside the recreate's own DO block.
        """
        from pgdeclare.executor import execute_migration

        schema = clean_schema
        execute_sql(
            db_url,
            f"""
            CREATE TABLE "{schema}"."t_base" (id integer, a integer);
            CREATE VIEW "{schema}"."v_moving" WITH (security_invoker = TRUE) AS
                SELECT id, a FROM "{schema}"."t_base";
            """,
        )

        old = View(
            schema=schema,
            name="v_moving",
            definition=f'SELECT id, a FROM "{schema}"."t_base"',
            reloptions=["security_invoker=true"],
        )
        # The new definition reads a view this same run creates, so PostgreSQL
        # rejects the recreate on the first pass and the executor defers it.
        new = View(
            schema=schema,
            name="v_moving",
            definition=f'SELECT id, a FROM "{schema}"."v_new"',
            reloptions=["security_invoker=true"],
        )
        dependency = View(
            schema=schema,
            name="v_new",
            definition=f'SELECT id, a FROM "{schema}"."t_base"',
            reloptions=["security_invoker=true"],
        )
        actual = Schema(views={f"{schema}.v_moving": old})
        desired = Schema(
            views={f"{schema}.v_moving": new, f"{schema}.v_new": dependency}
        )

        changes = compute_diff(desired, actual)
        success, executed_sql = execute_migration(db_url, list(changes))

        assert success is True
        # The recreate really did run last, the ordering that used to lose it.
        assert "DROP VIEW IF EXISTS" in executed_sql[-1]
        assert self._reloptions(db_url, schema, "v_moving") == ["security_invoker=true"]

    def test_rls_applies_through_the_view_for_a_non_superuser(
        self, db_url, clean_schema
    ):
        """The claim that matters, end to end.

        A non-superuser role reads a table through a view the differ built. With
        the declared security mode in force it sees only the rows RLS allows it.
        Without it, it sees everything, because the view owner is BYPASSRLS.
        """
        from pgdeclare.executor import execute_migration

        schema = clean_schema
        role = f"reader_{uuid.uuid4().hex[:8]}"
        execute_sql(db_url, f'CREATE ROLE "{role}"')
        try:
            execute_sql(
                db_url,
                f"""
                CREATE TABLE "{schema}"."secret" (id integer, owner text);
                INSERT INTO "{schema}"."secret" VALUES (1, 'mine'), (2, 'theirs');
                ALTER TABLE "{schema}"."secret" ENABLE ROW LEVEL SECURITY;
                CREATE POLICY only_mine ON "{schema}"."secret" FOR SELECT
                    USING (owner = current_setting('test.owner', true));
                GRANT USAGE ON SCHEMA "{schema}" TO "{role}";
                GRANT SELECT ON "{schema}"."secret" TO "{role}";
                """,
            )

            definition = f'SELECT id, owner FROM "{schema}"."secret"'
            view = View(
                schema=schema,
                name="v_secret",
                definition=definition,
                reloptions=["security_invoker=true"],
            )
            grant = Grant(
                object_type="TABLE",
                object_name="v_secret",
                schema=schema,
                privilege="SELECT",
                grantee=role,
            )
            desired = Schema(views={f"{schema}.v_secret": view}, grants=[grant])

            success, _ = execute_migration(
                db_url, list(compute_diff(desired, Schema()))
            )
            assert success is True

            def rows_seen_by_role():
                with psycopg2.connect(db_url) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute("SELECT set_config('test.owner', 'mine', false)")
                        cur.execute(f'SET ROLE "{role}"')
                        cur.execute(f'SELECT count(*) FROM "{schema}"."v_secret"')
                        return scalar(cur)

            # The view was created by the differ, and RLS reaches through it.
            assert self._reloptions(db_url, schema, "v_secret") == [
                "security_invoker=true"
            ]
            assert rows_seen_by_role() == 1

            # Drop the option and the same role reads every row: the leak.
            execute_sql(
                db_url, f'ALTER VIEW "{schema}"."v_secret" RESET (security_invoker)'
            )
            assert rows_seen_by_role() == 2

            # And the differ puts it back on the next converge, with no
            # definition change to prompt it.
            actual = Schema(
                views={
                    f"{schema}.v_secret": View(
                        schema=schema, name="v_secret", definition=definition
                    )
                },
                grants=[grant],
            )
            success, _ = execute_migration(db_url, list(compute_diff(desired, actual)))
            assert success is True
            assert rows_seen_by_role() == 1
        finally:
            execute_sql(db_url, f'DROP OWNED BY "{role}"')
            execute_sql(db_url, f'DROP ROLE "{role}"')


@pytest.mark.integration
class TestFreshBuildMatchesTheDeclaredSchema:
    """The fresh-build path must land every declared constraint.

    The differ has two ways to reach a constraint. A table that does not exist
    yet is written by CreateTable; a table that does exist goes through
    _diff_constraints and gets an ALTER TABLE. Those two paths can disagree for
    `enrollment`, whose declared UNIQUE (course_id, student_id) duplicates its
    primary key: PostgreSQL's transformIndexConstraints() silently discards a
    redundant index constraint declared INLINE, so an inlining fresh build loses
    it and the next diff adds it back.

    These tests introspect the real database after applying. The older
    test_idempotent_migration diffs a target against itself, which cannot see
    this class of defect at all: the database is never asked what it got.
    """

    @staticmethod
    def _redundant_unique_table(schema: str) -> Table:
        return Table(
            schema=schema,
            name="enrollment",
            columns={
                "course_id": Column(name="course_id", type="uuid", nullable=False),
                "student_id": Column(name="student_id", type="uuid", nullable=False),
            },
            constraints=[
                Constraint(
                    name="enrollment_course_id_student_id_key",
                    type="u",
                    definition="UNIQUE (course_id, student_id)",
                    table="enrollment",
                ),
                Constraint(
                    name="enrollment_pkey",
                    type="p",
                    definition="PRIMARY KEY (course_id, student_id)",
                    table="enrollment",
                ),
            ],
        )

    @staticmethod
    def _constraint_types(db_url: str, schema: str, table: str) -> dict:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT con.conname, con.contype::text
                    FROM pg_constraint con
                    JOIN pg_class c ON c.oid = con.conrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s
                    """,
                    (schema, table),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def test_unique_redundant_with_the_primary_key_survives_a_fresh_build(
        self, db_url, clean_schema
    ):
        """A UNIQUE that duplicates the PK column set must still be created.

        Before the fix CreateTable inlined both, PostgreSQL kept only the
        primary key, and the database came out one constraint short of what was
        declared, with no error and no notice.
        """
        from pgdeclare.executor import execute_migration

        schema = clean_schema
        desired = Schema(
            tables={f"{schema}.enrollment": self._redundant_unique_table(schema)}
        )

        changes = compute_diff(desired, Schema())
        success, _ = execute_migration(db_url, list(changes))
        assert success is True

        landed = self._constraint_types(db_url, schema, "enrollment")
        assert landed.get("enrollment_pkey") == "p"
        assert landed.get("enrollment_course_id_student_id_key") == "u", (
            "the declared UNIQUE was silently dropped by the fresh-build path; "
            f"the table came out with {sorted(landed)}"
        )

    def test_a_fresh_build_converges_in_one_run(self, db_url, clean_schema):
        """A second diff against the built database must find nothing.

        Convergence in one run, asserted against a real database rather than
        against the desired schema object.
        """
        from pgdeclare.executor import execute_migration
        from pgdeclare.introspection import introspect_schema

        schema = clean_schema
        desired = Schema(
            tables={f"{schema}.enrollment": self._redundant_unique_table(schema)}
        )

        success, _ = execute_migration(db_url, list(compute_diff(desired, Schema())))
        assert success is True

        actual = introspect_schema(connection_string=db_url, schemas=[schema])
        residual = compute_diff(desired, actual, schemas=[schema])

        # Grants on public / app_generated / extensions belong to the host
        # database this test borrows, not to the fixture, so scope the
        # assertion to the throwaway schema the test actually declared.
        mine = [c for c in residual if schema in (c.object_name or "")]

        assert mine == [], (
            "one run did not reach the declared schema; residual: "
            f"{[c.description for c in mine]}"
        )


def test_same_index_name_in_two_schemas_is_not_crossed(db_url, clean_schema):
    """An index is matched to its own schema, not a same-named one elsewhere.

    The other schema's index is left invalid by a failed concurrent build, so
    a cross-schema join would report the wrong validity for one of them.
    """
    schema = clean_schema
    other = f"{schema}_other"
    execute_sql(db_url, f'CREATE SCHEMA "{other}"')
    try:
        for name in (schema, other):
            execute_sql(db_url, f'CREATE TABLE "{name}".t (v int)')
        execute_sql(db_url, f'CREATE INDEX idx_v ON "{schema}".t (v)')
        execute_sql(db_url, f'INSERT INTO "{other}".t VALUES (1), (1)')
        with autocommit_connection(db_url) as conn:
            with conn.cursor() as cur:
                with pytest.raises(psycopg2.errors.UniqueViolation):
                    cur.execute(
                        f'CREATE UNIQUE INDEX CONCURRENTLY idx_v ON "{other}".t (v)'
                    )

        with psycopg2.connect(db_url) as conn:
            mine = _query_indexes(conn, schema)
            theirs = _query_indexes(conn, other)
        (mine_idx,) = [i for i in mine.values() if i.name == "idx_v"]
        (their_idx,) = [i for i in theirs.values() if i.name == "idx_v"]
        assert mine_idx.is_valid is True
        assert their_idx.is_valid is False
        assert f"{other}.t" not in mine_idx.definition
    finally:
        execute_sql(db_url, f'DROP SCHEMA IF EXISTS "{other}" CASCADE')
