"""
Tests for retry_executor.py - Retry-until-stable SQL execution.

Covers:
- Error categorization by PostgreSQL error codes
- Execute with retry logic
- Duplicate detection
- Idempotent statement handling (roles, extensions)
- Error formatting
"""

from unittest.mock import MagicMock, patch


from pgdeclare.retry_executor import (
    ErrorCategory,
    ERROR_CODE_CATEGORIES,
    IDEMPOTENT_DUPLICATE_STATEMENTS,
    FailedStatement,
    DuplicateDefinition,
    ExecutionResult,
    execute_with_retry,
    format_execution_errors,
    _format_location,
)
from pgdeclare.statement_parser import ParsedStatement

from conftest import (
    PgError,
    PGCODE_UNDEFINED_TABLE,
    PGCODE_UNDEFINED_FUNCTION,
    PGCODE_UNDEFINED_OBJECT,
    PGCODE_UNDEFINED_COLUMN,
    PGCODE_DUPLICATE_TABLE,
    PGCODE_DUPLICATE_OBJECT,
    PGCODE_DUPLICATE_FUNCTION,
    PGCODE_SYNTAX_ERROR,
    PGCODE_PERMISSION_DENIED,
    PGCODE_INVALID_FK,
)


class TestErrorCategorization:
    """Tests for PostgreSQL error code categorization."""

    def test_undefined_table_is_dependency(self):
        """42P01 (undefined_table) is a dependency error."""
        assert ERROR_CODE_CATEGORIES[PGCODE_UNDEFINED_TABLE] == ErrorCategory.DEPENDENCY

    def test_undefined_function_is_dependency(self):
        """42883 (undefined_function) is a dependency error."""
        assert (
            ERROR_CODE_CATEGORIES[PGCODE_UNDEFINED_FUNCTION] == ErrorCategory.DEPENDENCY
        )

    def test_undefined_object_is_dependency(self):
        """42704 (undefined_object) is a dependency error."""
        assert (
            ERROR_CODE_CATEGORIES[PGCODE_UNDEFINED_OBJECT] == ErrorCategory.DEPENDENCY
        )

    def test_undefined_column_is_dependency(self):
        """42703 (undefined_column) is a dependency error."""
        assert (
            ERROR_CODE_CATEGORIES[PGCODE_UNDEFINED_COLUMN] == ErrorCategory.DEPENDENCY
        )

    def test_invalid_fk_is_dependency(self):
        """42830 (invalid_foreign_key) is a dependency error."""
        assert ERROR_CODE_CATEGORIES[PGCODE_INVALID_FK] == ErrorCategory.DEPENDENCY

    def test_duplicate_table_is_duplicate(self):
        """42P07 (duplicate_table) is a duplicate error."""
        assert ERROR_CODE_CATEGORIES[PGCODE_DUPLICATE_TABLE] == ErrorCategory.DUPLICATE

    def test_duplicate_object_is_duplicate(self):
        """42710 (duplicate_object) is a duplicate error."""
        assert ERROR_CODE_CATEGORIES[PGCODE_DUPLICATE_OBJECT] == ErrorCategory.DUPLICATE

    def test_duplicate_function_is_duplicate(self):
        """42723 (duplicate_function) is a duplicate error."""
        assert (
            ERROR_CODE_CATEGORIES[PGCODE_DUPLICATE_FUNCTION] == ErrorCategory.DUPLICATE
        )

    def test_syntax_error_is_syntax(self):
        """42601 (syntax_error) is a syntax error."""
        assert ERROR_CODE_CATEGORIES[PGCODE_SYNTAX_ERROR] == ErrorCategory.SYNTAX

    def test_permission_denied_is_permission(self):
        """42501 (insufficient_privilege) is a permission error."""
        assert (
            ERROR_CODE_CATEGORIES[PGCODE_PERMISSION_DENIED] == ErrorCategory.PERMISSION
        )

    def test_unknown_code_defaults_to_other(self):
        """Unknown error codes default to OTHER category."""
        # FailedStatement should categorize unknown codes as OTHER
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
        )
        error = PgError("Unknown error", "99999")
        failed = FailedStatement(statement=stmt, error=error)

        assert failed.category == ErrorCategory.OTHER

    def test_error_without_pgcode_is_other(self):
        """Errors without pgcode attribute are categorized as OTHER."""
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
        )
        error = Exception("Generic error")
        failed = FailedStatement(statement=stmt, error=error)

        assert failed.category == ErrorCategory.OTHER
        assert failed.error_code is None


class TestFailedStatement:
    """Tests for FailedStatement dataclass."""

    def test_extracts_pgcode_from_exception(self):
        """FailedStatement extracts pgcode from PostgreSQL exceptions."""
        stmt = ParsedStatement(
            raw_sql="CREATE TABLE t(id int);",
            statement_type="CREATE TABLE",
        )
        error = PgError("relation already exists", PGCODE_DUPLICATE_TABLE)
        failed = FailedStatement(statement=stmt, error=error)

        assert failed.error_code == PGCODE_DUPLICATE_TABLE
        assert failed.category == ErrorCategory.DUPLICATE

    def test_preserves_original_error(self):
        """FailedStatement preserves the original error message."""
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
        )
        error = PgError("syntax error at or near 'SELEC'", PGCODE_SYNTAX_ERROR)
        failed = FailedStatement(statement=stmt, error=error)

        assert "syntax error" in str(failed.error)


class TestDuplicateDefinition:
    """Tests for DuplicateDefinition dataclass."""

    def test_stores_locations(self):
        """DuplicateDefinition stores first and second location."""
        dup = DuplicateDefinition(
            duplicate_key="TABLE:public.test",
            first_location="schema/01_tables.sql:10",
            second_location="schema/02_tables.sql:5",
        )

        assert dup.duplicate_key == "TABLE:public.test"
        assert dup.first_location == "schema/01_tables.sql:10"
        assert dup.second_location == "schema/02_tables.sql:5"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result_fields(self):
        """Successful execution result has correct fields."""
        result = ExecutionResult(
            success=True,
            executed_count=5,
            created_objects={"TABLE:public.a", "TABLE:public.b"},
        )

        assert result.success is True
        assert result.executed_count == 5
        assert len(result.failed_statements) == 0
        assert len(result.duplicates) == 0
        assert len(result.created_objects) == 2

    def test_failure_result_includes_failed_statements(self):
        """Failed execution result includes failed statements."""
        stmt = ParsedStatement(
            raw_sql="CREATE TABLE t(id int);",
            statement_type="CREATE TABLE",
        )
        failed = FailedStatement(
            statement=stmt,
            error=PgError("error", PGCODE_SYNTAX_ERROR),
        )
        result = ExecutionResult(
            success=False,
            executed_count=3,
            failed_statements=[failed],
        )

        assert result.success is False
        assert len(result.failed_statements) == 1

    def test_duplicate_result_includes_locations(self):
        """Result with duplicates includes location information."""
        dup = DuplicateDefinition(
            duplicate_key="TABLE:public.test",
            first_location="a.sql:1",
            second_location="b.sql:2",
        )
        result = ExecutionResult(
            success=False,
            executed_count=2,
            duplicates=[dup],
        )

        assert result.success is False
        assert len(result.duplicates) == 1


class TestExecuteWithRetry:
    """Tests for execute_with_retry function using mocked database."""

    def test_all_succeed_first_pass(
        self, mock_psycopg2_connect, mock_connection, statement_factory
    ):
        """All statements succeeding on first pass returns success."""
        statements = [
            statement_factory.create_table(name="a"),
            statement_factory.create_table(name="b"),
        ]

        result = execute_with_retry("postgresql://test", statements)

        assert result.success is True
        assert result.executed_count == 2
        assert len(result.failed_statements) == 0

    def test_dependency_resolved_by_retry(self, statement_factory):
        """Dependency error on first pass resolves on retry."""
        # Create statements where the second depends on the first
        stmt_a = statement_factory.create_table(name="parent")
        stmt_b = statement_factory.create_table(name="child")  # Depends on parent

        # Track call count to simulate dependency resolution
        call_count = {"child": 0}

        def mock_execute(sql):
            if "child" in sql:
                call_count["child"] += 1
                if call_count["child"] == 1:
                    # First attempt fails with dependency error
                    raise PgError(
                        "relation 'parent' does not exist", PGCODE_UNDEFINED_TABLE
                    )
            # Success

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.autocommit = True

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt_b, stmt_a])

        assert result.success is True
        # Both statements should have executed
        assert result.executed_count == 2

    def test_stuck_returns_failure(self, statement_factory):
        """Unresolvable dependencies return failure."""
        stmt = statement_factory.create_table(name="dependent")

        # Always fail with dependency error
        def mock_execute(sql):
            raise PgError("relation does not exist", PGCODE_UNDEFINED_TABLE)

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt], max_iterations=5)

        assert result.success is False
        assert len(result.failed_statements) == 1
        assert result.failed_statements[0].category == ErrorCategory.DEPENDENCY

    def test_max_iterations_reached(self, statement_factory):
        """Max iterations limit prevents infinite loops."""
        # Create a statement that always fails but makes "progress"
        # by having different statements fail each time (simulated)
        stmt = statement_factory.create_table(name="test")

        execution_count = {"count": 0}

        def mock_execute(sql):
            execution_count["count"] += 1
            raise PgError("dependency error", PGCODE_UNDEFINED_TABLE)

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt], max_iterations=3)

        # Should stop after detecting no progress (stuck)
        assert result.success is False

    def test_duplicate_detection(self, statement_factory):
        """Duplicate object definitions are detected and reported."""
        # Two statements creating the same table
        stmt1 = statement_factory.create_table(
            name="dup", source_file="a.sql", line_number=1
        )
        stmt2 = statement_factory.create_table(
            name="dup", source_file="b.sql", line_number=5
        )

        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()  # All succeed
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt1, stmt2])

        # Should detect the duplicate
        assert result.success is False
        assert len(result.duplicates) == 1
        assert result.duplicates[0].duplicate_key == "TABLE:public.dup"
        assert "a.sql:1" in result.duplicates[0].first_location
        assert "b.sql:5" in result.duplicates[0].second_location

    def test_idempotent_role_skip(self, statement_factory):
        """CREATE ROLE duplicate errors are treated as success."""
        stmt = statement_factory.create_role(name="existing_role")

        def mock_execute(sql):
            if "CREATE ROLE" in sql:
                raise PgError(
                    'role "existing_role" already exists', PGCODE_DUPLICATE_OBJECT
                )

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt])

        # Should succeed because CREATE ROLE is idempotent
        assert result.success is True
        assert result.executed_count == 1

    def test_idempotent_extension_skip(self, statement_factory):
        """CREATE EXTENSION duplicate errors are treated as success."""
        stmt = statement_factory.create_extension(name="uuid-ossp")

        def mock_execute(sql):
            if "CREATE EXTENSION" in sql:
                raise PgError(
                    'extension "uuid-ossp" already exists', PGCODE_DUPLICATE_OBJECT
                )

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt])

        # Should succeed because CREATE EXTENSION is idempotent
        assert result.success is True

    def test_index_duplicate_not_skipped(self, statement_factory):
        """CREATE INDEX duplicate errors are NOT skipped (indicates schema issue)."""
        stmt = statement_factory.create_index(name="idx_test")

        def mock_execute(sql):
            if "CREATE INDEX" in sql:
                raise PgError(
                    'relation "idx_test" already exists', PGCODE_DUPLICATE_TABLE
                )

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt])

        # Should fail because index duplicates indicate improper schema reset
        assert result.success is False
        assert len(result.failed_statements) == 1

    def test_syntax_error_immediately_fails(self, statement_factory):
        """Syntax errors don't retry and immediately fail."""
        stmt = ParsedStatement(
            raw_sql="CRAETE TABLE typo(id int);",  # Typo in CREATE
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.typo",
            source_file="test.sql",
            line_number=1,
        )

        def mock_execute(sql):
            raise PgError('syntax error at or near "CRAETE"', PGCODE_SYNTAX_ERROR)

        mock_cursor = MagicMock()
        mock_cursor.execute = mock_execute
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn):
            result = execute_with_retry("postgresql://test", [stmt])

        assert result.success is False
        assert len(result.failed_statements) == 1
        assert result.failed_statements[0].category == ErrorCategory.SYNTAX


class TestIdempotentStatements:
    """Tests for IDEMPOTENT_DUPLICATE_STATEMENTS configuration."""

    def test_create_role_is_idempotent(self):
        """CREATE ROLE is in idempotent statements list."""
        assert "CREATE ROLE" in IDEMPOTENT_DUPLICATE_STATEMENTS

    def test_create_extension_is_idempotent(self):
        """CREATE EXTENSION is in idempotent statements list."""
        assert "CREATE EXTENSION" in IDEMPOTENT_DUPLICATE_STATEMENTS

    def test_create_table_not_idempotent(self):
        """CREATE TABLE is NOT in idempotent statements list."""
        assert "CREATE TABLE" not in IDEMPOTENT_DUPLICATE_STATEMENTS

    def test_create_index_not_idempotent(self):
        """CREATE INDEX is NOT in idempotent statements list."""
        assert "CREATE INDEX" not in IDEMPOTENT_DUPLICATE_STATEMENTS


class TestFormatLocation:
    """Tests for _format_location helper function."""

    def test_format_with_file_and_line(self):
        """Format location with both file and line number."""
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
            source_file="schema/tables.sql",
            line_number=42,
        )
        assert _format_location(stmt) == "schema/tables.sql:42"

    def test_format_with_file_only(self):
        """Format location with file but no line number."""
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
            source_file="schema/tables.sql",
            line_number=None,
        )
        assert _format_location(stmt) == "schema/tables.sql"

    def test_format_with_no_location(self):
        """Format location with no file information."""
        stmt = ParsedStatement(
            raw_sql="SELECT 1;",
            statement_type="SELECT",
            source_file=None,
            line_number=None,
        )
        assert _format_location(stmt) == "<unknown>"


class TestFormatExecutionErrors:
    """Tests for format_execution_errors function."""

    def test_format_duplicates_with_locations(self):
        """Duplicate errors are formatted with locations."""
        result = ExecutionResult(
            success=False,
            executed_count=0,
            duplicates=[
                DuplicateDefinition(
                    duplicate_key="TABLE:public.test",
                    first_location="a.sql:1",
                    second_location="b.sql:5",
                )
            ],
        )

        formatted = format_execution_errors(result)

        assert "Duplicate" in formatted
        assert "TABLE:public.test" in formatted
        assert "a.sql:1" in formatted
        assert "b.sql:5" in formatted

    def test_format_dependency_errors(self):
        """Dependency errors are formatted clearly."""
        stmt = ParsedStatement(
            raw_sql="CREATE TABLE child (parent_id uuid REFERENCES parent(id));",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.child",
            source_file="child.sql",
            line_number=1,
        )
        result = ExecutionResult(
            success=False,
            executed_count=0,
            failed_statements=[
                FailedStatement(
                    statement=stmt,
                    error=PgError(
                        'relation "parent" does not exist', PGCODE_UNDEFINED_TABLE
                    ),
                )
            ],
        )

        formatted = format_execution_errors(result)

        assert "Missing dependencies" in formatted
        assert "child.sql:1" in formatted
        assert "CREATE TABLE" in formatted

    def test_format_syntax_errors_with_context(self):
        """Syntax errors include the error message."""
        stmt = ParsedStatement(
            raw_sql="CRAETE TABLE t(id int);",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.t",
            source_file="test.sql",
            line_number=5,
        )
        result = ExecutionResult(
            success=False,
            executed_count=0,
            failed_statements=[
                FailedStatement(
                    statement=stmt,
                    error=PgError(
                        'syntax error at or near "CRAETE"', PGCODE_SYNTAX_ERROR
                    ),
                )
            ],
        )

        formatted = format_execution_errors(result)

        assert "Syntax errors" in formatted
        assert "test.sql:5" in formatted
        assert "CRAETE" in formatted

    def test_format_mixed_errors(self):
        """Multiple error types are grouped correctly."""
        stmt1 = ParsedStatement(
            raw_sql="...",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.a",
            source_file="a.sql",
            line_number=1,
        )
        stmt2 = ParsedStatement(
            raw_sql="...",
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name="public.b",
            source_file="b.sql",
            line_number=1,
        )

        result = ExecutionResult(
            success=False,
            executed_count=0,
            failed_statements=[
                FailedStatement(
                    statement=stmt1,
                    error=PgError("undefined table", PGCODE_UNDEFINED_TABLE),
                ),
                FailedStatement(
                    statement=stmt2,
                    error=PgError("syntax error", PGCODE_SYNTAX_ERROR),
                ),
            ],
        )

        formatted = format_execution_errors(result)

        # Both error categories should appear
        assert "Missing dependencies" in formatted
        assert "Syntax errors" in formatted

    def test_format_empty_result(self):
        """Empty/successful result formats without errors."""
        result = ExecutionResult(success=True, executed_count=5)

        formatted = format_execution_errors(result)

        # Should be mostly empty - no errors to format
        assert "Duplicate" not in formatted
        assert "Missing dependencies" not in formatted
        assert "Syntax errors" not in formatted


class TestErrorCategoryEnum:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """All expected error categories exist."""
        assert hasattr(ErrorCategory, "DEPENDENCY")
        assert hasattr(ErrorCategory, "SYNTAX")
        assert hasattr(ErrorCategory, "DUPLICATE")
        assert hasattr(ErrorCategory, "PERMISSION")
        assert hasattr(ErrorCategory, "OTHER")

    def test_category_values(self):
        """Error category values are lowercase strings."""
        assert ErrorCategory.DEPENDENCY.value == "dependency"
        assert ErrorCategory.SYNTAX.value == "syntax"
        assert ErrorCategory.DUPLICATE.value == "duplicate"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.OTHER.value == "other"
