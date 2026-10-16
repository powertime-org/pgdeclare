"""
Retry-until-stable SQL executor.

Executes SQL statements with retry logic for dependency resolution and
duplicate detection for OR REPLACE statements.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import psycopg2

from .config import PgDeclareConfig, logger, resolve
from .statement_parser import ParsedStatement


class ErrorCategory(Enum):
    """Categories of PostgreSQL errors for diagnostics."""

    DEPENDENCY = "dependency"  # Missing table/function/type
    SYNTAX = "syntax"  # Syntax error
    DUPLICATE = "duplicate"  # Object already exists
    PERMISSION = "permission"  # Permission denied
    UNAVAILABLE = "unavailable"  # Postgres went away, not a schema defect
    OTHER = "other"  # Other errors


# PostgreSQL error codes mapped to categories
# See: https://www.postgresql.org/docs/current/errcodes-appendix.html
ERROR_CODE_CATEGORIES = {
    # Dependency errors (missing objects)
    "42P01": ErrorCategory.DEPENDENCY,  # undefined_table
    "42883": ErrorCategory.DEPENDENCY,  # undefined_function
    "42704": ErrorCategory.DEPENDENCY,  # undefined_object (type, etc.)
    "42703": ErrorCategory.DEPENDENCY,  # undefined_column
    "3F000": ErrorCategory.DEPENDENCY,  # invalid_schema_name
    "42830": ErrorCategory.DEPENDENCY,  # invalid_foreign_key (no matching unique constraint)
    # Duplicate errors
    "42P07": ErrorCategory.DUPLICATE,  # duplicate_table
    "42710": ErrorCategory.DUPLICATE,  # duplicate_object (includes roles)
    "42723": ErrorCategory.DUPLICATE,  # duplicate_function
    "42P16": ErrorCategory.DUPLICATE,  # invalid_table_definition (e.g., multiple primary keys)
    # Syntax errors
    "42601": ErrorCategory.SYNTAX,  # syntax_error
    "42000": ErrorCategory.SYNTAX,  # syntax_error_or_access_rule_violation
    # Permission errors
    "42501": ErrorCategory.PERMISSION,  # insufficient_privilege
    # The server going away. Never a schema defect, and must never be grouped
    # with one.
    "08000": ErrorCategory.UNAVAILABLE,  # connection_exception
    "08001": ErrorCategory.UNAVAILABLE,  # sqlclient_unable_to_establish_connection
    "08003": ErrorCategory.UNAVAILABLE,  # connection_does_not_exist
    "08004": ErrorCategory.UNAVAILABLE,  # sqlserver_rejected_establishment
    "08006": ErrorCategory.UNAVAILABLE,  # connection_failure
    "08007": ErrorCategory.UNAVAILABLE,  # transaction_resolution_unknown
    "08P01": ErrorCategory.UNAVAILABLE,  # protocol_violation
    "53000": ErrorCategory.UNAVAILABLE,  # insufficient_resources
    "53100": ErrorCategory.UNAVAILABLE,  # disk_full
    "53200": ErrorCategory.UNAVAILABLE,  # out_of_memory
    "53300": ErrorCategory.UNAVAILABLE,  # too_many_connections
    "57P01": ErrorCategory.UNAVAILABLE,  # admin_shutdown
    "57P02": ErrorCategory.UNAVAILABLE,  # crash_shutdown
    "57P03": ErrorCategory.UNAVAILABLE,  # cannot_connect_now
}

# Statement types where "already exists" errors should be treated as success
# Only for cluster-wide objects that persist across schema resets
IDEMPOTENT_DUPLICATE_STATEMENTS = {
    "CREATE ROLE",
    "CREATE EXTENSION",
    "CREATE SCHEMA",
}


@dataclass
class FailedStatement:
    """A statement that failed to execute."""

    statement: ParsedStatement
    error: Exception
    error_code: Optional[str] = None
    category: ErrorCategory = ErrorCategory.OTHER

    def __post_init__(self):
        """Extract error code and category from exception.

        pgcode is psycopg2's, so it is reached by getattr rather than an
        attribute read on the declared Exception.
        """
        pgcode = getattr(self.error, "pgcode", None)
        if pgcode:
            self.error_code = str(pgcode)
            self.category = ERROR_CODE_CATEGORIES.get(
                self.error_code, ErrorCategory.OTHER
            )


@dataclass
class DuplicateDefinition:
    """A duplicate object definition detected during execution."""

    duplicate_key: str
    first_location: str  # file:line of first definition
    second_location: str  # file:line of duplicate


@dataclass
class ExecutionResult:
    """Result of executing statements with retry."""

    success: bool
    executed_count: int
    failed_statements: list[FailedStatement] = field(default_factory=list)
    duplicates: list[DuplicateDefinition] = field(default_factory=list)
    created_objects: set[str] = field(
        default_factory=set
    )  # duplicate_keys of created objects


class RetryExecutorError(Exception):
    """Error during retry execution."""

    def __init__(
        self,
        message: str,
        failed_statements: list[FailedStatement],
        duplicates: list[DuplicateDefinition],
    ):
        super().__init__(message)
        self.failed_statements = failed_statements
        self.duplicates = duplicates


def execute_with_retry(
    connection_string: str,
    statements: list[ParsedStatement],
    max_iterations: int = 100,
    during: str = "the schema load",
    config: Optional[PgDeclareConfig] = None,
) -> ExecutionResult:
    """
    Execute SQL statements with retry-until-stable logic.

    Repeatedly attempts to execute failed statements until:
    - All statements succeed (success)
    - No new statements succeed in an iteration (stuck)
    - Max iterations reached

    Detects duplicates for OR REPLACE/IF NOT EXISTS statements by tracking
    which objects have been created.

    Args:
        connection_string: PostgreSQL connection string
        statements: List of parsed SQL statements
        max_iterations: Maximum retry iterations (safety limit)
        during: What this load is, for the message when Postgres dies under it
        config: Engine configuration (`classify_failure` decides which errors
            are infrastructure failures)

    Returns:
        ExecutionResult with success status and diagnostics

    Raises:
        RetryExecutorError: If execution gets stuck or duplicates are detected
        Exception: Whatever `config.classify_failure` returns for an error, when
            it classifies the error as the server becoming unreachable mid-load. Raised on
            the spot, with the error that named the death, because the passes
            that follow one see only `connection already closed` and reporting
            those blames the schema for an infrastructure failure.
    """
    classify_failure = resolve(config).classify_failure
    pending = list(statements)
    created_signatures: dict[str, str] = {}  # duplicate_key -> "file:line"
    duplicates: list[DuplicateDefinition] = []
    total_executed = 0
    iteration = 0

    # Reuse a single connection for all statements
    try:
        conn = psycopg2.connect(connection_string)
    except Exception as exc:
        failure = classify_failure(exc, during, None)
        if failure is not None:
            raise failure from exc
        raise
    conn.autocommit = True  # DDL should run outside transaction

    try:
        while pending and iteration < max_iterations:
            iteration += 1
            failed: list[FailedStatement] = []
            executed_this_round = 0

            logger.info(
                f"  Retry iteration {iteration}: {len(pending)} statements pending"
            )

            for stmt in pending:
                try:
                    _execute_single_statement_with_conn(conn, stmt)
                    executed_this_round += 1
                    total_executed += 1

                    # Duplicate detection for tracked object types
                    if stmt.duplicate_key:
                        if stmt.duplicate_key in created_signatures:
                            # Duplicate detected
                            duplicates.append(
                                DuplicateDefinition(
                                    duplicate_key=stmt.duplicate_key,
                                    first_location=created_signatures[
                                        stmt.duplicate_key
                                    ],
                                    second_location=_format_location(stmt),
                                )
                            )
                        else:
                            created_signatures[stmt.duplicate_key] = _format_location(
                                stmt
                            )

                except Exception as e:
                    # THE BUG THIS GUARDS. A server that dies mid-pass
                    # leaves every later statement raising `connection already
                    # closed`, which is derived, not diagnostic. The pass made
                    # progress, so another one is taken, and that one fails
                    # wholly with the derived error - so "stuck with N
                    # unresolvable statements" ends up naming N schema files and
                    # discarding the error that named the cause. Stop here, on
                    # the error that named it.
                    failure = classify_failure(e, during, _format_location(stmt))
                    if failure is not None:
                        raise failure from e

                    failed_stmt = FailedStatement(statement=stmt, error=e)

                    # Handle idempotent duplicate errors for cluster-wide objects
                    # Only roles, extensions, and schemas are truly cluster-wide and safe to skip
                    # For schema objects like indexes, "already exists" indicates improper reset
                    if (
                        failed_stmt.category == ErrorCategory.DUPLICATE
                        and stmt.statement_type in IDEMPOTENT_DUPLICATE_STATEMENTS
                    ):
                        logger.info(
                            f"    Skipping {stmt.statement_type} (already exists): "
                            f"{stmt.object_name or '<unnamed>'}"
                        )
                        executed_this_round += 1
                        total_executed += 1
                        continue

                    failed.append(failed_stmt)

            # Check if we made progress
            if len(failed) == len(pending):
                # No progress - we're stuck
                logger.warning(
                    f"Retry executor stuck with {len(failed)} unresolvable statements"
                )
                return ExecutionResult(
                    success=False,
                    executed_count=total_executed,
                    failed_statements=failed,
                    duplicates=duplicates,
                    created_objects=set(created_signatures.keys()),
                )

            pending = [f.statement for f in failed]

        # Check for duplicates after successful execution
        if duplicates:
            logger.error(f"Found {len(duplicates)} duplicate definitions")
            return ExecutionResult(
                success=False,
                executed_count=total_executed,
                failed_statements=[],
                duplicates=duplicates,
                created_objects=set(created_signatures.keys()),
            )

        if pending:
            # Max iterations reached
            logger.error(
                f"Max iterations ({max_iterations}) reached with {len(pending)} pending"
            )
            return ExecutionResult(
                success=False,
                executed_count=total_executed,
                failed_statements=[
                    FailedStatement(statement=s, error=Exception("Max iterations"))
                    for s in pending
                ],
                duplicates=duplicates,
                created_objects=set(created_signatures.keys()),
            )

        return ExecutionResult(
            success=True,
            executed_count=total_executed,
            failed_statements=[],
            duplicates=duplicates,
            created_objects=set(created_signatures.keys()),
        )

    finally:
        conn.close()


def _execute_single_statement_with_conn(conn, stmt: ParsedStatement) -> None:
    """
    Execute a single SQL statement using an existing connection.

    Args:
        conn: Database connection
        stmt: Statement to execute

    Raises:
        Exception: If execution fails
    """
    with conn.cursor() as cur:
        cur.execute(stmt.raw_sql)


def _execute_single_statement(connection_string: str, stmt: ParsedStatement) -> None:
    """
    Execute a single SQL statement (opens new connection).

    Deprecated: Use _execute_single_statement_with_conn for better performance.

    Args:
        connection_string: PostgreSQL connection string
        stmt: Statement to execute

    Raises:
        Exception: If execution fails
    """
    with psycopg2.connect(connection_string) as conn:
        conn.autocommit = True  # DDL should run outside transaction
        with conn.cursor() as cur:
            cur.execute(stmt.raw_sql)


def _format_location(stmt: ParsedStatement) -> str:
    """Format statement location as file:line."""
    if stmt.source_file and stmt.line_number:
        return f"{stmt.source_file}:{stmt.line_number}"
    elif stmt.source_file:
        return stmt.source_file
    else:
        return "<unknown>"


def format_execution_errors(result: ExecutionResult) -> str:
    """
    Format execution errors for display.

    Groups errors by category and provides actionable messages.

    Args:
        result: ExecutionResult with failures

    Returns:
        Formatted error message
    """
    lines = []

    # Report duplicates first (these are always fatal)
    if result.duplicates:
        lines.append("\nDuplicate definitions found:\n")
        for dup in result.duplicates:
            lines.append(f"  {dup.duplicate_key}:")
            lines.append(f"    First:  {dup.first_location}")
            lines.append(f"    Second: {dup.second_location}")
            lines.append("")

    # Group failed statements by category
    if result.failed_statements:
        by_category: dict[ErrorCategory, list[FailedStatement]] = {}
        for failed in result.failed_statements:
            cat = failed.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(failed)

        if any(
            f.category == ErrorCategory.UNAVAILABLE for f in result.failed_statements
        ):
            # Guard for the path that reaches here without going through the
            # abort in execute_with_retry. What follows is a list of schema
            # files, and without this banner it reads as a branch defect.
            lines.append(
                "\nPostgres became unreachable during this load. The statements "
                "below did not\nfail because they are wrong; they failed because "
                "there was no server to run\nthem. This is an infrastructure "
                "failure, not a schema or branch defect.\n"
            )

        lines.append("\nSchema loading stuck with unresolvable statements:\n")

        # Report dependency errors (most common)
        if ErrorCategory.DEPENDENCY in by_category:
            lines.append("Missing dependencies:")
            for failed in by_category[ErrorCategory.DEPENDENCY]:
                stmt = failed.statement
                lines.append(f"  - {_format_location(stmt)}")
                lines.append(f"    {stmt.statement_type} {stmt.object_name or ''}")
                lines.append(f"    Error: {failed.error}")
                lines.append("")

        # Report syntax errors
        if ErrorCategory.SYNTAX in by_category:
            lines.append("Syntax errors:")
            for failed in by_category[ErrorCategory.SYNTAX]:
                stmt = failed.statement
                lines.append(f"  - {_format_location(stmt)}")
                lines.append(f"    Error: {failed.error}")
                lines.append("")

        # Report other errors
        for cat in [
            ErrorCategory.DUPLICATE,
            ErrorCategory.PERMISSION,
            ErrorCategory.UNAVAILABLE,
            ErrorCategory.OTHER,
        ]:
            if cat in by_category:
                lines.append(f"{cat.value.title()} errors:")
                for failed in by_category[cat]:
                    stmt = failed.statement
                    lines.append(f"  - {_format_location(stmt)}")
                    lines.append(f"    {stmt.statement_type} {stmt.object_name or ''}")
                    lines.append(f"    Error: {failed.error}")
                    lines.append("")

    return "\n".join(lines)


def execute_statements_in_transaction(
    connection_string: str,
    statements: list[ParsedStatement],
) -> ExecutionResult:
    """
    Execute statements in a single transaction (for use after retry succeeds).

    This is used for the actual plan database population where we want
    atomicity - either all statements succeed or none do.

    Args:
        connection_string: PostgreSQL connection string
        statements: List of statements to execute

    Returns:
        ExecutionResult
    """
    created_signatures: dict[str, str] = {}
    duplicates: list[DuplicateDefinition] = []

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            try:
                for stmt in statements:
                    cur.execute(stmt.raw_sql)

                    # Duplicate detection
                    if stmt.duplicate_key:
                        if stmt.duplicate_key in created_signatures:
                            duplicates.append(
                                DuplicateDefinition(
                                    duplicate_key=stmt.duplicate_key,
                                    first_location=created_signatures[
                                        stmt.duplicate_key
                                    ],
                                    second_location=_format_location(stmt),
                                )
                            )
                        else:
                            created_signatures[stmt.duplicate_key] = _format_location(
                                stmt
                            )

                conn.commit()

                if duplicates:
                    return ExecutionResult(
                        success=False,
                        executed_count=len(statements),
                        duplicates=duplicates,
                        created_objects=set(created_signatures.keys()),
                    )

                return ExecutionResult(
                    success=True,
                    executed_count=len(statements),
                    created_objects=set(created_signatures.keys()),
                )

            except Exception as e:
                conn.rollback()
                # Find which statement failed
                return ExecutionResult(
                    success=False,
                    executed_count=0,
                    failed_statements=[
                        FailedStatement(
                            statement=ParsedStatement(
                                raw_sql="<transaction>",
                                statement_type="TRANSACTION",
                            ),
                            error=e,
                        )
                    ],
                )
