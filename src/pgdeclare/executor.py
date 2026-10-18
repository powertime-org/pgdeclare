"""
Migration execution engine.

Executes schema changes with proper transaction handling:
- Phase 1: DDL changes with retry-until-stable ordering
- Phase 2: CONCURRENTLY index operations outside transaction (with retry)
- Phase 3: COMMENT ON, once every object exists in its final form

The retry-until-stable approach replaces complex priority-based ordering.
PostgreSQL itself discovers the correct execution order by:
1. Attempting each change
2. Deferring failed changes (dependency errors) to retry later
3. Repeating until all succeed or no progress is made (stuck)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import psycopg2
import psycopg2.errors

from .quoting import quote_ident
from .config import PgDeclareConfig, logger, plain, resolve
from .changes import (
    AddEnumValue,
    Change,
    CreateEnum,
    CreateIndex,
    CreateRole,
    SetComment,
)
from .differ import categorize_changes, split_index_operations
from .models import ChangeCategory
from .retry_executor import ERROR_CODE_CATEGORIES, ErrorCategory


# PostgreSQL error codes indicating missing dependencies (retryable)
DEPENDENCY_ERROR_CODES = {
    code
    for code, cat in ERROR_CODE_CATEGORIES.items()
    if cat == ErrorCategory.DEPENDENCY
}

# Change types that are idempotent (safe to skip if duplicate_object error)
# CreateRole is cluster-wide - may already exist from retry_executor.py
IDEMPOTENT_CHANGE_TYPES = (CreateRole,)

# PostgreSQL error code for "already exists" errors
DUPLICATE_ERROR_CODE = "42710"  # duplicate_object


@dataclass
class DeferredChange:
    """A change that failed and should be retried."""

    change: Change
    error: Exception
    error_code: str | None = None


@dataclass
class RetryResult:
    """Result of retry-until-stable execution."""

    success: bool
    executed_sql: list[str] = field(default_factory=list)
    stuck_changes: list[DeferredChange] = field(default_factory=list)


# Advisory lock ID for migration serialization
MIGRATION_LOCK_ID = 0x574C4D49  # An arbitrary, fixed key


def execute_migration(
    connection_string: str,
    changes: list[Change],
    offline_mode: bool = False,
    force: bool = False,
    dry_run: bool = False,
    drop_directives: set[str] | None = None,
    config: Optional[PgDeclareConfig] = None,
) -> tuple[bool, list[str]]:
    """
    Execute migration with proper transaction handling.

    Uses retry-until-stable ordering to discover the correct execution order,
    replacing complex priority-based topological sorting. PostgreSQL itself
    determines the correct order by failing operations with missing dependencies.

    Args:
        connection_string: PostgreSQL connection string
        changes: List of changes to apply
        offline_mode: If True, allows offline operations
        force: If True, bypasses @drop checks
        dry_run: If True, only prints SQL without executing
        drop_directives: Set of object names that have @drop directives
        config: Engine configuration (failure classifier)

    Returns:
        Tuple of (success, list of executed SQL statements)
    """
    if not changes:
        logger.info("No changes to apply")
        return True, []

    drop_directives = drop_directives or set()

    # Note: We no longer use topological_sort for priority-based ordering.
    # The retry-until-stable approach in _execute_ddl_phase discovers the
    # correct order by actually attempting execution.

    # Categorize changes (for offline/blocked checks, not ordering)
    categorized = categorize_changes(changes)

    # Check for offline operations
    offline_changes = categorized[ChangeCategory.OFFLINE]
    if offline_changes and not offline_mode:
        logger.error(
            f"Found {len(offline_changes)} offline operations. "
            "Use --offline flag to run them:"
        )
        for change in offline_changes:
            logger.error(f"  - {change.description}")
        return False, []

    # Check for blocked operations (require @drop)
    # A change is truly blocked only if it requires a directive AND doesn't have one
    def _has_drop_directive(change: Change) -> bool:
        """Check if a change has a matching @drop directive."""
        if not change.object_name:
            return False
        # Check both qualified (public.table.col) and unqualified (table.col) forms
        parts = change.object_name.split(".")
        if len(parts) == 3:
            # Schema.table.column - check both qualified and unqualified
            unqualified = f"{parts[1]}.{parts[2]}"
            return (
                change.object_name in drop_directives or unqualified in drop_directives
            )
        elif len(parts) == 2:
            # schema.table - accept the unqualified table name too, as
            # validate_drops_runtime does
            return change.object_name in drop_directives or parts[1] in drop_directives
        return change.object_name in drop_directives

    blocked_changes = [
        c
        for c in categorized[ChangeCategory.BLOCKED]
        if c.requires_drop_directive and not _has_drop_directive(c)
    ]
    if blocked_changes and not force:
        logger.error(
            f"Found {len(blocked_changes)} blocked operations requiring @drop:"
        )
        for change in blocked_changes:
            logger.error(f"  - {change.description} ({change.object_name})")
        logger.error(
            "Use @drop directive in @migrate-before block or --force to override"
        )
        return False, []

    # Split into DDL, index and comment operations. Comments go last: an
    # index only exists after the concurrent phase, and a view or function the
    # DDL phase drops and recreates takes its comment with it.
    comment_changes: list[Change] = [c for c in changes if isinstance(c, SetComment)]
    ddl_changes, index_changes = split_index_operations(
        [c for c in changes if not isinstance(c, SetComment)]
    )

    executed_sql: list[str] = []

    if dry_run:
        logger.info("Dry run - showing SQL that would be executed:")
        logger.info("")
        for change in changes:
            sql_stmt = change.to_sql()
            logger.info(f"-- {change.description}")
            logger.info(change.to_display_sql())
            logger.info("")
            executed_sql.append(sql_stmt)
        return True, executed_sql

    # Execute DDL changes in transaction
    if ddl_changes:
        success, ddl_sql = _execute_ddl_phase(
            connection_string, ddl_changes, config=config
        )
        executed_sql.extend(ddl_sql)
        if not success:
            return False, executed_sql

    # Execute index operations (CONCURRENTLY) outside transaction
    if index_changes:
        success, index_sql = _execute_index_phase(connection_string, index_changes)
        executed_sql.extend(index_sql)
        if not success:
            return False, executed_sql

    if comment_changes:
        success, comment_sql = _execute_comment_phase(
            connection_string, comment_changes, config=config
        )
        executed_sql.extend(comment_sql)
        if not success:
            return False, executed_sql

    logger.info(f"Successfully applied {len(changes)} changes")
    return True, executed_sql


def _execute_ddl_phase(
    connection_string: str,
    changes: list[Change],
    max_iterations: int = 100,
    config: Optional[PgDeclareConfig] = None,
) -> tuple[bool, list[str]]:
    """
    Execute DDL changes with retry-until-stable ordering.

    Uses PostgreSQL's dependency errors to discover the correct execution order.
    Changes that fail due to missing dependencies are deferred and retried.
    This replaces complex priority-based topological sorting.

    PostgreSQL requires enum changes (CREATE TYPE, ALTER TYPE ADD VALUE) to be
    committed before functions can reference the new enum values. This function
    handles enums separately:

    1. Enum changes - committed first so values are visible to subsequent DDL
    2. Other DDL changes - executed with retry-until-stable ordering
    """
    executed_sql: list[str] = []

    # Separate enum changes from other changes
    # Enum changes must be committed before functions can reference new values
    enum_changes = [c for c in changes if isinstance(c, (CreateEnum, AddEnumValue))]
    other_changes = [c for c in changes if c not in enum_changes]

    try:
        with psycopg2.connect(connection_string) as conn:
            # Acquire advisory lock
            if not acquire_advisory_lock(conn):
                logger.error(
                    "Failed to acquire migration lock - another migration may be running"
                )
                return False, executed_sql

            try:
                with conn.cursor() as cur:
                    # Phase 1: Execute enum changes first (must be committed before use)
                    if enum_changes:
                        logger.info(
                            f"Executing {len(enum_changes)} enum changes (separate commit)..."
                        )
                        for change in enum_changes:
                            sql_stmt = change.to_sql()
                            logger.info(f"  {change.description}")

                            try:
                                cur.execute(sql_stmt)
                                executed_sql.append(sql_stmt)
                            except Exception as e:
                                logger.error(f"Failed: {change.description}")
                                logger.error(f"SQL: {change.to_display_sql()}")
                                logger.error(f"Error: {e}")
                                raise

                        # CRITICAL: Commit enum changes so they're visible to subsequent DDL
                        conn.commit()
                        logger.info("Enum changes committed")

                    # Phase 2: Execute other DDL changes with retry-until-stable
                    if other_changes:
                        logger.info(
                            f"Executing {len(other_changes)} DDL changes "
                            f"(retry-until-stable)..."
                        )

                        result = _execute_with_retry_savepoints(
                            cur, conn, other_changes, max_iterations
                        )
                        executed_sql.extend(result.executed_sql)

                        if not result.success:
                            _report_stuck_changes(result.stuck_changes)
                            raise Exception(
                                f"DDL execution stuck with {len(result.stuck_changes)} "
                                f"unresolvable changes"
                            )

                        conn.commit()

                    logger.info("DDL phase complete")

            finally:
                release_advisory_lock(conn)

    except Exception as e:
        # A server that dies mid-phase surfaces as `connection already closed`.
        # Returning False here would make the caller report a schema failure
        # for what is a dead server, so let the death propagate as itself.
        failure = resolve(config).classify_failure(
            e, "the DDL phase of the schema diff", None
        )
        if failure is not None:
            raise failure from e

        logger.error(f"DDL phase failed: {e}")
        return False, executed_sql

    return True, executed_sql


def _execute_with_retry_savepoints(
    cur,
    conn,
    changes: list[Change],
    max_iterations: int = 100,
) -> RetryResult:
    """
    Execute changes with retry-until-stable using savepoints.

    Each change is executed within a savepoint. If it fails with a dependency
    error, the savepoint is rolled back and the change is deferred for retry.
    This allows PostgreSQL to discover the correct execution order.

    Args:
        cur: Database cursor
        conn: Database connection
        changes: List of changes to execute
        max_iterations: Maximum retry iterations (safety limit)

    Returns:
        RetryResult with success status and executed SQL
    """
    pending = list(changes)
    executed_sql: list[str] = []
    iteration = 0

    while pending and iteration < max_iterations:
        iteration += 1
        deferred: list[DeferredChange] = []
        progress_made = False

        logger.info(f"  Iteration {iteration}: {len(pending)} changes pending")

        for change in pending:
            # Create a unique savepoint name
            savepoint_name = f"change_{iteration}_{id(change)}"

            try:
                # Start savepoint
                cur.execute(f"SAVEPOINT {savepoint_name}")

                # Execute the change
                sql_stmt = change.to_sql()
                cur.execute(sql_stmt)

                # Success - release savepoint
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                executed_sql.append(sql_stmt)
                progress_made = True
                logger.info(f"    ✓ {change.description}")

            except Exception as e:
                # Rollback to savepoint
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

                error_code = getattr(e, "pgcode", None)

                # Handle idempotent duplicate errors for cluster-wide objects
                # (e.g., roles created by retry_executor.py in plan DB)
                if error_code == DUPLICATE_ERROR_CODE and isinstance(
                    change, IDEMPOTENT_CHANGE_TYPES
                ):
                    logger.info(
                        f"    ✓ {change.description} (already exists, skipping)"
                    )
                    progress_made = True
                    continue

                # Check if this is a dependency error (retryable)
                if error_code in DEPENDENCY_ERROR_CODES:
                    # Defer for retry
                    deferred.append(
                        DeferredChange(change=change, error=e, error_code=error_code)
                    )
                    logger.info(
                        f"    ⏳ Deferred: {change.description} "
                        f"(waiting for dependency)"
                    )
                else:
                    # Non-dependency error - fail immediately
                    logger.error(f"Failed: {change.description}")
                    logger.error(f"SQL: {change.to_display_sql()}")
                    logger.error(f"Error: {e}")
                    return RetryResult(
                        success=False,
                        executed_sql=executed_sql,
                        stuck_changes=[
                            DeferredChange(
                                change=change, error=e, error_code=error_code
                            )
                        ],
                    )

        # Check if we made progress
        if not progress_made and deferred:
            # Stuck - no progress was made
            logger.warning(
                f"Retry-until-stable stuck with {len(deferred)} unresolvable changes"
            )
            return RetryResult(
                success=False,
                executed_sql=executed_sql,
                stuck_changes=deferred,
            )

        # Continue with deferred changes
        pending = [d.change for d in deferred]

    # Check if we hit max iterations
    if pending:
        logger.error(
            f"Max iterations ({max_iterations}) reached with "
            f"{len(pending)} pending changes"
        )
        return RetryResult(
            success=False,
            executed_sql=executed_sql,
            stuck_changes=[
                DeferredChange(change=c, error=Exception("Max iterations"))
                for c in pending
            ],
        )

    return RetryResult(success=True, executed_sql=executed_sql)


def _report_stuck_changes(stuck: list[DeferredChange]) -> None:
    """Report details about stuck changes for debugging."""
    logger.error(f"\nMigration stuck with {len(stuck)} unresolvable changes:")

    # Group by error code for clearer reporting
    by_code: dict[str, list[DeferredChange]] = {}
    for item in stuck:
        code = item.error_code or "unknown"
        if code not in by_code:
            by_code[code] = []
        by_code[code].append(item)

    for code, items in by_code.items():
        code_desc = {
            "42P01": "Missing table",
            "42883": "Missing function",
            "42704": "Missing type/object",
            "42703": "Missing column",
            "3F000": "Missing schema",
            "42830": "Invalid foreign key",
        }.get(code, f"Error code {code}")

        logger.error(f"\n{code_desc}:")
        for item in items:
            logger.error(f"  - {item.change.description}")
            logger.error(f"    Error: {item.error}")
            logger.error(f"    SQL: {item.change.to_display_sql()[:200]}...")


def _execute_index_phase(
    connection_string: str,
    changes: list[Change],
    max_retries: int = 3,
) -> tuple[bool, list[str]]:
    """
    Execute index operations outside transaction (CONCURRENTLY).

    Retries failed CONCURRENTLY operations after cleaning up invalid indexes.
    """
    executed_sql: list[str] = []

    if not changes:
        return True, executed_sql

    logger.info(f"Executing {len(changes)} index operations (CONCURRENTLY)...")

    for change in changes:
        sql_stmt = change.to_sql()
        logger.info(f"  {change.description}")

        success = False
        retries = 0

        while not success and retries < max_retries:
            try:
                # Index operations must run with autocommit
                # Must set autocommit before any operations to avoid implicit transaction
                conn = psycopg2.connect(connection_string)
                conn.autocommit = True
                try:
                    with conn.cursor() as cur:
                        # For CreateIndex, clean up any invalid index with same name first
                        if isinstance(change, CreateIndex):
                            _cleanup_invalid_index(
                                cur, change.index.schema, change.index.name
                            )

                        cur.execute(sql_stmt)
                        executed_sql.append(sql_stmt)
                        success = True
                finally:
                    conn.close()

            except psycopg2.errors.UniqueViolation as e:
                # Index build failed due to duplicate - need to retry
                retries += 1
                logger.warning(f"Index operation failed (attempt {retries}): {e}")
                if retries < max_retries:
                    logger.info("Cleaning up and retrying...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error(f"Index operation failed after {max_retries} attempts")
                    return False, executed_sql

            except psycopg2.errors.OperationalError as e:
                # Could be concurrent access issue
                retries += 1
                logger.warning(f"Index operation failed (attempt {retries}): {e}")
                if retries < max_retries:
                    time.sleep(2)
                else:
                    logger.error(f"Index operation failed after {max_retries} attempts")
                    return False, executed_sql

            except Exception as e:
                logger.error(f"Index operation failed: {e}")
                return False, executed_sql

    logger.info("Index phase complete")
    return True, executed_sql


def _execute_comment_phase(
    connection_string: str,
    changes: list[Change],
    config: Optional[PgDeclareConfig] = None,
) -> tuple[bool, list[str]]:
    """Write comments in one transaction, after every other change.

    Nothing is left to wait for by now, so there is no retry: a failure is an
    error, and the transaction rolls back.
    """
    executed_sql: list[str] = []
    logger.info(f"Executing {len(changes)} comment changes...")
    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                for change in changes:
                    sql_stmt = change.to_sql()
                    logger.info(f"  {change.description}")
                    try:
                        cur.execute(sql_stmt)
                    except Exception as e:
                        logger.error(f"Failed: {change.description}")
                        logger.error(f"SQL: {change.to_display_sql()}")
                        logger.error(f"Error: {e}")
                        raise
                    executed_sql.append(sql_stmt)
            conn.commit()
    except Exception as e:
        failure = resolve(config).classify_failure(
            e, "the comment phase of the schema diff", None
        )
        if failure is not None:
            raise failure from e
        logger.error(f"Comment phase failed: {e}")
        return False, []

    logger.info("Comment phase complete")
    return True, executed_sql


def _cleanup_invalid_index(cur, schema: str, index_name: str) -> None:
    """Drop invalid index if it exists."""
    cur.execute(
        """
        SELECT 1 FROM pg_indexes i
        JOIN pg_class c ON c.relname = i.indexname
        JOIN pg_index ix ON ix.indexrelid = c.oid
        WHERE NOT ix.indisvalid
          AND i.schemaname = %s
          AND i.indexname = %s
        """,
        (schema, index_name),
    )

    if cur.fetchone():
        logger.info(f"  Dropping invalid index {schema}.{index_name}")
        cur.execute(
            f"DROP INDEX IF EXISTS {quote_ident(schema)}.{quote_ident(index_name)}"
        )


def acquire_advisory_lock(conn) -> bool:
    """
    Acquire advisory lock to prevent parallel migrations.

    Returns True if lock acquired, False otherwise.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        result = cur.fetchone()
        return result[0] if result else False


def release_advisory_lock(conn) -> None:
    """Release advisory lock."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))


def cleanup_invalid_indexes(connection_string: str, schema: str) -> list[str]:
    """
    Drop all invalid indexes in schema.

    Invalid indexes occur when CONCURRENTLY fails.

    Args:
        connection_string: PostgreSQL connection string
        schema: Schema name

    Returns:
        List of dropped index names
    """
    dropped: list[str] = []

    with psycopg2.connect(connection_string) as conn:
        conn.autocommit = True  # DROP INDEX CONCURRENTLY requires autocommit
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.indexname
                FROM pg_indexes i
                JOIN pg_class c ON c.relname = i.indexname
                JOIN pg_index ix ON ix.indexrelid = c.oid
                WHERE NOT ix.indisvalid
                  AND i.schemaname = %s
                """,
                (schema,),
            )

            for row in cur.fetchall():
                index_name = row[0]
                logger.info(f"Dropping invalid index: {schema}.{index_name}")
                cur.execute(
                    f"DROP INDEX CONCURRENTLY IF EXISTS {quote_ident(schema)}.{quote_ident(index_name)}"
                )
                dropped.append(index_name)

    return dropped


def format_changes_summary(
    changes: list[Change],
    decorate: Callable[[str, str], str] = plain,
) -> str:
    """
    Format a human-readable summary of changes.

    Groups changes by category and type for clear output. `decorate(style, text)`
    styles the category markers (see `PgDeclareConfig.decorate`).
    """
    if not changes:
        return "No changes"

    categorized = categorize_changes(changes)
    lines: list[str] = []

    # Summary counts
    total = len(changes)
    online = len(categorized[ChangeCategory.ONLINE])
    offline = len(categorized[ChangeCategory.OFFLINE])
    blocked = len(categorized[ChangeCategory.BLOCKED])
    try_first = len(categorized[ChangeCategory.TRY_FIRST])

    lines.append(f"Total changes: {total}")
    lines.append(f"  Online (safe): {online}")
    lines.append(f"  Offline (requires --offline): {offline}")
    lines.append(f"  Blocked (requires @drop): {blocked}")
    lines.append(f"  Try first (may fail): {try_first}")
    lines.append("")

    # Group by object type
    by_type: dict[str, list[Change]] = {}
    for change in changes:
        obj_type = change.object_type or type(change).__name__
        by_type.setdefault(obj_type, []).append(change)

    for obj_type, type_changes in sorted(by_type.items()):
        lines.append(f"{obj_type}:")
        for change in type_changes:
            category_marker = {
                ChangeCategory.ONLINE: decorate("green", "+"),
                ChangeCategory.OFFLINE: decorate("yellow", "~"),
                ChangeCategory.BLOCKED: decorate("red", "!"),
                ChangeCategory.TRY_FIRST: decorate("blue", "?"),
            }.get(change.category, " ")
            lines.append(f"  {category_marker} {change.description}")

    return "\n".join(lines)
