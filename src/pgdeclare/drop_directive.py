"""
@drop directive parsing and enforcement.

Implements data loss protection by requiring explicit @drop directives
for destructive operations (DROP TABLE, DROP COLUMN with data).
"""

from typing import Callable, Optional

import psycopg2

from .quoting import quote_ident
from .config import PgDeclareConfig, plain, logger, resolve
from .changes import Change, DropColumn, DropTable
from .models import Schema


def validate_drops_runtime(
    connection_string: str,
    changes: list[Change],
    drop_directives: set[str],
    config: Optional[PgDeclareConfig] = None,
) -> list[str]:
    """
    Runtime validation: check live data for each drop operation.

    Returns list of error messages for drops without @drop where data exists.

    Args:
        connection_string: PostgreSQL connection string
        changes: List of changes to validate
        drop_directives: Set of object names with @drop directive

    Returns:
        List of error messages (empty if all valid)
    """
    errors: list[str] = []

    for change in changes:
        if isinstance(change, DropTable):
            table_key = change.table_name
            qualified_key = f"{change.schema}.{change.table_name}"

            # Check if @drop directive exists
            has_directive = (
                table_key in drop_directives or qualified_key in drop_directives
            )

            if not has_directive:
                # Check if table has data
                has_data = check_table_has_data(
                    connection_string, change.schema, change.table_name, config
                )
                if has_data:
                    errors.append(
                        f"DROP TABLE {qualified_key} requires @drop directive "
                        "(table contains data)"
                    )

        elif isinstance(change, DropColumn):
            column_key = f"{change.table}.{change.column_name}"
            qualified_key = f"{change.schema}.{change.table}.{change.column_name}"

            # Check if @drop directive exists
            has_directive = (
                column_key in drop_directives or qualified_key in drop_directives
            )

            if not has_directive:
                # Check if column has data
                has_data = check_column_has_data(
                    connection_string,
                    change.schema,
                    change.table,
                    change.column_name,
                    config,
                )
                if has_data:
                    errors.append(
                        f"DROP COLUMN {qualified_key} requires @drop directive "
                        "(column contains data)"
                    )

    return errors


def validate_drops_ci(
    source_schema: Schema,
    main_branch_schema: Schema,
    drop_directives: set[str],
) -> list[str]:
    """
    CI validation: compare against main branch schema.

    Checks that any objects that exist in main but not in source have
    corresponding @drop directives.

    Args:
        source_schema: Schema from current branch
        main_branch_schema: Schema from main branch
        drop_directives: Set of object names with @drop directive

    Returns:
        List of error messages (empty if all valid)
    """
    errors: list[str] = []

    # Check for dropped tables
    for table_key, table in main_branch_schema.tables.items():
        if table_key not in source_schema.tables:
            # Table is being dropped
            table_name = table.name
            qualified_name = table_key

            has_directive = (
                table_name in drop_directives or qualified_name in drop_directives
            )

            if not has_directive:
                errors.append(
                    f"DROP TABLE {qualified_name} requires @drop directive "
                    "(table exists in main branch)"
                )
            # Skip column checks - whole table is being dropped
            continue

        # Table exists in both - check for dropped columns
        source_table = source_schema.tables[table_key]
        for col_name in table.columns:
            if col_name not in source_table.columns:
                column_key = f"{table.name}.{col_name}"
                qualified_key = f"{table_key}.{col_name}"

                has_directive = (
                    column_key in drop_directives or qualified_key in drop_directives
                )

                if not has_directive:
                    errors.append(
                        f"DROP COLUMN {qualified_key} requires @drop directive "
                        "(column exists in main branch)"
                    )

    return errors


def check_table_has_data(
    connection_string: str,
    schema: str,
    table: str,
    config: Optional[PgDeclareConfig] = None,
) -> bool:
    """
    Check if a table contains any data.

    Uses SET row_security = off to bypass RLS for accurate check.

    Args:
        connection_string: PostgreSQL connection string
        schema: Schema name
        table: Table name

    Returns:
        True if table has rows
    """
    if resolve(config).dry_run:
        # In dry-run mode, assume no data to allow diff to proceed
        return False

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Disable RLS for this check
                cur.execute("SET row_security = off")
                cur.execute(
                    f"SELECT 1 FROM {quote_ident(schema)}.{quote_ident(table)} LIMIT 1"
                )
                return cur.fetchone() is not None
    except psycopg2.Error as e:
        logger.warning(f"Could not check data in {schema}.{table}: {e}")
        # Err on the side of caution - assume data exists
        return True


def check_column_has_data(
    connection_string: str,
    schema: str,
    table: str,
    column: str,
    config: Optional[PgDeclareConfig] = None,
) -> bool:
    """
    Check if a column contains any non-NULL data.

    Uses SET row_security = off to bypass RLS for accurate check.

    Args:
        connection_string: PostgreSQL connection string
        schema: Schema name
        table: Table name
        column: Column name

    Returns:
        True if column has non-NULL values
    """
    if resolve(config).dry_run:
        # In dry-run mode, assume no data to allow diff to proceed
        return False

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Disable RLS for this check
                cur.execute("SET row_security = off")
                cur.execute(
                    f"SELECT 1 FROM {quote_ident(schema)}.{quote_ident(table)} "
                    f"WHERE {quote_ident(column)} IS NOT NULL LIMIT 1"
                )
                return cur.fetchone() is not None
    except psycopg2.Error as e:
        logger.warning(f"Could not check data in {schema}.{table}.{column}: {e}")
        # Err on the side of caution - assume data exists
        return True


def format_drop_errors(
    errors: list[str], decorate: Callable[[str, str], str] = plain
) -> str:
    """Format drop validation errors for display.

    `decorate(style, text)` styles the lines (see `PgDeclareConfig.decorate`).
    """
    if not errors:
        return ""

    lines = [
        decorate("red", "Data loss protection errors:"),
        "",
    ]

    for error in errors:
        lines.append(f"  {decorate('red', f'- {error}')}")

    lines.extend(
        [
            "",
            decorate(
                "dim", "To proceed, add @drop directive in a @migrate-before block:"
            ),
            "",
            "  -- @migrate-before:YYYY-MM-DD_description",
            "  -- @drop table_name.column_name",
            "  -- Your migration SQL here",
            "  -- @end-migrate",
            "",
            decorate(
                "dim", "Or use --force to bypass (not recommended for production)"
            ),
        ]
    )

    return "\n".join(lines)
