"""
SQL schema parser.

Parses SQL files to build a Schema object representing the desired state.
Uses the plan database approach: apply SQL to a temporary database and introspect.

Supports two modes for applying SQL:
1. Retry-until-stable: Parse SQL into statements, execute with retry for dependency resolution
2. Combined SQL: Build single SQL string and execute in one go (legacy mode)
"""

import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extensions import make_dsn

from .quoting import quote_ident
from .config import PgDeclareConfig, logger, resolve
from .introspection import introspect_schema
from .models import Schema
from .statement_parser import (
    ParsedStatement,
    parse_sql_content,
    collect_sql_files_recursive,
    parse_role_from_sql,
)
from .retry_executor import (
    execute_with_retry,
    format_execution_errors,
)


# Regex to match @migrate-before / @migrate-after blocks (including content through @end-migrate)
_MIGRATE_BLOCK_PATTERN = re.compile(
    r"--\s*@migrate-(?:before|after):[^\n]*\n"  # Opening line
    r"[\s\S]*?"  # Content - matches any char including newlines (non-greedy)
    r"--\s*@end-migrate[^\n]*\n?",  # Closing line
    re.IGNORECASE,
)


def strip_migrate_blocks(sql_content: str) -> str:
    """
    Strip @migrate-before / @migrate-after...@end-migrate blocks from SQL content.

    Migration blocks contain data migration SQL that should only run on existing
    databases, not when building the desired schema for comparison.

    Args:
        sql_content: Raw SQL content potentially containing migration blocks

    Returns:
        SQL content with migration blocks removed
    """
    return _MIGRATE_BLOCK_PATTERN.sub("", sql_content)


def _search_path_statement(config: PgDeclareConfig) -> ParsedStatement:
    """The SET search_path every schema load starts with."""
    return ParsedStatement(
        raw_sql=f"SET search_path TO {config.search_path_sql};",
        statement_type="SET",
        source_file="<setup>",
        line_number=1,
    )


def _bootstrap_statements(config: PgDeclareConfig) -> list[ParsedStatement]:
    """Statements from the configured bootstrap SQL files (working tree only)."""
    statements: list[ParsedStatement] = []
    for path in config.bootstrap_sql_files:
        if path.exists():
            statements.extend(
                parse_sql_content(
                    path.read_text(), source_file=config.bootstrap_source_label
                )
            )
    return statements


def _is_externally_provisioned(stmt: ParsedStatement, config: PgDeclareConfig) -> bool:
    """CREATE EXTENSION, and CREATE SCHEMA for a schema provisioned outside the engine."""
    if stmt.statement_type == "CREATE EXTENSION":
        return True
    return (
        stmt.statement_type == "CREATE SCHEMA"
        and stmt.object_name in config.externally_provisioned_schemas
    )


def _extract_role_passwords(statements: list[ParsedStatement]) -> dict[str, str]:
    """
    Build role_name -> password mapping from CREATE ROLE statements.

    PostgreSQL introspection cannot retrieve passwords from pg_roles,
    so we extract them from the declarative SQL to inject into the schema.

    Args:
        statements: List of parsed SQL statements

    Returns:
        Dictionary mapping role names to their passwords
    """
    passwords: dict[str, str] = {}

    for stmt in statements:
        if stmt.statement_type != "CREATE ROLE":
            continue

        role_info = parse_role_from_sql(stmt.raw_sql)
        if role_info and role_info.get("password"):
            passwords[role_info["name"]] = role_info["password"]

    return passwords


def _inject_role_passwords(schema: Schema, passwords: dict[str, str]) -> None:
    """
    Inject passwords from declarative SQL into introspected roles.

    When PostgreSQL introspects roles via pg_roles, password hashes are
    not retrievable. This function merges the passwords parsed from
    declarative SQL into the introspected Role objects.

    Args:
        schema: Schema object containing introspected roles
        passwords: Dictionary mapping role names to passwords
    """
    for role_name, password in passwords.items():
        if role_name in schema.roles:
            schema.roles[role_name].password = password


def build_desired_schema(
    sql_dir: Path,
    connection_string: str,
    plan_db_name: str,
    schemas: list[str] | None = None,
    use_retry_executor: bool = True,
    config: Optional[PgDeclareConfig] = None,
) -> Schema:
    """
    Build desired schema by applying SQL files to plan database and introspecting.

    This approach ensures we get an accurate representation of the desired state,
    including all implicit constraints and defaults that PostgreSQL applies.

    Args:
        sql_dir: Path to the SQL root directory
        connection_string: Connection string to postgres database (for admin ops)
        plan_db_name: Name of the plan database to use
        schemas: List of schemas to introspect (default: `config.schemas`)
        use_retry_executor: If True, use retry-until-stable execution (default: True)
        config: Engine configuration

    Returns:
        Schema object representing desired state
    """
    cfg = resolve(config)
    with cfg.lock():
        return _build_desired_schema(
            sql_dir, connection_string, plan_db_name, schemas, use_retry_executor, cfg
        )


def _build_desired_schema(
    sql_dir: Path,
    connection_string: str,
    plan_db_name: str,
    schemas: list[str] | None,
    use_retry_executor: bool,
    config: PgDeclareConfig,
) -> Schema:
    if schemas is None:
        schemas = list(config.schemas)

    if config.dry_run:
        logger.info("DRY RUN: Would build desired schema from SQL files")
        return Schema()

    # Parse SQL files to extract statements (always needed for password extraction)
    statements = _parse_all_sql_files(sql_dir, config)

    if use_retry_executor:
        logger.info(f"Parsed {len(statements)} SQL statements from {sql_dir}")

        # Apply to plan database using retry executor
        _apply_to_plan_database(
            connection_string,
            plan_db_name,
            sql_content="",  # Not used in retry mode
            use_retry=True,
            statements=statements,
            config=config,
        )
    else:
        # Legacy mode: build combined SQL content
        sql_content = _build_combined_sql(sql_dir, config)

        # Apply to plan database
        _apply_to_plan_database(
            connection_string, plan_db_name, sql_content, config=config
        )

    # Build plan database connection string
    # Extract base URL and replace database name
    plan_url = _replace_db_in_url(connection_string, plan_db_name)

    # Extract role passwords from declarative SQL
    # PostgreSQL introspection cannot retrieve password hashes from pg_roles,
    # so we parse them from the source SQL files
    passwords = _extract_role_passwords(statements)

    # Introspect the plan database
    schema = introspect_schema(plan_url, schemas, config=config)

    # Roles are cluster-wide, so building the plan creates every declared role.
    # A managed role still missing is declared nowhere.
    missing = [r for r in resolve(config).managed_roles if r not in schema.roles]
    if missing:
        raise ValueError(
            "managed role(s) not found: "
            + ", ".join(missing)
            + ". Declare each with CREATE ROLE in your schema files, "
            "or create it on the server."
        )

    # Inject passwords into introspected roles
    _inject_role_passwords(schema, passwords)

    return schema


def _parse_all_sql_files(
    sql_dir: Path, config: Optional[PgDeclareConfig] = None
) -> list[ParsedStatement]:
    """
    Parse all SQL files into statements for retry-until-stable execution.

    Collects, in order:
    1. A SET search_path statement (`config.search_path`)
    2. The configured bootstrap SQL files
    3. schemas/ - Tables, constraints, base functions (recursive)
    4. functions/ - Additional functions
    5. generated/ - Generated policies, triggers, grants

    Note: CREATE EXTENSION, and CREATE SCHEMA for an externally provisioned
    schema, are filtered out (provisioned cluster-wide by the caller).
    Note: migration blocks are stripped.

    Args:
        sql_dir: Path to the SQL root directory
        config: Engine configuration

    Returns:
        List of ParsedStatement objects
    """
    cfg = resolve(config)
    all_statements: list[ParsedStatement] = [_search_path_statement(cfg)]
    all_statements.extend(_bootstrap_statements(cfg))

    # Collect and parse schema files (recursive)
    schemas_dir = sql_dir / cfg.schemas_subdir
    if schemas_dir.exists():
        for sql_file in collect_sql_files_recursive(schemas_dir):
            relative_path = sql_file.relative_to(schemas_dir)
            content = sql_file.read_text()
            # Strip migration blocks
            content = strip_migrate_blocks(content)

            statements = parse_sql_content(content, source_file=str(relative_path))
            all_statements.extend(
                stmt for stmt in statements if not _is_externally_provisioned(stmt, cfg)
            )

    # Collect and parse function files, then generated files (flat)
    for subdir in (cfg.functions_subdir, cfg.generated_subdir):
        flat_dir = sql_dir / subdir
        if flat_dir.exists():
            for flat_file in sorted(flat_dir.glob("*.sql")):
                content = flat_file.read_text()
                statements = parse_sql_content(
                    content, source_file=f"{subdir}/{flat_file.name}"
                )
                all_statements.extend(statements)

    return all_statements


def _collect_sql_files_recursive(directory: Path) -> list[Path]:
    """
    Collect all SQL files recursively, alphabetically sorted.

    No special ordering is needed - the retry-until-stable executor handles
    all dependency resolution automatically.

    Args:
        directory: Root directory to search

    Returns:
        List of SQL file paths in alphabetical order
    """
    result: list[Path] = []
    items = sorted(directory.iterdir()) if directory.exists() else []

    for item in items:
        if item.is_dir():
            result.extend(_collect_sql_files_recursive(item))
        elif item.is_file() and item.suffix == ".sql":
            result.append(item)

    return result


def _build_combined_sql(sql_dir: Path, config: Optional[PgDeclareConfig] = None) -> str:
    """
    Build combined SQL from schema, functions, and generated files.

    Order:
    1. schemas/ - Tables, constraints, base functions (recursive collection)
    2. functions/ - Additional functions (auth schema-only, etc.)
    3. generated/ - Generated policies, triggers, grants

    The schemas/ directory supports subdirectories organized by domain.
    Files are collected recursively with alphabetical ordering.
    The retry-until-stable executor handles dependency resolution.

    Note: migration blocks are stripped from schema files since they contain
    data migration SQL that should only run on existing databases, not when
    building the desired schema for comparison.

    Returns:
        Combined SQL content
    """
    cfg = resolve(config)
    parts: list[str] = []

    # Add search_path setup
    parts.append(f"SET search_path TO {cfg.search_path_sql};\n\n")

    # Read schema files recursively
    schemas_dir = sql_dir / cfg.schemas_subdir
    if schemas_dir.exists():
        for schema_file in _collect_sql_files_recursive(schemas_dir):
            relative_path = schema_file.relative_to(schemas_dir)
            content = schema_file.read_text()
            # Strip migration blocks - they're for data migration, not schema definition
            content = strip_migrate_blocks(content)

            # Filter out externally provisioned statements (handled cluster-wide)
            statements = parse_sql_content(content, source_file=str(relative_path))
            filtered_statements = [
                stmt for stmt in statements if not _is_externally_provisioned(stmt, cfg)
            ]

            if filtered_statements:
                parts.append(f"-- From: {relative_path}\n")
                parts.append("\n".join(stmt.raw_sql for stmt in filtered_statements))
                parts.append("\n\n")

    # Read function files, then generated files
    for subdir in (cfg.functions_subdir, cfg.generated_subdir):
        flat_dir = sql_dir / subdir
        if flat_dir.exists():
            for flat_file in sorted(flat_dir.glob("*.sql")):
                content = flat_file.read_text()
                parts.append(f"-- From: {subdir}/{flat_file.name}\n")
                parts.append(content)
                parts.append("\n\n")

    return "".join(parts)


def _apply_to_plan_database(
    connection_string: str,
    plan_db_name: str,
    sql_content: str,
    use_retry: bool = False,
    statements: Optional[list[ParsedStatement]] = None,
    config: Optional[PgDeclareConfig] = None,
) -> None:
    """
    Apply SQL content to plan database.

    Drops and recreates `config.plan_recreate_schemas` first.

    Args:
        connection_string: Connection string to main database
        plan_db_name: Name of plan database
        sql_content: Combined SQL content (used if use_retry=False)
        use_retry: If True, use retry-until-stable execution
        statements: Parsed statements (required if use_retry=True)
        config: Engine configuration
    """
    cfg = resolve(config)
    plan_url = _replace_db_in_url(connection_string, plan_db_name)

    with psycopg2.connect(plan_url) as conn:
        conn.autocommit = True  # Schema operations need autocommit
        with conn.cursor() as cur:
            # Drop and recreate schemas
            for name in cfg.plan_recreate_schemas:
                cur.execute(f"DROP SCHEMA IF EXISTS {quote_ident(name)} CASCADE")
                cur.execute(f"CREATE SCHEMA {quote_ident(name)}")

            # Privileges on the recreated schemas. With the opt-in flag, every
            # schema is opened to PUBLIC, as on the live database. Otherwise a
            # recreated `public` gets PostgreSQL's stock ACL back (USAGE for
            # PUBLIC, and CREATE too before 15), so the plan matches a target
            # whose `public` schema was never touched; others stay owner-only.
            for name in cfg.plan_recreate_schemas:
                if cfg.grant_public_on_created_schemas:
                    cur.execute(f"GRANT ALL ON SCHEMA {quote_ident(name)} TO PUBLIC")
                elif name == "public":
                    stock = "USAGE" if conn.server_version >= 150000 else "ALL"
                    cur.execute(f"GRANT {stock} ON SCHEMA public TO PUBLIC")

    if use_retry and statements:
        # Use retry-until-stable execution
        logger.info(
            f"Executing {len(statements)} statements with retry-until-stable..."
        )
        result = execute_with_retry(plan_url, statements, config=cfg)

        if not result.success:
            error_msg = format_execution_errors(result)
            logger.error(f"Failed to apply schema to plan database:\n{error_msg}")
            raise Exception(f"Schema loading failed: {error_msg}")

        logger.info(f"Successfully executed {result.executed_count} statements")
    else:
        # Legacy mode: execute as single SQL block
        with psycopg2.connect(plan_url) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql_content)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to apply schema to plan database: {e}")
                    raise


def _replace_db_in_url(connection_string: str, new_db: str) -> str:
    """Replace database name in connection string."""
    # Handle postgresql:// URLs
    if "://" in connection_string:
        # Split on last /
        parts = connection_string.rsplit("/", 1)
        if len(parts) == 2:
            # Handle query string
            if "?" in parts[1]:
                _, query = parts[1].split("?", 1)
                return f"{parts[0]}/{new_db}?{query}"
            return f"{parts[0]}/{new_db}"

    # A key=value string: returning it unchanged would point the plan build
    # (which drops and recreates schemas) at the target database itself.
    return make_dsn(connection_string, dbname=new_db)


def ensure_schemas(
    connection_string: str,
    schemas: "list[str] | tuple[str, ...]",
    config: Optional[PgDeclareConfig] = None,
) -> None:
    """
    Ensure each schema exists in the database. With
    `config.grant_public_on_created_schemas`, also GRANT ALL on it to PUBLIC.

    Called before migrations to set up the schema structure.
    """
    cfg = resolve(config)
    if cfg.dry_run:
        logger.info(f"DRY RUN: Would ensure {', '.join(schemas)} schema exists")
        return

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            for name in schemas:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(name)}")
                if cfg.grant_public_on_created_schemas:
                    cur.execute(f"GRANT ALL ON SCHEMA {quote_ident(name)} TO PUBLIC")
        conn.commit()


def get_connection_string(
    user: str,
    password: str,
    host: str,
    # Ports often arrive as strings (environment variables, dotenv); this only
    # interpolates, so either is accepted.
    port: int | str,
    database: str,
) -> str:
    """Build PostgreSQL connection string."""
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _parse_all_sql_from_git(
    branch: str, repo_root: Path, config: Optional[PgDeclareConfig] = None
) -> list[ParsedStatement]:
    """
    Parse SQL files from a git branch into statements for retry execution.

    Mirrors `_parse_all_sql_files()` but reads from git instead of filesystem.
    This enables the retry-until-stable executor for git branch schema building,
    which handles dependency resolution robustly (including "role already exists"
    errors that the text-based filtering couldn't handle properly).

    Collects, in order:
    1. A SET search_path statement (`config.search_path`)
    2. The configured bootstrap SQL files (from the working tree, not git)
    3. schemas/ - Tables, constraints, base functions (recursive)
    4. functions/ - Additional functions
    5. generated/ - Generated policies, triggers, grants

    Note: CREATE EXTENSION, and CREATE SCHEMA for an externally provisioned
    schema, are filtered out (provisioned cluster-wide by the caller).
    Note: migration blocks are stripped.

    Args:
        branch: Git branch name (e.g., "main", "origin/main")
        repo_root: Path to the git repository root
        config: Engine configuration

    Returns:
        List of ParsedStatement objects
    """
    cfg = resolve(config)
    all_statements: list[ParsedStatement] = [_search_path_statement(cfg)]
    all_statements.extend(_bootstrap_statements(cfg))

    # Collect and parse schema files from git (recursive, DFS ordered)
    schemas_path = cfg.git_path(cfg.schemas_subdir)
    schema_files = git_list_sql_files_dfs(branch, schemas_path, repo_root)
    for file_path in schema_files:
        content = git_read_file(branch, file_path, repo_root)
        if content:
            # Strip migration blocks - they're for data migration
            content = strip_migrate_blocks(content)
            # Get relative path from schemas/ directory
            relative_path = file_path.replace(f"{schemas_path}/", "")

            statements = parse_sql_content(content, source_file=relative_path)
            all_statements.extend(
                stmt for stmt in statements if not _is_externally_provisioned(stmt, cfg)
            )

    # Collect and parse function files, then generated files, from git (flat)
    for subdir in (cfg.functions_subdir, cfg.generated_subdir):
        flat_files = _git_list_files(branch, f"{cfg.git_path(subdir)}/*.sql", repo_root)
        for file_path in sorted(flat_files):
            content = git_read_file(branch, file_path, repo_root)
            if content:
                file_name = Path(file_path).name
                statements = parse_sql_content(
                    content, source_file=f"{subdir}/{file_name}"
                )
                all_statements.extend(statements)

    return all_statements


def build_schema_from_git_branch(
    branch: str,
    connection_string: str,
    plan_db_name: str,
    repo_root: Path,
    schemas: list[str] | None = None,
    config: Optional[PgDeclareConfig] = None,
) -> Schema:
    """
    Build schema from SQL files in a specific git branch.

    This is used for CI validation to compare current branch against main.
    Uses retry-until-stable execution for robust dependency resolution,
    which handles "role already exists" and other dependency errors gracefully.

    Args:
        branch: Git branch name (e.g., "main", "origin/main")
        connection_string: Connection string to postgres database (for admin ops)
        plan_db_name: Name of the plan database to use
        repo_root: Path to the git repository root
        schemas: List of schemas to introspect (default: `config.schemas`)
        config: Engine configuration

    Returns:
        Schema object representing the branch's schema state
    """
    cfg = resolve(config)
    with cfg.lock():
        return _build_schema_from_git_branch(
            branch, connection_string, plan_db_name, repo_root, schemas, cfg
        )


def _build_schema_from_git_branch(
    branch: str,
    connection_string: str,
    plan_db_name: str,
    repo_root: Path,
    schemas: list[str] | None,
    config: PgDeclareConfig,
) -> Schema:
    if schemas is None:
        schemas = list(config.schemas)

    if config.dry_run:
        logger.info(f"DRY RUN: Would build schema from git branch {branch}")
        return Schema()

    # Parse SQL files from git into statements for retry execution
    statements = _parse_all_sql_from_git(branch, repo_root, config)
    logger.info(f"Parsed {len(statements)} SQL statements from git branch {branch}")

    # Extract role passwords before applying
    # PostgreSQL introspection cannot retrieve password hashes from pg_roles
    passwords = _extract_role_passwords(statements)

    # Apply to plan database using retry executor
    # This handles "role already exists" errors gracefully by treating them
    # as "already exists, skip" rather than hard failures
    _apply_to_plan_database(
        connection_string,
        plan_db_name,
        sql_content="",  # Not used in retry mode
        use_retry=True,
        statements=statements,
        config=config,
    )

    # Build plan database connection string
    plan_url = _replace_db_in_url(connection_string, plan_db_name)

    # Introspect the plan database
    schema = introspect_schema(plan_url, schemas, config=config)

    # Inject passwords into introspected roles
    _inject_role_passwords(schema, passwords)

    return schema


def _git_list_files(branch: str, pattern: str, repo_root: Path) -> list[str]:
    """
    List files matching a glob pattern in a git branch.

    Args:
        branch: Git branch name
        pattern: Glob pattern (e.g., "sql/functions/*.sql")
        repo_root: Path to the git repository root

    Returns:
        List of file paths
    """
    try:
        # Use git ls-tree to list files
        result = subprocess.run(  # dry-run: read-only, git ls-tree
            ["git", "ls-tree", "-r", "--name-only", branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Filter files matching the pattern
        all_files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return [f for f in all_files if fnmatch.fnmatch(f, pattern)]

    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not list files from git branch {branch}: {e}")
        return []


def git_list_sql_files_dfs(branch: str, base_path: str, repo_root: Path) -> list[str]:
    """
    List SQL files under a directory in a git branch using DFS ordering.

    Files are sorted using the same DFS logic as _collect_sql_files_dfs:
    directories and files are sorted alphabetically, with all files in a
    directory processed before moving to the next directory.

    Args:
        branch: Git branch name
        base_path: Base directory path (e.g., "sql/schemas")
        repo_root: Path to the git repository root

    Returns:
        List of file paths in DFS order
    """
    try:
        # Use git ls-tree to list all files recursively
        result = subprocess.run(  # dry-run: read-only, git ls-tree
            ["git", "ls-tree", "-r", "--name-only", branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        all_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Filter to SQL files under base_path
        sql_files = [
            f for f in all_files if f.startswith(base_path + "/") and f.endswith(".sql")
        ]

        # Sort using DFS ordering: sort by path components
        # This naturally gives us DFS order when paths are sorted alphabetically
        return sorted(sql_files)

    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not list files from git branch {branch}: {e}")
        return []


def git_read_file(branch: str, file_path: str, repo_root: Path) -> str | None:
    """
    Read a file from a git branch without checking it out.

    Args:
        branch: Git branch name
        file_path: Path to the file (relative to repo root)
        repo_root: Path to the git repository root

    Returns:
        File content or None if not found
    """
    try:
        result = subprocess.run(  # dry-run: read-only, git show
            ["git", "show", f"{branch}:{file_path}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    except subprocess.CalledProcessError:
        # File doesn't exist in this branch
        return None


def parse_do_blocks(
    sql_dir: Path, config: Optional[PgDeclareConfig] = None
) -> list[ParsedStatement]:
    """
    Parse DO blocks from SQL files.

    DO blocks are anonymous code blocks that must be idempotent since they
    run on every migration. They are used for setup tasks like configuring
    pg_timetable jobs.

    Args:
        sql_dir: Path to the SQL root directory
        config: Engine configuration

    Returns:
        List of ParsedStatement objects for DO blocks only
    """
    all_statements = _parse_all_sql_files(sql_dir, config)
    return [stmt for stmt in all_statements if stmt.statement_type == "DO"]


def execute_do_blocks(
    connection_string: str,
    do_blocks: list[ParsedStatement],
    dry_run: bool = False,
    search_path: str | None = None,
    config: Optional[PgDeclareConfig] = None,
) -> tuple[bool, list[str]]:
    """
    Execute DO blocks on the target database.

    DO blocks must be idempotent - they run on every migration. This is used
    for setup tasks like configuring pg_timetable cron jobs.

    Args:
        connection_string: PostgreSQL connection string
        do_blocks: List of DO block statements to execute
        dry_run: If True, only log what would be executed
        search_path: If provided, SET search_path before executing DO blocks.
            This overrides the database-level default for the session.
        config: Engine configuration (`report`/`decorate` carry the progress
            lines)

    Returns:
        Tuple of (success, list of executed SQL statements)
    """
    if not do_blocks:
        return True, []

    cfg = resolve(config)
    say, style = cfg.report, cfg.decorate
    executed_sql: list[str] = []

    if dry_run:
        say(style("dim", f"Would execute {len(do_blocks)} DO blocks"))
        for stmt in do_blocks:
            say(style("dim", f"  - {stmt.source_file}:{stmt.line_number}"))
            executed_sql.append(stmt.raw_sql)
        return True, executed_sql

    say(style("dim", f"Executing {len(do_blocks)} DO blocks..."))

    try:
        with psycopg2.connect(connection_string) as conn:
            conn.autocommit = True  # DO blocks may need immediate visibility
            with conn.cursor() as cur:
                if search_path:
                    cur.execute(f"SET search_path TO {search_path};")
                for stmt in do_blocks:
                    try:
                        cur.execute(stmt.raw_sql)
                        executed_sql.append(stmt.raw_sql)
                        logger.debug(
                            f"  ✓ DO block: {stmt.source_file}:{stmt.line_number}"
                        )
                    except Exception as e:
                        say(
                            style(
                                "red",
                                f"DO block failed ({stmt.source_file}:{stmt.line_number}): {e}",
                            )
                        )
                        return False, executed_sql

        say(style("green", f"✓ Executed {len(do_blocks)} DO blocks"))
        return True, executed_sql

    except Exception as e:
        say(style("red", f"Failed to execute DO blocks: {e}"))
        return False, executed_sql
