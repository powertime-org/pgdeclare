"""
Inline migration blocks (@migrate-before / @migrate-after) and their ledger.

Schema files may embed imperative SQL that runs once per database:

```sql
-- @migrate-before:2025-12-08_backfill_archived
-- @drop account.legacy_status
UPDATE account SET archived_at = NOW() WHERE legacy_status = 'archived' AND archived_at IS NULL;
-- @end-migrate

-- @migrate-after:2026-03-08_backfill_new_column
UPDATE my_table SET new_col = compute_value();
-- @end-migrate
```

- @migrate-before runs BEFORE the schema diff (renames, drops, backfills on
  existing columns).
- @migrate-after runs AFTER the schema diff (backfills on columns the diff
  creates).

Each executed block is recorded in the ledger table (`config.ledger_table`)
with a hash of its SQL, so it never runs twice and an edit after it ran is
refused rather than silently ignored.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.errors
from psycopg2 import sql as pgsql
from psycopg2.extras import RealDictCursor

from .config import PgDeclareConfig, logger, resolve
from .schema_parser import git_list_sql_files_dfs, git_read_file


@dataclass
class MigrationBlock:
    """Represents a single @migrate-before / @migrate-after block."""

    id: str  # e.g., "2025-12-08_backfill_archived"
    source_file: Path
    sql: str
    sql_hash: str
    phase: str = "before"  # "before" or "after"
    drop_directives: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


_START_PATTERN = re.compile(r"--\s*@migrate-(before|after):(\S+)", re.IGNORECASE)
_END_PATTERN = re.compile(r"--\s*@end-migrate", re.IGNORECASE)
_DROP_PATTERN = re.compile(r"--\s*@drop\s+(\S+)", re.IGNORECASE)


def hash_block_sql(sql_content: str) -> str:
    """The ledger hash of a block's SQL: the first 16 hex digits of its SHA-256."""
    return hashlib.sha256(sql_content.encode()).hexdigest()[:16]


def parse_migration_blocks_from_content(
    content: str, source_file: Path
) -> list[MigrationBlock]:
    """
    Parse @migrate-before / @migrate-after blocks from SQL content.

    Args:
        content: SQL file content
        source_file: Path to the source file (for tracking)

    Returns:
        List of MigrationBlock objects in line order
    """
    blocks: list[MigrationBlock] = []

    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        start_match = _START_PATTERN.search(line)

        if start_match:
            phase = start_match.group(1).lower()  # before or after
            migration_id = start_match.group(2)
            line_start = i + 1  # 1-indexed

            # Collect lines until @end-migrate
            sql_lines: list[str] = []
            drop_directives: list[str] = []
            i += 1

            while i < len(lines):
                line = lines[i]

                if _END_PATTERN.search(line):
                    line_end = i + 1
                    break

                # Check for @drop directive
                drop_match = _DROP_PATTERN.search(line)
                if drop_match:
                    drop_directives.append(drop_match.group(1))
                elif not line.strip().startswith("--"):
                    # Only include non-comment lines in SQL
                    sql_lines.append(line)

                i += 1
            else:
                # No @end-migrate found
                logger.warning(
                    f"Missing @end-migrate for {migration_id} in {source_file}"
                )
                line_end = len(lines)

            sql_content = "\n".join(sql_lines).strip()
            if sql_content:
                blocks.append(
                    MigrationBlock(
                        id=migration_id,
                        source_file=source_file,
                        sql=sql_content,
                        sql_hash=hash_block_sql(sql_content),
                        phase=phase,
                        drop_directives=drop_directives,
                        line_start=line_start,
                        line_end=line_end,
                    )
                )

        i += 1

    return blocks


def parse_migration_blocks(
    sql_dir: Path, config: Optional[PgDeclareConfig] = None
) -> list[MigrationBlock]:
    """
    Parse @migrate-before / @migrate-after blocks from schema files.

    Scans all .sql files under the schemas subdirectory of `sql_dir`,
    recursively, in path order.

    Args:
        sql_dir: Path to the SQL root directory
        config: Engine configuration

    Returns:
        List of MigrationBlock objects in file/line order
    """
    blocks: list[MigrationBlock] = []

    schemas_dir = sql_dir / resolve(config).schemas_subdir
    if not schemas_dir.exists():
        return blocks

    for sql_file in sorted(schemas_dir.glob("**/*.sql")):
        content = sql_file.read_text()
        blocks.extend(parse_migration_blocks_from_content(content, sql_file))

    return blocks


def parse_migration_blocks_from_git(
    branch: str, repo_root: Path, config: Optional[PgDeclareConfig] = None
) -> list[MigrationBlock]:
    """
    Parse @migrate-before / @migrate-after blocks from a git branch's schema files.

    Reads SQL files from the branch without checking it out.

    Args:
        branch: Git branch name (e.g., "main", "origin/main")
        repo_root: Path to the git repository root
        config: Engine configuration

    Returns:
        List of MigrationBlock objects

    Raises:
        Exception: If the branch doesn't exist or contains no schema SQL files
    """
    import subprocess

    cfg = resolve(config)

    # Validate branch exists
    try:
        subprocess.run(  # dry-run: read-only, git rev-parse --verify
            ["git", "rev-parse", "--verify", branch],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise Exception(
            f"Branch '{branch}' not found. Use --base to specify an existing branch."
        )

    schemas_path = cfg.git_path(cfg.schemas_subdir)
    sql_files = git_list_sql_files_dfs(branch, schemas_path, repo_root)

    if not sql_files:
        raise Exception(
            f"No SQL files found in {schemas_path}/ on branch '{branch}'. "
            "Is the SQL layout configured for this repository?"
        )

    blocks: list[MigrationBlock] = []

    for file_path in sql_files:
        content = git_read_file(branch, file_path, repo_root)
        if content:
            blocks.extend(parse_migration_blocks_from_content(content, Path(file_path)))

    return blocks


def _ledger(config: PgDeclareConfig) -> pgsql.Composed:
    return pgsql.SQL("{}.{}").format(
        pgsql.Identifier(config.ledger_schema), pgsql.Identifier(config.ledger_name)
    )


def ensure_ledger(
    connection_string: str, config: Optional[PgDeclareConfig] = None
) -> None:
    """Create the ledger schema and table if they do not exist."""
    cfg = resolve(config)
    if cfg.dry_run:
        return

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                pgsql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(
                    pgsql.Identifier(cfg.ledger_schema)
                )
            )
            cur.execute(
                pgsql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT NOW(),
                        source_file TEXT,
                        sql_hash TEXT,
                        sql_executed TEXT
                    );
                    """
                ).format(_ledger(cfg))
            )
        conn.commit()


def get_applied_migrations(
    connection_string: str, config: Optional[PgDeclareConfig] = None
) -> dict[str, tuple[datetime, str]]:
    """
    Get all applied migrations from the ledger.

    Returns:
        Dict mapping migration id to (applied_at, sql_hash). Empty when the
        ledger does not exist yet, and on a dry run.
    """
    cfg = resolve(config)
    if cfg.dry_run:
        return {}

    result: dict[str, tuple[datetime, str]] = {}

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    pgsql.SQL("SELECT id, applied_at, sql_hash FROM {}").format(
                        _ledger(cfg)
                    )
                )
                for row in cur.fetchall():
                    result[row["id"]] = (row["applied_at"], row["sql_hash"])
    except psycopg2.errors.UndefinedTable:
        # Ledger doesn't exist yet - that's OK
        pass

    return result


def execute_migration_block(
    connection_string: str,
    block: MigrationBlock,
    force_rerun: bool = False,
    config: Optional[PgDeclareConfig] = None,
) -> tuple[bool, Optional[str]]:
    """
    Execute a migration block if not already applied, and record it.

    Args:
        connection_string: PostgreSQL connection string
        block: MigrationBlock to execute
        force_rerun: If True, rerun even if already applied
        config: Engine configuration

    Returns:
        Tuple of (success, error_message)
    """
    cfg = resolve(config)
    if cfg.dry_run:
        logger.info(f"DRY RUN: Would execute migration: {block.id}")
        return True, None

    ensure_ledger(connection_string, cfg)

    applied = get_applied_migrations(connection_string, cfg)

    if block.id in applied:
        _applied_at, stored_hash = applied[block.id]

        if stored_hash != block.sql_hash:
            if force_rerun:
                logger.warning(f"Migration {block.id} hash changed - forcing rerun")
            else:
                return False, (
                    f"Migration {block.id} has been modified after application. "
                    f"Stored hash: {stored_hash}, Current hash: {block.sql_hash}. "
                    "Run with force to execute it again."
                )
        elif not force_rerun:
            logger.info(f"Skipping already-applied migration: {block.id}")
            return True, None

    logger.info(f"Executing migration: {block.id}")

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Execute migration SQL
                cur.execute(block.sql)

                # Record in the ledger
                cur.execute(
                    pgsql.SQL(
                        """
                        INSERT INTO {} (id, source_file, sql_hash, sql_executed)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            applied_at = NOW(),
                            sql_hash = EXCLUDED.sql_hash,
                            sql_executed = EXCLUDED.sql_executed
                        """
                    ).format(_ledger(cfg)),
                    (
                        block.id,
                        str(block.source_file),
                        block.sql_hash,
                        block.sql,
                    ),
                )

            conn.commit()

        if cfg.on_migration_applied is not None:
            cfg.on_migration_applied(block, connection_string)

        logger.info(f"Migration {block.id} applied successfully")
        return True, None

    except Exception as e:
        logger.error(f"Migration {block.id} failed: {e}")
        return False, str(e)


def mark_blocks_applied(
    connection_string: str,
    blocks: list[MigrationBlock],
    config: Optional[PgDeclareConfig] = None,
) -> None:
    """Record blocks in the ledger without executing their SQL.

    For a fresh install, whose freshly built schema already reflects every
    inline migration, and for simulating a deployed database's history.
    Existing rows are left as they are. The ledger must exist.
    """
    if not blocks:
        return

    cfg = resolve(config)
    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            for block in blocks:
                cur.execute(
                    pgsql.SQL(
                        """
                        INSERT INTO {}
                            (id, source_file, sql_hash, sql_executed)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """
                    ).format(_ledger(cfg)),
                    (
                        block.id,
                        str(block.source_file),
                        block.sql_hash,
                        block.sql,
                    ),
                )
        conn.commit()


def delete_ledger_entries(
    connection_string: str,
    migration_ids: list[str],
    config: Optional[PgDeclareConfig] = None,
) -> None:
    """Remove migrations' rows from the ledger, in one transaction."""
    cfg = resolve(config)
    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            for migration_id in migration_ids:
                cur.execute(
                    pgsql.SQL("DELETE FROM {} WHERE id = %s").format(_ledger(cfg)),
                    (migration_id,),
                )
        conn.commit()


def list_ledger(
    connection_string: str, config: Optional[PgDeclareConfig] = None
) -> list[dict]:
    """Every ledger row (id, applied_at, source_file, sql_hash), newest first."""
    cfg = resolve(config)
    ensure_ledger(connection_string, cfg)

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                pgsql.SQL(
                    """
                    SELECT id, applied_at, source_file, sql_hash
                    FROM {}
                    ORDER BY applied_at DESC
                    """
                ).format(_ledger(cfg))
            )
            return [dict(row) for row in cur.fetchall()]
