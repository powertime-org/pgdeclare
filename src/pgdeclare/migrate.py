"""
The migration pipeline: @migrate-before blocks, schema diff, @migrate-after
blocks, DO blocks, then a convergence check.

This is the engine's whole "bring a database to the declared schema" run. The
caller provisions what the engine does not own (the plan database, extensions,
schemas the bootstrap SQL expects) before calling `apply_declared_schema`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .changes import Change
from .config import PgDeclareConfig, logger, resolve
from .differ import compute_diff
from .drop_directive import format_drop_errors, validate_drops_runtime
from .executor import execute_migration, format_changes_summary
from .introspection import check_table_has_data, introspect_schema
from .ledger import (
    MigrationBlock,
    ensure_ledger,
    execute_migration_block,
    get_applied_migrations,
    mark_blocks_applied,
    parse_migration_blocks,
)
from .models import Schema
from .schema_parser import build_desired_schema, execute_do_blocks, parse_do_blocks

# Roles and role memberships are cluster-wide: applying them in one database
# changes them for every database on the server.
CLUSTER_OBJECT_TYPES = frozenset({"ROLE", "ROLE_MEMBERSHIP"})


def is_cluster_change(change: Change) -> bool:
    """True for a change that acts on the whole server, not one database."""
    return change.object_type in CLUSTER_OBJECT_TYPES


@dataclass
class SchemaRunResult:
    """What one run of the pipeline did (or, on a dry run, planned)."""

    desired: Schema
    changes: list[Change] = field(default_factory=list)
    executed_sql: list[str] = field(default_factory=list)
    skipped_cluster_changes: list[Change] = field(default_factory=list)
    migration_blocks: list[MigrationBlock] = field(default_factory=list)
    fresh_install: bool = False


class SchemaConvergenceError(Exception):
    """A schema run applied its changes and still did not reach the declared schema."""


def compute_residual_changes(
    desired: Schema,
    connection_string: str,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
    *,
    skip_cluster_changes: bool = False,
) -> list[Change]:
    """Re-diff `desired` against the database, after changes have been applied.

    The list is empty when the run converged. Anything in it is a declaration
    the run failed to reach, which only the *next* diff would otherwise report.
    With `skip_cluster_changes`, role-level changes are left out (a rehearsal
    on a clone does not apply them).
    """
    cfg = resolve(config)
    residual_actual = introspect_schema(
        connection_string=connection_string,
        schemas=list(cfg.schemas),
        config=cfg,
    )
    residual = compute_diff(
        desired,
        residual_actual,
        None,
        schemas=list(cfg.schemas),
        locked_functions=locked_functions,
        config=cfg,
    )
    if skip_cluster_changes:
        residual = [c for c in residual if not is_cluster_change(c)]
    return residual


def assert_schema_converged(
    desired: Schema,
    connection_string: str,
    force: bool = False,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
    *,
    skip_cluster_changes: bool = False,
) -> None:
    """Fail the run if a second diff would still find work to do."""
    cfg = resolve(config)
    logger.info("Verifying the schema converged...")
    residual = compute_residual_changes(
        desired,
        connection_string,
        locked_functions,
        cfg,
        skip_cluster_changes=skip_cluster_changes,
    )

    if not residual:
        cfg.report(
            cfg.decorate("green", "✓ Schema converged (a second diff finds no changes)")
        )
        return

    logger.error(
        f"Schema did not converge: {len(residual)} change(s) remain after "
        f"applying this run. One run must reach the declared schema."
    )
    for change in residual[:20]:
        logger.error(f"  - {change.description}")
    if len(residual) > 20:
        logger.error(f"  ... and {len(residual) - 20} more")

    if force:
        logger.warning("--force: continuing despite non-convergence")
        return

    raise SchemaConvergenceError(
        f"Schema did not converge: {len(residual)} change(s) remain after applying this run"
    )


def report_pending_migration_blocks(
    blocks: list[MigrationBlock],
    applied: dict,
    force: bool = False,
) -> None:
    """Say which inline migrations a real run would execute, without running any.

    A pending @migrate-before block can rename or drop what the diff below is
    computed against, so the preview says plainly that it did not apply them.
    """
    pending = [b for b in blocks if force or b.id not in applied]
    if not pending:
        logger.info("Dry run: no pending inline migrations")
        return
    for block in pending:
        logger.info(f"Dry run: would execute @migrate-{block.phase} {block.id}")
    if any(b.phase == "before" for b in pending):
        logger.warning(
            "Dry run: the diff below is computed WITHOUT the pending "
            "@migrate-before blocks, so a real run may plan different DDL."
        )


def apply_declared_schema(
    connection_string: str,
    sql_dir: Path,
    plan_db_name: str,
    *,
    offline: bool = False,
    force: bool = False,
    dry_run: bool = False,
    expect_converged: bool = False,
    prove: bool = False,
    prepare_clone: Optional[Callable[[str], None]] = None,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
) -> None:
    """Bring the database at `connection_string` to the schema `sql_dir` declares.

    Steps: build the desired schema in the plan database; introspect the live
    one; run pending @migrate-before blocks (re-introspecting after); diff;
    enforce @drop directives; apply; run pending @migrate-after blocks; run the
    DO blocks; on a fresh database record every inline block as satisfied; and
    finally check that a second diff finds nothing.

    On a fresh database (no tables) inline blocks are upgrade-path SQL, so they
    are recorded rather than executed.

    Args:
        connection_string: The database to migrate
        sql_dir: The SQL root directory
        plan_db_name: Scratch database, on the same server, the desired schema
            is built in. It must exist.
        offline: Allow offline operations (SET NOT NULL, type changes)
        force: Bypass @drop directive checks, re-run modified blocks, and
            continue despite non-convergence
        dry_run: Preview only. Nothing is written to the live database: no
            inline block, no ledger row, no DDL, no DO block. Only the plan
            database is rebuilt.
        expect_converged: Any difference fails the run before anything is applied
        prove: With `dry_run`, rehearse the real run on a throwaway clone of
            the database instead of previewing it (see
            `pgdeclare.prove.prove_declared_schema`). Needs `dry_run=True`.
        prepare_clone: With `prove`, called with the clone's connection
            string before the run (create schemas, extensions)
        locked_functions: Passed to `compute_diff` (SECDEF PUBLIC EXECUTE revoke)
        config: Engine configuration

    Raises:
        SchemaConvergenceError: expect_converged and the database differs, or
            the run did not converge.
        ProveFailedError: `prove` and the rehearsal failed.
        Exception: Any other failure, with the reason in the message.
    """
    if prove:
        if not dry_run:
            raise ValueError("prove=True rehearses on a clone; pass dry_run=True")
        from .prove import prove_declared_schema

        prove_declared_schema(
            connection_string,
            sql_dir,
            plan_db_name,
            offline=offline,
            force=force,
            prepare=prepare_clone,
            locked_functions=locked_functions,
            config=config,
        )
        return
    run_declared_schema(
        connection_string,
        sql_dir,
        plan_db_name,
        offline=offline,
        force=force,
        dry_run=dry_run,
        expect_converged=expect_converged,
        locked_functions=locked_functions,
        config=config,
    )


def run_declared_schema(
    connection_string: str,
    sql_dir: Path,
    plan_db_name: str,
    *,
    offline: bool = False,
    force: bool = False,
    dry_run: bool = False,
    expect_converged: bool = False,
    skip_cluster_changes: bool = False,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
) -> SchemaRunResult:
    """`apply_declared_schema`'s pipeline, returning what it did.

    `skip_cluster_changes` leaves role and role-membership changes out of the
    plan and the convergence check: they act on the whole server, so a
    rehearsal on a clone must not apply them.
    """
    cfg = resolve(config)
    say, style = cfg.report, cfg.decorate

    # Step 1: Build desired schema from SQL files
    logger.info("Building desired schema from SQL files...")
    desired = build_desired_schema(
        sql_dir=sql_dir,
        connection_string=connection_string,
        plan_db_name=plan_db_name,
        schemas=list(cfg.schemas),
        config=cfg,
    )

    # Step 2: Introspect actual database state
    logger.info("Introspecting current database state...")
    actual = introspect_schema(
        connection_string=connection_string,
        schemas=list(cfg.schemas),
        config=cfg,
    )

    # Step 3: Parse and execute @migrate-before blocks (only on existing databases)
    # On fresh installs, inline migrations are upgrade-path SQL. Skip execution
    # and record them as satisfied once the schema is created so follow-up
    # runs do not replay them against the current schema.
    is_fresh_install = len(actual.tables) == 0
    dry_run_pending_before = False
    migration_blocks = parse_migration_blocks(sql_dir, cfg)
    pre_migrations_executed = 0
    if is_fresh_install:
        logger.info(
            "Fresh database - skipping inline migration execution during bootstrap"
        )
        post_blocks = []
    else:
        logger.info("Processing inline migrations...")
        pre_blocks = [b for b in migration_blocks if b.phase == "before"]
        post_blocks = [b for b in migration_blocks if b.phase == "after"]
        if dry_run:
            applied = get_applied_migrations(connection_string, cfg)
            report_pending_migration_blocks(migration_blocks, applied, force=force)
            dry_run_pending_before = any(
                force or b.id not in applied for b in pre_blocks
            )
            pre_blocks = []
        for block in pre_blocks:
            success, error = execute_migration_block(
                connection_string, block, force_rerun=force, config=cfg
            )
            if not success and error:
                logger.error(f"Migration {block.id} failed: {error}")
                if not force:
                    raise Exception(f"Migration failed: {error}")
            pre_migrations_executed += 1

    # Step 4: Re-introspect if pre-migrations ran (they may have renamed tables/columns)
    if pre_migrations_executed > 0:
        logger.info("Re-introspecting database after migrations...")
        actual = introspect_schema(
            connection_string=connection_string,
            schemas=list(cfg.schemas),
            config=cfg,
        )

    # Step 5: Compute schema diff
    logger.info("Computing schema diff...")

    def check_data_fn(
        schema: str, table_name: str, column: Optional[str] = None
    ) -> bool:
        return check_table_has_data(
            connection_string, schema, table_name, column, config=cfg
        )

    changes = compute_diff(
        desired,
        actual,
        check_data_fn,
        schemas=list(cfg.schemas),
        locked_functions=locked_functions,
        config=cfg,
    )
    result = SchemaRunResult(
        desired=desired,
        migration_blocks=migration_blocks,
        fresh_install=is_fresh_install,
    )
    if skip_cluster_changes:
        result.skipped_cluster_changes = [c for c in changes if is_cluster_change(c)]
        changes = [c for c in changes if not is_cluster_change(c)]
    result.changes = changes

    if not changes:
        say(style("green", "✓ No schema changes needed"))

    if changes and expect_converged:
        # A database that was already migrated, or restored from a template of
        # this schema, must already be what the schema files declare. Applying
        # the difference would heal it silently, and a template whose key
        # missed an input would then never be seen; so say what differs and stop.
        say(format_changes_summary(changes, style))
        raise SchemaConvergenceError(
            f"--expect-converged: the database differs from the schema files "
            f"by {len(changes)} change(s). If it was restored from a DB "
            "template, the template key missed an input."
        )

    if changes:
        # Step 6: Display changes
        say(style("blue", "Schema changes detected:"))
        say(format_changes_summary(changes, style))

        # Step 7: Collect @drop directives from all migration blocks and validate
        logger.info("Validating @drop directives...")
        drop_directives: set[str] = set()
        for block in migration_blocks:
            drop_directives.update(block.drop_directives)
        drop_errors = validate_drops_runtime(
            connection_string=connection_string,
            changes=changes,
            drop_directives=drop_directives,
            config=cfg,
        )

        if drop_errors and not force and dry_run_pending_before:
            # The diff was computed without the pending @migrate-before blocks,
            # so a rename they perform still looks like a drop plus an add.
            logger.warning(format_drop_errors(drop_errors, cfg.decorate))
            logger.warning(
                "Dry run: pending @migrate-before blocks may resolve these "
                "(a rename shows as a drop plus an add until it runs)."
            )
        elif drop_errors and not force:
            logger.error(format_drop_errors(drop_errors, cfg.decorate))
            raise Exception("@drop directive required for data-loss operations")

        if drop_errors and force:
            logger.warning("Forcing execution despite @drop violations")

        # Step 8: Execute changes
        if dry_run:
            logger.info("Dry run - no changes applied")
            logger.info("SQL that would be executed:")
            for change in changes:
                logger.info(change.to_display_sql())
        else:
            success, executed_sql = execute_migration(
                connection_string=connection_string,
                changes=changes,
                offline_mode=offline,
                force=force,
                dry_run=False,
                drop_directives=drop_directives,
                config=cfg,
            )
            result.executed_sql = executed_sql

            if success:
                say(style("green", "✓ Schema changes applied successfully"))
            else:
                raise Exception("Migration execution failed")

    # Step 9: Execute @migrate-after blocks (need columns/tables created by schema diff)
    if not is_fresh_install and post_blocks and not dry_run:
        logger.info("Processing post-diff migrations (@migrate-after)...")
        for block in post_blocks:
            success, error = execute_migration_block(
                connection_string, block, force_rerun=force, config=cfg
            )
            if not success and error:
                logger.error(f"Migration {block.id} failed: {error}")
                if not force:
                    raise Exception(f"Migration failed: {error}")

    # Step 10: Execute DO blocks (idempotent setup tasks)
    # DO blocks run on every migration and must be idempotent.
    do_blocks = parse_do_blocks(sql_dir, cfg)
    if do_blocks:
        success, _do_sql = execute_do_blocks(
            connection_string=connection_string,
            do_blocks=do_blocks,
            dry_run=dry_run,
            search_path=cfg.search_path_sql,
            config=cfg,
        )
        if not success:
            raise Exception("DO block execution failed")

    if is_fresh_install and migration_blocks and not dry_run:
        logger.info("Fresh database - recording inline migrations as already satisfied")
        ensure_ledger(connection_string, cfg)
        mark_blocks_applied(connection_string, migration_blocks, cfg)

    # Step 11: Convergence check. One run must reach the declared
    # schema; a second run finding work means this one did not converge.
    # Re-diff the same desired schema against the database just written.
    if not dry_run:
        assert_schema_converged(
            desired=desired,
            connection_string=connection_string,
            force=force,
            locked_functions=locked_functions,
            config=cfg,
            skip_cluster_changes=skip_cluster_changes,
        )
    return result
