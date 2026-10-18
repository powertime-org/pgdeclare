"""
Prove a run before it happens: rehearse it on a throwaway clone.

A dry run can only plan the diff; it cannot run the pending @migrate-before
blocks, so a rename still looks like a drop plus an add, and a backfill's effect
is invisible. Proving clones the target database on the same server
(`CREATE DATABASE ... TEMPLATE`), runs the whole real pipeline against the clone
(before-blocks, diff, @drop checks, DDL, after-blocks, DO blocks, convergence
check), and then checks two things: a second diff finds nothing, and no table
lost rows unless a @drop directive covers it. The ledger is written only in the
clone, and the clone is always dropped. The target itself is only read, by the
clone's copy.

Roles and role memberships are cluster-wide, so applying them on the clone would
change the server for every database; the rehearsal leaves them out and lists
them instead.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.errors
from psycopg2 import sql as pgsql
from psycopg2.extensions import parse_dsn

from .changes import Change
from .config import PgDeclareConfig, logger, resolve
from .ledger import get_applied_migrations
from .migrate import compute_residual_changes, run_declared_schema
from .schema_parser import _replace_db_in_url

# Postgres caps identifiers at 63 bytes; the suffix is 23 of them.
_CLONE_SUFFIX = "_pgdeclare_prove_"
_MAX_IDENTIFIER = 63


class ProveError(Exception):
    """The rehearsal could not be set up (no clone, no privilege)."""


class ProveFailedError(ProveError):
    """The rehearsal ran and the real run would fail or lose data."""

    def __init__(self, reasons: list[str], result: Optional[ProveResult] = None):
        self.reasons = reasons
        self.result = result
        super().__init__("prove failed: " + "; ".join(reasons))


@dataclass(frozen=True)
class TableRows:
    """A table's row count before and after the rehearsal.

    `after` is None when the table no longer exists; `renamed_to` is set when
    it survived under another name (tables are matched by OID, which a clone
    keeps and a rename does not change).
    """

    table: str
    before: int
    after: Optional[int]
    renamed_to: Optional[str] = None
    covered_by_drop: bool = False

    @property
    def lost_rows(self) -> bool:
        return self.after is None or self.after < self.before


@dataclass(frozen=True)
class LedgerEntry:
    """A ledger row the real run would write."""

    id: str
    phase: str
    executed: bool  # False: recorded without running (fresh database)


@dataclass
class ProveResult:
    """What the real run would do, as rehearsed on the clone."""

    database: str
    clone: str
    changes: list[Change] = field(default_factory=list)
    executed_sql: list[str] = field(default_factory=list)
    ledger_entries: list[LedgerEntry] = field(default_factory=list)
    skipped_cluster_changes: list[Change] = field(default_factory=list)
    row_changes: list[TableRows] = field(default_factory=list)
    residual: list[Change] = field(default_factory=list)


def clone_name(database: str) -> str:
    """A fresh random clone name, `<database>_pgdeclare_prove_<hex>`, 63 bytes max."""
    token = secrets.token_hex(4)
    room = _MAX_IDENTIFIER - len(_CLONE_SUFFIX) - len(token)
    base = database.encode()[:room].decode(errors="ignore")
    return f"{base}{_CLONE_SUFFIX}{token}"


def _autocommit(
    dsn: str, statement: pgsql.Composable, params: tuple | None = None
) -> list:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            return cur.fetchall() if cur.description else []
    finally:
        conn.close()


def _other_sessions(admin_dsn: str, database: str) -> int:
    rows = _autocommit(
        admin_dsn,
        pgsql.SQL(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid() "
            "AND backend_type = 'client backend'"
        ),
        (database,),
    )
    return int(rows[0][0])


def _create_clone(admin_dsn: str, database: str, clone: str) -> None:
    try:
        _autocommit(
            admin_dsn,
            pgsql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                pgsql.Identifier(clone), pgsql.Identifier(database)
            ),
        )
    except psycopg2.errors.ObjectInUse as exc:
        try:
            count = _other_sessions(admin_dsn, database)
        except psycopg2.Error:
            count = 0
        # Autovacuum workers are cancelled by CREATE DATABASE itself; only
        # client sessions are worth naming, and they may have left meanwhile.
        others = f"{count} other session(s)" if count else "other sessions"
        raise ProveError(
            f"cannot clone {database!r} to prove the run: {others} are connected "
            "to it. Proving copies the database with CREATE DATABASE ... TEMPLATE, "
            "which needs exclusive access to the target. Run it while nothing else "
            "is connected, or against a staging copy of the database"
        ) from exc
    except psycopg2.errors.InsufficientPrivilege as exc:
        raise ProveError(
            f"cannot clone {database!r} to prove the run: {str(exc).strip()}. "
            "Proving needs CREATEDB, and the role must own the target database "
            "(or be a superuser)"
        ) from exc


def _drop_clone(admin_dsn: str, clone: str) -> None:
    _autocommit(
        admin_dsn,
        pgsql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
            pgsql.Identifier(clone)
        ),
    )


def table_row_counts(dsn: str, schemas: list[str]) -> dict[int, tuple[str, int]]:
    """Exact row counts of the tables in `schemas`, keyed by OID."""
    counts: dict[int, tuple[str, int]] = {}
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.oid, n.nspname, c.relname
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relkind IN ('r', 'p') AND n.nspname = ANY(%s)
                ORDER BY n.nspname, c.relname
                """,
                (schemas,),
            )
            tables = cur.fetchall()
            for oid, schema, name in tables:
                cur.execute(
                    pgsql.SQL("SELECT count(*) FROM {}.{}").format(
                        pgsql.Identifier(schema), pgsql.Identifier(name)
                    )
                )
                row = cur.fetchone()
                counts[oid] = (f"{schema}.{name}", int(row[0]) if row else 0)
    finally:
        conn.close()
    return counts


def compare_row_counts(
    before: dict[int, tuple[str, int]],
    after: dict[int, tuple[str, int]],
    drop_directives: set[str],
) -> list[TableRows]:
    """The tables whose row count changed, vanished, or were renamed."""

    def covered(qualified: str) -> bool:
        return qualified in drop_directives or (
            qualified.split(".", 1)[1] in drop_directives
        )

    rows: list[TableRows] = []
    for oid, (name, count) in before.items():
        survivor = after.get(oid)
        if survivor is None:
            rows.append(TableRows(name, count, None, covered_by_drop=covered(name)))
            continue
        new_name, new_count = survivor
        if new_count == count and new_name == name:
            continue
        rows.append(
            TableRows(
                name,
                count,
                new_count,
                renamed_to=new_name if new_name != name else None,
                covered_by_drop=covered(name) or covered(new_name),
            )
        )
    return rows


def prove_declared_schema(
    connection_string: str,
    sql_dir: Path,
    plan_db_name: str,
    *,
    offline: bool = False,
    force: bool = False,
    prepare: Optional[Callable[[str], None]] = None,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
) -> ProveResult:
    """Rehearse `apply_declared_schema` on a throwaway clone of the database.

    Clones the target with `CREATE DATABASE <clone> TEMPLATE <target>` on the
    same server (which needs CREATEDB, ownership of the target, and no other
    session connected to it), records each managed table's row count, runs the
    real pipeline against the clone with the same `offline` and `force`, then
    requires that a second diff is empty and that no table lost rows or
    vanished unless a @drop directive names it. The clone is always dropped.
    Nothing is written to the target.

    Args:
        connection_string: The target database (never written)
        sql_dir: The SQL root directory
        plan_db_name: Existing scratch database on the same server; the
            declared schema is built in it, and the clone is created and
            dropped through a connection to it
        offline: As for `apply_declared_schema`
        force: As for `apply_declared_schema`; the rehearsal still fails on
            residual changes and unexplained row loss
        prepare: Called with the clone's connection string before the run,
            for what the caller provisions outside the engine (schemas,
            extensions)
        locked_functions: Passed to `compute_diff`
        config: Engine configuration; `dry_run` and `on_migration_applied`
            are ignored for the clone

    Returns:
        A ProveResult saying what the real run would do.

    Raises:
        ProveError: The clone could not be created.
        ProveFailedError: The real run would fail, would not converge, or
            would lose rows no @drop directive covers.
    """
    cfg = resolve(config)
    say, style = cfg.report, cfg.decorate
    database = parse_dsn(connection_string).get("dbname")
    if not database:
        raise ProveError("the connection string names no database")
    if database == plan_db_name:
        raise ProveError("the plan database cannot be the target database")
    clone = clone_name(database)
    clone_dsn = _replace_db_in_url(connection_string, clone)
    admin_dsn = _replace_db_in_url(connection_string, plan_db_name)
    # The clone run reports through the summary below, writes its ledger in the
    # clone only, and must not fire callbacks meant for real applications.
    run_cfg = replace(
        cfg, dry_run=False, on_migration_applied=None, report=lambda _text: None
    )
    schemas = list(cfg.schemas)

    logger.info(f"Cloning {database} to {clone} to prove the run...")
    _create_clone(admin_dsn, database, clone)
    try:
        before = table_row_counts(clone_dsn, schemas)
        ledger_before = get_applied_migrations(clone_dsn, run_cfg)
        if prepare is not None:
            prepare(clone_dsn)
        try:
            run = run_declared_schema(
                clone_dsn,
                sql_dir,
                plan_db_name,
                offline=offline,
                force=force,
                skip_cluster_changes=True,
                locked_functions=locked_functions,
                config=run_cfg,
            )
        except Exception as exc:  # noqa: BLE001 - every failure fails the proof
            raise ProveFailedError(
                [f"the real run would fail: {str(exc).strip()}"]
            ) from exc

        result = ProveResult(
            database=database,
            clone=clone,
            changes=run.changes,
            executed_sql=run.executed_sql,
            skipped_cluster_changes=run.skipped_cluster_changes,
        )
        phases = {b.id: b.phase for b in run.migration_blocks}
        ledger_after = get_applied_migrations(clone_dsn, run_cfg)
        result.ledger_entries = [
            LedgerEntry(block_id, phases.get(block_id, "?"), not run.fresh_install)
            for block_id, entry in sorted(ledger_after.items())
            if ledger_before.get(block_id) != entry
        ]
        result.residual = compute_residual_changes(
            run.desired,
            clone_dsn,
            locked_functions,
            run_cfg,
            skip_cluster_changes=True,
        )
        drop_directives: set[str] = set()
        for block in run.migration_blocks:
            drop_directives.update(block.drop_directives)
        result.row_changes = compare_row_counts(
            before, table_row_counts(clone_dsn, schemas), drop_directives
        )
    finally:
        _drop_clone(admin_dsn, clone)
        logger.info(f"Dropped the prove clone {clone}")

    report_prove_result(result, say, style)

    reasons: list[str] = []
    if result.residual:
        reasons.append(
            f"the schema did not converge: {len(result.residual)} change(s) remain "
            "after the run ("
            + ", ".join(c.description for c in result.residual[:5])
            + (", ..." if len(result.residual) > 5 else "")
            + ")"
        )
    lost = [r for r in result.row_changes if r.lost_rows and not r.covered_by_drop]
    for r in lost:
        if r.after is None:
            reasons.append(
                f"table {r.table} ({r.before} row(s)) would vanish without a "
                "@drop directive"
            )
        else:
            reasons.append(
                f"table {r.table} would lose {r.before - r.after} of {r.before} "
                "row(s) without a @drop directive"
            )
    if reasons:
        raise ProveFailedError(reasons, result)
    say(style("green", "✓ Proved: the run converges and keeps every row"))
    return result


def report_prove_result(
    result: ProveResult,
    say: Callable[[str], None],
    style: Callable[[str, str], str],
) -> None:
    """Say what the rehearsal did, as the real run would do it."""
    say(style("blue", f"Prove: rehearsed on a clone of {result.database}"))
    if result.ledger_entries:
        for entry in result.ledger_entries:
            how = "would run" if entry.executed else "would be recorded, not run"
            say(f"  @migrate-{entry.phase}:{entry.id} {how}")
    else:
        say("  No inline migrations would run")
    if result.executed_sql:
        say(f"  DDL the run would execute ({len(result.executed_sql)} statement(s)):")
        for statement in result.executed_sql:
            for line in statement.strip().splitlines():
                say(f"    {line}")
    else:
        say("  No DDL would run")
    for change in result.skipped_cluster_changes:
        say(
            f"  Not rehearsed (cluster-wide, would apply for real): "
            f"{change.description}"
        )
    for rows in result.row_changes:
        if rows.after is None:
            what = f"{rows.before} row(s), table dropped"
        else:
            what = f"{rows.before} -> {rows.after} row(s)"
        if rows.renamed_to:
            what += f", renamed to {rows.renamed_to}"
        if rows.lost_rows and rows.covered_by_drop:
            what += " (covered by @drop)"
        say(f"  {rows.table}: {what}")
