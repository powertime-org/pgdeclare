"""The `pgdeclare` command line: a thin convenience over the library.

    pgdeclare diff  [--json] [--exit-code]
    pgdeclare apply [--dry-run] [--offline] [--force]

Both build the declared schema in a scratch "plan" database on the same server
(created on demand and dropped afterwards, unless `--plan-database` names one
you manage), introspect the target database, and compare the two. Everything
here goes through the public API; anything the flags do not cover is a
`PgDeclareConfig` field away in Python.

Exit codes: 0 success (or nothing to do), 1 error, 2 changes pending
(`diff --exit-code` only).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psycopg2
from psycopg2 import sql as pgsql

from . import __version__
from .changes import Change
from .config import PgDeclareConfig
from .differ import compute_diff
from .executor import format_changes_summary
from .introspection import check_table_has_data, introspect_schema
from .ledger import get_applied_migrations, parse_migration_blocks
from .migrate import SchemaConvergenceError, apply_declared_schema
from .schema_parser import build_desired_schema, ensure_schemas
from .statement_parser import collect_sql_files_recursive, parse_sql_content

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CHANGES = 2


class CliError(Exception):
    """A failure the CLI reports as one line, without a traceback."""


def _database_name(url: str) -> str:
    if "://" not in url:
        raise CliError(
            "the database URL must be a postgresql:// URL "
            "(key=value connection strings are not supported)"
        )
    name = urlsplit(url).path.lstrip("/")
    if not name:
        raise CliError("the database URL names no database")
    return name


def _with_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return parts._replace(path=f"/{name}").geturl()


def _autocommit(
    url: str, statement: str | pgsql.Composable, params: Any = None
) -> list:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            return cur.fetchall() if cur.description else []
    finally:
        conn.close()


@contextmanager
def plan_database(url: str, requested: str | None) -> Iterator[str]:
    """Yield the plan database's name, creating it if it does not exist.

    A database this call created is dropped on the way out; one that already
    existed is left alone (its managed schemas are rebuilt on every run).
    """
    name = requested or f"{_database_name(url)}_pgdeclare_plan"
    exists = _autocommit(
        url, pgsql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), (name,)
    )
    created = False
    if not exists:
        _autocommit(url, pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(name)))
        created = True
    try:
        yield name
    finally:
        if created:
            _autocommit(
                url,
                pgsql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    pgsql.Identifier(name)
                ),
            )


_IF_NOT_EXISTS = re.compile(r"^\s*CREATE\s+EXTENSION\s+(?!IF\s+NOT\s+EXISTS)", re.I)


def declared_extensions(sql_dir: Path, config: PgDeclareConfig) -> list[str]:
    """The CREATE EXTENSION statements in the schema files, made idempotent.

    The engine leaves extensions to the caller (they are cluster-level and often
    need a superuser). The CLI installs the declared ones so a schema that uses
    one can be built at all.
    """
    statements: list[str] = []
    schemas_dir = sql_dir / config.schemas_subdir
    if not schemas_dir.exists():
        return statements
    for path in collect_sql_files_recursive(schemas_dir):
        for stmt in parse_sql_content(path.read_text(), source_file=str(path)):
            if stmt.statement_type == "CREATE EXTENSION":
                raw = stmt.raw_sql.strip()
                statements.append(
                    _IF_NOT_EXISTS.sub("CREATE EXTENSION IF NOT EXISTS ", raw)
                )
    return statements


@contextmanager
def _bootstrap_file(statements: list[str]) -> Iterator[tuple[Path, ...]]:
    """Write the extension statements to a file the engine loads first."""
    if not statements:
        yield ()
        return
    with tempfile.TemporaryDirectory(prefix="pgdeclare-") as tmp:
        path = Path(tmp) / "extensions.sql"
        path.write_text("\n".join(statements) + "\n")
        yield (path,)


def _build_config(
    args: argparse.Namespace, report: Callable[[str], None], bootstrap: tuple[Path, ...]
) -> PgDeclareConfig:
    schemas = tuple(args.schema or ("public",))
    kwargs: dict[str, Any] = {
        "schemas": schemas,
        "search_path": schemas,
        "plan_recreate_schemas": schemas,
        "secdef_schemas": schemas,
        "managed_roles": tuple(args.role or ()),
        "bootstrap_sql_files": bootstrap,
        "bootstrap_source_label": "<declared extensions>",
        "report": report,
    }
    if args.ledger_table:
        kwargs["ledger_table"] = args.ledger_table
    return PgDeclareConfig(**kwargs)


def change_to_dict(change: Change) -> dict[str, Any]:
    """The JSON shape of one planned change."""
    return {
        "kind": type(change).__name__,
        "category": change.category.value,
        "object_type": change.object_type,
        "object_name": change.object_name,
        "description": change.description,
        "requires_drop_directive": change.requires_drop_directive,
        "sql": change.to_display_sql(),
    }


def _stderr(text: str) -> None:
    print(text, file=sys.stderr)


def cmd_diff(args: argparse.Namespace) -> int:
    url = args.database_url
    sql_dir = Path(args.sql_dir)
    probe = _build_config(args, _stderr, ())
    with _bootstrap_file(declared_extensions(sql_dir, probe)) as bootstrap:
        cfg = _build_config(args, _stderr, bootstrap)
        with plan_database(url, args.plan_database) as plan_db:
            desired = build_desired_schema(sql_dir, url, plan_db, config=cfg)
        actual = introspect_schema(url, config=cfg)

    def has_data(schema: str, table: str, column: str | None = None) -> bool:
        return check_table_has_data(url, schema, table, column, config=cfg)

    changes = compute_diff(
        desired, actual, has_data, schemas=list(cfg.schemas), config=cfg
    )
    applied = get_applied_migrations(url, cfg)
    pending = [b for b in parse_migration_blocks(sql_dir, cfg) if b.id not in applied]

    if args.json:
        payload = {
            "fresh_database": not actual.tables,
            "changes": [change_to_dict(c) for c in changes],
            "pending_migrations": [
                {
                    "id": b.id,
                    "phase": b.phase,
                    "source_file": str(b.source_file),
                    "sql_hash": b.sql_hash,
                }
                for b in pending
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        if changes:
            print(format_changes_summary(changes))
            print()
            for change in changes:
                print(f"-- {change.description}")
                print(change.to_display_sql())
        else:
            print("No schema changes.")
        fresh = not actual.tables
        for block in pending:
            how = " (fresh database: recorded, not run)" if fresh else ""
            print(f"Pending inline migration: @migrate-{block.phase}:{block.id}{how}")
        if any(b.phase == "before" for b in pending):
            print(
                "Note: the diff above does not include the effect of the pending "
                "@migrate-before blocks."
            )

    if args.exit_code and (changes or pending):
        return EXIT_CHANGES
    return EXIT_OK


def cmd_apply(args: argparse.Namespace) -> int:
    url = args.database_url
    sql_dir = Path(args.sql_dir)
    probe = _build_config(args, print, ())
    extensions = declared_extensions(sql_dir, probe)
    with _bootstrap_file(extensions) as bootstrap:
        cfg = _build_config(args, print, bootstrap)
        if not args.dry_run:
            ensure_schemas(url, cfg.schemas, cfg)
            for statement in extensions:
                _autocommit(url, statement)
        with plan_database(url, args.plan_database) as plan_db:
            apply_declared_schema(
                url,
                sql_dir,
                plan_db,
                offline=args.offline,
                force=args.force,
                dry_run=args.dry_run,
                config=cfg,
            )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgdeclare",
        description="Declarative Postgres schemas: diff the live database "
        "against your SQL files and converge it.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="postgresql:// URL of the target database (default: $DATABASE_URL)",
    )
    common.add_argument(
        "--sql-dir",
        default="sql",
        help="SQL root; schema files live in its schemas/ subdirectory (default: sql)",
    )
    common.add_argument(
        "--schema",
        action="append",
        metavar="NAME",
        help="a schema to manage; repeatable (default: public)",
    )
    common.add_argument(
        "--role",
        action="append",
        metavar="NAME",
        help="a role whose definition, memberships and grants are managed; repeatable",
    )
    common.add_argument(
        "--ledger-table",
        metavar="SCHEMA.TABLE",
        help="inline-migration ledger (default: pgdeclare.applied_migrations)",
    )
    common.add_argument(
        "--plan-database",
        metavar="NAME",
        help="scratch database the declared schema is built in "
        "(default: <database>_pgdeclare_plan, created and dropped per run)",
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", help="log the engine's progress"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    diff = sub.add_parser(
        "diff", parents=[common], help="print the changes apply would make"
    )
    diff.add_argument("--json", action="store_true", help="machine-readable output")
    diff.add_argument(
        "--exit-code",
        action="store_true",
        help="exit 2 when there are changes or pending inline migrations",
    )
    diff.set_defaults(func=cmd_diff)

    apply = sub.add_parser(
        "apply",
        parents=[common],
        help="run @migrate-before blocks, the diff, then @migrate-after blocks",
    )
    apply.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only: nothing is written to the target database",
    )
    apply.add_argument(
        "--offline",
        action="store_true",
        help="allow changes that rewrite or lock tables (SET NOT NULL, type changes)",
    )
    apply.add_argument(
        "--force",
        action="store_true",
        help="bypass @drop checks, re-run edited blocks, ignore non-convergence",
    )
    apply.set_defaults(func=cmd_apply)
    return parser


class _Formatter(logging.Formatter):
    """Progress lines bare; warnings and errors with their level."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def _configure_logging(verbose: bool) -> None:
    """Send the engine's log to stderr: INFO when verbose (or dry run), else WARNING."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter())
    engine = logging.getLogger("pgdeclare")
    engine.handlers[:] = [handler]
    engine.setLevel(logging.INFO if verbose else logging.WARNING)
    engine.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(
        verbose=args.verbose or getattr(args, "dry_run", False),
    )

    if not args.database_url:
        _stderr(
            "pgdeclare: error: no database (pass --database-url or set DATABASE_URL)"
        )
        return EXIT_ERROR
    if not Path(args.sql_dir).is_dir():
        _stderr(f"pgdeclare: error: SQL directory not found: {args.sql_dir}")
        return EXIT_ERROR

    try:
        return args.func(args)
    except SchemaConvergenceError as exc:
        _stderr(f"pgdeclare: error: {exc}")
    except CliError as exc:
        _stderr(f"pgdeclare: error: {exc}")
    except psycopg2.Error as exc:
        _stderr(f"pgdeclare: database error: {str(exc).strip()}")
    except Exception as exc:  # noqa: BLE001 - the CLI reports, the library raises
        _stderr(f"pgdeclare: error: {exc}")
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
