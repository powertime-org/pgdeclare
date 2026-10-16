"""
Engine configuration.

Everything the engine needs to know about the project that uses it lives on one
frozen `PgDeclareConfig`, passed explicitly to the public entry points. The
defaults describe a plain PostgreSQL database: one `public` schema, no managed
roles, no extra bootstrap SQL, no lock, no failure classifier.

Logging goes to the standard `logging` logger named ``pgdeclare``, in plain
text. Whoever embeds the engine decides how (and whether) to render it.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("pgdeclare")


def _no_classifier(
    exc: BaseException, during: str, location: Optional[str] = None
) -> Optional[BaseException]:
    """Default failure classifier: every error is a SQL error, none is infrastructure."""
    return None


def plain(style: str, text: str) -> str:
    """Default text decorator: no styling."""
    return text


def _log_info(text: str) -> None:
    """Default reporter: user-facing progress lines go to the logger at INFO."""
    logger.info(text)


@dataclass(frozen=True)
class PgDeclareConfig:
    """How the engine sees one project.

    Attributes:
        schemas: Schemas the engine manages: introspected, diffed, and used to
            strip schema qualifiers from types during comparison.
        search_path: search_path used while loading SQL into the plan database,
            while introspecting column types, and for DO blocks.
        managed_roles: Roles whose definitions, memberships and grants are
            introspected and diffed. Roles outside this list are never touched.
        ledger_table: ``schema.table`` of the inline-migration ledger.
        schemas_subdir: Directory under the SQL root holding the hand-written
            schema files (recursive, may carry @migrate blocks).
        functions_subdir: Directory under the SQL root holding extra function
            files (flat).
        generated_subdir: Directory under the SQL root holding generated SQL
            (flat).
        git_sql_prefix: Path of the SQL root relative to the repository root,
            used when reading a branch's files with git.
        bootstrap_sql_files: SQL files loaded before the schema files when the
            desired schema is built (read from the working tree, never from git).
        bootstrap_source_label: The source label bootstrap statements carry in
            diagnostics.
        plan_recreate_schemas: Schemas dropped and recreated in the plan
            database before each build.
        externally_provisioned_schemas: ``CREATE SCHEMA`` statements for these
            names are skipped when building the desired schema, because
            something outside the engine (cluster-wide extension setup, say)
            creates them.
        secdef_schemas: Schemas whose SECURITY DEFINER functions are eligible
            for the locked-function PUBLIC EXECUTE revoke.
        grant_public_on_created_schemas: GRANT ALL on every schema the engine
            creates (`ensure_schemas`, the plan database) to PUBLIC. Off by
            default: anyone who can connect could then create objects there.
        dry_run: Preview only: nothing is written, introspection returns an
            empty schema, data probes report "no data".
        lock: Context-manager factory wrapped around every plan-database build.
        classify_failure: ``(exc, during, location) -> exception | None``.
            Returns an exception to raise in place of ``exc`` when the error is
            an infrastructure failure (server gone) rather than a SQL error.
        decorate: ``(style, text) -> text`` for user-facing text. ``style`` is
            one of "red", "yellow", "green", "blue", "dim".
        report: Where user-facing progress lines go.
        on_migration_applied: Called with ``(block, connection_string)`` after
            an inline migration block is executed and recorded.
    """

    schemas: tuple[str, ...] = ("public",)
    search_path: tuple[str, ...] = ("public",)
    managed_roles: tuple[str, ...] = ()
    ledger_table: str = "pgdeclare.applied_migrations"
    schemas_subdir: str = "schemas"
    functions_subdir: str = "functions"
    generated_subdir: str = "generated"
    git_sql_prefix: str = "sql"
    bootstrap_sql_files: tuple[Path, ...] = ()
    bootstrap_source_label: str = "<bootstrap>"
    plan_recreate_schemas: tuple[str, ...] = ("public",)
    externally_provisioned_schemas: tuple[str, ...] = ()
    secdef_schemas: tuple[str, ...] = ("public",)
    grant_public_on_created_schemas: bool = False
    dry_run: bool = False
    lock: Callable[[], AbstractContextManager[object]] = nullcontext
    classify_failure: Callable[
        [BaseException, str, Optional[str]], Optional[BaseException]
    ] = _no_classifier
    decorate: Callable[[str, str], str] = plain
    report: Callable[[str], None] = _log_info
    on_migration_applied: Optional[Callable[[object, str], None]] = None

    @property
    def search_path_sql(self) -> str:
        """The search path as a comma-separated list, ready for SET search_path."""
        return ", ".join(self.search_path)

    @property
    def ledger_schema(self) -> str:
        schema, _, _table = self.ledger_table.rpartition(".")
        return schema or "public"

    @property
    def ledger_name(self) -> str:
        return self.ledger_table.rpartition(".")[2]

    def git_path(self, subdir: str) -> str:
        """Repository-relative path of an SQL subdirectory, for git reads."""
        prefix = self.git_sql_prefix.strip("/")
        return f"{prefix}/{subdir}" if prefix else subdir


DEFAULT_CONFIG = PgDeclareConfig()


def resolve(config: Optional[PgDeclareConfig]) -> PgDeclareConfig:
    """The config to use when a caller passed none."""
    return DEFAULT_CONFIG if config is None else config
