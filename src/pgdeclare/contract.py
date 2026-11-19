"""Inline migrations that take away what a running application may still use.

A deploy that migrates while the previous version of the application still
serves can hold back the differ's contract changes (`expand_only`), but an
inline `@migrate-before` / `@migrate-after` block is SQL the engine runs as
written. A pending block that drops, renames or moves an object, or drops or
retypes a column, breaks that version's queries from the moment it runs, and
cannot be postponed past a diff that may depend on it. An expand run refuses
such a block, so the migration has to run with the application stopped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pglast import parse_sql
from pglast.enums import AlterTableType, DropBehavior, ObjectType

from .ledger import MigrationBlock

# Dropping these takes nothing a query names.
_HARMLESS_DROPS = frozenset(
    {
        ObjectType.OBJECT_INDEX,
        ObjectType.OBJECT_TRIGGER,
        ObjectType.OBJECT_STATISTIC_EXT,
    }
)
# Renaming these changes nothing a query names.
_HARMLESS_RENAMES = frozenset(
    {
        ObjectType.OBJECT_INDEX,
        ObjectType.OBJECT_TRIGGER,
        ObjectType.OBJECT_TABCONSTRAINT,
        ObjectType.OBJECT_DOMCONSTRAINT,
        ObjectType.OBJECT_POLICY,
        ObjectType.OBJECT_STATISTIC_EXT,
    }
)
_BREAKING_TABLE_CMDS = frozenset(
    {
        AlterTableType.AT_DropColumn,
        AlterTableType.AT_AlterColumnType,
        AlterTableType.AT_SetNotNull,
    }
)
# A DO block is PL/pgSQL the parser does not look into, so its text is read.
_DO_BREAKING = re.compile(
    r"\bDROP\s+(?!INDEX\b|TRIGGER\b|DEFAULT\b|NOT\s+NULL\b|IDENTITY\b|EXPRESSION\b)\w+"
    r"|\bRENAME\b|\bSET\s+SCHEMA\b|\bSET\s+NOT\s+NULL\b|\bALTER\s+COLUMN\b[^;]*\bTYPE\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BreakingBlock:
    """A pending inline migration and what in it breaks the serving version."""

    block: MigrationBlock
    statements: tuple[str, ...]

    def describe(self) -> str:
        first = " ".join(self.statements[0].split())
        if len(first) > 120:
            first = first[:117] + "..."
        more = (
            f" (and {len(self.statements) - 1} more)"
            if len(self.statements) > 1
            else ""
        )
        return f"@migrate-{self.block.phase}:{self.block.id}: {first}{more}"


def _as_enum(enum_type, value):
    try:
        return enum_type(value)
    except ValueError:
        return None


def _breaks(stmt) -> bool:
    kind = type(stmt).__name__
    if kind == "DropStmt":
        return (
            _as_enum(ObjectType, stmt.removeType) not in _HARMLESS_DROPS
            or _as_enum(DropBehavior, stmt.behavior) == DropBehavior.DROP_CASCADE
        )
    if kind == "RenameStmt":
        return _as_enum(ObjectType, stmt.renameType) not in _HARMLESS_RENAMES
    if kind == "AlterObjectSchemaStmt":
        return True
    if kind == "AlterTableStmt":
        return any(
            _as_enum(AlterTableType, cmd.subtype) in _BREAKING_TABLE_CMDS
            for cmd in stmt.cmds or ()
            if type(cmd).__name__ == "AlterTableCmd"
        )
    if kind == "DoStmt":
        body = " ".join(str(getattr(arg.arg, "sval", "")) for arg in stmt.args or ())
        return bool(_DO_BREAKING.search(body))
    return False


def breaking_statements(sql: str) -> list[str]:
    """The statements in `sql` that take away what an older application uses.

    SQL that does not parse counts as breaking: what it does is unknown.
    """
    try:
        raw = parse_sql(sql)
    except Exception:
        return [sql.strip()]
    # pglast's locations count bytes, not characters.
    data = sql.encode()
    found: list[str] = []
    for item in raw:
        if _breaks(item.stmt):
            start = item.stmt_location or 0
            end = start + item.stmt_len if item.stmt_len else len(data)
            found.append(data[start:end].decode(errors="replace").strip())
    return found


def breaking_blocks(blocks: list[MigrationBlock], applied: dict) -> list[BreakingBlock]:
    """The blocks not yet in the ledger whose SQL breaks the serving version."""
    found: list[BreakingBlock] = []
    for block in blocks:
        if block.id in applied:
            continue
        statements = breaking_statements(block.sql)
        if statements:
            found.append(BreakingBlock(block, tuple(statements)))
    return found
