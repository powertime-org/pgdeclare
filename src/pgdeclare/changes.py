"""
Schema change types and SQL generation.

Each Change subclass represents a specific database modification with:
- Category (online/offline/blocked)
- SQL statement generation
- Description for logging
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from .quoting import do_block, dollar_tag, quote_ident, quote_literal, role_ident
from .models import (
    ChangeCategory,
    Collation,
    Column,
    Comment,
    CompositeType,
    Constraint,
    DefaultPrivilege,
    Domain,
    EnumType,
    EventTrigger,
    Extension,
    ForeignServer,
    ForeignTable,
    Function,
    Grant,
    Index,
    MaterializedView,
    ObjectOwner,
    Policy,
    RangeType,
    Publication,
    PublicationTable,
    Role,
    RoleMembership,
    Rule,
    Sequence,
    Table,
    Trigger,
    UserMapping,
    View,
    routine_args_sql,
    routine_keyword,
    alter_fdw_options_sql,
    fdw_options_sql,
    option_names_changed,
)


# Constraint types CreateTable must NOT inline into the CREATE TABLE body; the
# differ emits each of them as its own AddConstraint instead.
#
# 'f' (FOREIGN KEY) is deferred so circular references resolve (post <->
# comment).
#
# 'u' (UNIQUE) and 'x' (EXCLUDE) are deferred because PostgreSQL SILENTLY
# DISCARDS a redundant index constraint declared inline.
# transformIndexConstraints() in parse_utilcmd.c deduplicates the inline index
# list by column set, keeping the primary key if there is one and otherwise the
# first constraint, so
#
#   CREATE TABLE enrollment (
#     ...,
#     CONSTRAINT enrollment_pkey PRIMARY KEY (course_id, student_id),
#     CONSTRAINT enrollment_course_id_student_id_key UNIQUE (course_id, student_id)
#   );
#
# creates ONLY enrollment_pkey, with no error and no notice. ALTER TABLE ADD
# CONSTRAINT does no such deduplication, so the plan database, which is built by
# executing declarative files that use standalone ALTER TABLE, keeps both.
# Inlined, fresh installs would go through CreateTable and lose the UNIQUE while
# existing databases went through _diff_constraints and gained it. The two paths
# would disagree, which is the one thing the declarative model exists to prevent.
DEFERRED_CONSTRAINT_TYPES = frozenset({"f", "u", "x"})


@dataclass
class Change(ABC):
    """Base class for all schema changes."""

    category: ChangeCategory
    description: str
    requires_drop_directive: bool = False
    object_type: str = ""
    object_name: str = ""

    @abstractmethod
    def to_sql(self) -> str:
        """Generate SQL statement for this change."""
        pass

    def to_display_sql(self) -> str:
        """The SQL for printing, logging and JSON: secrets masked.

        `to_sql()` is what executes. Anything shown to a person or written to a
        log goes through this instead, so a declared role password never leaves
        the process.
        """
        return self.to_sql()


# =============================================================================
# Table Changes
# =============================================================================


@dataclass
class CreateTable(Change):
    """Create a new table."""

    table: Table = field(default_factory=lambda: Table(schema="public", name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = self.table.qualified_name
        if not self.description:
            if self.table.partition_of is not None:
                self.description = (
                    f"Create partition {self.table.qualified_name} of "
                    f"{self.table.parent_key} ({self.table.partition_bound})"
                )
            else:
                self.description = f"Create table {self.table.qualified_name}"

    def _inline_constraints(self) -> list[str]:
        # Only PRIMARY KEY and CHECK constraints are inlined. FOREIGN KEY,
        # UNIQUE and EXCLUDE are emitted as separate AddConstraint changes by
        # the differ; see DEFERRED_CONSTRAINT_TYPES for why.
        return [
            f"    CONSTRAINT {quote_ident(constraint.name)} {constraint.definition}"
            for constraint in self.table.constraints
            if constraint.type not in DEFERRED_CONSTRAINT_TYPES
        ]

    def _partition_sql(self) -> str:
        """`CREATE TABLE ... PARTITION OF parent ... FOR VALUES ...`.

        The columns, their defaults and NOT NULL come from the parent; a
        partition's own overrides are written by the differ afterwards.
        """
        table = self.table
        assert table.partition_of is not None
        parent = (
            f"{quote_ident(table.partition_of[0])}.{quote_ident(table.partition_of[1])}"
        )
        sql = f"CREATE TABLE {quote_ident(table.schema)}.{quote_ident(table.name)} PARTITION OF {parent}"
        constraints = self._inline_constraints()
        if constraints:
            sql += f" (\n{',\n'.join(constraints)}\n)"
        sql += f" {table.partition_bound}"
        if table.partition_key:
            sql += f" PARTITION BY {table.partition_key}"
        return sql + ";"

    def to_sql(self) -> str:
        if self.table.partition_of is not None:
            return self._partition_sql()
        columns_sql = []
        for col in self.table.columns.values():
            col_def = f"    {quote_ident(col.name)} {col.type}"
            if col.collation:
                col_def += f" COLLATE {col.collation}"
            # Add IDENTITY clause before NOT NULL
            if col.identity == "a":
                col_def += " GENERATED ALWAYS AS IDENTITY"
            elif col.identity == "d":
                col_def += " GENERATED BY DEFAULT AS IDENTITY"
            if not col.nullable:
                col_def += " NOT NULL"
            if col.generated:
                # STORED generated column: never carries a DEFAULT.
                col_def += f" GENERATED ALWAYS AS ({col.generated}) STORED"
            elif col.default and not col.identity:
                # IDENTITY columns shouldn't have explicit DEFAULT
                col_def += f" DEFAULT {col.default}"
            columns_sql.append(col_def)

        all_parts = columns_sql + self._inline_constraints()
        partition_by = (
            f" PARTITION BY {self.table.partition_key}"
            if self.table.partition_key
            else ""
        )
        return f"CREATE TABLE {quote_ident(self.table.schema)}.{quote_ident(self.table.name)} (\n{',\n'.join(all_parts)}\n){partition_by};"


def _qualified(name: tuple[str, str]) -> str:
    return f"{quote_ident(name[0])}.{quote_ident(name[1])}"


@dataclass
class AttachPartition(Change):
    """Attach a table as a partition, detaching it from its current parent first.

    Used for a changed bound (detach and attach again to the same parent), a
    partition moved to another parent, and a plain table declared as a
    partition. ATTACH scans the table to check every row fits the bound, and a
    default partition to check none of its rows do, under a lock, so this is
    offline. Both statements run in one transaction.
    """

    schema: str = "public"
    table: str = ""
    parent: tuple[str, str] = ("public", "")
    bound: str = ""
    detach_from: Optional[tuple[str, str]] = None
    old_bound: Optional[str] = None
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            parent = f"{self.parent[0]}.{self.parent[1]}"
            if self.detach_from == self.parent:
                self.description = (
                    f"Change bound of partition {self.object_name} of {parent}: "
                    f"{self.old_bound} -> {self.bound} (DETACH, then ATTACH, "
                    "which scans the partition to validate it)"
                )
            else:
                self.description = (
                    f"Attach {self.object_name} to {parent} as a partition "
                    f"({self.bound}; ATTACH scans the table to validate it)"
                )

    def to_sql(self) -> str:
        table = f"{quote_ident(self.schema)}.{quote_ident(self.table)}"
        statements = []
        if self.detach_from is not None:
            statements.append(
                f"ALTER TABLE {_qualified(self.detach_from)} DETACH PARTITION {table};"
            )
        statements.append(
            f"ALTER TABLE {_qualified(self.parent)} ATTACH PARTITION {table} {self.bound};"
        )
        return "\n".join(statements)


@dataclass
class DetachPartition(Change):
    """Detach a partition that is now declared as a plain table."""

    schema: str = "public"
    table: str = ""
    parent: tuple[str, str] = ("public", "")
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            self.description = (
                f"Detach partition {self.object_name} from "
                f"{self.parent[0]}.{self.parent[1]}"
            )

    def to_sql(self) -> str:
        return (
            f"ALTER TABLE {_qualified(self.parent)} DETACH PARTITION "
            f"{quote_ident(self.schema)}.{quote_ident(self.table)};"
        )


@dataclass
class DropTable(Change):
    """Drop an existing table."""

    schema: str = "public"
    table_name: str = ""
    has_data: bool = False
    category: ChangeCategory = ChangeCategory.BLOCKED
    description: str = ""
    object_type: str = "TABLE"
    requires_drop_directive: bool = True

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table_name}"
        if not self.description:
            self.description = f"Drop table {self.object_name}"
        # Only require @drop if table has data
        self.requires_drop_directive = self.has_data

    def to_sql(self) -> str:
        return f"DROP TABLE IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.table_name)} CASCADE;"


# =============================================================================
# Column Changes
# =============================================================================


@dataclass
class AddColumn(Change):
    """Add a column to an existing table."""

    schema: str = "public"
    table: str = ""
    column: Column = field(
        default_factory=lambda: Column(name="", type="", nullable=True)
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column.name}"
        if not self.description:
            self.description = (
                f"Add column {self.column.name} to {self.schema}.{self.table}"
            )
        # Adding a STORED generated column to an existing (possibly populated)
        # table takes ACCESS EXCLUSIVE and rewrites the whole table to compute
        # values, so it must be gated behind --offline. (New tables get the
        # column inlined via CreateTable, which never reaches AddColumn.)
        if self.column.generated:
            self.category = ChangeCategory.OFFLINE

    def to_sql(self) -> str:
        sql = f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ADD COLUMN {quote_ident(self.column.name)} {self.column.type}"
        if self.column.collation:
            sql += f" COLLATE {self.column.collation}"
        # Add IDENTITY clause before NOT NULL
        if self.column.identity == "a":
            sql += " GENERATED ALWAYS AS IDENTITY"
        elif self.column.identity == "d":
            sql += " GENERATED BY DEFAULT AS IDENTITY"
        if not self.column.nullable:
            sql += " NOT NULL"
        if self.column.generated:
            # STORED generated column: never carries a DEFAULT.
            sql += f" GENERATED ALWAYS AS ({self.column.generated}) STORED"
        elif self.column.default and not self.column.identity:
            # IDENTITY columns shouldn't have explicit DEFAULT
            sql += f" DEFAULT {self.column.default}"
        return sql + ";"


@dataclass
class DropColumn(Change):
    """Drop a column from a table."""

    schema: str = "public"
    table: str = ""
    column_name: str = ""
    has_data: bool = False
    category: ChangeCategory = ChangeCategory.BLOCKED
    description: str = ""
    object_type: str = "COLUMN"
    requires_drop_directive: bool = True

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column_name}"
        if not self.description:
            self.description = (
                f"Drop column {self.column_name} from {self.schema}.{self.table}"
            )
        # Only require @drop if column has data
        self.requires_drop_directive = self.has_data

    def to_sql(self) -> str:
        # Use CASCADE to handle any dependent constraints, indexes, or views
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} DROP COLUMN {quote_ident(self.column_name)} CASCADE;"


@dataclass
class AlterColumnType(Change):
    """Change column type (may require data conversion)."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    old_type: str = ""
    new_type: str = ""
    # The COLLATE clause written with the type: a quoted collation name, or
    # None for no clause (the new type's default).
    collation: Optional[str] = None
    category: ChangeCategory = ChangeCategory.TRY_FIRST
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            if self.old_type == self.new_type:
                self.description = (
                    f"Alter column {self.column} collation to {self.collation}"
                )
            else:
                self.description = f"Alter column {self.column} type from {self.old_type} to {self.new_type}"

    def to_sql(self) -> str:
        collate = f" COLLATE {self.collation}" if self.collation else ""
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} TYPE {self.new_type}{collate} USING {quote_ident(self.column)}::{self.new_type};"


@dataclass
class SetNotNull(Change):
    """Add NOT NULL constraint to column (offline - requires full table scan)."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Set NOT NULL on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} SET NOT NULL;"


@dataclass
class DropNotNull(Change):
    """Remove NOT NULL constraint from column."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Drop NOT NULL on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} DROP NOT NULL;"


@dataclass
class SetDefault(Change):
    """Set default value for column."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    default_value: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Set default on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} SET DEFAULT {self.default_value};"


@dataclass
class DropDefault(Change):
    """Remove default value from column."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Drop default on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} DROP DEFAULT;"


@dataclass
class AlterColumnSetExpression(Change):
    """Replace the generation expression of a STORED generated column (PG17+).

    SET EXPRESSION rewrites the whole table under ACCESS EXCLUSIVE to recompute
    stored values, so it is categorized OFFLINE (gated behind --offline). Note:
    this is deliberately NOT keyed to AlterColumnType's category (TRY_FIRST),
    which execute_migration() would let run on a live DB without the gate.
    """

    schema: str = "public"
    table: str = ""
    column: str = ""
    expression: str = ""
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Set generation expression on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} SET EXPRESSION AS ({self.expression});"


@dataclass
class AlterColumnDropExpression(Change):
    """Convert a STORED generated column into a normal base column (PG13+).

    DROP EXPRESSION retains the existing stored values (no rewrite), so ONLINE.
    """

    schema: str = "public"
    table: str = ""
    column: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Drop generation expression on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} DROP EXPRESSION;"


@dataclass
class AddIdentity(Change):
    """Add IDENTITY to a column.

    When adding IDENTITY to a column with existing data, the underlying sequence
    must be synced to avoid duplicate key violations. This generates a DO block
    that adds the identity and syncs the sequence atomically.
    """

    schema: str = "public"
    table: str = ""
    column: str = ""
    identity_type: str = "d"  # 'a' (ALWAYS) or 'd' (BY DEFAULT)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            kind = "ALWAYS" if self.identity_type == "a" else "BY DEFAULT"
            self.description = f"Add GENERATED {kind} AS IDENTITY on {self.column}"

    def to_sql(self) -> str:
        kind = "ALWAYS" if self.identity_type == "a" else "BY DEFAULT"
        # Use DO block to atomically add identity and sync sequence with existing data
        table = f"{quote_ident(self.schema)}.{quote_ident(self.table)}"
        select_max = f'SELECT COALESCE(MAX("%I"), 0) FROM {table}'
        body = f"""DECLARE
    seq_name text;
    max_val bigint;
BEGIN
    ALTER TABLE {table} ALTER COLUMN {quote_ident(self.column)} ADD GENERATED {kind} AS IDENTITY;
    -- Sync sequence with existing data to avoid duplicate key violations
    SELECT pg_get_serial_sequence({quote_literal(f"{self.schema}.{self.table}")}, {quote_literal(self.column)}) INTO seq_name;
    IF seq_name IS NOT NULL THEN
        EXECUTE format({quote_literal(select_max)}, {quote_literal(self.column)}) INTO max_val;
        IF max_val > 0 THEN
            PERFORM setval(seq_name, max_val);
        END IF;
    END IF;
END"""
        tag = dollar_tag(body)
        return f"DO {tag}\n{body} {tag};"


@dataclass
class DropIdentity(Change):
    """Remove IDENTITY from a column."""

    schema: str = "public"
    table: str = ""
    column: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLUMN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.column}"
        if not self.description:
            self.description = f"Drop IDENTITY on {self.column}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ALTER COLUMN {quote_ident(self.column)} DROP IDENTITY IF EXISTS;"


# =============================================================================
# Constraint Changes
# =============================================================================


@dataclass
class AddConstraint(Change):
    """Add a constraint to a table."""

    schema: str = "public"
    table: str = ""
    constraint: Constraint = field(
        default_factory=lambda: Constraint(name="", type="", definition="")
    )
    not_valid: bool = True  # Use NOT VALID for online addition
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "CONSTRAINT"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.constraint.name}"
        if not self.description:
            self.description = f"Add constraint {self.constraint.name}"

    def to_sql(self) -> str:
        not_valid_clause = (
            " NOT VALID"
            if self.not_valid and self.constraint.type in ("f", "c")
            else ""
        )
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ADD CONSTRAINT {quote_ident(self.constraint.name)} {self.constraint.definition}{not_valid_clause};"


@dataclass
class ValidateConstraint(Change):
    """Validate a NOT VALID constraint."""

    schema: str = "public"
    table: str = ""
    constraint_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "CONSTRAINT"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.constraint_name}"
        if not self.description:
            self.description = f"Validate constraint {self.constraint_name}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} VALIDATE CONSTRAINT {quote_ident(self.constraint_name)};"


@dataclass
class DropConstraint(Change):
    """Drop a constraint from a table."""

    schema: str = "public"
    table: str = ""
    constraint_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "CONSTRAINT"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.constraint_name}"
        if not self.description:
            self.description = f"Drop constraint {self.constraint_name}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} DROP CONSTRAINT IF EXISTS {quote_ident(self.constraint_name)};"


@dataclass
class DropAndCreateConstraint(Change):
    """Drop and recreate a constraint (required when definition changes)."""

    schema: str = "public"
    table: str = ""
    constraint: Constraint = field(
        default_factory=lambda: Constraint(name="", type="", definition="")
    )
    # Re-add CHECK and FOREIGN KEY constraints NOT VALID (validated afterwards).
    # False where Postgres refuses NOT VALID: a foreign key on a partitioned
    # table before PostgreSQL 18.
    not_valid: bool = True
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "CONSTRAINT"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.constraint.name}"
        if not self.description:
            self.description = (
                f"Drop and recreate constraint {self.constraint.name} on {self.table}"
            )

    def to_sql(self) -> str:
        # Wrap in DO block for atomicity - if ADD fails, DROP is rolled back
        drop_sql = f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} DROP CONSTRAINT IF EXISTS {quote_ident(self.constraint.name)}"
        # Add the new constraint (NOT VALID for CHECK/FK constraints)
        not_valid_clause = (
            " NOT VALID"
            if self.not_valid and self.constraint.type in ("f", "c")
            else ""
        )
        add_sql = f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ADD CONSTRAINT {quote_ident(self.constraint.name)} {self.constraint.definition}{not_valid_clause}"
        # Escape single quotes for EXECUTE
        escaped_add = add_sql.replace("'", "''")
        return do_block(f"BEGIN {drop_sql}; EXECUTE '{escaped_add}'; END")


# =============================================================================
# Function Changes
# =============================================================================


def grant_replay_note(count: int) -> str:
    """Plan-summary suffix for a recreate that replays `count` grants."""
    if not count:
        return ""
    return f" (replaying {count} grant{'s' if count > 1 else ''})"


def _replay_grants_sql(grants: list[Grant]) -> str:
    """Render grants as `EXECUTE '...';` statements for a recreate's DO block.

    DROP wipes an object's ACL. Replaying the desired grants inside the same DO
    block as the drop+create makes the ACL atomic with the object, so it cannot
    be lost when the executor reorders a deferred recreate past a standalone
    GRANT. Returns "" when there is nothing to replay.
    """
    if not grants:
        return ""

    parts = []
    for grant in grants:
        grant_sql = GrantPrivilege(grant=grant).to_sql().rstrip().rstrip(";")
        escaped = grant_sql.replace("'", "''")
        parts.append(f" EXECUTE '{escaped}';")
    return "".join(parts)


def _revoke_public_execute_sql(
    schema: str, name: str, args: str, keyword: str = "FUNCTION"
) -> str:
    """Bare `REVOKE EXECUTE ... FROM public` (no trailing `;`, no DO wrapper).

    Used for SECDEF lockdowns. Emitted either appended atomically to a
    function's CREATE/recreate (so PostgreSQL's default PUBLIC EXECUTE grant is
    revoked in the same change), or standalone for pure ACL drift.
    """
    # GRANT/REVOKE has no AGGREGATE object type; an aggregate is a FUNCTION here.
    if keyword == "AGGREGATE":
        keyword = "FUNCTION"
    return f"REVOKE EXECUTE ON {keyword} {quote_ident(schema)}.{quote_ident(name)}({args}) FROM public"


@dataclass
class CreateFunction(Change):
    """Create a function, procedure or aggregate (CREATE OR REPLACE)."""

    function: Function = field(
        default_factory=lambda: Function(
            schema="public", name="", args="", full_definition=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FUNCTION"
    # When True, append a REVOKE EXECUTE ... FROM public so the function's
    # default PUBLIC EXECUTE (granted by PostgreSQL at CREATE time) is revoked
    # atomically in the same change. Set by _diff_function_public_execute for
    # locked SECDEF functions being created.
    revoke_public_execute: bool = False

    def __post_init__(self):
        self.object_name = self.function.signature
        if not self.description:
            self.description = (
                f"Create {self.function.keyword.lower()} {self.function.signature}"
            )

    def to_sql(self) -> str:
        # The full_definition already contains CREATE OR REPLACE FUNCTION
        # (or PROCEDURE, or AGGREGATE)
        sql = self.function.full_definition
        if self.revoke_public_execute:
            revoke = _revoke_public_execute_sql(
                self.function.schema,
                self.function.name,
                self.function.args,
                self.function.keyword,
            )
            # Run in the same change (one savepoint/unit) so the default PUBLIC
            # EXECUTE granted at CREATE time is immediately revoked.
            return f"{sql.rstrip().rstrip(';')};\n{revoke};"
        return sql


@dataclass
class ReplaceFunction(Change):
    """Replace an existing function (same as CreateFunction but semantically different)."""

    function: Function = field(
        default_factory=lambda: Function(
            schema="public", name="", args="", full_definition=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FUNCTION"

    def __post_init__(self):
        self.object_name = self.function.signature
        if not self.description:
            self.description = (
                f"Replace {self.function.keyword.lower()} {self.function.signature}"
            )

    def to_sql(self) -> str:
        return self.function.full_definition


@dataclass
class DropFunction(Change):
    """Drop a function, procedure or aggregate."""

    schema: str = "public"
    name: str = ""
    args: str = ""
    # pg_proc.prokind of the routine dropped; DROP FUNCTION refuses a
    # procedure or an aggregate.
    kind: str = "f"
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FUNCTION"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}({self.args})"
        if not self.description:
            keyword = routine_keyword(self.kind).lower()
            self.description = f"Drop {keyword} {self.object_name}"

    def to_sql(self) -> str:
        # Use CASCADE to handle trigger dependencies
        keyword = routine_keyword(self.kind)
        args = routine_args_sql(self.kind, self.args)
        return f"DROP {keyword} IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.name)}({args}) CASCADE;"


@dataclass
class DropAndCreateFunction(Change):
    """Drop and recreate a routine.

    Required when the return type, the parameter names or the kind (function,
    procedure, aggregate) change, none of which CREATE OR REPLACE can do.
    """

    function: Function = field(
        default_factory=lambda: Function(
            schema="public", name="", args="", full_definition=""
        )
    )
    old_args: Optional[str] = (
        None  # For parameter renames: args of the function to drop
    )
    # pg_proc.prokind of the routine to drop, when it differs from the new
    # one's: DROP has to name what is there now.
    old_kind: Optional[str] = None
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FUNCTION"
    # See CreateFunction.revoke_public_execute. The REVOKE is emitted INSIDE the
    # DO block, after the recreate, so the drop+create+revoke are one atomic
    # unit, closing the deferred-recreate clobber.
    revoke_public_execute: bool = False
    # DROP wipes the function's ACL. Same reasoning as
    # DropAndCreateView.grants - replay the desired grants inside the DO block.
    grants: list[Grant] = field(default_factory=list)

    def __post_init__(self):
        self.object_name = self.function.signature
        if not self.description:
            self.description = (
                f"Drop and recreate {self.function.keyword.lower()} "
                f"{self.function.signature}" + grant_replay_note(len(self.grants))
            )

    def to_sql(self) -> str:
        # Use old_args if provided (for parameter renames), otherwise use current args
        # Wrap in DO block for atomicity - if CREATE fails, DROP is rolled back
        # CASCADE is safe here: dependents (triggers/views) are dropped atomically but
        # recreated when their owning tables are processed; if CREATE fails, everything rolls back
        drop_kind = self.old_kind if self.old_kind is not None else self.function.kind
        drop_args = routine_args_sql(
            drop_kind,
            self.old_args if self.old_args is not None else self.function.args,
        )
        drop_sql = f"DROP {routine_keyword(drop_kind)} IF EXISTS {quote_ident(self.function.schema)}.{quote_ident(self.function.name)}({drop_args}) CASCADE"
        # Escape single quotes in function definition for EXECUTE
        escaped_def = self.function.full_definition.replace("'", "''")
        inner = f"{drop_sql}; EXECUTE '{escaped_def}';"
        inner += _replay_grants_sql(self.grants)
        if self.revoke_public_execute:
            revoke = _revoke_public_execute_sql(
                self.function.schema,
                self.function.name,
                self.function.args,
                self.function.keyword,
            ).replace("'", "''")
            inner += f" EXECUTE '{revoke}';"
        return do_block(f"BEGIN {inner} END")


# =============================================================================
# Index Changes
# =============================================================================


@dataclass
class CreateIndex(Change):
    """Create an index (CONCURRENTLY by default)."""

    index: Index = field(
        default_factory=lambda: Index(name="", table="", schema="public", definition="")
    )
    concurrent: bool = True
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "INDEX"

    def __post_init__(self):
        self.object_name = self.index.qualified_name
        if not self.description:
            self.description = f"Create index {self.index.name} on {self.index.table}"

    def to_sql(self) -> str:
        if self.concurrent:
            # Modify CREATE INDEX to CREATE INDEX CONCURRENTLY
            definition = self.index.definition
            if "CREATE INDEX " in definition and "CONCURRENTLY" not in definition:
                definition = definition.replace(
                    "CREATE INDEX ", "CREATE INDEX CONCURRENTLY ", 1
                )
            return definition
        return self.index.definition


@dataclass
class DropIndex(Change):
    """Drop an index (CONCURRENTLY by default)."""

    schema: str = "public"
    index_name: str = ""
    concurrent: bool = True
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "INDEX"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.index_name}"
        if not self.description:
            self.description = f"Drop index {self.index_name}"

    def to_sql(self) -> str:
        concurrently = "CONCURRENTLY " if self.concurrent else ""
        return f"DROP INDEX {concurrently}IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.index_name)};"


# Postgres truncates identifiers to NAMEDATALEN - 1 bytes.
_MAX_IDENTIFIER_BYTES = 63


def _truncate_identifier(name: str, reserve: int = 0) -> str:
    raw = name.encode()[: _MAX_IDENTIFIER_BYTES - reserve]
    return raw.decode(errors="ignore")


@dataclass
class CreatePartitionedIndex(CreateIndex):
    """Create an index on a partitioned table without a long lock.

    `CREATE INDEX` on a partitioned table cannot be CONCURRENTLY, and builds
    every partition's index under a lock. Instead, the parent's index is
    created `ON ONLY` the parent (invalid and empty), each partition's index
    is built CONCURRENTLY and attached to it, and the parent's index turns
    valid once every partition has one. A sub-partitioned partition gets an
    `ON ONLY` index of its own, filled the same way.

    The partitions are read when the change runs, so ones created at runtime
    and not declared get the index too, and a run that died half way resumes:
    an existing parent index is kept and only partitions without an attached
    index are built.
    """

    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Create index {self.index.name} on partitioned table "
                f"{self.index.table} (per partition CONCURRENTLY, then attached)"
            )
        super().__post_init__()

    def to_sql(self) -> str:
        return (
            f"{self.index.definition};\n"
            "-- then, for each partition: CREATE INDEX CONCURRENTLY on it and "
            f"ALTER INDEX {quote_ident(self.index.schema)}."
            f"{quote_ident(self.index.name)} ATTACH PARTITION <its index>"
        )

    def execute(self, cur) -> list[str]:
        """Run the build on an autocommit cursor; returns the statements run."""
        executed: list[str] = []

        def run(statement: str) -> None:
            cur.execute(statement)
            executed.append(statement)

        index_oid = _index_oid(cur, self.index.schema, self.index.name)
        if index_oid is None:
            run(self.index.definition)
            index_oid = _index_oid(cur, self.index.schema, self.index.name)
            assert index_oid is not None
        _fill_partitioned_index(cur, index_oid, run)
        return executed


def _index_oid(cur, schema: str, name: str) -> Optional[int]:
    cur.execute(
        """
        SELECT c.oid FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = %s AND c.relkind IN ('i', 'I')
        """,
        (schema, name),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _free_index_name(cur, schema: str, base: str) -> str:
    """`base`, or `base_N`, whichever is not yet a relation in `schema`."""
    candidate = _truncate_identifier(base)
    n = 0
    while True:
        cur.execute(
            """
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, candidate),
        )
        if cur.fetchone() is None:
            return candidate
        n += 1
        suffix = f"_{n}"
        candidate = _truncate_identifier(base, len(suffix.encode())) + suffix


def _fill_partitioned_index(cur, index_oid: int, run) -> None:
    """Give every partition of `index_oid`'s table an index attached to it."""
    cur.execute(
        """
        SELECT n.nspname, c.relname, t.relname, ix.indisunique, ix.indrelid,
               pg_get_indexdef(c.oid),
               format('CREATE %%sINDEX %%s ON ONLY %%s.%%s ',
                      CASE WHEN ix.indisunique THEN 'UNIQUE ' ELSE '' END,
                      quote_ident(c.relname), quote_ident(tn.nspname),
                      quote_ident(t.relname))
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index ix ON ix.indexrelid = c.oid
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_namespace tn ON tn.oid = t.relnamespace
        WHERE c.oid = %s
        """,
        (index_oid,),
    )
    row = cur.fetchone()
    assert row is not None
    index_schema, index_name, _table, unique, table_oid, definition, prefix = row
    if not definition.startswith(prefix):
        raise ValueError(
            f"cannot build partitioned index {quote_ident(index_schema)}."
            f"{quote_ident(index_name)}: unexpected definition {definition!r}"
        )
    # Everything after the table name: "USING btree (col) INCLUDE ... WHERE ...".
    tail = definition[len(prefix) :]
    parent_index = f"{quote_ident(index_schema)}.{quote_ident(index_name)}"
    create = "CREATE UNIQUE INDEX" if unique else "CREATE INDEX"

    cur.execute(
        """
        SELECT c.oid, n.nspname, c.relname, c.relkind,
               (SELECT x.indexrelid FROM pg_inherits ii
                JOIN pg_index x ON x.indexrelid = ii.inhrelid
                WHERE ii.inhparent = %s AND x.indrelid = c.oid) AS attached,
               (SELECT x.indisvalid FROM pg_inherits ii
                JOIN pg_index x ON x.indexrelid = ii.inhrelid
                WHERE ii.inhparent = %s AND x.indrelid = c.oid) AS attached_valid
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE i.inhparent = %s
        ORDER BY n.nspname, c.relname
        """,
        (index_oid, index_oid, table_oid),
    )
    partitions = cur.fetchall()
    for _oid, schema, name, relkind, attached, attached_valid in partitions:
        if relkind == "f":
            continue  # Foreign tables cannot be indexed; Postgres skips them too.
        table = f"{quote_ident(schema)}.{quote_ident(name)}"
        if attached is not None:
            if relkind == "p" and not attached_valid:
                _fill_partitioned_index(cur, attached, run)
            continue
        child_name = _free_index_name(cur, schema, f"{name}_{index_name}")
        child = f"{quote_ident(schema)}.{quote_ident(child_name)}"
        if relkind == "p":
            run(f"{create} {quote_ident(child_name)} ON ONLY {table} {tail}")
            child_oid = _index_oid(cur, schema, child_name)
            assert child_oid is not None
            _fill_partitioned_index(cur, child_oid, run)
        else:
            try:
                run(
                    f"{create} CONCURRENTLY {quote_ident(child_name)} ON {table} {tail}"
                )
            except Exception:
                # A failed CONCURRENTLY build leaves an invalid index behind.
                cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {child}")
                raise
        run(f"ALTER INDEX {parent_index} ATTACH PARTITION {child}")


# =============================================================================
# Policy Changes
# =============================================================================


@dataclass
class CreatePolicy(Change):
    """Create an RLS policy."""

    policy: Policy = field(
        default_factory=lambda: Policy(name="", table="", schema="public", command="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "POLICY"

    def __post_init__(self):
        self.object_name = self.policy.qualified_key
        if not self.description:
            self.description = (
                f"Create policy {self.policy.name} on {self.policy.table}"
            )

    def to_sql(self) -> str:
        sql = f"CREATE POLICY {quote_ident(self.policy.name)} ON {quote_ident(self.policy.schema)}.{quote_ident(self.policy.table)}"
        if not self.policy.permissive:
            # PERMISSIVE is the PG default; only emit AS RESTRICTIVE for the
            # non-default case (cleaner diffs, matches pg_dump conventions).
            sql += " AS RESTRICTIVE"
        if self.policy.command and self.policy.command != "ALL":
            sql += f" FOR {self.policy.command}"
        if self.policy.roles:
            sql += f" TO {', '.join(role_ident(r) for r in self.policy.roles)}"
        if self.policy.using_expr:
            sql += f" USING ({self.policy.using_expr})"
        if self.policy.check_expr:
            sql += f" WITH CHECK ({self.policy.check_expr})"
        return sql + ";"


@dataclass
class DropPolicy(Change):
    """Drop an RLS policy."""

    schema: str = "public"
    table: str = ""
    policy_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "POLICY"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.policy_name}"
        if not self.description:
            self.description = f"Drop policy {self.policy_name} on {self.table}"

    def to_sql(self) -> str:
        return f"DROP POLICY IF EXISTS {quote_ident(self.policy_name)} ON {quote_ident(self.schema)}.{quote_ident(self.table)};"


@dataclass
class DropAndCreatePolicy(Change):
    """Drop and recreate an RLS policy (required when policy expression changes)."""

    policy: Policy = field(
        default_factory=lambda: Policy(name="", table="", schema="public", command="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "POLICY"

    def __post_init__(self):
        self.object_name = self.policy.qualified_key
        if not self.description:
            self.description = (
                f"Drop and recreate policy {self.policy.name} on {self.policy.table}"
            )

    def to_sql(self) -> str:
        # Wrap in DO block for atomicity - if CREATE fails, DROP is rolled back
        drop_sql = f"DROP POLICY IF EXISTS {quote_ident(self.policy.name)} ON {quote_ident(self.policy.schema)}.{quote_ident(self.policy.table)}"

        # Build create statement
        create_sql = f"CREATE POLICY {quote_ident(self.policy.name)} ON {quote_ident(self.policy.schema)}.{quote_ident(self.policy.table)}"
        if not self.policy.permissive:
            create_sql += " AS RESTRICTIVE"
        if self.policy.command and self.policy.command != "ALL":
            create_sql += f" FOR {self.policy.command}"
        if self.policy.roles:
            create_sql += f" TO {', '.join(role_ident(r) for r in self.policy.roles)}"
        if self.policy.using_expr:
            create_sql += f" USING ({self.policy.using_expr})"
        if self.policy.check_expr:
            create_sql += f" WITH CHECK ({self.policy.check_expr})"

        # Escape single quotes for EXECUTE
        escaped_create = create_sql.replace("'", "''")
        return do_block(f"BEGIN {drop_sql}; EXECUTE '{escaped_create}'; END")


@dataclass
class EnableRLS(Change):
    """Enable row-level security on a table."""

    schema: str = "public"
    table: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            self.description = f"Enable RLS on {self.table}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} ENABLE ROW LEVEL SECURITY;"


@dataclass
class DisableRLS(Change):
    """Disable row-level security on a table."""

    schema: str = "public"
    table: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            self.description = f"Disable RLS on {self.table}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} DISABLE ROW LEVEL SECURITY;"


@dataclass
class ForceRLS(Change):
    """Force row-level security on a table (applies to table owner too)."""

    schema: str = "public"
    table: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            self.description = f"Force RLS on {self.table}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} FORCE ROW LEVEL SECURITY;"


@dataclass
class NoForceRLS(Change):
    """Remove force row-level security from a table."""

    schema: str = "public"
    table: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}"
        if not self.description:
            self.description = f"Remove force RLS on {self.table}"

    def to_sql(self) -> str:
        return f"ALTER TABLE {quote_ident(self.schema)}.{quote_ident(self.table)} NO FORCE ROW LEVEL SECURITY;"


# =============================================================================
# Trigger Changes
# =============================================================================


@dataclass
class CreateTrigger(Change):
    """Create a trigger."""

    trigger: Trigger = field(
        default_factory=lambda: Trigger(
            name="", table="", schema="public", definition=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TRIGGER"

    def __post_init__(self):
        self.object_name = self.trigger.qualified_key
        if not self.description:
            self.description = (
                f"Create trigger {self.trigger.name} on {self.trigger.table}"
            )

    def to_sql(self) -> str:
        # Use CREATE OR REPLACE for idempotent trigger creation (PostgreSQL 14+)
        # Note: CONSTRAINT TRIGGERs don't support OR REPLACE, need DROP + CREATE
        definition = self.trigger.definition
        upper_def = definition.upper()
        if upper_def.startswith("CREATE CONSTRAINT TRIGGER"):
            # Constraint triggers don't support OR REPLACE - prepend DROP IF EXISTS
            drop_sql = f"DROP TRIGGER IF EXISTS {quote_ident(self.trigger.name)} ON {quote_ident(self.trigger.schema)}.{quote_ident(self.trigger.table)};"
            return drop_sql + "\n" + definition + ";"
        elif upper_def.startswith("CREATE TRIGGER"):
            definition = "CREATE OR REPLACE" + definition[6:]
        return definition + ";"


@dataclass
class DropTrigger(Change):
    """Drop a trigger."""

    schema: str = "public"
    table: str = ""
    trigger_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TRIGGER"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.trigger_name}"
        if not self.description:
            self.description = f"Drop trigger {self.trigger_name} on {self.table}"

    def to_sql(self) -> str:
        return f"DROP TRIGGER IF EXISTS {quote_ident(self.trigger_name)} ON {quote_ident(self.schema)}.{quote_ident(self.table)};"


# =============================================================================
# Enum Changes
# =============================================================================


@dataclass
class CreateEnum(Change):
    """Create an enum type."""

    enum: EnumType = field(
        default_factory=lambda: EnumType(schema="public", name="", values=[])
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = self.enum.qualified_name
        if not self.description:
            self.description = f"Create enum type {self.enum.name}"

    def to_sql(self) -> str:
        values_str = ", ".join(quote_literal(v) for v in self.enum.values)
        return f"CREATE TYPE {quote_ident(self.enum.schema)}.{quote_ident(self.enum.name)} AS ENUM ({values_str});"


@dataclass
class AddEnumValue(Change):
    """Add a value to an existing enum type."""

    schema: str = "public"
    enum_name: str = ""
    value: str = ""
    after: Optional[str] = None  # Insert after this value
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.enum_name}"
        if not self.description:
            self.description = f"Add value '{self.value}' to enum {self.enum_name}"

    def to_sql(self) -> str:
        sql = f"ALTER TYPE {quote_ident(self.schema)}.{quote_ident(self.enum_name)} ADD VALUE {quote_literal(self.value)}"
        if self.after:
            sql += f" AFTER {quote_literal(self.after)}"
        return sql + ";"


@dataclass
class DropEnumValue(Change):
    """
    Drop a value from an enum type.

    Note: PostgreSQL doesn't support dropping enum values directly.
    This requires recreating the type, which is an offline operation.
    """

    schema: str = "public"
    enum_name: str = ""
    value: str = ""
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.enum_name}"
        if not self.description:
            self.description = f"Drop value '{self.value}' from enum {self.enum_name} (requires type recreation)"

    def to_sql(self) -> str:
        # This is a placeholder - actual implementation requires
        # recreating the enum type, which needs special handling
        note = f"Cannot directly drop enum value '{self.value}' from {self.enum_name}; requires type recreation"
        return "-- " + " ".join(note.splitlines())


# =============================================================================
# Composite Type Changes
# =============================================================================


@dataclass
class CreateCompositeType(Change):
    """Create a composite type.

    Composite types are structured types with named columns, used for:
    - Function parameters and return types
    - Array element types (e.g., sort_spec[])
    - Table column types

    Example:
        CREATE TYPE sort_spec AS (
            column_name TEXT,
            direction sort_direction
        );
    """

    composite_type: CompositeType = field(
        default_factory=lambda: CompositeType(schema="public", name="", columns=[])
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = self.composite_type.qualified_name
        if not self.description:
            self.description = f"Create composite type {self.composite_type.name}"

    def to_sql(self) -> str:
        cols = ", ".join(
            f"{quote_ident(name)} {typ}" for name, typ in self.composite_type.columns
        )
        return f"CREATE TYPE {quote_ident(self.composite_type.schema)}.{quote_ident(self.composite_type.name)} AS ({cols});"


# =============================================================================
# Domain Changes
# =============================================================================

_NOT_VALID_SUFFIX = " NOT VALID"


def _domain_ident(schema: str, name: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(name)}"


def _valid_check(definition: str) -> str:
    """A CHECK definition without pg_get_constraintdef's NOT VALID marker."""
    if definition.endswith(_NOT_VALID_SUFFIX):
        return definition[: -len(_NOT_VALID_SUFFIX)]
    return definition


def _create_domain_sql(domain: Domain) -> str:
    sql = f"CREATE DOMAIN {_domain_ident(domain.schema, domain.name)} AS {domain.base_type}"
    if domain.collation:
        sql += f" COLLATE {domain.collation}"
    if domain.default is not None:
        sql += f" DEFAULT {domain.default}"
    if not domain.nullable:
        sql += " NOT NULL"
    for constraint in sorted(domain.constraints, key=lambda c: c.name):
        sql += f" CONSTRAINT {quote_ident(constraint.name)} {_valid_check(constraint.definition)}"
    return sql + ";"


@dataclass
class CreateDomain(Change):
    """Create a domain with its default, NOT NULL and CHECK constraints."""

    domain: Domain = field(
        default_factory=lambda: Domain(schema="public", name="", base_type="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = self.domain.qualified_name
        if not self.description:
            self.description = f"Create domain {self.domain.qualified_name}"

    def to_sql(self) -> str:
        return _create_domain_sql(self.domain)


@dataclass
class DropDomain(Change):
    """Drop a domain that is no longer declared.

    Never CASCADE: a column still typed with the domain is dropped or retyped
    by its own change, and the executor retries this drop until it has been.
    """

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Drop domain {self.object_name}"

    def to_sql(self) -> str:
        return f"DROP DOMAIN IF EXISTS {_domain_ident(self.schema, self.name)};"


@dataclass
class DropAndCreateDomain(Change):
    """Recreate a domain whose base type or collation changed.

    The differ emits this only when nothing in the database uses the domain,
    so the plain DROP cannot cascade into data.
    """

    domain: Domain = field(
        default_factory=lambda: Domain(schema="public", name="", base_type="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = self.domain.qualified_name
        if not self.description:
            self.description = f"Drop and recreate domain {self.domain.qualified_name}"

    def to_sql(self) -> str:
        drop = f"DROP DOMAIN {_domain_ident(self.domain.schema, self.domain.name)};"
        return f"{drop}\n{_create_domain_sql(self.domain)}"


@dataclass
class AlterDomainSetDefault(Change):
    """Set a domain's default."""

    schema: str = "public"
    name: str = ""
    default_value: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Set default on domain {self.object_name}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.name)} SET DEFAULT {self.default_value};"


@dataclass
class AlterDomainDropDefault(Change):
    """Remove a domain's default."""

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Drop default on domain {self.object_name}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.name)} DROP DEFAULT;"


@dataclass
class AlterDomainSetNotNull(Change):
    """Make a domain NOT NULL (offline: scans every column typed with it)."""

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Set NOT NULL on domain {self.object_name}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.name)} SET NOT NULL;"


@dataclass
class AlterDomainDropNotNull(Change):
    """Allow NULL in a domain."""

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Drop NOT NULL on domain {self.object_name}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.name)} DROP NOT NULL;"


@dataclass
class AddDomainConstraint(Change):
    """Add a CHECK constraint to a domain, NOT VALID until validated."""

    schema: str = "public"
    domain: str = ""
    constraint: Constraint = field(
        default_factory=lambda: Constraint(name="", type="c", definition="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.domain}.{self.constraint.name}"
        if not self.description:
            self.description = f"Add constraint {self.constraint.name} to domain {self.schema}.{self.domain}"

    def to_sql(self) -> str:
        return (
            f"ALTER DOMAIN {_domain_ident(self.schema, self.domain)} "
            f"ADD CONSTRAINT {quote_ident(self.constraint.name)} "
            f"{_valid_check(self.constraint.definition)}{_NOT_VALID_SUFFIX};"
        )


@dataclass
class DropDomainConstraint(Change):
    """Drop a CHECK constraint from a domain."""

    schema: str = "public"
    domain: str = ""
    constraint_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.domain}.{self.constraint_name}"
        if not self.description:
            self.description = f"Drop constraint {self.constraint_name} from domain {self.schema}.{self.domain}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.domain)} DROP CONSTRAINT IF EXISTS {quote_ident(self.constraint_name)};"


@dataclass
class DropAndCreateDomainConstraint(Change):
    """Replace a domain CHECK constraint whose definition changed."""

    schema: str = "public"
    domain: str = ""
    constraint: Constraint = field(
        default_factory=lambda: Constraint(name="", type="c", definition="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.domain}.{self.constraint.name}"
        if not self.description:
            self.description = f"Drop and recreate constraint {self.constraint.name} on domain {self.schema}.{self.domain}"

    def to_sql(self) -> str:
        target = _domain_ident(self.schema, self.domain)
        name = quote_ident(self.constraint.name)
        drop_sql = f"ALTER DOMAIN {target} DROP CONSTRAINT IF EXISTS {name}"
        add_sql = f"ALTER DOMAIN {target} ADD CONSTRAINT {name} {_valid_check(self.constraint.definition)}{_NOT_VALID_SUFFIX}"
        return do_block(f"BEGIN {drop_sql}; {add_sql}; END")


@dataclass
class ValidateDomainConstraint(Change):
    """Validate a NOT VALID domain constraint against existing data."""

    schema: str = "public"
    domain: str = ""
    constraint_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DOMAIN"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.domain}.{self.constraint_name}"
        if not self.description:
            self.description = f"Validate constraint {self.constraint_name} on domain {self.schema}.{self.domain}"

    def to_sql(self) -> str:
        return f"ALTER DOMAIN {_domain_ident(self.schema, self.domain)} VALIDATE CONSTRAINT {quote_ident(self.constraint_name)};"


# =============================================================================
# Range Type Changes
# =============================================================================


def _create_range_sql(range_type: RangeType) -> str:
    options = [f"SUBTYPE = {range_type.subtype}"]
    if range_type.subtype_opclass:
        options.append(f"SUBTYPE_OPCLASS = {range_type.subtype_opclass}")
    if range_type.collation:
        options.append(f"COLLATION = {range_type.collation}")
    if range_type.canonical:
        options.append(f"CANONICAL = {range_type.canonical}")
    if range_type.subtype_diff:
        options.append(f"SUBTYPE_DIFF = {range_type.subtype_diff}")
    if range_type.multirange_type_name:
        options.append(f"MULTIRANGE_TYPE_NAME = {range_type.multirange_type_name}")
    target = _domain_ident(range_type.schema, range_type.name)
    return f"CREATE TYPE {target} AS RANGE ({', '.join(options)});"


@dataclass
class CreateShellType(Change):
    """Create a shell type, for a range whose canonical function takes it.

    The canonical function's signature names the range type, so the type has
    to exist (as a shell) before the function, and the function before the
    range is defined over the shell.
    """

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Create shell type {self.object_name}"

    def to_sql(self) -> str:
        return f"CREATE TYPE {_domain_ident(self.schema, self.name)};"


@dataclass
class CreateRangeType(Change):
    """Create a range type, with its multirange named explicitly."""

    range_type: RangeType = field(
        default_factory=lambda: RangeType(schema="public", name="", subtype="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = self.range_type.qualified_name
        if not self.description:
            self.description = f"Create range type {self.range_type.qualified_name}"

    def to_sql(self) -> str:
        return _create_range_sql(self.range_type)


@dataclass
class DropRangeType(Change):
    """Drop a range type (and its multirange) that is no longer declared.

    Never a blind CASCADE: a column still typed with the range is dropped or
    retyped by its own change, and the executor retries this drop until it
    has been. A range with a canonical function is the exception PostgreSQL
    forces: the function takes the type and the type names the function, so
    neither can be dropped first. That drop checks, in the same statement,
    that nothing but the canonical function still depends on the type, and
    only then drops it with CASCADE.
    """

    schema: str = "public"
    name: str = ""
    has_canonical: bool = False
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Drop range type {self.object_name}"

    def to_sql(self) -> str:
        target = _domain_ident(self.schema, self.name)
        if not self.has_canonical:
            return f"DROP TYPE IF EXISTS {target};"
        literal = quote_literal(target)
        return do_block(
            "DECLARE r regtype := to_regtype("
            + literal
            + "); BEGIN IF r IS NULL THEN RETURN; END IF; "
            "IF EXISTS (SELECT 1 FROM pg_range g "
            "JOIN pg_type t ON t.oid = g.rngtypid "
            "JOIN pg_type m ON m.oid = g.rngmultitypid "
            "JOIN pg_depend d ON d.refclassid = 'pg_type'::regclass "
            "AND d.refobjid IN (t.oid, t.typarray, m.oid, m.typarray) "
            "AND d.deptype = 'n' "
            "WHERE g.rngtypid = r "
            "AND NOT (d.classid = 'pg_proc'::regclass AND d.objid = g.rngcanonical) "
            "AND NOT EXISTS (SELECT 1 FROM pg_depend o "
            "WHERE o.classid = d.classid AND o.objid = d.objid "
            "AND o.refclassid = 'pg_type'::regclass "
            "AND o.refobjid IN (t.oid, m.oid) AND o.deptype = 'i')) "
            "THEN RAISE EXCEPTION USING ERRCODE = 'dependent_objects_still_exist', "
            "MESSAGE = format('range type %s is still in use', r); END IF; "
            f"DROP TYPE {target} CASCADE; END"
        )


@dataclass
class DropAndCreateRangeType(Change):
    """Recreate a range type whose definition changed.

    The differ emits this only when nothing in the database uses the range
    or its multirange, so the plain DROP cannot cascade into data.
    """

    range_type: RangeType = field(
        default_factory=lambda: RangeType(schema="public", name="", subtype="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "TYPE"

    def __post_init__(self):
        self.object_name = self.range_type.qualified_name
        if not self.description:
            self.description = (
                f"Drop and recreate range type {self.range_type.qualified_name}"
            )

    def to_sql(self) -> str:
        target = _domain_ident(self.range_type.schema, self.range_type.name)
        return f"DROP TYPE {target};\n{_create_range_sql(self.range_type)}"


# =============================================================================
# Collation Changes
# =============================================================================


def _create_collation_sql(collation: Collation) -> str:
    options = [f"provider = {collation.provider}"]
    if collation.provider == "libc":
        options.append(f"lc_collate = {quote_literal(collation.lc_collate or '')}")
        options.append(f"lc_ctype = {quote_literal(collation.lc_ctype or '')}")
    else:
        options.append(f"locale = {quote_literal(collation.locale or '')}")
    if collation.rules:
        options.append(f"rules = {quote_literal(collation.rules)}")
    if not collation.deterministic:
        options.append("deterministic = false")
    target = _domain_ident(collation.schema, collation.name)
    return f"CREATE COLLATION {target} ({', '.join(options)});"


@dataclass
class CreateCollation(Change):
    """Create a collation."""

    collation: Collation = field(
        default_factory=lambda: Collation(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLLATION"

    def __post_init__(self):
        self.object_name = self.collation.qualified_name
        if not self.description:
            self.description = f"Create collation {self.collation.qualified_name}"

    def to_sql(self) -> str:
        return _create_collation_sql(self.collation)


@dataclass
class DropCollation(Change):
    """Drop a collation that is no longer declared.

    Never CASCADE: a column, domain or range still using it is dropped or
    changed by its own change, and the executor retries this drop until then.
    """

    schema: str = "public"
    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLLATION"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}"
        if not self.description:
            self.description = f"Drop collation {self.object_name}"

    def to_sql(self) -> str:
        return f"DROP COLLATION IF EXISTS {_domain_ident(self.schema, self.name)};"


@dataclass
class DropAndCreateCollation(Change):
    """Recreate a collation whose definition changed.

    PostgreSQL has no ALTER for a collation's locale or provider. The differ
    emits this only when nothing uses the collation.
    """

    collation: Collation = field(
        default_factory=lambda: Collation(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COLLATION"

    def __post_init__(self):
        self.object_name = self.collation.qualified_name
        if not self.description:
            self.description = (
                f"Drop and recreate collation {self.collation.qualified_name}"
            )

    def to_sql(self) -> str:
        target = _domain_ident(self.collation.schema, self.collation.name)
        return f"DROP COLLATION {target};\n{_create_collation_sql(self.collation)}"


# =============================================================================
# Sequence Changes
# =============================================================================


SEQUENCE_TYPES = frozenset({"smallint", "integer", "bigint"})


def _sequence_ident(seq: Sequence) -> str:
    return f"{quote_ident(seq.schema)}.{quote_ident(seq.name)}"


def _sequence_options_sql(seq: Sequence) -> str:
    """Every option, spelled out: the catalog's values, not PostgreSQL's defaults.

    Writing all of them keeps CREATE and ALTER independent of the defaults
    (which depend on the type and the sign of INCREMENT) and makes an `AS`
    change unable to move MINVALUE/MAXVALUE behind the declaration's back.
    """
    # The type comes from format_type() on the catalog or the plan database;
    # a sequence can only be one of these, so anything else is refused rather
    # than written into SQL.
    if seq.data_type not in SEQUENCE_TYPES:
        raise ValueError(
            f"sequence {seq.qualified_name}: unexpected type {seq.data_type!r}"
        )
    cycle = "CYCLE" if seq.cycle else "NO CYCLE"
    return (
        f"AS {seq.data_type} INCREMENT BY {int(seq.increment)}"
        f" MINVALUE {int(seq.min_value)} MAXVALUE {int(seq.max_value)}"
        f" START WITH {int(seq.start)} CACHE {int(seq.cache)} {cycle}"
    )


@dataclass
class CreateSequence(Change):
    """Create a sequence. Ownership is a separate SetSequenceOwner change.

    A table's `nextval()` default needs the sequence first, and OWNED BY needs
    the table first, so the two cannot be one statement.
    """

    sequence: Sequence = field(
        default_factory=lambda: Sequence(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SEQUENCE"

    def __post_init__(self):
        self.object_name = self.sequence.qualified_name
        if not self.description:
            self.description = f"Create sequence {self.sequence.qualified_name}"

    def to_sql(self) -> str:
        return f"CREATE SEQUENCE {_sequence_ident(self.sequence)} {_sequence_options_sql(self.sequence)};"


@dataclass
class AlterSequence(Change):
    """Converge a sequence's options in place.

    Never RESTART: `START WITH` only changes the value a future
    `ALTER SEQUENCE ... RESTART` would use, so the current value survives.
    """

    sequence: Sequence = field(
        default_factory=lambda: Sequence(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SEQUENCE"

    def __post_init__(self):
        self.object_name = self.sequence.qualified_name
        if not self.description:
            self.description = (
                f"Alter options of sequence {self.sequence.qualified_name}"
            )

    def to_sql(self) -> str:
        return f"ALTER SEQUENCE {_sequence_ident(self.sequence)} {_sequence_options_sql(self.sequence)};"


@dataclass
class SetSequenceOwner(Change):
    """`ALTER SEQUENCE ... OWNED BY table.column` (or `OWNED BY NONE`)."""

    sequence: Sequence = field(
        default_factory=lambda: Sequence(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SEQUENCE"

    def __post_init__(self):
        self.object_name = self.sequence.qualified_name
        if not self.description:
            owner = self.sequence.owned_by
            target = ".".join(owner) if owner else "nothing"
            self.description = (
                f"Set sequence {self.sequence.qualified_name} owned by {target}"
            )

    def to_sql(self) -> str:
        owner = self.sequence.owned_by
        if owner is None:
            target = "NONE"
        else:
            target = ".".join(quote_ident(part) for part in owner)
        return f"ALTER SEQUENCE {_sequence_ident(self.sequence)} OWNED BY {target};"


@dataclass
class DropSequence(Change):
    """Drop a sequence that is no longer declared.

    No CASCADE: if a column default still calls nextval() on it, PostgreSQL
    refuses and the run fails rather than silently dropping the default. The
    differ orders this after the table changes, so a default being removed in
    the same run is gone first. IF EXISTS because dropping the owning table or
    column already dropped an owned sequence.
    """

    schema: str = "public"
    sequence_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SEQUENCE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.sequence_name}"
        if not self.description:
            self.description = f"Drop sequence {self.object_name}"

    def to_sql(self) -> str:
        return f"DROP SEQUENCE IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.sequence_name)};"


# =============================================================================
# View Changes
# =============================================================================


def view_options_note(view: Union[View, MaterializedView]) -> str:
    """Plan-summary suffix naming the view options a change applies."""
    if not view.reloptions:
        return ""
    return f" WITH ({', '.join(view.reloptions)})"


@dataclass
class CreateView(Change):
    """Create or replace a view."""

    view: View = field(
        default_factory=lambda: View(schema="public", name="", definition="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "VIEW"

    def __post_init__(self):
        self.object_name = self.view.qualified_name
        if not self.description:
            self.description = f"Create view {self.view.name}" + view_options_note(
                self.view
            )

    def to_sql(self) -> str:
        # The WITH clause carries options such as security_invoker. Emitting it
        # here is what makes a fresh install honour the view's declaration.
        return (
            f"CREATE OR REPLACE VIEW {quote_ident(self.view.schema)}.{quote_ident(self.view.name)}"
            f"{self.view.options_clause} AS {self.view.definition}"
        )


@dataclass
class DropView(Change):
    """Drop a view."""

    schema: str = "public"
    view_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "VIEW"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.view_name}"
        if not self.description:
            self.description = f"Drop view {self.view_name}"

    def to_sql(self) -> str:
        return f"DROP VIEW IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.view_name)};"


@dataclass
class DropAndCreateView(Change):
    """Drop and recreate a view (required when columns change order or type)."""

    view: View = field(
        default_factory=lambda: View(schema="public", name="", definition="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "VIEW"
    # DROP wipes the view's ACL, so the desired grants are replayed
    # INSIDE the DO block, after the recreate. A standalone GrantPrivilege cannot
    # be relied on: the retry-until-stable executor defers a recreate whose new
    # definition depends on an object built later in the same run, which lands
    # the grant on the pre-drop view and then throws it away.
    grants: list[Grant] = field(default_factory=list)
    # Views and materialized views that read from this one. DROP refuses while
    # they exist, so they are dropped first and rebuilt after, in this same
    # statement. Attached by compute_diff.
    dependents: list["DependentRelation"] = field(default_factory=list)

    def __post_init__(self):
        self.object_name = self.view.qualified_name
        if not self.description:
            self.description = (
                f"Drop and recreate view {self.view.name}"
                + view_options_note(self.view)
                + grant_replay_note(len(self.grants))
            )

    def to_sql(self) -> str:
        # Wrap in DO block for atomicity - if CREATE fails, DROP is rolled back
        drop_sql = f"DROP VIEW IF EXISTS {quote_ident(self.view.schema)}.{quote_ident(self.view.name)}"
        # The WITH clause rides on the CREATE, inside the same DO block, for the
        # same reason the grants do: a trailing standalone
        # `ALTER VIEW ... SET (...)` would land on the pre-drop view whenever the
        # executor defers this recreate to a later iteration, and the DROP would
        # then throw the option away. Object and security mode are one statement.
        create_sql = (
            f"CREATE VIEW {quote_ident(self.view.schema)}.{quote_ident(self.view.name)}"
            f"{self.view.options_clause} AS {self.view.definition}"
        )
        # Escape single quotes in view definition for EXECUTE
        escaped_create = create_sql.replace("'", "''")
        inner = _drop_dependents_sql(self.dependents)
        inner += f"{drop_sql}; EXECUTE '{escaped_create}';"
        inner += _replay_grants_sql(self.grants)
        inner += _recreate_dependents_sql(self.dependents)
        return do_block(f"BEGIN {inner} END")


@dataclass
class AlterViewOptions(Change):
    """Converge an existing view's `WITH (...)` options without recreating it.

    This is the path for a view whose definition already matches the
    declaration but whose `security_invoker` (say) was never written, so a
    drop-and-recreate would never fire. `ALTER VIEW ... SET (...)` fixes the security mode in place: cheap, and
    it does not disturb dependent views the way a DROP would.
    """

    view: View = field(
        default_factory=lambda: View(schema="public", name="", definition="")
    )
    # Options currently on the object, so the ones no longer declared are RESET.
    current_reloptions: list[str] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "VIEW"

    def __post_init__(self):
        self.object_name = self.view.qualified_name
        if not self.description:
            desired = ", ".join(self.view.reloptions) or "no options"
            self.description = f"Set options on view {self.view.name} to {desired}"

    def _removed_option_names(self) -> list[str]:
        desired_names = {opt.partition("=")[0] for opt in self.view.reloptions}
        removed = {
            opt.partition("=")[0] for opt in self.current_reloptions
        } - desired_names
        return sorted(removed)

    def to_sql(self) -> str:
        target = f"{quote_ident(self.view.schema)}.{quote_ident(self.view.name)}"
        statements: list[str] = []
        removed = self._removed_option_names()
        if removed:
            statements.append(f"ALTER VIEW {target} RESET ({', '.join(removed)});")
        if self.view.reloptions:
            statements.append(
                f"ALTER VIEW {target} SET ({', '.join(self.view.reloptions)});"
            )
        return " ".join(statements)


# =============================================================================
# Materialized View Changes
# =============================================================================


def _execute_sql(statement: str) -> str:
    """`statement` as an `EXECUTE '...';` inside a DO block."""
    escaped = statement.replace("'", "''")
    return f" EXECUTE '{escaped}';"


def _create_matview_sql(matview: MaterializedView) -> str:
    """`CREATE MATERIALIZED VIEW ... AS <query> WITH [NO] DATA` (no trailing `;`)."""
    # pg_get_viewdef ends the query with `;`, and WITH DATA has to follow it.
    query = matview.definition.strip().rstrip(";").rstrip()
    data = "WITH DATA" if matview.with_data else "WITH NO DATA"
    return (
        f"CREATE MATERIALIZED VIEW {quote_ident(matview.schema)}.{quote_ident(matview.name)}"
        f"{matview.options_clause} AS {query} {data}"
    )


def _rebuild_indexes_sql(indexes: list[Index]) -> str:
    """Rebuild a recreated materialized view's indexes inside its DO block.

    DROP takes the indexes with it, and the diff, computed against the snapshot
    taken before the run, still sees them, so nothing else would bring them
    back. They are built in the same statement, not CONCURRENTLY: the object is
    new, nobody can be reading it yet, and the index can never be missing.
    """
    return "".join(_execute_sql(index.definition) for index in indexes)


@dataclass
class DependentRelation:
    """A view or materialized view that reads from a relation being recreated.

    `relation` is its declared definition, or None when it is no longer
    declared: then it is only dropped (its own DropView or
    DropMaterializedView finds it gone).
    """

    schema: str
    name: str
    kind: str  # 'v' (view) or 'm' (materialized view)
    relation: Optional[Union[View, MaterializedView]] = None
    grants: list[Grant] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def rebuilds_data(self) -> bool:
        return isinstance(self.relation, MaterializedView)

    def drop_sql(self) -> str:
        keyword = "MATERIALIZED VIEW" if self.kind == "m" else "VIEW"
        return f"DROP {keyword} IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.name)};"

    def create_sql(self) -> str:
        relation = self.relation
        if relation is None:
            return ""
        if isinstance(relation, MaterializedView):
            create = _create_matview_sql(relation)
        else:
            create = (
                f"CREATE VIEW {quote_ident(relation.schema)}.{quote_ident(relation.name)}"
                f"{relation.options_clause} AS {relation.definition}"
            )
        return (
            _execute_sql(create)
            + _replay_grants_sql(self.grants)
            + _rebuild_indexes_sql(self.indexes)
        )


def _drop_dependents_sql(dependents: list[DependentRelation]) -> str:
    """Drop dependents, innermost first. `dependents` is in creation order."""
    return "".join(f"{d.drop_sql()} " for d in reversed(dependents))


def _recreate_dependents_sql(dependents: list[DependentRelation]) -> str:
    """Rebuild the still-declared dependents, each after what it reads from."""
    return "".join(d.create_sql() for d in dependents)


def dependents_note(dependents: list[DependentRelation]) -> str:
    """Plan-summary suffix for a recreate that carries its dependents along."""
    rebuilt = [d.name for d in dependents if d.relation is not None]
    dropped = [d.name for d in dependents if d.relation is None]
    note = ""
    if rebuilt:
        note += f" (recreating dependent {', '.join(rebuilt)})"
    if dropped:
        note += f" (dropping dependent {', '.join(dropped)})"
    return note


def index_rebuild_note(count: int) -> str:
    """Plan-summary suffix for a matview recreate that rebuilds `count` indexes."""
    if not count:
        return ""
    return f" (rebuilding {count} index{'es' if count > 1 else ''})"


@dataclass
class CreateMaterializedView(Change):
    """Create a materialized view, populated or not as declared."""

    matview: MaterializedView = field(
        default_factory=lambda: MaterializedView(
            schema="public", name="", definition=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "MATERIALIZED VIEW"

    def __post_init__(self):
        self.object_name = self.matview.qualified_name
        if not self.description:
            data = "" if self.matview.with_data else " WITH NO DATA"
            self.description = (
                f"Create materialized view {self.matview.name}"
                + view_options_note(self.matview)
                + data
            )

    def to_sql(self) -> str:
        return _create_matview_sql(self.matview) + ";"


@dataclass
class DropMaterializedView(Change):
    """Drop a materialized view. Its data is derived, so nothing is lost."""

    schema: str = "public"
    matview_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "MATERIALIZED VIEW"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.matview_name}"
        if not self.description:
            self.description = f"Drop materialized view {self.matview_name}"

    def to_sql(self) -> str:
        return f"DROP MATERIALIZED VIEW IF EXISTS {quote_ident(self.schema)}.{quote_ident(self.matview_name)};"


@dataclass
class DropAndCreateMaterializedView(Change):
    """Drop and recreate a materialized view whose query changed.

    There is no `CREATE OR REPLACE MATERIALIZED VIEW`. The recreate re-runs the
    query to repopulate the object (unless it is declared `WITH NO DATA`), and
    the table lock it holds until commit blocks every reader of the
    materialized view for that long, so it is OFFLINE.

    Like DropAndCreateView, everything the DROP throws away is rebuilt inside
    the same DO block: the declared grants, the declared indexes, and the
    views and materialized views that read from this one. No execution order
    can leave the object without them.
    """

    matview: MaterializedView = field(
        default_factory=lambda: MaterializedView(
            schema="public", name="", definition=""
        )
    )
    category: ChangeCategory = ChangeCategory.OFFLINE
    description: str = ""
    object_type: str = "MATERIALIZED VIEW"
    grants: list[Grant] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)
    dependents: list[DependentRelation] = field(default_factory=list)

    def __post_init__(self):
        self.object_name = self.matview.qualified_name
        if not self.description:
            self.description = (
                f"Drop and recreate materialized view {self.matview.name}"
                + view_options_note(self.matview)
                + (
                    " (rebuilds its data)"
                    if self.matview.with_data
                    else " WITH NO DATA"
                )
            )

    def to_sql(self) -> str:
        target = f"{quote_ident(self.matview.schema)}.{quote_ident(self.matview.name)}"
        inner = _drop_dependents_sql(self.dependents)
        inner += f"DROP MATERIALIZED VIEW IF EXISTS {target};"
        inner += _execute_sql(_create_matview_sql(self.matview))
        inner += _replay_grants_sql(self.grants)
        inner += _rebuild_indexes_sql(self.indexes)
        inner += _recreate_dependents_sql(self.dependents)
        return do_block(f"BEGIN {inner} END")


@dataclass
class AlterMaterializedViewOptions(Change):
    """Converge a materialized view's storage parameters in place."""

    matview: MaterializedView = field(
        default_factory=lambda: MaterializedView(
            schema="public", name="", definition=""
        )
    )
    current_reloptions: list[str] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "MATERIALIZED VIEW"

    def __post_init__(self):
        self.object_name = self.matview.qualified_name
        if not self.description:
            desired = ", ".join(self.matview.reloptions) or "no options"
            self.description = (
                f"Set storage options on materialized view {self.matview.name} "
                f"to {desired}"
            )

    def to_sql(self) -> str:
        target = f"{quote_ident(self.matview.schema)}.{quote_ident(self.matview.name)}"
        desired_names = {opt.partition("=")[0] for opt in self.matview.reloptions}
        removed = sorted(
            {opt.partition("=")[0] for opt in self.current_reloptions} - desired_names
        )
        statements: list[str] = []
        if removed:
            statements.append(
                f"ALTER MATERIALIZED VIEW {target} RESET ({', '.join(removed)});"
            )
        if self.matview.reloptions:
            statements.append(
                f"ALTER MATERIALIZED VIEW {target} SET ({', '.join(self.matview.reloptions)});"
            )
        return " ".join(statements)


# =============================================================================
# Grant/Revoke Changes
# =============================================================================


@dataclass
class GrantPrivilege(Change):
    """Grant a privilege."""

    grant: Grant = field(
        default_factory=lambda: Grant(
            object_type="", object_name="", schema="public", privilege="", grantee=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "GRANT"

    def __post_init__(self):
        self.object_name = self.grant.key
        if not self.description:
            self.description = f"Grant {self.grant.privilege} on {self.grant.object_name} to {self.grant.grantee}"

    def to_sql(self) -> str:
        if self.grant.object_type == "SCHEMA":
            return f"GRANT {self.grant.privilege} ON SCHEMA {quote_ident(self.grant.object_name)} TO {role_ident(self.grant.grantee)};"
        elif self.grant.column:
            return f"GRANT {self.grant.privilege} ({quote_ident(self.grant.column)}) ON {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} TO {role_ident(self.grant.grantee)};"
        elif self.grant.object_type in ("FUNCTION", "PROCEDURE"):
            return f"GRANT {self.grant.privilege} ON {self.grant.object_type} {self.grant.object_sql} TO {role_ident(self.grant.grantee)};"
        elif self.grant.object_type == "SEQUENCE":
            return f"GRANT {self.grant.privilege} ON SEQUENCE {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} TO {role_ident(self.grant.grantee)};"
        else:
            return f"GRANT {self.grant.privilege} ON {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} TO {role_ident(self.grant.grantee)};"


@dataclass
class RevokePrivilege(Change):
    """Revoke a privilege."""

    grant: Grant = field(
        default_factory=lambda: Grant(
            object_type="", object_name="", schema="public", privilege="", grantee=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "REVOKE"

    def __post_init__(self):
        self.object_name = self.grant.key
        if not self.description:
            self.description = f"Revoke {self.grant.privilege} on {self.grant.object_name} from {self.grant.grantee}"

    def to_sql(self) -> str:
        if self.grant.object_type == "SCHEMA":
            return f"REVOKE {self.grant.privilege} ON SCHEMA {quote_ident(self.grant.object_name)} FROM {role_ident(self.grant.grantee)};"
        elif self.grant.column:
            # Wrap in DO block: column may have been dropped in this migration batch
            revoke_sql = f"REVOKE {self.grant.privilege} ({quote_ident(self.grant.column)}) ON {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} FROM {role_ident(self.grant.grantee)}"
            return do_block(
                f"BEGIN {revoke_sql}; EXCEPTION WHEN undefined_column THEN NULL; END"
            )
        elif self.grant.object_type in ("FUNCTION", "PROCEDURE"):
            # Wrap in DO block: function may have been dropped (CASCADE removes
            # grants too), or recreated this run as the other kind of routine
            # (wrong_object_type), which took its old ACL with it.
            revoke_sql = f"REVOKE {self.grant.privilege} ON {self.grant.object_type} {self.grant.object_sql} FROM {role_ident(self.grant.grantee)}"
            return do_block(
                f"BEGIN {revoke_sql}; EXCEPTION WHEN undefined_function"
                " OR wrong_object_type THEN NULL; END"
            )
        elif self.grant.object_type == "SEQUENCE":
            # Wrap in DO block: the sequence may have been dropped in this
            # batch, with its owning table or by DropSequence.
            revoke_sql = f"REVOKE {self.grant.privilege} ON SEQUENCE {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} FROM {role_ident(self.grant.grantee)}"
            return do_block(
                f"BEGIN {revoke_sql}; EXCEPTION WHEN undefined_table THEN NULL; END"
            )
        else:
            return f"REVOKE {self.grant.privilege} ON {quote_ident(self.grant.schema)}.{quote_ident(self.grant.object_name)} FROM {role_ident(self.grant.grantee)};"


@dataclass
class RevokePublicExecute(Change):
    """Revoke PUBLIC EXECUTE on a SECDEF function for the drift case.

    Used only when a locked function already exists and is NOT being created or
    recreated this run (so PostgreSQL cannot reset the ACL afterward). For the
    create/recreate case the REVOKE is appended atomically to the function change
    instead (see CreateFunction/DropAndCreateFunction.revoke_public_execute).
    A plain unwrapped REVOKE is correct here because the function is known to
    exist; no DO/EXCEPTION guard is needed.
    """

    schema: str = "public"
    name: str = ""
    args: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "REVOKE"
    keyword: str = "FUNCTION"  # PROCEDURE for a SECURITY DEFINER procedure

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.name}({self.args})"
        if not self.description:
            self.description = f"Revoke PUBLIC EXECUTE on {self.object_name}"

    def to_sql(self) -> str:
        revoke = _revoke_public_execute_sql(
            self.schema, self.name, self.args, self.keyword
        )
        return f"{revoke};"


# =============================================================================
# Schema Changes
# =============================================================================


@dataclass
class CreateSchema(Change):
    """Create a schema."""

    schema_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SCHEMA"

    def __post_init__(self):
        self.object_name = self.schema_name
        if not self.description:
            self.description = f"Create schema {self.schema_name}"

    def to_sql(self) -> str:
        return f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.schema_name)};"


# =============================================================================
# Role Changes
# =============================================================================


@dataclass
class CreateRole(Change):
    """Create a database role."""

    role: Role = field(default_factory=lambda: Role(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "ROLE"

    def __post_init__(self):
        self.object_name = self.role.name
        if not self.description:
            self.description = f"Create role {self.role.name}"

    def to_sql(self) -> str:
        return self._sql(mask_password=False)

    def to_display_sql(self) -> str:
        return self._sql(mask_password=True)

    def _sql(self, mask_password: bool) -> str:
        options = self.role.options_for(mask_password)
        options_str = " ".join(options) if options else ""
        if options_str:
            return f"CREATE ROLE {quote_ident(self.role.name)} WITH {options_str};"
        return f"CREATE ROLE {quote_ident(self.role.name)};"


@dataclass
class AlterRole(Change):
    """Alter a database role's attributes."""

    role: Role = field(default_factory=lambda: Role(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "ROLE"

    def __post_init__(self):
        self.object_name = self.role.name
        if not self.description:
            self.description = f"Alter role {self.role.name}"

    def to_sql(self) -> str:
        return self._sql(mask_password=False)

    def to_display_sql(self) -> str:
        return self._sql(mask_password=True)

    def _sql(self, mask_password: bool) -> str:
        options = self.role.options_for(mask_password)
        options_str = " ".join(options) if options else ""
        if options_str:
            return f"ALTER ROLE {quote_ident(self.role.name)} WITH {options_str};"
        return "-- No changes for role " + " ".join(self.role.name.splitlines())


@dataclass
class DropRole(Change):
    """Drop a database role."""

    role_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "ROLE"

    def __post_init__(self):
        self.object_name = self.role_name
        if not self.description:
            self.description = f"Drop role {self.role_name}"

    def to_sql(self) -> str:
        return f"DROP ROLE IF EXISTS {quote_ident(self.role_name)};"


def _default_privileges_prefix(entry: DefaultPrivilege) -> str:
    """`ALTER DEFAULT PRIVILEGES FOR ROLE r [IN SCHEMA s]`, every name quoted.

    The target role is always written out, so the statement does not depend on
    which role the migrator connects as.
    """
    sql = f"ALTER DEFAULT PRIVILEGES FOR ROLE {role_ident(entry.target_role)}"
    if entry.schema is not None:
        sql += f" IN SCHEMA {quote_ident(entry.schema)}"
    return sql


def _default_privileges_scope(entry: DefaultPrivilege) -> str:
    where = f"in schema {entry.schema}" if entry.schema is not None else "everywhere"
    return f"{entry.object_type.lower()} of {entry.target_role} {where}"


@dataclass
class GrantDefaultPrivilege(Change):
    """ALTER DEFAULT PRIVILEGES ... GRANT: future objects get the privilege."""

    entry: DefaultPrivilege = field(
        default_factory=lambda: DefaultPrivilege(
            target_role="", schema=None, object_type="TABLES", privilege="", grantee=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DEFAULT_PRIVILEGE"

    def __post_init__(self):
        self.object_name = self.entry.key
        if not self.description:
            self.description = (
                f"Grant {self.entry.privilege} on future "
                f"{_default_privileges_scope(self.entry)} to {self.entry.grantee}"
            )

    def to_sql(self) -> str:
        option = " WITH GRANT OPTION" if self.entry.grantable else ""
        return (
            f"{_default_privileges_prefix(self.entry)} GRANT {self.entry.privilege} "
            f"ON {self.entry.object_type} TO {role_ident(self.entry.grantee)}{option};"
        )


@dataclass
class RevokeDefaultPrivilege(Change):
    """ALTER DEFAULT PRIVILEGES ... REVOKE. Objects that exist keep their ACL."""

    entry: DefaultPrivilege = field(
        default_factory=lambda: DefaultPrivilege(
            target_role="", schema=None, object_type="TABLES", privilege="", grantee=""
        )
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "DEFAULT_PRIVILEGE"

    def __post_init__(self):
        self.object_name = self.entry.key
        if not self.description:
            self.description = (
                f"Revoke {self.entry.privilege} on future "
                f"{_default_privileges_scope(self.entry)} from {self.entry.grantee}"
            )

    def to_sql(self) -> str:
        return (
            f"{_default_privileges_prefix(self.entry)} REVOKE {self.entry.privilege} "
            f"ON {self.entry.object_type} FROM {role_ident(self.entry.grantee)};"
        )


@dataclass
class GrantRoleMembership(Change):
    """Grant a role to another role (GRANT role TO member)."""

    membership: RoleMembership = field(
        default_factory=lambda: RoleMembership(role="", member="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "ROLE_MEMBERSHIP"

    def __post_init__(self):
        self.object_name = f"{self.membership.role}:{self.membership.member}"
        if not self.description:
            self.description = (
                f"Grant role {self.membership.role} to {self.membership.member}"
            )

    def to_sql(self) -> str:
        admin_option = " WITH ADMIN OPTION" if self.membership.admin_option else ""
        return f"GRANT {quote_ident(self.membership.role)} TO {quote_ident(self.membership.member)}{admin_option};"


@dataclass
class RevokeRoleMembership(Change):
    """Revoke a role from another role (REVOKE role FROM member)."""

    membership: RoleMembership = field(
        default_factory=lambda: RoleMembership(role="", member="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "ROLE_MEMBERSHIP"

    def __post_init__(self):
        self.object_name = f"{self.membership.role}:{self.membership.member}"
        if not self.description:
            self.description = (
                f"Revoke role {self.membership.role} from {self.membership.member}"
            )

    def to_sql(self) -> str:
        return f"REVOKE {quote_ident(self.membership.role)} FROM {quote_ident(self.membership.member)};"


# =============================================================================
# Comment Changes
# =============================================================================


@dataclass
class SetComment(Change):
    """Set, change or remove (`IS NULL`) the comment on an object.

    The executor runs these in a phase of their own, after the DDL and the
    concurrent index builds, so the object exists, in its final form, whatever
    order the retry loop chose and whether or not it was dropped and recreated.
    """

    comment: Comment = field(
        default_factory=lambda: Comment(object_type="", schema="", name="", text="")
    )
    # True writes `IS NULL`: the declaration was removed.
    remove: bool = False
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "COMMENT"

    def __post_init__(self):
        self.object_name = self.comment.key
        if not self.description:
            kind, _, target = self.comment.key.partition(":")
            verb = "Remove comment on" if self.remove else "Set comment on"
            self.description = f"{verb} {kind.lower()} {target}"

    def to_sql(self) -> str:
        text = "NULL" if self.remove else quote_literal(self.comment.text)
        return f"COMMENT ON {self.comment.target_sql} IS {text};"


# =============================================================================
# Rewrite Rule Changes
# =============================================================================

# ev_enabled / evtenabled -> the ENABLE/DISABLE clause that sets it.
_ENABLED_CLAUSE = {
    "O": "ENABLE",
    "D": "DISABLE",
    "R": "ENABLE REPLICA",
    "A": "ENABLE ALWAYS",
}


def _rule_state_sql(rule: Rule) -> str:
    return (
        f"ALTER TABLE {quote_ident(rule.schema)}.{quote_ident(rule.table)} "
        f"{_ENABLED_CLAUSE[rule.enabled]} RULE {quote_ident(rule.name)};"
    )


@dataclass
class CreateRule(Change):
    """Create or replace a rewrite rule on a table.

    `CREATE OR REPLACE RULE` keeps a replaced rule's enabled state, so the
    state is written whenever it is not the default or has to change.
    """

    rule: Rule = field(
        default_factory=lambda: Rule(schema="public", table="", name="", definition="")
    )
    set_enabled: bool = False
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "RULE"

    def __post_init__(self):
        self.object_name = self.rule.qualified_key
        if not self.description:
            self.description = f"Create rule {self.rule.name} on {self.rule.table}"

    def to_sql(self) -> str:
        definition = self.rule.definition.strip().rstrip(";")
        if definition.upper().startswith("CREATE RULE"):
            definition = "CREATE OR REPLACE RULE" + definition[len("CREATE RULE") :]
        sql = definition + ";"
        if self.set_enabled or self.rule.enabled != "O":
            sql += "\n" + _rule_state_sql(self.rule)
        return sql


@dataclass
class AlterRuleEnabled(Change):
    """Enable or disable a rewrite rule (ENABLE / DISABLE / REPLICA / ALWAYS)."""

    rule: Rule = field(
        default_factory=lambda: Rule(schema="public", table="", name="", definition="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "RULE"

    def __post_init__(self):
        self.object_name = self.rule.qualified_key
        if not self.description:
            clause = _ENABLED_CLAUSE[self.rule.enabled].capitalize()
            self.description = f"{clause} rule {self.rule.name} on {self.rule.table}"

    def to_sql(self) -> str:
        return _rule_state_sql(self.rule)


@dataclass
class DropRule(Change):
    """Drop a rewrite rule."""

    schema: str = "public"
    table: str = ""
    rule_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "RULE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table}.{self.rule_name}"
        if not self.description:
            self.description = f"Drop rule {self.rule_name} on {self.table}"

    def to_sql(self) -> str:
        return f"DROP RULE IF EXISTS {quote_ident(self.rule_name)} ON {quote_ident(self.schema)}.{quote_ident(self.table)};"


# =============================================================================
# Event Trigger Changes
# =============================================================================


def _event_trigger_state_sql(name: str, enabled: str) -> str:
    return f"ALTER EVENT TRIGGER {quote_ident(name)} {_ENABLED_CLAUSE[enabled]};"


@dataclass
class CreateEventTrigger(Change):
    """Create an event trigger; with `replace`, drop the old one first.

    There is no `CREATE OR REPLACE EVENT TRIGGER`. Creating one needs a
    superuser.
    """

    trigger: EventTrigger = field(
        default_factory=lambda: EventTrigger(
            name="", event="", function_schema="", function_name=""
        )
    )
    replace: bool = False
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EVENT TRIGGER"

    def __post_init__(self):
        self.object_name = self.trigger.name
        if not self.description:
            verb = "Drop and recreate" if self.replace else "Create"
            self.description = f"{verb} event trigger {self.trigger.name}"

    def to_sql(self) -> str:
        t = self.trigger
        sql = ""
        if self.replace:
            sql += f"DROP EVENT TRIGGER IF EXISTS {quote_ident(t.name)};\n"
        sql += f"CREATE EVENT TRIGGER {quote_ident(t.name)} ON {quote_ident(t.event)}"
        if t.tags:
            sql += f" WHEN TAG IN ({', '.join(quote_literal(tag) for tag in t.tags)})"
        sql += f" EXECUTE FUNCTION {quote_ident(t.function_schema)}.{quote_ident(t.function_name)}();"
        if t.enabled != "O":
            sql += "\n" + _event_trigger_state_sql(t.name, t.enabled)
        return sql


@dataclass
class AlterEventTriggerEnabled(Change):
    """Enable or disable an event trigger."""

    name: str = ""
    enabled: str = "O"
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EVENT TRIGGER"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            clause = _ENABLED_CLAUSE[self.enabled].capitalize()
            self.description = f"{clause} event trigger {self.name}"

    def to_sql(self) -> str:
        return _event_trigger_state_sql(self.name, self.enabled)


@dataclass
class DropEventTrigger(Change):
    """Drop an event trigger."""

    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EVENT TRIGGER"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = f"Drop event trigger {self.name}"

    def to_sql(self) -> str:
        return f"DROP EVENT TRIGGER IF EXISTS {quote_ident(self.name)};"


# =============================================================================
# Publication Changes
# =============================================================================


def _publication_table_sql(table: PublicationTable) -> str:
    """One published table: name, column list and row filter."""
    sql = f"{quote_ident(table.schema)}.{quote_ident(table.name)}"
    if table.columns:
        sql += f" ({', '.join(quote_ident(c) for c in table.columns)})"
    if table.row_filter is not None:
        sql += f" WHERE ({table.row_filter})"
    return sql


def _publication_objects_sql(
    tables: list[PublicationTable], schemas: tuple[str, ...] | list[str] = ()
) -> str:
    """`TABLE a, b (c) WHERE (...), TABLES IN SCHEMA s, t`.

    One TABLE keyword for the whole list, which PostgreSQL 14 also parses.
    """
    parts: list[str] = []
    if tables:
        parts.append("TABLE " + ", ".join(_publication_table_sql(t) for t in tables))
    if schemas:
        parts.append("TABLES IN SCHEMA " + ", ".join(quote_ident(s) for s in schemas))
    return ", ".join(parts)


def _publication_options_sql(publication: Publication) -> str:
    options = [
        f"publish = {quote_literal(', '.join(publication.publish))}",
        f"publish_via_partition_root = {'true' if publication.via_root else 'false'}",
    ]
    if publication.generated_columns is not None:
        options.append(f"publish_generated_columns = {publication.generated_columns}")
    return f"({', '.join(options)})"


def _create_publication_sql(publication: Publication) -> str:
    sql = f"CREATE PUBLICATION {quote_ident(publication.name)}"
    if publication.all_tables:
        sql += " FOR ALL TABLES"
    else:
        objects = _publication_objects_sql(
            list(publication.tables.values()), publication.schemas
        )
        if objects:
            sql += f" FOR {objects}"
    return sql + f" WITH {_publication_options_sql(publication)};"


def _drop_publication_table_body(publication: str, table: PublicationTable) -> str:
    """PL/pgSQL that drops `table` from `publication` if it is still a member.

    A column drop with CASCADE or a table drop earlier in the same run may
    already have removed it, and ALTER PUBLICATION ... DROP TABLE has no
    IF EXISTS.
    """
    relation = f"{quote_ident(table.schema)}.{quote_ident(table.name)}"
    statement = f"ALTER PUBLICATION {quote_ident(publication)} DROP TABLE {relation}"
    return (
        "IF EXISTS (SELECT 1 FROM pg_catalog.pg_publication_rel pr "
        "JOIN pg_catalog.pg_publication p ON p.oid = pr.prpubid "
        f"WHERE p.pubname = {quote_literal(publication)} "
        f"AND pr.prrelid = pg_catalog.to_regclass({quote_literal(relation)})) "
        f"THEN EXECUTE {quote_literal(statement)}; END IF;"
    )


@dataclass
class CreatePublication(Change):
    """Create a publication with its tables, schemas and options."""

    publication: Publication = field(default_factory=lambda: Publication(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.publication.name
        if not self.description:
            self.description = f"Create publication {self.publication.name}"

    def to_sql(self) -> str:
        return _create_publication_sql(self.publication)


@dataclass
class DropAndCreatePublication(Change):
    """Drop and recreate a publication: FOR ALL TABLES cannot be altered."""

    publication: Publication = field(default_factory=lambda: Publication(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.publication.name
        if not self.description:
            self.description = f"Drop and recreate publication {self.publication.name}"

    def to_sql(self) -> str:
        return (
            f"DROP PUBLICATION IF EXISTS {quote_ident(self.publication.name)};\n"
            + _create_publication_sql(self.publication)
        )


@dataclass
class DropPublication(Change):
    """Drop a publication."""

    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = f"Drop publication {self.name}"

    def to_sql(self) -> str:
        return f"DROP PUBLICATION IF EXISTS {quote_ident(self.name)};"


@dataclass
class AlterPublicationOptions(Change):
    """`ALTER PUBLICATION ... SET (publish = ..., ...)`."""

    publication: Publication = field(default_factory=lambda: Publication(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.publication.name
        if not self.description:
            self.description = f"Set options of publication {self.publication.name}"

    def to_sql(self) -> str:
        return (
            f"ALTER PUBLICATION {quote_ident(self.publication.name)} "
            f"SET {_publication_options_sql(self.publication)};"
        )


@dataclass
class AddPublicationObjects(Change):
    """`ALTER PUBLICATION ... ADD TABLE ..., TABLES IN SCHEMA ...`."""

    name: str = ""
    tables: list[PublicationTable] = field(default_factory=list)
    schemas: list[str] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            added = [t.qualified_name for t in self.tables] + [
                f"schema {s}" for s in self.schemas
            ]
            self.description = f"Add {', '.join(added)} to publication {self.name}"

    def to_sql(self) -> str:
        objects = _publication_objects_sql(self.tables, self.schemas)
        return f"ALTER PUBLICATION {quote_ident(self.name)} ADD {objects};"


@dataclass
class DropPublicationObjects(Change):
    """Remove tables and schemas from a publication.

    Each table is dropped only while it is still a member (see
    `_drop_publication_table_body`).
    """

    name: str = ""
    tables: list[PublicationTable] = field(default_factory=list)
    schemas: list[str] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            removed = [t.qualified_name for t in self.tables] + [
                f"schema {s}" for s in self.schemas
            ]
            self.description = (
                f"Remove {', '.join(removed)} from publication {self.name}"
            )

    def to_sql(self) -> str:
        statements: list[str] = []
        if self.tables:
            body = " ".join(
                _drop_publication_table_body(self.name, t) for t in self.tables
            )
            statements.append(do_block(f"BEGIN {body} END"))
        if self.schemas:
            schemas = ", ".join(quote_ident(s) for s in self.schemas)
            statements.append(
                f"ALTER PUBLICATION {quote_ident(self.name)} DROP TABLES IN SCHEMA {schemas};"
            )
        return "\n".join(statements)


@dataclass
class ReplacePublicationTable(Change):
    """Re-add a published table with a new column list or row filter.

    There is no ALTER for a member's column list or filter, so the table is
    dropped from the publication and added back in one statement.
    """

    name: str = ""
    table: PublicationTable = field(
        default_factory=lambda: PublicationTable(schema="public", name="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "PUBLICATION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = (
                f"Change columns or row filter of {self.table.qualified_name} "
                f"in publication {self.name}"
            )

    def to_sql(self) -> str:
        add = (
            f"ALTER PUBLICATION {quote_ident(self.name)} ADD "
            f"{_publication_objects_sql([self.table])}"
        )
        body = _drop_publication_table_body(self.name, self.table)
        return do_block(f"BEGIN {body} EXECUTE {quote_literal(add)}; END")


# =============================================================================
# Extension Changes
# =============================================================================


@dataclass
class CreateExtension(Change):
    """`CREATE EXTENSION`, with the schema and version the declaration names."""

    extension: Extension = field(default_factory=lambda: Extension(name=""))
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EXTENSION"

    def __post_init__(self):
        self.object_name = self.extension.name
        if not self.description:
            self.description = f"Create extension {self.extension.name}"

    def to_sql(self) -> str:
        ext = self.extension
        sql = f"CREATE EXTENSION IF NOT EXISTS {quote_ident(ext.name)}"
        if ext.schema is not None:
            sql += f" WITH SCHEMA {quote_ident(ext.schema)}"
        if ext.version is not None:
            sql += f" VERSION {quote_literal(ext.version)}"
        if ext.cascade:
            sql += " CASCADE"
        return sql + ";"


@dataclass
class AlterExtensionUpdate(Change):
    """`ALTER EXTENSION ... UPDATE TO` the declared version."""

    name: str = ""
    version: str = ""
    current_version: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EXTENSION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = (
                f"Update extension {self.name} from {self.current_version} "
                f"to {self.version}"
            )

    def to_sql(self) -> str:
        return (
            f"ALTER EXTENSION {quote_ident(self.name)} "
            f"UPDATE TO {quote_literal(self.version)};"
        )


@dataclass
class AlterExtensionSchema(Change):
    """`ALTER EXTENSION ... SET SCHEMA` (relocatable extensions only)."""

    name: str = ""
    schema: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EXTENSION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = f"Move extension {self.name} to schema {self.schema}"

    def to_sql(self) -> str:
        return (
            f"ALTER EXTENSION {quote_ident(self.name)} "
            f"SET SCHEMA {quote_ident(self.schema)};"
        )


@dataclass
class DropExtension(Change):
    """Drop an extension a `-- @drop extension:NAME` directive consents to.

    Never CASCADE: while a column, function or view still uses one of the
    extension's objects, PostgreSQL refuses, and the run fails rather than
    dropping data along with it.
    """

    name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "EXTENSION"

    def __post_init__(self):
        self.object_name = self.name
        if not self.description:
            self.description = (
                f"Drop extension {self.name} (@drop extension:{self.name})"
            )

    def to_sql(self) -> str:
        return f"DROP EXTENSION IF EXISTS {quote_ident(self.name)};"


# =============================================================================
# Foreign Data Changes
# =============================================================================


def _server_ident(name: str) -> str:
    return quote_ident(name)


@dataclass
class CreateForeignServer(Change):
    """`CREATE SERVER ... FOREIGN DATA WRAPPER ... OPTIONS (...)`."""

    server: ForeignServer = field(
        default_factory=lambda: ForeignServer(name="", fdw="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SERVER"

    def __post_init__(self):
        self.object_name = self.server.name
        if not self.description:
            self.description = (
                f"Create server {self.server.name} (foreign data wrapper "
                f"{self.server.fdw})"
            )

    def to_sql(self) -> str:
        srv = self.server
        sql = f"CREATE SERVER {_server_ident(srv.name)}"
        if srv.type is not None:
            sql += f" TYPE {quote_literal(srv.type)}"
        if srv.version is not None:
            sql += f" VERSION {quote_literal(srv.version)}"
        sql += f" FOREIGN DATA WRAPPER {quote_ident(srv.fdw)}"
        options = fdw_options_sql(srv.options)
        if options:
            sql += f" {options}"
        return sql + ";"


@dataclass
class AlterForeignServer(Change):
    """`ALTER SERVER ... [VERSION ...] OPTIONS (ADD/SET/DROP ...)`."""

    server: ForeignServer = field(
        default_factory=lambda: ForeignServer(name="", fdw="")
    )
    current_options: dict[str, str] = field(default_factory=dict)
    # Write VERSION: the declaration names one the server does not have.
    set_version: bool = False
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "SERVER"

    def __post_init__(self):
        self.object_name = self.server.name
        if not self.description:
            what: list[str] = []
            if self.set_version:
                what.append("version")
            if alter_fdw_options_sql(self.server.options, self.current_options):
                what.append("options")
            self.description = (
                f"Alter {' and '.join(what) or 'nothing'} of server {self.server.name}"
            )

    def to_sql(self) -> str:
        sql = f"ALTER SERVER {_server_ident(self.server.name)}"
        if self.set_version and self.server.version is not None:
            sql += f" VERSION {quote_literal(self.server.version)}"
        options = alter_fdw_options_sql(self.server.options, self.current_options)
        if options:
            sql += f" {options}"
        return sql + ";"


@dataclass
class CreateUserMapping(Change):
    """`CREATE USER MAPPING FOR ... SERVER ... OPTIONS (...)`.

    Options may hold a password, so the display SQL masks every value but
    the ones listed in USER_MAPPING_PUBLIC_OPTIONS.
    """

    mapping: UserMapping = field(
        default_factory=lambda: UserMapping(server="", user="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "USER MAPPING"

    def __post_init__(self):
        self.object_name = f"{self.mapping.user}@{self.mapping.server}"
        if not self.description:
            self.description = f"Create user mapping for {self.mapping.label}"

    def to_sql(self) -> str:
        return self._sql(mask=False)

    def to_display_sql(self) -> str:
        return self._sql(mask=True)

    def _sql(self, mask: bool) -> str:
        m = self.mapping
        sql = f"CREATE USER MAPPING FOR {m.user_sql} SERVER {_server_ident(m.server)}"
        masked = m.secret_options if mask else frozenset()
        options = fdw_options_sql(m.options or {}, masked)
        if options:
            sql += f" {options}"
        return sql + ";"


@dataclass
class AlterUserMapping(Change):
    """`ALTER USER MAPPING ... OPTIONS (ADD/SET/DROP ...)`, values masked for display."""

    mapping: UserMapping = field(
        default_factory=lambda: UserMapping(server="", user="")
    )
    current_options: dict[str, str] = field(default_factory=dict)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "USER MAPPING"

    def __post_init__(self):
        self.object_name = f"{self.mapping.user}@{self.mapping.server}"
        if not self.description:
            names = option_names_changed(
                self.mapping.options or {}, self.current_options
            )
            self.description = (
                f"Alter options ({', '.join(names)}) of user mapping for "
                f"{self.mapping.label}"
            )

    def to_sql(self) -> str:
        return self._sql(mask=False)

    def to_display_sql(self) -> str:
        return self._sql(mask=True)

    def _sql(self, mask: bool) -> str:
        m = self.mapping
        masked = m.secret_options if mask else frozenset()
        options = alter_fdw_options_sql(m.options or {}, self.current_options, masked)
        return (
            f"ALTER USER MAPPING FOR {m.user_sql} SERVER {_server_ident(m.server)} "
            f"{options};"
        )


@dataclass
class DropUserMapping(Change):
    """Drop a managed role's user mapping that is no longer declared."""

    mapping: UserMapping = field(
        default_factory=lambda: UserMapping(server="", user="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "USER MAPPING"

    def __post_init__(self):
        self.object_name = f"{self.mapping.user}@{self.mapping.server}"
        if not self.description:
            self.description = f"Drop user mapping for {self.mapping.label}"

    def to_sql(self) -> str:
        m = self.mapping
        return f"DROP USER MAPPING IF EXISTS FOR {m.user_sql} SERVER {_server_ident(m.server)};"


def _foreign_table_ident(schema: str, name: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(name)}"


def foreign_column_sql(column: Column, options: Optional[dict[str, str]]) -> str:
    """One column of a CREATE FOREIGN TABLE or an ADD COLUMN."""
    sql = f"{quote_ident(column.name)} {column.type}"
    rendered = fdw_options_sql(options or {})
    if rendered:
        sql += f" {rendered}"
    if not column.nullable:
        sql += " NOT NULL"
    if column.generated:
        sql += f" GENERATED ALWAYS AS ({column.generated}) STORED"
    elif column.default:
        sql += f" DEFAULT {column.default}"
    return sql


def _create_foreign_table_sql(table: ForeignTable) -> str:
    """`CREATE FOREIGN TABLE ... SERVER ... OPTIONS (...)` (no trailing `;`)."""
    columns = ",\n".join(
        f"    {foreign_column_sql(col, table.column_options.get(col.name))}"
        for col in table.columns.values()
    )
    sql = (
        f"CREATE FOREIGN TABLE {_foreign_table_ident(table.schema, table.name)} "
        f"(\n{columns}\n) SERVER {_server_ident(table.server)}"
    )
    options = fdw_options_sql(table.options)
    if options:
        sql += f" {options}"
    return sql


@dataclass
class CreateForeignTable(Change):
    """Create a foreign table with its columns, server and options."""

    table: ForeignTable = field(
        default_factory=lambda: ForeignTable(schema="public", name="", server="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FOREIGN TABLE"

    def __post_init__(self):
        self.object_name = self.table.qualified_name
        if not self.description:
            self.description = (
                f"Create foreign table {self.table.qualified_name} "
                f"on server {self.table.server}"
            )

    def to_sql(self) -> str:
        return _create_foreign_table_sql(self.table) + ";"


@dataclass
class AlterForeignTable(Change):
    """One `ALTER FOREIGN TABLE` carrying every column and option action.

    A foreign table holds no rows locally, so adding, dropping and retyping
    columns only changes the catalog; none of it needs `--offline` or `@drop`.
    `actions` are rendered, quoted SQL fragments built by the differ.
    """

    schema: str = "public"
    table_name: str = ""
    actions: list[str] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FOREIGN TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table_name}"
        if not self.description:
            self.description = f"Alter foreign table {self.object_name}"

    def to_sql(self) -> str:
        return (
            f"ALTER FOREIGN TABLE {_foreign_table_ident(self.schema, self.table_name)} "
            + ", ".join(self.actions)
            + ";"
        )


@dataclass
class DropForeignTable(Change):
    """Drop an undeclared foreign table. No rows live locally, so no `@drop`.

    No CASCADE: a view still reading from it makes PostgreSQL refuse, which
    the executor retries while that view is dropped in the same run.
    """

    schema: str = "public"
    table_name: str = ""
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FOREIGN TABLE"

    def __post_init__(self):
        self.object_name = f"{self.schema}.{self.table_name}"
        if not self.description:
            self.description = f"Drop foreign table {self.object_name}"

    def to_sql(self) -> str:
        return f"DROP FOREIGN TABLE IF EXISTS {_foreign_table_ident(self.schema, self.table_name)};"


@dataclass
class DropAndCreateForeignTable(Change):
    """Recreate a foreign table whose server changed (there is no ALTER for it).

    The drop, the create and the replayed grants are one DO block, for the
    reason DropAndCreateView gives.
    """

    table: ForeignTable = field(
        default_factory=lambda: ForeignTable(schema="public", name="", server="")
    )
    grants: list[Grant] = field(default_factory=list)
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "FOREIGN TABLE"

    def __post_init__(self):
        self.object_name = self.table.qualified_name
        if not self.description:
            self.description = (
                f"Drop and recreate foreign table {self.table.qualified_name} "
                f"on server {self.table.server}"
            )

    def to_sql(self) -> str:
        t = self.table
        inner = (
            f"DROP FOREIGN TABLE IF EXISTS {_foreign_table_ident(t.schema, t.name)};"
        )
        inner += _execute_sql(_create_foreign_table_sql(t))
        inner += _replay_grants_sql(self.grants)
        return do_block(f"BEGIN {inner} END")


# =============================================================================
# Ownership Changes
# =============================================================================


@dataclass
class AlterOwner(Change):
    """`ALTER <kind> ... OWNER TO role`, for an owner the schema files state.

    The executor runs these after the DDL and index phases, with the comments,
    so an object created or recreated in this run already exists in its final
    form (a recreate resets the owner to the migrating role).
    """

    owner: ObjectOwner = field(
        default_factory=lambda: ObjectOwner(kind="TABLE", schema="", name="", owner="")
    )
    category: ChangeCategory = ChangeCategory.ONLINE
    description: str = ""
    object_type: str = "OWNER"

    def __post_init__(self):
        self.object_name = self.owner.key
        if not self.description:
            kind, _, target = self.owner.key.partition(":")
            self.description = (
                f"Set owner of {kind.lower()} {target} to {self.owner.owner}"
            )

    def to_sql(self) -> str:
        return (
            f"ALTER {self.owner.target_sql} OWNER TO {quote_ident(self.owner.owner)};"
        )
