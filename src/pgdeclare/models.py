"""
Schema diff object model.

Dataclasses representing PostgreSQL database objects for schema comparison.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from .quoting import quote_ident, quote_literal


class ChangeCategory(Enum):
    """Migration change categories determining execution behavior."""

    ONLINE = "online"  # Safe to run with app up
    OFFLINE = "offline"  # Requires --offline flag, app must be stopped
    BLOCKED = "blocked"  # Requires @drop directive for data loss acknowledgment
    TRY_FIRST = "try_first"  # Let Postgres try (e.g., type casting), may fail


@dataclass
class Column:
    """Database column definition."""

    name: str
    type: str
    nullable: bool
    default: Optional[str] = None
    identity: Optional[str] = None  # 'a' (ALWAYS), 'd' (BY DEFAULT), or None
    # Generation expression for a STORED generated column (the text from
    # pg_get_expr), or None when the column is not generated. Mutually
    # exclusive with `default`: a generated column never carries a DEFAULT.
    generated: Optional[str] = None
    # Quoted, schema-qualified collation (`"public"."de"`) when it differs from
    # the type's own; None for the type's default and for types without one.
    collation: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        return (
            self.name == other.name
            and self.type == other.type
            and self.nullable == other.nullable
            and self.default == other.default
            and self.identity == other.identity
            and self.generated == other.generated
            and self.collation == other.collation
        )


@dataclass
class Constraint:
    """Database constraint definition."""

    name: str
    type: str  # 'p' (PK), 'f' (FK), 'u' (unique), 'c' (check)
    definition: str  # from pg_get_constraintdef()
    table: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constraint):
            return NotImplemented
        return (
            self.name == other.name
            and self.type == other.type
            and self.definition == other.definition
        )


@dataclass
class Table:
    """Database table definition."""

    schema: str
    name: str
    columns: dict[str, Column] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    rls_enabled: bool = False  # ALTER TABLE ... ENABLE ROW LEVEL SECURITY
    rls_forced: bool = False  # ALTER TABLE ... FORCE ROW LEVEL SECURITY
    # Declarative partitioning. `partition_key` is pg_get_partkeydef() of a
    # partitioned table ("RANGE (created_at)"), None for a plain table.
    # `partition_of` is the (schema, name) of the parent when this table is a
    # partition, and `partition_bound` its bound as pg_get_expr() prints it
    # ("FOR VALUES FROM (...) TO (...)" or "DEFAULT"). A sub-partitioned child
    # has all three.
    partition_key: Optional[str] = None
    partition_of: Optional[tuple[str, str]] = None
    partition_bound: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def parent_key(self) -> Optional[str]:
        """`schema.name` of the partitioned parent, or None for a non-partition."""
        if self.partition_of is None:
            return None
        return f"{self.partition_of[0]}.{self.partition_of[1]}"


_ROUTINE_KEYWORDS = {"p": "PROCEDURE", "a": "AGGREGATE"}


def routine_keyword(kind: str) -> str:
    """FUNCTION, PROCEDURE or AGGREGATE for a pg_proc.prokind."""
    return _ROUTINE_KEYWORDS.get(kind, "FUNCTION")


def routine_args_sql(kind: str, args: str) -> str:
    """Identity arguments as DDL writes them.

    `pg_get_function_identity_arguments` prints nothing for an aggregate that
    takes no arguments, but DROP and COMMENT ON AGGREGATE spell it `(*)`.
    """
    return "*" if kind == "a" and not args else args


@dataclass
class Function:
    """A function, procedure, aggregate or window function (one pg_proc row).

    For an aggregate, which `pg_get_functiondef` cannot print, the engine
    renders `full_definition` itself from pg_aggregate.
    """

    schema: str
    name: str
    args: str  # Function argument signature from pg_get_function_identity_arguments
    full_definition: str  # Complete function definition from pg_get_functiondef
    definition_hash: str = ""  # Hash of full_definition for quick comparison
    security_definer: bool = False
    search_path: Optional[str] = None  # For SECURITY DEFINER functions
    # Whether PUBLIC retains EXECUTE on this function. Default True mirrors
    # PostgreSQL: a freshly created function (NULL proacl) grants PUBLIC EXECUTE.
    # Consumed only by _diff_function_public_execute (the SECDEF lockdown
    # pass); NOT part of the managed-role grant set-diff.
    public_execute: bool = True
    # pg_proc.prokind: 'f' function, 'p' procedure, 'a' aggregate, 'w' window
    # function. CREATE OR REPLACE cannot change it, so a change of kind is a
    # drop and re-create.
    kind: str = "f"
    # Aggregates only: the result type and pg_aggregate.aggkind, neither of
    # which CREATE OR REPLACE AGGREGATE can change.
    result_type: str = ""
    # Aggregates only: signatures of the functions the aggregate calls (state,
    # final, combine and so on). Dropping one of them drops the aggregate.
    support_functions: tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def signature(self) -> str:
        """Full function signature including arguments."""
        return f"{self.schema}.{self.name}({self.args})"

    @property
    def keyword(self) -> str:
        """The object keyword DDL uses for this routine."""
        return routine_keyword(self.kind)

    @property
    def args_sql(self) -> str:
        """The argument list as written in DDL (`*` for a zero-argument aggregate)."""
        return routine_args_sql(self.kind, self.args)

    @property
    def sql_name(self) -> str:
        """`"schema"."name"(args)`, as DROP, COMMENT and REVOKE name it."""
        return f"{quote_ident(self.schema)}.{quote_ident(self.name)}({self.args_sql})"

    def returns_primitive_type(self) -> bool:
        """
        Check if function returns a primitive type (not a table/composite type).

        Functions returning table types must be created AFTER tables.
        Functions returning primitive types can be created BEFORE tables
        (for use in column defaults).
        """
        # Primitive types that can be created before tables
        primitive_patterns = [
            "void",
            "boolean",
            "bool",
            "text",
            "integer",
            "int",
            "bigint",
            "smallint",
            "numeric",
            "decimal",
            "real",
            "double precision",
            "uuid",
            "timestamp",
            "timestamptz",
            "date",
            "time",
            "timetz",
            "interval",
            "json",
            "jsonb",
            "bytea",
            "oid",
            "name",
            "char",
            "character",
            "varchar",
            "inet",
            "cidr",
            "macaddr",
            "bit",
            "varbit",
            "money",
            "point",
            "line",
            "lseg",
            "box",
            "path",
            "polygon",
            "circle",
            "xml",
            "tsvector",
            "tsquery",
            "regtype",
            "regclass",
            "regproc",
            "regoper",
            "trigger",
            "event_trigger",
            "language_handler",
            "fdw_handler",
            "index_am_handler",
            "tsm_handler",
            "record",
            "anyarray",
            "anyelement",
            "anyenum",
            "anyrange",
            "cstring",
            "internal",
            "opaque",
            "unknown",
            # Enum types - detected by checking if they're defined
        ]

        # Parse return type from definition
        import re

        match = re.search(r"RETURNS\s+(\S+)", self.full_definition, re.IGNORECASE)
        if not match:
            return True  # No RETURNS clause, assume primitive (trigger functions)

        return_type = match.group(1).lower().strip('"')

        # SETOF and TABLE(...) are special
        if return_type == "setof" or return_type == "table":
            # Check if SETOF/TABLE of primitive type
            setof_match = re.search(
                r"RETURNS\s+SETOF\s+(\S+)", self.full_definition, re.IGNORECASE
            )
            if setof_match:
                inner_type = setof_match.group(1).lower().strip('"')
                return inner_type in primitive_patterns
            # TABLE(...) returns are primitive (inline definition)
            table_match = re.search(
                r"RETURNS\s+TABLE\s*\(", self.full_definition, re.IGNORECASE
            )
            if table_match:
                return True  # TABLE(...) is inline, not dependent on existing tables
            return False

        # Check if it's a known primitive type
        return return_type in primitive_patterns

    def get_return_clause(self) -> str:
        """
        Extract the full RETURNS clause from the function definition.

        This includes TABLE(...) column definitions for functions returning tables.
        Used to detect when a function's return type has changed.
        """
        import re

        # Match RETURNS ... up to LANGUAGE or AS keyword
        if self.kind == "a":
            return self.result_type
        match = re.search(
            r"RETURNS\s+(.*?)(?=\s+LANGUAGE|\s+AS\s)",
            self.full_definition,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Function):
            return NotImplemented
        # Compare by definition hash if available, otherwise by full definition
        if self.definition_hash and other.definition_hash:
            return self.definition_hash == other.definition_hash
        return self.full_definition == other.full_definition


@dataclass
class Index:
    """Database index definition."""

    name: str
    table: str
    schema: str
    definition: str  # Full CREATE INDEX statement from pg_indexes
    is_valid: bool = True  # False if index creation failed (CONCURRENTLY)
    # An index on a partitioned table (relkind 'I'). Its definition reads
    # `ON ONLY`, and it is built per partition and attached, not in one go.
    partitioned: bool = False

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.definition == other.definition


@dataclass
class Policy:
    """Row-level security policy definition."""

    name: str
    table: str
    schema: str
    command: str  # SELECT, INSERT, UPDATE, DELETE, ALL
    using_expr: Optional[str] = None  # USING clause
    check_expr: Optional[str] = None  # WITH CHECK clause
    roles: list[str] = field(default_factory=list)  # Roles policy applies to
    permissive: bool = True  # PERMISSIVE (default) vs RESTRICTIVE

    @property
    def qualified_key(self) -> str:
        """Unique key for policy lookup."""
        return f"{self.schema}.{self.table}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Policy):
            return NotImplemented
        return (
            self.name == other.name
            and self.table == other.table
            and self.command == other.command
            and self.using_expr == other.using_expr
            and self.check_expr == other.check_expr
            and self.permissive == other.permissive
        )


@dataclass
class Trigger:
    """Database trigger definition."""

    name: str
    table: str
    schema: str
    definition: str  # Full trigger definition from pg_get_triggerdef
    function_schema: str = ""
    function_name: str = ""

    @property
    def qualified_key(self) -> str:
        """Unique key for trigger lookup."""
        return f"{self.schema}.{self.table}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trigger):
            return NotImplemented
        return self.definition == other.definition


@dataclass
class Rule:
    """A rewrite rule on a table (never a view's `_RETURN` rule)."""

    schema: str
    table: str
    name: str
    definition: str  # pg_get_ruledef(), trailing semicolon included
    # pg_rewrite.ev_enabled: 'O' (origin, the default), 'D' (disabled),
    # 'R' (replica) or 'A' (always).
    enabled: str = "O"

    @property
    def qualified_key(self) -> str:
        return f"{self.schema}.{self.table}.{self.name}"


@dataclass
class EventTrigger:
    """A database-level event trigger."""

    name: str
    event: str  # ddl_command_start, ddl_command_end, sql_drop, ...
    function_schema: str
    function_name: str
    # WHEN TAG IN (...) filter, sorted; empty for no filter.
    tags: tuple[str, ...] = ()
    # pg_event_trigger.evtenabled: 'O', 'D', 'R' or 'A', as for rules.
    enabled: str = "O"


@dataclass
class PublicationTable:
    """One table published by a publication."""

    schema: str
    name: str
    # Column list (PostgreSQL 15+), in column order; empty for all columns.
    columns: tuple[str, ...] = ()
    # Row filter (PostgreSQL 15+) as pg_get_expr() prints it, or None.
    row_filter: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


PUBLISH_ACTIONS = ("insert", "update", "delete", "truncate")


@dataclass
class Publication:
    """A logical replication publication (database-level)."""

    name: str
    all_tables: bool = False  # FOR ALL TABLES
    tables: dict[str, PublicationTable] = field(
        default_factory=dict
    )  # key = schema.table
    # TABLES IN SCHEMA (PostgreSQL 15+), sorted.
    schemas: tuple[str, ...] = ()
    publish: tuple[str, ...] = PUBLISH_ACTIONS
    via_root: bool = False  # publish_via_partition_root
    # publish_generated_columns (PostgreSQL 18+): 'none' or 'stored'; None on
    # servers that do not have the option.
    generated_columns: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.all_tables and not self.tables and not self.schemas


@dataclass
class EnumType:
    """PostgreSQL enum type definition."""

    schema: str
    name: str
    values: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnumType):
            return NotImplemented
        return self.name == other.name and self.values == other.values


@dataclass
class CompositeType:
    """PostgreSQL composite type definition (typtype = 'c').

    Composite types are structured types with named columns, similar to a row type.
    They are created with CREATE TYPE name AS (column1 type1, column2 type2, ...).

    Example:
        CREATE TYPE sort_spec AS (
            column_name TEXT,
            direction sort_direction
        );
    """

    schema: str
    name: str
    columns: list[tuple[str, str]] = field(
        default_factory=list
    )  # [(column_name, type_name), ...]

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeType):
            return NotImplemented
        return (
            self.schema == other.schema
            and self.name == other.name
            and self.columns == other.columns
        )


@dataclass
class Domain:
    """PostgreSQL domain type definition (typtype = 'd').

    `base_type` is format_type() of the underlying type, `collation` the
    quoted, schema-qualified collation when it differs from the base type's
    default (None otherwise). NOT NULL is modelled on the domain, like a
    column's, so PostgreSQL 18's NOT NULL constraint rows are not managed twice.
    `constraints` holds the CHECK constraints, matched by name.

    `dependents` describes what uses the domain in the live database (columns,
    function arguments, other domains). It is not part of equality: it decides
    whether a base-type change may drop and recreate the domain.
    """

    schema: str
    name: str
    base_type: str
    nullable: bool = True
    default: Optional[str] = None
    collation: Optional[str] = None
    constraints: list[Constraint] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return NotImplemented
        return (
            self.schema == other.schema
            and self.name == other.name
            and self.base_type == other.base_type
            and self.nullable == other.nullable
            and self.default == other.default
            and self.collation == other.collation
            and sorted(self.constraints, key=lambda c: c.name)
            == sorted(other.constraints, key=lambda c: c.name)
        )


@dataclass
class RangeType:
    """A range type (CREATE TYPE ... AS RANGE) and its multirange.

    `subtype` is format_type() of the subtype. The other options are quoted,
    schema-qualified names, or None where PostgreSQL chose the default: the
    subtype's default btree operator class, the subtype's own collation, no
    canonical or subtype_diff function. `multirange_type_name` is always set
    from the catalog, so a declared name and a derived one compare the same.

    `dependents` describes what uses the range or its multirange in the live
    database, and `canonical_signature` is the canonical function's
    `Function.signature`. Neither takes part in equality.
    """

    schema: str
    name: str
    subtype: str
    subtype_opclass: Optional[str] = None
    collation: Optional[str] = None
    canonical: Optional[str] = None
    subtype_diff: Optional[str] = None
    multirange_type_name: Optional[str] = None
    dependents: list[str] = field(default_factory=list, compare=False)
    canonical_signature: Optional[str] = field(default=None, compare=False)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass
class Collation:
    """A collation (CREATE COLLATION).

    `provider` is `libc`, `icu` or `builtin`. A libc collation has
    `lc_collate` and `lc_ctype`; an ICU or builtin one has `locale`, and
    `rules` holds ICU tailoring rules (PostgreSQL 16+). The collation version
    is runtime state and is not compared.

    `dependents` describes what uses the collation in the live database
    (columns, domains, range types, indexes). It is not part of equality.
    """

    schema: str
    name: str
    provider: str = "libc"
    locale: Optional[str] = None
    lc_collate: Optional[str] = None
    lc_ctype: Optional[str] = None
    deterministic: bool = True
    rules: Optional[str] = None
    dependents: list[str] = field(default_factory=list, compare=False)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass
class Sequence:
    """A standalone sequence (CREATE SEQUENCE, or the one a serial column makes).

    Identity-column sequences are not modelled here: the column owns them.
    Options are compared as the catalog stores them (pg_sequence), so a
    declaration that relies on defaults and one that spells them out are equal.
    The current value is state, not schema, and is never compared.
    """

    schema: str
    name: str
    data_type: str = "bigint"  # format_type(seqtypid): smallint, integer, bigint
    increment: int = 1
    min_value: int = 1
    max_value: int = 9223372036854775807
    start: int = 1
    cache: int = 1
    cycle: bool = False
    # OWNED BY: (schema, table, column) of the owning column, or None.
    owned_by: Optional[tuple[str, str, str]] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def options_equal(self, other: "Sequence") -> bool:
        """Whether every CREATE/ALTER SEQUENCE option matches (not ownership)."""
        return (
            self.data_type == other.data_type
            and self.increment == other.increment
            and self.min_value == other.min_value
            and self.max_value == other.max_value
            and self.start == other.start
            and self.cache == other.cache
            and self.cycle == other.cycle
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return NotImplemented
        return (
            self.schema == other.schema
            and self.name == other.name
            and self.options_equal(other)
            and self.owned_by == other.owned_by
        )


def normalize_view_reloptions(raw: Optional[list[str]]) -> list[str]:
    """Canonicalize a view's `pg_class.reloptions` array for comparison.

    Postgres stores view options as `name=value` strings. Whitespace and case
    vary with how the DDL was written (`WITH (security_invoker = TRUE)` in the
    schema files vs `security_invoker=true` from the catalog), so both sides are
    lowercased, stripped, and sorted before they are compared or re-emitted.
    """
    if not raw:
        return []

    normalized: list[str] = []
    for option in raw:
        name, sep, value = option.partition("=")
        name = name.strip().lower()
        if not name:
            continue
        normalized.append(f"{name}={value.strip().lower()}" if sep else name)
    return sorted(normalized)


@dataclass
class View:
    """Database view definition."""

    schema: str
    name: str
    definition: str  # View query definition
    # `WITH (...)` options, chiefly `security_invoker=true`. Without them the
    # differ would model views as schema/name/definition only, so a declared
    # security_invoker would be dropped the moment a view round-tripped through
    # the plan database, and `__eq__` could not even see the drift. Owner-rights
    # views owned by a BYPASSRLS superuser mean RLS on the base tables never
    # applies to anything read through the view, so this failed open.
    reloptions: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.reloptions = normalize_view_reloptions(self.reloptions)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def options_clause(self) -> str:
        """`" WITH (...)"` for a CREATE VIEW, or `""` when there are no options."""
        if not self.reloptions:
            return ""
        return f" WITH ({', '.join(self.reloptions)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, View):
            return NotImplemented
        # Normalize whitespace for comparison
        return (
            self.definition.strip() == other.definition.strip()
            and self.reloptions == other.reloptions
        )


@dataclass
class MaterializedView:
    """Materialized view definition (`CREATE MATERIALIZED VIEW`)."""

    schema: str
    name: str
    definition: str  # Query text from pg_get_viewdef, as pg_matviews reports it
    # Storage parameters (`WITH (fillfactor = 70)`, autovacuum settings),
    # canonicalized like a view's options.
    reloptions: list[str] = field(default_factory=list)
    # Whether the object is populated. In the plan database this is the
    # declaration (`WITH DATA` vs `WITH NO DATA`), and it decides how a create or
    # recreate populates the object. It is not compared: a live matview that an
    # application has refreshed, or emptied with `REFRESH ... WITH NO DATA`, has
    # not drifted from its declaration.
    with_data: bool = True

    def __post_init__(self):
        self.reloptions = normalize_view_reloptions(self.reloptions)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def options_clause(self) -> str:
        """`" WITH (...)"` for a CREATE MATERIALIZED VIEW, or `""`."""
        if not self.reloptions:
            return ""
        return f" WITH ({', '.join(self.reloptions)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaterializedView):
            return NotImplemented
        return (
            self.definition.strip() == other.definition.strip()
            and self.reloptions == other.reloptions
        )


@dataclass
class Grant:
    """Privilege grant definition."""

    object_type: str  # TABLE, COLUMN, FUNCTION, PROCEDURE, SCHEMA, SEQUENCE
    object_name: str  # Object name (may include schema)
    schema: str
    privilege: str  # SELECT, INSERT, UPDATE, DELETE, EXECUTE, USAGE, etc.
    grantee: str  # Role receiving the grant
    column: Optional[str] = None  # For column-level grants
    # Routines only: the identity signature (pg_get_function_identity_arguments),
    # so each overload is its own object. None when unknown (a grant built by
    # hand), which names the routine without arguments.
    args: Optional[str] = None

    @property
    def object_sql(self) -> str:
        """The routine as GRANT names it: with its signature when that is known.

        An ordered-set aggregate's identity arguments (`... ORDER BY ...`) are
        not valid in GRANT, so such a routine is named without them.
        """
        name = f"{quote_ident(self.schema)}.{quote_ident(self.object_name)}"
        if self.args is None or "ORDER BY" in self.args:
            return name
        return f"{name}({self.args})"

    @property
    def target(self) -> str:
        """The object this grant is on, without privilege or grantee."""
        if self.column:
            return f"{self.object_type}:{self.schema}.{self.object_name}:{self.column}"
        if self.args is not None:
            return f"{self.object_type}:{self.schema}.{self.object_name}({self.args})"
        return f"{self.object_type}:{self.schema}.{self.object_name}"

    @property
    def key(self) -> str:
        """Unique key for grant lookup."""
        return f"{self.target}:{self.privilege}:{self.grantee}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grant):
            return NotImplemented
        return self.key == other.key


# pg_default_acl.defaclobjtype -> the keyword ALTER DEFAULT PRIVILEGES uses.
# ROUTINES is an alias of FUNCTIONS and is stored as 'f'.
DEFAULT_PRIVILEGE_OBJECT_TYPES = {
    "r": "TABLES",
    "S": "SEQUENCES",
    "f": "FUNCTIONS",
    "T": "TYPES",
    "n": "SCHEMAS",
}


@dataclass
class DefaultPrivilege:
    """One privilege of an ALTER DEFAULT PRIVILEGES entry (pg_default_acl).

    `target_role` is the role whose future objects receive the privilege (the
    `FOR ROLE`; the session user when the declaration names none). `schema` is
    the `IN SCHEMA`, or None for an entry that applies in every schema.
    `object_type` is the plural keyword: TABLES, SEQUENCES, FUNCTIONS, TYPES or
    SCHEMAS.
    """

    target_role: str
    schema: Optional[str]
    object_type: str
    privilege: str
    grantee: str
    grantable: bool = False

    @property
    def key(self) -> str:
        """Unique key, grant option included, so a change of it re-grants."""
        scope = self.schema if self.schema is not None else "*"
        option = ":grantable" if self.grantable else ""
        return (
            f"{self.target_role}:{scope}:{self.object_type}:"
            f"{self.privilege}:{self.grantee}{option}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultPrivilege):
            return NotImplemented
        return self.key == other.key


@dataclass
class Role:
    """Database role definition."""

    name: str
    can_login: bool = False
    superuser: bool = False
    create_db: bool = False
    create_role: bool = False
    inherit: bool = True
    replication: bool = False
    bypass_rls: bool = False
    connection_limit: int = -1  # -1 means unlimited
    password: Optional[str] = None  # Password for login roles (dev only)

    @property
    def options(self) -> list[str]:
        """Get list of WITH options for this role."""
        return self.options_for(mask_password=False)

    def options_for(self, mask_password: bool) -> list[str]:
        """WITH options for this role; the password masked for display."""
        opts = []
        if self.can_login:
            opts.append("LOGIN")
        else:
            opts.append("NOLOGIN")
        if self.superuser:
            opts.append("SUPERUSER")
        if self.create_db:
            opts.append("CREATEDB")
        if self.create_role:
            opts.append("CREATEROLE")
        if not self.inherit:
            opts.append("NOINHERIT")
        if self.replication:
            opts.append("REPLICATION")
        if self.bypass_rls:
            opts.append("BYPASSRLS")
        if self.connection_limit != -1:
            opts.append(f"CONNECTION LIMIT {self.connection_limit}")
        if self.password:
            password = "***" if mask_password else self.password
            opts.append(f"PASSWORD {quote_literal(password)}")
        return opts

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Role):
            return NotImplemented
        # Compare all attributes except password (sensitive)
        return (
            self.name == other.name
            and self.can_login == other.can_login
            and self.superuser == other.superuser
            and self.create_db == other.create_db
            and self.create_role == other.create_role
            and self.inherit == other.inherit
            and self.replication == other.replication
            and self.bypass_rls == other.bypass_rls
            and self.connection_limit == other.connection_limit
        )


@dataclass
class RoleMembership:
    """Role membership (GRANT role TO member)."""

    role: str  # Role being granted
    member: str  # Role receiving membership
    admin_option: bool = False  # Can member grant this role to others?

    @property
    def key(self) -> str:
        """Unique key for membership lookup."""
        return f"{self.role}:{self.member}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoleMembership):
            return NotImplemented
        return (
            self.role == other.role
            and self.member == other.member
            and self.admin_option == other.admin_option
        )


# Object kinds that sit on a relation and are named `ON schema.table`.
_COMMENT_ON_RELATION = frozenset({"CONSTRAINT", "TRIGGER", "POLICY"})
_COMMENT_ON_ROUTINE = {"FUNCTION": "f", "PROCEDURE": "p", "AGGREGATE": "a"}


@dataclass
class Comment:
    """A `COMMENT ON` an object the engine manages.

    `object_type` is the keyword the COMMENT statement uses: TABLE, VIEW,
    COLUMN, INDEX, FUNCTION, PROCEDURE, AGGREGATE, TYPE, COLLATION, CONSTRAINT,
    TRIGGER or POLICY. `name` is the object itself, or for a column,
    constraint, trigger or policy the relation it belongs to, with the member
    in `member`. `args` is a routine's identity signature, as
    `pg_get_function_identity_arguments` prints it.
    """

    object_type: str
    schema: str
    name: str
    text: str
    member: str = ""
    args: str = ""

    @property
    def key(self) -> str:
        """Unique key for the commented object (not the text)."""
        if self.object_type in _COMMENT_ON_ROUTINE:
            return f"{self.object_type}:{self.schema}.{self.name}({self.args})"
        if self.member:
            return f"{self.object_type}:{self.schema}.{self.name}.{self.member}"
        return f"{self.object_type}:{self.schema}.{self.name}"

    @property
    def target_sql(self) -> str:
        """The object as written after `COMMENT ON`, every name quoted."""
        qualified = f"{quote_ident(self.schema)}.{quote_ident(self.name)}"
        kind = _COMMENT_ON_ROUTINE.get(self.object_type)
        if kind is not None:
            args = routine_args_sql(kind, self.args)
            return f"{self.object_type} {qualified}({args})"
        if self.object_type == "COLUMN":
            return f"COLUMN {qualified}.{quote_ident(self.member)}"
        if self.object_type in _COMMENT_ON_RELATION:
            return f"{self.object_type} {quote_ident(self.member)} ON {qualified}"
        return f"{self.object_type} {qualified}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Comment):
            return NotImplemented
        return self.key == other.key and self.text == other.text


def parse_options(raw: Optional[list[str]]) -> dict[str, str]:
    """A catalog `OPTIONS` array (`{name=value,...}`) as a dict, in order."""
    options: dict[str, str] = {}
    for item in raw or []:
        name, _, value = item.partition("=")
        options[name] = value
    return options


def fdw_options_sql(
    options: dict[str, str], masked: frozenset[str] = frozenset()
) -> str:
    """`OPTIONS (name 'value', ...)`, or `""` when there are none.

    The value of every option named in `masked` is written as `'***'`.
    """
    if not options:
        return ""
    parts = [
        f"{quote_ident(name)} {quote_literal('***' if name in masked else value)}"
        for name, value in options.items()
    ]
    return f"OPTIONS ({', '.join(parts)})"


def alter_fdw_options_sql(
    desired: dict[str, str],
    actual: dict[str, str],
    masked: frozenset[str] = frozenset(),
) -> str:
    """`OPTIONS (ADD ..., SET ..., DROP ...)` turning `actual` into `desired`.

    Returns `""` when the two are equal. Values of options in `masked` are
    written as `'***'`.
    """
    parts: list[str] = []
    for name, value in desired.items():
        shown = quote_literal("***" if name in masked else value)
        if name not in actual:
            parts.append(f"ADD {quote_ident(name)} {shown}")
        elif actual[name] != value:
            parts.append(f"SET {quote_ident(name)} {shown}")
    for name in actual:
        if name not in desired:
            parts.append(f"DROP {quote_ident(name)}")
    if not parts:
        return ""
    return f"OPTIONS ({', '.join(parts)})"


def option_names_changed(desired: dict[str, str], actual: dict[str, str]) -> list[str]:
    """Names of the options that differ, for plan descriptions (never values)."""
    names = {n for n in desired if actual.get(n) != desired[n]}
    names |= {n for n in actual if n not in desired}
    return sorted(names)


@dataclass
class Extension:
    """An extension declared with `CREATE EXTENSION`, or installed in a database.

    On the declared side `schema` and `version` are None unless the statement
    names them (`WITH SCHEMA`, `VERSION`): an unstated one is not compared, so
    a database on an older version of an extension is not updated behind the
    declaration's back. On the introspected side both are always set.
    """

    name: str
    schema: Optional[str] = None
    version: Optional[str] = None
    cascade: bool = False


@dataclass
class ForeignServer:
    """A foreign server (`CREATE SERVER ... FOREIGN DATA WRAPPER ...`)."""

    name: str
    fdw: str
    type: Optional[str] = None
    version: Optional[str] = None
    options: dict[str, str] = field(default_factory=dict)


# User-mapping options whose values are not secret. Every other option (a
# `password` above all) is masked in plan output and logs, and left alone on
# the target when the declaration does not set it, so a password provisioned by
# hand outside the schema files survives.
USER_MAPPING_PUBLIC_OPTIONS = frozenset({"user", "password_required"})


@dataclass
class UserMapping:
    """A user mapping (`CREATE USER MAPPING FOR user SERVER server`).

    `user` is the role name, or `public` for `FOR PUBLIC`. `options` is None
    when the connecting role is not allowed to read them (pg_user_mappings
    hides them); then they are not compared.
    """

    server: str
    user: str
    options: Optional[dict[str, str]] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.server, self.user)

    @property
    def secret_options(self) -> frozenset[str]:
        """Names of the options whose values are masked for display."""
        return frozenset(
            n for n in (self.options or {}) if n not in USER_MAPPING_PUBLIC_OPTIONS
        )

    @property
    def user_sql(self) -> str:
        """The mapped user as written after FOR: `PUBLIC` or a quoted role."""
        return "PUBLIC" if self.user == "public" else quote_ident(self.user)

    @property
    def label(self) -> str:
        return f"{self.user} on server {self.server}"


@dataclass
class ForeignTable:
    """A foreign table (relkind 'f') in a managed schema.

    `column_options` holds each column's `OPTIONS (...)` (postgres_fdw's
    `column_name`, say), keyed by column name; columns without options are
    absent.
    """

    schema: str
    name: str
    server: str
    columns: dict[str, Column] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)
    column_options: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


# The keyword `ALTER <kind> ... OWNER TO` takes for each owner kind.
OWNER_KINDS = frozenset(
    {
        "TABLE",
        "FOREIGN TABLE",
        "VIEW",
        "MATERIALIZED VIEW",
        "SEQUENCE",
        "FUNCTION",
        "PROCEDURE",
        "AGGREGATE",
        "TYPE",
        "DOMAIN",
        "COLLATION",
        "SCHEMA",
    }
)
# Owner kinds that name a routine, with its identity arguments.
ROUTINE_OWNER_KINDS = frozenset({"FUNCTION", "PROCEDURE", "AGGREGATE"})


@dataclass
class ObjectOwner:
    """The owner of a managed object (`pg_class.relowner` and friends).

    `kind` is one of OWNER_KINDS. A schema has its name in `name` and an
    empty `schema`; a function, procedure or aggregate (a window function is a
    FUNCTION) carries its identity arguments in `args`.
    """

    kind: str
    schema: str
    name: str
    owner: str
    args: str = ""

    @property
    def key(self) -> str:
        if self.kind == "SCHEMA":
            return f"SCHEMA:{self.name}"
        if self.kind in ROUTINE_OWNER_KINDS:
            return f"{self.kind}:{self.schema}.{self.name}({self.args})"
        return f"{self.kind}:{self.schema}.{self.name}"

    @property
    def target_sql(self) -> str:
        """The object as written after `ALTER`, every name quoted."""
        if self.kind not in OWNER_KINDS:
            raise ValueError(f"unexpected owner kind {self.kind!r}")
        if self.kind == "SCHEMA":
            return f"SCHEMA {quote_ident(self.name)}"
        qualified = f"{quote_ident(self.schema)}.{quote_ident(self.name)}"
        if self.kind in ROUTINE_OWNER_KINDS:
            # A zero-argument aggregate is spelled `(*)`.
            args = "*" if self.kind == "AGGREGATE" and not self.args else self.args
            return f"{self.kind} {qualified}({args})"
        return f"{self.kind} {qualified}"


@dataclass
class Dependency:
    """Object dependency from pg_depend."""

    dependent_type: str  # Class of dependent object
    dependent_oid: int
    referenced_type: str  # Class of referenced object
    referenced_oid: int
    dep_type: str  # 'n' (normal), 'a' (auto), 'i' (internal)


@dataclass
class Schema:
    """Complete database schema representation for one or more PostgreSQL schemas."""

    tables: dict[str, Table] = field(default_factory=dict)  # key = schema.table
    functions: dict[str, Function] = field(
        default_factory=dict
    )  # key = schema.name(args)
    indexes: dict[str, Index] = field(default_factory=dict)  # key = schema.index
    policies: dict[str, Policy] = field(
        default_factory=dict
    )  # key = schema.table.policy
    triggers: dict[str, Trigger] = field(
        default_factory=dict
    )  # key = schema.table.trigger
    enums: dict[str, EnumType] = field(default_factory=dict)  # key = schema.enum
    composite_types: dict[str, CompositeType] = field(
        default_factory=dict
    )  # key = schema.type
    domains: dict[str, Domain] = field(default_factory=dict)  # key = schema.domain
    ranges: dict[str, RangeType] = field(default_factory=dict)  # key = schema.range
    collations: dict[str, Collation] = field(
        default_factory=dict
    )  # key = schema.collation
    views: dict[str, View] = field(default_factory=dict)  # key = schema.view
    materialized_views: dict[str, MaterializedView] = field(
        default_factory=dict
    )  # key = schema.matview
    # Views and materialized views that read from a view or materialized view:
    # key = schema.name of the referenced relation, value = schema.name of each
    # direct dependent. Dropping a relation fails while a dependent exists, so a
    # recreate has to take its dependents down and rebuild them with it.
    relation_dependents: dict[str, list[str]] = field(default_factory=dict)
    sequences: dict[str, Sequence] = field(
        default_factory=dict
    )  # key = schema.sequence
    grants: list[Grant] = field(default_factory=list)
    default_privileges: list[DefaultPrivilege] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    roles: dict[str, Role] = field(default_factory=dict)  # key = role name
    role_memberships: list[RoleMembership] = field(default_factory=list)
    comments: dict[str, Comment] = field(default_factory=dict)  # key = Comment.key
    rules: dict[str, Rule] = field(default_factory=dict)  # key = schema.table.rule
    # Database-level objects: every one in the database is introspected, and
    # the differ decides which of them it may touch.
    publications: dict[str, Publication] = field(default_factory=dict)  # key = name
    event_triggers: dict[str, EventTrigger] = field(default_factory=dict)  # key = name
    # Extensions: on the declared side, the CREATE EXTENSION statements of the
    # schema files; on the live side, every installed extension.
    extensions: dict[str, Extension] = field(default_factory=dict)  # key = name
    # Extensions a `-- @drop extension:NAME` directive consents to dropping.
    extension_drops: set[str] = field(default_factory=set)
    foreign_servers: dict[str, ForeignServer] = field(
        default_factory=dict
    )  # key = server name
    user_mappings: dict[tuple[str, str], UserMapping] = field(
        default_factory=dict
    )  # key = (server, user)
    foreign_tables: dict[str, ForeignTable] = field(
        default_factory=dict
    )  # key = schema.table
    # Owners, filled only with `PgDeclareConfig.manage_ownership`. On the
    # declared side only owners the schema files state (a managed role other
    # than the role building the plan) are kept.
    owners: dict[str, ObjectOwner] = field(default_factory=dict)  # key = .key

    def get_table(self, schema: str, name: str) -> Optional[Table]:
        """Get table by schema and name."""
        return self.tables.get(f"{schema}.{name}")

    def get_function(self, schema: str, name: str, args: str) -> Optional[Function]:
        """Get function by signature."""
        return self.functions.get(f"{schema}.{name}({args})")

    def get_grants_for_object(
        self, object_type: str, schema: str, object_name: str
    ) -> list[Grant]:
        """Get all grants for a specific object."""
        return [
            g
            for g in self.grants
            if g.object_type == object_type
            and g.schema == schema
            and g.object_name == object_name
        ]

    def has_comment_target(self, comment: Comment) -> bool:
        """Whether the object `comment` is on exists in this schema."""
        relation = f"{comment.schema}.{comment.name}"
        kind = comment.object_type
        if kind == "TABLE":
            return relation in self.tables
        if kind == "VIEW":
            return relation in self.views
        if kind == "COLUMN":
            table = self.tables.get(relation)
            if table is not None:
                return comment.member in table.columns
            # View columns are not modelled; a view whose columns change is
            # recreated, which the differ accounts for.
            return relation in self.views
        if kind == "CONSTRAINT":
            table = self.tables.get(relation)
            return table is not None and any(
                c.name == comment.member for c in table.constraints
            )
        if kind == "TRIGGER":
            return f"{relation}.{comment.member}" in self.triggers
        if kind == "POLICY":
            return f"{relation}.{comment.member}" in self.policies
        if kind == "INDEX":
            return relation in self.indexes
        if kind in _COMMENT_ON_ROUTINE:
            func = self.functions.get(f"{relation}({comment.args})")
            return func is not None and func.keyword == kind
        if kind == "TYPE":
            return (
                relation in self.enums
                or relation in self.composite_types
                or relation in self.ranges
            )
        if kind == "COLLATION":
            return relation in self.collations
        return False

    def has_owner_target(self, owner: ObjectOwner) -> bool:
        """Whether the object `owner` describes is one this schema models."""
        relation = f"{owner.schema}.{owner.name}"
        kind = owner.kind
        if kind == "TABLE":
            return relation in self.tables
        if kind == "FOREIGN TABLE":
            return relation in self.foreign_tables
        if kind == "VIEW":
            return relation in self.views
        if kind == "MATERIALIZED VIEW":
            return relation in self.materialized_views
        if kind == "SEQUENCE":
            return relation in self.sequences
        if kind in ROUTINE_OWNER_KINDS:
            func = self.functions.get(f"{relation}({owner.args})")
            return func is not None and func.keyword == kind
        if kind == "TYPE":
            return (
                relation in self.enums
                or relation in self.composite_types
                or relation in self.ranges
            )
        if kind == "DOMAIN":
            return relation in self.domains
        if kind == "COLLATION":
            return relation in self.collations
        # Schemas: the caller only asks about managed ones.
        return kind == "SCHEMA"

    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(name)

    def get_role_memberships_for(self, member: str) -> list[RoleMembership]:
        """Get all role memberships where the given role is the member."""
        return [m for m in self.role_memberships if m.member == member]

    def get_role_members(self, role: str) -> list[RoleMembership]:
        """Get all roles that are members of the given role."""
        return [m for m in self.role_memberships if m.role == role]
