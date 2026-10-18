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

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass
class Function:
    """Database function definition."""

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

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def signature(self) -> str:
        """Full function signature including arguments."""
        return f"{self.schema}.{self.name}({self.args})"

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
class Grant:
    """Privilege grant definition."""

    object_type: str  # TABLE, FUNCTION, SCHEMA, SEQUENCE
    object_name: str  # Object name (may include schema)
    schema: str
    privilege: str  # SELECT, INSERT, UPDATE, DELETE, EXECUTE, USAGE, etc.
    grantee: str  # Role receiving the grant
    column: Optional[str] = None  # For column-level grants

    @property
    def key(self) -> str:
        """Unique key for grant lookup."""
        if self.column:
            return f"{self.object_type}:{self.schema}.{self.object_name}:{self.column}:{self.privilege}:{self.grantee}"
        return f"{self.object_type}:{self.schema}.{self.object_name}:{self.privilege}:{self.grantee}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grant):
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


@dataclass
class Comment:
    """A `COMMENT ON` an object the engine manages.

    `object_type` is the keyword the COMMENT statement uses: TABLE, VIEW,
    COLUMN, INDEX, FUNCTION, TYPE, CONSTRAINT, TRIGGER or POLICY. `name` is
    the object itself, or for a column, constraint, trigger or policy the
    relation it belongs to, with the member in `member`. `args` is a function's
    identity signature, as `pg_get_function_identity_arguments` prints it.
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
        if self.object_type == "FUNCTION":
            return f"FUNCTION:{self.schema}.{self.name}({self.args})"
        if self.member:
            return f"{self.object_type}:{self.schema}.{self.name}.{self.member}"
        return f"{self.object_type}:{self.schema}.{self.name}"

    @property
    def target_sql(self) -> str:
        """The object as written after `COMMENT ON`, every name quoted."""
        qualified = f"{quote_ident(self.schema)}.{quote_ident(self.name)}"
        if self.object_type == "FUNCTION":
            return f"FUNCTION {qualified}({self.args})"
        if self.object_type == "COLUMN":
            return f"COLUMN {qualified}.{quote_ident(self.member)}"
        if self.object_type in _COMMENT_ON_RELATION:
            return f"{self.object_type} {quote_ident(self.member)} ON {qualified}"
        return f"{self.object_type} {qualified}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Comment):
            return NotImplemented
        return self.key == other.key and self.text == other.text


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
    views: dict[str, View] = field(default_factory=dict)  # key = schema.view
    grants: list[Grant] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    roles: dict[str, Role] = field(default_factory=dict)  # key = role name
    role_memberships: list[RoleMembership] = field(default_factory=list)
    comments: dict[str, Comment] = field(default_factory=dict)  # key = Comment.key

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
        if kind == "FUNCTION":
            return f"{relation}({comment.args})" in self.functions
        if kind == "TYPE":
            return relation in self.enums or relation in self.composite_types
        return False

    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(name)

    def get_role_memberships_for(self, member: str) -> list[RoleMembership]:
        """Get all role memberships where the given role is the member."""
        return [m for m in self.role_memberships if m.member == member]

    def get_role_members(self, role: str) -> list[RoleMembership]:
        """Get all roles that are members of the given role."""
        return [m for m in self.role_memberships if m.role == role]
