"""
Database introspection using pg_catalog.

Queries PostgreSQL system catalogs to build a complete Schema object
representing the current database state.
"""

import hashlib
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .quoting import quote_ident
from .config import PgDeclareConfig, logger, resolve
from .models import (
    Column,
    CompositeType,
    Constraint,
    Dependency,
    EnumType,
    Function,
    Grant,
    Index,
    MaterializedView,
    Policy,
    Role,
    RoleMembership,
    Schema,
    Table,
    Trigger,
    View,
)


def introspect_schema(
    connection_string: str,
    schemas: list[str] | None = None,
    managed_roles: list[str] | None = None,
    config: Optional[PgDeclareConfig] = None,
) -> Schema:
    """
    Query pg_catalog to build complete Schema object for specified schemas.

    Args:
        connection_string: PostgreSQL connection string
        schemas: List of schema names to introspect (default: `config.schemas`)
        managed_roles: List of role names to introspect and query grants for
            (default: `config.managed_roles`)
        config: Engine configuration

    Returns:
        Schema object with all database objects
    """
    cfg = resolve(config)
    if schemas is None:
        schemas = list(cfg.schemas)

    if managed_roles is None:
        managed_roles = list(cfg.managed_roles)

    if cfg.dry_run:
        logger.info("DRY RUN: Would introspect current database state")
        return Schema()

    result = Schema()

    with psycopg2.connect(connection_string) as conn:
        # Query each object type
        for schema in schemas:
            tables = _query_tables(conn, schema, cfg.search_path_sql)
            result.tables.update(tables)

            functions = _query_functions(conn, schema)
            result.functions.update(functions)

            indexes = _query_indexes(conn, schema)
            result.indexes.update(indexes)

            policies = _query_policies(conn, schema)
            result.policies.update(policies)

            triggers = _query_triggers(conn, schema)
            result.triggers.update(triggers)

            enums = _query_enums(conn, schema)
            result.enums.update(enums)

            composite_types = _query_composite_types(conn, schema)
            result.composite_types.update(composite_types)

            views = _query_views(conn, schema)
            result.views.update(views)

            result.materialized_views.update(_query_materialized_views(conn, schema))

            for role in managed_roles:
                grants = _query_grants(conn, schema, role)
                result.grants.extend(grants)

        # Query dependencies (cross-schema)
        result.dependencies = _query_dependencies(conn, schemas)
        result.relation_dependents = _query_relation_dependents(conn, schemas)

        # Query schema-level grants for all managed roles
        for role in managed_roles:
            schema_grants = _query_schema_grants(conn, role)
            result.grants.extend(schema_grants)

        # Query roles and memberships for managed roles
        result.roles = _query_roles(conn, managed_roles)
        result.role_memberships = _query_role_memberships(conn, managed_roles)

    return result


def _query_tables(conn, schema: str, search_path: str = "public") -> dict[str, Table]:
    """Query tables and their columns from pg_catalog."""
    tables: dict[str, Table] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Query tables (including RLS state)
        cur.execute(
            """
            SELECT
                c.relname AS table_name,
                c.relrowsecurity AS rls_enabled,
                c.relforcerowsecurity AS rls_forced
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
              AND n.nspname = %s
              AND c.relname NOT LIKE 'pg_%%'
              AND NOT c.relname ~ '^_'
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_class'::regclass AND ext.objid = c.oid
                    AND ext.deptype = 'e'
              )
            ORDER BY c.relname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            table_name = row["table_name"]
            table = Table(
                schema=schema,
                name=table_name,
                rls_enabled=row["rls_enabled"],
                rls_forced=row["rls_forced"],
            )

            # Query columns for this table
            table.columns = _query_columns(conn, schema, table_name, search_path)

            # Query constraints for this table
            table.constraints = _query_constraints(conn, schema, table_name)

            tables[f"{schema}.{table_name}"] = table

    return tables


def _query_columns(
    conn, schema: str, table_name: str, search_path: str = "public"
) -> dict[str, Column]:
    """Query columns for a specific table.

    `search_path` is set first so format_type() names types the way the SQL
    files that created them did.
    """
    columns: dict[str, Column] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Set search_path to ensure format_type returns consistent names
        # This matches the search_path used when SQL files are executed
        cur.execute(f"SET search_path TO {search_path}")
        cur.execute(
            """
            SELECT
                a.attname AS column_name,
                format_type(a.atttypid, a.atttypmod) AS data_type,
                NOT a.attnotnull AS is_nullable,
                pg_get_expr(d.adbin, d.adrelid) AS column_default,
                a.attidentity AS identity_type,
                a.attgenerated AS generated_kind
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE c.relname = %s
              AND n.nspname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (table_name, schema),
        )

        for row in cur.fetchall():
            # attidentity: 'a' = ALWAYS, 'd' = BY DEFAULT, '' = not identity
            identity = row["identity_type"] if row["identity_type"] else None

            # attgenerated: 's' = STORED generated column, 'v' = VIRTUAL
            # (newer Postgres, unsupported here), '' = not generated. For a
            # STORED column pg_get_expr() returns the *generation* expression
            # (stored in pg_attrdef), not a DEFAULT, so route it into
            # `generated` and keep `default` NULL.
            generated_kind = row["generated_kind"] or ""
            generated = None
            default = row["column_default"]
            if generated_kind == "s":
                generated = default
                default = None
            elif generated_kind == "v":
                raise ValueError(
                    f"column {quote_ident(schema)}.{quote_ident(table_name)}.{quote_ident(row['column_name'])}: "
                    "virtual generated columns are not supported by the "
                    "schema-diff tool; use a STORED generated column, or expose "
                    "the value via an IMMUTABLE function. (A raw text::enum cast "
                    "is not IMMUTABLE - an enum RENAME VALUE can change the "
                    "label<->value mapping - so route enum-valued expressions "
                    "through a deliberately IMMUTABLE-marked function.)"
                )

            col = Column(
                name=row["column_name"],
                type=row["data_type"],
                nullable=row["is_nullable"],
                default=default,
                identity=identity,
                generated=generated,
            )
            columns[col.name] = col

    return columns


def _query_constraints(conn, schema: str, table_name: str) -> list[Constraint]:
    """Query constraints for a specific table."""
    constraints: list[Constraint] = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                con.conname AS constraint_name,
                con.contype AS constraint_type,
                pg_get_constraintdef(con.oid) AS constraint_def
            FROM pg_constraint con
            JOIN pg_class c ON c.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s
              AND n.nspname = %s
              AND con.contype != 't'  -- Exclude trigger constraints (handled by _query_triggers)
              -- PostgreSQL 18 records NOT NULL as a pg_constraint row (contype 'n').
              -- Nullability is modelled on the column (SetNotNull/DropNotNull), so
              -- these rows would be managed twice. Earlier versions have none.
              AND con.contype != 'n'
            ORDER BY con.conname
            """,
            (table_name, schema),
        )

        for row in cur.fetchall():
            constraints.append(
                Constraint(
                    name=row["constraint_name"],
                    type=row["constraint_type"],
                    definition=row["constraint_def"],
                    table=table_name,
                )
            )

    return constraints


def _query_functions(conn, schema: str) -> dict[str, Function]:
    """Query functions from pg_proc."""
    functions: dict[str, Function] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.proname AS function_name,
                pg_get_function_identity_arguments(p.oid) AS args,
                p.prosecdef AS security_definer,
                pg_get_functiondef(p.oid) AS full_definition,
                p.oid AS func_oid,
                -- PUBLIC retains EXECUTE if the ACL is default (NULL => PUBLIC
                -- EXECUTE) or an explicit grantee=0 EXECUTE entry exists.
                (
                    p.proacl IS NULL
                    OR EXISTS (
                        SELECT 1 FROM aclexplode(p.proacl) a
                        WHERE a.grantee = 0 AND a.privilege_type = 'EXECUTE'
                    )
                ) AS public_execute
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %s
              AND p.prokind = 'f'
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_proc'::regclass AND ext.objid = p.oid
                    AND ext.deptype = 'e'
              )
            ORDER BY p.oid
            """,
            (schema,),
        )

        for row in cur.fetchall():
            full_def = row["full_definition"] or ""
            func = Function(
                schema=schema,
                name=row["function_name"],
                args=row["args"] or "",
                full_definition=full_def,
                definition_hash=hashlib.sha256(full_def.encode()).hexdigest()[:16],
                security_definer=row["security_definer"],
                public_execute=row["public_execute"],
            )
            functions[func.signature] = func

    return functions


def _query_indexes(conn, schema: str) -> dict[str, Index]:
    """Query indexes from pg_indexes.

    Excludes indexes that back constraints (PRIMARY KEY, UNIQUE, EXCLUDE).
    These are managed via _query_constraints and AddConstraint/DropConstraint.
    Including them here would cause duplicate operations since AddConstraint
    automatically creates the backing index.
    """
    indexes: dict[str, Index] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                i.indexname AS index_name,
                i.tablename AS table_name,
                i.indexdef AS index_definition,
                ix.indisvalid AS is_valid
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.indexname
            JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = i.schemaname
            JOIN pg_index ix ON ix.indexrelid = c.oid
            WHERE i.schemaname = %s
              -- Exclude indexes that back constraints (PK, UNIQUE, EXCLUDE)
              -- These are managed by _query_constraints / AddConstraint
              AND NOT EXISTS (
                  SELECT 1 FROM pg_constraint con
                  WHERE con.conindid = c.oid
              )
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_class'::regclass AND ext.objid = ix.indrelid
                    AND ext.deptype = 'e'
              )
            ORDER BY i.indexname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            idx = Index(
                name=row["index_name"],
                table=row["table_name"],
                schema=schema,
                definition=row["index_definition"],
                is_valid=row["is_valid"],
            )
            indexes[idx.qualified_name] = idx

    return indexes


def _query_policies(conn, schema: str) -> dict[str, Policy]:
    """Query RLS policies from pg_policy."""
    policies: dict[str, Policy] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                pol.polname AS policy_name,
                c.relname AS table_name,
                CASE pol.polcmd
                    WHEN 'r' THEN 'SELECT'
                    WHEN 'a' THEN 'INSERT'
                    WHEN 'w' THEN 'UPDATE'
                    WHEN 'd' THEN 'DELETE'
                    WHEN '*' THEN 'ALL'
                END AS command,
                pg_get_expr(pol.polqual, pol.polrelid) AS using_expr,
                pg_get_expr(pol.polwithcheck, pol.polrelid) AS check_expr,
                pol.polpermissive AS permissive,
                ARRAY(
                    SELECT rolname FROM pg_roles
                    WHERE oid = ANY(pol.polroles)
                ) AS roles
            FROM pg_policy pol
            JOIN pg_class c ON pol.polrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
            ORDER BY c.relname, pol.polname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            policy = Policy(
                name=row["policy_name"],
                table=row["table_name"],
                schema=schema,
                command=row["command"],
                using_expr=row["using_expr"],
                check_expr=row["check_expr"],
                roles=row["roles"] or [],
                permissive=bool(row["permissive"]),
            )
            policies[policy.qualified_key] = policy

    return policies


def _query_triggers(conn, schema: str) -> dict[str, Trigger]:
    """Query triggers from pg_trigger."""
    triggers: dict[str, Trigger] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                t.tgname AS trigger_name,
                c.relname AS table_name,
                pg_get_triggerdef(t.oid) AS trigger_definition,
                np.nspname AS function_schema,
                p.proname AS function_name
            FROM pg_trigger t
            JOIN pg_class c ON t.tgrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_proc p ON p.oid = t.tgfoid
            JOIN pg_namespace np ON np.oid = p.pronamespace
            WHERE n.nspname = %s
              AND NOT t.tgisinternal
            ORDER BY c.relname, t.tgname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            trigger = Trigger(
                name=row["trigger_name"],
                table=row["table_name"],
                schema=schema,
                definition=row["trigger_definition"],
                function_schema=row["function_schema"],
                function_name=row["function_name"],
            )
            triggers[trigger.qualified_key] = trigger

    return triggers


def _query_enums(conn, schema: str) -> dict[str, EnumType]:
    """Query enum types from pg_type."""
    enums: dict[str, EnumType] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                t.typname AS enum_name,
                array_agg(e.enumlabel ORDER BY e.enumsortorder) AS enum_values
            FROM pg_type t
            JOIN pg_enum e ON t.oid = e.enumtypid
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE n.nspname = %s
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_type'::regclass AND ext.objid = t.oid
                    AND ext.deptype = 'e'
              )
            GROUP BY t.typname
            ORDER BY t.typname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            enum = EnumType(
                schema=schema,
                name=row["enum_name"],
                values=row["enum_values"] or [],
            )
            enums[enum.qualified_name] = enum

    return enums


def _query_composite_types(conn, schema: str) -> dict[str, CompositeType]:
    """Query composite types from pg_type/pg_attribute.

    Composite types (typtype = 'c') are user-defined types with named columns.
    This excludes:
    - Table row types (have a corresponding pg_class entry with reltype = type OID)
    - View row types
    - Domain types
    - Enum types (typtype = 'e')
    """
    composite_types: dict[str, CompositeType] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                n.nspname AS schema_name,
                t.typname AS type_name,
                array_agg(a.attname ORDER BY a.attnum) AS column_names,
                array_agg(pg_catalog.format_type(a.atttypid, a.atttypmod) ORDER BY a.attnum) AS column_types
            FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            JOIN pg_class c ON c.oid = t.typrelid
            JOIN pg_attribute a ON a.attrelid = t.typrelid
            WHERE t.typtype = 'c'
              AND n.nspname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
              -- Exclude table/view row types: these have a relkind of 'r' (table) or 'v' (view)
              -- Composite types have relkind = 'c' (composite type)
              AND c.relkind = 'c'
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_type'::regclass AND ext.objid = t.oid
                    AND ext.deptype = 'e'
              )
            GROUP BY n.nspname, t.typname
            ORDER BY t.typname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            column_names = row["column_names"] or []
            column_types = row["column_types"] or []
            columns = list(zip(column_names, column_types))

            comp_type = CompositeType(
                schema=schema,
                name=row["type_name"],
                columns=columns,
            )
            composite_types[comp_type.qualified_name] = comp_type

    return composite_types


def _query_views(conn, schema: str) -> dict[str, View]:
    """Query views from pg_views, plus their reloptions from pg_class.

    The definition still comes from `pg_views` so the comparison text is
    byte-identical to what it has always been (a change there would recreate
    every view in the database). `pg_views` has no reloptions column, so the
    `WITH (security_invoker = TRUE)` half of the declaration is joined in from
    `pg_class`.
    """
    views: dict[str, View] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                v.viewname AS view_name,
                v.definition AS view_definition,
                c.reloptions AS reloptions
            FROM pg_views AS v
            JOIN pg_namespace AS n ON n.nspname = v.schemaname
            JOIN pg_class AS c
              ON c.relname = v.viewname
             AND c.relnamespace = n.oid
             AND c.relkind = 'v'
            WHERE v.schemaname = %s
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_class'::regclass AND ext.objid = c.oid
                    AND ext.deptype = 'e'
              )
            ORDER BY v.viewname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            view = View(
                schema=schema,
                name=row["view_name"],
                definition=row["view_definition"] or "",
                reloptions=list(row["reloptions"] or []),
            )
            views[view.qualified_name] = view

    return views


def _query_materialized_views(conn, schema: str) -> dict[str, MaterializedView]:
    """Query materialized views from pg_matviews, plus storage options from pg_class."""
    matviews: dict[str, MaterializedView] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.relname AS matview_name,
                mv.definition AS matview_definition,
                c.reloptions AS reloptions,
                c.relispopulated AS is_populated
            FROM pg_matviews AS mv
            JOIN pg_namespace AS n ON n.nspname = mv.schemaname
            JOIN pg_class AS c
              ON c.relname = mv.matviewname
             AND c.relnamespace = n.oid
             AND c.relkind = 'm'
            WHERE mv.schemaname = %s
              -- Objects an extension owns belong to the extension, not the schema files.
              AND NOT EXISTS (
                  SELECT 1 FROM pg_depend ext
                  WHERE ext.classid = 'pg_class'::regclass AND ext.objid = c.oid
                    AND ext.deptype = 'e'
              )
            ORDER BY c.relname
            """,
            (schema,),
        )

        for row in cur.fetchall():
            matview = MaterializedView(
                schema=schema,
                name=row["matview_name"],
                definition=row["matview_definition"] or "",
                reloptions=list(row["reloptions"] or []),
                with_data=bool(row["is_populated"]),
            )
            matviews[matview.qualified_name] = matview

    return matviews


def _query_relation_dependents(conn, schemas: list[str]) -> dict[str, list[str]]:
    """Views and materialized views that read from another one, within `schemas`.

    A view's query is stored as a rewrite rule, and the rule depends on every
    relation the query reads. Only edges between views and materialized views
    are kept: those are the ones a drop-and-recreate has to carry along.
    """
    dependents: dict[str, list[str]] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT DISTINCT
                ref_ns.nspname AS ref_schema,
                ref.relname AS ref_name,
                dep_ns.nspname AS dep_schema,
                dep.relname AS dep_name
            FROM pg_depend AS d
            JOIN pg_rewrite AS r ON r.oid = d.objid
            JOIN pg_class AS dep ON dep.oid = r.ev_class
            JOIN pg_namespace AS dep_ns ON dep_ns.oid = dep.relnamespace
            JOIN pg_class AS ref ON ref.oid = d.refobjid
            JOIN pg_namespace AS ref_ns ON ref_ns.oid = ref.relnamespace
            WHERE d.classid = 'pg_rewrite'::regclass
              AND d.refclassid = 'pg_class'::regclass
              AND d.deptype = 'n'
              AND dep.oid <> ref.oid
              AND dep.relkind IN ('v', 'm')
              AND ref.relkind IN ('v', 'm')
              AND dep_ns.nspname = ANY(%s)
              AND ref_ns.nspname = ANY(%s)
            ORDER BY 1, 2, 3, 4
            """,
            (list(schemas), list(schemas)),
        )

        for row in cur.fetchall():
            key = f"{row['ref_schema']}.{row['ref_name']}"
            dependents.setdefault(key, []).append(
                f"{row['dep_schema']}.{row['dep_name']}"
            )

    return dependents


def _query_grants(conn, schema: str, grantee: str) -> list[Grant]:
    """Query grants for a specific role."""
    grants: list[Grant] = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Table grants (no column_name in role_table_grants)
        cur.execute(
            """
            SELECT
                c.relname AS object_name,
                'TABLE' AS object_type,
                rtg.privilege_type,
                rtg.grantee
            FROM information_schema.role_table_grants rtg
            JOIN pg_class c ON c.relname = rtg.table_name
            JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = rtg.table_schema
            WHERE rtg.grantee = %s
              AND rtg.table_schema = %s
            """,
            (grantee, schema),
        )

        for row in cur.fetchall():
            grants.append(
                Grant(
                    object_type=row["object_type"],
                    object_name=row["object_name"],
                    schema=schema,
                    privilege=row["privilege_type"],
                    grantee=row["grantee"],
                    column=None,
                )
            )

        # Build set of (table, privilege) pairs that have table-level grants.
        # When a table-level grant exists, PostgreSQL also reports per-column
        # entries in role_column_grants. We skip those to avoid redundant
        # column-level grants in the diff output.
        table_grant_keys = {(g.object_name, g.privilege) for g in grants}

        # Column grants (separate query)
        # Join with information_schema.columns to filter out grants for non-existent columns
        # (can happen if column was dropped but grant metadata is stale)
        cur.execute(
            """
            SELECT
                rcg.table_name AS object_name,
                'COLUMN' AS object_type,
                rcg.privilege_type,
                rcg.grantee,
                rcg.column_name
            FROM information_schema.role_column_grants rcg
            JOIN information_schema.columns c
              ON c.table_schema = rcg.table_schema
              AND c.table_name = rcg.table_name
              AND c.column_name = rcg.column_name
            WHERE rcg.grantee = %s
              AND rcg.table_schema = %s
            """,
            (grantee, schema),
        )

        for row in cur.fetchall():
            # Skip column-level grants that are implied by a table-level grant
            if (row["object_name"], row["privilege_type"]) in table_grant_keys:
                continue
            grants.append(
                Grant(
                    object_type=row["object_type"],
                    object_name=row["object_name"],
                    schema=schema,
                    privilege=row["privilege_type"],
                    grantee=row["grantee"],
                    column=row["column_name"],
                )
            )

        # Materialized view grants. information_schema.role_table_grants and
        # role_column_grants only cover tables, views and foreign tables, so
        # read the ACLs directly. acldefault() stands in for a NULL ACL, the
        # way information_schema does it, so an owning role reads the same.
        cur.execute(
            """
            SELECT c.relname AS object_name, a.privilege_type
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            CROSS JOIN LATERAL aclexplode(coalesce(c.relacl, acldefault('r', c.relowner))) a
            JOIN pg_roles g ON g.oid = a.grantee
            WHERE c.relkind = 'm' AND n.nspname = %s AND g.rolname = %s
            """,
            (schema, grantee),
        )
        matview_rows = cur.fetchall()
        for row in matview_rows:
            grants.append(
                Grant(
                    object_type="TABLE",
                    object_name=row["object_name"],
                    schema=schema,
                    privilege=row["privilege_type"],
                    grantee=grantee,
                    column=None,
                )
            )
        matview_grant_keys = {
            (row["object_name"], row["privilege_type"]) for row in matview_rows
        }
        cur.execute(
            """
            SELECT c.relname AS object_name, att.attname AS column_name,
                   a.privilege_type
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute att
              ON att.attrelid = c.oid AND att.attnum > 0 AND NOT att.attisdropped
            CROSS JOIN LATERAL aclexplode(att.attacl) a
            JOIN pg_roles g ON g.oid = a.grantee
            WHERE c.relkind = 'm' AND n.nspname = %s AND g.rolname = %s
            """,
            (schema, grantee),
        )
        for row in cur.fetchall():
            if (row["object_name"], row["privilege_type"]) in matview_grant_keys:
                continue
            grants.append(
                Grant(
                    object_type="COLUMN",
                    object_name=row["object_name"],
                    schema=schema,
                    privilege=row["privilege_type"],
                    grantee=grantee,
                    column=row["column_name"],
                )
            )

        # Function grants
        cur.execute(
            """
            SELECT
                routine_name AS object_name,
                'FUNCTION' AS object_type,
                privilege_type,
                grantee,
                NULL AS column_name
            FROM information_schema.role_routine_grants
            WHERE grantee = %s
              AND routine_schema = %s
            """,
            (grantee, schema),
        )

        for row in cur.fetchall():
            grants.append(
                Grant(
                    object_type=row["object_type"],
                    object_name=row["object_name"],
                    schema=schema,
                    privilege=row["privilege_type"],
                    grantee=row["grantee"],
                    column=None,
                )
            )

    return grants


def _query_schema_grants(conn, grantee: str) -> list[Grant]:
    """Query schema-level USAGE grants for a specific role.

    Queries all non-system schemas to detect grants like:
    GRANT USAGE ON SCHEMA extensions TO app_user;

    Uses has_schema_privilege() since information_schema.role_usage_grants
    doesn't reliably show schema grants.
    """
    grants: list[Grant] = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Query all non-system schemas where grantee has USAGE
        # Note: %% escapes % in psycopg2 parameterized queries
        cur.execute(
            """
            SELECT nspname AS schema_name
            FROM pg_namespace
            WHERE nspname NOT LIKE 'pg_%%'
              AND nspname != 'information_schema'
              AND has_schema_privilege(%s, nspname, 'USAGE')
            """,
            (grantee,),
        )
        for row in cur.fetchall():
            grants.append(
                Grant(
                    object_type="SCHEMA",
                    object_name=row["schema_name"],
                    schema=row["schema_name"],
                    privilege="USAGE",
                    grantee=grantee,
                    column=None,
                )
            )
    return grants


def _query_dependencies(conn, schemas: list[str]) -> list[Dependency]:
    """Query object dependencies from pg_depend."""
    dependencies: list[Dependency] = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Build schema filter

        cur.execute(
            """
            SELECT
                d.classid::regclass::text AS dependent_type,
                d.objid AS dependent_oid,
                d.refclassid::regclass::text AS referenced_type,
                d.refobjid AS referenced_oid,
                d.deptype AS dep_type
            FROM pg_depend d
            WHERE d.deptype IN ('n', 'a')
              AND d.classid != 'pg_class'::regclass
            LIMIT 10000
            """,
        )

        for row in cur.fetchall():
            dependencies.append(
                Dependency(
                    dependent_type=row["dependent_type"],
                    dependent_oid=row["dependent_oid"],
                    referenced_type=row["referenced_type"],
                    referenced_oid=row["referenced_oid"],
                    dep_type=row["dep_type"],
                )
            )

    return dependencies


def check_table_has_data(
    connection_string: str,
    schema: str,
    table: str,
    column: Optional[str] = None,
    config: Optional[PgDeclareConfig] = None,
) -> bool:
    """
    Check if a table or column contains data.

    Uses SET row_security = off to bypass RLS for accurate check.

    Args:
        connection_string: PostgreSQL connection string
        schema: Schema name
        table: Table name
        column: Optional column name (checks for non-NULL values)

    Returns:
        True if data exists
    """
    if resolve(config).dry_run:
        # In dry-run mode, assume no data to allow diff to proceed
        return False

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            # Disable RLS for this check
            cur.execute("SET row_security = off")

            if column:
                # Check if column has non-NULL values
                cur.execute(
                    f"SELECT 1 FROM {quote_ident(schema)}.{quote_ident(table)} WHERE {quote_ident(column)} IS NOT NULL LIMIT 1"
                )
            else:
                # Check if table has any rows
                cur.execute(
                    f"SELECT 1 FROM {quote_ident(schema)}.{quote_ident(table)} LIMIT 1"
                )

            return cur.fetchone() is not None


def get_invalid_indexes(
    connection_string: str, schema: str, config: Optional[PgDeclareConfig] = None
) -> list[str]:
    """
    Get list of invalid indexes in schema.

    Invalid indexes occur when CONCURRENTLY fails.

    Args:
        connection_string: PostgreSQL connection string
        schema: Schema name

    Returns:
        List of invalid index names
    """
    if resolve(config).dry_run:
        return []

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.indexname
                FROM pg_indexes i
                JOIN pg_class c ON c.relname = i.indexname
                JOIN pg_index ix ON ix.indexrelid = c.oid
                WHERE NOT ix.indisvalid
                  AND i.schemaname = %s
                """,
                (schema,),
            )
            return [row[0] for row in cur.fetchall()]


def _query_roles(conn, role_names: list[str]) -> dict[str, Role]:
    """
    Query role definitions from pg_roles.

    Only queries the specified managed roles, not all roles in the database.

    Args:
        conn: Database connection
        role_names: List of role names to query

    Returns:
        Dict of role name to Role object
    """
    roles: dict[str, Role] = {}

    if not role_names:
        return roles

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Build parameterized query for role names
        placeholders = ", ".join(["%s"] * len(role_names))
        cur.execute(
            f"""
            SELECT
                rolname AS name,
                rolcanlogin AS can_login,
                rolsuper AS superuser,
                rolcreatedb AS create_db,
                rolcreaterole AS create_role,
                rolinherit AS inherit,
                rolreplication AS replication,
                rolbypassrls AS bypass_rls,
                rolconnlimit AS connection_limit
            FROM pg_roles
            WHERE rolname IN ({placeholders})
            ORDER BY rolname
            """,
            tuple(role_names),
        )

        for row in cur.fetchall():
            role = Role(
                name=row["name"],
                can_login=row["can_login"],
                superuser=row["superuser"],
                create_db=row["create_db"],
                create_role=row["create_role"],
                inherit=row["inherit"],
                replication=row["replication"],
                bypass_rls=row["bypass_rls"],
                connection_limit=row["connection_limit"],
                # Note: password is not queried for security reasons
            )
            roles[role.name] = role

    return roles


def _query_role_memberships(conn, role_names: list[str]) -> list[RoleMembership]:
    """
    Query role memberships from pg_auth_members.

    Returns memberships where either the granted role or the member is in the
    managed roles list. This captures:
    - Roles granted TO managed roles (e.g., GRANT pg_read_all_data TO app_user)
    - Managed roles granted TO other roles (e.g., GRANT app_user TO admin)

    Args:
        conn: Database connection
        role_names: List of managed role names

    Returns:
        List of RoleMembership objects
    """
    memberships: list[RoleMembership] = []

    if not role_names:
        return memberships

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Build parameterized query for role names
        placeholders = ", ".join(["%s"] * len(role_names))
        cur.execute(
            f"""
            SELECT
                r.rolname AS role,
                m.rolname AS member,
                am.admin_option
            FROM pg_auth_members am
            JOIN pg_roles r ON r.oid = am.roleid
            JOIN pg_roles m ON m.oid = am.member
            WHERE r.rolname IN ({placeholders})
               OR m.rolname IN ({placeholders})
            ORDER BY r.rolname, m.rolname
            """,
            tuple(role_names) + tuple(role_names),
        )

        for row in cur.fetchall():
            membership = RoleMembership(
                role=row["role"],
                member=row["member"],
                admin_option=row["admin_option"],
            )
            memberships.append(membership)

    return memberships
