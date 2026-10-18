"""
Schema diff computation.

Compares desired schema state against actual database state and produces
a list of Change objects needed to synchronize them.
"""

from dataclasses import replace as dataclass_replace
from typing import Any, Callable, Optional, Union

from .changes import (
    AddColumn,
    AddConstraint,
    AddDomainConstraint,
    AddEnumValue,
    AddIdentity,
    AddPublicationObjects,
    AlterColumnDropExpression,
    AlterColumnSetExpression,
    AlterColumnType,
    AlterMaterializedViewOptions,
    AlterDomainDropDefault,
    AlterDomainDropNotNull,
    AlterDomainSetDefault,
    AlterDomainSetNotNull,
    AlterEventTriggerEnabled,
    AlterPublicationOptions,
    AlterRuleEnabled,
    AlterExtensionSchema,
    AlterExtensionUpdate,
    AlterForeignServer,
    AlterForeignTable,
    AlterOwner,
    AlterRole,
    AlterSequence,
    AlterUserMapping,
    AlterViewOptions,
    AttachPartition,
    Change,
    CreateCollation,
    CreateCompositeType,
    CreateDomain,
    CreateEnum,
    CreateEventTrigger,
    CreateExtension,
    CreateForeignServer,
    CreateForeignTable,
    CreateFunction,
    CreateIndex,
    CreateMaterializedView,
    CreatePartitionedIndex,
    CreatePolicy,
    CreateRangeType,
    CreatePublication,
    CreateRole,
    CreateRule,
    CreateSequence,
    CreateShellType,
    CreateTable,
    CreateTrigger,
    CreateUserMapping,
    CreateView,
    DEFERRED_CONSTRAINT_TYPES,
    DisableRLS,
    DropAndCreateConstraint,
    DependentRelation,
    DropAndCreateCollation,
    DetachPartition,
    DropAndCreateDomain,
    DropAndCreateDomainConstraint,
    DropAndCreateForeignTable,
    DropAndCreateFunction,
    DropAndCreateMaterializedView,
    DropAndCreatePolicy,
    DropAndCreateRangeType,
    DropAndCreatePublication,
    DropAndCreateView,
    DropCollation,
    DropColumn,
    DropConstraint,
    DropDefault,
    DropDomain,
    DropDomainConstraint,
    DropEventTrigger,
    DropExtension,
    DropForeignTable,
    DropFunction,
    DropIdentity,
    DropIndex,
    DropMaterializedView,
    DropNotNull,
    DropPolicy,
    DropRangeType,
    DropPublication,
    DropPublicationObjects,
    DropRule,
    DropSequence,
    DropTable,
    DropTrigger,
    DropUserMapping,
    DropView,
    EnableRLS,
    ForceRLS,
    GrantDefaultPrivilege,
    GrantPrivilege,
    GrantRoleMembership,
    NoForceRLS,
    ReplaceFunction,
    RevokeDefaultPrivilege,
    RevokePrivilege,
    RevokePublicExecute,
    ReplacePublicationTable,
    RevokeRoleMembership,
    SetComment,
    SetDefault,
    SetSequenceOwner,
    SetNotNull,
    ValidateConstraint,
    dependents_note,
    foreign_column_sql,
    ValidateDomainConstraint,
    grant_replay_note,
    index_rebuild_note,
)
from .config import PgDeclareConfig, resolve
from .quoting import quote_ident
from .models import (
    ROUTINE_OWNER_KINDS,
    ChangeCategory,
    Column,
    Comment,
    Domain,
    ForeignTable,
    Function,
    Grant,
    Index,
    MaterializedView,
    RangeType,
    Publication,
    Schema,
    Table,
    USER_MAPPING_PUBLIC_OPTIONS,
    View,
    alter_fdw_options_sql,
)


def compute_diff(
    desired: Schema,
    actual: Schema,
    check_data_fn: Optional[Callable[..., bool]] = None,
    schemas: list[str] | None = None,
    locked_functions: Optional[set[str]] = None,
    config: Optional[PgDeclareConfig] = None,
) -> list[Change]:
    """
    Compare desired schema state against actual database state.

    Args:
        desired: Schema object representing desired state (from SQL files)
        actual: Schema object representing current database state
        check_data_fn: Optional function to check if table/column has data
                      Signature: check_data_fn(schema, table, column=None) -> bool
        schemas: Optional list of managed schema names (e.g. ["public", "app_generated"]).
                 Used to strip schema qualifiers from types during comparison,
                 preventing false-positive diffs when format_type() disambiguates.
        locked_functions: Optional set of lowercased function names (SECDEF, in
                 one of `config.secdef_schemas`) that must NOT retain PUBLIC
                 EXECUTE. Computed by the caller, which decides which
                 functions are meant to be callable by everyone. Defaults to
                 empty (no PUBLIC-revoke behavior).

    Returns:
        List of Change objects needed to transform actual into desired
    """
    changes: list[Change] = []

    # Partitions created at runtime (pg_partman, a cron job) and not declared
    # are never touched: they leave the comparison here, with everything on
    # them. partition_notes() reports them.
    parents_with_default = {
        t.parent_key for t in actual.tables.values() if t.partition_bound == "DEFAULT"
    }
    actual = _without_undeclared_partitions(desired, actual)
    # An owner's privileges are implicit, but read back as grants to it. Where
    # the schema files state the owner, those are the owner's, not grants to
    # diff: `_diff_owners` moves them with the object.
    owner_grants = _stated_owner_grants(desired)
    if owner_grants:
        desired = dataclass_replace(
            desired,
            grants=[
                g for g in desired.grants if not _is_stated_owner_grant(g, owner_grants)
            ],
        )
        actual = dataclass_replace(
            actual,
            grants=[
                g for g in actual.grants if not _is_stated_owner_grant(g, owner_grants)
            ],
        )

    # Diff roles first (cluster-wide, needed before schemas)
    changes.extend(_diff_roles(desired, actual))
    changes.extend(_diff_role_memberships(desired, actual))

    # Default privileges no longer declared are revoked before anything is
    # created, and new ones granted after the object grants. An object created
    # in this run then gets exactly the grants the diff gives it: the plan
    # database built it with the declared defaults already in force (see
    # schema_parser), so its grants are in the desired schema either way.
    default_privilege_revokes, default_privilege_grants = _diff_default_privileges(
        desired, actual
    )
    changes.extend(default_privilege_revokes)

    # Extensions, then foreign servers and user mappings: types, functions and
    # foreign tables need them. All are database-wide, not per schema.
    changes.extend(_diff_extensions(desired, actual))
    changes.extend(_diff_foreign_servers(desired, actual))
    changes.extend(_diff_user_mappings(desired, actual))

    # Diff each object type
    # Order matters for dependency resolution:
    # - Enums before composite types (composite types may reference enums)
    # - Collations before domains, ranges and tables, which may use them;
    #   collation drops last, after everything that could still use one
    # - Domains before tables (columns may be typed with a domain)
    # - Range types after domains and before composite types and tables
    #   (a range may be over a domain; a column or attribute may be a range);
    #   range drops after functions, like domain drops
    # - Composite types before functions (functions may use composite types)
    # - Views before functions (functions may SELECT from views, so views
    #   must be updated first when column types change)
    # - Domain drops after tables, views and functions, which drop or retype
    #   whatever still uses the domain
    # - Sequences before tables (a column default calls nextval() on one);
    #   OWNED BY and sequence drops after tables and views (OWNED BY needs the
    #   column, and a drop must follow the removal of any default using it)
    changes.extend(_diff_enums(desired, actual))
    collations_before, collations_after = _diff_collations(desired, actual)
    changes.extend(collations_before)
    changes.extend(_diff_domains(desired, actual, managed_schemas=schemas))
    changes.extend(_diff_ranges(desired, actual, managed_schemas=schemas))
    changes.extend(_diff_composite_types(desired, actual))
    sequences_before, sequences_after = _diff_sequences(desired, actual)
    changes.extend(sequences_before)
    changes.extend(
        _diff_tables(
            desired,
            actual,
            check_data_fn,
            managed_schemas=schemas,
            parents_with_default=parents_with_default,
        )
    )
    changes.extend(_diff_foreign_tables(desired, actual, managed_schemas=schemas))
    changes.extend(_diff_views(desired, actual))
    changes.extend(_diff_materialized_views(desired, actual))
    changes.extend(sequences_after)
    changes.extend(_diff_functions(desired, actual))
    changes.extend(_drop_ranges(desired, actual))
    changes.extend(_drop_domains(desired, actual))
    changes.extend(collations_after)
    changes.extend(_drop_extensions(desired, actual))
    changes.extend(_diff_indexes(desired, actual))
    changes.extend(_diff_policies(desired, actual))
    changes.extend(_diff_triggers(desired, actual))
    changes.extend(_diff_rules(desired, actual))
    managed = list(schemas) if schemas is not None else list(resolve(config).schemas)
    changes.extend(_diff_publications(desired, actual, managed))
    # Event triggers last among the DDL: one created here fires on every DDL
    # statement after it.
    changes.extend(_diff_event_triggers(desired, actual, managed, changes))
    changes.extend(_diff_grants(desired, actual))
    changes.extend(default_privilege_grants)
    changes = _skip_revokes_on_dropped_relations(changes)
    changes = _skip_canonical_function_drops(changes, actual)

    # Re-apply grants for objects being dropped and recreated.
    #
    # Drop-and-recreate strips all grants, and _diff_grants compares against the
    # DB snapshot taken before changes execute, so it never sees the loss. This
    # used to append a standalone GrantPrivilege after the recreate, which is not
    # enough: the executor runs changes retry-until-stable, so a recreate whose
    # new definition depends on an object built later in the same run is deferred
    # to a later iteration. The GRANT then lands on the pre-drop view and the
    # recreate throws it away, leaving the view owner-only until the next
    # converge.
    #
    # The grants are therefore carried inside the recreate's own DO block, the
    # same way the SECDEF REVOKE is atomic with the function recreate. Object
    # and ACL are one unit, so no execution order can separate them.
    #
    # A recreated view or materialized view also takes its dependents down with
    # it; those ride inside the same statement too (_carry_recreate_dependents).
    changes = _carry_recreate_dependents(changes, desired, actual)
    recreations: dict[
        tuple[str, str],
        list[
            Union[
                DropAndCreateView,
                DropAndCreateFunction,
                DropAndCreateMaterializedView,
                DropAndCreateForeignTable,
            ]
        ],
    ] = {}
    for c in changes:
        if isinstance(c, DropAndCreateView):
            recreations.setdefault((c.view.schema, c.view.name), []).append(c)
        elif isinstance(c, DropAndCreateFunction):
            recreations.setdefault((c.function.schema, c.function.name), []).append(c)
        elif isinstance(c, DropAndCreateMaterializedView):
            recreations.setdefault((c.matview.schema, c.matview.name), []).append(c)
        elif isinstance(c, DropAndCreateForeignTable):
            recreations.setdefault((c.table.schema, c.table.name), []).append(c)
    if recreations:
        for grant in desired.grants:
            for recreate in recreations.get((grant.schema, grant.object_name), []):
                if isinstance(
                    recreate, (DropAndCreateMaterializedView, DropAndCreateForeignTable)
                ) and not _is_relation_grant(grant):
                    continue
                if (
                    isinstance(recreate, DropAndCreateFunction)
                    and grant.args is not None
                    and grant.args != recreate.function.args
                ):
                    continue  # another overload of the same name
                recreate.grants.append(grant)
        # Say so in the plan, so a recreate's ACL work is visible.
        for recreated in recreations.values():
            for recreate in recreated:
                recreate.description += grant_replay_note(len(recreate.grants))

    # Revoke PUBLIC EXECUTE on locked SECDEF functions. Runs after the
    # function diff so it can flag the relevant create/recreate changes, and
    # after grants so standalone revokes land in the grant phase.
    changes.extend(
        _diff_function_public_execute(
            desired,
            actual,
            changes,
            locked_functions,
            resolve(config).secdef_schemas,
        )
    )

    # Schedule ValidateConstraint for any constraints added with NOT VALID.
    # This ensures FK/CHECK constraints are validated after the DDL phase.
    #
    # DropAndCreateConstraint (a tightened CHECK, say) re-adds NOT VALID too, so
    # it needs the same follow-up. Without it the constraint stays NOT VALID
    # forever, pg_get_constraintdef keeps reporting the ' NOT VALID' suffix, and
    # the next diff sees drift and re-emits the same drop/recreate on every
    # deploy.
    validation_changes: list[Change] = []
    for change in changes:
        # Only FK ('f') and CHECK ('c') constraints use NOT VALID
        if isinstance(change, AddConstraint):
            needs_validation = change.not_valid and change.constraint.type in (
                "f",
                "c",
            )
        elif isinstance(change, DropAndCreateConstraint):
            needs_validation = change.not_valid and change.constraint.type in (
                "f",
                "c",
            )
        else:
            continue

        if needs_validation:
            validation_changes.append(
                ValidateConstraint(
                    schema=change.schema,
                    table=change.table,
                    constraint_name=change.constraint.name,
                )
            )
    # Domain CHECK constraints are always added NOT VALID, for the same reason.
    for change in changes:
        if isinstance(change, (AddDomainConstraint, DropAndCreateDomainConstraint)):
            validation_changes.append(
                ValidateDomainConstraint(
                    schema=change.schema,
                    domain=change.domain,
                    constraint_name=change.constraint.name,
                )
            )
    changes.extend(validation_changes)

    # Owners, like comments, after everything else: a recreate resets the
    # owner, so the diff needs the full change list (dependents included).
    changes.extend(_diff_owners(desired, actual, changes))

    # Comments last: a recreate this run wipes the object's comment, so the
    # diff needs the full change list to know which ones to write again.
    changes.extend(_diff_comments(desired, actual, changes))

    return changes


def undeclared_partitions(desired: Schema, actual: Schema) -> list[Table]:
    """Live partitions the schema files do not declare, sub-partitions included.

    Applications often create partitions at runtime (pg_partman, a cron job).
    Such a partition is never dropped or altered; the diff leaves it, and
    everything on it, out of the comparison.
    """
    return [
        table
        for key, table in sorted(actual.tables.items())
        if table.partition_of is not None and key not in desired.tables
    ]


def partition_notes(desired: Schema, actual: Schema) -> list[str]:
    """One line per undeclared live partition of a declared table, for the output.

    These are notes, not changes: the partition is left in place. Partitions of
    a table that is itself going away are not mentioned; its drop takes them.
    """

    def root(table: Table) -> str:
        seen: set[str] = set()
        while table.parent_key is not None and table.parent_key not in seen:
            seen.add(table.parent_key)
            parent = actual.tables.get(table.parent_key)
            if parent is None:
                return table.parent_key
            table = parent
        return table.qualified_name

    return [
        f"Partition {table.qualified_name} of {table.parent_key} "
        f"({table.partition_bound}) is not declared; left in place."
        for table in undeclared_partitions(desired, actual)
        if root(table) in desired.tables
    ]


def _without_undeclared_partitions(desired: Schema, actual: Schema) -> Schema:
    """`actual` without the undeclared partitions and what sits on them."""
    gone = {t.qualified_name for t in undeclared_partitions(desired, actual)}
    if not gone:
        return actual

    def on_gone(schema: str, relation: str) -> bool:
        return f"{schema}.{relation}" in gone

    indexes = {
        k: i for k, i in actual.indexes.items() if not on_gone(i.schema, i.table)
    }
    gone_indexes = actual.indexes.keys() - indexes.keys()

    def keep_comment(c: Comment) -> bool:
        if c.object_type == "INDEX":
            return f"{c.schema}.{c.name}" not in gone_indexes
        if c.object_type in ("TABLE", "COLUMN", "CONSTRAINT", "TRIGGER", "POLICY"):
            return not on_gone(c.schema, c.name)
        return True

    return dataclass_replace(
        actual,
        tables={k: t for k, t in actual.tables.items() if k not in gone},
        indexes=indexes,
        policies={
            k: p for k, p in actual.policies.items() if not on_gone(p.schema, p.table)
        },
        triggers={
            k: t for k, t in actual.triggers.items() if not on_gone(t.schema, t.table)
        },
        grants=[
            g
            for g in actual.grants
            if not (_is_relation_grant(g) and on_gone(g.schema, g.object_name))
        ],
        comments={k: c for k, c in actual.comments.items() if keep_comment(c)},
    )


def _skip_revokes_on_dropped_relations(changes: list[Change]) -> list[Change]:
    """Leave out revokes on a table, view or materialized view this run drops.

    The DROP takes the ACL with it. A REVOKE ordered after it would fail on the
    missing relation and stay deferred until the run reported itself stuck.
    """
    dropped: set[tuple[str, str]] = set()
    for c in changes:
        if isinstance(c, DropMaterializedView):
            dropped.add((c.schema, c.matview_name))
        elif isinstance(c, DropView):
            dropped.add((c.schema, c.view_name))
        elif isinstance(c, (DropTable, DropForeignTable)):
            dropped.add((c.schema, c.table_name))
    if not dropped:
        return changes
    return [
        c
        for c in changes
        if not (
            isinstance(c, RevokePrivilege)
            and _is_relation_grant(c.grant)
            and (c.grant.schema, c.grant.object_name) in dropped
        )
    ]


def _is_relation_grant(grant: Grant) -> bool:
    """A grant on a table-like object (table, view, materialized view)."""
    return grant.object_type in ("TABLE", "COLUMN")


def _relation_dependent_order(root: str, dependents: dict[str, list[str]]) -> list[str]:
    """Every view or materialized view reading from `root`, directly or not.

    Returned in creation order: each one after everything it reads from, so
    dropping in reverse and creating in order never trips over a dependency.
    """
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        for dependent in dependents.get(name, []):
            if dependent not in seen:
                seen.add(dependent)
                visit(dependent)
                order.append(dependent)

    visit(root)
    # Post-order puts the deepest dependents first; reversed, each relation
    # precedes everything that reads from it.
    order.reverse()
    return order


def _carry_recreate_dependents(
    changes: list[Change], desired: Schema, actual: Schema
) -> list[Change]:
    """Fold a recreated relation's dependents, and a matview's indexes, into it.

    DROP VIEW / DROP MATERIALIZED VIEW refuse while another view reads from the
    object, and a materialized view's DROP takes its indexes along. The diff is
    computed against the snapshot taken before the run, so it never sees
    either loss. Each recreate therefore drops its dependents first and builds
    them, their grants and their indexes back, inside its own statement.

    A dependent folded into a recreate no longer needs its own recreate,
    options change or index changes: the recreate builds it from its
    declaration, so those changes are removed.
    """
    roots = [
        c
        for c in changes
        if isinstance(c, (DropAndCreateView, DropAndCreateMaterializedView))
    ]
    if not roots:
        return changes

    def root_name(c: Union[DropAndCreateView, DropAndCreateMaterializedView]) -> str:
        return (
            c.view.qualified_name
            if isinstance(c, DropAndCreateView)
            else c.matview.qualified_name
        )

    def relation_grants(schema: str, name: str) -> list[Grant]:
        return [
            g
            for g in desired.grants
            if g.schema == schema and g.object_name == name and _is_relation_grant(g)
        ]

    def relation_indexes(schema: str, name: str) -> list[Index]:
        return [
            i
            for i in desired.indexes.values()
            if i.schema == schema and i.table == name
        ]

    plans: dict[str, list[str]] = {
        root_name(c): _relation_dependent_order(
            root_name(c), actual.relation_dependents
        )
        for c in roots
    }
    folded: set[str] = {name for order in plans.values() for name in order}

    # Relations whose indexes a recreate rebuilds: a recreated matview, and
    # every still-declared matview dependent.
    rebuilt_indexes: set[str] = set()

    kept: list[Change] = []
    for c in changes:
        if isinstance(c, (DropAndCreateView, DropAndCreateMaterializedView)):
            name = root_name(c)
            if name in folded:
                continue  # Another recreate rebuilds it.
            dependents: list[DependentRelation] = []
            for dep_name in plans[name]:
                matview = actual.materialized_views.get(dep_name)
                view = actual.views.get(dep_name)
                existing = matview if matview is not None else view
                if existing is None:
                    # Outside the managed schemas: it cannot be rebuilt, so
                    # leave it in place and let the DROP report it.
                    continue
                relation: Optional[Union[View, MaterializedView]] = (
                    desired.materialized_views.get(dep_name)
                    or desired.views.get(dep_name)
                )
                dependent = DependentRelation(
                    schema=existing.schema,
                    name=existing.name,
                    kind="m" if matview is not None else "v",
                    relation=relation,
                )
                if relation is not None:
                    dependent.grants = relation_grants(relation.schema, relation.name)
                    if isinstance(relation, MaterializedView):
                        dependent.indexes = relation_indexes(
                            relation.schema, relation.name
                        )
                        rebuilt_indexes.add(relation.qualified_name)
                dependents.append(dependent)
            c.dependents = dependents
            if isinstance(c, DropAndCreateMaterializedView):
                c.indexes = relation_indexes(c.matview.schema, c.matview.name)
                rebuilt_indexes.add(name)
                c.description += index_rebuild_note(len(c.indexes))
            if any(d.rebuilds_data for d in dependents):
                # A view recreate that rebuilds a materialized view's data holds
                # that lock too.
                c.category = ChangeCategory.OFFLINE
            c.description += dependents_note(dependents)
            kept.append(c)
        elif (
            isinstance(
                c,
                (
                    AlterViewOptions,
                    AlterMaterializedViewOptions,
                ),
            )
            and c.object_name in folded
        ):
            continue
        else:
            kept.append(c)

    if not rebuilt_indexes:
        return kept

    def index_relation(c: Change) -> Optional[str]:
        if isinstance(c, CreateIndex):
            return f"{c.index.schema}.{c.index.table}"
        if isinstance(c, DropIndex):
            index = actual.indexes.get(f"{c.schema}.{c.index_name}")
            return f"{index.schema}.{index.table}" if index is not None else None
        return None

    return [c for c in kept if index_relation(c) not in rebuilt_indexes]


def _diff_function_public_execute(
    desired: Schema,
    actual: Schema,
    changes: list[Change],
    locked_functions: Optional[set[str]],
    secdef_schemas: tuple[str, ...] = ("public",),
) -> list[Change]:
    """Revoke PUBLIC EXECUTE on locked SECDEF functions.

    `locked_functions` is the set of lowercased SECDEF function names, in one of
    `secdef_schemas`,
    names that must not retain PUBLIC EXECUTE (the caller decides which). For
    each desired function whose
    name is locked:

    - If it is being created or recreated this run, flag that change so the
      REVOKE is appended atomically (PostgreSQL grants PUBLIC EXECUTE at CREATE
      time; an atomic revoke closes the deferred-create/recreate clobber).
    - Else, if the live function still has PUBLIC EXECUTE (drift), emit a
      standalone RevokePublicExecute.
    - Else (already converged, no recreate) emit nothing -> clean diff.
    """
    if not locked_functions:
        return []

    extra: list[Change] = []

    # Index the create/recreate changes by full signature so a locked function
    # gets its REVOKE appended atomically instead of as a standalone statement.
    # Keying by name alone would let an overload (or an aggregate) of the same
    # name absorb the flag, and the locked routine would keep PUBLIC EXECUTE.
    create_changes: dict[str, Union[CreateFunction, DropAndCreateFunction]] = {}
    for c in changes:
        if isinstance(c, (CreateFunction, DropAndCreateFunction)):
            create_changes[c.function.signature] = c

    for func in desired.functions.values():
        if not func.security_definer or func.schema not in secdef_schemas:
            continue
        if func.name.lower() not in locked_functions:
            continue

        recreate = create_changes.get(func.signature)
        if recreate is not None:
            # Atomic path: append the REVOKE to the (re)creation change.
            recreate.revoke_public_execute = True
            continue

        # Drift path: revoke only if the live function still has PUBLIC EXECUTE.
        actual_func = actual.functions.get(func.signature)
        if actual_func is not None and actual_func.public_execute:
            extra.append(
                RevokePublicExecute(
                    schema=func.schema,
                    name=func.name,
                    args=func.args,
                    keyword=func.keyword,
                )
            )

    return extra


def _recreated_objects(changes: list[Change]) -> tuple[set[str], set[str]]:
    """Comment keys of the objects this run creates or drops and recreates.

    Returns `(objects, relations)`: the `Comment.key` of every object built
    fresh, and the `schema.name` of every table or view built fresh, whose
    column comments are therefore gone too.
    """
    objects: set[str] = set()
    relations: set[str] = set()

    def mark(kind: str, schema: str, name: str, member: str = "", args: str = ""):
        probe = Comment(kind, schema, name, "", member=member, args=args)
        objects.add(probe.key)

    for c in changes:
        if isinstance(c, CreateTable):
            mark("TABLE", c.table.schema, c.table.name)
            relations.add(c.table.qualified_name)
            for constraint in c.table.constraints:
                mark("CONSTRAINT", c.table.schema, c.table.name, constraint.name)
        elif isinstance(c, (CreateView, DropAndCreateView)):
            mark("VIEW", c.view.schema, c.view.name)
            relations.add(c.view.qualified_name)
        elif isinstance(c, AddColumn):
            mark("COLUMN", c.schema, c.table, c.column.name)
        elif isinstance(c, (AddConstraint, DropAndCreateConstraint)):
            mark("CONSTRAINT", c.schema, c.table, c.constraint.name)
        elif isinstance(c, CreateIndex):
            mark("INDEX", c.index.schema, c.index.name)
        elif isinstance(c, (CreateFunction, DropAndCreateFunction)):
            f = c.function
            mark(f.keyword, f.schema, f.name, args=f.args)
        elif isinstance(c, CreateTrigger):
            mark("TRIGGER", c.trigger.schema, c.trigger.table, c.trigger.name)
        elif isinstance(c, (CreatePolicy, DropAndCreatePolicy)):
            p = c.policy
            mark("POLICY", p.schema, p.table, p.name)
        elif isinstance(c, CreateEnum):
            mark("TYPE", c.enum.schema, c.enum.name)
        elif isinstance(c, CreateCompositeType):
            mark("TYPE", c.composite_type.schema, c.composite_type.name)
        elif isinstance(c, (CreateRangeType, DropAndCreateRangeType)):
            mark("TYPE", c.range_type.schema, c.range_type.name)
        elif isinstance(c, (CreateCollation, DropAndCreateCollation)):
            mark("COLLATION", c.collation.schema, c.collation.name)
    return objects, relations


def _diff_comments(
    desired: Schema, actual: Schema, changes: list[Change]
) -> list[Change]:
    """Diff `COMMENT ON` text between desired and actual schemas.

    A declared comment is written when the live text differs, and also when
    the object is created or recreated this run: a DROP takes the comment with
    it, and the snapshot in `actual` predates the drop. A comment no longer
    declared is removed with `IS NULL`, unless its object is going away or
    being rebuilt anyway.
    """
    recreated, recreated_relations = _recreated_objects(changes)

    def rebuilt(comment: Comment) -> bool:
        if comment.key in recreated:
            return True
        return (
            comment.object_type == "COLUMN"
            and f"{comment.schema}.{comment.name}" in recreated_relations
        )

    result: list[Change] = []
    for key in sorted(desired.comments):
        comment = desired.comments[key]
        current = actual.comments.get(key)
        if current is None or current.text != comment.text or rebuilt(comment):
            result.append(SetComment(comment=comment))

    for key in sorted(actual.comments.keys() - desired.comments.keys()):
        comment = actual.comments[key]
        if desired.has_comment_target(comment) and not rebuilt(comment):
            result.append(SetComment(comment=comment, remove=True))

    return result


def _diff_tables(
    desired: Schema,
    actual: Schema,
    check_data_fn: Optional[Callable[..., bool]] = None,
    managed_schemas: list[str] | None = None,
    parents_with_default: Optional[set[Optional[str]]] = None,
) -> list[Change]:
    """Diff tables between desired and actual schemas.

    A partition's columns belong to its parent: adding, dropping or retyping
    one happens on the parent and Postgres carries it to every partition. Only
    a partition's own defaults and NOT NULL are diffed, after everything else,
    since the parent's changes recurse into its partitions first.
    """
    changes: list[Change] = []
    partition_columns: list[Change] = []
    parents_with_default = parents_with_default or set()

    # Tables to create
    for key, table in desired.tables.items():
        if key not in actual.tables:
            create = CreateTable(table=table)
            if (
                table.partition_of is not None
                and table.partition_bound != "DEFAULT"
                and table.parent_key in parents_with_default
            ):
                # Postgres scans the default partition for rows that now
                # belong to the new one, under a lock.
                create.category = ChangeCategory.OFFLINE
                create.description += " (scans the default partition)"
            changes.append(create)
            if table.partition_of is not None:
                partition_columns.extend(
                    _diff_partition_columns(table, None, desired, actual)
                )
            # CreateTable inlines only PRIMARY KEY and CHECK. Everything in
            # DEFERRED_CONSTRAINT_TYPES is added here instead: FOREIGN KEY so
            # circular dependencies resolve (post <-> comment), UNIQUE and
            # EXCLUDE because an inline index constraint that duplicates another
            # one's column set is silently discarded by PostgreSQL.
            for constraint in table.constraints:
                if constraint.type in DEFERRED_CONSTRAINT_TYPES:
                    changes.append(
                        AddConstraint(
                            schema=table.schema,
                            table=table.name,
                            constraint=constraint,
                            not_valid=False,  # New tables are empty, no validation needed
                        )
                    )
            # Enable RLS on new table if desired
            if table.rls_enabled:
                changes.append(EnableRLS(schema=table.schema, table=table.name))
            if table.rls_forced:
                changes.append(ForceRLS(schema=table.schema, table=table.name))
        else:
            # Table exists - diff partitioning, columns, constraints, RLS state
            actual_table = actual.tables[key]
            _check_partitioning(table, actual_table)
            if table.partition_of is not None:
                assert table.partition_bound is not None
                if (actual_table.partition_of, actual_table.partition_bound) != (
                    table.partition_of,
                    table.partition_bound,
                ):
                    changes.append(
                        AttachPartition(
                            schema=table.schema,
                            table=table.name,
                            parent=table.partition_of,
                            bound=table.partition_bound,
                            detach_from=actual_table.partition_of,
                            old_bound=actual_table.partition_bound,
                        )
                    )
                partition_columns.extend(
                    _diff_partition_columns(table, actual_table, desired, actual)
                )
            else:
                if actual_table.partition_of is not None:
                    changes.append(
                        DetachPartition(
                            schema=table.schema,
                            table=table.name,
                            parent=actual_table.partition_of,
                        )
                    )
                changes.extend(
                    _diff_columns(table, actual_table, check_data_fn, managed_schemas)
                )
            changes.extend(_diff_constraints(table, actual_table))
            changes.extend(_diff_rls(table, actual_table))

    changes.extend(partition_columns)

    # Tables to drop
    for key, table in actual.tables.items():
        if key not in desired.tables:
            has_data = False
            if check_data_fn:
                has_data = check_data_fn(table.schema, table.name)
            changes.append(
                DropTable(
                    schema=table.schema,
                    table_name=table.name,
                    has_data=has_data,
                )
            )

    return changes


def _check_partitioning(desired: Table, actual: Table) -> None:
    """Refuse what Postgres cannot do in place: a partitioning change of a table."""
    name = f"{quote_ident(desired.schema)}.{quote_ident(desired.name)}"
    how = (
        "Do it in an explicit @migrate-before block: create the new table, "
        "copy the rows (INSERT ... SELECT), drop the old one and rename the new "
        "one into place."
    )
    if desired.partition_key == actual.partition_key:
        return
    if actual.partition_key is None:
        raise ValueError(
            f"cannot convert table {name} into a partitioned table "
            f"(PARTITION BY {desired.partition_key}): PostgreSQL has no ALTER "
            f"for it. {how}"
        )
    if desired.partition_key is None:
        raise ValueError(
            f"cannot convert partitioned table {name} (PARTITION BY "
            f"{actual.partition_key}) into a plain table: PostgreSQL has no "
            f"ALTER for it. {how}"
        )
    raise ValueError(
        f"cannot change the partition key of {name} from PARTITION BY "
        f"{actual.partition_key} to PARTITION BY {desired.partition_key}: "
        f"PostgreSQL has no ALTER for it. {how}"
    )


def _diff_partition_columns(
    desired_child: Table,
    actual_child: Optional[Table],
    desired: Schema,
    actual: Schema,
) -> list[Change]:
    """A partition's own column defaults and NOT NULL.

    A partition takes its columns, defaults and NOT NULL from its parent when
    it is created, and a parent's ALTER COLUMN ... SET/DROP DEFAULT or NOT NULL
    recurses into it, overwriting a partition's own value. So the value a
    partition holds once the parent's changes have run is the parent's
    declared one, unless the parent's value is unchanged and the partition is
    already attached to it, in which case it keeps its own. Changes are
    emitted where that differs from the declaration.
    """
    parent_key = desired_child.parent_key
    desired_parent = desired.tables.get(parent_key) if parent_key else None
    if desired_parent is None:
        return []
    actual_parent = actual.tables.get(parent_key) if parent_key else None
    attached = (
        actual_child is not None
        and actual_child.partition_of == desired_child.partition_of
    )
    schema, table = desired_child.schema, desired_child.name
    category = ChangeCategory.ONLINE if actual_child is None else ChangeCategory.OFFLINE
    changes: list[Change] = []

    for name, col in desired_child.columns.items():
        parent_col = desired_parent.columns.get(name)
        if parent_col is None or col.identity or col.generated:
            continue
        current = actual_child.columns.get(name) if actual_child else None
        actual_parent_col = actual_parent.columns.get(name) if actual_parent else None

        def held(attr: str, normalize: Callable[[Any], Any]) -> Any:
            # What the partition holds after the parent's own changes.
            assert parent_col is not None
            if (
                attached
                and current is not None
                and actual_parent_col is not None
                and normalize(getattr(parent_col, attr))
                == normalize(getattr(actual_parent_col, attr))
            ):
                return getattr(current, attr)
            return getattr(parent_col, attr)

        nullable = held("nullable", bool)
        if col.nullable != nullable:
            if col.nullable:
                changes.append(DropNotNull(schema=schema, table=table, column=name))
            else:
                changes.append(
                    SetNotNull(
                        schema=schema, table=table, column=name, category=category
                    )
                )

        default = held("default", _normalize_default)
        if _normalize_default(col.default) != _normalize_default(default):
            if col.default is None:
                changes.append(DropDefault(schema=schema, table=table, column=name))
            else:
                changes.append(
                    SetDefault(
                        schema=schema,
                        table=table,
                        column=name,
                        default_value=col.default,
                    )
                )

    return changes


def _diff_rls(desired_table: Table, actual_table: Table) -> list[Change]:
    """Diff RLS state between desired and actual table."""
    changes: list[Change] = []

    # Enable/Disable RLS
    if desired_table.rls_enabled and not actual_table.rls_enabled:
        changes.append(EnableRLS(schema=desired_table.schema, table=desired_table.name))
    elif not desired_table.rls_enabled and actual_table.rls_enabled:
        changes.append(
            DisableRLS(schema=desired_table.schema, table=desired_table.name)
        )

    # Force/No Force RLS (only meaningful if RLS is enabled)
    if desired_table.rls_forced and not actual_table.rls_forced:
        changes.append(ForceRLS(schema=desired_table.schema, table=desired_table.name))
    elif not desired_table.rls_forced and actual_table.rls_forced:
        changes.append(
            NoForceRLS(schema=desired_table.schema, table=desired_table.name)
        )

    return changes


def _diff_columns(
    desired_table: Table,
    actual_table: Table,
    check_data_fn: Optional[Callable[..., bool]] = None,
    managed_schemas: list[str] | None = None,
) -> list[Change]:
    """Diff columns between desired and actual table."""
    changes: list[Change] = []
    schema = desired_table.schema
    table_name = desired_table.name

    # Columns to add
    for col_name, col in desired_table.columns.items():
        if col_name not in actual_table.columns:
            changes.append(
                AddColumn(
                    schema=schema,
                    table=table_name,
                    column=col,
                )
            )
        else:
            # Column exists - check for changes
            actual_col = actual_table.columns[col_name]
            changes.extend(
                _diff_column_properties(
                    schema, table_name, col, actual_col, managed_schemas
                )
            )

    # Columns to drop
    for col_name, col in actual_table.columns.items():
        if col_name not in desired_table.columns:
            has_data = False
            if check_data_fn:
                has_data = check_data_fn(schema, table_name, col_name)
            changes.append(
                DropColumn(
                    schema=schema,
                    table=table_name,
                    column_name=col_name,
                    has_data=has_data,
                )
            )

    return changes


def _diff_column_properties(
    schema: str,
    table: str,
    desired: Column,
    actual: Column,
    managed_schemas: list[str] | None = None,
) -> list[Change]:
    """Diff properties of a single column."""
    changes: list[Change] = []

    # Generation transitions (STORED generated columns). Handle these before the
    # type/nullable/default branches because the available DDL differs and the
    # naive drop+add path is unsafe (it would CASCADE-drop dependent views and
    # trip data-loss protection on populated tables).
    if desired.generated != actual.generated:
        if actual.generated and desired.generated:
            # Expression changed: PG17 SET EXPRESSION rewrites stored values in
            # place (no CASCADE, no type change). Same type, so no AlterColumnType
            # and no DEFAULT handling (both defaults are None for generated cols).
            changes.append(
                AlterColumnSetExpression(
                    schema=schema,
                    table=table,
                    column=desired.name,
                    expression=desired.generated,
                )
            )
            # A NOT NULL flip on an otherwise-unchanged generated column is still
            # a valid in-place ALTER; fall through to the nullability branch.
        elif actual.generated and not desired.generated:
            # generated -> plain: DROP EXPRESSION converts to a normal base
            # column, retaining existing data. Then fall through so the desired
            # plain column's DEFAULT / nullability get applied normally.
            changes.append(
                AlterColumnDropExpression(
                    schema=schema, table=table, column=desired.name
                )
            )
        else:
            # plain -> generated: there is no in-place DDL to add a generation
            # expression to an existing column, and an automatic drop+add would
            # lose the column's real (non-derived) data. Fail fast.
            raise ValueError(
                f'cannot convert existing column "{schema}"."{table}".'
                f'"{desired.name}" into a stored generated column via '
                "schema-diff (Postgres has no in-place ALTER to add a generation "
                "expression, and an automatic drop+add would lose the column's "
                "existing data). Apply this in an explicit @migrate-before block. "
                "Note: a raw text::enum cast is not IMMUTABLE - route any "
                "enum-valued generation expression through an IMMUTABLE function."
            )

    # Type change (skip for generated columns: a same-type SET EXPRESSION needs
    # no AlterColumnType, and a type change on a generated column is not handled
    # automatically here).
    type_changed = _normalize_type(desired.type, managed_schemas) != _normalize_type(
        actual.type, managed_schemas
    )
    if not desired.generated and (
        type_changed or desired.collation != actual.collation
    ):
        # Without a COLLATE clause a retyped column takes the new type's
        # default collation; a column only losing its explicit collation has
        # to be given that default by name.
        collation = desired.collation
        if collation is None and not type_changed:
            collation = 'pg_catalog."default"'
        changes.append(
            AlterColumnType(
                schema=schema,
                table=table,
                column=desired.name,
                old_type=actual.type,
                new_type=desired.type,
                collation=collation,
            )
        )

    # Nullability change
    if desired.nullable != actual.nullable:
        if desired.nullable:
            changes.append(DropNotNull(schema=schema, table=table, column=desired.name))
        else:
            changes.append(SetNotNull(schema=schema, table=table, column=desired.name))

    # Default change (skip for identity columns - they manage their own default)
    if not desired.identity and not actual.identity:
        desired_default = _normalize_default(desired.default)
        actual_default = _normalize_default(actual.default)

        if desired_default != actual_default:
            if desired_default is None:
                changes.append(
                    DropDefault(schema=schema, table=table, column=desired.name)
                )
            else:
                changes.append(
                    SetDefault(
                        schema=schema,
                        table=table,
                        column=desired.name,
                        default_value=desired.default or "",
                    )
                )

    # Identity change
    if desired.identity != actual.identity:
        if actual.identity and not desired.identity:
            # Remove identity
            changes.append(
                DropIdentity(schema=schema, table=table, column=desired.name)
            )
        elif desired.identity and not actual.identity:
            # Add identity
            changes.append(
                AddIdentity(
                    schema=schema,
                    table=table,
                    column=desired.name,
                    identity_type=desired.identity or "",
                )
            )
        else:
            # Change identity type (e.g., ALWAYS <-> BY DEFAULT)
            # Need to drop and add
            changes.append(
                DropIdentity(schema=schema, table=table, column=desired.name)
            )
            changes.append(
                AddIdentity(
                    schema=schema,
                    table=table,
                    column=desired.name,
                    identity_type=desired.identity or "",
                )
            )

    return changes


def _diff_constraints(
    desired_table: Table,
    actual_table: Table,
) -> list[Change]:
    """Diff constraints between desired and actual table."""
    changes: list[Change] = []
    schema = desired_table.schema
    table_name = desired_table.name

    # Build lookup maps
    desired_constraints = {c.name: c for c in desired_table.constraints}
    actual_constraints = {c.name: c for c in actual_table.constraints}

    # Constraints to add
    for name, constraint in desired_constraints.items():
        # Before PostgreSQL 18 a partitioned table's foreign key cannot be
        # added NOT VALID, so it is validated as it is added: a scan of every
        # partition under a lock.
        validated_now = desired_table.partition_key is not None and (
            constraint.type == "f"
        )
        category = ChangeCategory.OFFLINE if validated_now else ChangeCategory.ONLINE
        if name not in actual_constraints:
            changes.append(
                AddConstraint(
                    schema=schema,
                    table=table_name,
                    constraint=constraint,
                    not_valid=not validated_now,
                    category=category,
                )
            )
        elif constraint.definition != actual_constraints[name].definition:
            # Constraint changed - use combined drop+create to ensure correct ordering
            changes.append(
                DropAndCreateConstraint(
                    schema=schema,
                    table=table_name,
                    constraint=constraint,
                    not_valid=not validated_now,
                    category=category,
                )
            )

    # Constraints to drop
    for name in actual_constraints:
        if name not in desired_constraints:
            changes.append(
                DropConstraint(
                    schema=schema,
                    table=table_name,
                    constraint_name=name,
                )
            )

    return changes


def _diff_functions(desired: Schema, actual: Schema) -> list[Change]:
    """Diff functions between desired and actual schemas."""
    changes: list[Change] = []

    # Build lookup by qualified name (without args) to detect parameter renames
    actual_by_qualified_name: dict[str, list[Function]] = {}
    for func in actual.functions.values():
        qname = func.qualified_name
        if qname not in actual_by_qualified_name:
            actual_by_qualified_name[qname] = []
        actual_by_qualified_name[qname].append(func)

    # Track functions that will be dropped due to parameter rename
    functions_to_drop_for_rename: set[str] = set()

    # Functions to create or replace
    for key, func in desired.functions.items():
        if key not in actual.functions:
            # Check if this is a parameter rename (same qualified name, different args)
            existing_funcs = actual_by_qualified_name.get(func.qualified_name, [])
            rename_source = None
            for existing in existing_funcs:
                if existing.signature not in desired.functions:
                    # Found an old version with different args - this is a parameter rename
                    rename_source = existing
                    break

            if rename_source:
                # Parameter names changed - use DropAndCreateFunction with the OLD args
                # to ensure proper DROP before CREATE
                functions_to_drop_for_rename.add(rename_source.signature)
                changes.append(
                    DropAndCreateFunction(
                        function=func,
                        old_args=rename_source.args,  # Pass old args for correct DROP
                        old_kind=rename_source.kind,
                    )
                )
            else:
                changes.append(CreateFunction(function=func))
        elif func.definition_hash != actual.functions[key].definition_hash:
            # A changed return type or kind requires DROP + CREATE: CREATE OR
            # REPLACE refuses both.
            actual_func = actual.functions[key]
            if (
                func.kind != actual_func.kind
                or func.get_return_clause() != actual_func.get_return_clause()
            ):
                # Use combined Drop+Create to keep them together
                changes.append(
                    DropAndCreateFunction(function=func, old_kind=actual_func.kind)
                )
            else:
                changes.append(ReplaceFunction(function=func))

    # An aggregate goes with any support function this run drops (DROP ...
    # CASCADE), so one that is otherwise unchanged is recreated with it.
    dropped: set[str] = set(functions_to_drop_for_rename)
    for c in changes:
        if isinstance(c, DropAndCreateFunction):
            f = c.function
            old_args = c.old_args if c.old_args is not None else f.args
            dropped.add(f"{f.schema}.{f.name}({old_args})")
    dropped.update(k for k in actual.functions if k not in desired.functions)
    changed = {
        c.function.signature
        for c in changes
        if isinstance(c, (CreateFunction, ReplaceFunction, DropAndCreateFunction))
    }
    for key, func in desired.functions.items():
        existing = actual.functions.get(key)
        if (
            func.kind == "a"
            and existing is not None
            and key not in changed
            and dropped.intersection(existing.support_functions)
        ):
            changes.append(DropAndCreateFunction(function=func, old_kind=existing.kind))

    # Functions to drop (excluding those handled by parameter rename above)
    for key, func in actual.functions.items():
        if key not in desired.functions and key not in functions_to_drop_for_rename:
            changes.append(
                DropFunction(
                    schema=func.schema,
                    name=func.name,
                    args=func.args,
                    kind=func.kind,
                )
            )

    # Aggregates are built after the functions they call. The executor would
    # retry an early one, but a recreated support function's DROP ... CASCADE
    # would then take the freshly built aggregate with it.
    def builds_aggregate(c: Change) -> bool:
        return (
            isinstance(c, (CreateFunction, ReplaceFunction, DropAndCreateFunction))
            and c.function.kind == "a"
        )

    return [c for c in changes if not builds_aggregate(c)] + [
        c for c in changes if builds_aggregate(c)
    ]


def _diff_indexes(desired: Schema, actual: Schema) -> list[Change]:
    """Diff indexes between desired and actual schemas."""
    changes: list[Change] = []

    # An index on a partitioned table is built per partition and attached
    # (CreatePartitionedIndex), and dropped without CONCURRENTLY, which
    # Postgres refuses for one; that drop runs in the DDL transaction.
    def create(index: Index) -> CreateIndex:
        if index.partitioned:
            return CreatePartitionedIndex(index=index)
        return CreateIndex(index=index, concurrent=True)

    # Indexes to create
    for key, index in desired.indexes.items():
        current = actual.indexes.get(key)
        if current is None:
            changes.append(create(index))
        elif index.definition != current.definition:
            # Index changed - drop and recreate
            changes.append(
                DropIndex(
                    schema=index.schema,
                    index_name=index.name,
                    concurrent=not current.partitioned,
                )
            )
            changes.append(create(index))
        elif index.partitioned and not current.is_valid:
            # A partitioned index is invalid until every partition has an
            # index attached to it: a run died half way. Finish the job.
            changes.append(create(index))

    # Indexes to drop
    for key, index in actual.indexes.items():
        if key not in desired.indexes:
            changes.append(
                DropIndex(
                    schema=index.schema,
                    index_name=index.name,
                    concurrent=not index.partitioned,
                )
            )

    return changes


def _diff_rules(desired: Schema, actual: Schema) -> list[Change]:
    """Diff rewrite rules on tables.

    `CREATE OR REPLACE RULE` changes a rule in place, event included. A rule on
    a table this run drops goes with the table.
    """
    changes: list[Change] = []
    for key, rule in desired.rules.items():
        current = actual.rules.get(key)
        if current is None:
            changes.append(CreateRule(rule=rule))
        elif rule.definition != current.definition:
            changes.append(
                CreateRule(rule=rule, set_enabled=rule.enabled != current.enabled)
            )
        elif rule.enabled != current.enabled:
            changes.append(AlterRuleEnabled(rule=rule))

    for key, rule in actual.rules.items():
        if key in desired.rules:
            continue
        table = f"{rule.schema}.{rule.table}"
        if table in actual.tables and table not in desired.tables:
            continue  # DROP TABLE takes the rule with it.
        changes.append(
            DropRule(schema=rule.schema, table=rule.table, rule_name=rule.name)
        )
    return changes


def publication_is_droppable(
    publication: Publication, actual: Schema, managed_schemas: list[str]
) -> bool:
    """Whether an undeclared live publication may be dropped.

    Only when everything it publishes is something pgdeclare manages: at least
    one table or schema, every table one introspected in a managed schema, and
    every TABLES IN SCHEMA schema a managed one. FOR ALL TABLES publications,
    empty ones, and any that reach a table or schema outside that set belong
    to someone else (a replication tool, another application) and are left
    alone.
    """
    if publication.all_tables or publication.is_empty:
        return False
    if any(key not in actual.tables for key in publication.tables):
        return False
    return all(schema in managed_schemas for schema in publication.schemas)


def _diff_publications(
    desired: Schema, actual: Schema, managed_schemas: list[str]
) -> list[Change]:
    """Diff publications.

    Declared publications converge in place: options with SET (...), tables
    and schemas with ADD and DROP, a member whose column list or row filter
    changed is dropped and added back. Switching to or from FOR ALL TABLES
    needs a drop and re-create. Undeclared ones are dropped only when
    `publication_is_droppable` says so.
    """
    changes: list[Change] = []
    for name, pub in desired.publications.items():
        current = actual.publications.get(name)
        if current is None:
            changes.append(CreatePublication(publication=pub))
            continue
        if pub.all_tables != current.all_tables:
            changes.append(DropAndCreatePublication(publication=pub))
            continue
        if (pub.publish, pub.via_root, pub.generated_columns) != (
            current.publish,
            current.via_root,
            current.generated_columns,
        ):
            changes.append(AlterPublicationOptions(publication=pub))
        if pub.all_tables:
            continue

        removed_tables = [
            t for key, t in current.tables.items() if key not in pub.tables
        ]
        removed_schemas = [s for s in current.schemas if s not in pub.schemas]
        if removed_tables or removed_schemas:
            changes.append(
                DropPublicationObjects(
                    name=name, tables=removed_tables, schemas=removed_schemas
                )
            )
        for key, table in pub.tables.items():
            old = current.tables.get(key)
            if old is not None and old != table:
                changes.append(ReplacePublicationTable(name=name, table=table))
        added_tables = [t for key, t in pub.tables.items() if key not in current.tables]
        added_schemas = [s for s in pub.schemas if s not in current.schemas]
        if added_tables or added_schemas:
            changes.append(
                AddPublicationObjects(
                    name=name, tables=added_tables, schemas=added_schemas
                )
            )

    for name, current in actual.publications.items():
        if name not in desired.publications and publication_is_droppable(
            current, actual, managed_schemas
        ):
            changes.append(DropPublication(name=name))
    return changes


def _diff_event_triggers(
    desired: Schema,
    actual: Schema,
    managed_schemas: list[str],
    changes: list[Change],
) -> list[Change]:
    """Diff event triggers.

    There is no ALTER for an event trigger's event, filter or function, so a
    change to any of them drops and re-creates it; the enabled state has its
    own ALTER. A function recreated this run (DROP ... CASCADE) takes its
    event triggers with it, so those are re-created too. An undeclared event
    trigger is dropped only when its function lives in a managed schema.
    """
    recreated = {
        (c.function.schema, c.function.name)
        for c in changes
        if isinstance(c, DropAndCreateFunction)
    }
    result: list[Change] = []
    for name, trigger in desired.event_triggers.items():
        current = actual.event_triggers.get(name)
        if current is None:
            result.append(CreateEventTrigger(trigger=trigger))
        elif (
            dataclass_replace(current, enabled=trigger.enabled) != trigger
            or (trigger.function_schema, trigger.function_name) in recreated
        ):
            result.append(CreateEventTrigger(trigger=trigger, replace=True))
        elif current.enabled != trigger.enabled:
            result.append(AlterEventTriggerEnabled(name=name, enabled=trigger.enabled))

    for name, current in actual.event_triggers.items():
        if (
            name not in desired.event_triggers
            and current.function_schema in managed_schemas
        ):
            result.append(DropEventTrigger(name=name))
    return result


def _diff_policies(desired: Schema, actual: Schema) -> list[Change]:
    """Diff RLS policies between desired and actual schemas."""
    changes: list[Change] = []

    # Policies to create or replace
    for key, policy in desired.policies.items():
        if key not in actual.policies:
            changes.append(CreatePolicy(policy=policy))
        elif policy != actual.policies[key]:
            # Policy changed - use combined drop+create to ensure correct order
            changes.append(DropAndCreatePolicy(policy=policy))

    # Policies to drop
    for key, policy in actual.policies.items():
        if key not in desired.policies:
            changes.append(
                DropPolicy(
                    schema=policy.schema,
                    table=policy.table,
                    policy_name=policy.name,
                )
            )

    return changes


def _diff_triggers(desired: Schema, actual: Schema) -> list[Change]:
    """Diff triggers between desired and actual schemas."""
    changes: list[Change] = []

    # Triggers to create
    for key, trigger in desired.triggers.items():
        if key not in actual.triggers:
            changes.append(CreateTrigger(trigger=trigger))
        elif trigger.definition != actual.triggers[key].definition:
            # Trigger changed - drop and recreate
            changes.append(
                DropTrigger(
                    schema=trigger.schema,
                    table=trigger.table,
                    trigger_name=trigger.name,
                )
            )
            changes.append(CreateTrigger(trigger=trigger))

    # Triggers to drop
    for key, trigger in actual.triggers.items():
        if key not in desired.triggers:
            changes.append(
                DropTrigger(
                    schema=trigger.schema,
                    table=trigger.table,
                    trigger_name=trigger.name,
                )
            )

    return changes


def _diff_enums(desired: Schema, actual: Schema) -> list[Change]:
    """Diff enum types between desired and actual schemas."""
    changes: list[Change] = []

    # Enums to create or modify
    for key, enum in desired.enums.items():
        if key not in actual.enums:
            changes.append(CreateEnum(enum=enum))
        else:
            # Enum exists - check for new values
            actual_enum = actual.enums[key]
            for value in enum.values:
                if value not in actual_enum.values:
                    # Find the value to insert after
                    idx = enum.values.index(value)
                    after = enum.values[idx - 1] if idx > 0 else None
                    changes.append(
                        AddEnumValue(
                            schema=enum.schema,
                            enum_name=enum.name,
                            value=value,
                            after=after,
                        )
                    )

    # Note: We don't automatically drop enum values as it requires type recreation

    return changes


def _diff_composite_types(desired: Schema, actual: Schema) -> list[Change]:
    """Diff composite types between desired and actual schemas.

    Composite types must be created before functions that use them.
    The retry-until-stable execution will handle dependency ordering if
    a composite type references an enum that hasn't been created yet.

    Note: ALTER TYPE for column changes is complex and not supported yet.
    Changing a composite type's columns requires DROP + CREATE, which
    requires CASCADE and handling dependent objects. For now, we only
    support creating new composite types.
    """
    changes: list[Change] = []

    # Composite types to create
    for key, comp_type in desired.composite_types.items():
        if key not in actual.composite_types:
            changes.append(CreateCompositeType(composite_type=comp_type))
        # Note: We don't diff column changes for existing composite types yet
        # This would require DROP TYPE ... CASCADE + CREATE TYPE, which is complex

    # Note: We don't automatically drop composite types as it may break dependent objects

    return changes


def _diff_domains(
    desired: Schema,
    actual: Schema,
    managed_schemas: list[str] | None = None,
) -> list[Change]:
    """Create domains and converge the ones that exist.

    Default, NOT NULL and CHECK constraints are altered in place. A changed
    base type or collation has no ALTER, so the domain is dropped and
    recreated, which is only safe while nothing uses it; otherwise the change
    is refused with the dependents named. Drops of undeclared domains come
    from _drop_domains, later in the plan.
    """
    changes: list[Change] = []

    for key, domain in desired.domains.items():
        current = actual.domains.get(key)
        if current is None:
            changes.append(CreateDomain(domain=domain))
            continue

        if _normalize_type(domain.base_type, managed_schemas) != _normalize_type(
            current.base_type, managed_schemas
        ) or (domain.collation or "") != (current.collation or ""):
            if current.dependents:
                raise ValueError(
                    f'cannot change the base type of domain "{domain.schema}".'
                    f'"{domain.name}" from {current.base_type} to '
                    f"{domain.base_type}: PostgreSQL has no ALTER for it, and "
                    "dropping and recreating the domain would drop what uses "
                    f"it ({'; '.join(current.dependents)}). Move those to the "
                    "new type in an explicit @migrate-before block, or declare "
                    "a new domain and retype the columns to it."
                )
            changes.append(DropAndCreateDomain(domain=domain))
            continue

        changes.extend(_diff_domain_properties(domain, current))

    return changes


def _diff_domain_properties(desired: Domain, actual: Domain) -> list[Change]:
    """Default, NOT NULL and CHECK constraint changes of an existing domain."""
    changes: list[Change] = []
    schema, name = desired.schema, desired.name

    desired_default = _normalize_default(desired.default)
    if desired_default != _normalize_default(actual.default):
        if desired_default is None:
            changes.append(AlterDomainDropDefault(schema=schema, name=name))
        else:
            changes.append(
                AlterDomainSetDefault(
                    schema=schema, name=name, default_value=desired.default or ""
                )
            )

    if desired.nullable != actual.nullable:
        if desired.nullable:
            changes.append(AlterDomainDropNotNull(schema=schema, name=name))
        else:
            changes.append(AlterDomainSetNotNull(schema=schema, name=name))

    desired_checks = {c.name: c for c in desired.constraints}
    actual_checks = {c.name: c for c in actual.constraints}
    # Drops first: a renamed constraint must not briefly coexist with its
    # replacement, and a dropped CHECK may be what the new one would violate.
    for cname in sorted(actual_checks.keys() - desired_checks.keys()):
        changes.append(
            DropDomainConstraint(schema=schema, domain=name, constraint_name=cname)
        )
    for cname, constraint in sorted(desired_checks.items()):
        if cname not in actual_checks:
            changes.append(
                AddDomainConstraint(schema=schema, domain=name, constraint=constraint)
            )
        elif constraint.definition != actual_checks[cname].definition:
            changes.append(
                DropAndCreateDomainConstraint(
                    schema=schema, domain=name, constraint=constraint
                )
            )

    return changes


def _in_use_error(kind: str, qualified: str, dependents: list[str]) -> ValueError:
    schema, _, name = qualified.partition(".")
    return ValueError(
        f'cannot change {kind} "{schema}"."{name}": PostgreSQL has no ALTER for '
        f"it, and dropping and recreating the {kind} would drop what uses it "
        f"({'; '.join(dependents)}). Move those off it in an explicit "
        f"@migrate-before block, or declare a new {kind} under another name "
        "and move them to it."
    )


def _diff_collations(
    desired: Schema, actual: Schema
) -> tuple[list[Change], list[Change]]:
    """Diff collations. Returns (changes before everything, drops after it).

    A changed collation is dropped and recreated, which is only safe while
    nothing uses it; otherwise the change is refused naming the dependents.
    Drops come last, after the columns, domains and ranges that used them.
    """
    before: list[Change] = []
    for key, collation in sorted(desired.collations.items()):
        current = actual.collations.get(key)
        if current is None:
            before.append(CreateCollation(collation=collation))
        elif collation != current:
            if current.dependents:
                raise _in_use_error("collation", key, current.dependents)
            before.append(DropAndCreateCollation(collation=collation))
    after: list[Change] = [
        DropCollation(schema=c.schema, name=c.name)
        for key, c in sorted(actual.collations.items())
        if key not in desired.collations
    ]
    return before, after


def _ranges_equal(
    desired: RangeType, actual: RangeType, managed_schemas: list[str] | None
) -> bool:
    return (
        _normalize_type(desired.subtype, managed_schemas)
        == _normalize_type(actual.subtype, managed_schemas)
        and dataclass_replace(desired, subtype=actual.subtype) == actual
    )


def _diff_ranges(
    desired: Schema,
    actual: Schema,
    managed_schemas: list[str] | None = None,
) -> list[Change]:
    """Create range types and recreate the ones whose definition changed.

    No option of a range has an ALTER, so a change drops and recreates the
    type (its multirange goes with it), only while nothing uses either;
    otherwise the change is refused naming the dependents. A range with a
    canonical function gets a shell type first: the function's signature
    names the type, and the range definition names the function. Drops of
    undeclared ranges come from _drop_ranges, later in the plan.
    """
    changes: list[Change] = []
    for key, range_type in sorted(desired.ranges.items()):
        current = actual.ranges.get(key)
        if current is None:
            if range_type.canonical:
                changes.append(
                    CreateShellType(schema=range_type.schema, name=range_type.name)
                )
            changes.append(CreateRangeType(range_type=range_type))
            continue
        if _ranges_equal(range_type, current, managed_schemas):
            continue
        if current.dependents:
            raise _in_use_error("range type", key, current.dependents)
        if range_type.canonical:
            # The canonical function would have to take the type that is
            # about to be dropped.
            raise ValueError(
                f'cannot change range type "{range_type.schema}".'
                f'"{range_type.name}" to one with a canonical function in '
                "place: the function's signature names the type. Drop the "
                "type and the function in an explicit @migrate-before block "
                "and let the next run create both."
            )
        changes.append(DropAndCreateRangeType(range_type=range_type))
    return changes


def _drop_ranges(desired: Schema, actual: Schema) -> list[Change]:
    """Drop range types that are no longer declared.

    Emitted after the table, view and function changes so whatever used the
    range is gone first; the executor also retries a drop that still finds a
    dependent.
    """
    return [
        DropRangeType(
            schema=r.schema, name=r.name, has_canonical=r.canonical is not None
        )
        for key, r in sorted(actual.ranges.items())
        if key not in desired.ranges
    ]


def _skip_canonical_function_drops(
    changes: list[Change], actual: Schema
) -> list[Change]:
    """Leave a dropped range's canonical function to the range's own drop.

    The function and the type depend on each other, so DropRangeType drops
    the function with the type, after checking nothing else uses it. A
    separate DROP FUNCTION ... CASCADE would take the type, and whatever
    still uses it, down without that check.
    """
    dropped = {f"{c.schema}.{c.name}" for c in changes if isinstance(c, DropRangeType)}
    signatures = {
        r.canonical_signature
        for key, r in actual.ranges.items()
        if key in dropped and r.canonical_signature
    }
    if not signatures:
        return changes
    return [
        c
        for c in changes
        if not (
            isinstance(c, DropFunction)
            and f"{c.schema}.{c.name}({c.args})" in signatures
        )
    ]


def _drop_domains(desired: Schema, actual: Schema) -> list[Change]:
    """Drop domains that are no longer declared.

    Emitted after the table, view and function changes so the columns and
    signatures that used the domain are gone first; the executor also retries
    a drop that still finds a dependent.
    """
    return [
        DropDomain(schema=domain.schema, name=domain.name)
        for key, domain in sorted(actual.domains.items())
        if key not in desired.domains
    ]


def _diff_views(desired: Schema, actual: Schema) -> list[Change]:
    """Diff views between desired and actual schemas."""
    changes: list[Change] = []

    # Views to create or replace
    for key, view in desired.views.items():
        if key not in actual.views:
            changes.append(CreateView(view=view))
        elif view.definition.strip() != actual.views[key].definition.strip():
            # Use DropAndCreateView to handle column order/type changes
            # that CREATE OR REPLACE VIEW can't handle. The recreate carries the
            # declared WITH (...) options itself.
            changes.append(DropAndCreateView(view=view))
        elif view.reloptions != actual.views[key].reloptions:
            # Definition already matches, only the security mode drifted (say,
            # `security_invoker` was declared but never written). ALTER VIEW
            # converges it in place.
            changes.append(
                AlterViewOptions(
                    view=view,
                    current_reloptions=list(actual.views[key].reloptions),
                )
            )

    # Views to drop
    for key, view in actual.views.items():
        if key not in desired.views:
            changes.append(DropView(schema=view.schema, view_name=view.name))

    return changes


def _diff_materialized_views(desired: Schema, actual: Schema) -> list[Change]:
    """Diff materialized views between desired and actual schemas.

    Only the query and the storage options are compared. Whether a live
    materialized view is populated is runtime state (an application refreshes
    it); `WITH [NO] DATA` from the declaration decides how a create or recreate
    populates it.
    """
    changes: list[Change] = []

    for key, matview in desired.materialized_views.items():
        current = actual.materialized_views.get(key)
        if current is None:
            changes.append(CreateMaterializedView(matview=matview))
        elif matview.definition.strip() != current.definition.strip():
            changes.append(DropAndCreateMaterializedView(matview=matview))
        elif matview.reloptions != current.reloptions:
            changes.append(
                AlterMaterializedViewOptions(
                    matview=matview, current_reloptions=list(current.reloptions)
                )
            )

    for key, matview in actual.materialized_views.items():
        if key not in desired.materialized_views:
            changes.append(
                DropMaterializedView(schema=matview.schema, matview_name=matview.name)
            )

    return changes


def _diff_sequences(
    desired: Schema, actual: Schema
) -> tuple[list[Change], list[Change]]:
    """Diff sequences. Returns (changes before tables, changes after them).

    Creates and option changes go before the tables, so a new table's
    nextval() default finds its sequence. OWNED BY goes after, because it
    names a column that may be created in the same run; releasing an old owner
    goes before, since dropping that owner would drop the sequence. Drops go
    after too:
    DROP SEQUENCE has no CASCADE, so a default still using the sequence must
    already be gone (or the drop fails, which is the point).
    """
    before: list[Change] = []
    after: list[Change] = []

    for key, seq in desired.sequences.items():
        current = actual.sequences.get(key)
        if current is None:
            before.append(CreateSequence(sequence=seq))
            if seq.owned_by is not None:
                after.append(SetSequenceOwner(sequence=seq))
            continue
        if not seq.options_equal(current):
            before.append(AlterSequence(sequence=seq))
        if seq.owned_by != current.owned_by:
            if current.owned_by is not None:
                # Release the old owner before the tables change: dropping the
                # old owning column or table would otherwise take the sequence
                # with it.
                before.append(
                    SetSequenceOwner(sequence=dataclass_replace(seq, owned_by=None))
                )
            if seq.owned_by is not None:
                after.append(SetSequenceOwner(sequence=seq))

    for key, seq in actual.sequences.items():
        if key not in desired.sequences:
            after.append(DropSequence(schema=seq.schema, sequence_name=seq.name))

    return before, after


def _diff_grants(desired: Schema, actual: Schema) -> list[Change]:
    """Diff grants between desired and actual schemas."""
    changes: list[Change] = []

    # Build lookup sets
    desired_keys = {g.key for g in desired.grants}
    actual_keys = {g.key for g in actual.grants}
    desired_by_key = {g.key: g for g in desired.grants}
    actual_by_key = {g.key: g for g in actual.grants}

    # Revokes go FIRST. A table-level REVOKE also strips every column-level
    # grant of that privilege on the table, so moving a role from a table-wide
    # INSERT to per-column INSERT (the diff emits column GRANTs plus a table
    # REVOKE) must revoke before it grants. Granting first lets the REVOKE wipe
    # the column grants it had just written, leaving the role with no INSERT at
    # all and the convergence check failing. The executor runs changes in list
    # order within an iteration, so this order is the execution order. Sorted
    # for a stable plan.
    for key in sorted(actual_keys - desired_keys):
        changes.append(RevokePrivilege(grant=actual_by_key[key]))

    for key in sorted(desired_keys - actual_keys):
        changes.append(GrantPrivilege(grant=desired_by_key[key]))

    return changes


def _diff_default_privileges(
    desired: Schema, actual: Schema
) -> tuple[list[Change], list[Change]]:
    """Revokes and grants of ALTER DEFAULT PRIVILEGES entries.

    Both sides hold only entries whose grantee is a managed role (see
    introspection), so an entry for any other grantee is never touched. A
    change of grant option revokes the privilege and grants it again.
    """
    desired_by_key = {p.key: p for p in desired.default_privileges}
    actual_by_key = {p.key: p for p in actual.default_privileges}
    revokes: list[Change] = [
        RevokeDefaultPrivilege(entry=actual_by_key[key])
        for key in sorted(actual_by_key.keys() - desired_by_key.keys())
    ]
    grants: list[Change] = [
        GrantDefaultPrivilege(entry=desired_by_key[key])
        for key in sorted(desired_by_key.keys() - actual_by_key.keys())
    ]
    return revokes, grants


def _diff_roles(desired: Schema, actual: Schema) -> list[Change]:
    """
    Diff roles between desired and actual schemas.

    Creates missing roles, alters changed roles.
    Note: Role drops are not generated to avoid accidental role deletion.
    """
    changes: list[Change] = []

    # Roles to create
    for name, role in desired.roles.items():
        if name not in actual.roles:
            changes.append(CreateRole(role=role))
        else:
            # Check if role attributes changed
            actual_role = actual.roles[name]
            if role != actual_role:
                changes.append(AlterRole(role=role))

    # Note: We intentionally don't generate DropRole changes
    # to avoid accidentally dropping roles that have dependencies.
    # If needed, drops should be explicitly managed.

    return changes


def _diff_role_memberships(desired: Schema, actual: Schema) -> list[Change]:
    """
    Diff role memberships between desired and actual schemas.

    Grants missing memberships, revokes extra memberships for any membership
    where at least one side (role or member) is managed (declared in SQL).
    This makes SQL authoritative for all grants involving managed roles.
    """
    changes: list[Change] = []

    # Get managed role names from desired schema
    managed_roles = set(desired.roles.keys())

    # Build lookup sets
    desired_keys = {m.key for m in desired.role_memberships}
    actual_keys = {m.key for m in actual.role_memberships}
    desired_by_key = {m.key: m for m in desired.role_memberships}
    actual_by_key = {m.key: m for m in actual.role_memberships}

    # Memberships to grant - grant all missing memberships declared in SQL
    for key in desired_keys - actual_keys:
        changes.append(GrantRoleMembership(membership=desired_by_key[key]))

    # Memberships to revoke - revoke if EITHER role or member is managed
    # This makes SQL authoritative for all grants involving managed roles
    for key in actual_keys - desired_keys:
        membership = actual_by_key[key]
        if membership.role in managed_roles or membership.member in managed_roles:
            changes.append(RevokeRoleMembership(membership=membership))

    # Check for admin_option changes (only for managed memberships)
    for key in desired_keys & actual_keys:
        desired_m = desired_by_key[key]
        actual_m = actual_by_key[key]
        if desired_m.admin_option != actual_m.admin_option:
            # Need to revoke and re-grant with new option
            changes.append(RevokeRoleMembership(membership=actual_m))
            changes.append(GrantRoleMembership(membership=desired_m))

    return changes


def _normalize_type(type_str: str, managed_schemas: list[str] | None = None) -> str:
    """Normalize type string for comparison."""
    if not type_str:
        return ""
    # Remove extra whitespace and normalize common variations
    normalized = " ".join(type_str.split()).lower()

    # Strip managed schema qualifiers from types for comparison.
    # e.g., "app_generated.asset_type" -> "asset_type"
    # This prevents false-positive diffs when the same type exists
    # in multiple schemas and PostgreSQL's format_type() disambiguates.
    # Uses replace() (not startswith) to handle array types like
    # "app_generated.asset_type[]".
    if managed_schemas:
        for schema in managed_schemas:
            normalized = normalized.replace(f"{schema.lower()}.", "")

    # Handle common aliases
    replacements = {
        "integer": "int4",
        "bigint": "int8",
        "smallint": "int2",
        "boolean": "bool",
        "character varying": "varchar",
        "timestamp without time zone": "timestamp",
        "timestamp with time zone": "timestamptz",
        "double precision": "float8",
        "real": "float4",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _normalize_default(default: Optional[str]) -> Optional[str]:
    """Normalize default value for comparison.

    Uses case-insensitive comparison to handle PostgreSQL's type name lowercasing
    (e.g., '::JSONB' vs '::jsonb') while preserving expression structure.

    Previous implementation used split("::")[0] which incorrectly truncated
    complex expressions like (now() + '7 days'::INTERVAL).
    """
    if default is None:
        return None
    # Normalize whitespace and case - preserve structure
    return default.strip().lower()


def categorize_changes(changes: list[Change]) -> dict[ChangeCategory, list[Change]]:
    """
    Group changes by their category.

    Returns:
        Dict mapping category to list of changes in that category
    """
    result: dict[ChangeCategory, list[Change]] = {
        ChangeCategory.ONLINE: [],
        ChangeCategory.OFFLINE: [],
        ChangeCategory.BLOCKED: [],
        ChangeCategory.TRY_FIRST: [],
    }

    for change in changes:
        result[change.category].append(change)

    return result


def split_index_operations(
    changes: list[Change],
) -> tuple[list[Change], list[Change]]:
    """
    Split changes into DDL (transaction) and index (concurrent) operations.

    Index operations with concurrent=True need to run outside a transaction.
    For index changes, drops must come before creates (to handle recreate scenarios).

    Returns:
        Tuple of (ddl_changes, index_changes)
    """
    ddl_changes = []
    index_changes = []

    for change in changes:
        if isinstance(change, (CreateIndex, DropIndex)) and getattr(
            change, "concurrent", False
        ):
            index_changes.append(change)
        else:
            ddl_changes.append(change)

    # Sort index changes: DropIndex before CreateIndex
    # This ensures drop+recreate pairs execute in correct order
    index_changes.sort(key=lambda c: 0 if isinstance(c, DropIndex) else 1)

    return ddl_changes, index_changes


# =============================================================================
# Extensions, foreign data, ownership
# =============================================================================

# Installed in every database by initdb; never worth a note.
_BUILTIN_EXTENSIONS = frozenset({"plpgsql"})


def _diff_extensions(desired: Schema, actual: Schema) -> list[Change]:
    """Create declared extensions; converge a stated version or schema.

    An unstated version or schema is not compared. Undeclared extensions are
    never dropped here: see `_drop_extensions` and `undeclared_extensions`.
    """
    changes: list[Change] = []
    for name, ext in desired.extensions.items():
        current = actual.extensions.get(name)
        if current is None:
            changes.append(CreateExtension(extension=ext))
            continue
        if ext.version is not None and ext.version != current.version:
            changes.append(
                AlterExtensionUpdate(
                    name=name,
                    version=ext.version,
                    current_version=current.version or "",
                )
            )
        if ext.schema is not None and ext.schema != current.schema:
            changes.append(AlterExtensionSchema(name=name, schema=ext.schema))
    return changes


def _drop_extensions(desired: Schema, actual: Schema) -> list[Change]:
    """Drop the undeclared extensions a `@drop extension:NAME` directive names."""
    return [
        DropExtension(name=name)
        for name in sorted(actual.extensions)
        if name not in desired.extensions and name in desired.extension_drops
    ]


def undeclared_extensions(desired: Schema, actual: Schema) -> list[str]:
    """Installed extensions the schema files neither declare nor consent to drop.

    They are left in place (dropping one cascades to every column of its
    types) and reported, so a forgotten declaration is visible.
    """
    return [
        name
        for name in sorted(actual.extensions)
        if name not in desired.extensions
        and name not in desired.extension_drops
        and name not in _BUILTIN_EXTENSIONS
    ]


def _diff_foreign_servers(desired: Schema, actual: Schema) -> list[Change]:
    """Create declared foreign servers and converge their version and options.

    Undeclared servers are never dropped: other schemas' foreign tables and
    user mappings may depend on them. The wrapper and TYPE have no ALTER, and
    recreating the server would cascade to its mappings and foreign tables,
    so a change to either is refused.
    """
    changes: list[Change] = []
    for name, srv in desired.foreign_servers.items():
        current = actual.foreign_servers.get(name)
        if current is None:
            changes.append(CreateForeignServer(server=srv))
            continue
        if srv.fdw != current.fdw or srv.type != current.type:
            raise ValueError(
                f"cannot change the foreign data wrapper or TYPE of server "
                f"{quote_ident(name)} ({current.fdw}, {current.type!r} -> "
                f"{srv.fdw}, {srv.type!r}): PostgreSQL has no ALTER for them, "
                "and recreating the server drops its user mappings and foreign "
                "tables. Recreate it in an explicit @migrate-before block."
            )
        set_version = srv.version is not None and srv.version != current.version
        if set_version or srv.options != current.options:
            changes.append(
                AlterForeignServer(
                    server=srv,
                    current_options=dict(current.options),
                    set_version=set_version,
                )
            )
    return changes


def _diff_user_mappings(desired: Schema, actual: Schema) -> list[Change]:
    """Create and converge declared user mappings.

    A secret option (anything but USER_MAPPING_PUBLIC_OPTIONS) the declaration
    does not set is left alone, so a password provisioned outside the schema
    files survives. An undeclared mapping is dropped only when it is a managed
    role's on a declared server; anything else is not this schema's business.
    """
    changes: list[Change] = []
    managed_roles = set(desired.roles)
    for key, mapping in desired.user_mappings.items():
        current = actual.user_mappings.get(key)
        if current is None:
            changes.append(CreateUserMapping(mapping=mapping))
            continue
        if mapping.options is None or current.options is None:
            continue  # Options not readable on one side: not compared.
        compared = {
            name: value
            for name, value in current.options.items()
            if name in mapping.options or name in USER_MAPPING_PUBLIC_OPTIONS
        }
        if mapping.options != compared:
            changes.append(AlterUserMapping(mapping=mapping, current_options=compared))
    for key, current in actual.user_mappings.items():
        if (
            key not in desired.user_mappings
            and current.server in desired.foreign_servers
            and current.user in managed_roles
        ):
            changes.append(DropUserMapping(mapping=current))
    return changes


def _diff_foreign_tables(
    desired: Schema, actual: Schema, managed_schemas: list[str] | None = None
) -> list[Change]:
    """Create, alter and drop foreign tables.

    A foreign table keeps no rows locally, so column changes are catalog-only
    and a drop needs no `@drop`. A changed server has no ALTER: the table is
    recreated, its grants replayed in the same statement.
    """
    changes: list[Change] = []
    for key, table in desired.foreign_tables.items():
        current = actual.foreign_tables.get(key)
        if current is None:
            changes.append(CreateForeignTable(table=table))
        elif table.server != current.server:
            changes.append(DropAndCreateForeignTable(table=table))
        else:
            actions = _foreign_table_actions(table, current, managed_schemas)
            if actions:
                changes.append(
                    AlterForeignTable(
                        schema=table.schema, table_name=table.name, actions=actions
                    )
                )
    for key, table in actual.foreign_tables.items():
        if key not in desired.foreign_tables:
            changes.append(DropForeignTable(schema=table.schema, table_name=table.name))
    return changes


def _foreign_table_actions(
    desired: ForeignTable, actual: ForeignTable, managed_schemas: list[str] | None
) -> list[str]:
    """The `ALTER FOREIGN TABLE` actions turning `actual` into `desired`."""
    actions: list[str] = []
    for name in actual.columns:
        if name not in desired.columns:
            actions.append(f"DROP COLUMN {quote_ident(name)}")
    for name, column in desired.columns.items():
        if name not in actual.columns:
            actions.append(
                "ADD COLUMN "
                + foreign_column_sql(column, desired.column_options.get(name))
            )
    for name, column in desired.columns.items():
        current = actual.columns.get(name)
        if current is None:
            continue
        ident = quote_ident(name)
        if column.generated != current.generated:
            raise ValueError(
                f"cannot change the generation expression of foreign table column "
                f"{quote_ident(desired.schema)}.{quote_ident(desired.name)}.{ident}; "
                "drop and re-add the column in an explicit @migrate-before block."
            )
        if _normalize_type(column.type, managed_schemas) != _normalize_type(
            current.type, managed_schemas
        ):
            actions.append(f"ALTER COLUMN {ident} TYPE {column.type}")
        if column.nullable != current.nullable:
            verb = "DROP" if column.nullable else "SET"
            actions.append(f"ALTER COLUMN {ident} {verb} NOT NULL")
        if _normalize_default(column.default) != _normalize_default(current.default):
            if column.default is None:
                actions.append(f"ALTER COLUMN {ident} DROP DEFAULT")
            else:
                actions.append(f"ALTER COLUMN {ident} SET DEFAULT {column.default}")
        column_options = alter_fdw_options_sql(
            desired.column_options.get(name, {}), actual.column_options.get(name, {})
        )
        if column_options:
            actions.append(f"ALTER COLUMN {ident} {column_options}")
    table_options = alter_fdw_options_sql(desired.options, actual.options)
    if table_options:
        actions.append(table_options)
    return actions


def _owner_keys_rebuilt(changes: list[Change]) -> set[str]:
    """Owner keys of the objects this run creates or drops and recreates.

    A fresh object belongs to the migrating role, whatever the live owner was.
    """
    keys: set[str] = set()

    def mark(kind: str, schema: str, name: str, args: str = "") -> None:
        if kind in ROUTINE_OWNER_KINDS:
            keys.add(f"{kind}:{schema}.{name}({args})")
        else:
            keys.add(f"{kind}:{schema}.{name}")

    for c in changes:
        if isinstance(c, CreateTable):
            mark("TABLE", c.table.schema, c.table.name)
        elif isinstance(c, (CreateForeignTable, DropAndCreateForeignTable)):
            mark("FOREIGN TABLE", c.table.schema, c.table.name)
        elif isinstance(c, (CreateView, DropAndCreateView)):
            mark("VIEW", c.view.schema, c.view.name)
        elif isinstance(c, (CreateMaterializedView, DropAndCreateMaterializedView)):
            mark("MATERIALIZED VIEW", c.matview.schema, c.matview.name)
        elif isinstance(c, CreateSequence):
            mark("SEQUENCE", c.sequence.schema, c.sequence.name)
        elif isinstance(c, (CreateFunction, DropAndCreateFunction)):
            f = c.function
            mark(f.keyword, f.schema, f.name, f.args)
        elif isinstance(c, CreateEnum):
            mark("TYPE", c.enum.schema, c.enum.name)
        elif isinstance(c, CreateCompositeType):
            mark("TYPE", c.composite_type.schema, c.composite_type.name)
        elif isinstance(c, (CreateDomain, DropAndCreateDomain)):
            mark("DOMAIN", c.domain.schema, c.domain.name)
        elif isinstance(c, (CreateRangeType, DropAndCreateRangeType)):
            mark("TYPE", c.range_type.schema, c.range_type.name)
        elif isinstance(c, (CreateCollation, DropAndCreateCollation)):
            mark("COLLATION", c.collation.schema, c.collation.name)
        if isinstance(c, (DropAndCreateView, DropAndCreateMaterializedView)):
            for dep in c.dependents:
                if dep.relation is not None:
                    kind = "MATERIALIZED VIEW" if dep.kind == "m" else "VIEW"
                    mark(kind, dep.schema, dep.name)
    return keys


def _diff_owners(
    desired: Schema, actual: Schema, changes: list[Change]
) -> list[Change]:
    """`ALTER ... OWNER TO` for every stated owner the target does not have.

    `desired.owners` holds only owners the schema files state (see
    `PgDeclareConfig.manage_ownership`), so nothing else is touched. An object
    built fresh in this run gets its owner written whatever the live one was.
    """
    rebuilt = _owner_keys_rebuilt(changes)
    result: list[Change] = []
    for key in sorted(desired.owners):
        wanted = desired.owners[key]
        current = actual.owners.get(key)
        if current is None or current.owner != wanted.owner or key in rebuilt:
            result.append(AlterOwner(owner=wanted))
    return result


# Owner kind -> the grant object types its implicit privileges read back as.
# Relations of every kind read back as TABLE (and their columns as COLUMN), an
# aggregate or window function as FUNCTION. Types, domains and collations have
# no modelled grants.
_OWNER_GRANT_TYPES: dict[str, tuple[str, ...]] = {
    "SCHEMA": ("SCHEMA",),
    "TABLE": ("TABLE", "COLUMN"),
    "VIEW": ("TABLE", "COLUMN"),
    "MATERIALIZED VIEW": ("TABLE", "COLUMN"),
    "FOREIGN TABLE": ("TABLE", "COLUMN"),
    "SEQUENCE": ("SEQUENCE",),
    "FUNCTION": ("FUNCTION",),
    "PROCEDURE": ("PROCEDURE",),
    "AGGREGATE": ("FUNCTION",),
}

# (grant object type, schema, object name, routine signature or None, owner)
_OwnerGrantKey = tuple[str, str, str, Optional[str], str]


def _stated_owner_grants(desired: Schema) -> set[_OwnerGrantKey]:
    """The exact objects whose stated owner's privileges are not grants.

    Keyed by grant object type and, for a routine, its signature, so a grant
    to the same role on another object of that name (a function named like
    the owned table, another overload) is still diffed.
    """
    keys: set[_OwnerGrantKey] = set()
    for o in desired.owners.values():
        schema = o.name if o.kind == "SCHEMA" else o.schema
        args = o.args if o.kind in ROUTINE_OWNER_KINDS else None
        for grant_type in _OWNER_GRANT_TYPES.get(o.kind, ()):
            keys.add((grant_type, schema, o.name, args, o.owner))
    return keys


def _is_stated_owner_grant(grant: Grant, owner_grants: set[_OwnerGrantKey]) -> bool:
    args = grant.args
    if grant.object_type in ("FUNCTION", "PROCEDURE") and args is None:
        # A routine grant without a signature cannot be pinned to one
        # overload, so it is never taken for an owner's privilege.
        return False
    key = (grant.object_type, grant.schema, grant.object_name, args, grant.grantee)
    return key in owner_grants
