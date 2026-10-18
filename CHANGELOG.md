# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0 the API
may change in minor releases.

## [Unreleased]

### Added

- Procedures, aggregates and window functions are managed like functions:
  created, replaced with `CREATE OR REPLACE` when the signature, return type
  and kind are unchanged, otherwise dropped and re-created, and dropped when
  undeclared. Aggregates are rendered from `pg_aggregate`, built after the
  functions they call, and re-created when one of those is dropped. Grants to
  managed roles use `ON PROCEDURE` for procedures, and `COMMENT ON PROCEDURE`
  and `COMMENT ON AGGREGATE` are managed.
- Default privileges (`ALTER DEFAULT PRIVILEGES [FOR ROLE] [IN SCHEMA] GRANT
  ... ON TABLES | SEQUENCES | FUNCTIONS | ROUTINES | TYPES | SCHEMAS`) granted
  to managed roles, schema-less or in a managed schema: declaring grants,
  removing revokes, a change converges. The plan database runs them before
  the other statements, so every declared object they cover gets the grant.
- Range types (`CREATE TYPE ... AS RANGE`) in the managed schemas, with
  their multirange: created with every option (subtype, operator class,
  collation, canonical and subtype_diff functions, multirange name); dropped
  and recreated on a change while nothing uses them, otherwise refused naming
  the dependents; dropped when undeclared, after the columns using them.
  Comments on them are managed.
- Collations (`CREATE COLLATION`, libc, ICU and builtin providers) in the
  managed schemas: created, recreated on a change while nothing uses them
  (refused naming the dependents otherwise), dropped when undeclared after
  everything using them. Comments on them are managed.
- Column collations are compared and converged (`ALTER COLUMN ... TYPE ...
  COLLATE`), and a new table or column is created with its declared
  `COLLATE`, which used to be dropped.
- Publications: created, converged in place (`ALTER PUBLICATION ... ADD` /
  `DROP` tables and `TABLES IN SCHEMA`, column lists and row filters on
  PostgreSQL 15+, `SET (...)` publish options) and dropped when undeclared,
  but only when every table they publish is managed. `FOR ALL TABLES` and
  empty publications are never dropped unless declared. Subscriptions are not
  managed.
- Rewrite rules on tables (`CREATE OR REPLACE RULE`, enabled state), never a
  view's `_RETURN` rule.
- Event triggers (event, `WHEN TAG` filter, function, enabled state); an
  undeclared one is dropped only when its function is in a managed schema.
  Creating one needs a superuser.
- Declarative partitioning: partitioned tables (`PARTITION BY RANGE`, `LIST`,
  `HASH`) and their partitions (`PARTITION OF ... FOR VALUES` or `DEFAULT`,
  sub-partitioned ones included). Columns, defaults, NOT NULL, constraints,
  indexes and triggers a partition inherits are not diffed on the partition.
  A changed bound is a `DETACH` and `ATTACH` (offline). A partition the schema
  files do not declare, such as one pg_partman created, is never dropped; the
  output notes it. An index on a partitioned table is built `CONCURRENTLY` on
  each partition and attached to an `ON ONLY` parent index. Converting a table
  to or from partitioned, or changing its partition key, is refused with an
  error pointing to `@migrate-before`.
- Foreign data: foreign tables in the managed schemas (created, altered with
  one `ALTER FOREIGN TABLE`, recreated on a new server, dropped when
  undeclared), foreign servers and user mappings (created and their options
  converged; servers are never dropped, a managed role's undeclared mapping on
  a declared server is). User-mapping option values other than `user` and
  `password_required` are masked in `diff`, JSON, dry-run, prove output and
  logs, and a secret option the declaration omits is left alone.
- Extensions are modelled: declared ones are created with their schema,
  version and `CASCADE`, a declared version is reached with `ALTER EXTENSION
  ... UPDATE TO`, and an undeclared extension is reported as a note rather
  than dropped, unless a `-- @drop extension:NAME` directive consents. The CLI
  no longer installs declared extensions in the target before the run; the
  diff does, so `diff` shows them.
- Opt-in ownership (`manage_ownership`, `--manage-ownership`): an owner the
  schema files state with `ALTER ... OWNER TO`, when it is a managed role, is
  converged on tables (partitioned ones too), foreign tables, views,
  materialized views, sequences, functions, procedures, aggregates, enum,
  composite and range types, domains, collations and the managed schemas,
  including after a recreate. The owner's implicit privileges are left out of
  the grant diff on that exact object only (kind and, for a routine,
  signature), so a grant to the owner on a same-named object or another
  overload is still converged.

### Changed

- README: the plan database's privilege needs, the PostgreSQL 14 `public`
  ownership caveat, a corrected Python example, and a comparison with other
  tools.

### Fixed

- `CREATE TABLE` quotes inline constraint names, so a name with a quote in it
  stays a name instead of breaking (or injecting into) the statement.
- A `DROP` written without `CASCADE` that fails because another object
  still depends on it (`2BP01`) is retried after the rest, like a missing
  dependency: views, materialized views, domains, range types, collations,
  extensions and foreign tables. Dropping a stack of views no longer depends
  on the order they were listed in.
- Grants on functions, procedures and aggregates name the overload, so a
  grant on one of several functions sharing a name is diffed and applied on
  that one instead of failing as ambiguous.
- No `REVOKE` is planned on a table, view or materialized view the same run
  drops; it used to fail on the missing relation and leave the run stuck.
- `serial` and `bigserial` columns on a fresh database: the table's
  `nextval()` default used to name a sequence that was never created.
- A managed role that exists nowhere is a clear error naming the role, not a
  raw `UndefinedObject` traceback from introspection.
- `apply --dry-run` with pending `@migrate-before` blocks reports drops it
  cannot rule out (a rename looks like drop plus add) as warnings, instead of
  failing with a `@drop` error the real run would not raise.
- A role without `CREATEDB` is told to create a plan database and pass
  `--plan-database`, instead of a bare permission error.
- A key=value connection string passed to the library no longer builds the
  plan in the target database: the plan database name is now substituted into
  it, as it already was for `postgresql://` URLs.

## [0.1.0] - 2026-10-16

First real release: a schema engine proven in production, extracted and made generic.

### Added

- Declarative schema files as the source of truth: the declared schema is built
  in a scratch plan database, introspected through `pg_catalog`, and diffed
  against the live database.
- `@migrate-before` / `@migrate-after` inline blocks for renames and
  backfills, run once per database and recorded in a hash ledger
  (`pgdeclare.applied_migrations` by default). An edited block is refused, not
  silently skipped.
- `@drop` directives: dropping a table or column that holds data needs
  explicit consent.
- Online / offline / blocked classification of every change; offline changes
  need an explicit opt-in.
- Tables, columns (identity, generated, defaults), constraints, indexes (built
  `CONCURRENTLY`), enums, composite types, views (including `WITH (...)`
  options such as `security_invoker`), functions, triggers, row level security
  and policies, and grants, roles and memberships for managed roles.
- `PgDeclareConfig`: every project-specific choice (schemas, search path,
  roles, ledger table, file layout, bootstrap SQL, locking, failure
  classification, output) in one frozen object with neutral defaults.
- The `pgdeclare` command line: `diff` (with `--json` and `--exit-code`) and
  `apply` (with `--dry-run`, `--offline`, `--force`).
- Typed package (`py.typed`).

### Hardening carried over from production use

- Retry-until-stable execution, both when loading the schema files and when
  applying changes, instead of a hand-maintained dependency order.
- A convergence check: after applying, a second diff must find nothing.
- Revokes are ordered before grants, so narrowing a table-wide privilege to
  per-column privileges cannot wipe the new column grants.
- Grants on a recreated view or function, and a SECURITY DEFINER function's
  `PUBLIC` EXECUTE revoke, are applied inside the same statement as the
  recreate, so no execution order can leave the object with the wrong ACL.
- UNIQUE and EXCLUDE constraints are never inlined into `CREATE TABLE`, where
  PostgreSQL silently discards one that duplicates the primary key.
- Recreated CHECK and FOREIGN KEY constraints are validated, so they do not
  stay `NOT VALID` and reappear in every diff.
- A server that goes away mid-run is reported as an infrastructure failure, not
  as a defect in the schema files.
- Identifiers are quoted with embedded quotes doubled, literals escaped, and
  `DO` blocks use a dollar tag their body cannot contain, so a hostile object
  name in the live catalog cannot inject SQL.
- Declared role passwords are masked in every printed, logged or JSON output
  (`Change.to_display_sql()`); only executed SQL carries them.
- Schemas the engine creates are no longer opened to `PUBLIC` unless
  `grant_public_on_created_schemas` is set.
- Objects owned by extensions are left out of introspection.
- PostgreSQL 18 `NOT NULL` constraint rows are not managed twice; nullability
  stays a column property on every version.

## [0.0.1] - 2026-10-16

Placeholder release to reserve the name on PyPI. No functionality.

[0.1.0]: https://github.com/powertime-org/pgdeclare/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/powertime-org/pgdeclare/releases/tag/v0.0.1
