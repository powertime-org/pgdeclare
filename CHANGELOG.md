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

### Changed

- README: the plan database's privilege needs, the PostgreSQL 14 `public`
  ownership caveat, a corrected Python example, and a comparison with other
  tools.

### Fixed

- `CREATE TABLE` quotes inline constraint names, so a name with a quote in it
  stays a name instead of breaking (or injecting into) the statement.
- A `DROP` that fails because another object still depends on it
  (`2BP01`) is retried after the rest, like a missing dependency, so dropping
  a stack of views no longer depends on the order they were listed in.
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
