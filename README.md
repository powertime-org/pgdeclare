# pgdeclare

pgdeclare is a Postgres schema differ.

Why we built it: [Your Database Schema Should Be Code You Can Read](https://blog.powertime.org/pgdeclare).

## Overview

**Your database schema should be code you can read.**

With migration files, reviewers read deltas, not the schema, and an LLM reading the repository has to reconstruct the state too, getting it wrong in the same places people do. pgdeclare keeps the schema as declarations: one `CREATE TABLE` per table, in plain SQL files, always describing the database as it is now. You change a table by editing its definition. pgdeclare builds the declared schema in a scratch database, diffs it against the live one through `pg_catalog`, and applies the difference.

A diff cannot infer everything, such as a rename or a backfill. For those you write small imperative `@migrate-before` / `@migrate-after` blocks next to the DDL they belong to. Each runs once per database and is recorded in a hash ledger, so an edit after it ran is refused rather than silently ignored. Destructive changes on tables that hold data need an explicit `@drop` directive.  Migrations that have run can be pruned so they don't pile up.  History still exists, in git, where history belongs.

![The schema files are built into a scratch plan database and introspected. Pending @migrate-before blocks run on the live database, which is then introspected. The two are diffed, the difference is applied, @migrate-after and DO blocks run, and a second diff must come back empty.](docs/images/pipeline.svg)

Status: **0.1.0.** The engine has run the production schema of [Powertime](https://powertime.org) since it went into service on 10 December 2025. Backwards compatibility may not be preserved until 1.0.

## Design choices

- **Postgres parses your SQL.** The desired schema is not reconstructed from a parser's model of your files. pgdeclare executes them in a scratch "plan" database and introspects the result, so defaults, implicit constraints and type normalisation come out exactly as Postgres would produce them.
- **Postgres orders the work.** Statements, and the changes themselves, run retry-until-stable: anything that fails on a missing dependency is retried after the rest, until everything succeeds or no progress is made. You do not maintain a dependency order.

## Install

```bash
pip install pgdeclare
# or, inside a uv project
uv add pgdeclare
```

Requires Python 3.12+ and a PostgreSQL 14 to 18 server.

Connect as a role that owns the target database's schemas. Each run builds your schema in a scratch plan database on the same server, which by default is created and dropped per run, so the role needs `CREATEDB`. Without it, create a plan database you own (`createdb -O app app_plan`) and pass `--plan-database app_plan`.

## Quickstart

A schema tree:

```text
sql/
└── schemas/
    ├── 01_types.sql
    └── 02_note.sql
```

```sql
-- sql/schemas/01_types.sql
CREATE TYPE mood AS ENUM ('calm', 'loud');
```

```sql
-- sql/schemas/02_note.sql
CREATE TABLE IF NOT EXISTS note (
    id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    author text NOT NULL,
    body   text NOT NULL,
    mood   mood NOT NULL DEFAULT 'calm'
);

CREATE INDEX IF NOT EXISTS note_author_idx ON note (author);
```

### Command line

```bash
export DATABASE_URL=postgresql://app:secret@localhost:5432/app

pgdeclare diff                 # what would change, as a summary plus SQL
pgdeclare diff --json          # the same, machine-readable
pgdeclare diff --exit-code     # exit 2 when anything is pending (for CI)
pgdeclare apply --dry-run      # plan only, nothing written
pgdeclare apply --dry-run --prove   # rehearse the real run on a throwaway clone
pgdeclare apply                # converge the database
```

Now add a column and a backfill:

```sql
-- sql/schemas/02_note.sql
CREATE TABLE IF NOT EXISTS note (
    id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    author text NOT NULL,
    body   text NOT NULL,
    mood   mood NOT NULL DEFAULT 'calm',
    title  text
);

CREATE INDEX IF NOT EXISTS note_author_idx ON note (author);

-- @migrate-after:2026-01-15_note_titles
UPDATE note SET title = left(body, 40) WHERE title IS NULL;
-- @end-migrate
```

`pgdeclare apply` adds the column, then runs the block once and records it.

Options shared by both commands:

| Flag | Default | Meaning |
|------|---------|---------|
| `--database-url URL` | `$DATABASE_URL` | Target database (a `postgresql://` URL) |
| `--sql-dir DIR` | `sql` | SQL root; schema files live in `DIR/schemas/` |
| `--schema NAME` | `public` | A schema to manage; repeat for several |
| `--role NAME` | none | A role whose definition, memberships and grants are managed; repeatable |
| `--manage-ownership` | off | Converge object owners the schema files state (see [What it manages](#what-it-manages)) |
| `--ledger-table SCHEMA.TABLE` | `pgdeclare.applied_migrations` | Where executed blocks are recorded |
| `--plan-database NAME` | `<db>_pgdeclare_plan_<random>` | Scratch database the declared schema is built in. Never the target database; an existing one must be owned by you and is left in place, otherwise it is created and dropped per run |
| `-v`, `--verbose` | off | Log the engine's progress |

`apply` also takes `--dry-run`, `--offline` (allow changes that lock or rewrite tables) and `--force` (bypass `@drop` checks, re-run edited blocks, continue despite non-convergence).

`diff --prove` and `apply --dry-run --prove` also rehearse the real run on a throwaway clone of the target, inline migrations included (see [Safety](#safety)). With `diff --json` the result is under a `prove` key. A failed proof exits `1` with the reason.

Exit codes: `0` success or nothing to do, `1` error, `2` changes pending (`diff --exit-code` only).

The CLI also creates the managed schemas in the target database and installs any `CREATE EXTENSION` your schema files declare in the plan database; in the target, the diff creates them. The library leaves the managed schemas, and the plan database's extensions, to the caller.

### Python

```python
from pathlib import Path

from pgdeclare import (
    PgDeclareConfig,
    apply_declared_schema,
    build_desired_schema,
    compute_diff,
    format_changes_summary,
    introspect_schema,
)

url = "postgresql://app:secret@localhost:5432/app"
config = PgDeclareConfig(schemas=("public",))

# Preview: build the declared schema in a plan database (it must exist) and diff.
desired = build_desired_schema(Path("sql"), url, "app_plan", config=config)
actual = introspect_schema(url, config=config)
changes = compute_diff(desired, actual, schemas=list(config.schemas), config=config)
print(format_changes_summary(changes))

# Converge: before-blocks, diff, apply, after-blocks, DO blocks, convergence check.
apply_declared_schema(url, Path("sql"), "app_plan", config=config)
```

`apply_declared_schema` raises `SchemaConvergenceError` if a second diff after applying still finds work, and a plain `Exception` with the reason for any other failure. `prove_declared_schema(url, Path("sql"), "app_plan", config=config)` (or `apply_declared_schema(..., dry_run=True, prove=True)`) rehearses the run on a clone; it returns a `ProveResult` and raises `ProveFailedError` with the reasons. Progress goes to the standard `logging` logger `pgdeclare`, and user-facing lines to `config.report`.

## Directives

Directives are SQL comments inside the schema files.

### `@migrate-before` / `@migrate-after`

```sql
-- @migrate-before:2026-02-01_rename_body
ALTER TABLE note RENAME COLUMN body TO content;
-- @end-migrate
```

- `@migrate-before` blocks run before the diff. Use them for renames, and for anything the diff must not see as drop-plus-add.
- `@migrate-after` blocks run after the diff. Use them for backfills on columns the diff just created.
- The id after the colon is the ledger key. It must be unique; a date prefix keeps them sorted and readable.
- A block runs once per database. Its SQL is hashed (SHA-256, first 16 hex digits). Editing a block after it ran is an error until you re-run with `--force`.
- Blocks are stripped before the schema is built, so the plan database never sees them.
- On a fresh database (no tables in the managed schemas) blocks are recorded as satisfied without running: the schema is built straight to its final state, and the upgrade path does not apply.
- Comment lines inside a block are not part of its SQL.

### `@drop`

Dropping a table or a column that contains data needs explicit consent, written inside a block:

```sql
-- @migrate-before:2026-03-01_drop_legacy
-- @drop note.legacy_flag
-- @drop old_table
SELECT 1;
-- @end-migrate
```

Columns are named `table.column` or `schema.table.column`; tables `table` or `schema.table`. Empty tables and columns can be dropped without one. `--force` overrides the check.

An extension is never dropped just because its `CREATE EXTENSION` is gone (that would cascade to every column of its types); the run notes it instead. `-- @drop extension:NAME` consents to `DROP EXTENSION NAME`, which still fails while anything uses the extension.

### `DO` blocks

Top-level `DO $$ ... $$;` blocks in the schema files run on every apply, after the diff. They must be idempotent.

## Layout

Under the SQL root (`--sql-dir`, or the `sql_dir` argument):

| Directory | Read | Holds |
|-----------|------|-------|
| `schemas/` | recursively, path order | Hand-written DDL; may carry directives |
| `functions/` | `*.sql`, flat | Extra function definitions |
| `generated/` | `*.sql`, flat | Generated SQL (policies, grants and so on) |

## What it manages

Within the managed schemas:

| Object | Create | Change | Drop |
|--------|--------|--------|------|
| Tables | yes | via columns and constraints | yes (`@drop` if it holds data) |
| Columns | yes | type, NOT NULL, default, identity, generated expression | yes (`@drop` if it holds data) |
| Constraints (PK, FK, UNIQUE, CHECK, EXCLUDE) | yes | drop and re-add | yes |
| Indexes (on tables and materialized views) | yes, `CONCURRENTLY` | drop and re-create | yes |
| Enum types | yes | add values (in position) | no (values are never removed) |
| Composite types | yes | no | no |
| Domains | yes | default, NOT NULL, CHECK constraints; drop and re-create on a base type or collation change while nothing uses it | yes, after the columns that use it |
| Sequences (standalone and serial; not identity) | yes, with every option | `ALTER SEQUENCE` options (never `RESTART`; the current value is kept); `OWNED BY` | yes, without `CASCADE` (refused while a default still uses it) |
| Views | yes | drop and re-create; `WITH (...)` options such as `security_invoker` | yes |
| Materialized views | yes, `WITH DATA` or `WITH NO DATA` as declared | query: drop and re-create (offline, rebuilds the data); storage options in place | yes |
| Functions | yes | replace, or drop and re-create when the signature changes | yes |
| Triggers | yes | drop and re-create | yes |
| Row level security | enable, force, and policies | policies drop and re-create | yes |
| Grants to managed roles | tables, views, materialized views, columns, functions, sequences, schemas | revokes before grants | yes |
| Managed roles | yes | attributes | never dropped |
| Role memberships | yes | admin option | yes |
| Extensions (`CREATE EXTENSION`) | yes, with the `SCHEMA`, `VERSION` and `CASCADE` declared | `ALTER EXTENSION ... UPDATE TO` a declared version; `SET SCHEMA` to a declared schema | only with `@drop extension:NAME`; otherwise noted |
| Foreign servers | yes, with `TYPE`, `VERSION` and options | `VERSION` and options in place (a new wrapper or `TYPE` is refused) | never dropped |
| User mappings | yes, with options | options in place; a secret option the declaration omits is left alone | a managed role's mapping on a declared server |
| Foreign tables | yes, with server, options, columns and column options | columns (type, NOT NULL, default, options) and options in one `ALTER FOREIGN TABLE`; a new server drops and re-creates | yes (no rows live locally, so no `@drop`) |
| Owners (opt-in, `manage_ownership`) | `ALTER ... OWNER TO` | tables, foreign tables, views, materialized views, sequences, functions, types, domains, managed schemas | no |
| Comments (`COMMENT ON`) | tables, columns (of tables and views), views, functions, indexes, constraints, triggers, policies, enum and composite types | text; written again after the object is recreated | `IS NULL` when the declaration is removed |

Not managed: foreign data wrappers, comments on schemas, domains, sequences, materialized views and other unmanaged objects, owners not stated in the schema files (or with `manage_ownership` off), and anything outside the managed schemas. Tables whose names start with `_` are ignored. Objects owned by an extension are never introspected.

An owner is managed only when the schema files state it with `ALTER ... OWNER TO`, the owner is a managed role, and `manage_ownership` is on; the role that builds the plan cannot be stated as an owner, since everything it creates is already its own. An owned object's implicit privileges are not diffed as grants to its owner. Sequences linked to a column follow their table's owner.

A comment the schema files do not declare is removed, including one added by hand. Comments are written in a phase of their own after the DDL and index builds, so an object dropped and recreated in the same run gets its comment back.

Grants on a recreated view or function are replayed inside the same statement as the recreate, so no execution order can leave the object without its ACL. A recreated materialized view rebuilds its indexes the same way, and a recreated view or materialized view drops the views and materialized views that read from it and builds them back, from their declarations, in that statement too. Whether a materialized view is populated is not compared: a `REFRESH` is runtime state, not drift. When a set of SECURITY DEFINER functions is passed as `locked_functions`, their default `PUBLIC` EXECUTE is revoked atomically with their creation.

## Configuration

Everything project-specific is a field of the frozen `PgDeclareConfig`, passed to the entry points. The defaults describe a plain database.

| Field | Default | Meaning |
|-------|---------|---------|
| `schemas` | `("public",)` | Schemas introspected and diffed |
| `search_path` | `("public",)` | `search_path` used to load SQL and introspect types |
| `managed_roles` | `()` | Roles whose definitions, memberships and grants are diffed; others are never touched |
| `ledger_table` | `"pgdeclare.applied_migrations"` | `schema.table` of the inline-migration ledger |
| `schemas_subdir` | `"schemas"` | Hand-written schema files, recursive |
| `functions_subdir` | `"functions"` | Extra function files, flat |
| `generated_subdir` | `"generated"` | Generated SQL, flat |
| `git_sql_prefix` | `"sql"` | SQL root relative to the repository, for reading another branch with git |
| `bootstrap_sql_files` | `()` | SQL files loaded before the schema files when building the plan |
| `bootstrap_source_label` | `"<bootstrap>"` | Source label for bootstrap statements in errors |
| `plan_recreate_schemas` | `("public",)` | Schemas dropped and recreated in the plan database before each build |
| `externally_provisioned_schemas` | `()` | `CREATE SCHEMA` for these is skipped when building the plan |
| `secdef_schemas` | `("public",)` | Schemas whose SECURITY DEFINER functions may be locked |
| `manage_ownership` | `False` | Converge owners the schema files state with `ALTER ... OWNER TO` (a managed role only); others stay unmanaged |
| `grant_public_on_created_schemas` | `False` | `GRANT ALL ... TO PUBLIC` on every schema the engine creates (`ensure_schemas`, the plan database). Off by default: anyone who can connect could create objects there |
| `dry_run` | `False` | Global preview: nothing written, introspection returns an empty schema |
| `lock` | `nullcontext` | Context-manager factory around every plan build |
| `classify_failure` | none | Turns "the server went away" into your own exception instead of a schema error |
| `decorate` | plain text | `(style, text) -> text` for user-facing output (colour, say) |
| `report` | logger at INFO | Where user-facing progress lines go |
| `on_migration_applied` | `None` | Called with `(block, connection_string)` after a block runs |

The plan database must be on the same server as the target and must already exist; the library rebuilds `plan_recreate_schemas` in it on every run, and drops the foreign servers it can (with their user mappings) so a reused plan database keeps none that are no longer declared. `CREATE EXTENSION` statements in schema files are skipped when building the plan, so provision extensions in the plan database yourself (or list them in `bootstrap_sql_files`); the declarations are read from the files, and the diff creates them in the target.

## Safety

- **Preview first.** `pgdeclare diff` and `apply --dry-run` write nothing to the target database. Declared roles are still created on the server when the plan is built, since roles are cluster-wide. A dry run cannot show the effect of pending `@migrate-before` blocks, and says so: a rename shows as a drop plus an add, reported as a warning rather than a `@drop` error. `--prove` can.
- **Prove on a clone.** `--prove` copies the target with `CREATE DATABASE <db>_pgdeclare_prove_<random> TEMPLATE <db>` on the same server and runs the whole real pipeline there with the same `--offline` and `--force`: `@migrate-before` blocks, diff, `@drop` checks, DDL, `@migrate-after` blocks, `DO` blocks. It reports the ledger entries and the DDL the real run would write, then fails if a second diff still finds work (even with `--force`) or if any table lost rows or vanished without a `@drop` directive naming it. Tables are matched by OID, so a rename is not a loss. The ledger is written only in the clone, the clone is always dropped, and nothing is written to the target. Its costs:
  - **Exclusive access.** Postgres refuses to copy a database other sessions are connected to, so run it in a quiet window or against a staging copy. A busy production database will fail with a clear error.
  - **Disk and time.** The clone is a full copy of the target, so it needs as much free disk, takes time proportional to its size, and the row check counts every managed table exactly.
  - **Privileges.** The role needs `CREATEDB` and must own the target database (or be a superuser).
  - **Cluster-wide changes are not rehearsed.** Role and role-membership changes would apply to the whole server, so they are listed, not run.
- **Online, offline, blocked.** Every change is classified. Online changes apply directly. Offline changes (SET NOT NULL on a column or a domain, column type changes, a new generation expression) lock or rewrite tables and need `--offline` (`offline=True`). Blocked changes (drops of data) need `@drop`.
- **One migration at a time.** The DDL phase takes a session advisory lock on the target database and fails fast if another run holds it.
- **Transactions.** Enum changes commit first (Postgres requires it); the rest of the DDL runs in one transaction with savepoints, so a stuck run changes nothing. Index builds run `CONCURRENTLY` afterwards; an invalid index left by a failed build is dropped and retried.
- **Names are data.** Every identifier written into SQL is quoted with its embedded quotes doubled, every literal escaped, and `DO` blocks use a dollar tag their body cannot contain, so an object name chosen by a less privileged role cannot inject SQL into the migrator's session.
- **Passwords are not shown.** A role password declared in the schema files is executed but masked as `PASSWORD '***'` in `diff`, `diff --json`, dry-run output and logs (`Change.to_display_sql()`). Roles are cluster-wide, so building the plan database on the same server creates declared roles there too.
- **User-mapping options are not shown.** Their values may hold a password, so every option value except `user` and `password_required` is masked as `'***'` in the same places, and in the statements `--prove` reports; plan descriptions name the options, never their values. A secret option the declaration does not set (a password provisioned by hand, say) is left as it is on the target.
- **Convergence check.** After applying, the database is diffed again. Anything left over fails the run, because one run must reach the declared schema.

## Supported PostgreSQL versions

The suite runs against PostgreSQL 14, 15, 16, 17 and 18 in CI.

- `security_invoker` views need PostgreSQL 15 or later.
- On 14, `public` in a new database belongs to the bootstrap superuser, so a non-superuser needs `ALTER SCHEMA public OWNER TO <role>` in its plan database.
- Changing the expression of a generated column uses `ALTER COLUMN ... SET EXPRESSION`, which needs PostgreSQL 17 or later. On 14 to 16, do that change by hand in a `@migrate-before` block.
- PostgreSQL 18 records `NOT NULL` as a constraint; pgdeclare models it on the column on every version, so plans are identical.

## How it compares

As of October 2026. [pgschema](https://github.com/pgplex/pgschema) and [pg-delta](https://github.com/supabase/pg-toolbelt) cover more object types than pgdeclare 0.1.0 (see [What it manages](#what-it-manages)). pgdeclare's difference is where the steps a diff cannot infer live: next to the DDL they belong to, run once, and recorded.

| Tool | Renames and backfills | Records what ran | Free licence | Notes |
|------|-----------------------|------------------|--------------|-------|
| pgdeclare | `@migrate` blocks beside the DDL | Hash ledger; edited blocks are refused | Apache-2.0 | Re-diffs after applying and fails the run if anything is left |
| [pgschema](https://github.com/pgplex/pgschema) | Outside the tool | No; state only | Apache-2.0 | Terraform-style plan and apply; broad object coverage |
| [pg-delta](https://github.com/supabase/pg-toolbelt) | Renames come out as drop plus add, edited by hand; never writes row changes | Through Supabase migration files | PostgreSQL | Alpha; very broad coverage; `prove` tests a plan on a clone first |
| [Atlas](https://atlasgo.io) | Versioned migration files | Yes, in versioned mode | Apache-2.0 Community Edition | The Community Edition omits views, functions, triggers and RLS policies; the full binary is under the Atlas MSA |
| [pg-schema-diff](https://github.com/stripe/pg-schema-diff) | Renames come out as drop plus add | No | MIT | Strong lock avoidance for online changes |
| [migra](https://github.com/djrobstep/migra) | Not handled | No | Unlicense | Deprecated; last release 2022 |

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/powertime-org/pgdeclare). See [CONTRIBUTING.md](CONTRIBUTING.md) for running the tests, and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Extracted from [Powertime](https://powertime.org), where it manages the application's schema.
