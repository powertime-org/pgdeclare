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
# or
uv add pgdeclare
```

Requires Python 3.12+ and a PostgreSQL 14 to 18 server.

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
| `--ledger-table SCHEMA.TABLE` | `pgdeclare.applied_migrations` | Where executed blocks are recorded |
| `--plan-database NAME` | `<db>_pgdeclare_plan_<random>` | Scratch database the declared schema is built in. Never the target database; an existing one must be owned by you and is left in place, otherwise it is created and dropped per run |
| `-v`, `--verbose` | off | Log the engine's progress |

`apply` also takes `--dry-run`, `--offline` (allow changes that lock or rewrite tables) and `--force` (bypass `@drop` checks, re-run edited blocks, continue despite non-convergence).

Exit codes: `0` success or nothing to do, `1` error, `2` changes pending (`diff --exit-code` only).

The CLI also creates the managed schemas in the target database and installs any `CREATE EXTENSION` your schema files declare, in both the plan and the target database. The library leaves both to the caller.

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
config = PgDeclareConfig(schemas=("public",), managed_roles=("app_reader",))

# Preview: build the declared schema in a plan database (it must exist) and diff.
desired = build_desired_schema(Path("sql"), url, "app_plan", config=config)
actual = introspect_schema(url, config=config)
changes = compute_diff(desired, actual, schemas=list(config.schemas), config=config)
print(format_changes_summary(changes))

# Converge: before-blocks, diff, apply, after-blocks, DO blocks, convergence check.
apply_declared_schema(url, Path("sql"), "app_plan", config=config)
```

`apply_declared_schema` raises `SchemaConvergenceError` if a second diff after applying still finds work, and a plain `Exception` with the reason for any other failure. Progress goes to the standard `logging` logger `pgdeclare`, and user-facing lines to `config.report`.

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
| Views | yes | drop and re-create; `WITH (...)` options such as `security_invoker` | yes |
| Materialized views | yes, `WITH DATA` or `WITH NO DATA` as declared | query: drop and re-create (offline, rebuilds the data); storage options in place | yes |
| Functions | yes | replace, or drop and re-create when the signature changes | yes |
| Triggers | yes | drop and re-create | yes |
| Row level security | enable, force, and policies | policies drop and re-create | yes |
| Grants to managed roles | tables, views, materialized views, columns, functions, schemas | revokes before grants | yes |
| Managed roles | yes | attributes | never dropped |
| Role memberships | yes | admin option | yes |

Not managed: extensions, sequences outside identity columns, domains, comments, ownership, and anything outside the managed schemas. Tables whose names start with `_` are ignored. Objects owned by an extension are never introspected.

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
| `grant_public_on_created_schemas` | `False` | `GRANT ALL ... TO PUBLIC` on every schema the engine creates (`ensure_schemas`, the plan database). Off by default: anyone who can connect could create objects there |
| `dry_run` | `False` | Global preview: nothing written, introspection returns an empty schema |
| `lock` | `nullcontext` | Context-manager factory around every plan build |
| `classify_failure` | none | Turns "the server went away" into your own exception instead of a schema error |
| `decorate` | plain text | `(style, text) -> text` for user-facing output (colour, say) |
| `report` | logger at INFO | Where user-facing progress lines go |
| `on_migration_applied` | `None` | Called with `(block, connection_string)` after a block runs |

The plan database must be on the same server as the target and must already exist; the library rebuilds `plan_recreate_schemas` in it on every run. `CREATE EXTENSION` statements in schema files are skipped when building the plan, so provision extensions in the plan database yourself (or list them in `bootstrap_sql_files`).

## Safety

- **Preview first.** `pgdeclare diff` and `apply --dry-run` write nothing to the target database. A dry run cannot show the effect of pending `@migrate-before` blocks, and says so.
- **Online, offline, blocked.** Every change is classified. Online changes apply directly. Offline changes (SET NOT NULL, column type changes, a new generation expression) lock or rewrite tables and need `--offline` (`offline=True`). Blocked changes (drops of data) need `@drop`.
- **One migration at a time.** The DDL phase takes a session advisory lock on the target database and fails fast if another run holds it.
- **Transactions.** Enum changes commit first (Postgres requires it); the rest of the DDL runs in one transaction with savepoints, so a stuck run changes nothing. Index builds run `CONCURRENTLY` afterwards; an invalid index left by a failed build is dropped and retried.
- **Names are data.** Every identifier written into SQL is quoted with its embedded quotes doubled, every literal escaped, and `DO` blocks use a dollar tag their body cannot contain, so an object name chosen by a less privileged role cannot inject SQL into the migrator's session.
- **Passwords are not shown.** A role password declared in the schema files is executed but masked as `PASSWORD '***'` in `diff`, `diff --json`, dry-run output and logs (`Change.to_display_sql()`). Roles are cluster-wide, so building the plan database on the same server creates declared roles there too.
- **Convergence check.** After applying, the database is diffed again. Anything left over fails the run, because one run must reach the declared schema.

## Supported PostgreSQL versions

The suite runs against PostgreSQL 14, 15, 16, 17 and 18 in CI.

- `security_invoker` views need PostgreSQL 15 or later.
- Changing the expression of a generated column uses `ALTER COLUMN ... SET EXPRESSION`, which needs PostgreSQL 17 or later. On 14 to 16, do that change by hand in a `@migrate-before` block.
- PostgreSQL 18 records `NOT NULL` as a constraint; pgdeclare models it on the column on every version, so plans are identical.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/powertime-org/pgdeclare). See [CONTRIBUTING.md](CONTRIBUTING.md) for running the tests, and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Extracted from [Powertime](https://powertime.org), where it manages the application's schema.
