# pgdeclare

pgdeclare is a Postgres schema differ.

Why we built it: [Your Database Schema Should Be Code You Can Read](https://blog.powertime.org/pgdeclare).

## Overview

**Your database schema should be code you can read.**

With migration files, reviewers read deltas, not the schema, and an LLM reading the repository has to reconstruct the state too, getting it wrong in the same places people do. pgdeclare keeps the schema as declarations: one `CREATE TABLE` per table, in plain SQL files, always describing the database as it is now. You change a table by editing its definition. pgdeclare builds the declared schema in a scratch database, diffs it against the live one through `pg_catalog`, and applies the difference.

A diff cannot infer everything, such as a rename or a backfill. For those you write small imperative `@migrate-before` / `@migrate-after` blocks next to the DDL they belong to. Each runs once per database and is recorded in a hash ledger, so an edit after it ran is refused rather than silently ignored. Destructive changes on tables that hold data need an explicit `@drop` directive.  Migrations that have run can be pruned so they don't pile up.  History still exists, in git, where history belongs.

![The schema files are built into a scratch plan database and introspected. Pending @migrate-before blocks run on the live database, which is then introspected. The two are diffed, the difference is applied, @migrate-after and DO blocks run, and a second diff must come back empty.](https://raw.githubusercontent.com/powertime-org/pgdeclare/main/docs/images/pipeline.svg)

Status: **0.1.1.** The engine has run the production schema of [Powertime](https://powertime.org) since it went into service on 10 December 2025. Backwards compatibility may not be preserved until 1.0.

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

Every command, flag and exit code is in [docs/cli.md](https://github.com/powertime-org/pgdeclare/blob/main/docs/cli.md); the library API is in [docs/python.md](https://github.com/powertime-org/pgdeclare/blob/main/docs/python.md).

## Documentation

- [Command line](https://github.com/powertime-org/pgdeclare/blob/main/docs/cli.md): flags, exit codes, `--prove`.
- [Python API](https://github.com/powertime-org/pgdeclare/blob/main/docs/python.md): the library entry points and their errors.
- [Schema files](https://github.com/powertime-org/pgdeclare/blob/main/docs/schema-files.md): directory layout, `@migrate-before` / `@migrate-after`, `@drop`, `DO` blocks.
- [Coverage](https://github.com/powertime-org/pgdeclare/blob/main/docs/coverage.md): every object kind pgdeclare manages, how, and what it leaves alone.
- [Configuration](https://github.com/powertime-org/pgdeclare/blob/main/docs/configuration.md): every `PgDeclareConfig` field.
- [Safety](https://github.com/powertime-org/pgdeclare/blob/main/docs/safety.md): previews, `--prove`, online/offline/blocked changes, quoting, masking, and the supported PostgreSQL versions (14 to 18).

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/powertime-org/pgdeclare). See [CONTRIBUTING.md](https://github.com/powertime-org/pgdeclare/blob/main/CONTRIBUTING.md) for running the tests, and [SECURITY.md](https://github.com/powertime-org/pgdeclare/blob/main/SECURITY.md) for reporting vulnerabilities.

## License

Apache License 2.0. See [LICENSE](https://github.com/powertime-org/pgdeclare/blob/main/LICENSE) and [NOTICE](https://github.com/powertime-org/pgdeclare/blob/main/NOTICE).

Extracted from [Powertime](https://powertime.org), where it manages the application's schema.
