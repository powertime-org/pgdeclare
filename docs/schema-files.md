# Schema files

## Layout

Under the SQL root (`--sql-dir`, or the `sql_dir` argument):

| Directory | Read | Holds |
|-----------|------|-------|
| `schemas/` | recursively, path order | Hand-written DDL; may carry directives |
| `functions/` | `*.sql`, flat | Extra function definitions |
| `generated/` | `*.sql`, flat | Generated SQL (policies, grants and so on) |

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

[Back to the README](../README.md)
