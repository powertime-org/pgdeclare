# Safety

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
- Publication column lists, row filters and `TABLES IN SCHEMA` need PostgreSQL 15 or later; `publish_generated_columns` needs 18.
- PostgreSQL 18 records `NOT NULL` as a constraint; pgdeclare models it on the column on every version, so plans are identical.

[Back to the README](../README.md)
