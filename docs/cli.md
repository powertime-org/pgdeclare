# Command line

The commands are shown in the [README quickstart](../README.md#quickstart).

Options shared by both commands:

| Flag | Default | Meaning |
|------|---------|---------|
| `--database-url URL` | `$DATABASE_URL` | Target database (a `postgresql://` URL) |
| `--sql-dir DIR` | `sql` | SQL root; schema files live in `DIR/schemas/` |
| `--schema NAME` | `public` | A schema to manage; repeat for several |
| `--role NAME` | none | A role whose definition, memberships and grants are managed; repeatable |
| `--manage-ownership` | off | Converge object owners the schema files state (see [What it manages](coverage.md#what-it-manages)) |
| `--ledger-table SCHEMA.TABLE` | `pgdeclare.applied_migrations` | Where executed blocks are recorded |
| `--plan-database NAME` | `<db>_pgdeclare_plan_<random>` | Scratch database the declared schema is built in. Never the target database; an existing one must be owned by you and is left in place, otherwise it is created and dropped per run |
| `-v`, `--verbose` | off | Log the engine's progress |

`apply` also takes `--dry-run`, `--offline` (allow changes that lock or rewrite tables) and `--force` (bypass `@drop` checks, re-run edited blocks, continue despite non-convergence).

`diff --prove` and `apply --dry-run --prove` also rehearse the real run on a throwaway clone of the target, inline migrations included (see [Safety](safety.md#safety)). With `diff --json` the result is under a `prove` key. A failed proof exits `1` with the reason.

Exit codes: `0` success or nothing to do, `1` error, `2` changes pending (`diff --exit-code` only).

The CLI also creates the managed schemas in the target database and installs any `CREATE EXTENSION` your schema files declare in the plan database; in the target, the diff creates them. The library leaves the managed schemas, and the plan database's extensions, to the caller.

[Back to the README](../README.md)
