# Configuration

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

[Back to the README](../README.md)
