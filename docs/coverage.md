# Coverage

## What it manages

Within the managed schemas:

| Object | Create | Change | Drop |
|--------|--------|--------|------|
| Tables | yes | via columns and constraints | yes (`@drop` if it holds data) |
| Partitioned tables (`PARTITION BY RANGE`, `LIST`, `HASH`) | yes | via columns and constraints, which Postgres carries to every partition; never the partition key (refused) | yes (`@drop` if it holds data), with all its partitions |
| Partitions (`PARTITION OF ... FOR VALUES` or `DEFAULT`, sub-partitioned too) | yes (offline beside a default partition, which Postgres scans) | bound: `DETACH`, then `ATTACH` (offline: `ATTACH` scans); own defaults, NOT NULL and constraints | never: an undeclared partition is left in place and noted |
| Columns | yes | type, collation, NOT NULL, default, identity, generated expression | yes (`@drop` if it holds data) |
| Constraints (PK, FK, UNIQUE, CHECK, EXCLUDE) | yes | drop and re-add | yes |
| Indexes (on tables, partitioned tables and materialized views) | yes, `CONCURRENTLY` (on a partitioned table: per partition, then attached) | drop and re-create | yes (on a partitioned table without `CONCURRENTLY`, which Postgres refuses there) |
| Enum types | yes | add values (in position) | no (values are never removed) |
| Composite types | yes | no | no |
| Range types (with their multirange) | yes, with every option (a shell type first when there is a canonical function) | drop and re-create while nothing uses the range or its multirange | yes, after the columns that use it (with its canonical function) |
| Collations (libc, ICU, builtin) | yes | drop and re-create while nothing uses it | yes, after the columns, domains and ranges that use it |
| Domains | yes | default, NOT NULL, CHECK constraints; drop and re-create on a base type or collation change while nothing uses it | yes, after the columns that use it |
| Sequences (standalone and serial; not identity) | yes, with every option | `ALTER SEQUENCE` options (never `RESTART`; the current value is kept); `OWNED BY` | yes, without `CASCADE` (refused while a default still uses it) |
| Views | yes | drop and re-create; `WITH (...)` options such as `security_invoker` | yes |
| Materialized views | yes, `WITH DATA` or `WITH NO DATA` as declared | query: drop and re-create (offline, rebuilds the data); storage options in place | yes |
| Functions | yes | replace, or drop and re-create when the signature changes | yes |
| Procedures and window functions | yes | as functions; drop and re-create when the kind changes | yes |
| Aggregates (normal, ordered-set, hypothetical-set) | yes, after the functions they call | `CREATE OR REPLACE AGGREGATE`; drop and re-create when the result type or kind changes, or when a function it calls is dropped | yes |
| Triggers | yes | drop and re-create | yes |
| Row level security | enable, force, and policies | policies drop and re-create | yes |
| Grants to managed roles | tables, views, materialized views, columns, functions, procedures, aggregates, sequences, schemas | revokes before grants | yes |
| Default privileges (`ALTER DEFAULT PRIVILEGES`) to managed roles | on tables, sequences, functions, types and schemas; for any target role (`FOR ROLE`, else the connecting role); schema-less or `IN SCHEMA` a managed schema | revoke and re-grant | yes |
| Managed roles | yes | attributes | never dropped |
| Role memberships | yes | admin option | yes |
| Extensions (`CREATE EXTENSION`) | yes, with the `SCHEMA`, `VERSION` and `CASCADE` declared | `ALTER EXTENSION ... UPDATE TO` a declared version; `SET SCHEMA` to a declared schema | only with `@drop extension:NAME`; otherwise noted |
| Foreign servers | yes, with `TYPE`, `VERSION` and options | `VERSION` and options in place (a new wrapper or `TYPE` is refused) | never dropped |
| User mappings | yes, with options | options in place; a secret option the declaration omits is left alone | a managed role's mapping on a declared server |
| Foreign tables | yes, with server, options, columns and column options | columns (type, NOT NULL, default, options) and options in one `ALTER FOREIGN TABLE`; a new server drops and re-creates | yes (no rows live locally, so no `@drop`) |
| Owners (opt-in, `manage_ownership`) | `ALTER ... OWNER TO` | tables (partitioned ones too), foreign tables, views, materialized views, sequences, functions, procedures, aggregates, enum, composite and range types, domains, collations, managed schemas | no |
| Rewrite rules (`CREATE RULE`, on tables; never a view's `_RETURN` rule) | yes | `CREATE OR REPLACE RULE`; enabled state | yes |
| Publications (database-level) | yes: tables, column lists and row filters (15+), `TABLES IN SCHEMA` (15+), `FOR ALL TABLES`, options | `ALTER PUBLICATION ... ADD` / `DROP` members and `SET (...)` options; a member with a new column list or row filter is dropped and added back; drop and re-create to switch `FOR ALL TABLES` | yes, when every table it publishes is managed (see below) |
| Event triggers (database-level; superuser) | yes, with `WHEN TAG IN` | drop and re-create; enabled state with `ALTER EVENT TRIGGER` | yes, when its function is in a managed schema |
| Comments (`COMMENT ON`) | tables, columns (of tables and views), views, functions, procedures, aggregates, indexes, constraints, triggers, policies, enum, composite and range types, collations | text; written again after the object is recreated | `IS NULL` when the declaration is removed |

Not managed: subscriptions (they hold connection strings and secrets, and creating one starts replication); foreign data wrappers; operators, operator classes and families; casts; text search configurations, dictionaries, parsers and templates; statistics objects (`CREATE STATISTICS`); languages; transforms; security labels; table inheritance (`INHERITS`) and replica identity; role settings (`ALTER ROLE ... SET`); collation versions (`ALTER COLLATION ... REFRESH VERSION`); partitions the schema files do not declare; comments on objects the Comments row does not list (schemas, domains, sequences, materialized views, foreign tables and the rest); owners not stated in the schema files (or with `manage_ownership` off); grants to roles that are not managed; default privileges for grantees that are not managed roles (`PUBLIC` included, so `REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC` is not) or that take away a target role's own built-in privileges; and anything outside the managed schemas. Tables whose names start with `_` are ignored. Objects owned by an extension are never introspected.

Partitions created at runtime (by pg_partman or a cron job, say) are never dropped. A live partition the schema files do not declare is left in place with everything on it, and `diff` and `apply` print a note naming it (`notes` in `diff --json`); to drop one, write the `DROP TABLE` in a `@migrate-before` block. An index on a partitioned table is created `ON ONLY` the parent, built `CONCURRENTLY` on each partition (the undeclared ones too) and attached, so no partition is locked for the build; a run that dies half way is finished by the next. Postgres has no `ALTER` that turns a table into a partitioned one, back, or changes a partition key: pgdeclare refuses those with an error, and the conversion goes in a `@migrate-before` block. A foreign key on a partitioned table is validated as it is added, which is offline, since PostgreSQL before 18 cannot add one `NOT VALID`.

An owner is managed only when the schema files state it with `ALTER ... OWNER TO`, the owner is a managed role, and `manage_ownership` is on; the role that builds the plan cannot be stated as an owner, since everything it creates is already its own. An owned object's implicit privileges are not diffed as grants to its owner; that applies to that exact object only, so a grant to the same role on another object of the same name, or on another overload of a function, is still diffed. Sequences linked to a column follow their table's owner.

A comment the schema files do not declare is removed, including one added by hand. Comments are written in a phase of their own after the DDL and index builds, so an object dropped and recreated in the same run gets its comment back.

Declared default privileges run first when the plan is built, whatever file they are in, so every declared object they cover gets the grant, as it would on a database that has had them since its first apply. Defaults no longer declared are revoked before anything is created and new ones granted after, so an object created in the same run gets exactly the grants the plan gives it. The target role is always written as `FOR ROLE`, so one omitted in the schema files means the role pgdeclare connects as.

Publications and event triggers belong to the database rather than a schema, so pgdeclare decides which live ones are its own. A declared one is always converged. An undeclared publication is dropped only when it publishes at least one table, every table it publishes is a table pgdeclare manages (in a managed schema, not `_`-prefixed, not owned by an extension; partitioned tables and their declared partitions count, a partition the schema files do not declare does not) and every `TABLES IN SCHEMA` schema is a managed one. `FOR ALL TABLES` publications, empty ones and any that reach further are left alone unless declared. An undeclared event trigger is dropped only when its function is in a managed schema. Creating an event trigger needs a superuser, both on the target and in the plan database. Event triggers are built last, in the plan and in the run, so they do not fire on the rest of the DDL; a function recreated with `DROP ... CASCADE` gets its event triggers back in the same run.

Grants on a recreated view or function are replayed inside the same statement as the recreate, so no execution order can leave the object without its ACL. A recreated materialized view rebuilds its indexes the same way, and a recreated view or materialized view drops the views and materialized views that read from it and builds them back, from their declarations, in that statement too. Whether a materialized view is populated is not compared: a `REFRESH` is runtime state, not drift. When a set of SECURITY DEFINER functions is passed as `locked_functions`, their default `PUBLIC` EXECUTE is revoked atomically with their creation.

## Compared with other tools

As of October 2026. Against [pg-delta's coverage list](https://github.com/supabase/pg-toolbelt/blob/main/packages/pg-delta/COVERAGE.md), pgdeclare manages the same object kinds except subscriptions (deliberately unmanaged), foreign data wrappers, security labels, table inheritance (`INHERITS`), replica identity, role settings and CHECK constraints on foreign tables; its comments and grants reach fewer kinds of object, and grants go only to managed roles. It manages every object kind [pgschema](https://github.com/pgplex/pgschema) lists, except grants on types and domains, `WITH GRANT OPTION`, and comments on materialized views.

See the [README](../README.md#how-it-compares) for the overall comparison.

[Back to the README](../README.md)
