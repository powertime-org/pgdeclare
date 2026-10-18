"""
Declarative schema diff engine for PostgreSQL.

Schema files declare the desired state; the engine builds that state in a
scratch "plan" database, introspects both it and the live database through
pg_catalog, and plans the changes between them:

- Change categorization (online, offline, blocked)
- @drop directive enforcement for data loss protection
- Two-phase index operations (CONCURRENTLY outside the transaction)
- Retry-until-stable execution instead of hand-maintained dependency order
- Inline @migrate-before / @migrate-after blocks with a hash-checked ledger

Everything project-specific (managed schemas, search_path, roles, ledger table,
file layout, bootstrap SQL, locking, failure classification) comes in through
`PgDeclareConfig`. The package imports nothing from the application that
embeds it.
"""

__version__ = "0.1.0"

from .config import DEFAULT_CONFIG, PgDeclareConfig
from .models import (
    ChangeCategory,
    Column,
    Comment,
    Constraint,
    Dependency,
    Domain,
    EnumType,
    Function,
    Grant,
    Index,
    MaterializedView,
    Policy,
    Role,
    RoleMembership,
    Schema,
    Sequence,
    Table,
    Trigger,
    View,
)
from .introspection import (
    introspect_schema,
    check_table_has_data,
    get_invalid_indexes,
)
from .differ import (
    compute_diff,
    categorize_changes,
    split_index_operations,
)
from .executor import (
    execute_migration,
    cleanup_invalid_indexes,
    format_changes_summary,
)
from .drop_directive import (
    validate_drops_runtime,
    validate_drops_ci,
    format_drop_errors,
)
from .schema_parser import (
    build_desired_schema,
    build_schema_from_git_branch,
    ensure_schemas,
    execute_do_blocks,
    get_connection_string,
    git_list_sql_files_dfs,
    git_read_file,
    parse_do_blocks,
    strip_migrate_blocks,
)
from .ledger import (
    MigrationBlock,
    delete_ledger_entries,
    ensure_ledger,
    execute_migration_block,
    get_applied_migrations,
    hash_block_sql,
    list_ledger,
    mark_blocks_applied,
    parse_migration_blocks,
    parse_migration_blocks_from_content,
    parse_migration_blocks_from_git,
)
from .migrate import (
    SchemaConvergenceError,
    apply_declared_schema,
    assert_schema_converged,
    compute_residual_changes,
    report_pending_migration_blocks,
)

__all__ = [
    "__version__",
    # Configuration
    "DEFAULT_CONFIG",
    "PgDeclareConfig",
    # Models
    "ChangeCategory",
    "Column",
    "Comment",
    "Constraint",
    "Dependency",
    "Domain",
    "EnumType",
    "Function",
    "Grant",
    "Index",
    "MaterializedView",
    "Policy",
    "Role",
    "RoleMembership",
    "Schema",
    "Sequence",
    "Table",
    "Trigger",
    "View",
    # Introspection
    "introspect_schema",
    "check_table_has_data",
    "get_invalid_indexes",
    # Differ
    "compute_diff",
    "categorize_changes",
    "split_index_operations",
    # Executor
    "execute_migration",
    "cleanup_invalid_indexes",
    "format_changes_summary",
    # Drop directive
    "validate_drops_runtime",
    "validate_drops_ci",
    "format_drop_errors",
    # Schema parser
    "build_desired_schema",
    "build_schema_from_git_branch",
    "ensure_schemas",
    "execute_do_blocks",
    "get_connection_string",
    "git_list_sql_files_dfs",
    "git_read_file",
    "parse_do_blocks",
    "strip_migrate_blocks",
    # Inline migrations and their ledger
    "MigrationBlock",
    "delete_ledger_entries",
    "ensure_ledger",
    "execute_migration_block",
    "get_applied_migrations",
    "hash_block_sql",
    "list_ledger",
    "mark_blocks_applied",
    "parse_migration_blocks",
    "parse_migration_blocks_from_content",
    "parse_migration_blocks_from_git",
    # Pipeline
    "SchemaConvergenceError",
    "apply_declared_schema",
    "assert_schema_converged",
    "compute_residual_changes",
    "report_pending_migration_blocks",
]
