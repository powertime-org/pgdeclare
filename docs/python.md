# Python API

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

[Back to the README](../README.md)
