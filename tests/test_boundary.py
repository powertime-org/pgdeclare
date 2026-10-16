"""The package stays generic.

pgdeclare was extracted from an application. It must not carry any
application's names, must import nothing beyond the standard library and its
two declared dependencies, and its defaults must describe a plain database.
Everything project-specific arrives through `PgDeclareConfig`.

An embedding project can assert that none of its own names (schemas, roles,
product names) leaked into the package by listing them, comma-separated, in
PGDECLARE_FORBIDDEN_NAMES when it runs this suite.
"""

import ast
import os
import re
import sys
from pathlib import Path

import pytest

import pgdeclare

PACKAGE = Path(pgdeclare.__file__).parent
FILES = sorted(PACKAGE.rglob("*.py"))
SOURCES = sorted(
    p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
)

# Names no generic package should carry, plus whatever the environment adds.
NAMES = ["acme_internal"] + [
    n.strip()
    for n in os.environ.get("PGDECLARE_FORBIDDEN_NAMES", "").split(",")
    if n.strip()
]
FORBIDDEN = re.compile("|".join(re.escape(n) for n in NAMES), re.IGNORECASE)
THIRD_PARTY = {"pglast", "psycopg2"}


def _imports(path: Path) -> list[tuple[int, str]]:
    """(level, module) for every import in `path`. level 0 is absolute."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.level, node.module or ""))
    return found


def _allowed_root(module: str) -> bool:
    """An absolute import may reach the stdlib, the package, or a declared dependency."""
    root = module.split(".")[0]
    return (
        root == "pgdeclare"
        or root in THIRD_PARTY
        or root in sys.stdlib_module_names
        or root == "__future__"
    )


def _depth(path: Path) -> int:
    """How many relative levels stay inside the package from `path`."""
    return len(path.relative_to(PACKAGE).parts)


def test_the_package_has_files():
    assert PACKAGE.is_dir() and FILES
    assert (PACKAGE / "py.typed").is_file()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_imports_are_stdlib_the_package_or_declared_dependencies(path):
    for level, module in _imports(path):
        assert level <= _depth(path), (
            f"{path.name} imports {'.' * level}{module}, outside the package"
        )
        if level == 0:
            assert _allowed_root(module), f"{path.name} imports {module}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_application_names_in_the_package(path):
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = FORBIDDEN.search(line)
        assert match is None, (
            f"{path.name}:{number} names an application ({match.group(0)!r}): "
            f"{line.strip()}"
        )


def test_the_guard_can_fail():
    assert FORBIDDEN.search("SET search_path TO public, acme_internal")
    assert FORBIDDEN.search("GRANT SELECT ON t TO ACME_INTERNAL")
    probe = PACKAGE / "probe.py"
    assert not _allowed_root("requests.adapters")
    assert not _allowed_root("rich")
    assert _allowed_root("psycopg2.extras") and _allowed_root("json")
    tree = ast.parse("from ...elsewhere import x\n")
    levels = [n.level for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert max(levels) > _depth(probe)


def test_the_default_config_is_neutral():
    config = pgdeclare.DEFAULT_CONFIG
    assert config.schemas == ("public",)
    assert config.search_path == ("public",)
    assert config.managed_roles == ()
    assert config.bootstrap_sql_files == ()
    assert config.externally_provisioned_schemas == ()
    assert config.ledger_table == "pgdeclare.applied_migrations"
    assert config.dry_run is False


def test_the_public_api_is_exported():
    for name in pgdeclare.__all__:
        assert hasattr(pgdeclare, name), name
    assert pgdeclare.__version__ == "0.1.0"
