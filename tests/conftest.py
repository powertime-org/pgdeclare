"""Shared fixtures for the pgdeclare test suite.

Unit tests need nothing. Integration tests need a PostgreSQL server: set
TEST_DATABASE_URL to a postgresql:// URL whose role can create databases,
schemas and roles, and they run; leave it unset and they skip.

Under pytest-xdist every worker gets a database of its own, created from
TEST_DATABASE_URL's server and dropped at the end. Workers that shared one
database would contend for the migration advisory lock (which is per database)
and for the ledger table, and fail at random.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import patch
from urllib.parse import urlsplit

import psycopg2
import pytest


def _with_database(url: str, name: str) -> str:
    return urlsplit(url)._replace(path=f"/{name}").geturl()


def _admin(url: str, sql: str) -> None:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


@pytest.fixture(scope="session", autouse=True)
def _per_worker_database() -> Iterator[None]:
    """Point TEST_DATABASE_URL at a database owned by this xdist worker."""
    url = os.environ.get("TEST_DATABASE_URL")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not url or not worker:
        yield
        return

    name = f"pgdeclare_test_{worker}_{uuid.uuid4().hex[:6]}"
    _admin(url, f'CREATE DATABASE "{name}"')
    os.environ["TEST_DATABASE_URL"] = _with_database(url, name)
    try:
        yield
    finally:
        os.environ["TEST_DATABASE_URL"] = url
        _admin(url, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


# =============================================================================
# Mock Database Connection Fixtures
# =============================================================================


@dataclass
class MockCursor:
    """Mock database cursor for unit tests."""

    rows: list[tuple] = field(default_factory=list)
    # DB-API leaves description None outside a result set; keep that rather
    # than inventing an empty list the real driver would never return.
    description: Optional[list[tuple]] = None
    rowcount: int = 0
    _executed_queries: list[str] = field(default_factory=list)
    _error_on_execute: Optional[Exception] = None

    def execute(self, query: str, params=None):
        """Record executed query and optionally raise an error."""
        self._executed_queries.append(query)
        if self._error_on_execute:
            raise self._error_on_execute

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> Optional[tuple]:
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@dataclass
class MockConnection:
    """Mock database connection for unit tests."""

    cursor_factory: Optional[MockCursor] = None
    autocommit: bool = False
    _closed: bool = False
    _committed: bool = False
    _rolled_back: bool = False

    def __post_init__(self):
        self.cursor_factory = self.cursor_factory or MockCursor()

    def cursor(self):
        return self.cursor_factory

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def mock_cursor():
    """Create a mock cursor for testing."""
    return MockCursor()


@pytest.fixture
def mock_connection(mock_cursor):
    """Create a mock database connection."""
    return MockConnection(cursor_factory=mock_cursor)


@pytest.fixture
def mock_psycopg2_connect(mock_connection):
    """Patch psycopg2.connect to return mock connection."""
    with patch("psycopg2.connect", return_value=mock_connection) as mock_connect:
        yield mock_connect


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def integration_db_url() -> str:
    """The test database URL; skips the test when TEST_DATABASE_URL is unset."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set - skipping integration test")
    return url


# =============================================================================
# Test Data Factories
# =============================================================================


class StatementFactory:
    """Factory for creating ParsedStatement test objects."""

    @staticmethod
    def create_table(
        name: str = "test_table",
        schema: str = "public",
        source_file: str = "test.sql",
        line_number: int = 1,
    ):
        """Create a CREATE TABLE ParsedStatement."""
        from pgdeclare.statement_parser import ParsedStatement

        return ParsedStatement(
            raw_sql=f'CREATE TABLE "{schema}"."{name}" (id uuid PRIMARY KEY);',
            statement_type="CREATE TABLE",
            object_type="TABLE",
            object_name=f"{schema}.{name}",
            source_file=source_file,
            line_number=line_number,
            is_idempotent=False,
        )

    @staticmethod
    def create_function(
        name: str = "test_func",
        schema: str = "public",
        args: str = "",
        source_file: str = "test.sql",
        line_number: int = 1,
    ):
        """Create a CREATE FUNCTION ParsedStatement."""
        from pgdeclare.statement_parser import ParsedStatement

        signature = f"{schema}.{name}({args})"
        return ParsedStatement(
            raw_sql=f"CREATE OR REPLACE FUNCTION {schema}.{name}({args}) RETURNS void AS $$ BEGIN END; $$ LANGUAGE plpgsql;",
            statement_type="CREATE FUNCTION",
            object_type="FUNCTION",
            object_name=f"{schema}.{name}",
            signature=signature,
            source_file=source_file,
            line_number=line_number,
            is_idempotent=True,
        )

    @staticmethod
    def create_index(
        name: str = "test_idx",
        table: str = "test_table",
        source_file: str = "test.sql",
        line_number: int = 1,
    ):
        """Create a CREATE INDEX ParsedStatement."""
        from pgdeclare.statement_parser import ParsedStatement

        return ParsedStatement(
            raw_sql=f"CREATE INDEX {name} ON public.{table}(id);",
            statement_type="CREATE INDEX",
            object_type="INDEX",
            object_name=name,
            source_file=source_file,
            line_number=line_number,
            is_idempotent=False,
        )

    @staticmethod
    def create_role(
        name: str = "test_role",
        source_file: str = "test.sql",
        line_number: int = 1,
    ):
        """Create a CREATE ROLE ParsedStatement."""
        from pgdeclare.statement_parser import ParsedStatement

        return ParsedStatement(
            raw_sql=f'CREATE ROLE "{name}";',
            statement_type="CREATE ROLE",
            object_type="ROLE",
            object_name=name,
            source_file=source_file,
            line_number=line_number,
            is_idempotent=False,
        )

    @staticmethod
    def create_extension(
        name: str = "uuid-ossp",
        source_file: str = "test.sql",
        line_number: int = 1,
    ):
        """Create a CREATE EXTENSION ParsedStatement."""
        from pgdeclare.statement_parser import ParsedStatement

        return ParsedStatement(
            raw_sql=f'CREATE EXTENSION IF NOT EXISTS "{name}";',
            statement_type="CREATE EXTENSION",
            object_type="EXTENSION",
            object_name=name,
            source_file=source_file,
            line_number=line_number,
            is_idempotent=True,
        )


@pytest.fixture
def statement_factory():
    """Provide the StatementFactory for creating test statements."""
    return StatementFactory


# =============================================================================
# PostgreSQL Error Simulation
# =============================================================================


class PgError(Exception):
    """Mock PostgreSQL error with pgcode attribute."""

    def __init__(self, message: str, pgcode: str):
        super().__init__(message)
        self.pgcode = pgcode


@pytest.fixture
def pg_error_factory():
    """Factory for creating mock PostgreSQL errors."""

    def create_error(code: str, message: str = "Error") -> PgError:
        return PgError(message, code)

    return create_error


# Error code constants for convenience
PGCODE_UNDEFINED_TABLE = "42P01"
PGCODE_UNDEFINED_FUNCTION = "42883"
PGCODE_UNDEFINED_OBJECT = "42704"
PGCODE_UNDEFINED_COLUMN = "42703"
PGCODE_DUPLICATE_TABLE = "42P07"
PGCODE_DUPLICATE_OBJECT = "42710"
PGCODE_DUPLICATE_FUNCTION = "42723"
PGCODE_SYNTAX_ERROR = "42601"
PGCODE_PERMISSION_DENIED = "42501"
PGCODE_INVALID_FK = "42830"
PGCODE_DATATYPE_MISMATCH = "42804"


# =============================================================================
# Migration Ledger Fixtures
# =============================================================================

LEDGER = "pgdeclare.applied_migrations"


class MigrationsTableHelper:
    """Manipulate the inline-migration ledger directly, to stage odd states.

    - A migration marked as applied whose objects still exist
    - A migration not marked whose objects were created or removed
    - Partial migration states
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._ensure_table()

    def _execute(self, sql: str, params: Any = None) -> list[tuple]:
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
        return rows

    def _ensure_table(self) -> None:
        from pgdeclare import ensure_ledger

        ensure_ledger(self.db_url)

    def mark_as_applied(self, migration_id: str, sql_hash: str = "test_hash") -> None:
        """Mark a migration as applied without actually running it."""
        self._execute(
            f"""
            INSERT INTO {LEDGER} (id, sql_hash, sql_executed, source_file)
            VALUES (%s, %s, 'test sql', 'test.sql')
            ON CONFLICT (id) DO UPDATE SET sql_hash = EXCLUDED.sql_hash
            """,
            (migration_id, sql_hash),
        )

    def remove_applied_mark(self, migration_id: str) -> None:
        """Remove the applied mark for a migration."""
        self._execute(f"DELETE FROM {LEDGER} WHERE id = %s", (migration_id,))

    def is_applied(self, migration_id: str) -> bool:
        """Check if a migration is marked as applied."""
        return bool(
            self._execute(f"SELECT 1 FROM {LEDGER} WHERE id = %s", (migration_id,))
        )

    def get_all_applied(self) -> list[str]:
        """Get all applied migration IDs."""
        return [row[0] for row in self._execute(f"SELECT id FROM {LEDGER}")]

    def cleanup(self, prefix: str = "test_") -> None:
        """Remove all test migrations matching a prefix."""
        self._execute(f"DELETE FROM {LEDGER} WHERE id LIKE %s", (f"{prefix}%",))


@pytest.fixture
def migrations_table_helper(
    integration_db_url,
) -> Generator[MigrationsTableHelper, None, None]:
    """A MigrationsTableHelper; test migrations are removed afterwards."""
    helper = MigrationsTableHelper(integration_db_url)
    yield helper
    helper.cleanup("test_")


# =============================================================================
# Server version gates
# =============================================================================


def server_version_num(url: str) -> int:
    """The server's version as an integer (170004 for 17.4)."""
    conn = psycopg2.connect(url)
    try:
        return conn.server_version
    finally:
        conn.close()


def require_server_version(url: str, minimum: int, feature: str) -> None:
    """Skip the calling test on a server older than `minimum` (e.g. 150000)."""
    if server_version_num(url) < minimum:
        pytest.skip(f"{feature} needs PostgreSQL {minimum // 10000}+")
