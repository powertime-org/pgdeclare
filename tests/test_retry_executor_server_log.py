"""A schema load keeps its expected failures out of the server log.

Retry-until-stable finds the load order by failing, and each failed attempt
would otherwise be an ERROR in the server log with its full statement text:
thousands of lines per load of a large schema, burying the errors that matter.
The load's own session raises its logging thresholds when the role may; no
other session logs any less, and a role that may not loads as before.
"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit

import psycopg2
import pytest

from dbutil import scalar
from pgdeclare.retry_executor import (
    QUIET_SESSION_SETTINGS,
    _quiet_expected_failures,
    execute_with_retry,
)
from pgdeclare.statement_parser import ParsedStatement

from conftest import MockConnection, MockCursor


def _stmt(sql: str, kind: str, name: str) -> ParsedStatement:
    return ParsedStatement(
        raw_sql=sql,
        statement_type=kind,
        object_type=kind.split()[-1],
        object_name=name,
        source_file="test.sql",
        line_number=1,
        is_idempotent=False,
    )


def _out_of_order_load(table: str) -> list[ParsedStatement]:
    """A load whose first statement fails on the first pass, recording the
    settings its session ran under once its dependency exists."""
    return [
        _stmt(
            f"CREATE TABLE {table} AS SELECT "
            "current_setting('log_min_messages') AS min_messages, "
            "current_setting('log_min_error_statement') AS min_error_statement "
            f"FROM {table}_dep",
            "CREATE TABLE",
            table,
        ),
        _stmt(
            f"CREATE TABLE {table}_dep AS SELECT 1 AS one",
            "CREATE TABLE",
            f"{table}_dep",
        ),
    ]


def _query(url: str, sql: str):
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()
    finally:
        conn.close()


def _defaults(url: str) -> tuple[str, str]:
    row = _query(
        url,
        "SELECT current_setting('log_min_messages'), "
        "current_setting('log_min_error_statement')",
    )
    assert row is not None
    return row[0], row[1]


def test_a_privileged_load_logs_no_expected_failure(integration_db_url):
    table = f"quiet_{uuid.uuid4().hex[:8]}"
    before = _defaults(integration_db_url)
    if _query(integration_db_url, "SELECT current_setting('is_superuser')") != ("on",):
        pytest.skip("TEST_DATABASE_URL's role is not a superuser")

    result = execute_with_retry(integration_db_url, _out_of_order_load(table))

    assert result.success, result.failed_statements
    # The statement that failed on the first pass ran in a session that logs
    # neither the ERROR nor its statement text...
    assert _query(
        integration_db_url, f"SELECT min_messages, min_error_statement FROM {table}"
    ) == tuple(value for _, value in QUIET_SESSION_SETTINGS)
    # ...and every other session, on this database too, logs as before.
    assert _defaults(integration_db_url) == before


def test_a_role_that_may_not_set_them_loads_as_before(integration_db_url):
    role = f"pgdeclare_quiet_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    dbname = urlsplit(integration_db_url).path.lstrip("/")
    admin = psycopg2.connect(integration_db_url)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
            cur.execute(f'GRANT CONNECT, TEMP ON DATABASE "{dbname}" TO {role}')
            cur.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {role}")
        parts = urlsplit(integration_db_url)
        netloc = f"{role}:{password}@{parts.hostname}" + (
            f":{parts.port}" if parts.port else ""
        )
        role_url = parts._replace(netloc=netloc).geturl()
        before = _defaults(role_url)
        table = f"loud_{uuid.uuid4().hex[:8]}"

        result = execute_with_retry(role_url, _out_of_order_load(table))

        assert result.success, result.failed_statements
        assert (
            _query(role_url, f"SELECT min_messages, min_error_statement FROM {table}")
            == before
        )
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP OWNED BY {role}")
            cur.execute(f"DROP ROLE IF EXISTS {role}")
        admin.close()


def test_granted_set_is_enough_from_postgres_15(integration_db_url):
    with psycopg2.connect(integration_db_url) as probe:
        version = probe.server_version
    probe.close()
    if version < 150000:
        pytest.skip("GRANT SET ON PARAMETER needs PostgreSQL 15+")
    role = f"pgdeclare_quiet_{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    admin = psycopg2.connect(integration_db_url)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
            for name, _ in QUIET_SESSION_SETTINGS:
                cur.execute(f"GRANT SET ON PARAMETER {name} TO {role}")
        parts = urlsplit(integration_db_url)
        netloc = f"{role}:{password}@{parts.hostname}" + (
            f":{parts.port}" if parts.port else ""
        )
        conn = psycopg2.connect(parts._replace(netloc=netloc).geturl())
        conn.autocommit = True
        try:
            assert _quiet_expected_failures(conn) is True
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('log_min_messages')")
                assert scalar(cur) == "fatal"
        finally:
            conn.close()
    finally:
        with admin.cursor() as cur:
            for name, _ in QUIET_SESSION_SETTINGS:
                cur.execute(f"REVOKE SET ON PARAMETER {name} FROM {role}")
            cur.execute(f"DROP ROLE IF EXISTS {role}")
        admin.close()


def test_a_connection_that_cannot_answer_is_left_alone():
    """Quietening is best effort: a session it cannot ask loads as before."""
    assert _quiet_expected_failures(MockConnection(MockCursor())) is False
