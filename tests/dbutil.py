"""Small database-cursor helpers shared by the DB-backed test modules."""

from typing import Any


def scalar(cur) -> Any:
    """First column of the single row a query is expected to return.

    `cur.fetchone()[0]` is typed `tuple | None`, and when a query unexpectedly
    matches nothing it fails as an unhelpful "NoneType is not subscriptable"
    several frames from the assertion that cared.
    """
    row = cur.fetchone()
    assert row is not None, "expected exactly one row, got none"
    return row[0]
