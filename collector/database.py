"""Accès aux bases existantes et aides SQL pour les outils de lecture, sans migration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
import pathlib
import sqlite3


def validate_time_range(since: int | None, until: int | None) -> None:
    """Valide une fenêtre semi-ouverte [since, until) en epoch ms."""
    if since is not None and until is not None and until <= since:
        raise ValueError("--until doit être supérieur à --since")


def sample_filter_sql(
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    alias: str = "",
) -> tuple[str, list[str | int]]:
    """Construit la clause WHERE et ses paramètres pour filtrer `samples`."""
    prefix = f"{alias}." if alias else ""
    filters: list[str] = []
    params: list[str | int] = []
    if boot_id is not None:
        filters.append(f"{prefix}boot_id = ?")
        params.append(boot_id)
    if since is not None:
        filters.append(f"{prefix}ts >= ?")
        params.append(since)
    if until is not None:
        filters.append(f"{prefix}ts < ?")
        params.append(until)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    return where, params


@contextmanager
def read_snapshot(path: str | pathlib.Path, *, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
    uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, isolation_level=None, timeout=timeout)) as conn:
        conn.execute("BEGIN")
        yield conn
