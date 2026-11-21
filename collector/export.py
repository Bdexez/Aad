"""Export CSV/JSONL des mesures, annotations et événements, sans modifier la base."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import csv
import json
import pathlib
import sqlite3
import sys
from typing import TextIO

from .database import read_snapshot, sample_filter_sql, validate_time_range
from .storage import quote_identifier


def _iter_samples(
    conn: sqlite3.Connection,
    *,
    boot_id: str | None,
    since: int | None,
    until: int | None,
    columns: Sequence[str] | None,
    reserved_label_column: str,
) -> tuple[list[str], Iterator[tuple[tuple, list[dict]]]]:
    fields = conn.execute("PRAGMA table_info(samples)").fetchall()
    available = [field[1] for field in fields]
    selected = available if columns is None else list(columns)
    if not selected:
        raise ValueError("aucune colonne de mesures à exporter")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"colonnes inconnues: {', '.join(sorted(unknown))}")
    if len(set(selected)) != len(selected) or reserved_label_column in selected:
        raise ValueError(f"colonnes dupliquées ou nom réservé: {reserved_label_column}")
    primary = [field[1] for field in fields if field[5]]
    if len(primary) != 1:
        raise ValueError("la table samples doit avoir une clé primaire unique")
    key = f"s.{quote_identifier(primary[0])}"
    projection = ", ".join(f"s.{quote_identifier(field)}" for field in selected)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    cursor = conn.execute(
        f"SELECT {key}, {projection}, l.id, l.ts_start, l.ts_end, l.label, l.notes"
        " FROM samples AS s LEFT JOIN labels AS l"
        " ON l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end)"
        + where + f" ORDER BY s.ts, {key}, l.id",
        params,
    )

    def stream() -> Iterator[tuple[tuple, list[dict]]]:
        previous = None
        values: tuple = ()
        labels: list[dict] = []
        width = 1 + len(selected)
        for row in cursor:
            if row[0] != previous:
                if previous is not None:
                    yield values, labels
                previous = row[0]
                values = row[1:width]
                labels = []
            label_id, start, end, label, notes = row[width:]
            if label_id is not None:
                labels.append({
                    "id": label_id, "label": label, "notes": notes,
                    "ts_start": start, "ts_end": end,
                })
        if previous is not None:
            yield values, labels

    return selected, stream()


def export_csv(
    db: str | pathlib.Path,
    target: TextIO,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    columns: Sequence[str] | None = None,
) -> int:
    """Écrit une ligne par mesure, en flux, depuis un snapshot SQLite cohérent.

    Les fenêtres sont [début, fin[. labels_json conserve toutes les annotations
    qui se chevauchent, leurs notes et leurs bornes ; [] n'invente aucun label.
    Les anciennes bases sont lisibles sans déclencher leur migration.
    """
    validate_time_range(since, until)
    with read_snapshot(db) as conn:
        selected, rows = _iter_samples(
            conn, boot_id=boot_id, since=since, until=until,
            columns=columns, reserved_label_column="labels_json",
        )
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow([*selected, "labels_json"])
        count = 0
        for values, labels in rows:
            writer.writerow([*values, json.dumps(labels, ensure_ascii=False, separators=(",", ":"))])
            count += 1
        return count


def export_jsonl(
    db: str | pathlib.Path,
    target: TextIO,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    columns: Sequence[str] | None = None,
) -> int:
    """Écrit un objet JSON par mesure (JSON Lines), préservant les types natifs."""
    validate_time_range(since, until)
    with read_snapshot(db) as conn:
        selected, rows = _iter_samples(
            conn, boot_id=boot_id, since=since, until=until,
            columns=columns, reserved_label_column="labels",
        )
        count = 0
        for values, labels in rows:
            record = dict(zip(selected, values))
            record["labels"] = labels
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        return count


def export_events(
    db: str | pathlib.Path,
    target: TextIO,
    *,
    fmt: str = "csv",
    since: int | None = None,
    until: int | None = None,
    source: str | None = None,
    kind: str | None = None,
    severity: str | None = None,
) -> int:
    """Exporte la table `events` en CSV ou JSONL depuis un snapshot lecture seule."""
    if fmt not in ("csv", "jsonl"):
        raise ValueError(f"format non supporté: {fmt!r}")
    validate_time_range(since, until)
    with read_snapshot(db) as conn:
        fields = [field[1] for field in conn.execute("PRAGMA table_info(events)").fetchall()]
        if not {"ts", "source", "kind"} <= set(fields):
            raise ValueError("la base ne contient pas de table events compatible")
        filters = []
        params: list[str | int] = []
        for value, condition in (
            (since, "ts >= ?"),
            (until, "ts < ?"),
            (source, "source = ?"),
            (kind, "kind = ?"),
            (severity, "severity = ?"),
        ):
            if value is not None:
                filters.append(condition)
                params.append(value)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        order = "ORDER BY ts, id" if "id" in fields else "ORDER BY ts"
        projection = ", ".join(quote_identifier(field) for field in fields)
        cursor = conn.execute(f"SELECT {projection} FROM events{where} {order}", params)
        count = 0
        if fmt == "csv":
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(fields)
            for row in cursor:
                writer.writerow(row)
                count += 1
        else:
            for row in cursor:
                target.write(
                    json.dumps(dict(zip(fields, row)), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                count += 1
        return count


@contextmanager
def _output(destination: str):
    if destination == "-":
        yield sys.stdout
        return
    path = pathlib.Path(destination)
    stream = path.open("x", encoding="utf-8", newline="")
    try:
        with stream:
            yield stream
    except BaseException:
        # Seul le fichier créé par cet appel est supprimé en cas d'échec.
        # Un export partiel ne doit pas passer pour un dataset complet.
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exporter les mesures, labels ou événements")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante, ouverte en lecture seule")
    parser.add_argument("--output", default="-", help="nouveau fichier de sortie, ou - pour stdout (défaut)")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv", help="format de sortie (défaut : csv)")
    parser.add_argument("--events", action="store_true", help="exporter la table events au lieu de samples")
    parser.add_argument("--boot", dest="boot_id", help="limiter les mesures à un boot_id")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--columns", nargs="+", help="colonnes de mesures à exporter, dans l'ordre choisi")
    parser.add_argument("--source", help="filtrer les événements par source (avec --events)")
    parser.add_argument("--kind", help="filtrer les événements par genre (avec --events)")
    parser.add_argument("--severity", help="filtrer les événements par gravité (avec --events)")
    args = parser.parse_args(argv)
    if args.events and (args.boot_id is not None or args.columns is not None):
        parser.error("--boot et --columns ne s'appliquent pas à --events")
    if not args.events and any(v is not None for v in (args.source, args.kind, args.severity)):
        parser.error("--source, --kind et --severity nécessitent --events")
    try:
        with _output(args.output) as target:
            if args.events:
                count = export_events(
                    args.db, target, fmt=args.format, since=args.since, until=args.until,
                    source=args.source, kind=args.kind, severity=args.severity,
                )
            elif args.format == "jsonl":
                count = export_jsonl(
                    args.db, target, boot_id=args.boot_id, since=args.since,
                    until=args.until, columns=args.columns,
                )
            else:
                count = export_csv(
                    args.db, target, boot_id=args.boot_id, since=args.since,
                    until=args.until, columns=args.columns,
                )
            target.flush()
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"export impossible: {exc}")
    noun = "événement(s)" if args.events else "mesure(s)"
    print(f"{count} {noun} exporté(s)" if args.events else f"{count} {noun} exportée(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
