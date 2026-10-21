"""Consolidation de plusieurs bases SQLite AAD en une nouvelle base autonome.

Ouvre chaque source en snapshot lecture seule (sans migration), aligne
l'union des colonnes dynamiques de `samples`, rejette les conflits de types,
et déduplique `boots`, `events` (par `dedup`) et `labels`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib
import sqlite3

from .database import read_snapshot
from .main import SCHEMA
from .storage import COLUMN_TYPES, Storage, quote_identifier

_BATCH_SIZE = 2000
_META_COLUMNS = frozenset({"sample_id", "ts", "boot_id"})


def merge_databases(
    sources: Sequence[str | pathlib.Path],
    destination: str | pathlib.Path,
    *,
    schema: pathlib.Path = SCHEMA,
) -> dict:
    """Fusionne >= 2 bases AAD existantes dans un nouveau fichier `destination`."""
    if len(sources) < 2:
        raise ValueError("au moins deux bases sources sont requises pour une fusion")
    src_paths = [pathlib.Path(s) for s in sources]
    for path in src_paths:
        if not path.is_file():
            raise FileNotFoundError(f"base source introuvable: {path}")
    dest = pathlib.Path(destination)
    if dest.exists():
        raise FileExistsError(f"la destination existe déjà: {dest}")

    # Passe 1 : découvrir et valider l'union des schémas sur toutes les sources.
    col_types: dict[str, str] = {}
    for path in src_paths:
        with read_snapshot(path) as conn:
            fields = conn.execute("PRAGMA table_info(samples)").fetchall()
            names = {f[1] for f in fields}
            if not {"ts", "boot_id"} <= names:
                raise ValueError(f"{path}: table samples incompatible")
            for _, name, raw_type, *_ in fields:
                if name in _META_COLUMNS:
                    continue
                ctype = (raw_type or "REAL").upper()
                if ctype not in COLUMN_TYPES:
                    ctype = "REAL"
                prev = col_types.get(name)
                if prev is not None and prev != ctype:
                    raise ValueError(
                        f"conflit de type pour la colonne {name!r}: {prev} vs {ctype}"
                    )
                col_types[name] = ctype

    # Créer la destination atomiquement (nettoyée si échec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.open("xb").close()
    total_boots = 0
    total_samples = 0
    events_inserted = 0
    events_skipped = 0
    labels_inserted = 0
    labels_skipped = 0
    try:
        with Storage(dest, schema) as target:
            ordered_cols = sorted(col_types)
            target.ensure_columns(ordered_cols, col_types)

            for path in src_paths:
                with read_snapshot(path) as conn:
                    # 1. boots
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='boots'").fetchone():
                        for boot_id, ts_first, kernel, notes in conn.execute(
                            "SELECT boot_id, ts_first, kernel, notes FROM boots"
                        ):
                            cur = target.conn.execute(
                                "INSERT OR IGNORE INTO boots (boot_id, ts_first, kernel, notes)"
                                " VALUES (?, ?, ?, ?)",
                                (boot_id, ts_first, kernel, notes),
                            )
                            total_boots += cur.rowcount

                    # 2. samples (par lots pour rester borné en RAM)
                    src_cols = [
                        f[1] for f in conn.execute("PRAGMA table_info(samples)").fetchall()
                        if f[1] != "sample_id"
                    ]
                    proj = ", ".join(quote_identifier(c) for c in src_cols)
                    order_col = "sample_id" if any(
                        f[1] == "sample_id" for f in conn.execute("PRAGMA table_info(samples)").fetchall()
                    ) else "ts"
                    cursor = conn.execute(
                        f"SELECT {proj} FROM samples ORDER BY {quote_identifier(order_col)}"
                    )
                    batch: list[dict] = []
                    for row in cursor:
                        batch.append(dict(zip(src_cols, row)))
                        if len(batch) >= _BATCH_SIZE:
                            total_samples += target.insert_samples(batch)
                            batch.clear()
                    if batch:
                        total_samples += target.insert_samples(batch)

                    # 3. events (déduplication via dedup)
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone():
                        ev_cols = {f[1] for f in conn.execute("PRAGMA table_info(events)").fetchall()}
                        dedup_expr = "dedup" if "dedup" in ev_cols else "NULL"
                        for ts, source, kind, severity, message, dedup in conn.execute(
                            f"SELECT ts, source, kind, severity, message, {dedup_expr} FROM events ORDER BY ts"
                        ):
                            if target.add_event(ts, source, kind, severity or "", message or "", dedup):
                                events_inserted += 1
                            else:
                                events_skipped += 1

                    # 4. labels (idempotent sur (ts_start, ts_end, label))
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='labels'").fetchone():
                        entries = [
                            {"ts_start": r[0], "ts_end": r[1], "label": r[2], "notes": r[3] or ""}
                            for r in conn.execute(
                                "SELECT ts_start, ts_end, label, notes FROM labels ORDER BY ts_start, id"
                            )
                        ]
                        ins, skp = target.import_labels(entries)
                        labels_inserted += ins
                        labels_skipped += skp
    except BaseException:
        dest.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            dest.with_name(dest.name + suffix).unlink(missing_ok=True)
        raise

    return {
        "destination": str(dest),
        "sources": [str(p) for p in src_paths],
        "columns": sorted(col_types),
        "boots": total_boots,
        "samples": total_samples,
        "events_inserted": events_inserted,
        "events_skipped": events_skipped,
        "labels_inserted": labels_inserted,
        "labels_skipped": labels_skipped,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fusionner plusieurs bases SQLite AAD dans une nouvelle base"
    )
    parser.add_argument("sources", nargs="+", type=pathlib.Path, help="bases SQLite sources (>= 2)")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="nouveau fichier SQLite de destination")
    parser.add_argument("--schema", type=pathlib.Path, default=SCHEMA)
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)
    try:
        result = merge_databases(args.sources, args.output, schema=args.schema)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"fusion impossible: {exc}")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"fusion créée : {result['destination']} "
        f"({result['samples']} mesures, {result['boots']} boots, "
        f"{result['events_inserted']} événements, {result['labels_inserted']} labels)"
    )


if __name__ == "__main__":
    main()
