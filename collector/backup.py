"""Copie cohérente d'une base SQLite active, y compris son WAL validé."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import closing
import json
import math
import pathlib
import sqlite3
import time

from .database import read_snapshot

_EXPECTED_TABLES = ("samples", "boots", "events", "labels")
_EXPECTED_INDEXES = (
    "idx_samples_ts",
    "idx_samples_boot",
    "idx_events_ts",
    "idx_events_kind",
    "idx_events_dedup",
    "idx_labels_ts",
)


def verify_database(db: str | pathlib.Path) -> dict:
    """Vérifie l'intégrité et le schéma d'une base SQLite en lecture seule."""
    path = pathlib.Path(db)
    with read_snapshot(path) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise sqlite3.DatabaseError(f"quick_check a échoué: {quick_check}")
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = [table for table in _EXPECTED_TABLES if table not in existing]
        if "samples" in missing:
            raise ValueError("la base ne contient pas de table samples")
        existing_indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        missing_indexes = [idx for idx in _EXPECTED_INDEXES if idx not in existing_indexes]
        counts = {
            table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in _EXPECTED_TABLES
            if table in existing
        }
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    db_bytes = path.stat().st_size
    wal_path = path.with_name(path.name + "-wal")
    shm_path = path.with_name(path.name + "-shm")
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
    shm_bytes = shm_path.stat().st_size if shm_path.exists() else 0
    total_bytes = db_bytes + wal_bytes + shm_bytes
    n_samples = counts.get("samples", 0)
    return {
        "path": str(path),
        "quick_check": quick_check,
        "journal_mode": journal_mode,
        "missing_tables": missing,
        "missing_indexes": missing_indexes,
        "counts": counts,
        "size": {
            "db_bytes": db_bytes,
            "wal_bytes": wal_bytes,
            "shm_bytes": shm_bytes,
            "total_bytes": total_bytes,
            "bytes_per_sample": total_bytes / n_samples if n_samples else None,
        },
        "ok": quick_check == "ok" and not missing,
    }


def backup_database(
    db: str | pathlib.Path,
    destination: str | pathlib.Path,
    *,
    timeout: float = 30.0,
) -> pathlib.Path:
    """Crée une nouvelle sauvegarde autonome sans arrêter le collecteur.

    L'API backup de SQLite inclut les transactions validées dans le WAL.
    La destination doit être absente et une copie abandonnée est supprimée.
    Les données encore dans les buffers RAM du collecteur ne sont pas incluses.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("le délai de sauvegarde doit être un nombre fini > 0")
    destination = pathlib.Path(destination)
    deadline = time.monotonic() + timeout

    def check_deadline(status, remaining, total):
        if time.monotonic() >= deadline:
            raise TimeoutError("délai de sauvegarde dépassé")

    # L'attente SQLite par défaut de 5 s masquerait les petits délais choisis
    # par l'utilisateur. L'API backup réessaie avec son propre callback borné.
    with read_snapshot(db, timeout=0.0) as source:
        # Création exclusive avant que SQLite ouvre le fichier : protège aussi
        # la source si elle a été choisie par erreur comme destination.
        destination.open("xb").close()
        try:
            with closing(sqlite3.connect(destination, isolation_level=None)) as target:
                source.backup(target, pages=256, progress=check_deadline, sleep=0.05)
                if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise sqlite3.DatabaseError("la copie SQLite ne passe pas quick_check")
                target.execute("PRAGMA journal_mode=DELETE")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sauvegarder ou vérifier une base AAD")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--output", type=pathlib.Path, help="nouveau fichier de sauvegarde")
    parser.add_argument("--verify", action="store_true", help="vérifier l'intégrité d'une base en lecture seule")
    parser.add_argument("--json", action="store_true", help="sortie JSON (avec --verify)")
    parser.add_argument("--timeout", type=float, default=30.0, help="délai de copie en secondes (défaut : 30)")
    args = parser.parse_args(argv)
    if args.verify and args.output is not None:
        parser.error("--verify et --output sont mutuellement exclusifs")
    if args.json and not args.verify:
        parser.error("--json nécessite --verify")
    if not args.verify and args.output is None:
        parser.error("--output est requis (ou utilisez --verify)")
    try:
        if args.verify:
            result = verify_database(args.db)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                counts = ", ".join(f"{k}={v}" for k, v in result["counts"].items())
                size_kb = result["size"]["total_bytes"] / 1024.0
                print(f"base valide ({result['journal_mode']}, {size_kb:.1f} KiB) : {counts}")
                if result["missing_indexes"]:
                    print(f"index manquants : {', '.join(result['missing_indexes'])}")
            return
        destination = backup_database(args.db, args.output, timeout=args.timeout)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"opération impossible: {exc}")
    print(f"sauvegarde créée : {destination}")


if __name__ == "__main__":
    main()
