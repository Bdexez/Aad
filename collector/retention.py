"""Rétention et purge contrôlée des anciennes mesures (dry-run par défaut).

Permet de maîtriser la taille disque sur plusieurs mois de collecte tout en
préservant par défaut les annotations (`labels`). En mode dry-run (défaut), la
base est ouverte en lecture seule sans aucune modification.
"""

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


def prune_database(
    db: str | pathlib.Path,
    *,
    before_ts: int,
    dry_run: bool = True,
    prune_labels: bool = False,
    vacuum: bool = False,
) -> dict:
    """Purge les mesures, événements (et boots orphelins) antérieurs à `before_ts`."""
    path = pathlib.Path(db)
    if not path.is_file():
        raise FileNotFoundError(f"base introuvable: {path}")

    if dry_run:
        with read_snapshot(path) as conn:
            samples_n = conn.execute("SELECT COUNT(*) FROM samples WHERE ts < ?", (before_ts,)).fetchone()[0]
            events_n = conn.execute("SELECT COUNT(*) FROM events WHERE ts < ?", (before_ts,)).fetchone()[0]
            boots_n = conn.execute(
                "SELECT COUNT(*) FROM boots WHERE boot_id NOT IN"
                " (SELECT DISTINCT boot_id FROM samples WHERE ts >= ?)",
                (before_ts,),
            ).fetchone()[0]
            labels_n = (
                conn.execute(
                    "SELECT COUNT(*) FROM labels WHERE ts_end IS NOT NULL AND ts_end <= ?",
                    (before_ts,),
                ).fetchone()[0]
                if prune_labels
                else 0
            )
        return {
            "path": str(path),
            "before_ts": before_ts,
            "dry_run": True,
            "vacuumed": False,
            "samples": samples_n,
            "events": events_n,
            "boots": boots_n,
            "labels": labels_n,
        }

    with closing(sqlite3.connect(path, isolation_level=None, timeout=30.0)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            samples_n = conn.execute("DELETE FROM samples WHERE ts < ?", (before_ts,)).rowcount
            events_n = conn.execute("DELETE FROM events WHERE ts < ?", (before_ts,)).rowcount
            boots_n = conn.execute(
                "DELETE FROM boots WHERE boot_id NOT IN (SELECT DISTINCT boot_id FROM samples)"
            ).rowcount
            labels_n = (
                conn.execute(
                    "DELETE FROM labels WHERE ts_end IS NOT NULL AND ts_end <= ?",
                    (before_ts,),
                ).rowcount
                if prune_labels
                else 0
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        if vacuum:
            conn.execute("VACUUM")
    return {
        "path": str(path),
        "before_ts": before_ts,
        "dry_run": False,
        "vacuumed": vacuum,
        "samples": samples_n,
        "events": events_n,
        "boots": boots_n,
        "labels": labels_n,
    }


def prune_boot(
    db: str | pathlib.Path,
    *,
    boot_id: str,
    dry_run: bool = True,
    vacuum: bool = False,
) -> dict:
    """Supprime toutes les mesures et la ligne `boots` d'un `boot_id` précis."""
    boot_id = boot_id.strip()
    if not boot_id:
        raise ValueError("--boot ne peut pas être vide")
    path = pathlib.Path(db)
    if not path.is_file():
        raise FileNotFoundError(f"base introuvable: {path}")

    if dry_run:
        with read_snapshot(path) as conn:
            samples_n = conn.execute("SELECT COUNT(*) FROM samples WHERE boot_id = ?", (boot_id,)).fetchone()[0]
            boots_n = conn.execute("SELECT COUNT(*) FROM boots WHERE boot_id = ?", (boot_id,)).fetchone()[0]
        return {
            "path": str(path),
            "boot_id": boot_id,
            "dry_run": True,
            "vacuumed": False,
            "samples": samples_n,
            "events": 0,
            "boots": boots_n,
            "labels": 0,
        }

    with closing(sqlite3.connect(path, isolation_level=None, timeout=30.0)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            samples_n = conn.execute("DELETE FROM samples WHERE boot_id = ?", (boot_id,)).rowcount
            boots_n = conn.execute("DELETE FROM boots WHERE boot_id = ?", (boot_id,)).rowcount
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        if vacuum:
            conn.execute("VACUUM")
    return {
        "path": str(path),
        "boot_id": boot_id,
        "dry_run": False,
        "vacuumed": vacuum,
        "samples": samples_n,
        "events": 0,
        "boots": boots_n,
        "labels": 0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Purge contrôlée d'une base AAD (simulation dry-run par défaut)"
    )
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    cutoff = parser.add_mutually_exclusive_group(required=True)
    cutoff.add_argument("--before-ts", type=int, help="supprimer les entrées strictement antérieures à cet epoch ms")
    cutoff.add_argument("--keep-days", type=float, help="conserver les N derniers jours")
    cutoff.add_argument("--boot", dest="boot_id", help="supprimer toutes les mesures d'un boot_id précis")
    parser.add_argument("--apply", action="store_true", help="appliquer réellement la purge (sinon dry-run)")
    parser.add_argument("--prune-labels", action="store_true",
                        help="supprimer aussi les labels fermés terminés avant la borne")
    parser.add_argument("--vacuum", action="store_true", help="compacter la base après purge (avec --apply)")
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)

    if args.vacuum and not args.apply:
        parser.error("--vacuum nécessite --apply")
    if args.boot_id is not None and args.prune_labels:
        parser.error("--prune-labels ne s'applique pas à --boot")

    try:
        if args.boot_id is not None:
            result = prune_boot(
                args.db,
                boot_id=args.boot_id,
                dry_run=not args.apply,
                vacuum=args.vacuum,
            )
        else:
            if args.keep_days is not None:
                if not math.isfinite(args.keep_days) or args.keep_days <= 0:
                    parser.error("--keep-days doit être un nombre fini > 0")
                before_ts = int(time.time() * 1000) - int(args.keep_days * 86_400_000)
            else:
                before_ts = args.before_ts
            result = prune_database(
                args.db,
                before_ts=before_ts,
                dry_run=not args.apply,
                prune_labels=args.prune_labels,
                vacuum=args.vacuum,
            )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"purge impossible: {exc}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    mode = "DRY-RUN (aucune modification)" if result["dry_run"] else "APPLIQUÉ"
    scope = f"boot={result['boot_id']}" if "boot_id" in result else f"avant ts={result['before_ts']}"
    print(f"Purge [{mode}] {scope} :")
    print(
        f"  samples={result['samples']}, events={result['events']}, "
        f"boots={result['boots']}, labels={result['labels']}"
    )


if __name__ == "__main__":
    main()
