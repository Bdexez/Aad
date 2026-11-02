"""Synthèse globale de santé d'une campagne AAD en une seule commande.

Combine en lecture seule : intégrité & taille disque (`backup.verify_database`),
complétude (`report.dataset_report`), bornes physiques (`analyze.check_physical_bounds`)
et audit des annotations (`analyze.audit_labels`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib
import sqlite3

from .analyze import audit_labels, check_physical_bounds
from .backup import verify_database
from .database import resolve_last_minutes
from .report import dataset_report


def campaign_summary(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Agrège les indicateurs clés d'intégrité et de qualité de la base."""
    integrity = verify_database(db)
    rep = dataset_report(db, boot_id=boot_id, since=since, until=until)
    bounds = check_physical_bounds(db, boot_id=boot_id, since=since, until=until)
    labels = audit_labels(db, boot_id=boot_id, since=since, until=until)
    healthy = (
        integrity["ok"]
        and bounds["total_violations"] == 0
        and len(labels["empty_windows"]) == 0
    )
    return {
        "healthy": healthy,
        "integrity": integrity,
        "samples": rep["samples"],
        "boots": len(rep["boots"]),
        "metrics_count": len(rep["metrics"]),
        "events_total": sum(e["count"] for e in rep["events"]),
        "physical_bounds": {
            "checked_metrics": bounds["checked_metrics"],
            "total_violations": bounds["total_violations"],
        },
        "labels_audit": {
            "total_windows": labels["total_windows"],
            "open_windows": len(labels["open_windows"]),
            "empty_windows": len(labels["empty_windows"]),
            "sampleless_windows": len(labels["sampleless_windows"]),
            "cross_label_overlap_samples": labels["cross_label_overlap_samples"],
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthèse de santé globale d'une base AAD")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--boot", dest="boot_id", help="limiter à un boot_id")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--last-minutes", type=float, help="limiter aux N dernières minutes")
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)
    try:
        since, until = resolve_last_minutes(args.last_minutes, args.since, args.until)
        result = campaign_summary(args.db, boot_id=args.boot_id, since=since, until=until)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"synthèse impossible: {exc}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    verdict = "OK" if result["healthy"] else "ATTENTION"
    size_kb = result["integrity"]["size"]["total_bytes"] / 1024.0
    samples = result["samples"]
    cov = "indéfinie" if samples["labelled_pct"] is None else f"{samples['labelled_pct']:.1f} %"
    print(f"État global : {verdict} ({size_kb:.1f} KiB, {result['integrity']['journal_mode']})")
    print(f"Mesures     : {samples['count']} sur {result['boots']} boot(s) · {result['metrics_count']} métrique(s)")
    print(f"Annotations : {cov} · {result['labels_audit']['total_windows']} fenêtre(s) "
          f"({result['labels_audit']['open_windows']} ouv., {result['labels_audit']['cross_label_overlap_samples']} multi-labels)")
    print(f"Bornes      : {result['physical_bounds']['total_violations']} violation(s) sur "
          f"{result['physical_bounds']['checked_metrics']} sonde(s)")
    print(f"Événements  : {result['events_total']}")


if __name__ == "__main__":
    main()
