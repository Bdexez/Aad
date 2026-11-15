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

from .analyze import analyze_dataset, audit_labels, check_physical_bounds
from .backup import verify_database
from .database import resolve_boot_id, resolve_last_minutes
from .report import dataset_report


def campaign_summary(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Agrège les indicateurs clés d'intégrité, de qualité et de projection disque."""
    integrity = verify_database(db)
    rep = dataset_report(db, boot_id=boot_id, since=since, until=until)
    ana = analyze_dataset(db, boot_id=boot_id, since=since, until=until)
    bounds = check_physical_bounds(db, boot_id=boot_id, since=since, until=until)
    labels = audit_labels(db, boot_id=boot_id, since=since, until=until)

    medians = [b["median_dt_ms"] for b in ana["cadence"] if b["median_dt_ms"] and b["median_dt_ms"] > 0]
    median_dt_ms = (sum(medians) / len(medians)) if medians else None
    bps = integrity["size"]["bytes_per_sample"]
    mib_per_day = None
    if median_dt_ms is not None and bps is not None:
        samples_per_day = 86_400_000.0 / median_dt_ms
        mib_per_day = (samples_per_day * bps) / (1024.0 * 1024.0)

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
        "storage_projection": {
            "median_dt_ms": median_dt_ms,
            "bytes_per_sample": bps,
            "estimated_mib_per_day": mib_per_day,
            "estimated_mib_14d": None if mib_per_day is None else mib_per_day * 14.0,
            "estimated_mib_30d": None if mib_per_day is None else mib_per_day * 30.0,
        },
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


def format_summary_markdown(result: dict) -> str:
    verdict = "OK" if result["healthy"] else "ATTENTION"
    size_kb = result["integrity"]["size"]["total_bytes"] / 1024.0
    samples = result["samples"]
    cov = "indéfinie" if samples["labelled_pct"] is None else f"{samples['labelled_pct']:.1f} %"
    proj = result["storage_projection"]
    proj_str = (
        f"~{proj['estimated_mib_per_day']:.1f} MiB/j (~{proj['estimated_mib_14d']:.0f} MiB / 14 j)"
        if proj["estimated_mib_per_day"] is not None
        else "n/a"
    )
    return "\n".join([
        f"# Synthèse AAD — {verdict}",
        "",
        "| Indicateur | Valeur |",
        "| :--- | :--- |",
        f"| Intégrité | `{result['integrity']['quick_check']}` ({size_kb:.1f} KiB, `{result['integrity']['journal_mode']}`) |",
        f"| Mesures | {samples['count']} sur {result['boots']} boot(s) ({result['metrics_count']} métriques) |",
        f"| Annotations | {cov} · {result['labels_audit']['total_windows']} fenêtre(s) ({result['labels_audit']['open_windows']} ouv.) |",
        f"| Bornes physiques | {result['physical_bounds']['total_violations']} violation(s) sur {result['physical_bounds']['checked_metrics']} sonde(s) |",
        f"| Projection disque | {proj_str} |",
        f"| Événements | {result['events_total']} |",
    ])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthèse de santé globale d'une base AAD")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--boot", dest="boot_id", help="limiter à un boot_id")
    parser.add_argument("--latest-boot", action="store_true", help="limiter au boot le plus récent")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--last-minutes", type=float, help="limiter aux N dernières minutes")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    fmt.add_argument("--markdown", action="store_true", help="sortie Markdown GFM")
    args = parser.parse_args(argv)
    try:
        boot_id = resolve_boot_id(args.db, boot_id=args.boot_id, latest_boot=args.latest_boot)
        since, until = resolve_last_minutes(args.last_minutes, args.since, args.until)
        result = campaign_summary(args.db, boot_id=boot_id, since=since, until=until)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"synthèse impossible: {exc}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.markdown:
        print(format_summary_markdown(result))
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
    proj = result["storage_projection"]
    if proj["estimated_mib_per_day"] is not None:
        print(
            f"Projection  : ~{proj['estimated_mib_per_day']:.1f} MiB/j "
            f"(~{proj['estimated_mib_14d']:.0f} MiB / 14 j)"
        )
    print(f"Événements  : {result['events_total']}")


if __name__ == "__main__":
    main()
