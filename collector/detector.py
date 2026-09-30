"""Détecteur statistique d'anomalies de référence (zéro dépendance, explicable).

Apprend une distribution de référence (médiane/MAD, repli sur moyenne/écart-type)
sur une fenêtre ou un label de base (ex. `normal`), puis évalue une période cible
en streaming depuis un snapshot SQLite lecture seule. Regroupe les dépassements
consécutifs en épisodes et identifie les capteurs contributeurs principaux.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import pathlib
import sqlite3

from .analyze import _MAD_SCALE, _numeric_metrics, _quantile
from .database import read_snapshot, sample_filter_sql, validate_time_range
from .storage import quote_identifier


@dataclass(frozen=True)
class FeatureBaseline:
    column: str
    center: float
    scale: float
    method: str  # "mad" | "std"
    present: int


def _label_exists_clause(label: str) -> tuple[str, list[str]]:
    return (
        "EXISTS (SELECT 1 FROM labels AS l WHERE l.label = ?"
        " AND l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end))",
        [label],
    )


def _scoped_where(
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    label: str | None = None,
) -> tuple[str, list[str | int]]:
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    if label is not None:
        clause, l_params = _label_exists_clause(label)
        where = (where + " AND " + clause) if where else (" WHERE " + clause)
        params.extend(l_params)
    return where, params


def _fit_baseline(
    conn: sqlite3.Connection,
    columns: Sequence[str],
    where: str,
    params: Sequence[str | int],
) -> tuple[int, list[FeatureBaseline], list[str]]:
    total = conn.execute("SELECT COUNT(*) FROM samples AS s" + where, params).fetchone()[0]
    if total < 2:
        raise ValueError("la référence doit contenir au moins 2 mesures")
    prefix = (where + " AND ") if where else " WHERE "
    fitted: list[FeatureBaseline] = []
    skipped: list[str] = []
    for col in columns:
        qc = quote_identifier(col)
        values = [
            row[0]
            for row in conn.execute(
                f"SELECT {qc} FROM samples AS s{prefix}{qc} IS NOT NULL ORDER BY {qc}",
                params,
            )
        ]
        n = len(values)
        if n < 2:
            skipped.append(col)
            continue
        median = _quantile(values, 0.50)
        abs_devs = sorted(abs(v - median) for v in values)
        mad = _quantile(abs_devs, 0.50)
        if mad is not None and mad > 0.0:
            fitted.append(FeatureBaseline(col, median, mad / _MAD_SCALE, "mad", n))
            continue
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(max(0.0, var))
        if std > 0.0:
            fitted.append(FeatureBaseline(col, mean, std, "std", n))
        else:
            skipped.append(col)
    if not fitted:
        raise ValueError("aucune métrique numérique non constante dans la référence")
    return total, fitted, skipped


def _f1_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def detect_anomalies(
    db: str | pathlib.Path,
    *,
    baseline_label: str | None = None,
    baseline_boot: str | None = None,
    baseline_since: int | None = None,
    baseline_until: int | None = None,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    target_label: str | None = None,
    eval_label: str | None = None,
    columns: Sequence[str] | None = None,
    threshold: float = 4.0,
    min_samples: int = 3,
    top_contributors: int = 3,
    event_margin_ms: int = 5000,
) -> dict:
    """Ajuste une référence robuste et détecte les épisodes anormaux sur la cible."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("--threshold doit être un nombre fini > 0")
    if min_samples < 1:
        raise ValueError("--min-samples doit être >= 1")
    if top_contributors < 1:
        raise ValueError("top_contributors doit être >= 1")
    if event_margin_ms < 0:
        raise ValueError("--event-margin-ms doit être >= 0")
    validate_time_range(baseline_since, baseline_until)
    validate_time_range(since, until)
    if baseline_label is not None:
        baseline_label = baseline_label.strip()
        if not baseline_label:
            raise ValueError("--baseline-label ne peut pas être vide")
    if target_label is not None:
        target_label = target_label.strip()
        if not target_label:
            raise ValueError("--target-label ne peut pas être vide")
    if eval_label is not None:
        eval_label = eval_label.strip()
        if not eval_label:
            raise ValueError("--eval-label ne peut pas être vide")

    base_where, base_params = _scoped_where(
        boot_id=baseline_boot, since=baseline_since, until=baseline_until, label=baseline_label,
    )
    target_where, target_params = _scoped_where(
        boot_id=boot_id, since=since, until=until, label=target_label,
    )

    with read_snapshot(db) as conn:
        metrics, order_col = _numeric_metrics(conn)
        numeric_names = [name for name, _ in metrics]
        if columns is not None:
            selected = list(columns)
            if not selected:
                raise ValueError("aucune colonne sélectionnée")
            unknown = set(selected) - set(numeric_names)
            if unknown:
                raise ValueError(f"métriques numériques introuvables: {', '.join(sorted(unknown))}")
        else:
            selected = numeric_names

        baseline_count, fitted, skipped = _fit_baseline(conn, selected, base_where, base_params)

        projection = ", ".join(f"s.{quote_identifier(fb.column)}" for fb in fitted)
        eval_select = ""
        query_params: list[str | int] = []
        if eval_label is not None:
            clause, e_params = _label_exists_clause(eval_label)
            eval_select = f", ({clause}) AS is_eval_positive"
            query_params.extend(e_params)
        query_params.extend(target_params)

        cursor = conn.execute(
            f"SELECT s.boot_id, s.ts, {projection}{eval_select}"
            f" FROM samples AS s{target_where}"
            f" ORDER BY s.boot_id, s.{quote_identifier(order_col)}",
            query_params,
        )

        episodes: list[dict] = []
        total_scored = 0
        anomalous_samples = 0
        # Pour l'évaluation : on attend la fin de chaque run pour savoir s'il atteint min_samples.
        tp = fp = fn = tn = 0

        run_boot: str | None = None
        run_ts_start: int | None = None
        run_ts_end: int | None = None
        run_len = 0
        run_peak = 0.0
        run_rms_sum = 0.0
        run_eval_pos = 0
        # column -> (peak_abs_z, signed_peak_z, observed_at_peak, baseline_center)
        run_contrib: dict[str, tuple[float, float, float, float]] = {}

        def flush_run() -> None:
            nonlocal run_boot, run_ts_start, run_ts_end, run_len, run_peak, run_rms_sum
            nonlocal run_eval_pos, run_contrib, anomalous_samples, tp, fp, fn, tn
            if run_len == 0:
                return
            if run_len >= min_samples:
                anomalous_samples += run_len
                if eval_label is not None:
                    tp += run_eval_pos
                    fp += run_len - run_eval_pos
                ranked = sorted(
                    run_contrib.items(),
                    key=lambda kv: (-kv[1][0], kv[0]),
                )[:top_contributors]
                episodes.append({
                    "boot_id": run_boot,
                    "ts_start": run_ts_start,
                    "ts_end": run_ts_end,
                    "sample_count": run_len,
                    "peak_score": run_peak,
                    "mean_rms_z": run_rms_sum / run_len,
                    "top_contributors": [
                        {
                            "column": col,
                            "peak_z": signed_z,
                            "observed_at_peak": obs,
                            "baseline_center": center,
                        }
                        for col, (_, signed_z, obs, center) in ranked
                    ],
                })
            elif eval_label is not None:
                # Run trop court -> prédit négatif.
                fn += run_eval_pos
                tn += run_len - run_eval_pos
            run_boot = None
            run_ts_start = None
            run_ts_end = None
            run_len = 0
            run_peak = 0.0
            run_rms_sum = 0.0
            run_eval_pos = 0
            run_contrib = {}

        n_feat = len(fitted)
        for row in cursor:
            boot = row[0]
            ts = row[1]
            values = row[2 : 2 + n_feat]
            is_pos = bool(row[2 + n_feat]) if eval_label is not None else False
            total_scored += 1

            if run_len > 0 and boot != run_boot:
                flush_run()

            max_abs = 0.0
            sq_sum = 0.0
            n_obs = 0
            z_obs: list[tuple[FeatureBaseline, float, float]] = []
            for fb, val in zip(fitted, values):
                if val is None:
                    continue
                z = (val - fb.center) / fb.scale
                az = abs(z)
                if az > max_abs:
                    max_abs = az
                sq_sum += z * z
                n_obs += 1
                z_obs.append((fb, z, val))

            if n_obs > 0 and max_abs >= threshold:
                rms_z = math.sqrt(sq_sum / n_obs)
                if run_len == 0:
                    run_boot = boot
                    run_ts_start = ts
                run_ts_end = ts
                run_len += 1
                if max_abs > run_peak:
                    run_peak = max_abs
                run_rms_sum += rms_z
                if is_pos:
                    run_eval_pos += 1
                for fb, z, val in z_obs:
                    az = abs(z)
                    prev = run_contrib.get(fb.column)
                    if prev is None or az > prev[0]:
                        run_contrib[fb.column] = (az, z, val, fb.center)
            else:
                flush_run()
                if eval_label is not None:
                    if is_pos:
                        fn += 1
                    else:
                        tn += 1

        flush_run()

        has_events = bool(
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'").fetchone()
        )
        for ep in episodes:
            if has_events:
                ev_rows = conn.execute(
                    "SELECT ts, source, kind, severity, message FROM events"
                    " WHERE ts >= ? AND ts <= ? ORDER BY ts, id",
                    (ep["ts_start"] - event_margin_ms, ep["ts_end"] + event_margin_ms),
                ).fetchall()
                ep["events"] = [
                    {"ts": r[0], "source": r[1], "kind": r[2], "severity": r[3], "message": r[4]}
                    for r in ev_rows
                ]
            else:
                ep["events"] = []

    result = {
        "baseline": {
            "label": baseline_label,
            "boot_id": baseline_boot,
            "window": {"since": baseline_since, "until": baseline_until},
            "samples": baseline_count,
            "features_used": len(fitted),
            "features_skipped": skipped,
        },
        "target": {
            "label": target_label,
            "boot_id": boot_id,
            "window": {"since": since, "until": until},
            "samples": total_scored,
            "anomalous_samples": anomalous_samples,
            "anomalous_pct": (anomalous_samples / total_scored * 100.0) if total_scored else None,
        },
        "threshold": threshold,
        "min_samples": min_samples,
        "episodes": episodes,
    }
    if eval_label is not None:
        result["evaluation"] = {
            "label": eval_label,
            **_f1_metrics(tp, fp, fn, tn),
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Détecteur statistique d'anomalies AAD (z-score robuste / MAD)"
    )
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--baseline-label", help="label de référence (ex. normal)")
    parser.add_argument("--baseline-boot", help="boot_id de référence")
    parser.add_argument("--baseline-since", type=int, help="début référence (epoch ms inclusif)")
    parser.add_argument("--baseline-until", type=int, help="fin référence (epoch ms exclusif)")
    parser.add_argument("--boot", dest="boot_id", help="limiter la cible à un boot_id")
    parser.add_argument("--since", type=int, help="début cible (epoch ms inclusif)")
    parser.add_argument("--until", type=int, help="fin cible (epoch ms exclusif)")
    parser.add_argument("--target-label", help="limiter la cible à un label")
    parser.add_argument("--eval-label", help="évaluer précision/rappel/F1 face à un label terrain")
    parser.add_argument("--columns", nargs="+", help="sous-ensemble de métriques numériques")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="seuil de z-score absolu (défaut : 4.0)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="longueur minimale d'un épisode anormal (défaut : 3)")
    parser.add_argument("--event-margin-ms", type=int, default=5000,
                        help="marge (ms) pour associer les événements autour d'un épisode (défaut : 5000)")
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)

    if not any(v is not None for v in (
        args.baseline_label, args.baseline_boot, args.baseline_since, args.baseline_until
    )):
        parser.error("spécifiez une référence (--baseline-label, --baseline-boot ou --baseline-since/--baseline-until)")

    try:
        result = detect_anomalies(
            args.db,
            baseline_label=args.baseline_label,
            baseline_boot=args.baseline_boot,
            baseline_since=args.baseline_since,
            baseline_until=args.baseline_until,
            boot_id=args.boot_id,
            since=args.since,
            until=args.until,
            target_label=args.target_label,
            eval_label=args.eval_label,
            columns=args.columns,
            threshold=args.threshold,
            min_samples=args.min_samples,
            event_margin_ms=args.event_margin_ms,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    base = result["baseline"]
    tgt = result["target"]
    pct = "indéfinie" if tgt["anomalous_pct"] is None else f"{tgt['anomalous_pct']:.1f} %"
    print(f"Référence : {base['samples']} mesures, {base['features_used']} métrique(s) active(s)")
    print(f"Cible     : {tgt['samples']} mesures, {tgt['anomalous_samples']} anormale(s) ({pct})")
    print(f"Épisodes  : {len(result['episodes'])} (seuil z >= {result['threshold']}, min {result['min_samples']} mesures)")
    for idx, ep in enumerate(result["episodes"], 1):
        contribs = ", ".join(
            f"{c['column']} (z={c['peak_z']:+.1f}, val={c['observed_at_peak']:.2f} vs {c['baseline_center']:.2f})"
            for c in ep["top_contributors"]
        )
        print(
            f"  #{idx} boot={ep['boot_id']} [{ep['ts_start']}..{ep['ts_end']}] "
            f"n={ep['sample_count']} pic={ep['peak_score']:.1f} — {contribs}"
        )
        if ep["events"]:
            kinds = ", ".join(sorted({f"{e['source']}/{e['kind']}" for e in ep["events"]}))
            print(f"       événements proches ({len(ep['events'])}) : {kinds}")
    if "evaluation" in result:
        ev = result["evaluation"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(
            f"Évaluation ({ev['label']}) : precision={fmt(ev['precision'])}, "
            f"recall={fmt(ev['recall'])}, f1={fmt(ev['f1'])} "
            f"(tp={ev['tp']} fp={ev['fp']} fn={ev['fn']} tn={ev['tn']})"
        )


if __name__ == "__main__":
    main()
