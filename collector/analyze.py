"""Analyse exploratoire du dataset AAD, en lecture seule sans migration.

Fournit trois diagnostics essentiels avant d'entraîner un modèle d'anomalies :
- cadence d'échantillonnage par boot (médiane, p95, trous, retours d'horloge) ;
- statistiques descriptives par métrique numérique (calculées dans SQLite) ;
- équilibre des classes labellisées et comparaison entre deux expériences.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
import pathlib
import re
import sqlite3

from .database import read_snapshot, sample_filter_sql, validate_time_range
from .storage import quote_identifier

_META_COLUMNS = frozenset({"sample_id", "ts", "boot_id"})
#: Facteur de cohérence normale pour le z-score modifié d'Iglewicz & Hoaglin.
_MAD_SCALE = 0.6744897501960817


def _percentage(part: int, total: int) -> float | None:
    return part / total * 100.0 if total else None


def _quantile(sorted_values: Sequence[int | float], q: float) -> float | None:
    """Quantile par interpolation linéaire (méthode R7, identique à numpy/pandas)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _numeric_metrics(conn: sqlite3.Connection) -> tuple[list[tuple[str, str]], str]:
    fields = conn.execute("PRAGMA table_info(samples)").fetchall()
    names = {field[1] for field in fields}
    if not {"ts", "boot_id"} <= names:
        raise ValueError("la base ne contient pas de table samples compatible")
    order_col = "sample_id" if "sample_id" in names else "ts"
    metrics = [
        (field[1], field[2])
        for field in fields
        if field[1] not in _META_COLUMNS and (field[2] or "").upper() != "TEXT"
    ]
    return metrics, order_col


def _compute_numeric_stats(
    conn: sqlite3.Connection,
    metrics: list[tuple[str, str]],
    total_samples: int,
    where: str,
    params: Sequence[str | int],
) -> list[dict]:
    """Statistiques descriptives en deux passes SQLite (O(1) mémoire Python)."""
    if not metrics:
        return []
    agg_exprs = []
    for name, _ in metrics:
        q = quote_identifier(name)
        agg_exprs.append(f"COUNT({q}), MIN({q}), MAX({q}), AVG({q})")
    row = conn.execute(
        "SELECT " + ", ".join(agg_exprs) + " FROM samples AS s" + where,
        params,
    ).fetchone()

    pass1: list[tuple[str, str, int, float | None, float | None, float | None]] = []
    var_exprs: list[str] = []
    var_params: list[float] = []
    var_indices: list[int] = []
    for idx, (name, coltype) in enumerate(metrics):
        present, minimum, maximum, mean = row[idx * 4 : (idx + 1) * 4]
        pass1.append((name, coltype, present, minimum, maximum, mean))
        if present > 1 and mean is not None:
            q = quote_identifier(name)
            var_exprs.append(f"SUM(({q} - ?) * ({q} - ?))")
            var_params.extend((mean, mean))
            var_indices.append(idx)

    stds: dict[int, float] = {}
    if var_exprs:
        sq_row = conn.execute(
            "SELECT " + ", ".join(var_exprs) + " FROM samples AS s" + where,
            [*var_params, *params],
        ).fetchone()
        for idx, sum_sq in zip(var_indices, sq_row):
            present = pass1[idx][2]
            if sum_sq is not None and present > 1:
                stds[idx] = math.sqrt(max(0.0, sum_sq) / (present - 1))

    result = []
    for idx, (name, coltype, present, minimum, maximum, mean) in enumerate(pass1):
        result.append({
            "column": name,
            "type": coltype,
            "present": present,
            "missing": total_samples - present,
            "coverage_pct": _percentage(present, total_samples),
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": stds.get(idx),
            "is_constant": present > 0 and minimum == maximum,
        })
    return result


def _compute_cadence(
    conn: sqlite3.Connection,
    order_col: str,
    where: str,
    params: Sequence[str | int],
    gap_factor: float,
) -> list[dict]:
    cursor = conn.execute(
        f"SELECT s.boot_id, s.ts FROM samples AS s{where}"
        f" ORDER BY s.boot_id, s.{quote_identifier(order_col)}",
        params,
    )
    boots: list[dict] = []
    current_boot: str | None = None
    prev_ts: int | None = None
    samples = 0
    duplicates = 0
    regressions = 0
    positive_dts: list[int] = []

    def finish_boot() -> None:
        if current_boot is None:
            return
        positive_dts.sort()
        median_dt = _quantile(positive_dts, 0.5)
        p95_dt = _quantile(positive_dts, 0.95)
        gaps = 0
        if median_dt is not None and median_dt > 0:
            threshold = gap_factor * median_dt
            gaps = sum(1 for dt in positive_dts if dt > threshold)
        boots.append({
            "boot_id": current_boot,
            "samples": samples,
            "intervals": max(0, samples - 1),
            "min_dt_ms": positive_dts[0] if positive_dts else None,
            "median_dt_ms": median_dt,
            "p95_dt_ms": p95_dt,
            "max_dt_ms": positive_dts[-1] if positive_dts else None,
            "mean_dt_ms": sum(positive_dts) / len(positive_dts) if positive_dts else None,
            "duplicate_timestamps": duplicates,
            "clock_regressions": regressions,
            "gaps": gaps,
        })

    for boot, ts in cursor:
        if boot != current_boot:
            finish_boot()
            current_boot = boot
            prev_ts = ts
            samples = 1
            duplicates = 0
            regressions = 0
            positive_dts = []
            continue
        samples += 1
        dt = ts - prev_ts
        if dt == 0:
            duplicates += 1
        elif dt < 0:
            regressions += 1
        else:
            positive_dts.append(dt)
        prev_ts = ts

    finish_boot()
    return boots


def _compute_label_summary(
    conn: sqlite3.Connection,
    sample_where: str,
    sample_params: Sequence[str | int],
    since: int | None,
    until: int | None,
) -> list[dict]:
    label_filters: list[str] = []
    time_params: list[int] = []
    event_filters: list[str] = []
    if since is not None:
        label_filters.append("(ts_end IS NULL OR ts_end > ?)")
        event_filters.append("e.ts >= ?")
        time_params.append(since)
    if until is not None:
        label_filters.append("ts_start < ?")
        event_filters.append("e.ts < ?")
        time_params.append(until)
    label_where = (
        " WHERE (ts_end IS NULL OR ts_start < ts_end) AND " + " AND ".join(label_filters)
        if label_filters
        else ""
    )
    rows = conn.execute(
        "SELECT label, COUNT(*), COALESCE(SUM(ts_end IS NULL), 0),"
        " COALESCE(SUM(CASE WHEN ts_end IS NOT NULL THEN ts_end - ts_start ELSE 0 END), 0)"
        " FROM labels" + label_where + " GROUP BY label ORDER BY label",
        time_params,
    ).fetchall()

    sample_and = (sample_where + " AND ") if sample_where else " WHERE "
    event_where = (" WHERE " + " AND ".join(event_filters) + " AND ") if event_filters else " WHERE "

    summary = []
    for name, windows, open_windows, closed_duration_ms in rows:
        sample_count = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + sample_and
            + "EXISTS (SELECT 1 FROM labels AS l WHERE l.label = ?"
            " AND l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end))",
            [*sample_params, name],
        ).fetchone()[0]
        event_count = conn.execute(
            "SELECT COUNT(*) FROM events AS e" + event_where
            + "EXISTS (SELECT 1 FROM labels AS l WHERE l.label = ?"
            " AND l.ts_start <= e.ts AND (l.ts_end IS NULL OR e.ts < l.ts_end))",
            [*time_params, name],
        ).fetchone()[0]
        summary.append({
            "label": name,
            "windows": windows,
            "open_windows": open_windows,
            "closed_duration_ms": closed_duration_ms,
            "samples": sample_count,
            "events": event_count,
        })
    return summary


def analyze_dataset(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    gap_factor: float = 2.0,
) -> dict:
    """Analyse exploratoire sur un snapshot SQLite cohérent en lecture seule."""
    validate_time_range(since, until)
    if not math.isfinite(gap_factor) or gap_factor <= 1.0:
        raise ValueError("--gap-factor doit être un nombre fini > 1")
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, order_col = _numeric_metrics(conn)
        total_samples = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + where, params
        ).fetchone()[0]
        return {
            "boot_id": boot_id,
            "window": {"since": since, "until": until},
            "gap_factor": gap_factor,
            "samples": total_samples,
            "cadence": _compute_cadence(conn, order_col, where, params, gap_factor),
            "metrics": _compute_numeric_stats(conn, metrics, total_samples, where, params),
            "labels": _compute_label_summary(conn, where, params, since, until),
        }


def compare_labels(
    db: str | pathlib.Path,
    baseline_label: str,
    target_label: str,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Compare la distribution des métriques numériques entre deux labels.

    Mesure le décalage moyen (`delta_mean`) et le décalage normalisé par
    l'écart-type de la référence (`z_shift`), sans inférer de vérité terrain.
    """
    baseline_label = baseline_label.strip()
    target_label = target_label.strip()
    if not baseline_label or not target_label:
        raise ValueError("les noms de labels à comparer ne peuvent pas être vides")
    validate_time_range(since, until)
    base_where, base_params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    label_exists = (
        "EXISTS (SELECT 1 FROM labels AS l WHERE l.label = ?"
        " AND l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end))"
    )
    prefix = (base_where + " AND ") if base_where else " WHERE "
    single_where = prefix + label_exists
    overlap_where = prefix + label_exists + " AND " + label_exists

    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        baseline_params = [*base_params, baseline_label]
        target_params = [*base_params, target_label]
        baseline_count = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + single_where, baseline_params
        ).fetchone()[0]
        target_count = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + single_where, target_params
        ).fetchone()[0]
        overlap_count = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + overlap_where,
            [*base_params, baseline_label, target_label],
        ).fetchone()[0]

        base_stats = _compute_numeric_stats(conn, metrics, baseline_count, single_where, baseline_params)
        target_stats = _compute_numeric_stats(conn, metrics, target_count, single_where, target_params)

    comparisons = []
    for base, target in zip(base_stats, target_stats):
        b_mean, b_std = base["mean"], base["std"]
        t_mean, t_std = target["mean"], target["std"]
        delta = None if b_mean is None or t_mean is None else t_mean - b_mean
        z_shift = None if delta is None or b_std is None or b_std == 0.0 else delta / b_std
        comparisons.append({
            "column": base["column"],
            "baseline_present": base["present"],
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "target_present": target["present"],
            "target_mean": t_mean,
            "target_std": t_std,
            "delta_mean": delta,
            "z_shift": z_shift,
        })
    comparisons.sort(
        key=lambda item: (item["z_shift"] is None, -(abs(item["z_shift"]) if item["z_shift"] is not None else 0.0), item["column"])
    )
    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "baseline": {"label": baseline_label, "samples": baseline_count},
        "target": {"label": target_label, "samples": target_count},
        "overlap_samples": overlap_count,
        "metrics": comparisons,
    }


def correlate_metric(
    db: str | pathlib.Path,
    target_column: str,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    min_pairs: int = 3,
) -> dict:
    """Corrélation de Pearson entre une métrique cible et les autres sondes numériques.

    Calculée en deux passes SQLite sur l'intersection non-NULL de chaque paire
    (O(1) mémoire Python), sans imputation des valeurs manquantes.
    """
    if min_pairs < 2:
        raise ValueError("--min-pairs doit être >= 2")
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        numeric_names = [name for name, _ in metrics]
        if target_column not in numeric_names:
            raise ValueError(f"métrique numérique introuvable: {target_column!r}")
        others = [name for name in numeric_names if name != target_column]
        qt = quote_identifier(target_column)
        correlations: list[dict] = []
        if others:
            p1_exprs = []
            for col in others:
                qc = quote_identifier(col)
                cond = f"{qt} IS NOT NULL AND {qc} IS NOT NULL"
                p1_exprs.append(
                    f"COUNT(CASE WHEN {cond} THEN 1 END),"
                    f" AVG(CASE WHEN {cond} THEN {qt} END),"
                    f" AVG(CASE WHEN {cond} THEN {qc} END)"
                )
            p1_row = conn.execute(
                "SELECT " + ", ".join(p1_exprs) + " FROM samples AS s" + where,
                params,
            ).fetchone()

            p2_exprs = []
            p2_params: list[float] = []
            p2_cols: list[tuple[int, str, int]] = []
            pairs_by_col: dict[str, int] = {}
            for idx, col in enumerate(others):
                pairs, mx, my = p1_row[idx * 3 : (idx + 1) * 3]
                pairs_by_col[col] = pairs
                if pairs >= min_pairs and mx is not None and my is not None:
                    qc = quote_identifier(col)
                    cond = f"{qt} IS NOT NULL AND {qc} IS NOT NULL"
                    p2_exprs.append(
                        f"SUM(CASE WHEN {cond} THEN ({qt} - ?) * ({qc} - ?) END),"
                        f" SUM(CASE WHEN {cond} THEN ({qt} - ?) * ({qt} - ?) END),"
                        f" SUM(CASE WHEN {cond} THEN ({qc} - ?) * ({qc} - ?) END)"
                    )
                    p2_params.extend((mx, my, mx, mx, my, my))
                    p2_cols.append((len(p2_cols), col, pairs))

            r_by_col: dict[str, float | None] = {}
            if p2_exprs:
                p2_row = conn.execute(
                    "SELECT " + ", ".join(p2_exprs) + " FROM samples AS s" + where,
                    [*p2_params, *params],
                ).fetchone()
                for pos, col, _ in p2_cols:
                    sxy, sxx, syy = p2_row[pos * 3 : (pos + 1) * 3]
                    if sxy is None or sxx is None or syy is None:
                        r_by_col[col] = None
                        continue
                    denom = math.sqrt(max(0.0, sxx) * max(0.0, syy))
                    r_by_col[col] = max(-1.0, min(1.0, sxy / denom)) if denom > 0.0 else None

            for col in others:
                r = r_by_col.get(col)
                correlations.append({
                    "column": col,
                    "pairs": pairs_by_col[col],
                    "r": r,
                    "r_squared": None if r is None else r * r,
                })
            correlations.sort(
                key=lambda item: (
                    item["r"] is None,
                    -(abs(item["r"]) if item["r"] is not None else 0.0),
                    item["column"],
                )
            )
    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "target": target_column,
        "min_pairs": min_pairs,
        "correlations": correlations,
    }


def detect_outliers(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    z_threshold: float = 3.5,
    columns: Sequence[str] | None = None,
) -> dict:
    """Détecte les valeurs atypiques par médiane/MAD (z-score modifié) et IQR.

    Ne modifie ni les mesures ni les labels : signale simplement les sondes qui
    présentent des pics extrêmes ou des artefacts capteurs avant modélisation.
    """
    if not math.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("--z-threshold doit être un nombre fini > 0")
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    prefix = (where + " AND ") if where else " WHERE "
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        numeric_names = [name for name, _ in metrics]
        if columns is not None:
            selected = list(columns)
            if not selected:
                raise ValueError("aucune colonne spécifiée pour --outliers")
            unknown = set(selected) - set(numeric_names)
            if unknown:
                raise ValueError(f"métriques numériques introuvables: {', '.join(sorted(unknown))}")
        else:
            selected = numeric_names

        results: list[dict] = []
        for col in selected:
            qc = quote_identifier(col)
            values = [
                row[0]
                for row in conn.execute(
                    f"SELECT {qc} FROM samples AS s{prefix}{qc} IS NOT NULL ORDER BY {qc}",
                    params,
                )
            ]
            present = len(values)
            if not present:
                results.append({
                    "column": col, "present": 0, "p25": None, "median": None, "p75": None,
                    "iqr": None, "mad": None, "outlier_count": 0, "outlier_pct": None,
                    "min_outlier": None, "max_outlier": None,
                })
                continue
            p25 = _quantile(values, 0.25)
            median = _quantile(values, 0.50)
            p75 = _quantile(values, 0.75)
            iqr = p75 - p25
            abs_devs = sorted(abs(v - median) for v in values)
            mad = _quantile(abs_devs, 0.50)
            if mad is not None and mad > 0.0:
                outliers = [v for v in values if _MAD_SCALE * abs(v - median) / mad > z_threshold]
            else:
                outliers = []
            results.append({
                "column": col,
                "present": present,
                "p25": p25,
                "median": median,
                "p75": p75,
                "iqr": iqr,
                "mad": mad,
                "outlier_count": len(outliers),
                "outlier_pct": _percentage(len(outliers), present),
                "min_outlier": outliers[0] if outliers else None,
                "max_outlier": outliers[-1] if outliers else None,
            })
        results.sort(key=lambda item: (-item["outlier_count"], item["column"]))
    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "z_threshold": z_threshold,
        "metrics": results,
    }


def audit_labels(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Vérifie la qualité des fenêtres d'annotations avant entraînement.

    Signale les fenêtres vides, ouvertes, sans aucune mesure associée, et les
    chevauchements (en distinguant les chevauchements entre labels distincts).
    """
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    prefix = (where + " AND ") if where else " WHERE "
    with read_snapshot(db) as conn:
        _numeric_metrics(conn)  # valide la présence de samples
        rows = conn.execute(
            "SELECT id, ts_start, ts_end, label, notes FROM labels ORDER BY ts_start, id"
        ).fetchall()
        windows = [
            {"id": r[0], "ts_start": r[1], "ts_end": r[2], "label": r[3], "notes": r[4]}
            for r in rows
        ]
        empty_windows = [w for w in windows if w["ts_end"] is not None and w["ts_end"] <= w["ts_start"]]
        open_windows = [w for w in windows if w["ts_end"] is None]
        non_empty = [w for w in windows if w["ts_end"] is None or w["ts_end"] > w["ts_start"]]

        sampleless_windows = []
        for w in non_empty:
            if w["ts_end"] is None:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM samples AS s{prefix}s.ts >= ?",
                    [*params, w["ts_start"]],
                ).fetchone()[0]
            else:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM samples AS s{prefix}s.ts >= ? AND s.ts < ?",
                    [*params, w["ts_start"], w["ts_end"]],
                ).fetchone()[0]
            if count == 0:
                sampleless_windows.append(w)

        overlaps = []
        for i, a in enumerate(non_empty):
            for b in non_empty[i + 1 :]:
                if a["ts_end"] is not None and b["ts_start"] >= a["ts_end"]:
                    break  # trié par ts_start : aucun suivant ne peut chevaucher a
                if b["ts_end"] is not None and a["ts_start"] >= b["ts_end"]:
                    continue
                ov_start = max(a["ts_start"], b["ts_start"])
                if a["ts_end"] is None:
                    ov_end = b["ts_end"]
                elif b["ts_end"] is None:
                    ov_end = a["ts_end"]
                else:
                    ov_end = min(a["ts_end"], b["ts_end"])
                overlaps.append({
                    "id_a": a["id"],
                    "label_a": a["label"],
                    "id_b": b["id"],
                    "label_b": b["label"],
                    "same_label": a["label"] == b["label"],
                    "overlap_start": ov_start,
                    "overlap_end": ov_end,
                })

        cross_label_overlap_samples = conn.execute(
            "SELECT COUNT(*) FROM samples AS s" + prefix
            + "(SELECT COUNT(DISTINCT l.label) FROM labels AS l"
            " WHERE l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end)) > 1",
            params,
        ).fetchone()[0]

    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "total_windows": len(windows),
        "empty_windows": empty_windows,
        "open_windows": open_windows,
        "sampleless_windows": sampleless_windows,
        "overlaps": overlaps,
        "cross_label_overlap_samples": cross_label_overlap_samples,
    }


def analyze_drift(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    slices: int = 2,
) -> dict:
    """Mesure la dérive entre la première et la dernière tranche temporelle.

    Découpe l'étendue observée [min_ts, max_ts] en `slices` tranches de durée
    égale et compare les statistiques SQLite de la première et de la dernière
    tranche (útil pour repérer un encrassement ou un échauffement progressif).
    """
    if slices < 2:
        raise ValueError("--slices doit être >= 2")
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        total, min_ts, max_ts = conn.execute(
            "SELECT COUNT(*), MIN(s.ts), MAX(s.ts) FROM samples AS s" + where, params
        ).fetchone()
        if not total or min_ts is None or max_ts is None or max_ts <= min_ts:
            return {
                "boot_id": boot_id,
                "window": {"since": since, "until": until},
                "slices": slices,
                "first_slice": {"since": min_ts, "until": min_ts, "samples": 0},
                "last_slice": {"since": max_ts, "until": max_ts, "samples": 0},
                "metrics": [],
            }
        span = max_ts - min_ts + 1
        width = max(1, span // slices)
        first_until = min_ts + width
        last_since = max_ts + 1 - width
        last_until = max_ts + 1

        f_where, f_params = sample_filter_sql(boot_id=boot_id, since=min_ts, until=first_until, alias="s")
        l_where, l_params = sample_filter_sql(boot_id=boot_id, since=last_since, until=last_until, alias="s")
        first_count = conn.execute("SELECT COUNT(*) FROM samples AS s" + f_where, f_params).fetchone()[0]
        last_count = conn.execute("SELECT COUNT(*) FROM samples AS s" + l_where, l_params).fetchone()[0]
        first_stats = _compute_numeric_stats(conn, metrics, first_count, f_where, f_params)
        last_stats = _compute_numeric_stats(conn, metrics, last_count, l_where, l_params)

    drifts = []
    for first, last in zip(first_stats, last_stats):
        f_mean, f_std = first["mean"], first["std"]
        l_mean, l_std = last["mean"], last["std"]
        delta = None if f_mean is None or l_mean is None else l_mean - f_mean
        z_drift = None if delta is None or f_std is None or f_std == 0.0 else delta / f_std
        drifts.append({
            "column": first["column"],
            "first_present": first["present"],
            "first_mean": f_mean,
            "first_std": f_std,
            "last_present": last["present"],
            "last_mean": l_mean,
            "last_std": l_std,
            "delta_mean": delta,
            "z_drift": z_drift,
        })
    drifts.sort(
        key=lambda item: (item["z_drift"] is None, -(abs(item["z_drift"]) if item["z_drift"] is not None else 0.0), item["column"])
    )
    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "slices": slices,
        "first_slice": {"since": min_ts, "until": first_until, "samples": first_count},
        "last_slice": {"since": last_since, "until": last_until, "samples": last_count},
        "metrics": drifts,
    }


def analyze_by_group(
    db: str | pathlib.Path,
    group_column: str,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    columns: Sequence[str] | None = None,
    max_groups: int = 50,
) -> dict:
    """Statistiques descriptives ventilées par valeur discrète d'une colonne de contexte.

    Conserve `NULL` comme groupe explicite (contexte inconnu). Rejette les
    colonnes à trop forte cardinalité (`> max_groups`) pour éviter de grouper
    par erreur sur une métrique continue.
    """
    if max_groups < 1:
        raise ValueError("max_groups doit être >= 1")
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    prefix = (where + " AND ") if where else " WHERE "
    with read_snapshot(db) as conn:
        fields = conn.execute("PRAGMA table_info(samples)").fetchall()
        all_names = [f[1] for f in fields]
        if group_column not in all_names:
            raise ValueError(f"colonne de regroupement introuvable: {group_column!r}")
        numeric_metrics = [
            (f[1], f[2])
            for f in fields
            if f[1] not in _META_COLUMNS and f[1] != group_column and (f[2] or "").upper() != "TEXT"
        ]
        if columns is not None:
            numeric_by_name = dict(numeric_metrics)
            selected = list(columns)
            if not selected:
                raise ValueError("aucune colonne numérique sélectionnée")
            unknown = set(selected) - set(numeric_by_name)
            if unknown:
                raise ValueError(f"métriques numériques introuvables: {', '.join(sorted(unknown))}")
            metrics = [(col, numeric_by_name[col]) for col in selected]
        else:
            metrics = numeric_metrics

        qg = quote_identifier(group_column)
        group_rows = conn.execute(
            f"SELECT s.{qg}, COUNT(*) FROM samples AS s{where} GROUP BY s.{qg}"
            f" ORDER BY s.{qg} IS NULL, s.{qg}",
            params,
        ).fetchall()
        if len(group_rows) > max_groups:
            raise ValueError(
                f"trop de groupes distincts pour {group_column!r} ({len(group_rows)} > {max_groups})"
            )

        groups = []
        for g_val, g_count in group_rows:
            if g_val is None:
                g_where = f"{prefix}s.{qg} IS NULL"
                g_params = list(params)
            else:
                g_where = f"{prefix}s.{qg} = ?"
                g_params = [*params, g_val]
            stats = _compute_numeric_stats(conn, metrics, g_count, g_where, g_params)
            groups.append({
                "value": g_val,
                "samples": g_count,
                "metrics": stats,
            })

    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "group_by": group_column,
        "groups": groups,
    }


_PHYSICAL_BOUND_RULES: tuple[tuple[re.Pattern[str], float | None, float | None, str], ...] = (
    (re.compile(r"^(cpu|cpu\d+)_util$"), 0.0, 1.0, "ratio [0, 1]"),
    (re.compile(r"(_pct$|^gpu_(mem_)?util$|^psi_)"), 0.0, 100.0, "% [0, 100]"),
    (re.compile(r"(_w$|_watts$)"), 0.0, 1000.0, "W [0, 1000]"),
    (re.compile(r"_rpm$"), 0.0, 20000.0, "RPM [0, 20000]"),
    (re.compile(r"(_mhz$|^cpu_freq_avg$|^gpu_clock_)"), 0.0, 10000.0, "MHz [0, 10000]"),
    (re.compile(r"_v$"), 0.0, 20.0, "V [0, 20]"),
    (re.compile(r"(_tctl$|_tccd\d*$|^gpu_temp$|_temp\d*$|^nvme\d*_(composite|sensor)|^amdgpu_(edge|junction|mem)$)"), -20.0, 150.0, "°C [-20, 150]"),
    (re.compile(r"(_mb$|^load1$|^uptime_s$|^n_procs$|^err_|^kmsg_)"), 0.0, None, ">= 0"),
)


def _physical_bounds_for(column: str) -> tuple[float | None, float | None, str] | None:
    for pattern, lo, hi, unit in _PHYSICAL_BOUND_RULES:
        if pattern.search(column):
            return lo, hi, unit
    return None


def check_physical_bounds(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Vérifie si les métriques numériques connues respectent des bornes physiques plausibles.

    Calculé intégralement dans SQLite ; ne modifie aucune donnée.
    """
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        checked: list[tuple[str, float | None, float | None, str]] = []
        for col, _ in metrics:
            bounds = _physical_bounds_for(col)
            if bounds is not None:
                checked.append((col, *bounds))
        violations: list[dict] = []
        if checked:
            exprs = []
            query_params: list[float] = []
            for col, lo, hi, _ in checked:
                qc = quote_identifier(col)
                if lo is not None:
                    exprs.append(f"COUNT(CASE WHEN {qc} < ? THEN 1 END)")
                    query_params.append(lo)
                else:
                    exprs.append("0")
                if hi is not None:
                    exprs.append(f"COUNT(CASE WHEN {qc} > ? THEN 1 END)")
                    query_params.append(hi)
                else:
                    exprs.append("0")
                exprs.append(f"MIN({qc})")
                exprs.append(f"MAX({qc})")
            row = conn.execute(
                "SELECT " + ", ".join(exprs) + " FROM samples AS s" + where,
                [*query_params, *params],
            ).fetchone()
            for idx, (col, lo, hi, unit) in enumerate(checked):
                below, above, minimum, maximum = row[idx * 4 : (idx + 1) * 4]
                if below or above:
                    violations.append({
                        "column": col,
                        "unit": unit,
                        "expected_min": lo,
                        "expected_max": hi,
                        "below_min": below,
                        "above_max": above,
                        "observed_min": minimum,
                        "observed_max": maximum,
                    })
            violations.sort(key=lambda item: (-(item["below_min"] + item["above_max"]), item["column"]))
    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "checked_metrics": len(checked),
        "total_violations": sum(v["below_min"] + v["above_max"] for v in violations),
        "violations": violations,
    }


def detect_stuck_runs(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    min_run: int = 30,
    columns: Sequence[str] | None = None,
) -> dict:
    """Détecte les séquences consécutives de valeurs identiques (capteur figé).

    Réinitialise le suivi à chaque changement de `boot_id` ou de valeur `NULL`.
    Utile pour repérer un capteur gelé après une veille ou un blocage de pilote
    alors que la sonde n'est pas constante sur l'ensemble du dataset.
    """
    if min_run < 2:
        raise ValueError("--min-run doit être >= 2")
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
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

        findings: list[dict] = []
        if selected:
            proj = ", ".join(f"s.{quote_identifier(c)}" for c in selected)
            cursor = conn.execute(
                f"SELECT s.boot_id, s.ts, {proj} FROM samples AS s{where}"
                f" ORDER BY s.boot_id, s.{quote_identifier(order_col)}",
                params,
            )
            # Par colonne : (cur_val, cur_len, cur_start_ts, cur_end_ts)
            state: list[list] = [[None, 0, None, None] for _ in selected]
            # Par colonne : [run_count, stuck_samples, longest_len, longest_val, longest_boot, longest_start, longest_end]
            agg: list[list] = [[0, 0, 0, None, None, None, None] for _ in selected]
            current_boot: str | None = None

            def flush_col(idx: int, boot_name: str | None) -> None:
                val, length, start_ts, end_ts = state[idx]
                if length >= min_run:
                    a = agg[idx]
                    a[0] += 1
                    a[1] += length
                    if length > a[2]:
                        a[2] = length
                        a[3] = val
                        a[4] = boot_name
                        a[5] = start_ts
                        a[6] = end_ts
                state[idx] = [None, 0, None, None]

            for row in cursor:
                boot, ts = row[0], row[1]
                if boot != current_boot:
                    if current_boot is not None:
                        for i in range(len(selected)):
                            flush_col(i, current_boot)
                    current_boot = boot
                for i, val in enumerate(row[2:]):
                    if val is None:
                        flush_col(i, current_boot)
                        continue
                    cur_val, cur_len, cur_start, _ = state[i]
                    if cur_len > 0 and val == cur_val:
                        state[i][1] = cur_len + 1
                        state[i][3] = ts
                    else:
                        flush_col(i, current_boot)
                        state[i] = [val, 1, ts, ts]
            if current_boot is not None:
                for i in range(len(selected)):
                    flush_col(i, current_boot)

            for col, a in zip(selected, agg):
                if a[0] > 0:
                    findings.append({
                        "column": col,
                        "run_count": a[0],
                        "stuck_samples": a[1],
                        "longest_run": a[2],
                        "longest_value": a[3],
                        "longest_boot": a[4],
                        "longest_ts_start": a[5],
                        "longest_ts_end": a[6],
                    })
            findings.sort(key=lambda item: (-item["longest_run"], item["column"]))

    return {
        "boot_id": boot_id,
        "window": {"since": since, "until": until},
        "min_run": min_run,
        "stuck_metrics": findings,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse exploratoire du dataset AAD")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--boot", dest="boot_id", help="limiter les mesures à un boot_id")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--gap-factor", type=float, default=2.0,
                        help="multiple de la cadence médiane définissant un trou (défaut : 2.0)")
    parser.add_argument("--min-pairs", type=int, default=3,
                        help="paires minimales pour calculer une corrélation (défaut : 3)")
    parser.add_argument("--z-threshold", type=float, default=3.5,
                        help="seuil de z-score modifié (MAD) pour --outliers (défaut : 3.5)")
    parser.add_argument("--slices", type=int, default=2,
                        help="nombre de tranches temporelles pour --drift (défaut : 2)")
    parser.add_argument("--min-run", type=int, default=30,
                        help="longueur minimale d'une séquence figée pour --stuck-runs (défaut : 30)")
    parser.add_argument("--columns", nargs="+", help="sous-ensemble de métriques (pour --outliers, --by, --stuck-runs)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", nargs=2, metavar=("BASELINE", "TARGET"),
                      help="comparer les métriques numériques entre deux labels")
    mode.add_argument("--correlate", metavar="COLUMN",
                      help="corrélations de Pearson entre une métrique cible et les autres sondes")
    mode.add_argument("--outliers", action="store_true",
                      help="détecter les valeurs atypiques par médiane/MAD et IQR")
    mode.add_argument("--check-labels", action="store_true",
                      help="auditer les fenêtres d'annotations (vides, ouvertes, chevauchements)")
    mode.add_argument("--drift", action="store_true",
                      help="mesurer la dérive entre la première et la dernière tranche temporelle")
    mode.add_argument("--by", metavar="COLUMN",
                      help="ventiler les statistiques numériques par valeur discrète de contexte")
    mode.add_argument("--check-bounds", action="store_true",
                      help="vérifier les bornes physiques plausibles des sondes connues")
    mode.add_argument("--stuck-runs", action="store_true",
                      help="détecter les séquences consécutives de valeurs identiques (capteur figé)")
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)

    try:
        if args.compare:
            result = compare_labels(
                args.db, args.compare[0], args.compare[1],
                boot_id=args.boot_id, since=args.since, until=args.until,
            )
        elif args.correlate:
            result = correlate_metric(
                args.db, args.correlate,
                boot_id=args.boot_id, since=args.since, until=args.until,
                min_pairs=args.min_pairs,
            )
        elif args.outliers:
            result = detect_outliers(
                args.db, boot_id=args.boot_id, since=args.since, until=args.until,
                z_threshold=args.z_threshold, columns=args.columns,
            )
        elif args.check_labels:
            result = audit_labels(
                args.db, boot_id=args.boot_id, since=args.since, until=args.until,
            )
        elif args.drift:
            result = analyze_drift(
                args.db, boot_id=args.boot_id, since=args.since, until=args.until,
                slices=args.slices,
            )
        elif args.by:
            result = analyze_by_group(
                args.db, args.by, boot_id=args.boot_id, since=args.since,
                until=args.until, columns=args.columns,
            )
        elif args.check_bounds:
            result = check_physical_bounds(
                args.db, boot_id=args.boot_id, since=args.since, until=args.until,
            )
        elif args.stuck_runs:
            result = detect_stuck_runs(
                args.db, boot_id=args.boot_id, since=args.since, until=args.until,
                min_run=args.min_run, columns=args.columns,
            )
        else:
            result = analyze_dataset(
                args.db, boot_id=args.boot_id, since=args.since,
                until=args.until, gap_factor=args.gap_factor,
            )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.stuck_runs:
        print(f"Séquences figées (>= {result['min_run']} mesures identiques consécutives) :")
        for item in result["stuck_metrics"]:
            print(
                f"  {item['column']} : {item['run_count']} séquence(s), "
                f"max={item['longest_run']} (val={_fmt(item['longest_value'])}, boot={item['longest_boot']})"
            )
        return

    if args.check_bounds:
        print(
            f"Contrôle des bornes physiques ({result['checked_metrics']} sonde(s) vérifiée(s), "
            f"{result['total_violations']} violation(s)) :"
        )
        for v in result["violations"]:
            print(
                f"  {v['column']} ({v['unit']}) : <min={v['below_min']}, >max={v['above_max']} "
                f"(observé [{_fmt(v['observed_min'])}, {_fmt(v['observed_max'])}])"
            )
        return

    if args.by:
        print(f"Statistiques ventilées par {result['group_by']} :")
        for group in result["groups"]:
            label = "NULL" if group["value"] is None else str(group["value"])
            print(f"  [{label}] ({group['samples']} mesure(s))")
            for m in group["metrics"]:
                print(f"    {m['column']} : moy={_fmt(m['mean'])}, std={_fmt(m['std'])}, min={_fmt(m['min'])}, max={_fmt(m['max'])}")
        return

    if args.drift:
        f, l = result["first_slice"], result["last_slice"]
        print(f"Dérive temporelle ({result['slices']} tranches : {f['samples']} → {l['samples']} mesures) :")
        for item in result["metrics"]:
            print(
                f"  {item['column']} : {_fmt(item['first_mean'])} ± {_fmt(item['first_std'])}"
                f" → {_fmt(item['last_mean'])} ± {_fmt(item['last_std'])}"
                f" (Δ={_fmt(item['delta_mean'])}, z={_fmt(item['z_drift'])})"
            )
        return

    if args.check_labels:
        print(f"Audit des annotations ({result['total_windows']} fenêtre(s)) :")
        print(f"  Fenêtres vides : {len(result['empty_windows'])}")
        print(f"  Fenêtres ouvertes : {len(result['open_windows'])}")
        print(f"  Fenêtres sans mesure : {len(result['sampleless_windows'])}")
        cross = [ov for ov in result["overlaps"] if not ov["same_label"]]
        print(
            f"  Chevauchements : {len(result['overlaps'])} "
            f"(dont {len(cross)} inter-labels, "
            f"{result['cross_label_overlap_samples']} mesure(s) multi-labels)"
        )
        return

    if args.outliers:
        print(f"Valeurs atypiques (z-score modifié MAD > {result['z_threshold']}) :")
        for item in result["metrics"]:
            print(
                f"  {item['column']} : médiane={_fmt(item['median'])}, "
                f"IQR={_fmt(item['iqr'])}, MAD={_fmt(item['mad'])}, "
                f"outliers={item['outlier_count']}/{item['present']} ({_fmt(item['outlier_pct'], 1)} %)"
            )
        return

    if args.correlate:
        print(f"Corrélations de Pearson avec {result['target']} (min_pairs={result['min_pairs']}) :")
        for item in result["correlations"]:
            print(
                f"  {item['column']} : r={_fmt(item['r'], 3)}, "
                f"r²={_fmt(item['r_squared'], 3)} (n={item['pairs']})"
            )
        return

    if args.compare:
        b = result["baseline"]
        t = result["target"]
        print(f"Comparaison : {b['label']} ({b['samples']} mesures) vs {t['label']} ({t['samples']} mesures)")
        if result["overlap_samples"]:
            print(f"Attention : {result['overlap_samples']} mesure(s) portent les deux labels")
        for item in result["metrics"]:
            print(
                f"  {item['column']} : {_fmt(item['baseline_mean'])} ± {_fmt(item['baseline_std'])}"
                f" → {_fmt(item['target_mean'])} ± {_fmt(item['target_std'])}"
                f" (Δ={_fmt(item['delta_mean'])}, z={_fmt(item['z_shift'])})"
            )
        return

    print(f"Mesures analysées : {result['samples']}")
    print("Cadence par boot :")
    for boot in result["cadence"]:
        print(
            f"  boot {boot['boot_id']} : {boot['samples']} mesures, "
            f"médiane={_fmt(boot['median_dt_ms'], 1)} ms, p95={_fmt(boot['p95_dt_ms'], 1)} ms, "
            f"trous={boot['gaps']}, doublons={boot['duplicate_timestamps']}, "
            f"retours={boot['clock_regressions']}"
        )
    print("Métriques numériques :")
    for metric in result["metrics"]:
        flag = " [constante]" if metric["is_constant"] else ""
        print(
            f"  {metric['column']} : n={metric['present']}, "
            f"min={_fmt(metric['min'])}, max={_fmt(metric['max'])}, "
            f"moy={_fmt(metric['mean'])}, std={_fmt(metric['std'])}{flag}"
        )
    if result["labels"]:
        print("Classes labellisées :")
        for item in result["labels"]:
            print(
                f"  {item['label']} : {item['windows']} fenêtre(s) ({item['open_windows']} ouverte(s)), "
                f"{item['samples']} mesure(s), {item['events']} événement(s)"
            )


if __name__ == "__main__":
    main()
