"""Primitives partagées d'agrégation SQLite et de statistiques descriptives.

Extraites de `collector/analyze.py` pour être réutilisées proprement par les
modules d'analyse, de détection d'anomalies et de rapport HTML.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import sqlite3

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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Calcule les rangs moyens (gestion exacte des ex-aequo) pour Spearman."""
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson_lists(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Corrélation de Pearson en mémoire sur deux séquences finies de même taille."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(max(0.0, sxx) * max(0.0, syy))
    if denom <= 0.0:
        return None
    return max(-1.0, min(1.0, sxy / denom))


def _spearman_lists(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Corrélation de rang de Spearman (Pearson sur les rangs moyens)."""
    if len(xs) < 2:
        return None
    return _pearson_lists(_average_ranks(xs), _average_ranks(ys))


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
