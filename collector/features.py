"""Export ML avec ingénierie de features temporelles en streaming (zéro dépendance).

Calcule à la volée, sans fuite entre boots ni imputation des valeurs manquantes :
- variations instantanées (`_delta`) et taux par seconde (`_rate_s`) ;
- moyennes et écarts-types glissants (`_roll{N}_mean`, `_roll{N}_std`) ;
- décomposition binaire des flags NVML `gpu_throttle_mask`.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
import csv
import json
import math
import pathlib
import sqlite3
import sys
from typing import TextIO

from .database import read_snapshot, resolve_boot_id, resolve_last_minutes, sample_filter_sql, validate_time_range
from .export import _output
from .storage import quote_identifier

#: Masques NVML officiels (nvmlClocksEventReason* / nvmlClocksThrottleReason*).
THROTTLE_FLAGS: tuple[tuple[str, int], ...] = (
    ("gpu_throttle_idle", 0x01),
    ("gpu_throttle_power_cap", 0x04),
    ("gpu_throttle_hw_slowdown", 0x08),
    ("gpu_throttle_sw_thermal", 0x20),
    ("gpu_throttle_hw_thermal", 0x40),
    ("gpu_throttle_hw_power_brake", 0x80),
)


def decode_throttle_mask(mask: float | int | None) -> dict[str, int | None]:
    """Décompose un bitmask NVML en indicateurs binaires explicites (0/1/None)."""
    if mask is None or (isinstance(mask, float) and not math.isfinite(mask)) or mask < 0:
        return {name: None for name, _ in THROTTLE_FLAGS}
    bits = int(mask)
    return {name: int(bool(bits & flag)) for name, flag in THROTTLE_FLAGS}


def _rolling_stats(window: deque[float | None]) -> tuple[float | None, float | None]:
    present = [v for v in window if v is not None]
    n = len(present)
    if n == 0:
        return None, None
    mean = sum(present) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in present) / (n - 1)
    return mean, math.sqrt(max(0.0, var))


def export_features(
    db: str | pathlib.Path,
    target: TextIO,
    *,
    fmt: str = "csv",
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    rolling: int | None = None,
    deltas: bool = False,
    decode_throttle: bool = False,
    split_ratios: tuple[float, float, float] | None = None,
) -> int:
    """Exporte mesures + features dérivées + labels en flux depuis un snapshot lecture seule."""
    if columns is not None and exclude_columns is not None:
        raise ValueError("--columns et --exclude-columns sont mutuellement exclusifs")
    if fmt not in ("csv", "jsonl"):
        raise ValueError(f"format non supporté: {fmt!r}")
    if rolling is not None and rolling < 2:
        raise ValueError("--rolling doit être >= 2")
    if split_ratios is not None:
        if len(split_ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in split_ratios):
            raise ValueError("--split-ratios attend 3 nombres finis >= 0 (train val test)")
        ratio_sum = sum(split_ratios)
        if ratio_sum <= 0:
            raise ValueError("la somme de --split-ratios doit être > 0")
        norm_train = split_ratios[0] / ratio_sum
        norm_train_val = (split_ratios[0] + split_ratios[1]) / ratio_sum
    validate_time_range(since, until)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")

    with read_snapshot(db) as conn:
        fields = conn.execute("PRAGMA table_info(samples)").fetchall()
        available = [field[1] for field in fields]
        if not {"ts", "boot_id"} <= set(available):
            raise ValueError("la base ne contient pas de table samples compatible")
        primary = [field[1] for field in fields if field[5]]
        if len(primary) != 1:
            raise ValueError("la table samples doit avoir une clé primaire unique")

        numeric_cols = [
            field[1]
            for field in fields
            if field[1] not in ("sample_id", "ts", "boot_id") and (field[2] or "").upper() != "TEXT"
        ]
        if columns is None:
            excluded = set(exclude_columns or ())
            selected_metrics = [c for c in numeric_cols if c not in excluded]
            if not selected_metrics and excluded:
                raise ValueError("toutes les colonnes numériques ont été exclues")
        else:
            selected_metrics = list(columns)
            if not selected_metrics:
                raise ValueError("aucune colonne numérique sélectionnée")
            unknown = set(selected_metrics) - set(numeric_cols)
            if unknown:
                raise ValueError(f"colonnes numériques inconnues: {', '.join(sorted(unknown))}")
            if len(set(selected_metrics)) != len(selected_metrics):
                raise ValueError("colonnes dupliquées")

        meta_cols = [c for c in ("sample_id", "ts", "boot_id") if c in available]
        query_cols = [*meta_cols, *selected_metrics]
        key = f"s.{quote_identifier(primary[0])}"
        projection = ", ".join(f"s.{quote_identifier(c)}" for c in query_cols)

        total_for_split = 0
        if split_ratios is not None:
            total_for_split = conn.execute("SELECT COUNT(*) FROM samples AS s" + where, params).fetchone()[0]

        cursor = conn.execute(
            f"SELECT {key}, {projection}, l.id, l.ts_start, l.ts_end, l.label, l.notes"
            " FROM samples AS s LEFT JOIN labels AS l"
            " ON l.ts_start <= s.ts AND (l.ts_end IS NULL OR s.ts < l.ts_end)"
            + where + f" ORDER BY s.boot_id, s.ts, {key}, l.id",
            params,
        )

        # Construire l'en-tête de sortie
        out_cols: list[str] = [*meta_cols]
        for col in selected_metrics:
            out_cols.append(col)
            if deltas:
                out_cols.append(f"{col}_delta")
                out_cols.append(f"{col}_rate_s")
            if rolling is not None:
                out_cols.append(f"{col}_roll{rolling}_mean")
                out_cols.append(f"{col}_roll{rolling}_std")
        has_throttle = decode_throttle and "gpu_throttle_mask" in selected_metrics
        if has_throttle:
            out_cols.extend(name for name, _ in THROTTLE_FLAGS)
        if split_ratios is not None:
            out_cols.append("split")

        label_col = "labels_json" if fmt == "csv" else "labels"
        writer = csv.writer(target, lineterminator="\n") if fmt == "csv" else None
        if writer is not None:
            writer.writerow([*out_cols, label_col])

        idx_ts = query_cols.index("ts")
        idx_boot = query_cols.index("boot_id")
        metric_offset = len(meta_cols)

        # État par boot : réinitialisé à chaque changement de boot_id.
        current_boot: str | None = None
        prev_ts: int | None = None
        prev_vals: dict[str, float | None] = {}
        windows: dict[str, deque[float | None]] = {}

        def emit_sample(index: int, values: tuple, labels: list[dict]) -> None:
            nonlocal current_boot, prev_ts, prev_vals, windows
            boot = values[idx_boot]
            ts = values[idx_ts]
            if boot != current_boot:
                current_boot = boot
                prev_ts = None
                prev_vals = {}
                windows = {col: deque(maxlen=rolling) for col in selected_metrics} if rolling else {}

            dt_s = (ts - prev_ts) / 1000.0 if (prev_ts is not None and ts > prev_ts) else None
            row_out: list = list(values[:metric_offset])
            throttle_val = None
            for offset, col in enumerate(selected_metrics):
                val = values[metric_offset + offset]
                if col == "gpu_throttle_mask":
                    throttle_val = val
                row_out.append(val)
                if deltas:
                    prev = prev_vals.get(col)
                    if val is None or prev is None:
                        delta, rate = None, None
                    else:
                        delta = val - prev
                        rate = (delta / dt_s) if dt_s is not None else None
                    row_out.append(delta)
                    row_out.append(rate)
                    prev_vals[col] = val
                if rolling is not None:
                    win = windows[col]
                    win.append(val)
                    r_mean, r_std = _rolling_stats(win)
                    row_out.append(r_mean)
                    row_out.append(r_std)
            if has_throttle:
                decoded = decode_throttle_mask(throttle_val)
                row_out.extend(decoded[name] for name, _ in THROTTLE_FLAGS)
            if split_ratios is not None:
                frac = index / total_for_split if total_for_split else 0.0
                if frac < norm_train:
                    row_out.append("train")
                elif frac < norm_train_val:
                    row_out.append("val")
                else:
                    row_out.append("test")
            prev_ts = ts

            if writer is not None:
                writer.writerow([*row_out, json.dumps(labels, ensure_ascii=False, separators=(",", ":"))])
            else:
                record = dict(zip(out_cols, row_out))
                record[label_col] = labels
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        count = 0
        previous_pk = None
        sample_values: tuple = ()
        sample_labels: list[dict] = []
        width = 1 + len(query_cols)
        for row in cursor:
            if row[0] != previous_pk:
                if previous_pk is not None:
                    emit_sample(count, sample_values, sample_labels)
                    count += 1
                previous_pk = row[0]
                sample_values = row[1:width]
                sample_labels = []
            label_id, start, end, label, notes = row[width:]
            if label_id is not None:
                sample_labels.append({
                    "id": label_id, "label": label, "notes": notes,
                    "ts_start": start, "ts_end": end,
                })
        if previous_pk is not None:
            emit_sample(count, sample_values, sample_labels)
            count += 1
        return count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exporter un dataset AAD enrichi de features temporelles"
    )
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--output", default="-", help="nouveau fichier de sortie, ou - pour stdout")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv", help="format (défaut : csv)")
    parser.add_argument("--boot", dest="boot_id", help="limiter à un boot_id")
    parser.add_argument("--latest-boot", action="store_true", help="limiter au boot le plus récent")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--last-minutes", type=float, help="limiter aux N dernières minutes")
    parser.add_argument("--columns", nargs="+", help="métriques numériques à inclure")
    parser.add_argument("--exclude-columns", nargs="+", help="métriques numériques à exclure")
    parser.add_argument("--rolling", type=int, help="taille de fenêtre glissante (moyenne + écart-type)")
    parser.add_argument("--deltas", action="store_true", help="ajouter delta et vitesse par seconde")
    parser.add_argument("--decode-throttle", action="store_true",
                        help="décomposer gpu_throttle_mask en flags binaires NVML")
    parser.add_argument("--split-ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                        help="ajouter une colonne chronologique split (train/val/test) sans fuite")
    args = parser.parse_args(argv)
    try:
        boot_id = resolve_boot_id(args.db, boot_id=args.boot_id, latest_boot=args.latest_boot)
        since, until = resolve_last_minutes(args.last_minutes, args.since, args.until)
        with _output(args.output) as target:
            count = export_features(
                args.db, target, fmt=args.format, boot_id=boot_id,
                since=since, until=until, columns=args.columns,
                exclude_columns=args.exclude_columns,
                rolling=args.rolling, deltas=args.deltas,
                decode_throttle=args.decode_throttle,
                split_ratios=tuple(args.split_ratios) if args.split_ratios else None,
            )
            target.flush()
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"export de features impossible: {exc}")
    print(f"{count} mesure(s) enrichie(s) exportée(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
