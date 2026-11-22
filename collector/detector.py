"""Détecteur statistique d'anomalies de référence (zéro dépendance, explicable).

Apprend une distribution de référence (médiane/MAD, repli sur moyenne/écart-type)
sur une fenêtre ou un label de base (ex. `normal`), puis évalue une période cible
en streaming depuis un snapshot SQLite lecture seule. Regroupe les dépassements
consécutifs en épisodes et identifie les capteurs contributeurs principaux.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import json
import math
import pathlib
import sqlite3

from .analyze import _MAD_SCALE, _numeric_metrics, _quantile
from .database import read_snapshot, resolve_boot_id, resolve_last_minutes, sample_filter_sql, validate_time_range
from .export import _output
from .storage import quote_identifier


@dataclass(frozen=True)
class FeatureBaseline:
    column: str
    center: float
    scale: float
    method: str  # "mad" | "std"
    present: int
    intercept: float | None = None
    slope: float = 0.0


def _serialize_feature(f: FeatureBaseline) -> dict:
    return {
        "column": f.column,
        "center": f.center,
        "scale": f.scale,
        "method": f.method,
        "present": f.present,
        "intercept": f.intercept,
        "slope": f.slope,
    }


def _deserialize_feature(item: object) -> FeatureBaseline:
    if not isinstance(item, dict):
        raise ValueError("entrée de métrique invalide dans le modèle")
    col = item.get("column")
    center = item.get("center")
    scale = item.get("scale")
    method = item.get("method")
    present = item.get("present", 0)
    intercept = item.get("intercept")
    slope = item.get("slope", 0.0)
    if (
        not isinstance(col, str)
        or not isinstance(center, (int, float)) or not math.isfinite(center)
        or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0
        or method not in ("mad", "std")
        or not isinstance(present, int) or present < 0
        or (intercept is not None and (not isinstance(intercept, (int, float)) or not math.isfinite(intercept)))
        or not isinstance(slope, (int, float)) or not math.isfinite(slope)
    ):
        raise ValueError(f"paramètres invalides pour la métrique {col!r}")
    return FeatureBaseline(
        col, float(center), float(scale), method, present,
        None if intercept is None else float(intercept), float(slope),
    )


def save_baseline_model(
    path: str | pathlib.Path,
    baseline_meta: dict,
    features: Sequence[FeatureBaseline],
    grouped_features: dict[str, list[FeatureBaseline]] | None = None,
) -> pathlib.Path:
    """Enregistre un modèle de référence en JSON (refuse d'écraser un fichier existant)."""
    target = pathlib.Path(path)
    payload: dict = {
        "version": 1,
        "baseline": baseline_meta,
        "features": [_serialize_feature(f) for f in features],
    }
    if grouped_features:
        payload["grouped_features"] = {
            k: [_serialize_feature(f) for f in v] for k, v in grouped_features.items()
        }
    stream = target.open("x", encoding="utf-8")
    try:
        with stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def load_baseline_model(
    path: str | pathlib.Path,
) -> tuple[dict, list[FeatureBaseline], dict[str, list[FeatureBaseline]]]:
    """Charge et valide un modèle de référence JSON."""
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("modèle de référence invalide ou version non supportée")
    items = raw.get("features")
    if not isinstance(items, list) or not items:
        raise ValueError("le modèle ne contient aucune métrique")
    features = [_deserialize_feature(item) for item in items]
    grouped: dict[str, list[FeatureBaseline]] = {}
    raw_grouped = raw.get("grouped_features")
    if isinstance(raw_grouped, dict):
        for k, v in raw_grouped.items():
            if not isinstance(k, str) or not isinstance(v, list):
                raise ValueError("grouped_features invalide dans le modèle")
            grouped[k] = [_deserialize_feature(item) for item in v]
    meta = raw.get("baseline") if isinstance(raw.get("baseline"), dict) else {}
    return meta, features, grouped


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


def _robust_center_scale(values: list[float]) -> tuple[float, float, str] | None:
    n = len(values)
    if n < 2:
        return None
    ordered = sorted(values)
    median = _quantile(ordered, 0.50)
    abs_devs = sorted(abs(v - median) for v in ordered)
    mad = _quantile(abs_devs, 0.50)
    if mad is not None and mad > 0.0:
        return median, mad / _MAD_SCALE, "mad"
    mean = sum(ordered) / n
    var = sum((v - mean) ** 2 for v in ordered) / (n - 1)
    std = math.sqrt(max(0.0, var))
    if std > 0.0:
        return mean, std, "std"
    return None


def _fit_baseline(
    conn: sqlite3.Connection,
    columns: Sequence[str],
    where: str,
    params: Sequence[str | int],
    control_for: str | None = None,
) -> tuple[int, list[FeatureBaseline], list[str]]:
    total = conn.execute("SELECT COUNT(*) FROM samples AS s" + where, params).fetchone()[0]
    if total < 2:
        raise ValueError("la référence doit contenir au moins 2 mesures")
    prefix = (where + " AND ") if where else " WHERE "
    fitted: list[FeatureBaseline] = []
    skipped: list[str] = []
    qx = quote_identifier(control_for) if control_for is not None else None
    for col in columns:
        if col == control_for:
            continue
        qc = quote_identifier(col)
        if qx is None:
            values = [
                row[0]
                for row in conn.execute(
                    f"SELECT {qc} FROM samples AS s{prefix}{qc} IS NOT NULL",
                    params,
                )
            ]
            stats = _robust_center_scale(values)
            if stats is None:
                skipped.append(col)
            else:
                center, scale, method = stats
                fitted.append(FeatureBaseline(col, center, scale, method, len(values)))
        else:
            pairs = conn.execute(
                f"SELECT {qx}, {qc} FROM samples AS s{prefix}{qx} IS NOT NULL AND {qc} IS NOT NULL",
                params,
            ).fetchall()
            n = len(pairs)
            if n < 3:
                skipped.append(col)
                continue
            mx = sum(x for x, _ in pairs) / n
            my = sum(y for _, y in pairs) / n
            sxx = sum((x - mx) ** 2 for x, _ in pairs)
            if sxx <= 0.0:
                skipped.append(col)
                continue
            slope = sum((x - mx) * (y - my) for x, y in pairs) / sxx
            intercept = my - slope * mx
            residuals = [y - (intercept + slope * x) for x, y in pairs]
            stats = _robust_center_scale(residuals)
            if stats is None:
                skipped.append(col)
            else:
                center, scale, method = stats
                fitted.append(FeatureBaseline(col, center, scale, method, n, intercept, slope))
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


def _roc_pr_auc(scored: Sequence[tuple[float, bool]]) -> tuple[float | None, float | None]:
    """Calcule ROC-AUC (avec rangs moyens sur ex-aequo) et PR-AUC (Average Precision)."""
    n_pos = sum(1 for _, pos in scored if pos)
    n_neg = len(scored) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None

    # ROC-AUC via statistique de Mann-Whitney U (gère exactement les ex-aequo)
    asc = sorted(scored, key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(asc)
    while i < n:
        j = i + 1
        while j < n and asc[j][0] == asc[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_tie = sum(1 for k in range(i, j) if asc[k][1])
        rank_sum_pos += pos_in_tie * avg_rank
        i = j
    roc_auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # PR-AUC (Average Precision) en triant par score décroissant (groupé par ex-aequo)
    desc = sorted(scored, key=lambda item: -item[0])
    tp_cum = 0
    ap_sum = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and desc[j][0] == desc[i][0]:
            j += 1
        pos_in_tie = sum(1 for k in range(i, j) if desc[k][1])
        if pos_in_tie:
            tp_cum += pos_in_tie
            precision_at_j = tp_cum / j
            ap_sum += precision_at_j * pos_in_tie
        i = j
    pr_auc = ap_sum / n_pos
    return max(0.0, min(1.0, roc_auc)), max(0.0, min(1.0, pr_auc))


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
    exclude_columns: Sequence[str] | None = None,
    control_for: str | None = None,
    group_by: str | None = None,
    score_mode: str = "max",
    smooth: int = 1,
    threshold: float = 4.0,
    min_samples: int = 3,
    top_contributors: int = 3,
    event_margin_ms: int = 5000,
    load_model: str | pathlib.Path | None = None,
    save_model: str | pathlib.Path | None = None,
    export_scores: str | pathlib.Path | None = None,
) -> dict:
    """Ajuste ou charge une référence robuste et détecte les épisodes anormaux."""
    if columns is not None and exclude_columns is not None:
        raise ValueError("--columns et --exclude-columns sont mutuellement exclusifs")
    if score_mode not in ("max", "rms"):
        raise ValueError("--score-mode doit valoir 'max' ou 'rms'")
    if smooth < 1:
        raise ValueError("--smooth doit être >= 1")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("--threshold doit être un nombre fini > 0")
    if min_samples < 1:
        raise ValueError("--min-samples doit être >= 1")
    if top_contributors < 1:
        raise ValueError("top_contributors doit être >= 1")
    if event_margin_ms < 0:
        raise ValueError("--event-margin-ms doit être >= 0")
    if load_model is not None and save_model is not None:
        raise ValueError("--load-model et --save-model sont mutuellement exclusifs")
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
        all_cols = {f[1] for f in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if columns is not None:
            selected = list(columns)
            if not selected:
                raise ValueError("aucune colonne sélectionnée")
            unknown = set(selected) - set(numeric_names)
            if unknown:
                raise ValueError(f"métriques numériques introuvables: {', '.join(sorted(unknown))}")
        else:
            excluded = set(exclude_columns or ())
            selected = [c for c in numeric_names if c not in excluded]
            if not selected and excluded:
                raise ValueError("toutes les colonnes numériques ont été exclues")

        grouped_fitted: dict[str, dict[str, FeatureBaseline]] = {}
        if load_model is not None:
            loaded_meta, loaded_features, loaded_grouped = load_baseline_model(load_model)
            control_for = loaded_meta.get("control_for")
            group_by = loaded_meta.get("group_by")
            if control_for is not None and control_for not in numeric_names:
                raise ValueError(f"colonne de contrôle du modèle absente: {control_for!r}")
            if group_by is not None and group_by not in all_cols:
                raise ValueError(f"colonne de régime du modèle absente: {group_by!r}")
            selected_set = set(selected)
            fitted = [fb for fb in loaded_features if fb.column in selected_set]
            skipped = [fb.column for fb in loaded_features if fb.column not in selected_set]
            if not fitted:
                raise ValueError("aucune métrique du modèle n'est présente dans la base cible")
            for g_key, g_feats in loaded_grouped.items():
                grouped_fitted[g_key] = {fb.column: fb for fb in g_feats if fb.column in selected_set}
            baseline_meta = {
                **loaded_meta,
                "loaded_from": str(load_model),
                "features_used": len(fitted),
                "features_skipped": skipped,
            }
        else:
            if control_for is not None and control_for not in numeric_names:
                raise ValueError(f"colonne de contrôle introuvable: {control_for!r}")
            if group_by is not None and group_by not in all_cols:
                raise ValueError(f"colonne de régime introuvable: {group_by!r}")
            fit_cols = [c for c in selected if c != group_by]
            baseline_count, fitted, skipped = _fit_baseline(
                conn, fit_cols, base_where, base_params, control_for=control_for,
            )
            grouped_lists: dict[str, list[FeatureBaseline]] = {}
            if group_by is not None:
                qg = quote_identifier(group_by)
                g_prefix = (base_where + " AND ") if base_where else " WHERE "
                g_vals = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT s.{qg} FROM samples AS s{g_prefix}s.{qg} IS NOT NULL",
                        base_params,
                    )
                ]
                for g_val in g_vals:
                    gw = f"{g_prefix}s.{qg} = ?"
                    gp = [*base_params, g_val]
                    try:
                        _, gf, _ = _fit_baseline(conn, fit_cols, gw, gp, control_for=control_for)
                    except ValueError:
                        continue
                    grouped_lists[str(g_val)] = gf
                    grouped_fitted[str(g_val)] = {fb.column: fb for fb in gf}
            baseline_meta = {
                "label": baseline_label,
                "boot_id": baseline_boot,
                "window": {"since": baseline_since, "until": baseline_until},
                "control_for": control_for,
                "group_by": group_by,
                "groups_fitted": sorted(grouped_fitted),
                "samples": baseline_count,
                "features_used": len(fitted),
                "features_skipped": skipped,
            }
            if save_model is not None:
                save_baseline_model(save_model, baseline_meta, fitted, grouped_lists or None)

        projection = ", ".join(f"s.{quote_identifier(fb.column)}" for fb in fitted)
        control_select = f", s.{quote_identifier(control_for)}" if control_for is not None else ""
        group_select = f", s.{quote_identifier(group_by)}" if group_by is not None else ""
        eval_select = ""
        query_params: list[str | int] = []
        if eval_label is not None:
            clause, e_params = _label_exists_clause(eval_label)
            eval_select = f", ({clause}) AS is_eval_positive"
            query_params.extend(e_params)
        query_params.extend(target_params)

        cursor = conn.execute(
            f"SELECT s.boot_id, s.ts, {projection}{control_select}{group_select}{eval_select}"
            f" FROM samples AS s{target_where}"
            f" ORDER BY s.boot_id, s.{quote_identifier(order_col)}",
            query_params,
        )

        episodes: list[dict] = []
        total_scored = 0
        anomalous_samples = 0
        # Pour l'évaluation : on attend la fin de chaque run pour savoir s'il atteint min_samples.
        tp = fp = fn = tn = 0
        eval_scores: list[tuple[float, bool]] = []

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
        has_control = control_for is not None
        has_group = group_by is not None

        scores_path = pathlib.Path(export_scores) if export_scores is not None else None
        scores_stream = scores_path.open("x", encoding="utf-8", newline="") if scores_path is not None else None
        scores_writer = csv.writer(scores_stream, lineterminator="\n") if scores_stream is not None else None
        if scores_writer is not None:
            scores_writer.writerow(["ts", "boot_id", "score", "max_abs_z", "rms_z", "top_sensor", "exceeds_threshold"])
        smooth_window: deque[float] = deque(maxlen=smooth)
        last_boot: str | None = None
        try:
            for row in cursor:
                boot = row[0]
                ts = row[1]
                values = row[2 : 2 + n_feat]
                idx = 2 + n_feat
                ctrl_val = row[idx] if has_control else None
                if has_control:
                    idx += 1
                grp_val = row[idx] if has_group else None
                if has_group:
                    idx += 1
                is_pos = bool(row[idx]) if eval_label is not None else False
                total_scored += 1

                if boot != last_boot:
                    smooth_window.clear()
                    last_boot = boot
                if run_len > 0 and boot != run_boot:
                    flush_run()

                active_group = grouped_fitted.get(str(grp_val)) if (has_group and grp_val is not None) else None
                max_abs = 0.0
                top_sensor = ""
                sq_sum = 0.0
                n_obs = 0
                z_obs: list[tuple[FeatureBaseline, float, float, float]] = []
                for base_fb, val in zip(fitted, values):
                    if val is None:
                        continue
                    fb = active_group.get(base_fb.column, base_fb) if active_group is not None else base_fb
                    if fb.intercept is not None:
                        if ctrl_val is None:
                            continue
                        expected = fb.intercept + fb.slope * ctrl_val + fb.center
                    else:
                        expected = fb.center
                    z = (val - expected) / fb.scale
                    az = abs(z)
                    if az > max_abs:
                        max_abs = az
                        top_sensor = fb.column
                    sq_sum += z * z
                    n_obs += 1
                    z_obs.append((fb, z, val, expected))

                if n_obs > 0:
                    rms_z = math.sqrt(sq_sum / n_obs)
                    raw_score = rms_z if score_mode == "rms" else max_abs
                    smooth_window.append(raw_score)
                    sample_score = sum(smooth_window) / len(smooth_window)
                    if eval_label is not None:
                        eval_scores.append((sample_score, is_pos))
                else:
                    rms_z = 0.0
                    sample_score = 0.0

                exceeds = n_obs > 0 and sample_score >= threshold
                if scores_writer is not None:
                    scores_writer.writerow([
                        ts,
                        boot,
                        f"{sample_score:.6f}" if n_obs > 0 else "",
                        f"{max_abs:.6f}" if n_obs > 0 else "",
                        f"{rms_z:.6f}" if n_obs > 0 else "",
                        top_sensor,
                        int(exceeds),
                    ])

                if exceeds:
                    if run_len == 0:
                        run_boot = boot
                        run_ts_start = ts
                    run_ts_end = ts
                    run_len += 1
                    if sample_score > run_peak:
                        run_peak = sample_score
                    run_rms_sum += rms_z
                    if is_pos:
                        run_eval_pos += 1
                    for fb, z, val, expected in z_obs:
                        az = abs(z)
                        prev = run_contrib.get(fb.column)
                        if prev is None or az > prev[0]:
                            run_contrib[fb.column] = (az, z, val, expected)
                else:
                    flush_run()
                    if eval_label is not None:
                        if is_pos:
                            fn += 1
                        else:
                            tn += 1
        except BaseException:
            if scores_stream is not None:
                scores_stream.close()
            if scores_path is not None:
                scores_path.unlink(missing_ok=True)
            raise
        if scores_stream is not None:
            scores_stream.close()

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

        episode_eval = None
        if eval_label is not None:
            gt_rows = conn.execute(
                "SELECT id, ts_start, ts_end FROM labels WHERE label = ?"
                " AND (ts_end IS NULL OR ts_start < ts_end) ORDER BY ts_start, id",
                (eval_label,),
            ).fetchall()
            matched_gt: set[int] = set()
            matched_ep = 0
            for ep in episodes:
                hit = False
                for gt_id, gt_start, gt_end in gt_rows:
                    if gt_start <= ep["ts_end"] and (gt_end is None or ep["ts_start"] < gt_end):
                        matched_gt.add(gt_id)
                        hit = True
                if hit:
                    matched_ep += 1
            ep_prec = matched_ep / len(episodes) if episodes else None
            ep_rec = len(matched_gt) / len(gt_rows) if gt_rows else None
            ep_f1 = (
                2 * ep_prec * ep_rec / (ep_prec + ep_rec)
                if ep_prec is not None and ep_rec is not None and (ep_prec + ep_rec) > 0
                else None
            )
            episode_eval = {
                "gt_windows": len(gt_rows),
                "detected_episodes": len(episodes),
                "matched_gt_windows": len(matched_gt),
                "matched_episodes": matched_ep,
                "episode_precision": ep_prec,
                "episode_recall": ep_rec,
                "episode_f1": ep_f1,
            }

    result = {
        "baseline": baseline_meta,
        "target": {
            "label": target_label,
            "boot_id": boot_id,
            "window": {"since": since, "until": until},
            "samples": total_scored,
            "anomalous_samples": anomalous_samples,
            "anomalous_pct": (anomalous_samples / total_scored * 100.0) if total_scored else None,
        },
        "score_mode": score_mode,
        "smooth": smooth,
        "threshold": threshold,
        "min_samples": min_samples,
        "episodes": episodes,
    }
    if eval_label is not None:
        roc_auc, pr_auc = _roc_pr_auc(eval_scores)
        result["evaluation"] = {
            "label": eval_label,
            **_f1_metrics(tp, fp, fn, tn),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            **(episode_eval or {}),
        }
    return result


_DEFAULT_TUNE_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)


def _evaluate_scores_at_threshold(
    scored: Sequence[tuple[str, float | None, bool]],
    threshold: float,
    min_samples: int,
) -> dict:
    tp = fp = fn = tn = 0
    episodes = 0
    run_boot: str | None = None
    run_len = 0
    run_pos = 0

    def flush() -> None:
        nonlocal tp, fp, fn, tn, episodes, run_boot, run_len, run_pos
        if run_len == 0:
            return
        if run_len >= min_samples:
            episodes += 1
            tp += run_pos
            fp += run_len - run_pos
        else:
            fn += run_pos
            tn += run_len - run_pos
        run_boot = None
        run_len = 0
        run_pos = 0

    for boot, max_abs, is_pos in scored:
        if run_len > 0 and boot != run_boot:
            flush()
        if max_abs is not None and max_abs >= threshold:
            run_boot = boot
            run_len += 1
            if is_pos:
                run_pos += 1
        else:
            flush()
            if is_pos:
                fn += 1
            else:
                tn += 1
    flush()
    return {"threshold": threshold, "episodes": episodes, **_f1_metrics(tp, fp, fn, tn)}


def tune_threshold(
    db: str | pathlib.Path,
    *,
    eval_label: str,
    baseline_label: str | None = None,
    baseline_boot: str | None = None,
    baseline_since: int | None = None,
    baseline_until: int | None = None,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    target_label: str | None = None,
    columns: Sequence[str] | None = None,
    control_for: str | None = None,
    group_by: str | None = None,
    score_mode: str = "max",
    smooth: int = 1,
    thresholds: Sequence[float] | None = None,
    min_samples: int = 3,
    load_model: str | pathlib.Path | None = None,
    export_curve: str | pathlib.Path | None = None,
) -> dict:
    """Évalue une grille de seuils z en une passe et sélectionne le meilleur F1."""
    if score_mode not in ("max", "rms"):
        raise ValueError("--score-mode doit valoir 'max' ou 'rms'")
    if smooth < 1:
        raise ValueError("--smooth doit être >= 1")
    eval_label = eval_label.strip()
    if not eval_label:
        raise ValueError("--eval-label ne peut pas être vide")
    if min_samples < 1:
        raise ValueError("--min-samples doit être >= 1")
    grid = list(_DEFAULT_TUNE_THRESHOLDS if thresholds is None else thresholds)
    if not grid or any(not math.isfinite(t) or t <= 0 for t in grid):
        raise ValueError("--thresholds doit contenir des nombres finis > 0")
    validate_time_range(baseline_since, baseline_until)
    validate_time_range(since, until)

    base_where, base_params = _scoped_where(
        boot_id=baseline_boot, since=baseline_since, until=baseline_until, label=baseline_label,
    )
    target_where, target_params = _scoped_where(
        boot_id=boot_id, since=since, until=until, label=target_label,
    )

    with read_snapshot(db) as conn:
        metrics, order_col = _numeric_metrics(conn)
        numeric_names = [name for name, _ in metrics]
        all_cols = {f[1] for f in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if columns is not None:
            selected = list(columns)
            if not selected or (set(selected) - set(numeric_names)):
                raise ValueError("colonnes sélectionnées invalides")
        else:
            selected = numeric_names

        grouped_fitted: dict[str, dict[str, FeatureBaseline]] = {}
        if load_model is not None:
            loaded_meta, loaded_features, loaded_grouped = load_baseline_model(load_model)
            control_for = loaded_meta.get("control_for")
            group_by = loaded_meta.get("group_by")
            if control_for is not None and control_for not in numeric_names:
                raise ValueError(f"colonne de contrôle du modèle absente: {control_for!r}")
            if group_by is not None and group_by not in all_cols:
                raise ValueError(f"colonne de régime du modèle absente: {group_by!r}")
            selected_set = set(selected)
            fitted = [fb for fb in loaded_features if fb.column in selected_set]
            if not fitted:
                raise ValueError("aucune métrique du modèle n'est présente dans la base cible")
            for g_key, g_feats in loaded_grouped.items():
                grouped_fitted[g_key] = {fb.column: fb for fb in g_feats if fb.column in selected_set}
        else:
            if control_for is not None and control_for not in numeric_names:
                raise ValueError(f"colonne de contrôle introuvable: {control_for!r}")
            if group_by is not None and group_by not in all_cols:
                raise ValueError(f"colonne de régime introuvable: {group_by!r}")
            fit_cols = [c for c in selected if c != group_by]
            _, fitted, _ = _fit_baseline(conn, fit_cols, base_where, base_params, control_for=control_for)
            if group_by is not None:
                qg = quote_identifier(group_by)
                g_prefix = (base_where + " AND ") if base_where else " WHERE "
                g_vals = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT s.{qg} FROM samples AS s{g_prefix}s.{qg} IS NOT NULL",
                        base_params,
                    )
                ]
                for g_val in g_vals:
                    try:
                        _, gf, _ = _fit_baseline(
                            conn, fit_cols, f"{g_prefix}s.{qg} = ?", [*base_params, g_val], control_for=control_for,
                        )
                    except ValueError:
                        continue
                    grouped_fitted[str(g_val)] = {fb.column: fb for fb in gf}

        projection = ", ".join(f"s.{quote_identifier(fb.column)}" for fb in fitted)
        control_select = f", s.{quote_identifier(control_for)}" if control_for is not None else ""
        group_select = f", s.{quote_identifier(group_by)}" if group_by is not None else ""
        clause, e_params = _label_exists_clause(eval_label)
        cursor = conn.execute(
            f"SELECT s.boot_id, {projection}{control_select}{group_select}, ({clause})"
            f" FROM samples AS s{target_where}"
            f" ORDER BY s.boot_id, s.{quote_identifier(order_col)}",
            [*e_params, *target_params],
        )
        n_feat = len(fitted)
        has_control = control_for is not None
        has_group = group_by is not None
        scored: list[tuple[str, float | None, bool]] = []
        smooth_window: deque[float] = deque(maxlen=smooth)
        last_boot: str | None = None
        for row in cursor:
            boot = row[0]
            if boot != last_boot:
                smooth_window.clear()
                last_boot = boot
            values = row[1 : 1 + n_feat]
            idx = 1 + n_feat
            ctrl_val = row[idx] if has_control else None
            if has_control:
                idx += 1
            grp_val = row[idx] if has_group else None
            if has_group:
                idx += 1
            is_pos = bool(row[idx])
            active_group = grouped_fitted.get(str(grp_val)) if (has_group and grp_val is not None) else None
            max_abs = 0.0
            sq_sum = 0.0
            n_obs = 0
            for base_fb, val in zip(fitted, values):
                if val is None:
                    continue
                fb = active_group.get(base_fb.column, base_fb) if active_group is not None else base_fb
                if fb.intercept is not None:
                    if ctrl_val is None:
                        continue
                    expected = fb.intercept + fb.slope * ctrl_val + fb.center
                else:
                    expected = fb.center
                z = (val - expected) / fb.scale
                az = abs(z)
                if az > max_abs:
                    max_abs = az
                sq_sum += z * z
                n_obs += 1
            if n_obs > 0:
                raw_score = math.sqrt(sq_sum / n_obs) if score_mode == "rms" else max_abs
                smooth_window.append(raw_score)
                sample_score: float | None = sum(smooth_window) / len(smooth_window)
            else:
                sample_score = None
            scored.append((boot, sample_score, is_pos))

    results = [_evaluate_scores_at_threshold(scored, t, min_samples) for t in sorted(set(grid))]
    best = max(
        results,
        key=lambda r: (
            r["f1"] if r["f1"] is not None else -1.0,
            r["precision"] if r["precision"] is not None else -1.0,
            r["threshold"],
        ),
    )
    roc_auc, pr_auc = _roc_pr_auc([(s, p) for _, s, p in scored if s is not None])
    if export_curve is not None:
        curve_path = pathlib.Path(export_curve)
        stream = curve_path.open("x", encoding="utf-8", newline="")
        try:
            with stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["threshold", "episodes", "tp", "fp", "fn", "tn", "tpr", "fpr", "precision", "recall", "f1"])
                for r in results:
                    tpr = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) else None
                    fpr = r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) else None
                    fmt_opt = lambda v: "" if v is None else f"{v:.6f}"
                    writer.writerow([
                        r["threshold"], r["episodes"], r["tp"], r["fp"], r["fn"], r["tn"],
                        fmt_opt(tpr), fmt_opt(fpr), fmt_opt(r["precision"]), fmt_opt(r["recall"]), fmt_opt(r["f1"]),
                    ])
        except BaseException:
            curve_path.unlink(missing_ok=True)
            raise
    return {
        "eval_label": eval_label,
        "samples": len(scored),
        "min_samples": min_samples,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best": best,
        "grid": results,
    }


def benchmark_configs(
    db: str | pathlib.Path,
    *,
    eval_label: str,
    baseline_label: str | None = None,
    baseline_boot: str | None = None,
    baseline_since: int | None = None,
    baseline_until: int | None = None,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    target_label: str | None = None,
    columns: Sequence[str] | None = None,
    control_for: str | None = None,
    group_by: str | None = None,
    smooth: int = 1,
    thresholds: Sequence[float] | None = None,
    min_samples: int = 3,
) -> dict:
    """Compare plusieurs stratégies de détection sur un même dataset labellisé."""
    variants: list[tuple[str, str | None, str | None]] = [("raw", None, None)]
    if control_for is not None:
        variants.append((f"control({control_for})", control_for, None))
    if group_by is not None:
        variants.append((f"group({group_by})", None, group_by))
    if control_for is not None and group_by is not None:
        variants.append((f"control({control_for})+group({group_by})", control_for, group_by))

    runs: list[dict] = []
    for name, ctrl, grp in variants:
        for mode in ("max", "rms"):
            tuned = tune_threshold(
                db,
                eval_label=eval_label,
                baseline_label=baseline_label,
                baseline_boot=baseline_boot,
                baseline_since=baseline_since,
                baseline_until=baseline_until,
                boot_id=boot_id,
                since=since,
                until=until,
                target_label=target_label,
                columns=columns,
                control_for=ctrl,
                group_by=grp,
                score_mode=mode,
                smooth=smooth,
                thresholds=thresholds,
                min_samples=min_samples,
            )
            runs.append({
                "name": f"{name}/{mode}",
                "control_for": ctrl,
                "group_by": grp,
                "score_mode": mode,
                "roc_auc": tuned["roc_auc"],
                "pr_auc": tuned["pr_auc"],
                "best_threshold": tuned["best"]["threshold"],
                "best_f1": tuned["best"]["f1"],
                "best_precision": tuned["best"]["precision"],
                "best_recall": tuned["best"]["recall"],
            })

    runs.sort(
        key=lambda r: (
            -(r["pr_auc"] if r["pr_auc"] is not None else -1.0),
            -(r["roc_auc"] if r["roc_auc"] is not None else -1.0),
            -(r["best_f1"] if r["best_f1"] is not None else -1.0),
            r["name"],
        )
    )
    return {
        "eval_label": eval_label,
        "min_samples": min_samples,
        "best": runs[0] if runs else None,
        "runs": runs,
    }


def check_model_compatibility(
    db: str | pathlib.Path,
    model_path: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> dict:
    """Vérifie la compatibilité d'un modèle JSON sauvegardé avec une base cible."""
    validate_time_range(since, until)
    meta, features, grouped = load_baseline_model(model_path)
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        numeric_set = {name for name, _ in metrics}
        all_cols = {f[1] for f in conn.execute("PRAGMA table_info(samples)").fetchall()}

        model_cols = [fb.column for fb in features]
        matched = [c for c in model_cols if c in numeric_set]
        missing = [c for c in model_cols if c not in numeric_set]

        control_for = meta.get("control_for")
        control_ok = control_for is None or control_for in numeric_set

        group_by = meta.get("group_by")
        group_ok = group_by is None or group_by in all_cols
        seen_groups: list[str] = []
        unseen_groups: list[str] = []
        if group_by is not None and group_ok:
            qg = quote_identifier(group_by)
            prefix = (where + " AND ") if where else " WHERE "
            observed = {
                str(r[0])
                for r in conn.execute(
                    f"SELECT DISTINCT s.{qg} FROM samples AS s{prefix}s.{qg} IS NOT NULL",
                    params,
                )
            }
            trained = set(grouped)
            seen_groups = sorted(observed & trained)
            unseen_groups = sorted(observed - trained)

    compatible = bool(matched) and control_ok and group_ok
    return {
        "model": str(model_path),
        "compatible": compatible,
        "matched_features": matched,
        "missing_features": missing,
        "control_for": control_for,
        "control_for_present": control_ok,
        "group_by": group_by,
        "group_by_present": group_ok,
        "seen_groups": seen_groups,
        "unseen_groups": unseen_groups,
    }


def inspect_model(model_path: str | pathlib.Path) -> dict:
    """Retourne les paramètres lisibles d'un modèle JSON sauvegardé."""
    meta, features, grouped = load_baseline_model(model_path)
    return {
        "model": str(model_path),
        "baseline": meta,
        "features": [_serialize_feature(f) for f in features],
        "grouped_features": {
            k: [_serialize_feature(f) for f in v] for k, v in grouped.items()
        },
    }


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
    parser.add_argument("--latest-boot", action="store_true", help="limiter la cible au boot le plus récent")
    parser.add_argument("--since", type=int, help="début cible (epoch ms inclusif)")
    parser.add_argument("--until", type=int, help="fin cible (epoch ms exclusif)")
    parser.add_argument("--last-minutes", type=float, help="limiter la cible aux N dernières minutes")
    parser.add_argument("--target-label", help="limiter la cible à un label")
    parser.add_argument("--eval-label", help="évaluer précision/rappel/F1 face à un label terrain")
    parser.add_argument("--columns", nargs="+", help="sous-ensemble de métriques numériques")
    parser.add_argument("--exclude-columns", nargs="+", help="métriques numériques à exclure du scoring")
    parser.add_argument("--control-for", help="métrique de charge/contexte utilisée pour le scoring par résidus")
    parser.add_argument("--group-by", help="colonne de régime discret (ex. is_gaming) pour ajuster un profil par régime")
    parser.add_argument("--score-mode", choices=("max", "rms"), default="max",
                        help="agrégation du score par tick : max |z| (défaut) ou rms multivarié")
    parser.add_argument("--smooth", type=int, default=1,
                        help="lissage par moyenne mobile sur N ticks du score (défaut : 1)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="seuil de z-score absolu (défaut : 4.0)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="longueur minimale d'un épisode anormal (défaut : 3)")
    parser.add_argument("--event-margin-ms", type=int, default=5000,
                        help="marge (ms) pour associer les événements autour d'un épisode (défaut : 5000)")
    parser.add_argument("--save-model", type=pathlib.Path, help="sauvegarder le modèle ajusté en JSON")
    parser.add_argument("--load-model", type=pathlib.Path, help="charger un modèle JSON existant")
    parser.add_argument("--export-scores", type=pathlib.Path,
                        help="exporter les scores d'anomalie par échantillon en CSV")
    parser.add_argument("--export-curve", type=pathlib.Path,
                        help="exporter la courbe ROC/PR par seuil en CSV (avec --tune-threshold)")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="calibrer le seuil optimal (F1) sur --eval-label")
    parser.add_argument("--benchmark", action="store_true",
                        help="comparer plusieurs stratégies (raw/control-for/group-by × max/rms) sur --eval-label")
    parser.add_argument("--check-model", action="store_true",
                        help="vérifier la compatibilité de --load-model avec la base cible")
    parser.add_argument("--inspect-model", action="store_true",
                        help="afficher les paramètres appris de --load-model sans ouvrir de base")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        help="grille de seuils personnalisée pour --tune-threshold")
    parser.add_argument("--output", default="-",
                        help="nouveau fichier de sortie, ou - pour stdout (défaut)")
    parser.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")
    args = parser.parse_args(argv)

    has_baseline_opts = any(v is not None for v in (
        args.baseline_label, args.baseline_boot, args.baseline_since, args.baseline_until
    ))
    if args.load_model is not None and has_baseline_opts:
        parser.error("--load-model ne se combine pas avec les options --baseline-*")
    if args.load_model is None and not has_baseline_opts:
        parser.error("spécifiez une référence (--baseline-label, --baseline-boot, --baseline-since/--baseline-until ou --load-model)")
    if (args.check_model or args.inspect_model) and args.load_model is None:
        parser.error("--check-model et --inspect-model nécessitent --load-model")
    if args.tune_threshold and args.benchmark:
        parser.error("--tune-threshold et --benchmark sont mutuellement exclusifs")
    if (args.tune_threshold or args.benchmark) and not args.eval_label:
        parser.error("--tune-threshold et --benchmark nécessitent --eval-label")
    if args.benchmark and args.load_model is not None:
        parser.error("--benchmark nécessite une référence --baseline-* (pas --load-model)")

    try:
        since, until = resolve_last_minutes(args.last_minutes, args.since, args.until)
        boot_id = (
            None
            if args.inspect_model
            else resolve_boot_id(args.db, boot_id=args.boot_id, latest_boot=args.latest_boot)
        )
        if args.inspect_model:
            result = inspect_model(args.load_model)
        elif args.check_model:
            result = check_model_compatibility(
                args.db, args.load_model, boot_id=boot_id, since=since, until=until,
            )
        elif args.benchmark:
            result = benchmark_configs(
                args.db,
                eval_label=args.eval_label,
                baseline_label=args.baseline_label,
                baseline_boot=args.baseline_boot,
                baseline_since=args.baseline_since,
                baseline_until=args.baseline_until,
                boot_id=boot_id,
                since=since,
                until=until,
                target_label=args.target_label,
                columns=args.columns,
                control_for=args.control_for,
                group_by=args.group_by,
                smooth=args.smooth,
                thresholds=args.thresholds,
                min_samples=args.min_samples,
            )
        elif args.tune_threshold:
            result = tune_threshold(
                args.db,
                eval_label=args.eval_label,
                baseline_label=args.baseline_label,
                baseline_boot=args.baseline_boot,
                baseline_since=args.baseline_since,
                baseline_until=args.baseline_until,
                boot_id=boot_id,
                since=since,
                until=until,
                target_label=args.target_label,
                columns=args.columns,
                control_for=args.control_for,
                group_by=args.group_by,
                score_mode=args.score_mode,
                smooth=args.smooth,
                thresholds=args.thresholds,
                min_samples=args.min_samples,
                load_model=args.load_model,
                export_curve=args.export_curve,
            )
        else:
            result = detect_anomalies(
                args.db,
                baseline_label=args.baseline_label,
                baseline_boot=args.baseline_boot,
                baseline_since=args.baseline_since,
                baseline_until=args.baseline_until,
                boot_id=boot_id,
                since=since,
                until=until,
                target_label=args.target_label,
                eval_label=args.eval_label,
                columns=args.columns,
                exclude_columns=args.exclude_columns,
                control_for=args.control_for,
                group_by=args.group_by,
                score_mode=args.score_mode,
                smooth=args.smooth,
                threshold=args.threshold,
                min_samples=args.min_samples,
                event_margin_ms=args.event_margin_ms,
                load_model=args.load_model,
                save_model=args.save_model,
                export_scores=args.export_scores,
            )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))

    def emit_output(stream) -> None:
        if args.json:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return

        if args.inspect_model:
            base = result["baseline"]
            print(f"Modèle {result['model']} ({len(result['features'])} métriques) :", file=stream)
            if base.get("control_for"):
                print(f"  control_for : {base['control_for']}", file=stream)
            if base.get("group_by"):
                print(f"  group_by    : {base['group_by']} ({len(result['grouped_features'])} régime(s))", file=stream)
            for f in result["features"]:
                reg = f", slope={f['slope']:.3f}, intercept={f['intercept']:.2f}" if f["intercept"] is not None else ""
                print(
                    f"  {f['column']} : center={f['center']:.2f}, scale={f['scale']:.2f} "
                    f"[{f['method']}, n={f['present']}{reg}]",
                    file=stream,
                )
            return

        if args.check_model:
            verdict = "COMPATIBLE" if result["compatible"] else "INCOMPATIBLE"
            print(
                f"Modèle {result['model']} : {verdict} "
                f"({len(result['matched_features'])} présentes, {len(result['missing_features'])} manquantes)",
                file=stream,
            )
            if result["unseen_groups"]:
                print(f"  régimes non vus à l'apprentissage : {', '.join(result['unseen_groups'])}", file=stream)
            return

        if args.benchmark:
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(f"Benchmark ({result['eval_label']}, {len(result['runs'])} configurations) :", file=stream)
            for row in result["runs"]:
                print(
                    f"  {row['name']:34} pr_auc={fmt(row['pr_auc'])} roc_auc={fmt(row['roc_auc'])} "
                    f"best_f1={fmt(row['best_f1'])} (z>={row['best_threshold']})",
                    file=stream,
                )
            return

        if args.tune_threshold:
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            best = result["best"]
            print(
                f"Calibration ({result['eval_label']}, {result['samples']} mesures) — "
                f"meilleur seuil : z >= {best['threshold']} (f1={fmt(best['f1'])}, "
                f"precision={fmt(best['precision'])}, recall={fmt(best['recall'])})",
                file=stream,
            )
            for row in result["grid"]:
                print(
                    f"  z >= {row['threshold']:4.1f} : episodes={row['episodes']}, "
                    f"precision={fmt(row['precision'])}, recall={fmt(row['recall'])}, f1={fmt(row['f1'])}",
                    file=stream,
                )
            return

        base = result["baseline"]
        tgt = result["target"]
        pct = "indéfinie" if tgt["anomalous_pct"] is None else f"{tgt['anomalous_pct']:.1f} %"
        print(f"Référence : {base['samples']} mesures, {base['features_used']} métrique(s) active(s)", file=stream)
        print(f"Cible     : {tgt['samples']} mesures, {tgt['anomalous_samples']} anormale(s) ({pct})", file=stream)
        print(f"Épisodes  : {len(result['episodes'])} (seuil z >= {result['threshold']}, min {result['min_samples']} mesures)", file=stream)
        for idx, ep in enumerate(result["episodes"], 1):
            contribs = ", ".join(
                f"{c['column']} (z={c['peak_z']:+.1f}, val={c['observed_at_peak']:.2f} vs {c['baseline_center']:.2f})"
                for c in ep["top_contributors"]
            )
            print(
                f"  #{idx} boot={ep['boot_id']} [{ep['ts_start']}..{ep['ts_end']}] "
                f"n={ep['sample_count']} pic={ep['peak_score']:.1f} — {contribs}",
                file=stream,
            )
            if ep["events"]:
                kinds = ", ".join(sorted({f"{e['source']}/{e['kind']}" for e in ep["events"]}))
                print(f"       événements proches ({len(ep['events'])}) : {kinds}", file=stream)
        if "evaluation" in result:
            ev = result["evaluation"]
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(
                f"Évaluation ({ev['label']}) : precision={fmt(ev['precision'])}, "
                f"recall={fmt(ev['recall'])}, f1={fmt(ev['f1'])} "
                f"(tp={ev['tp']} fp={ev['fp']} fn={ev['fn']} tn={ev['tn']})",
                file=stream,
            )

    try:
        with _output(args.output) as stream:
            emit_output(stream)
            stream.flush()
    except OSError as exc:
        parser.error(f"écriture impossible: {exc}")


if __name__ == "__main__":
    main()
