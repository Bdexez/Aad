"""Rapport HTML autonome (zéro dépendance, sparklines SVG inline, lecture seule).

Permet d'inspecter visuellement une campagne AAD dans n'importe quel navigateur
hors-ligne : complétude, cadence, statistiques numériques avec mini-courbes SVG,
audit des labels et événements récents.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import html
import pathlib
import sqlite3

from .analyze import _numeric_metrics, analyze_dataset, audit_labels, check_physical_bounds
from .database import read_snapshot, resolve_last_minutes, sample_filter_sql, validate_time_range
from .detector import detect_anomalies
from .report import dataset_report
from .storage import quote_identifier


def _sparkline_svg(values: list[float], width: int = 160, height: int = 28) -> str:
    """Génère une mini-courbe SVG autonome (sans JS ni bibliothèque externe)."""
    if len(values) < 2:
        return '<svg width="160" height="28" aria-label="courbe indisponible"></svg>'
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = 2.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    step = usable_w / (len(values) - 1)
    pts = []
    for i, v in enumerate(values):
        x = pad + i * step
        y = height / 2.0 if span == 0 else pad + usable_h * (1.0 - (v - lo) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="1.5" points="{polyline}"/>'
        "</svg>"
    )


def _fetch_sparklines(
    db: str | pathlib.Path,
    *,
    boot_id: str | None,
    since: int | None,
    until: int | None,
    max_points: int = 120,
) -> dict[str, list[float]]:
    where, params = sample_filter_sql(boot_id=boot_id, since=since, until=until, alias="s")
    with read_snapshot(db) as conn:
        metrics, _ = _numeric_metrics(conn)
        if not metrics:
            return {}
        total, min_ts, max_ts = conn.execute(
            "SELECT COUNT(*), MIN(s.ts), MAX(s.ts) FROM samples AS s" + where, params
        ).fetchone()
        if not total or min_ts is None or max_ts is None:
            return {name: [] for name, _ in metrics}
        names = [name for name, _ in metrics]
        if total <= max_points or max_ts <= min_ts:
            proj = ", ".join(f"s.{quote_identifier(n)}" for n in names)
            rows = conn.execute(
                f"SELECT {proj} FROM samples AS s{where} ORDER BY s.ts", params
            ).fetchall()
        else:
            bucket = max(1, (max_ts - min_ts + 1) // max_points)
            aggs = ", ".join(f"AVG(s.{quote_identifier(n)})" for n in names)
            rows = conn.execute(
                f"SELECT {aggs} FROM samples AS s{where}"
                " GROUP BY (s.ts / ?) ORDER BY (s.ts / ?)",
                [*params, bucket, bucket],
            ).fetchall()
    series: dict[str, list[float]] = {n: [] for n in names}
    for row in rows:
        for name, val in zip(names, row):
            if val is not None:
                series[name].append(float(val))
    return series


def _fmt(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_html_report(
    db: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    title: str = "AAD — Rapport de campagne",
    baseline_label: str | None = None,
    load_model: str | pathlib.Path | None = None,
    threshold: float = 4.0,
    min_samples: int = 3,
) -> str:
    """Produit le document HTML complet depuis une base existante en lecture seule."""
    validate_time_range(since, until)
    rep = dataset_report(db, boot_id=boot_id, since=since, until=until)
    ana = analyze_dataset(db, boot_id=boot_id, since=since, until=until)
    aud = audit_labels(db, boot_id=boot_id, since=since, until=until)
    bounds = check_physical_bounds(db, boot_id=boot_id, since=since, until=until)
    sparks = _fetch_sparklines(db, boot_id=boot_id, since=since, until=until)
    detection = None
    if baseline_label is not None or load_model is not None:
        detection = detect_anomalies(
            db,
            baseline_label=baseline_label,
            load_model=load_model,
            boot_id=boot_id,
            since=since,
            until=until,
            threshold=threshold,
            min_samples=min_samples,
        )

    esc = html.escape
    samples = rep["samples"]
    cov = "indéfinie" if samples["labelled_pct"] is None else f"{samples['labelled_pct']:.1f} %"

    cadence_rows = "".join(
        f"<tr><td>{esc(str(b['boot_id']))}</td><td>{b['samples']}</td>"
        f"<td>{_fmt(b['median_dt_ms'], 1)} ms</td><td>{_fmt(b['p95_dt_ms'], 1)} ms</td>"
        f"<td>{b['gaps']}</td><td>{b['duplicate_timestamps']}</td><td>{b['clock_regressions']}</td></tr>"
        for b in ana["cadence"]
    )
    metric_rows = "".join(
        f"<tr><td><code>{esc(m['column'])}</code>{' <span class=\"tag\">constante</span>' if m['is_constant'] else ''}</td>"
        f"<td>{m['present']} / {samples['count']}</td>"
        f"<td>{_fmt(m['min'])}</td><td>{_fmt(m['mean'])}</td><td>{_fmt(m['std'])}</td><td>{_fmt(m['max'])}</td>"
        f"<td>{_sparkline_svg(sparks.get(m['column'], []))}</td></tr>"
        for m in ana["metrics"]
    )
    event_rows = "".join(
        f"<tr><td>{esc(str(e['source']))}</td><td>{esc(str(e['kind']))}</td>"
        f"<td>{esc(str(e['severity'] or ''))}</td><td>{e['count']}</td></tr>"
        for e in rep["events"]
    ) or '<tr><td colspan="4">Aucun événement</td></tr>'

    bounds_section = ""
    if bounds["violations"]:
        b_rows = "".join(
            f"<tr><td><code>{esc(v['column'])}</code></td><td>{esc(v['unit'])}</td>"
            f"<td>{v['below_min']}</td><td>{v['above_max']}</td>"
            f"<td>[{_fmt(v['observed_min'])}, {_fmt(v['observed_max'])}]</td></tr>"
            for v in bounds["violations"]
        )
        bounds_section = (
            f"<h2>Violations de bornes physiques ({bounds['total_violations']})</h2>"
            "<table><thead><tr><th>Colonne</th><th>Plage attendue</th><th>&lt; min</th>"
            "<th>&gt; max</th><th>Observé</th></tr></thead>"
            f"<tbody>{b_rows}</tbody></table>"
        )

    episodes_section = ""
    if detection is not None:
        def _fmt_contribs(items: list[dict]) -> str:
            return ", ".join(f"{c['column']} (z={c['peak_z']:+.1f})" for c in items)

        ep_rows = "".join(
            f"<tr><td>{esc(str(ep['boot_id']))}</td>"
            f"<td>{ep['ts_start']} → {ep['ts_end']}</td>"
            f"<td>{ep['sample_count']}</td>"
            f"<td>{ep['peak_score']:.2f}</td>"
            f"<td>{esc(_fmt_contribs(ep['top_contributors']))}</td>"
            f"<td>{len(ep.get('events', []))}</td></tr>"
            for ep in detection["episodes"]
        ) or '<tr><td colspan="6">Aucun épisode anormal détecté</td></tr>'
        episodes_section = (
            f"<h2>Épisodes anormaux détectés ({len(detection['episodes'])})</h2>"
            "<table><thead><tr><th>Boot</th><th>Fenêtre (epoch ms)</th><th>Mesures</th>"
            "<th>Pic |z|</th><th>Contributeurs</th><th>Événements</th></tr></thead>"
            f"<tbody>{ep_rows}</tbody></table>"
        )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
  h1, h2 {{ color: #f8fafc; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 1rem; }}
  .card .label {{ font-size: 0.85rem; color: #94a3b8; }}
  .card .val {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 0.55rem 0.75rem; border-bottom: 1px solid #334155; text-align: left; font-size: 0.9rem; }}
  th {{ background: #0b1220; color: #94a3b8; font-weight: 600; }}
  code {{ color: #7dd3fc; }}
  .tag {{ background: #b45309; color: #fef3c7; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>{esc(title)}</h1>
<div class="cards">
  <div class="card"><div class="label">Mesures</div><div class="val">{samples['count']}</div></div>
  <div class="card"><div class="label">Boots</div><div class="val">{len(rep['boots'])}</div></div>
  <div class="card"><div class="label">Couverture labels</div><div class="val">{esc(cov)}</div></div>
  <div class="card"><div class="label">Fenêtres labels</div><div class="val">{aud['total_windows']} ({len(aud['open_windows'])} ouv.)</div></div>
  <div class="card"><div class="label">Multi-labels</div><div class="val">{aud['cross_label_overlap_samples']}</div></div>
  <div class="card"><div class="label">Bornes physiques</div><div class="val">{bounds['total_violations']} viol.</div></div>
</div>
<h2>Cadence par boot</h2>
<table><thead><tr><th>Boot</th><th>Mesures</th><th>Médiane</th><th>p95</th><th>Trous</th><th>Doublons</th><th>Retours</th></tr></thead>
<tbody>{cadence_rows or '<tr><td colspan="7">Aucune mesure</td></tr>'}</tbody></table>
<h2>Métriques numériques</h2>
<table><thead><tr><th>Colonne</th><th>Présence</th><th>Min</th><th>Moyenne</th><th>Écart-type</th><th>Max</th><th>Tendance</th></tr></thead>
<tbody>{metric_rows or '<tr><td colspan="7">Aucune métrique numérique</td></tr>'}</tbody></table>
<h2>Événements</h2>
<table><thead><tr><th>Source</th><th>Genre</th><th>Gravité</th><th>Total</th></tr></thead>
<tbody>{event_rows}</tbody></table>
{bounds_section}
{episodes_section}
</body>
</html>
"""


def write_html_report(
    db: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    boot_id: str | None = None,
    since: int | None = None,
    until: int | None = None,
    title: str = "AAD — Rapport de campagne",
    baseline_label: str | None = None,
    load_model: str | pathlib.Path | None = None,
    threshold: float = 4.0,
    min_samples: int = 3,
) -> pathlib.Path:
    """Écrit le rapport HTML dans un nouveau fichier (refuse d'écraser l'existant)."""
    html_text = render_html_report(
        db, boot_id=boot_id, since=since, until=until, title=title,
        baseline_label=baseline_label, load_model=load_model,
        threshold=threshold, min_samples=min_samples,
    )
    dest = pathlib.Path(output)
    stream = dest.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(html_text)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Générer un rapport HTML autonome du dataset AAD")
    parser.add_argument("--db", default="data/metrics.db", help="base SQLite existante")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="nouveau fichier HTML de sortie")
    parser.add_argument("--boot", dest="boot_id", help="limiter à un boot_id")
    parser.add_argument("--since", type=int, help="epoch ms inclusif")
    parser.add_argument("--until", type=int, help="epoch ms exclusif")
    parser.add_argument("--last-minutes", type=float, help="limiter aux N dernières minutes")
    parser.add_argument("--title", default="AAD — Rapport de campagne", help="titre du rapport")
    parser.add_argument("--baseline-label", help="inclure la détection d'épisodes depuis un label de référence")
    parser.add_argument("--load-model", type=pathlib.Path, help="inclure la détection depuis un modèle JSON")
    parser.add_argument("--threshold", type=float, default=4.0, help="seuil z de détection (défaut : 4.0)")
    parser.add_argument("--min-samples", type=int, default=3, help="longueur minimale d'épisode (défaut : 3)")
    args = parser.parse_args(argv)
    try:
        since, until = resolve_last_minutes(args.last_minutes, args.since, args.until)
        dest = write_html_report(
            args.db, args.output, boot_id=args.boot_id,
            since=since, until=until, title=args.title,
            baseline_label=args.baseline_label, load_model=args.load_model,
            threshold=args.threshold, min_samples=args.min_samples,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"génération HTML impossible: {exc}")
    print(f"rapport HTML créé : {dest}")


if __name__ == "__main__":
    main()
