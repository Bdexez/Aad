"""Annotation manuelle des expériences.

Le collector produit des séries ; ce module produit la seule chose qu'aucun
capteur ne donne : la vérité terrain. Sans elle on n'a qu'un détecteur
non supervisé qu'on ne peut pas évaluer.

    python -m collector.label start airflow_blocked -n "grille avant obstruée"
    python -m collector.label list
    python -m collector.label end

`end` sans argument ferme tous les labels ouverts — le cas courant après une
expérience où l'on a oublié de noter l'id.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime as dt
import json
import pathlib
import sqlite3

from .main import SCHEMA, now_ms
from .storage import Storage

#: Indicatif, pas contraignant : un label hors liste est accepté (avec un
#: avertissement). Figer le vocabulaire trop tôt empêcherait de nommer
#: l'anomalie qu'on n'avait pas prévue — qui est justement celle qui compte.
KNOWN_LABELS = (
    "normal",
    "fan_curve_low",
    "airflow_blocked",
    "oc_unstable",
    "stress_ng",
    "gaming",
)


def human(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "en cours"
    return dt.datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Annotation des expériences (table labels)")
    ap.add_argument("--db", default="data/metrics.db")
    ap.add_argument("--schema", type=pathlib.Path, default=SCHEMA)
    sub = ap.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="ouvrir une fenêtre labellisée")
    start.add_argument("label")
    start.add_argument("-n", "--notes", default="")

    end = sub.add_parser("end", help="fermer une fenêtre (toutes si id omis)")
    end.add_argument("id", nargs="?", type=int)

    listing = sub.add_parser("list", help="lister les fenêtres ouvertes")
    listing.add_argument("--all", action="store_true", help="inclure les fenêtres terminées")
    listing.add_argument("--label", help="filtrer sur un nom de label exact")
    listing.add_argument("--json", action="store_true", help="sortie JSON pour les scripts")

    note = sub.add_parser("note", help="mettre à jour les notes d'une fenêtre existante")
    note.add_argument("id", type=int)
    note.add_argument("-n", "--notes", required=True)

    delete = sub.add_parser("delete", help="supprimer une fenêtre créée par erreur")
    delete.add_argument("id", type=int)

    args = ap.parse_args(argv)
    if args.command == "start":
        args.label = args.label.strip()
        if not args.label:
            ap.error("le label ne peut pas être vide")
    else:
        if not pathlib.Path(args.db).is_file():
            ap.error(f"base introuvable: {args.db}")
        if args.command in ("end", "note", "delete") and args.id is not None and args.id <= 0:
            ap.error("l'identifiant du label doit être > 0")
    try:
        with Storage(args.db, args.schema) as storage:
            if args.command == "start":
                if args.label not in KNOWN_LABELS:
                    print(f"note: '{args.label}' hors liste ({', '.join(KNOWN_LABELS)})")
                label_id = storage.start_label(now_ms(), args.label, args.notes)
                print(f"label {label_id} ouvert: {args.label}")
            elif args.command == "end":
                closed = storage.end_label(now_ms(), args.id)
                print(f"{closed} label(s) fermé(s)")
            elif args.command == "note":
                if not storage.update_label_notes(args.id, args.notes):
                    raise ValueError(f"label #{args.id} introuvable")
                print(f"notes du label #{args.id} mises à jour")
            elif args.command == "delete":
                if not storage.delete_label(args.id):
                    raise ValueError(f"label #{args.id} introuvable")
                print(f"label #{args.id} supprimé")
            else:
                rows = storage.list_labels(include_closed=args.all, label=args.label)
                if args.json:
                    payload = [
                        {"id": r[0], "ts_start": r[1], "ts_end": r[2], "label": r[3], "notes": r[4]}
                        for r in rows
                    ]
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    if not rows:
                        print("aucun label" if args.all else "aucun label ouvert")
                    for row in rows:
                        print(f"#{row[0]}  {human(row[1])} → {human(row[2])}  {row[3]}  {row[4]}")
    except (ValueError, OSError, sqlite3.Error) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
