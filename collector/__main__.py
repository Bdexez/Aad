"""Point d'entrée CLI unifié : `python -m collector <commande> [options]`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from . import (
    analyze,
    backup,
    detector,
    export,
    features,
    html_report,
    label,
    main as collect_main,
    report,
)

COMMANDS: dict[str, tuple[str, object]] = {
    "run": ("lancer la collecte continue", lambda argv: collect_main.main(argv)),
    "doctor": ("diagnostiquer les capteurs et sources d'erreurs", lambda argv: collect_main.main(["--doctor", *argv])),
    "scan": ("classer les erreurs noyau du boot courant", lambda argv: collect_main.main(["--scan", *argv])),
    "label": ("gérer les annotations d'expériences", label.main),
    "report": ("rapport de complétude du dataset", report.main),
    "html-report": ("générer un rapport HTML autonome avec sparklines SVG", html_report.main),
    "analyze": ("analyse exploratoire (cadence, stats, corrélations, dérive)", analyze.main),
    "detect": ("détecteur statistique d'anomalies et calibration", detector.main),
    "export": ("exporter mesures (CSV/JSONL/downsampled) ou événements", export.main),
    "features": ("exporter des features temporelles dérivées pour le ML", features.main),
    "backup": ("sauvegarder ou vérifier une base SQLite active", backup.main),
}


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            prog="python -m collector",
            description="AAD — Arch Anomaly Detector CLI unifié",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Commandes disponibles :\n" + "\n".join(
                f"  {name:10} {desc}" for name, (desc, _) in COMMANDS.items()
            ),
        )
        parser.print_help()
        return

    command, rest = args[0], args[1:]
    entry = COMMANDS.get(command)
    if entry is None:
        available = ", ".join(COMMANDS)
        print(f"erreur: commande inconnue {command!r} (attendu : {available})", file=sys.stderr)
        raise SystemExit(2)
    _, handler = entry
    handler(rest)


if __name__ == "__main__":
    main()
