"""Boucle de collecte à cadence fixe.

Lancement :
    sudo python -m collector.main --db data/metrics.db --period 1.0

Sans root, RAPL est simplement désactivé (energy_uj est en 0400) : le reste
collecte normalement.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import math
import os
import pathlib
import platform
import signal
import sqlite3
import threading
import time

from . import errors as error_catalogue
from . import readers as R
from .readers.base import Reader, ReaderState
from .registry import boot_id, find_all_hwmon, list_hwmon
from .storage import Storage

log = logging.getLogger("collector")

#: Compromis perte/usure : à 1 Hz, 30 samples = ~30 s de perte max sur kill -9.
#: Un commit par seconde userait le SSD pour rien.
FLUSH_EVERY = 30
#: ...mais avec --period 60, 30 samples feraient 30 min de données en RAM.
#: On flushe donc aussi au temps, le premier des deux qui tombe.
FLUSH_INTERVAL = 30.0
#: Garde-fou si la base devient inécrivable : on jette plutôt que d'OOM.
MAX_BUFFER = 10_000
MAX_EVENT_BUFFER = 10_000

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "schema.sql"


def validate_seconds(value: float, option: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{option} doit être un nombre fini > 0")


def close_reader(reader: Reader) -> None:
    try:
        reader.close()
    except Exception:
        log.exception("close() de %s a échoué", reader.name)


def candidate_readers(boot: str = "") -> list[Reader]:
    """Tous les readers envisagés, dans l'ordre où ils écriront leurs colonnes.

    Aucun n'est obligatoire : chacun décide dans setup() s'il a quelque chose
    à offrir sur cette machine. C'est ce qui permet au même dépôt de tourner
    sur un desktop 7600X + RTX 3060 Ti et sur un portable Ryzen à iGPU.
    """
    return [
        R.KmsgReader(boot=boot),
        R.ErrorCounterReader(boot=boot),
        R.K10TempReader(),
        R.GigabyteWmiReader(),
        R.RaplReader(),
        R.NvmlReader(),
        R.AmdGpuReader(),
        *(R.NvmeReader(i) for i in range(len(find_all_hwmon("nvme")))),
        R.PsiReader(),
        R.ProcReader(),
        R.ContextReader(),
    ]


def build_readers(on_disable=None, emitter=None, boot: str = "") -> list[ReaderState]:
    """Instancie et teste chaque reader ; ne garde que ceux qui répondent.

    L'émetteur d'événements est branché AVANT setup() : le rejeu du tampon
    noyau a lieu pendant la configuration, et ses erreurs doivent atterrir en
    base comme les autres.
    """
    active: list[ReaderState] = []
    owners: dict[str, str] = {}
    try:
        for reader in candidate_readers(boot):
            state = None
            try:
                reader.emitter = emitter
                if reader.setup():
                    candidate = ReaderState(reader, on_disable=on_disable)
                    for field in candidate.fields:
                        if field in owners:
                            raise ValueError(f"colonne {field} déjà produite par {owners[field]}")
                    state = candidate
                else:
                    log.info("reader indisponible: %s", reader.name)
            except Exception as exc:
                log.error("setup %s a échoué: %s", reader.name, exc)
            finally:
                if state is None:
                    close_reader(reader)
            if state is not None:
                active.append(state)
                owners.update({field: reader.name for field in state.fields})
                log.info("reader actif: %s -> %s", reader.name, ", ".join(state.fields))
        return active
    except BaseException:
        for state in active:
            close_reader(state.reader)
        raise


def now_ms() -> int:
    return int(time.time() * 1000)


class Collector:
    def __init__(self, db: str, period: float, schema: pathlib.Path = SCHEMA):
        validate_seconds(period, "--period")
        self.storage = Storage(db, schema)
        self.period = period
        self.running = True
        self._stop_event = threading.Event()
        self._closed = False
        self._event_buffer: deque[tuple] = deque()
        self._event_dedup: set[str] = set()
        self.readers: list[ReaderState] = []
        # Le boot_id est résolu avant les readers : KmsgReader s'en sert comme
        # préfixe de clé de déduplication (les numéros de séquence kmsg ne sont
        # uniques qu'à l'intérieur d'un boot).
        try:
            self.boot = boot_id()
            self.storage.register_boot(self.boot, now_ms(), platform.release())
            self.readers = build_readers(
                on_disable=self._record_failure,
                emitter=self._store_event,
                boot=self.boot,
            )

            for state in self.readers:
                self.storage.ensure_columns(state.fields, state.column_types)

            log.info("hwmon détectés: %s", ", ".join(sorted(list_hwmon())) or "aucun")
            self._store_event("collector", "start", "info", f"period={period}s")
        except BaseException:
            for state in self.readers:
                close_reader(state.reader)
            self.storage.close()
            raise

    def _store_event(
        self,
        source: str,
        kind: str,
        severity: str,
        message: str,
        ts: int | None = None,
        dedup: str | None = None,
    ) -> bool:
        """Accepte un événement, en RAM si nécessaire. False sur doublon.

        Une panne de stockage n'est pas une panne de capteur : elle ne doit
        pas désactiver les readers qui émettent des événements.
        """
        event = (now_ms() if ts is None else ts, source, kind, severity, message[:4000], dedup)
        if self._event_buffer:
            stored = self._queue_event(event)
        else:
            try:
                stored = self.storage.add_event(*event)
            except (sqlite3.Error, OSError):
                log.exception("écriture d'événement impossible, mise en attente")
                stored = self._queue_event(event)
        if stored and severity in (error_catalogue.CRITICAL, error_catalogue.ERROR):
            log.warning("[%s] %s: %s", severity, kind, message[:200])
        return stored

    def _queue_event(self, event: tuple) -> bool:
        dedup = event[-1]
        if dedup is not None and dedup in self._event_dedup:
            return False
        if len(self._event_buffer) >= MAX_EVENT_BUFFER:
            dropped = self._event_buffer.popleft()
            if dropped[-1] is not None:
                self._event_dedup.discard(dropped[-1])
            log.error("buffer d'événements saturé, événement le plus ancien abandonné")
        self._event_buffer.append(event)
        if dedup is not None:
            self._event_dedup.add(dedup)
        return True

    def _flush_events(self) -> None:
        while self._event_buffer:
            event = self._event_buffer[0]
            try:
                self.storage.add_event(*event)
            except (sqlite3.Error, OSError):
                log.exception("écriture de %d événement(s) en attente impossible",
                              len(self._event_buffer))
                break
            self._event_buffer.popleft()
            if event[-1] is not None:
                self._event_dedup.discard(event[-1])

    def _record_failure(self, name: str, message: str) -> None:
        self._store_event("collector", "reader_fail", "warning", f"{name}: {message}")

    def stop(self, *_):
        log.info("arrêt demandé")
        self.running = False
        self._stop_event.set()

    def tick(self) -> dict:
        row: dict = {"ts": now_ms(), "boot_id": self.boot}
        for state in self.readers:
            row.update(state.read())
        return row

    def flush(self, buffer: list[dict]) -> None:
        self._flush_events()
        if not buffer:
            return
        try:
            self.storage.insert_samples(buffer)
        except Exception:
            # Base pleine, disque en lecture seule, corruption : on log et on
            # continue. Le batch reste en RAM pour la prochaine tentative ;
            # seule la saturation du buffer autorise la perte des plus anciens.
            log.exception("écriture de %d samples impossible", len(buffer))
        else:
            buffer.clear()

    def run(self, duration: float | None = None) -> None:
        buffer: list[dict] = []
        # Cadence sans dérive : on vise des deadlines absolues, on ne fait pas
        # sleep(period) après un travail de durée variable — sinon on perd
        # quelques ms par tick et les timestamps glissent sur 14 jours.
        try:
            if duration is not None:
                validate_seconds(duration, "--duration")
            start = time.monotonic()
            end = None if duration is None else start + duration
            deadline = start
            last_flush = start
            since_flush = 0
            while self.running:
                now = time.monotonic()
                if end is not None and now >= end:
                    break
                if now >= deadline:
                    buffer.append(self.tick())
                    since_flush += 1
                    if len(buffer) > MAX_BUFFER:
                        dropped = len(buffer) - MAX_BUFFER
                        del buffer[:dropped]
                        log.error("buffer saturé, %d sample(s) les plus anciens abandonnés", dropped)
                    deadline += self.period
                    now = time.monotonic()
                    if deadline <= now:
                        # Un reader lent ne crée pas de ticks de rattrapage.
                        log.debug("tick en retard de %.3fs", now - deadline)
                        deadline = now + self.period

                if since_flush >= FLUSH_EVERY or now - last_flush >= FLUSH_INTERVAL:
                    self.flush(buffer)
                    last_flush = time.monotonic()
                    since_flush = 0

                if end is not None and time.monotonic() >= end:
                    break

                # Le flush a sa propre échéance, même si --period vaut 60 s.
                wake_at = min(deadline, last_flush + FLUSH_INTERVAL)
                if end is not None:
                    wake_at = min(wake_at, end)
                delay = wake_at - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(min(delay, threading.TIMEOUT_MAX))
        except KeyboardInterrupt:
            log.info("interruption clavier")
        finally:
            self.close(buffer)

    def close(self, buffer: list[dict] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        self._stop_event.set()
        buffer = [] if buffer is None else buffer
        try:
            for state in self.readers:
                close_reader(state.reader)
            self._store_event("collector", "stop", "info", "")
            self.flush(buffer)
            if buffer or self._event_buffer:
                log.error("arrêt avec %d sample(s) et %d événement(s) non enregistrés",
                          len(buffer), len(self._event_buffer))
            try:
                log.info("%d samples en base", self.storage.count_samples())
            except sqlite3.Error:
                log.exception("comptage des samples impossible à l'arrêt")
        finally:
            self.storage.close()
        log.info("collecteur fermé")


def _cpu_model() -> str:
    try:
        text = pathlib.Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return platform.machine()
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.machine()


def collect_doctor_info() -> dict:
    """Inspecte les sources et capacités de détection exposées par la machine."""
    hwmon = sorted(list_hwmon())
    readers_info: list[dict] = []
    counters = None
    kmsg = None
    try:
        for reader in candidate_readers(boot="doctor"):
            try:
                ok = reader.setup()
            except Exception as exc:
                readers_info.append({
                    "name": reader.name,
                    "status": "error",
                    "fields": [],
                    "error": str(exc),
                })
                close_reader(reader)
                continue
            if isinstance(reader, R.ErrorCounterReader) and ok:
                counters = reader
            if isinstance(reader, R.KmsgReader):
                kmsg = reader
            readers_info.append({
                "name": reader.name,
                "status": "ok" if ok else "unavailable",
                "fields": list(reader.fields) if ok else [],
                "error": None,
            })
            if reader is not counters and reader is not kmsg:
                close_reader(reader)

        kinds = error_catalogue.catalogue()
        if kmsg is not None and kmsg.backend is not None:
            kmsg_info: dict = {
                "available": True,
                "backend": kmsg.backend.name,
                "backfilled": kmsg.backfilled,
                "reason": None,
            }
        else:
            kmsg_info = {
                "available": False,
                "backend": None,
                "backfilled": 0,
                "reason": kmsg.unavailable_reason if kmsg else "reader absent",
            }
        counters_info = {
            "active": [
                {"field": field, "note": note.splitlines()[0]}
                for field, note in counters.describe()
            ] if counters is not None else [],
            "missing": counters.missing() if counters is not None else [],
        }
    finally:
        for reader in (kmsg, counters):
            if reader is not None:
                close_reader(reader)

    return {
        "machine": platform.node(),
        "kernel": platform.release(),
        "cpu": _cpu_model(),
        "hwmon": hwmon,
        "is_root": os.geteuid() == 0,
        "readers": readers_info,
        "error_detection": {
            "rules": len(error_catalogue.RULES),
            "kinds": len(kinds),
            "catch_all_level": error_catalogue.KERNEL_ERROR_LEVEL,
            "kmsg": kmsg_info,
            "counters": counters_info,
        },
    }


def run_doctor(*, as_json: bool = False) -> None:
    """`--doctor` : ce que cette machine permet de mesurer et de détecter.

    À lancer sur chaque machine : la couverture n'est pas la même partout, et
    c'est cette sortie qui le dit — y compris ce qui N'EST PAS surveillé.
    """
    info = collect_doctor_info()
    if as_json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    print(f"Machine   {info['machine']} · noyau {info['kernel']}")
    print(f"CPU       {info['cpu']}")
    print(f"hwmon     {', '.join(info['hwmon']) or 'aucun'}")
    print(f"Droits    {'root' if info['is_root'] else 'utilisateur ordinaire'}")

    print("\nSources de métriques")
    for reader in info["readers"]:
        if reader["status"] == "error":
            print(f"  [!!] {reader['name']:14} erreur de configuration: {reader['error']}")
            continue
        ok = reader["status"] == "ok"
        marker = "OK" if ok else "--"
        detail = ", ".join(reader["fields"]) if ok else "absent de cette machine"
        print(f"  [{marker}] {reader['name']:14} {detail[:96]}")

    det = info["error_detection"]
    print("\nDétection d'erreurs")
    print(f"  Étage 1  messages noyau : {det['rules']} règles, {det['kinds']} genres")
    kmsg = det["kmsg"]
    if kmsg["available"]:
        print(f"           source active : {kmsg['backend']}")
        print(f"           tampon du boot courant : {kmsg['backfilled']} erreur(s)")
    else:
        print(f"           INDISPONIBLE : {kmsg['reason']}")
    print(f"  Étage 2  attrape-tout : toute erreur noyau de priorité <= {det['catch_all_level']} non classée")
    active_counters = det["counters"]["active"]
    if active_counters:
        print(f"  Étage 3  compteurs matériels : {len(active_counters)} actifs")
        for item in active_counters:
            print(f"           {item['field']:20} {item['note'][:70]}")
        absent = det["counters"]["missing"]
        if absent:
            print(f"           non exposés ici : {', '.join(absent)}")
    else:
        print("  Étage 3  compteurs matériels : aucun")


def collect_scan_results() -> dict:
    """Classe tout le tampon noyau du boot courant, sans écrire en base."""
    found: list[tuple[str, str, str]] = []
    unavailable: list[dict[str, str]] = []

    def collect(source, kind, severity, message, ts=None, dedup=None):
        found.append((kind, severity, message))
        return True

    for reader in (R.KmsgReader(boot="scan"), R.ErrorCounterReader(boot="scan")):
        reader.emitter = collect
        try:
            if not reader.setup():
                reason = getattr(reader, "unavailable_reason", "") or "indisponible"
                unavailable.append({"reader": reader.name, "status": "unavailable", "reason": reason})
        except Exception as exc:
            unavailable.append({"reader": reader.name, "status": "error", "reason": str(exc)})
        finally:
            close_reader(reader)

    groups: dict[str, list[tuple[str, str]]] = {}
    for kind, severity, message in found:
        groups.setdefault(kind, []).append((severity, message))

    order = {error_catalogue.CRITICAL: 0, error_catalogue.ERROR: 1}
    ranked = sorted(
        groups.items(), key=lambda kv: (order.get(kv[1][0][0], 2), -len(kv[1]))
    )
    criticals = sum(1 for _, sev, _ in found if sev == error_catalogue.CRITICAL)
    return {
        "total": len(found),
        "criticals": criticals,
        "unavailable": unavailable,
        "groups": [
            {
                "kind": kind,
                "severity": entries[0][0],
                "count": len(entries),
                "messages": [message for _, message in entries],
            }
            for kind, entries in ranked
        ],
    }


def run_scan(top: int = 6, *, as_json: bool = False) -> None:
    """`--scan` : classe tout le tampon noyau du boot courant, sans écrire en base.

    C'est l'outil à lancer sur chaque machine pour voir *ce qui est réellement
    détecté* — et pour repérer un faux positif avant qu'il ne pollue 14 jours
    de dataset.
    """
    result = collect_scan_results()
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    for item in result["unavailable"]:
        marker = "!!" if item["status"] == "error" else "--"
        print(f"[{marker}] {item['reader']}: {item['reason']}")

    if not result["total"]:
        print("Aucune erreur détectée sur le boot courant.")
        return

    print(f"{result['total']} erreur(s) classée(s) sur le boot courant :\n")
    for group in result["groups"]:
        print(f"  {group['severity'].upper():8} {group['kind']:18} x{group['count']}")
        for message in group["messages"][:top]:
            print(f"           {message[:110]}")
        if group["count"] > top:
            print(f"           ... et {group['count'] - top} autre(s)")
    print(f"\nVerdict : {result['criticals']} critique(s). "
          + ("Rien de bloquant." if not result["criticals"] else "À examiner avant de collecter."))


def main() -> None:
    ap = argparse.ArgumentParser(description="AAD — collecte de télémétrie matérielle")
    ap.add_argument("--db", default="data/metrics.db")
    ap.add_argument("--period", type=float, default=1.0, help="secondes entre deux samples")
    ap.add_argument("--duration", type=float, help="arrêt automatique après N secondes")
    ap.add_argument("--schema", type=pathlib.Path, default=SCHEMA)
    ap.add_argument("--doctor", "--list-sensors", dest="doctor", action="store_true",
                    help="ce que cette machine permet de mesurer et de détecter, puis sortie")
    ap.add_argument("--scan", action="store_true",
                    help="classe les erreurs du boot courant sans rien écrire en base")
    ap.add_argument("--json", action="store_true",
                    help="sortie JSON (avec --doctor ou --scan)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.json and not (args.doctor or args.scan):
        ap.error("--json nécessite --doctor ou --scan")

    if args.doctor:
        run_doctor(as_json=args.json)
        return

    if args.scan:
        run_scan(as_json=args.json)
        return

    try:
        validate_seconds(args.period, "--period")
        if args.duration is not None:
            validate_seconds(args.duration, "--duration")
    except ValueError as exc:
        ap.error(str(exc))

    collector = Collector(args.db, args.period, args.schema)
    signal.signal(signal.SIGINT, collector.stop)
    signal.signal(signal.SIGTERM, collector.stop)
    collector.run(duration=args.duration)


if __name__ == "__main__":
    main()
