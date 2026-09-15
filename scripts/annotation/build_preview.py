#!/usr/bin/env python3
"""
Zbiór „zaliczka" — TYLKO kadry (początek/koniec) + wideo, bez adnotacji.

Szybko (bez modeli, ffmpeg): dla każdego oznaczonego wideo wycinamy kadr w
sekundzie start i end, układamy w:
    <high_resolution|low_resolution>/<emocja>/<emocja>_<NNN>_1.jpg (start), _2.jpg (end)

Podział rozdzielczości: krótszy bok >= 720 -> high_resolution, inaczej low_resolution.
Opcjonalnie (--drive) kopiuje WSZYSTKIE wideo zbioru na Drive do jednego folderu
`DATASET_150plus_videos/<res>/<emocja>/` (kopia serwerowa, nie rusza oryginałów).

fearful scalone z surprise.
"""

import argparse
import json
import logging
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.annotation.refine_startend import direct_license
from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

REPO = Path(__file__).resolve().parent.parent.parent
LABELS = REPO / "data" / "labels" / "dataset_final"
DL = Path("/tmp/preview_dl")
DL.mkdir(exist_ok=True)
CACHE = Path("/tmp/dogvids")
OUTPUT = REPO / "data" / "dataset_final" / "release_150plus_preview"
DRIVE_PARENT_NAME = "DATASET_150plus_videos"

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
MERGE: dict[str, str] = {}  # fearful i surprise rozdzielone
MIN_SIDE = 720

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_name2id: dict[str, str | None] = {}


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                raise
            logger.warning("Drive błąd (%s), ponawiam %d/%d", type(exc).__name__, i + 1, tries)
            time.sleep(3 * (i + 1))
            try:
                _drive.authenticate()
            except Exception:  # noqa: BLE001
                pass


def _load_records() -> list[dict]:
    records, seen_names, seen_fid, seen_old = [], set(), set(), set()
    for f in sorted(LABELS.glob("select_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("emotion") or r["fid"] in seen_fid:
                    continue
                if r.get("start_time") is None or r.get("end_time") is None:
                    continue
                seen_fid.add(r["fid"])
                seen_names.add(r.get("video"))
                records.append({"fid": r["fid"], "name": r.get("video"),
                                "emotion": MERGE.get(r["emotion"], r["emotion"]),
                                "start": r["start_time"], "end": r["end_time"]})
    for f in sorted(LABELS.glob("video_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("emotion") or r["video"] in seen_old or r["video"] in seen_names:
                    continue
                if r.get("start_time") is None or r.get("end_time") is None:
                    continue
                seen_old.add(r["video"])
                records.append({"fid": None, "name": r["video"],
                                "emotion": MERGE.get(r["emotion"], r["emotion"]),
                                "start": r["start_time"], "end": r["end_time"]})
    return records


def _find_drive(name: str) -> str | None:
    if name in _name2id:
        return _name2id[name]
    for cand in (f"{name}.mp4", f"{name}.webm", f"{name}.mov", name):
        safe = cand.replace("'", "\\'")
        r = _retry(lambda q=safe: _drive._service.files().list(
            q=f"name = '{q}' and trashed = false", fields="files(id)", pageSize=1).execute()).get("files", [])
        if r:
            _name2id[name] = r[0]["id"]
            return r[0]["id"]
    _name2id[name] = None
    return None


def _get_video(fid: str) -> Path | None:
    """Ścieżka do pliku wideo: cache odtwarzacza lub pobranie do DL."""
    cached = CACHE / f"{fid}.mp4"
    if cached.is_file():
        return cached
    dst = DL / f"{fid}.mp4"
    if dst.is_file():
        return dst
    try:
        _retry(lambda: _drive.download_file(fid, dst))
    except Exception:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        return None
    return dst if dst.is_file() else None


def _resolution(path: Path) -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        return 0, 0


def _extract(path: Path, t: float, dst: Path) -> bool:
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", str(max(0.0, t)), "-i", str(path),
                        "-frames:v", "1", "-q:v", "2", str(dst)],
                       capture_output=True, timeout=60, check=True)
        return dst.is_file()
    except Exception:  # noqa: BLE001
        return False


def build(do_drive: bool) -> None:
    records = _load_records()
    logger.info("Wideo do wycięcia: %d", len(records))
    _drive.authenticate()

    if OUTPUT.exists():
        import shutil
        shutil.rmtree(OUTPUT)
    for tier in ("high_resolution", "low_resolution"):
        for e in EMOTIONS:
            (OUTPUT / tier / e).mkdir(parents=True, exist_ok=True)

    counters: Counter = Counter()
    stats: Counter = Counter()
    manifest: list[dict] = []  # do kopiowania na Drive

    for n, rec in enumerate(records, 1):
        try:
            fid = rec["fid"] or _find_drive(rec["name"])
            path = _get_video(fid) if fid else None
            if path is None:
                stats["brak_pliku"] += 1
                continue
            w, h = _resolution(path)
            if min(w, h) == 0:
                stats["brak_rozdz"] += 1
                continue
            tier = "high_resolution" if min(w, h) >= MIN_SIDE else "low_resolution"
            emotion = rec["emotion"]
            counters[emotion] += 1
            base = f"{emotion}_{counters[emotion]:03d}"
            made = 0
            for role, t in (("start", rec["start"]), ("end", rec["end"])):
                num = 1 if role == "start" else 2
                dst = OUTPUT / tier / emotion / f"{base}_{num}.jpg"
                if _extract(path, float(t), dst):
                    made += 1
            if made == 0:
                counters[emotion] -= 1
                stats["brak_kadru"] += 1
            else:
                stats[tier] += 1
                # nazwa wideo = baza kadrów (spójne nazewnictwo wideo i obrazów)
                manifest.append({"fid": fid, "base": base, "tier": tier, "emotion": emotion})
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! %s: %s", rec["name"], exc)
            stats["blad"] += 1
        if n % 50 == 0:
            logger.info("  ... %d/%d | %s", n, len(records), dict(stats))

    (OUTPUT / "README.txt").write_text(
        "Zaliczka: TYLKO kadry (poczatek/koniec) + wideo, bez adnotacji.\n"
        "Podzial: high_resolution (krotszy bok >= 720) / low_resolution (< 720),\n"
        "wewnatrz foldery klas emocji. Pelny zbior (keypoints, AU, rasa, bbox) - wieczorem.\n",
        encoding="utf-8")
    logger.info("=== KADRY gotowe: %s ===", OUTPUT)
    for e in EMOTIONS:
        hi = len(list((OUTPUT / "high_resolution" / e).glob("*.jpg")))
        lo = len(list((OUTPUT / "low_resolution" / e).glob("*.jpg")))
        logger.info("  %-9s high %3d | low %3d kadrów", e, hi, lo)
    logger.info("statusy: %s", dict(stats))

    if do_drive:
        _copy_to_drive(manifest)


def _copy_to_drive(manifest: list[dict]) -> None:
    """Kopiuje wideo zbioru na Drive do DATASET_150plus_videos/<res>/<emocja>/."""
    logger.info("Kopiuję %d wideo na Drive do '%s'...", len(manifest), DRIVE_PARENT_NAME)
    parent = _drive.ensure_folder(DRIVE_PARENT_NAME, GDRIVE_FOLDER_ID)
    folders: dict[str, str] = {}

    def folder_for(tier: str, emotion: str) -> str:
        key = f"{tier}/{emotion}"
        if key not in folders:
            tier_id = folders.get(tier) or _drive.ensure_folder(tier, parent)
            folders[tier] = tier_id
            folders[key] = _drive.ensure_folder(emotion, tier_id)
        return folders[key]

    done = Counter()
    for i, m in enumerate(manifest, 1):
        try:
            target = folder_for(m["tier"], m["emotion"])
            _retry(lambda t=target, m=m: _drive._service.files().copy(
                fileId=m["fid"], body={"name": f"{m['base']}.mp4", "parents": [t]},
                fields="id").execute())
            done[m["tier"]] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! kopia %s: %s", m["name"], exc)
        if i % 50 == 0:
            logger.info("  ... skopiowano %d/%d", i, len(manifest))
    logger.info("Skopiowano na Drive: %s", dict(done))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zaliczka: kadry + wideo bez adnotacji")
    ap.add_argument("--drive", action="store_true", help="Skopiuj wideo na Drive do jednego folderu")
    args = ap.parse_args()
    build(args.drive)
