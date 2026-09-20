#!/usr/bin/env python3
"""
Buduje `docs/video_manifest.csv` — mapę „gdzie leży każde zweryfikowane wideo".

Czyta logi wyboru kadrów (`data/labels/dataset_final/select_*.jsonl`), a folder
na Google Drive dla każdego wideo ustala z listingu folderów źródłowych
(`OTOBRANE`/`SUROWE` z `select_frames.py`): najpierw po `fid`, a gdy loga nie
ma `fid` — po nazwie pliku.

Wynik (kolumny): `emotion, video, folder, fid, drive_link`. Wiersz na wideo,
deduplikacja po `fid` (a przy jego braku po nazwie).

Uruchom:
  PYTHONPATH=. python -m scripts.annotation.build_manifest
"""

import csv
import json
from pathlib import Path

from scripts.annotation.select_frames import (
    OTOBRANE,
    SUROWE,
    _drive,
    _find_folder,
)

LABELS_DIR = Path("data/labels/dataset_final")
OUT_PATH = Path("docs/video_manifest.csv")


def _drive_index() -> tuple[dict[str, str], dict[str, str]]:
    """Zwraca (fid -> folder, nazwa_pliku -> folder) ze wszystkich folderów źródłowych."""
    by_fid: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for folder in list(OTOBRANE) + SUROWE:
        fid = _find_folder(folder)
        if not fid:
            print(f"  ! brak folderu na Drive: {folder}")
            continue
        for f in _drive.list_files(fid, fields="id,name"):
            by_fid[f["id"]] = folder
            by_name.setdefault(f["name"], folder)
    return by_fid, by_name


def _load_selections() -> list[dict]:
    """Wczytuje logi select_*.jsonl, dedup po fid (a przy braku fid — po nazwie)."""
    seen: dict[str, dict] = {}
    for path in sorted(LABELS_DIR.glob("select_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("fid") or rec.get("video")
            if key:
                seen[key] = rec  # ostatni wpis wygrywa (najświeższa weryfikacja)
    return list(seen.values())


def build() -> None:
    by_fid, by_name = _drive_index()
    rows: list[dict] = []
    for rec in _load_selections():
        fid = rec.get("fid", "")
        name = rec.get("video", "")
        folder = by_fid.get(fid) or by_name.get(name, "")
        rows.append({
            "emotion": rec.get("emotion", ""),
            "video": name,
            "folder": folder,
            "fid": fid,
            "drive_link": f"https://drive.google.com/uc?id={fid}" if fid else "",
        })

    rows.sort(key=lambda r: (r["emotion"], r["video"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["emotion", "video", "folder", "fid", "drive_link"])
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
    print(f"=== {OUT_PATH}: {len(rows)} wideo ===")
    for e in sorted(counts):
        print(f"  {e:9} {counts[e]:3}")


if __name__ == "__main__":
    build()
