#!/usr/bin/env python3
"""
Zbiera do dwóch folderów na Drive (happy_final, sad_final) TYLKO te wideo, z
których pochodzą zatwierdzone kadry (po 250 na emocję).

Źródła etykiet:
- select_*.jsonl — nowe narzędzie: mamy wprost `fid` pliku na Drive.
- video_*.jsonl  — stare narzędzie: mamy tylko nazwę; plik szukamy po nazwie
  po CAŁYM Drive (leżą w innych folderach, czasem w 2 kopiach).

Kopiujemy przez Drive API (files.copy) — nic nie ląduje lokalnie. Kopia jest
niezależna, więc oryginały można później sprzątnąć bez ruszania *_final.

Uruchom: python -m scripts.annotation.build_final_folders          # dry-run
         python -m scripts.annotation.build_final_folders --go     # kopiuje
"""

import glob
import json
import os
import sys
from collections import defaultdict

from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

LAB = "data/labels/dataset_final"
EMOTIONS = ("happy", "sad")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v", ".mkv")


def _collect_labels() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(emocja -> zbiór fid z nowego narzędzia, emocja -> zbiór nazw ze starego)."""
    by_fid: dict[str, set[str]] = defaultdict(set)
    by_name: dict[str, set[str]] = defaultdict(set)
    for path in glob.glob(f"{LAB}/select_*.jsonl"):
        if "skip" in path:
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            emo = r.get("emotion")
            if emo in EMOTIONS and r.get("fid"):
                by_fid[emo].add(r["fid"])
    for path in glob.glob(f"{LAB}/video_*.jsonl"):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            emo = r.get("emotion")
            if emo in EMOTIONS:
                nm = r.get("video") or r.get("name")
                if nm:
                    by_name[emo].add(nm)
    return by_fid, by_name


def _find_by_name(drive: GoogleDriveUploader, name: str) -> dict | None:
    """Pierwszy plik wideo o danej nazwie (z rozszerzeniem lub bez) na całym Drive."""
    base = os.path.splitext(name)[0]
    q = f"name contains '{base}' and trashed = false and mimeType contains 'video/'"
    files = (drive._service.files()
             .list(q=q, fields="files(id,name,size)", pageSize=20)
             .execute().get("files", []))
    for f in files:
        stem = os.path.splitext(f["name"])[0]
        if stem == base or f["name"] == name:
            return f
    return files[0] if files else None


def main(go: bool) -> None:
    by_fid, by_name = _collect_labels()
    drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
    drive.authenticate()

    plan: dict[str, list[dict]] = {}  # emocja -> [{id,name,size}]
    missing: dict[str, list[str]] = defaultdict(list)
    for emo in EMOTIONS:
        items: dict[str, dict] = {}
        for fid in by_fid[emo]:  # nowe: mamy fid, dociągamy metadane
            try:
                m = drive._service.files().get(
                    fileId=fid, fields="id,name,size").execute()
                items[fid] = m
            except Exception as exc:  # noqa: BLE001
                missing[emo].append(f"fid:{fid} ({exc})")
        for nm in by_name[emo]:  # stare: szukamy po nazwie
            f = _find_by_name(drive, nm)
            if f:
                items.setdefault(f["id"], f)
            else:
                missing[emo].append(nm)
        plan[emo] = list(items.values())

    total_gb = 0.0
    for emo in EMOTIONS:
        size = sum(int(f.get("size", 0) or 0) for f in plan[emo])
        total_gb += size / 1e9
        print(f"{emo}_final: {len(plan[emo])} wideo, {size/1e9:.2f} GB, "
              f"nieodnalezionych {len(missing[emo])}")
        if missing[emo]:
            print(f"   brak: {missing[emo][:6]}")
    print(f"RAZEM do skopiowania: {total_gb:.2f} GB na Drive")

    if not go:
        print("\n[dry-run] nic nie skopiowano. Uruchom z --go, żeby wykonać.")
        return

    for emo in EMOTIONS:
        folder = drive.ensure_folder(f"{emo}_final", GDRIVE_FOLDER_ID)
        existing = {f["name"] for f in drive.list_files(folder, fields="id,name")}
        done = 0
        for f in plan[emo]:
            if f["name"] in existing:
                continue
            try:
                drive._service.files().copy(
                    fileId=f["id"], body={"name": f["name"], "parents": [folder]}
                ).execute()
                done += 1
                if done % 20 == 0:
                    print(f"  {emo}: skopiowano {done}...", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {emo} {f['name']}: {exc}", flush=True)
        print(f"GOTOWE {emo}_final: skopiowano {done} (pominięto istniejących "
              f"{len(existing)})", flush=True)


if __name__ == "__main__":
    main("--go" in sys.argv)
