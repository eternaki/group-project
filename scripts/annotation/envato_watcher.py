#!/usr/bin/env python3
"""
Przenosi pobrane z Envato wideo z ~/Downloads na Google Drive i kasuje lokalnie.

Uruchom: python -m scripts.annotation.envato_watcher <emocja>
Bierze pod uwagę TYLKO pliki wideo nowsze niż start watchera (żeby nie ruszać
istniejących plików użytkownika). Folder docelowy: envato_<emocja> na Drive.
"""

import sys
import time
from pathlib import Path

from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

DOWNLOADS = Path.home() / "Downloads"
VIDEO_EXT = (".mov", ".mp4", ".webm", ".m4v")
START = time.time()


def _stable(path: Path, wait: float = 2.0) -> bool:
    """True, jeśli rozmiar pliku nie zmienia się (pobieranie skończone)."""
    try:
        s1 = path.stat().st_size
        time.sleep(wait)
        return path.exists() and path.stat().st_size == s1 and s1 > 0
    except OSError:
        return False


def main(emotion: str) -> None:
    drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
    drive.authenticate()
    folder = drive.ensure_folder(f"envato_{emotion}", GDRIVE_FOLDER_ID)
    print(f"watcher: envato_{emotion} (folder {folder}); pilnuję {DOWNLOADS}", flush=True)
    moved = 0
    seen: set[str] = set()
    while True:
        for f in DOWNLOADS.glob("*"):
            if (f.suffix.lower() in VIDEO_EXT and f.name not in seen
                    and f.stat().st_mtime > START and not f.name.endswith(".crdownload")):
                if not _stable(f):
                    continue
                try:
                    drive.upload_file(f, remote_name=f.name, folder_id=folder, resumable=False)
                    f.unlink(missing_ok=True)
                    moved += 1
                    print(f"  ↑ {f.name} -> Drive, удалён локально ({moved})", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! błąd {f.name}: {exc}", flush=True)
                    seen.add(f.name)  # nie próbuj w kółko tego samego
        time.sleep(4)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sad")
