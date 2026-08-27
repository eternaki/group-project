#!/usr/bin/env python3
"""
Wysyła lokalne nagrania do folderu na Google Drive.

    python scripts/download/upload_to_drive.py data/drive_dogs/sobaki_local --subfolder sobaki_local

Wysyłanie NIE DZIAŁA na kluczu API — klucz daje wyłącznie odczyt plików
publicznych. Potrzebny jest OAuth: plik klienta z Google Cloud
(`secrets/credentials.json`, typ „Desktop app") i JEDNORAZOWE logowanie
w przeglądarce. Token odświeżania ląduje w `secrets/token.json`, więc kolejne
uruchomienia są już bezobsługowe.

Skrypt jest wznawialny: pomija nazwy, które w folderze docelowym już są.
Przerwane wysyłanie można więc powtórzyć bez kosztu.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.download.tiktok.drive_uploader import GoogleDriveUploader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRIVE_FOLDER_ID: str = "1jxUaN3Mq1ge8lFcPzwnN2ISl0E9k9mfQ"

CREDENTIALS_PATH: Path = REPO_ROOT / "secrets" / "credentials.json"
TOKEN_PATH: Path = REPO_ROOT / "secrets" / "token.json"

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".avi", ".webm", ".mkv")

PROGRESS_EVERY: int = 10


def _remote_names(uploader: GoogleDriveUploader, folder_id: str) -> set[str]:
    """
    Czyta nazwy plików leżących już w folderze docelowym.

    Args:
        uploader: Uwierzytelniony klient Dysku
        folder_id: Folder docelowy

    Returns:
        Nazwy plików obecnych na Dysku
    """
    service = uploader._service  # noqa: SLF001 — klasa nie wystawia listowania
    names: set[str] = set()
    token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(name)",
                pageSize=1000,
                pageToken=token,
            )
            .execute()
        )
        names.update(item["name"] for item in response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return names


def upload_directory(source: Path, subfolder: str, parent_id: str) -> tuple[int, int]:
    """
    Wysyła nagrania z katalogu do podfolderu na Dysku.

    Args:
        source: Katalog z nagraniami
        subfolder: Nazwa podfolderu na Dysku
        parent_id: Folder nadrzędny na Dysku

    Returns:
        Para (wysłane, pominięte jako już obecne)
    """
    uploader = GoogleDriveUploader(CREDENTIALS_PATH, TOKEN_PATH, parent_id)
    logger.info("Loguje sie do Dysku (przy pierwszym razie otworzy sie przegladarka)")
    uploader.authenticate()

    target_id = uploader.ensure_folder(subfolder, parent_id)
    logger.info(f"Folder docelowy: {subfolder} ({target_id})")

    already = _remote_names(uploader, target_id)
    videos = sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    pending = [video for video in videos if video.name not in already]
    logger.info(f"Nagran lokalnie {len(videos)}, na Dysku juz {len(videos) - len(pending)}")

    sent = 0
    for index, video in enumerate(pending, start=1):
        uploader.upload_file(video, folder_id=target_id)
        sent += 1
        if index % PROGRESS_EVERY == 0:
            logger.info(f"  {index}/{len(pending)}")
    return sent, len(videos) - len(pending)


def main() -> None:
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(description="Wysyla nagrania na Google Drive")
    parser.add_argument("source", type=Path, help="Katalog z nagraniami")
    parser.add_argument("--subfolder", required=True, help="Nazwa podfolderu na Dysku")
    parser.add_argument("--parent", default=DRIVE_FOLDER_ID, help="Folder nadrzedny na Dysku")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Nie ma katalogu: {args.source}")

    sent, skipped = upload_directory(args.source, args.subfolder, args.parent)
    logger.info(f"GOTOWE: wyslane {sent}, pominiete jako obecne {skipped}")


if __name__ == "__main__":
    main()
