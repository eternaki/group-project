#!/usr/bin/env python3
"""
Pobiera z folderu Google Drive TYLKO pliki wymienione w pliku manifestu
(jedna nazwa na linię) - używane do podziału pracy nad dog_tv_24_7_nareski
między kilka osób bez ręcznego zaznaczania plików w przeglądarce Drive.

    python -m scripts.download.download_by_manifest \
        --manifest data/dog_tv_partner.txt \
        --folder-id 1rUdyWfsn343tW-h6MT--vbbiOOMDIgrv \
        --output data/raw/dog_tv_24_7_nareski

Wymaga tego samego OAuth co reszta narzędzi w scripts/download/tiktok/
(secrets/gdrive_credentials.json - wspólny; secrets/token.json - osobisty,
tworzony przy pierwszym logowaniu pod WŁASNYM kontem Google).
"""

import argparse
from pathlib import Path

from scripts.download.tiktok.config import GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader


def main() -> None:
    parser = argparse.ArgumentParser(description="Pobiera pliki z Drive wg listy nazw")
    parser.add_argument("--manifest", required=True, type=Path, help="Plik z nazwami (jedna na linię)")
    parser.add_argument("--folder-id", required=True, help="ID folderu źródłowego na Drive")
    parser.add_argument("--output", required=True, type=Path, help="Katalog docelowy")
    args = parser.parse_args()

    wanted = {
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"W manifeście: {len(wanted)} plików")

    uploader = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, args.folder_id)
    uploader.authenticate()

    args.output.mkdir(parents=True, exist_ok=True)
    files = uploader.list_files(args.folder_id, fields="id,name,size")
    print(f"W folderze na Drive: {len(files)} plików")

    downloaded = skipped = missing = 0
    found_names = {f["name"] for f in files}
    for f in files:
        if f["name"] not in wanted:
            continue
        dest = args.output / f["name"]
        if dest.exists() and dest.stat().st_size == int(f.get("size", 0)):
            skipped += 1
            continue
        uploader.download_file(f["id"], dest)
        downloaded += 1
        if downloaded % 25 == 0:
            print(f"pobrano {downloaded}...")

    missing = len(wanted - found_names)
    print(f"GOTOWE: pobrano={downloaded}, juz bylo={skipped}, brak na Drive={missing}")


if __name__ == "__main__":
    main()
