#!/usr/bin/env python3
"""
Pobiera wideo z Envato po podpisanych linkach i wgrywa na Drive (envato_<emocja>).

Wejście: JSON z listą {id, url} (z download.data). Sekwencyjnie: pobierz jeden ->
wgraj na Drive -> skasuj lokalnie. Na dysku zawsze jeden plik.

Uruchom: python -m scripts.annotation.envato_download <emocja> <plik.json>
"""

import json
import sys
import time
from pathlib import Path

import requests

from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

TMP = Path("/tmp/envato_dl")
TMP.mkdir(exist_ok=True)


def _download_one(url: str, local: Path, tries: int = 3) -> bool:
    for i in range(tries):
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with local.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return True
        except Exception:  # noqa: BLE001
            local.unlink(missing_ok=True)
            time.sleep(2 * (i + 1))
    return False


def main(emotion: str, json_path: str) -> None:
    data = json.loads(Path(json_path).read_text())
    items = [x for x in data.get("items", []) if x.get("url")]
    drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
    drive.authenticate()
    folder = drive.ensure_folder(f"envato_{emotion}", GDRIVE_FOLDER_ID)
    # pomiń już wgrane (po nazwie envato_<uuid>.mov)
    existing = {f["name"] for f in drive.list_files(folder, fields="id,name")}
    items = [x for x in items if f"envato_{x['id']}.mov" not in existing]
    print(f"envato_{emotion}: {len(items)} do pobrania (pominięto {len(existing)} istniejących)", flush=True)

    ok = fail = 0
    for i, it in enumerate(items, 1):
        name = f"envato_{it['id']}.mov"
        local = TMP / name
        if _download_one(it["url"], local):
            try:
                drive.upload_file(local, remote_name=name, folder_id=folder, resumable=False)
                ok += 1
                mb = local.stat().st_size // (1 << 20)
                print(f"  [{i}/{len(items)}] ↑ {name} ({mb}MB)", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  [{i}/{len(items)}] ! upload {it['id']}: {exc}", flush=True)
        else:
            fail += 1
            print(f"  [{i}/{len(items)}] ! pobranie {it['id']} nieudane", flush=True)
        local.unlink(missing_ok=True)
    print(f"GOTOWE envato_{emotion}: wgrano {ok}, błędów {fail}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
