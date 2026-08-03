"""
Pobiera WSZYSTKIE pliki z publicznego folderu Google Drive (omija limit ~50 gdown).

Wymaga klucza Google Drive API (czyta z env GDRIVE_API_KEY lub pliku drive_key.txt
w korzeniu repo). Folder musi być udostępniony "każdy z linkiem".

Użycie:
    export GDRIVE_API_KEY=...            # lub utwórz drive_key.txt
    python scripts/download/download_drive_folder.py FOLDER_ID --out data/drive_dogs --limit 1500
"""

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.googleapis.com/drive/v3/files"


def _read_key() -> str:
    key = os.environ.get("GDRIVE_API_KEY", "").strip()
    if key:
        return key
    for cand in ("drive_key.txt", "key.txt"):
        p = Path(cand)
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    sys.exit("Brak klucza: ustaw GDRIVE_API_KEY lub utwórz drive_key.txt")


def _get_json(url: str) -> dict:
    import json

    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def list_folder(folder_id: str, key: str) -> list[dict]:
    """Listuje wszystkie pliki w folderze (z paginacją)."""
    files: list[dict] = []
    token = None
    while True:
        q = urllib.parse.quote(f"'{folder_id}' in parents and trashed=false")
        url = (
            f"{API}?q={q}&key={key}&pageSize=1000"
            f"&fields=nextPageToken,files(id,name,mimeType,size)"
            f"&supportsAllDrives=true&includeItemsFromAllDrives=true"
        )
        if token:
            url += f"&pageToken={token}"
        data = _get_json(url)
        files.extend(data.get("files", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return files


def download_file(file_id: str, dest: Path, key: str) -> bool:
    url = f"{API}/{file_id}?alt=media&key={key}&supportsAllDrives=true"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        return dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        print(f"  ! błąd {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_id")
    ap.add_argument("--out", type=Path, default=Path("data/drive_dogs"))
    ap.add_argument("--limit", type=int, default=0, help="0 = wszystkie")
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    key = _read_key()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Listowanie folderu...")
    all_files = list_folder(args.folder_id, key)
    # tylko pliki wideo (po mimeType / rozszerzeniu) i podfoldery pomijamy
    vids = [
        f for f in all_files
        if "folder" not in f.get("mimeType", "")
        and f["name"].lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))
    ]
    print(f"Znaleziono {len(vids)} wideo (z {len(all_files)} pozycji)")

    sel = vids[args.offset:]
    if args.limit:
        sel = sel[: args.limit]

    ok, skip = 0, 0
    for i, f in enumerate(sel):
        dest = args.out / f["name"]
        if dest.exists() and dest.stat().st_size > 0:
            skip += 1
            continue
        if download_file(f["id"], dest, key):
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sel)} (pobrano {ok}, pominięto {skip})")
            time.sleep(0.5)
    print(f"GOTOWE: pobrano {ok}, pominięto {skip}, razem na dysku do {len(sel)}")


if __name__ == "__main__":
    main()
