"""Jednorazowy skrypt: dociąga resztę dog_tv_24_7_nareski jako oddzielny proces."""

from pathlib import Path

from scripts.download.tiktok.config import GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

u = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
u.authenticate()

folder_id = "1rUdyWfsn343tW-h6MT--vbbiOOMDIgrv"
dest_dir = Path("D:/group-project/data/raw/dog_tv_24_7_nareski")
dest_dir.mkdir(parents=True, exist_ok=True)

files = u.list_files(folder_id, fields="id,name,size")
print("total files:", len(files))

downloaded = 0
skipped = 0
failed = 0
for i, f in enumerate(files):
    dest = dest_dir / f["name"]
    if dest.exists() and dest.stat().st_size == int(f.get("size", 0)):
        skipped += 1
        continue
    try:
        u.download_file(f["id"], dest)
        downloaded += 1
    except Exception as e:
        failed += 1
        print(f"FAILED {f['name']}: {e}")
    if (i + 1) % 50 == 0:
        print(f"progress: {i+1}/{len(files)} (downloaded={downloaded}, skipped={skipped}, failed={failed})")

print(f"DONE: downloaded={downloaded}, skipped={skipped}, failed={failed}")
